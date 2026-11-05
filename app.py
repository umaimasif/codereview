"""
AI Code Review Assistant (LangGraph + Groq)

Input : a GitHub PR URL, a .diff/.patch file path, or "-" to read a diff from stdin
Output: a Markdown report with comments, risk flags, and suggestions

Graph:
    fetch_diff --> split_diff --> review_files --> summarize --> END
         |
         +--(error / empty diff)--> END
"""

import argparse
import os
import re
import sys
from typing import List, Literal, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_FILES = 15
MAX_CHARS_PER_FILE = 12_000  # keep prompts small so we stay under token limits
SKIP_FILES = (".lock", "package-lock.json", "yarn.lock", ".min.js", ".svg", ".png", ".jpg")

Severity = Literal["low", "medium", "high"]


# ----------------------------------------------------------------------------
# 1) Shapes of data (what the LLM must return)
# ----------------------------------------------------------------------------
class Comment(BaseModel):
    line_hint: str = Field(description="The changed code line or snippet this comment is about")
    severity: Severity
    comment: str = Field(description="What is wrong or worth discussing")
    suggestion: str = Field(description="A concrete fix or improvement")


class FileReview(BaseModel):
    summary: str = Field(description="One or two sentences: what changed in this file")
    comments: List[Comment] = Field(default_factory=list)
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Short flags such as 'SQL injection', 'hardcoded secret', 'missing tests', "
        "'breaking API change', 'race condition'",
    )
    risk_level: Severity


# ----------------------------------------------------------------------------
# 2) The graph state (the "backpack" every node carries)
# ----------------------------------------------------------------------------
class ReviewState(TypedDict, total=False):
    source: str
    diff: str
    file_chunks: List[dict]  # [{"filename": str, "patch": str}]
    reviews: List[dict]  # [{"filename": str, "review": FileReview-as-dict}]
    overall_summary: str
    report: str
    error: str


def get_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Put it in your .env file.")
    return ChatGroq(model=MODEL_NAME, temperature=0)


# ----------------------------------------------------------------------------
# 3) Nodes
# ----------------------------------------------------------------------------
def fetch_diff(state: ReviewState) -> ReviewState:
    """Get the raw diff text from a PR URL, a file, or stdin."""
    source = state["source"]

    try:
        if source == "-":
            return {"diff": sys.stdin.read()}

        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                return {"diff": f.read()}

        match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", source)
        if match:
            owner, repo, number = match.groups()
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
            headers = {"Accept": "application/vnd.github.v3.diff"}
            token = os.getenv("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return {"diff": resp.text}

        return {"error": f"Don't know how to read source: {source}"}
    except Exception as e:  # network errors, bad file, 404, etc.
        return {"error": f"Failed to fetch diff: {e}"}


def route_after_fetch(state: ReviewState) -> str:
    if state.get("error") or not state.get("diff", "").strip():
        return "stop"
    return "continue"


def split_diff(state: ReviewState) -> ReviewState:
    """Cut one big diff into one chunk per file."""
    parts = re.split(r"(?=^diff --git )", state["diff"], flags=re.MULTILINE)
    chunks = []
    for part in parts:
        if not part.startswith("diff --git"):
            continue
        header = part.splitlines()[0]
        m = re.match(r"diff --git a/(.+?) b/(.+)", header)
        filename = m.group(2) if m else "unknown"
        if filename.endswith(SKIP_FILES) or "Binary files" in part:
            continue
        if len(part) > MAX_CHARS_PER_FILE:
            part = part[:MAX_CHARS_PER_FILE] + "\n... [diff truncated]"
        chunks.append({"filename": filename, "patch": part})

    if not chunks:
        return {"error": "No reviewable files found in the diff."}
    return {"file_chunks": chunks}


REVIEW_SYSTEM_PROMPT = """You are a senior software engineer doing a code review.
You get a unified diff for ONE file. Lines starting with '+' are added, '-' are removed.

Rules:
- Only comment on things that are actually in the diff. Do not invent code.
- Look for: bugs, security problems, bad error handling, performance issues,
  breaking changes, missing tests, unclear naming, and risky logic.
- Be specific and short. Every comment needs a concrete suggestion.
- If the change looks good, return an empty comments list and risk_level "low".
- risk_level: high = could cause outage/security issue/data loss,
  medium = likely bug or maintenance problem, low = minor or safe.
"""


def review_files(state: ReviewState) -> ReviewState:
    """Ask the LLM to review each file, one by one."""
    structured_llm = get_llm().with_structured_output(FileReview)
    reviews = []

    for chunk in state["file_chunks"]:
        print(f"  reviewing {chunk['filename']} ...", file=sys.stderr)
        try:
            result: FileReview = structured_llm.invoke(
                [
                    SystemMessage(content=REVIEW_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"File: {chunk['filename']}\n\nDiff:\n{chunk['patch']}"
                    ),
                ]
            )
            reviews.append({"filename": chunk["filename"], "review": result.model_dump()})
        except Exception as e:
            reviews.append(
                {
                    "filename": chunk["filename"],
                    "review": {
                        "summary": f"Could not review this file: {e}",
                        "comments": [],
                        "risk_flags": ["review failed"],
                        "risk_level": "medium",
                    },
                }
            )
    return {"reviews": reviews}


RANK = {"low": 0, "medium": 1, "high": 2}
ICON = {"low": "🟢", "medium": "🟡", "high": "🔴"}


def summarize(state: ReviewState) -> ReviewState:
    """Make an overall summary and a final Markdown report."""
    reviews = state["reviews"]
    overall_risk = max((r["review"]["risk_level"] for r in reviews), key=RANK.get)

    # One extra LLM call: a short PR-level summary
    bullets = "\n".join(f"- {r['filename']}: {r['review']['summary']}" for r in reviews)
    try:
        overall_summary = (
            get_llm()
            .invoke(
                [
                    SystemMessage(
                        content="Write a 2-3 sentence summary of this pull request for a "
                        "reviewer, based on the per-file summaries. No fluff."
                    ),
                    HumanMessage(content=bullets),
                ]
            )
            .content
        )
    except Exception:
        overall_summary = bullets

    lines = [
        "# 🤖 AI Code Review",
        "",
        f"**Overall risk:** {ICON[overall_risk]} {overall_risk.upper()}",
        "",
        "## Summary",
        overall_summary,
        "",
    ]

    all_flags = sorted({f for r in reviews for f in r["review"]["risk_flags"]})
    if all_flags:
        lines += ["## ⚠️ Risk flags", *[f"- {f}" for f in all_flags], ""]

    lines.append("## File-by-file review")
    for r in reviews:
        rv = r["review"]
        lines += [
            "",
            f"### `{r['filename']}`  {ICON[rv['risk_level']]} {rv['risk_level']}",
            rv["summary"],
        ]
        if not rv["comments"]:
            lines.append("\n_No issues found._")
        for c in sorted(rv["comments"], key=lambda c: -RANK[c["severity"]]):
            lines += [
                "",
                f"- {ICON[c['severity']]} **{c['severity'].upper()}** — `{c['line_hint']}`",
                f"  - **Issue:** {c['comment']}",
                f"  - **Suggestion:** {c['suggestion']}",
            ]

    return {"overall_summary": overall_summary, "report": "\n".join(lines)}


# ----------------------------------------------------------------------------
# 4) Build the graph
# ----------------------------------------------------------------------------
def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_diff", fetch_diff)
    g.add_node("split_diff", split_diff)
    g.add_node("review_files", review_files)
    g.add_node("summarize", summarize)

    g.add_edge(START, "fetch_diff")
    g.add_conditional_edges(
        "fetch_diff", route_after_fetch, {"continue": "split_diff", "stop": END}
    )
    g.add_conditional_edges(
        "split_diff",
        lambda s: "stop" if s.get("error") else "continue",
        {"continue": "review_files", "stop": END},
    )
    g.add_edge("review_files", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


# ----------------------------------------------------------------------------
# 5) Command line
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI code review for a PR diff")
    parser.add_argument("source", help="GitHub PR URL, path to .diff file, or '-' for stdin")
    parser.add_argument("--out", help="Save the report to this Markdown file")
    args = parser.parse_args()

    app = build_graph()
    final = app.invoke({"source": args.source})

    if final.get("error"):
        print(f"Error: {final['error']}", file=sys.stderr)
        sys.exit(1)

    print(final["report"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(final["report"])
        print(f"\nSaved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
