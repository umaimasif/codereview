"""
Streamlit web page for the AI Code Review Assistant.

- The PR link is typed into an INPUT BOX by the user (never stored in secrets).
- Only private keys (GROQ_API_KEY, GITHUB_TOKEN, APP_PASSWORD) live in Streamlit Secrets.
"""

import os
import re

import streamlit as st

st.set_page_config(page_title="AI Code Review", page_icon="🤖", layout="centered")


def get_secret(name: str):
    """Read a key from Streamlit Secrets, or from the environment/.env locally."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


# Copy keys into environment variables so app.py can read them
for _key in ("GROQ_API_KEY", "GITHUB_TOKEN", "GROQ_MODEL"):
    _value = get_secret(_key)
    if _value:
        os.environ[_key] = _value

from app import build_graph  # noqa: E402  (import after env vars are set)


@st.cache_resource
def get_graph():
    return build_graph()


PR_PATTERN = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")

st.title("🤖 AI Code Review")
st.write("Paste a GitHub Pull Request link and get comments, risk flags, and suggestions.")

# Optional password gate (set APP_PASSWORD in Secrets to turn it on)
app_password = get_secret("APP_PASSWORD")
if app_password:
    typed = st.text_input("Password", type="password")
    if typed != app_password:
        st.info("Enter the password to use this app.")
        st.stop()

# ---- The INPUT BOX for the PR link ----
pr_url = st.text_input(
    "GitHub Pull Request link",
    placeholder="https://github.com/owner/repo/pull/1",
).strip()

if st.button("Review this PR", type="primary"):
    if not pr_url:
        st.warning("Please paste a Pull Request link first.")
    elif not PR_PATTERN.match(pr_url):
        st.error(
            "That is not a Pull Request link. It must look like "
            "`https://github.com/owner/repo/pull/1`. "
            "A normal repo link will not work, because this tool reviews a PR's changes."
        )
    else:
        with st.spinner("Reviewing your PR... this can take a minute."):
            try:
                result = get_graph().invoke({"source": pr_url})
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        if result.get("error"):
            st.error(result["error"])
        else:
            st.markdown(result["report"])
            st.download_button(
                "⬇️ Download report (.md)",
                data=result["report"],
                file_name="review.md",
                mime="text/markdown",
            )
