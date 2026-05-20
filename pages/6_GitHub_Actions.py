"""
GitHub Actions — fetch and display recent workflow runs

Pulls from the public REST API: /repos/{owner}/{repo}/actions/runs
For private repos, set GITHUB_TOKEN in your environment.
"""

import os
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

# Importing dashboard.data triggers the load_dotenv() call inside, so
# GITHUB_TOKEN gets picked up from .env even if you navigate straight here.
from dashboard import data as _dashboard_data  # noqa: F401 (side effect: load .env)
from dashboard.components import sidebar_refresh, status_chip
from safe_io import run_utf8


st.set_page_config(page_title="GitHub Actions", page_icon="•", layout="wide")
sidebar_refresh()
st.title("GitHub Actions")


# ─────────────────────────────────────────────────────────────────────
# Repo discovery — try the git remote, fall back to manual entry
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _detect_repo() -> Optional[str]:
    """Read .git/config to find the GitHub repo (owner/name)."""
    try:
        r = run_utf8(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, timeout=2,
        )
        url = r.stdout.strip()
        if "github.com" not in url:
            return None
        # Parse: git@github.com:owner/repo.git  OR  https://github.com/owner/repo.git
        if url.startswith("git@"):
            owner_repo = url.split(":", 1)[-1]
        else:
            owner_repo = url.split("github.com/", 1)[-1]
        return owner_repo.removesuffix(".git").strip("/")
    except Exception:
        return None


repo_default = _detect_repo() or "uwuexdeemeow/Stock-Market-AI-Bot"

c1, c2 = st.columns([3, 1])
with c1:
    repo = st.text_input("Repository (owner/name)", value=repo_default, key="gha_repo")
with c2:
    workflow_filter = st.text_input("Workflow file (optional)",
                                     value="daily_paper_trading.yml",
                                     help="e.g. daily_paper_trading.yml — blank = all workflows",
                                     key="gha_workflow")

# Surface token detection up front — most 404s on private repos are
# really "no auth" in disguise.
_token_detected = bool(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
if _token_detected:
    status_chip("GITHUB_TOKEN detected", "ok")
else:
    status_chip("GITHUB_TOKEN not set — private repos will return 404", "warn")
    with st.expander("How to add a token"):
        st.markdown("""
1. Open https://github.com/settings/personal-access-tokens/new
2. **Repository access** → Only select repositories → pick this repo
3. **Repository permissions** → Actions → **Read-only**
4. Click **Generate token**, copy the `github_pat_...` value
5. Add it to your `.env` file:
   ```
   GITHUB_TOKEN=github_pat_...
   ```
6. Restart Streamlit (Ctrl+C, then `streamlit run dashboard.py`)

Classic tokens also work — at https://github.com/settings/tokens/new
tick the `repo` scope.
        """)


# ─────────────────────────────────────────────────────────────────────
# Fetch runs from the GitHub REST API
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_runs(repo: str, workflow: str = "", limit: int = 20) -> dict:
    """Hit the GitHub API for recent workflow runs.

    Uses `requests` (which respects the certifi CA bundle by default).
    Falls back to urllib + certifi SSL context if `requests` isn't
    installed.  Both routes work without manually setting REQUESTS_CA_BUNDLE.
    """
    base = f"https://api.github.com/repos/{repo}/actions"
    if workflow:
        url = f"{base}/workflows/{workflow}/runs?per_page={limit}"
    else:
        url = f"{base}/runs?per_page={limit}"

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "stock-bot-dashboard",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # ── Path 1: requests (preferred — handles SSL via certifi) ─────────
    try:
        import requests
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 404:
            return {"_error": f"404 Not Found — repo {repo!r} or workflow {workflow!r} doesn't exist"}
        if r.status_code == 403 and "rate limit" in r.text.lower():
            return {"_error": f"403 Rate limit exceeded.  Set GITHUB_TOKEN to authenticate."}
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except ImportError:
        pass  # requests not installed — try urllib path
    except Exception as exc:
        # If requests failed for another reason, try urllib as fallback
        urllib_err = str(exc)

    # ── Path 2: urllib + explicit certifi CA bundle ────────────────────
    import urllib.request
    import json as _json
    import ssl
    try:
        # Build an SSL context that explicitly uses certifi's CA bundle.
        # macOS Python distributions ship without the system CA chain —
        # this is the standard fix.
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            return _json.loads(resp.read().decode())
    except Exception as exc:
        msg = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in msg:
            msg += (
                "\n\nFix: install certifi:  pip install certifi"
                "\nOr on macOS run:  /Applications/Python\\ 3.x/Install\\ Certificates.command"
            )
        return {"_error": msg}


# Fetch & render
with st.spinner("Fetching runs from GitHub..."):
    payload = fetch_runs(repo, workflow_filter or "", limit=20)

if "_error" in payload:
    err = payload["_error"]
    st.error(f"GitHub API error: {err}")
    if "rate limit" in err.lower() or "403" in err:
        st.info(
            "Hit unauthenticated rate limit (60 req/h). Set `GITHUB_TOKEN` "
            "in your environment for 5,000 req/h. Generate a token at "
            "https://github.com/settings/tokens (only needs `repo` scope for "
            "private repos, no scope for public)."
        )
    if "404" in err:
        st.info("Repo not found. Check the owner/name format above.")
    st.stop()

runs = payload.get("workflow_runs", [])
total = payload.get("total_count", len(runs))

if not runs:
    st.warning(f"No workflow runs found for `{repo}`.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# Summary metrics — last N runs at a glance
# ─────────────────────────────────────────────────────────────────────
def _conclusion_status(conclusion: str) -> str:
    """Map GitHub conclusion → our status_chip color."""
    return {
        "success": "ok",
        "failure": "fail",
        "cancelled": "warn",
        "skipped": "unknown",
        "timed_out": "fail",
        "neutral": "unknown",
        "action_required": "warn",
    }.get(conclusion or "", "unknown")


success_count = sum(1 for r in runs if r.get("conclusion") == "success")
failure_count = sum(1 for r in runs if r.get("conclusion") == "failure")
in_progress = sum(1 for r in runs if r.get("status") == "in_progress")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Runs shown", len(runs))
with m2:
    st.metric("Successful", success_count)
with m3:
    st.metric("Failed", failure_count, delta_color="inverse")
with m4:
    st.metric("In progress", in_progress)

st.divider()


# ─────────────────────────────────────────────────────────────────────
# Filter controls
# ─────────────────────────────────────────────────────────────────────
fc1, fc2 = st.columns([2, 1])
with fc1:
    status_filter = st.multiselect(
        "Filter by conclusion",
        ["success", "failure", "cancelled", "skipped", "timed_out", "in_progress"],
        default=[],
        help="Empty = show all",
    )
with fc2:
    show_per_page = st.selectbox("Show per page", [5, 10, 20, 50], index=2)


def _passes_filter(run: dict) -> bool:
    if not status_filter:
        return True
    if run.get("status") == "in_progress" and "in_progress" in status_filter:
        return True
    return run.get("conclusion") in status_filter


filtered = [r for r in runs if _passes_filter(r)][:show_per_page]


# ─────────────────────────────────────────────────────────────────────
# Run list — one collapsible per run
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"##### Showing {len(filtered)} of {len(runs)} runs")

for run in filtered:
    rid = run.get("id")
    name = run.get("name", "unknown")
    workflow_name = run.get("display_title") or run.get("head_commit", {}).get("message", "")
    workflow_name = (workflow_name or "")[:90]
    conclusion = run.get("conclusion") or run.get("status") or "unknown"
    status = run.get("status", "unknown")
    branch = run.get("head_branch", "?")
    actor = run.get("triggering_actor", {}).get("login", "?")
    event = run.get("event", "?")
    html_url = run.get("html_url", "")

    # Parse timestamps
    created = run.get("created_at", "")
    updated = run.get("updated_at", "")
    duration = ""
    if created and updated:
        try:
            t0 = datetime.fromisoformat(created.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            secs = (t1 - t0).total_seconds()
            if secs >= 60:
                duration = f"{int(secs // 60)}m {int(secs % 60)}s"
            else:
                duration = f"{int(secs)}s"
        except Exception:
            pass

    # Build the expander header
    when = ""
    if created:
        try:
            t = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - t).total_seconds()
            if age < 3600:
                when = f"{int(age / 60)}m ago"
            elif age < 86400:
                when = f"{int(age / 3600)}h ago"
            else:
                when = f"{int(age / 86400)}d ago"
        except Exception:
            when = created

    # Status icon at the start of the header (simple unicode, not emoji)
    icon = {
        "success": "●",
        "failure": "●",
        "cancelled": "○",
        "in_progress": "◐",
    }.get(conclusion, "·")

    header = f"{icon}  Run #{run.get('run_number', '?')}  ·  {conclusion}  ·  {when}  ·  {duration}"

    with st.expander(header):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{name}** — {workflow_name}")
            st.caption(f"Branch: `{branch}`  ·  Trigger: `{event}`  ·  Actor: `{actor}`")
            st.caption(f"Created: {created}  ·  Updated: {updated}")
            if html_url:
                st.markdown(f"[Open in GitHub →]({html_url})")
        with c2:
            status_chip(conclusion, _conclusion_status(conclusion))
            if status == "in_progress":
                status_chip("running", "warn")


st.divider()
st.caption(
    "Polls the GitHub REST API every 60s (cached). Set `GITHUB_TOKEN` "
    "for higher rate limits or to view private repos."
)
