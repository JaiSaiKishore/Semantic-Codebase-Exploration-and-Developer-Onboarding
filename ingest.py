import os
import sys
import subprocess
from typing import List, Dict, Any, Tuple
from github import Github, Auth
from dotenv import load_dotenv

load_dotenv()

def parse_repo_url(repo_url: str) -> Tuple[str, str, str]:
    """Extracts (owner, name, full_name) from any GitHub URL or shorthand."""
    clean = repo_url.strip().rstrip("/")
    # Strip URL queries or anchors
    clean = clean.split("?")[0].split("#")[0].strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    
    if "github.com/" in clean:
        clean = clean.split("github.com/")[-1]
    elif "github.com:" in clean:
        clean = clean.split("github.com:")[-1]

    clean = clean.strip("/")
    parts = clean.split("/")
    if len(parts) >= 2:
        owner = parts[0]
        name = parts[1]
        # Ignore tree/blob segments if pasted (e.g. owner/repo/tree/main)
        return owner, name, f"{owner}/{name}"
    
    return "custom", clean, clean

def clone_repository(repo_url: str) -> Tuple[str, str]:
    """
    Clones a GitHub repository to local disk using an optimized shallow clone
    (--depth 1 --single-branch --no-tags) or reuses/refreshes an existing clone.
    """
    owner, name, full_name = parse_repo_url(repo_url)
    
    if full_name == "langchain-ai/langchain" and os.path.exists("./repo_data"):
        return os.path.abspath("./repo_data"), full_name
        
    target_dir = os.path.abspath(f"./repos/{owner}_{name}")
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    if os.path.exists(os.path.join(target_dir, ".git")):
        print(f"Repository already cloned at {target_dir}. Reusing local files...")
        try:
            # Fast, non-blocking check for updates (5s timeout)
            subprocess.run(
                ["git", "-C", target_dir, "pull", "--ff-only"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
        except Exception:
            pass
    else:
        if os.path.exists(target_dir):
            import shutil, stat
            def rm_ro(func, path, _):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(target_dir, onerror=rm_ro)
            
        print(f"Cloning {full_name} (--depth 1 --single-branch --no-tags) into {target_dir}...")
        clone_url = repo_url if repo_url.startswith(("http://", "https://", "git@")) else f"https://github.com/{full_name}.git"
        
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", "--no-tags", clone_url, target_dir],
            capture_output=True,
            text=True,
            timeout=180
        )
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() if proc.stderr else f"Exit code {proc.returncode}"
            raise RuntimeError(f"Git clone failed: {err_msg}")
        
    return target_dir, full_name

def fetch_github_issues(full_name: str, max_issues: int = 25) -> List[Dict[str, Any]]:
    """Fetches closed issues and pull requests from GitHub REST API, with fallback to unauthenticated public client."""
    token = os.getenv("GITHUB_TOKEN")
    issues = []
    
    print(f"Fetching recent closed Issues and PRs for {full_name} via PyGithub...")
    g = None
    if token:
        try:
            g = Github(auth=Auth.Token(token), timeout=10)
            repo = g.get_repo(full_name)
            issues = list(repo.get_issues(state="closed")[:max_issues])
        except Exception as auth_err:
            print(f"Notice: GITHUB_TOKEN authentication failed ({auth_err}). Falling back to unauthenticated public API...")
            g = None

    if g is None:
        try:
            g = Github(timeout=10)
            repo = g.get_repo(full_name)
            issues = list(repo.get_issues(state="closed")[:max_issues])
        except Exception as unauth_err:
            print(f"Notice: GitHub unauthenticated fetch skipped ({unauth_err})")
            return []

    issue_data = []
    for issue in issues:
        is_pr = bool(issue.pull_request)
        issue_type = "PR" if is_pr else "Issue"
        issue_data.append({
            "number": issue.number,
            "title": issue.title,
            "body": issue.body or "No description provided.",
            "is_pr": is_pr,
            "repo": full_name
        })
        print(f"[{issue_type} #{issue.number}]: {issue.title}")
        
    print(f"Successfully fetched {len(issue_data)} issues/PRs.")
    return issue_data

def run_ingest(repo_url: str):
    target_dir, full_name = clone_repository(repo_url)
    issues = fetch_github_issues(full_name)
    return target_dir, full_name, issues
