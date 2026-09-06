import os
import subprocess
from github import Github
from github import Auth
from dotenv import load_dotenv

load_dotenv()

# Target repository for our testbed
TARGET_REPO_URL = "https://github.com/langchain-ai/langchain"
TARGET_REPO_NAME = "langchain-ai/langchain"
LOCAL_DIR = "./repo_data"

# 1. Clone the raw source code locally
if not os.path.exists(LOCAL_DIR):
    print(f"Cloning repository into {LOCAL_DIR}...")
    subprocess.run(["git", "clone", TARGET_REPO_URL, LOCAL_DIR])
else:
    print("Repository already cloned locally.")

# 2. Authenticate with GitHub API to fetch context
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set the GITHUB_TOKEN environment variable.")

auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)
repo = g.get_repo(TARGET_REPO_NAME)

print(f"\nFetching recent closed Issues and PRs for {TARGET_REPO_NAME}...")
issues = repo.get_issues(state="closed")

# Fetching the first 20 for initial testing to avoid long wait times
for issue in issues[:20]:
    issue_type = "PR" if issue.pull_request else "Issue"
    print(f"[{issue_type} #{issue.number}]: {issue.title}")