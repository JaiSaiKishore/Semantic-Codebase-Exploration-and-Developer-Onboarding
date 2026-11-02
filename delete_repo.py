import sys
import os
import stat
import shutil
import urllib.parse
import urllib.request
from ingest import parse_repo_url
from db_config import code_col, history_col

def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def list_indexed_repos():
    repos = set()
    try:
        c_data = code_col.get(include=["metadatas"])
        for m in c_data.get("metadatas", []):
            if m and m.get("repo"):
                repos.add(m["repo"])
        h_data = history_col.get(include=["metadatas"])
        for m in h_data.get("metadatas", []):
            if m and m.get("repo"):
                repos.add(m["repo"])
    except Exception:
        pass
    return sorted(list(repos))

def delete_repository_data(repo_identifier: str):
    owner, name, full_name = parse_repo_url(repo_identifier)
    print(f"\nDeleting all stored data for: {full_name}...")

    # 1. Notify running FastAPI server if active
    try:
        url = f"http://localhost:8000/api/repository?repo_name={urllib.parse.quote(full_name)}"
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print(f"  [OK] Cleared {full_name} from running server cache.")
    except Exception:
        pass

    # 2. Delete code chunks from ChromaDB
    try:
        existing_c = code_col.get(where={"repo": full_name})
        c_ids = existing_c.get("ids", [])
        if c_ids:
            code_col.delete(ids=c_ids)
        print(f"  [OK] Removed {len(c_ids)} code chunks from ChromaDB.")
    except Exception as e:
        print(f"  [-] ChromaDB code notice: {e}")

    # 3. Delete GitHub issue chunks from ChromaDB
    try:
        existing_h = history_col.get(where={"repo": full_name})
        h_ids = existing_h.get("ids", [])
        if h_ids:
            history_col.delete(ids=h_ids)
        print(f"  [OK] Removed {len(h_ids)} issue & PR chunks from ChromaDB.")
    except Exception as e:
        print(f"  [-] ChromaDB history notice: {e}")

    # 4. Delete cloned repository directory from disk
    repo_dir = os.path.abspath(f"./repos/{owner}_{name}")
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, onerror=remove_readonly)
        print(f"  [OK] Deleted cloned directory: {repo_dir}")
    else:
        print(f"  [-] No cloned directory at: {repo_dir}")

    print(f"\nDone! Repository '{full_name}' has been completely removed.")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        target = sys.argv[1].strip()
        if target.lower() in ("--all", "all"):
            from reset_db import reset_all_data
            reset_all_data()
        else:
            delete_repository_data(target)
    else:
        repos = list_indexed_repos()
        print("==================================================")
        print("        DELETE REPOSITORY DATA UTILITY            ")
        print("==================================================")
        if not repos:
            print("No repositories are currently indexed in ChromaDB.")
            print("\nUsage: python delete_repo.py <owner/repo or github_url>")
        else:
            print("Currently indexed repositories in ChromaDB:")
            for idx, r in enumerate(repos, 1):
                print(f"  [{idx}] {r}")
            print("  [all] Delete ALL repositories")
            choice = input("\nEnter repository name, number, or 'all' to delete (or press Enter to cancel): ").strip()
            if choice.lower() == "all":
                from reset_db import reset_all_data
                reset_all_data()
            elif choice.isdigit() and 1 <= int(choice) <= len(repos):
                delete_repository_data(repos[int(choice) - 1])
            elif choice:
                delete_repository_data(choice)
            else:
                print("Cancelled.")