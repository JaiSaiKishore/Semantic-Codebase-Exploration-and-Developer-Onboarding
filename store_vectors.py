import os
import sys
import hashlib
from typing import List, Dict, Any
from db_config import code_col, history_col
from ingest import clone_repository, fetch_github_issues, parse_repo_url
from chunk_code import extract_code_blocks, extract_ast_blocks

SUPPORTED_CODE_EXTS = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".md",
    ".html", ".css", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt"
}

def locate_source_files(repo_dir: str) -> List[str]:
    """Locates all source code and documentation files, excluding build/dependency directories."""
    found = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in (
            ".git", "node_modules", "venv", ".venv", "__pycache__", 
            "dist", "build", "coverage", ".github", ".system_generated", "target", "vendor"
        )]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_CODE_EXTS:
                found.append(os.path.join(root, f))
    return found[:250]

def store_repository_vectors(repo_url: str) -> Dict[str, Any]:
    owner, name, full_name = parse_repo_url(repo_url)
    print(f"\n==========================================")
    print(f"Indexing Repository Pipeline: {full_name}")
    print(f"==========================================")

    repo_dir, full_name = clone_repository(repo_url)
    
    print(f"Locating code and doc files in {repo_dir}...")
    core_files = locate_source_files(repo_dir)
    print(f"Extracting semantic AST chunks from {len(core_files)} files...")

    all_blocks = []
    for f in core_files:
        blocks = extract_code_blocks(f, full_name)
        all_blocks.extend(blocks)
    print(f"Extracted {len(all_blocks)} intact AST blocks.")

    if all_blocks:
        print(f"Generating FastEmbed vectors and upserting {len(all_blocks)} chunks into codebase_chunks...")
        batch_size = 50
        for i in range(0, len(all_blocks), batch_size):
            batch = all_blocks[i:i+batch_size]
            docs = [b["code"] for b in batch]
            metas = [{
                "file": b["file"],
                "name": b["name"],
                "type": b["type"],
                "start_line": b["start_line"],
                "end_line": b["end_line"],
                "repo": full_name
            } for b in batch]
            ids = [f"{owner}_{name}_{hashlib.md5((b['file'] + str(b['start_line']) + b['name']).encode()).hexdigest()[:16]}" for b in batch]
            code_col.upsert(documents=docs, metadatas=metas, ids=ids)
        print("Code chunks indexed successfully.")

    issues = fetch_github_issues(full_name, max_issues=25)
    if issues:
        print(f"Generating FastEmbed vectors and upserting {len(issues)} issues into github_history...")
        h_docs, h_metas, h_ids = [], [], []
        for issue in issues:
            h_docs.append(f"Repo: {full_name}\nType: {'PR' if issue['is_pr'] else 'Issue'} #{issue['number']}\nTitle: {issue['title']}\nBody: {issue['body'][:1200]}")
            h_metas.append({
                "repo": full_name,
                "number": issue["number"],
                "title": issue["title"],
                "is_pr": issue["is_pr"]
            })
            h_ids.append(f"hist_{owner}_{name}_{issue['number']}")
        history_col.upsert(documents=h_docs, metadatas=h_metas, ids=h_ids)
        print("GitHub issues indexed successfully.")

    result = {
        "repo": full_name,
        "repo_dir": repo_dir,
        "code_chunks": len(all_blocks),
        "history_chunks": len(issues),
        "total_codebase_chunks_in_db": code_col.count(),
        "total_history_chunks_in_db": history_col.count()
    }
    print(f"\nPipeline finished: {result}")
    return result

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/sindresorhus/yocto-queue.git"
    store_repository_vectors(url)