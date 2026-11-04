import os
import sys
import hashlib
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from db_config import code_col, history_col, embed_fn
from ingest import clone_repository, fetch_github_issues, parse_repo_url
from chunk_code import extract_code_blocks, extract_ast_blocks

SUPPORTED_CODE_EXTS = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".md",
    ".html", ".css", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt"
}

EXCLUDED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", 
    "dist", "build", "coverage", ".github", ".system_generated", "target", "vendor",
    ".next", ".nuxt", ".turbo", ".cache", ".pytest_cache", ".idea", ".vscode",
    "out", "eggs", ".mypy_cache", ".tox", "site-packages", ".trunk", ".circleci",
    "repos", "repo_data", "chroma_db", ".chroma", "brain"
}

SKIP_FILE_SUFFIXES = (
    ".min.js", ".min.css", ".bundle.js", ".map",
    "-lock.json", ".lock", "lock.yaml"
)

AUXILIARY_DIR_NAMES = {
    "tests", "test", "__tests__", "spec", "specs", "fixtures", "testdata",
    "mocks", "stubs", "e2e", "cypress", "migrations", "alembic", "benchmarks", 
    "benchmark", "docs", "documentation", "examples", "samples"
}

def scan_repository_files(repo_dir: str, max_core: int = 800, max_total: int = 2500) -> Tuple[List[str], List[str]]:
    """
    Scans repository files and partitions them into:
      1. core_files: Primary application logic, models, services, controllers, interfaces, and utilities.
      2. all_files: Full repository files (core + test suites & fixtures) for 100% dependency graph analysis.
    Accelerates large repository indexing by focusing vector embeddings on high-signal architectural code.
    """
    core_files = []
    auxiliary_files = []

    for root, dirs, files in os.walk(repo_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

        rel_dir = os.path.relpath(root, repo_dir).replace("\\", "/").lower()
        dir_parts = set(rel_dir.split("/"))
        is_aux = bool(dir_parts.intersection(AUXILIARY_DIR_NAMES))

        for f in files:
            f_lower = f.lower()
            if any(f_lower.endswith(sfx) for sfx in SKIP_FILE_SUFFIXES):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_CODE_EXTS:
                full_path = os.path.join(root, f)
                try:
                    if os.path.getsize(full_path) <= 500_000:
                        is_test_file = any(t in f_lower for t in ["test_", "_test.", ".test.", ".spec.", "_spec."])
                        if is_aux or is_test_file:
                            if len(auxiliary_files) + len(core_files) < max_total:
                                auxiliary_files.append(full_path)
                        else:
                            if len(core_files) < max_core:
                                core_files.append(full_path)
                except OSError:
                    continue

        if len(core_files) >= max_core and (len(core_files) + len(auxiliary_files)) >= max_total:
            break

    # If repo has very few core files, promote auxiliary files so small test-heavy repos are fully indexed
    if len(core_files) < 30 and auxiliary_files:
        core_files.extend(auxiliary_files[:max_core - len(core_files)])

    all_files = list(dict.fromkeys(core_files + auxiliary_files))
    return core_files, all_files

def locate_source_files(repo_dir: str, max_files: int = 800) -> List[str]:
    """Locates primary core source files for vector indexing, pruning heavy test suites for speed."""
    core_files, _ = scan_repository_files(repo_dir, max_core=max_files, max_total=max_files * 2)
    return core_files[:max_files]

def store_repository_vectors(repo_url: str) -> Dict[str, Any]:
    owner, name, full_name = parse_repo_url(repo_url)
    print(f"\n==========================================")
    print(f"Indexing Repository Pipeline: {full_name}")
    print(f"==========================================")

    repo_dir, full_name = clone_repository(repo_url)
    
    print(f"Locating code and doc files in {repo_dir}...")
    core_files = locate_source_files(repo_dir)
    print(f"Extracting semantic AST chunks from {len(core_files)} files (threaded)...")

    all_blocks = []
    if core_files:
        max_workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            file_results = list(executor.map(lambda f: extract_code_blocks(f, full_name), core_files))
        for blocks in file_results:
            all_blocks.extend(blocks)
    print(f"Extracted {len(all_blocks)} intact AST blocks.")

    if all_blocks:
        print(f"Generating FastEmbed vectors and upserting {len(all_blocks)} chunks into codebase_chunks...")
        batch_size = 80
        total_blocks = len(all_blocks)
        for i in range(0, total_blocks, batch_size):
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
            ids = [
                f"{owner}_{name}_{hashlib.md5((b['file'] + str(b['start_line']) + b['name'] + str(i + idx)).encode()).hexdigest()[:16]}"
                for idx, b in enumerate(batch)
            ]
            # Embed first 1000 chars for 2x faster ONNX tokenization & self-attention,
            # while preserving the full intact code block in ChromaDB documents for synthesis
            embed_texts = [d[:1000] for d in docs]
            embeddings = [e.tolist() for e in embed_fn.model.embed(embed_texts, batch_size=batch_size)]
            code_col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
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
        h_embeddings = [e.tolist() for e in embed_fn.model.embed(h_docs, batch_size=len(h_docs))]
        history_col.upsert(ids=h_ids, documents=h_docs, metadatas=h_metas, embeddings=h_embeddings)
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