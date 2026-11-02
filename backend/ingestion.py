import os
import hashlib
from typing import Optional, Dict, Any, List
from ingest import clone_repository, fetch_github_issues, parse_repo_url
from chunk_code import extract_code_blocks
from store_vectors import locate_source_files

INDEX_STATUS: Dict[str, Dict[str, Any]] = {}

def parse_repo_identifier(repo_url: str) -> tuple[str, str, str]:
    """Extracts (owner, name, full_name) using shared parsing logic."""
    return parse_repo_url(repo_url)

def run_repo_ingestion(repo_url: str, code_col, history_col, on_ready_callback=None) -> Dict[str, Any]:
    owner, name, full_name = parse_repo_identifier(repo_url)
    
    # Check if already indexed in ChromaDB
    try:
        existing_code = code_col.get(where={"repo": full_name})
        existing_hist = history_col.get(where={"repo": full_name})
        code_count = len(existing_code.get("ids", []))
        hist_count = len(existing_hist.get("ids", []))
        
        if code_count > 0:
            status = {
                "status": "ready",
                "step": 4,
                "total_steps": 4,
                "percent": 100,
                "progress": f"Ready ({code_count} code chunks, {hist_count} history items)",
                "detail": "Loaded existing vectors from ChromaDB",
                "code_chunks": code_count,
                "history_chunks": hist_count,
                "error": None
            }
            INDEX_STATUS[full_name] = status
            if on_ready_callback:
                on_ready_callback(full_name, "ready")
            return status
    except Exception as check_err:
        print(f"Index existence check notice: {check_err}")

    # Step 1: Cloning Repository
    INDEX_STATUS[full_name] = {
        "status": "indexing",
        "step": 1,
        "total_steps": 4,
        "percent": 12,
        "progress": f"Step 1/4: Cloning {full_name} from GitHub...",
        "detail": "Running git clone --depth 1",
        "code_chunks": 0,
        "history_chunks": 0,
        "error": None
    }

    try:
        repo_dir, full_name = clone_repository(repo_url)
        
        # Step 2: Locate & Parse AST Blocks
        src_files = locate_source_files(repo_dir)
        total_files = len(src_files)
        INDEX_STATUS[full_name].update({
            "step": 2,
            "percent": 25,
            "progress": f"Step 2/4: Parsing AST structures from {total_files} source files...",
            "detail": f"Found {total_files} source/doc files in repository"
        })

        all_blocks = []
        for idx, src_file in enumerate(src_files, 1):
            blocks = extract_code_blocks(src_file, full_name)
            all_blocks.extend(blocks)
            pct = 25 + int((idx / max(total_files, 1)) * 30)
            INDEX_STATUS[full_name].update({
                "step": 2,
                "percent": pct,
                "progress": f"Step 2/4: Parsing AST ({idx}/{total_files} files) — {len(all_blocks)} blocks extracted",
                "detail": f"Parsed: {os.path.basename(src_file)}",
                "code_chunks": len(all_blocks)
            })

        # Step 3: Compute FastEmbed Vectors & Upsert into ChromaDB
        total_blocks = len(all_blocks)
        INDEX_STATUS[full_name].update({
            "step": 3,
            "percent": 60,
            "progress": f"Step 3/4: Generating FastEmbed vectors for {total_blocks} code blocks...",
            "detail": "Initializing ONNX BAAI/bge-small-en-v1.5 embeddings",
            "code_chunks": total_blocks
        })

        if all_blocks:
            batch_size = 40
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
                ids = [f"{owner}_{name}_{hashlib.md5((b['file'] + str(b['start_line']) + b['name']).encode()).hexdigest()[:16]}" for b in batch]
                code_col.upsert(documents=docs, metadatas=metas, ids=ids)
                done_chunks = min(i + batch_size, total_blocks)
                pct = 60 + int((done_chunks / max(total_blocks, 1)) * 25)
                INDEX_STATUS[full_name].update({
                    "step": 3,
                    "percent": pct,
                    "progress": f"Step 3/4: Indexed {done_chunks}/{total_blocks} code chunks into ChromaDB...",
                    "detail": f"Batch upserted ({done_chunks} of {total_blocks} chunks)"
                })

        # Step 4: Fetch & Index Closed GitHub Issues/PRs
        INDEX_STATUS[full_name].update({
            "step": 4,
            "percent": 88,
            "progress": f"Step 4/4: Fetching closed GitHub Issues & PRs for {full_name}...",
            "detail": "Querying GitHub REST API for closed issues & pull requests"
        })
        issues = fetch_github_issues(full_name, max_issues=25)
        if issues:
            INDEX_STATUS[full_name].update({
                "percent": 95,
                "progress": f"Step 4/4: Embedding {len(issues)} GitHub issues & PRs into ChromaDB...",
                "detail": f"Fetched {len(issues)} closed issues/PRs",
                "history_chunks": len(issues)
            })
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

        final_status = {
            "status": "ready",
            "step": 4,
            "total_steps": 4,
            "percent": 100,
            "progress": f"Ready! ({len(all_blocks)} code chunks, {len(issues)} issues indexed)",
            "detail": f"Completed indexing for {full_name}",
            "code_chunks": len(all_blocks),
            "history_chunks": len(issues),
            "error": None
        }
        INDEX_STATUS[full_name] = final_status
        if on_ready_callback:
            on_ready_callback(full_name, "ready")
        return final_status

    except Exception as e:
        err_msg = str(e)
        err_status = {
            "status": "error",
            "step": 0,
            "total_steps": 4,
            "percent": 0,
            "progress": f"Indexing failed: {err_msg}",
            "detail": err_msg,
            "code_chunks": 0,
            "history_chunks": 0,
            "error": err_msg
        }
        INDEX_STATUS[full_name] = err_status
        if on_ready_callback:
            on_ready_callback(full_name, "error")
        return err_status