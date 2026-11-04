import os
import hashlib
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from ingest import clone_repository, fetch_github_issues, parse_repo_url
from chunk_code import extract_code_blocks
from store_vectors import scan_repository_files, locate_source_files
from backend.architecture import get_or_build_architecture_model, ARCHITECTURE_MODELS
from db_config import embed_fn

INDEX_STATUS: Dict[str, Dict[str, Any]] = {}

def parse_repo_identifier(repo_url: str) -> tuple[str, str, str]:
    """Extracts (owner, name, full_name) using shared parsing logic."""
    return parse_repo_url(repo_url)

def resolve_repo_dir(target: str) -> Optional[str]:
    """Resolves a repository identifier (full_name, URL, or shorthand) to its local directory path."""
    if not target:
        return None
    if target in ("codebase", "local", "current"):
        local_dir = os.path.abspath(".")
        if os.path.exists(os.path.join(local_dir, "backend")):
            return local_dir
        return None

    owner, name, full_name = parse_repo_identifier(target)
    if full_name == "langchain-ai/langchain" and os.path.exists("./repo_data"):
        return os.path.abspath("./repo_data")

    primary = os.path.abspath(f"./repos/{owner}_{name}")
    if os.path.exists(primary):
        return primary

    repos_base = os.path.abspath("./repos")
    if os.path.exists(repos_base):
        for entry in os.listdir(repos_base):
            full_path = os.path.join(repos_base, entry)
            if not os.path.isdir(full_path):
                continue
            entry_lower = entry.lower()
            name_lower = name.lower()
            owner_name_lower = f"{owner}_{name}".lower()
            if entry_lower == owner_name_lower or entry_lower == name_lower or entry_lower.endswith(f"_{name_lower}"):
                return full_path
    return None

def run_repo_ingestion(repo_url: str, code_col, history_col, on_ready_callback=None) -> Dict[str, Any]:
    owner, name, full_name = parse_repo_identifier(repo_url)
    
    # Check if already indexed in ChromaDB
    try:
        existing_code = code_col.get(where={"repo": full_name})
        existing_hist = history_col.get(where={"repo": full_name})
        code_count = len(existing_code.get("ids", []))
        hist_count = len(existing_hist.get("ids", []))
        
        if code_count > 0:
            repo_dir = resolve_repo_dir(full_name) or os.path.abspath(f"./repos/{owner}_{name}")
            try:
                arch_model = get_or_build_architecture_model(repo_dir, full_name)
                raw_patterns = arch_model.get("detected_patterns") or arch_model.get("patterns") or []
                pattern_names = [p.get("pattern") or p.get("name") if isinstance(p, dict) else str(p) for p in raw_patterns]
                arch_summary = {
                    "health_score": arch_model.get("health_score", 100),
                    "dominant_pattern": arch_model.get("dominant_pattern", "Modular Application"),
                    "total_modules": arch_model.get("total_modules", 0),
                    "violations_count": arch_model.get("conformance", {}).get("total_violations", 0),
                    "detected_patterns": pattern_names
                }
            except Exception:
                arch_summary = {
                    "health_score": 100,
                    "dominant_pattern": "Modular Application",
                    "total_modules": 0,
                    "violations_count": 0,
                    "detected_patterns": []
                }

            status = {
                "status": "ready",
                "step": 4,
                "total_steps": 4,
                "percent": 100,
                "progress": f"Ready ({code_count} code chunks, {hist_count} history items)",
                "detail": "Loaded existing vectors from ChromaDB",
                "code_chunks": code_count,
                "history_chunks": hist_count,
                "architecture": arch_summary,
                "error": None
            }
            INDEX_STATUS[full_name] = status
            if on_ready_callback:
                on_ready_callback(full_name, "ready")
            return status
    except Exception as check_err:
        print(f"Index existence check notice: {check_err}")

    # Step 1: Cloning Repository (5% - 20%)
    INDEX_STATUS[full_name] = {
        "status": "indexing",
        "step": 1,
        "total_steps": 4,
        "percent": 10,
        "progress": f"Step 1/4: Cloning {full_name} from GitHub...",
        "detail": "Running shallow git clone (--depth 1 --single-branch --no-tags)",
        "code_chunks": 0,
        "history_chunks": 0,
        "error": None
    }

    try:
        repo_dir, full_name = clone_repository(repo_url)
        
        INDEX_STATUS[full_name].update({
            "percent": 20,
            "progress": f"Step 1/4: Cloned {full_name} successfully.",
            "detail": f"Local repository ready at {os.path.basename(repo_dir)}"
        })

        # Step 2: Locate Files & Construct Architecture Model in Parallel (20% - 50%)
        core_files, all_files = scan_repository_files(repo_dir, max_core=800, max_total=2500)
        total_core = len(core_files)
        total_all = len(all_files)
        INDEX_STATUS[full_name].update({
            "step": 2,
            "percent": 22,
            "progress": f"Step 2/4: Discovered {total_core} core files ({total_all} total in graph)...",
            "detail": "Starting multi-threaded AST extraction and architecture analysis..."
        })

        all_blocks = []
        if total_core > 0:
            max_workers = min(8, os.cpu_count() or 4)
            # Process core source files in parallel threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(extract_code_blocks, f, full_name): f for f in core_files}
                completed_count = 0
                for future in as_completed(future_to_file):
                    completed_count += 1
                    try:
                        blocks = future.result()
                        all_blocks.extend(blocks)
                    except Exception as parse_err:
                        f_name = os.path.basename(future_to_file[future])
                        print(f"Warning: Failed to parse {f_name}: {parse_err}")

                    if completed_count % 10 == 0 or completed_count == total_core:
                        pct = 20 + int((completed_count / total_core) * 25)
                        INDEX_STATUS[full_name].update({
                            "step": 2,
                            "percent": pct,
                            "progress": f"Step 2/4: Parsing AST ({completed_count}/{total_core} files) — {len(all_blocks)} blocks extracted",
                            "detail": f"Extracted {len(all_blocks)} intact syntax blocks across {completed_count} files",
                            "code_chunks": len(all_blocks)
                        })

        # Run Architecture Intelligence Engine
        INDEX_STATUS[full_name].update({
            "step": 2,
            "percent": 48,
            "progress": f"Step 2/4: Discovering architecture patterns & dependency rules for {full_name}...",
            "detail": f"Analyzing dependency graph ({total_all} files), layer boundaries, and rule conformance..."
        })
        arch_model = get_or_build_architecture_model(repo_dir, full_name, all_files, all_blocks)

        # Inject high-level architectural semantic blocks for direct RAG vector retrieval
        raw_layers = arch_model.get("layers", {})
        if isinstance(raw_layers, dict):
            layer_names = list(raw_layers.keys())
        elif isinstance(raw_layers, list):
            layer_names = [l.get("layer", str(l)) if isinstance(l, dict) else str(l) for l in raw_layers]
        else:
            layer_names = []
        if not layer_names and "layers_detail" in arch_model:
            layer_names = [l.get("layer", "") for l in arch_model.get("layers_detail", []) if isinstance(l, dict) and l.get("layer")]

        layers_display = ", ".join(layer_names) if layer_names else "Presentation, Application, Domain, Infrastructure, Shared / Utility"

        file_naming_info = arch_model.get("conventions", {}).get("file_naming", {})
        conv_name = file_naming_info.get("convention") or file_naming_info.get("dominant_style") or "standard"
        conv_rate = file_naming_info.get("consistency_rate") or f"{file_naming_info.get('consistency_pct', 100)}%"

        arch_summary_code = (
            f"# Architecture Intelligence Model for {full_name}\n"
            f"# Dominant Pattern: {arch_model.get('dominant_pattern', 'Unknown')}\n"
            f"# Architectural Health Score: {arch_model.get('health_score', 100)}/100\n"
            f"# Inferred Architectural Layers: {layers_display}\n"
            f"# Dependency Graph: {arch_model.get('total_modules', 0)} modules, {arch_model.get('total_dependency_edges', 0)} dependency connections\n"
            f"# Conformance Status: {arch_model.get('conformance', {}).get('total_violations', 0)} total rule violations ({arch_model.get('conformance', {}).get('high_severity', 0)} High Severity)\n"
            f"# Code Quality Conventions: File Naming: {conv_name} ({conv_rate})\n"
        )
        all_blocks.append({
            "type": "architecture_model",
            "name": f"Architecture Overview ({arch_model.get('dominant_pattern', 'Multi-Tier')})",
            "code": arch_summary_code,
            "start_line": 1,
            "end_line": 1,
            "file": "ARCHITECTURE_MODEL.md",
            "repo": full_name
        })

        # Step 3: Compute FastEmbed Vectors & Upsert into ChromaDB (50% - 85%)
        total_blocks = len(all_blocks)
        INDEX_STATUS[full_name].update({
            "step": 3,
            "percent": 52,
            "progress": f"Step 3/4: Generating FastEmbed vectors for {total_blocks} code blocks...",
            "detail": "Initializing ONNX BAAI/bge-small-en-v1.5 embeddings",
            "code_chunks": total_blocks
        })

        if all_blocks:
            batch_size = 80
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
                # Guaranteed unique IDs using index to prevent collisions
                ids = [
                    f"{owner}_{name}_{hashlib.md5((b['file'] + str(b['start_line']) + b['name'] + str(i + idx)).encode()).hexdigest()[:16]}"
                    for idx, b in enumerate(batch)
                ]

                # Embed first 1000 chars for 2x faster ONNX tokenization & self-attention,
                # while preserving the full intact code block in ChromaDB documents for synthesis
                embed_texts = [d[:1000] for d in docs]
                embeddings = [e.tolist() for e in embed_fn.model.embed(embed_texts, batch_size=batch_size)]
                # Upsert directly with embeddings to avoid ChromaDB re-embedding & reduce SQLite lock contention
                code_col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

                done_chunks = min(i + batch_size, total_blocks)
                pct = 52 + int((done_chunks / max(total_blocks, 1)) * 33)
                INDEX_STATUS[full_name].update({
                    "step": 3,
                    "percent": pct,
                    "progress": f"Step 3/4: Indexed {done_chunks}/{total_blocks} code chunks into ChromaDB...",
                    "detail": f"Batch upserted ({done_chunks} of {total_blocks} chunks)"
                })
        else:
            INDEX_STATUS[full_name].update({
                "step": 3,
                "percent": 85,
                "progress": "Step 3/4: Zero code chunks to embed.",
                "detail": "Skipping code embedding step"
            })

        # Step 4: Fetch & Index Closed GitHub Issues/PRs (85% - 100%)
        INDEX_STATUS[full_name].update({
            "step": 4,
            "percent": 87,
            "progress": f"Step 4/4: Fetching closed GitHub Issues & PRs for {full_name}...",
            "detail": "Querying GitHub REST API for closed issues & pull requests"
        })
        issues = fetch_github_issues(full_name, max_issues=25)
        if issues:
            INDEX_STATUS[full_name].update({
                "percent": 94,
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
            h_embeddings = [e.tolist() for e in embed_fn.model.embed(h_docs, batch_size=len(h_docs))]
            history_col.upsert(ids=h_ids, documents=h_docs, metadatas=h_metas, embeddings=h_embeddings)

        raw_patterns = arch_model.get("detected_patterns") or arch_model.get("patterns") or []
        pattern_names = []
        for p in raw_patterns:
            if isinstance(p, dict):
                pattern_names.append(p.get("pattern") or p.get("name") or "Architectural Pattern")
            else:
                pattern_names.append(str(p))

        final_status = {
            "status": "ready",
            "step": 4,
            "total_steps": 4,
            "percent": 100,
            "progress": f"Ready! ({len(all_blocks)} code chunks, {len(issues)} issues, Architecture Model built)",
            "detail": f"Completed indexing & architecture analysis for {full_name}",
            "code_chunks": len(all_blocks),
            "history_chunks": len(issues),
            "architecture": {
                "health_score": arch_model.get("health_score", 100),
                "dominant_pattern": arch_model.get("dominant_pattern", "Modular Application"),
                "total_modules": arch_model.get("total_modules", 0),
                "violations_count": arch_model.get("conformance", {}).get("total_violations", 0),
                "detected_patterns": pattern_names
            },
            "error": None
        }
        INDEX_STATUS[full_name] = final_status
        if on_ready_callback:
            on_ready_callback(full_name, "ready")
        return final_status

    except Exception as e:
        err_msg = str(e)
        last_status = INDEX_STATUS.get(full_name, {})
        last_step = last_status.get("step", 1)
        last_pct = last_status.get("percent", 15)
        err_status = {
            "status": "error",
            "step": last_step,
            "total_steps": 4,
            "percent": last_pct,
            "progress": f"Indexing failed at Step {last_step}/4: {err_msg}",
            "detail": err_msg,
            "code_chunks": last_status.get("code_chunks", 0),
            "history_chunks": 0,
            "error": err_msg
        }
        INDEX_STATUS[full_name] = err_status
        if on_ready_callback:
            on_ready_callback(full_name, "error")
        return err_status
