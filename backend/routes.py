import os
import hmac
import hashlib
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import workflow, code_col, history_col
from backend.ingestion import parse_repo_identifier, run_repo_ingestion, INDEX_STATUS, resolve_repo_dir
from backend.architecture import get_or_build_architecture_model, ARCHITECTURE_MODELS
from store_vectors import scan_repository_files

compiled_app = {}
current_repository = {
    "url": "",
    "name": "",
    "status": "unset"
}

GITHUB_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "my_super_secret_webhook_key")

def update_current_repo_status(repo_name: str, new_status: str):
    if current_repository.get("name") == repo_name or not current_repository.get("name"):
        current_repository["name"] = repo_name
        current_repository["status"] = new_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints.db"))
    with SqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
        compiled_app["agent"] = workflow.compile(checkpointer=checkpointer)
        print("Database connection opened and Graph compiled.")
        yield
    print("Database connection closed.")

app = FastAPI(title="Semantic Codebase RAG API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_index = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "index.html")

# ── 1. Frontend Endpoints ────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
@app.get("", response_class=FileResponse)
async def serve_index():
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    raise HTTPException(status_code=404, detail="frontend/index.html not found")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/api/health")
async def health_check():
    agent_ready = "agent" in compiled_app and compiled_app["agent"] is not None
    return {
        "status": "healthy" if agent_ready else "initializing",
        "agent_ready": agent_ready,
        "database": "connected" if agent_ready else "disconnected",
        "repository": current_repository
    }

# ── 2. Repository Management Endpoints ───────────────────────────────────
class RepositoryRequest(BaseModel):
    repo_url: str

@app.get("/api/repository")
async def get_repository():
    return current_repository

@app.get("/api/repository/status")
async def get_repo_status(repo_name: Optional[str] = None):
    target = repo_name or current_repository["name"]
    if not target:
        return {
            "repo_name": "",
            "status": "unset",
            "step": 0,
            "total_steps": 4,
            "percent": 0,
            "progress": "No repository set",
            "detail": "",
            "code_chunks": 0,
            "history_chunks": 0,
            "error": None
        }

    _, _, normalized = parse_repo_identifier(target)
    status_info = INDEX_STATUS.get(normalized) or INDEX_STATUS.get(target)
    if not status_info:
        try:
            res = code_col.get(where={"repo": normalized})
            h_res = history_col.get(where={"repo": normalized})
            code_count = len(res.get("ids", []))
            hist_count = len(h_res.get("ids", []))

            if code_count > 0:
                status_info = {
                    "status": "ready",
                    "step": 4,
                    "total_steps": 4,
                    "percent": 100,
                    "progress": f"Ready ({code_count} code chunks, {hist_count} history items)",
                    "detail": f"Indexed and ready in ChromaDB",
                    "code_chunks": code_count,
                    "history_chunks": hist_count,
                    "error": None
                }
                current_repository["name"] = normalized
                current_repository["status"] = "ready"
            else:
                status_info = {
                    "status": "not_indexed",
                    "step": 0,
                    "total_steps": 4,
                    "percent": 0,
                    "progress": "Not indexed yet",
                    "detail": "",
                    "code_chunks": 0,
                    "history_chunks": 0,
                    "error": None
                }
        except Exception as e:
            status_info = {
                "status": "not_indexed",
                "step": 0,
                "total_steps": 4,
                "percent": 0,
                "progress": f"Status check notice: {str(e)}",
                "detail": "",
                "code_chunks": 0,
                "history_chunks": 0,
                "error": None
            }
        INDEX_STATUS[normalized] = status_info

    return {"repo_name": normalized, **status_info}

@app.get("/api/repositories")
async def list_repositories():
    try:
        data = code_col.get(include=["metadatas"])
        repos = set(m.get("repo") for m in data.get("metadatas", []) if m and m.get("repo"))
        return {"repositories": sorted(list(repos))}
    except Exception:
        return {"repositories": []}

@app.post("/api/repository")
async def set_and_index_repository(request: RepositoryRequest, background_tasks: BackgroundTasks):
    repo_url = request.repo_url.strip()
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repository URL is required")
        
    owner, name, full_name = parse_repo_identifier(repo_url)
    current_repository["url"] = repo_url
    current_repository["name"] = full_name

    # Check if already indexed in ChromaDB
    try:
        existing_code = code_col.get(where={"repo": full_name})
        existing_hist = history_col.get(where={"repo": full_name})
        code_cnt = len(existing_code.get("ids", []))
        hist_cnt = len(existing_hist.get("ids", []))
        
        if code_cnt > 0:
            current_repository["status"] = "ready"
            INDEX_STATUS[full_name] = {
                "status": "ready",
                "step": 4,
                "total_steps": 4,
                "percent": 100,
                "progress": f"Ready ({code_cnt} code chunks, {hist_cnt} history items)",
                "detail": "Already indexed in ChromaDB",
                "code_chunks": code_cnt,
                "history_chunks": hist_cnt,
                "error": None
            }
            return {
                "status": "ready",
                "repository": current_repository,
                "code_chunks": code_cnt,
                "history_chunks": hist_cnt,
                "message": f"Repository {full_name} is already indexed ({code_cnt} code chunks, {hist_cnt} issues) and ready!"
            }
    except Exception:
        pass

    # Check if already indexing
    existing_status = INDEX_STATUS.get(full_name)
    if existing_status and existing_status.get("status") == "indexing":
        return {
            "status": "indexing",
            "repository": current_repository,
            "repo_name": full_name,
            "message": existing_status.get("progress", f"Indexing {full_name} is already in progress...")
        }

    # Queue background indexing task
    current_repository["status"] = "indexing"
    INDEX_STATUS[full_name] = {
        "status": "indexing",
        "step": 1,
        "total_steps": 4,
        "percent": 8,
        "progress": f"Step 1/4: Cloning {full_name} from GitHub...",
        "detail": "Starting background ingestion pipeline...",
        "code_chunks": 0,
        "history_chunks": 0,
        "error": None
    }
    background_tasks.add_task(run_repo_ingestion, repo_url, code_col, history_col, update_current_repo_status)

    return {
        "status": "indexing",
        "repository": current_repository,
        "repo_name": full_name,
        "message": f"Step 1/4: Cloning {full_name} and starting AST pipeline..."
    }

@app.delete("/api/repository")
async def delete_repository(repo_name: str):
    owner, name, full_name = parse_repo_identifier(repo_name)
    try:
        c_res = code_col.get(where={"repo": full_name})
        if c_res and c_res.get("ids"):
            code_col.delete(ids=c_res["ids"])
        h_res = history_col.get(where={"repo": full_name})
        if h_res and h_res.get("ids"):
            history_col.delete(ids=h_res["ids"])
        repo_dir = os.path.join(".", "repos", f"{owner}_{name}")
        if os.path.exists(repo_dir):
            import shutil, stat
            def rm_ro(func, path, _):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(repo_dir, onerror=rm_ro)
        INDEX_STATUS.pop(full_name, None)
        ARCHITECTURE_MODELS.pop(full_name, None)
        if current_repository.get("name") == full_name:
            current_repository["name"] = ""
            current_repository["url"] = ""
            current_repository["status"] = "unset"
        return {"status": "success", "message": f"Deleted all data for {full_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_all_endpoint():
    try:
        c_data = code_col.get()
        if c_data and c_data.get("ids"):
            ids = c_data["ids"]
            for i in range(0, len(ids), 500):
                code_col.delete(ids=ids[i:i+500])
        h_data = history_col.get()
        if h_data and h_data.get("ids"):
            ids = h_data["ids"]
            for i in range(0, len(ids), 500):
                history_col.delete(ids=ids[i:i+500])
        INDEX_STATUS.clear()
        ARCHITECTURE_MODELS.clear()
        current_repository["url"] = ""
        current_repository["name"] = ""
        current_repository["status"] = "unset"
        return {"status": "success", "message": "All repository caches and collections cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 2B. Architecture Intelligence Endpoints ──────────────────────────────
def _resolve_architecture_model(repo_identifier: Optional[str], force_refresh: bool = False) -> Dict[str, Any]:
    target = repo_identifier or current_repository["name"]
    if not target:
        raise HTTPException(status_code=400, detail="No repository specified. Please set a repository first.")
    owner, name, full_name = parse_repo_identifier(target)
    canonical_name = full_name if target not in ("codebase", "local", "current") else "codebase"

    # 1. Check in-memory cache (only if not forcing refresh)
    if not force_refresh and canonical_name in ARCHITECTURE_MODELS:
        return ARCHITECTURE_MODELS[canonical_name]

    # 2. Check disk repository directory via robust resolver
    repo_dir = resolve_repo_dir(target)
    if repo_dir and os.path.exists(repo_dir):
        _, all_files = scan_repository_files(repo_dir, max_core=800, max_total=2500)
        if all_files:
            return get_or_build_architecture_model(repo_dir, canonical_name, all_files, force_refresh=force_refresh)

    # 3. Check if repository is currently indexing
    status_info = INDEX_STATUS.get(full_name)
    if status_info and status_info.get("status") == "indexing":
        pct = status_info.get("percent", 0)
        raise HTTPException(
            status_code=202, 
            detail=f"Repository '{full_name}' is currently being indexed ({pct}% completed). Architecture analysis will be ready once ingestion completes."
        )

    raise HTTPException(
        status_code=404, 
        detail=f"Architecture model not found for '{full_name}'. Please ensure repository is indexed."
    )

@app.get("/api/architecture")
async def get_repository_architecture(repo_name: Optional[str] = None, refresh: bool = False):
    return _resolve_architecture_model(repo_name, force_refresh=refresh)

@app.post("/api/architecture/reanalyze")
async def reanalyze_repository_architecture(repo_name: Optional[str] = None):
    return _resolve_architecture_model(repo_name, force_refresh=True)

@app.get("/api/architecture/conformance")
async def get_architecture_conformance(repo_name: Optional[str] = None):
    model = _resolve_architecture_model(repo_name)
    return {
        "repository": model.get("repository"),
        "health_score": model.get("health_score"),
        "dominant_pattern": model.get("dominant_pattern"),
        "conformance": model.get("conformance"),
        "cycles": model.get("cycles")
    }

@app.get("/api/architecture/graph")
async def get_architecture_graph(repo_name: Optional[str] = None):
    model = _resolve_architecture_model(repo_name)
    return {
        "repository": model.get("repository"),
        "graph": model.get("graph"),
        "coupling_metrics": model.get("coupling_metrics")
    }

@app.get("/api/architecture/conventions")
async def get_architecture_conventions(repo_name: Optional[str] = None):
    model = _resolve_architecture_model(repo_name)
    return {
        "repository": model.get("repository"),
        "conventions": model.get("conventions")
    }

# ── 3. Chat Endpoint ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question: str
    repo: Optional[str] = None

class ChatResponse(BaseModel):
    route: str
    answer: str
    context: Optional[str] = None
    trace: Optional[List[str]] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    agent = compiled_app.get("agent")
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    raw_repo = request.repo or current_repository["name"]
    if not raw_repo:
        raise HTTPException(status_code=400, detail="No repository specified. Please set a repository first.")
        
    _, _, repo_target = parse_repo_identifier(raw_repo)

    # Check if target repository is still indexing
    status_info = INDEX_STATUS.get(repo_target)
    if status_info and status_info.get("status") == "indexing":
        pct = status_info.get("percent", 0)
        return ChatResponse(
            route="indexing",
            answer=f"Repository **{repo_target}** is currently indexing ({pct}% completed). Please wait until the ingestion pipeline finishes before asking questions about this codebase.",
            context="",
            trace=[f"Active target repository: {repo_target}", f"Indexing status: {pct}% complete"]
        )

    config = {"configurable": {"thread_id": request.session_id}}
    
    try:
        result = await asyncio.to_thread(agent.invoke, {"question": request.question, "repo": repo_target}, config=config)
        return ChatResponse(
            route=result.get("route", "codebase"),
            answer=result.get("generation", "No answer generated."),
            context=result.get("context", ""),
            trace=result.get("trace", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 4. Webhook and Background Tree-Sitter Re-indexing ─────────────────────
def process_github_update(payload: dict):
    try:
        repo_name = payload.get("repository", {}).get("full_name")
        commits = payload.get("commits", [])
        files_to_update = set()
        for c in commits:
            for f in c.get("added", []) + c.get("modified", []):
                files_to_update.add(f)
                    
        if not files_to_update:
            print(f"  [Webhook] No code files modified in payload for {repo_name}.")
            return

        from chunk_code import extract_code_blocks
        owner, name, full_name = parse_repo_identifier(repo_name or "")
        repo_dir = os.path.join(".", "repo_data") if full_name == "langchain-ai/langchain" else os.path.join(".", "repos", f"{owner}_{name}")
        
        print(f"  [Webhook Sync] Fetching git delta for {full_name}...")
        if os.path.exists(os.path.join(repo_dir, ".git")):
            import subprocess
            try:
                subprocess.run(["git", "-C", repo_dir, "pull", "--ff-only"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            except Exception:
                pass

        total_upserted = 0
        for rel_path in files_to_update:
            full_path = os.path.join(repo_dir, rel_path)
            if os.path.exists(full_path):
                blocks = extract_code_blocks(full_path, full_name)
                if blocks:
                    ids = [f"{owner}_{name}_{hashlib.md5((b['file'] + str(b['start_line']) + b['name']).encode()).hexdigest()[:16]}" for b in blocks]
                    docs = [b["code"] for b in blocks]
                    metas = [{
                        "file": b["file"],
                        "name": b["name"],
                        "type": b["type"],
                        "start_line": b["start_line"],
                        "end_line": b["end_line"],
                        "repo": full_name
                    } for b in blocks]
                    code_col.upsert(documents=docs, metadatas=metas, ids=ids)
                    total_upserted += len(blocks)
                    print(f"  [Incremental AST Upsert] File '{rel_path}': {len(blocks)} AST chunks upserted into codebase_chunks.")
        print(f"  [Webhook Done] Total {total_upserted} chunks synchronized in ChromaDB for {full_name}.\n")
    except Exception as e:
        print(f"Error in process_github_update: {e}")

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature")
        
    body = await request.body()
    mac = hmac.new(GITHUB_SECRET.encode(), msg=body, digestmod=hashlib.sha256)
    expected_signature = "sha256=" + mac.hexdigest()
    
    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    event = request.headers.get("X-GitHub-Event", "push")
    repo_name = payload.get("repository", {}).get("full_name", "unknown/repo")
    
    print("\n" + "=" * 60)
    print(" [GITHUB WEBHOOK EVENT LOG]")
    print(f" Event: {event}")
    print(f" Repository: {repo_name}")
    print(f" HMAC SHA-256 Signature: {signature[:18]}... [VERIFIED OK]")
    print("=" * 60)
    
    if event in ["push", "pull_request"]:
        background_tasks.add_task(process_github_update, payload)
        
    return {"status": "Accepted", "event": event, "repository": repo_name, "hmac_verified": True}

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")