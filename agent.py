import re
import os
from typing import Dict, TypedDict, Annotated, Optional, List, Tuple
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from db_config import client, embed_fn, code_col, history_col

load_dotenv()

# 1. Initialize Cross-Encoder Re-ranker
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. Initialize Groq LLM
groq_api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0,
    api_key=groq_api_key
)

# 3. Define Graph State
class GraphState(TypedDict):
    messages: Annotated[list, add_messages] 
    question: str
    repo: Optional[str]
    route: str
    context: str
    context_found: bool
    search_keywords: Optional[str]
    trace: List[str]
    generation: str

# Regex patterns for deterministic word-boundary routing
HISTORY_PATTERN = re.compile(
    r'\b(cve|cves|vulnerability|vulnerabilities|pull request|pull requests|pr|prs|issue|issues|bug|bugs|commit|commits|patch|patches|changelog)\b',
    re.IGNORECASE
)
CODE_PATTERN = re.compile(
    r'\b(code|implementation|implement|function|functions|class|classes|method|methods|constructor|queue|node|stack|list|data structure|file|files|ast|syntax|algorithm|logic|how it works|flow|architecture|component|components|structure|primary|overview)\b',
    re.IGNORECASE
)

def normalize_repo_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    clean = raw.strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    if "github.com/" in clean:
        clean = clean.split("github.com/")[-1]
    clean = clean.strip("/")
    parts = clean.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return clean

def get_repo_metadata_summary(repo: Optional[str]) -> str:
    """Returns a compact summary of files and symbol names indexed for the target repo."""
    if not repo:
        return "Unknown files"
    try:
        data = code_col.get(where={"repo": repo}, include=["metadatas"])
        metas = data.get("metadatas", [])
        files = sorted(list({os.path.basename(m.get("file", "")) for m in metas if m and m.get("file")}))
        symbols = sorted(list({m.get("name", "") for m in metas if m and m.get("name")}))
        return f"Files: {', '.join(files[:25])} | Symbols: {', '.join(symbols[:40])}"
    except Exception:
        return "Unknown files"

# ── Node 1: Intent Router ────────────────────────────────────────────────
def route_question(state: GraphState):
    question = state["question"].strip()
    repo = normalize_repo_name(state.get("repo")) or "codebase"
    trace = [f"Active target repository: {repo}"]
    
    has_history = bool(HISTORY_PATTERN.search(question))
    has_code = bool(CODE_PATTERN.search(question))
    
    if has_history and not has_code:
        route = "github_history"
        trace.append("Query routed to: github_history (matched issue/PR/history intent)")
        return {"route": route, "trace": trace}
    if has_code and not has_history:
        route = "codebase"
        trace.append("Query routed to: codebase (matched code/AST/architecture intent)")
        return {"route": route, "trace": trace}
        
    system = """You are a strict routing assistant. 
Analyze the user's latest question AND the provided chat history.
If the conversation topic is primarily about a vulnerability, CVE, bug report, pull request, or GitHub discussion history, reply ONLY with 'github_history'. 
If the topic is about code implementation, functions, algorithms, architecture, or how a specific class/method works, reply ONLY with 'codebase'."""
    
    messages_to_llm = [SystemMessage(content=system)] + state.get("messages", []) + [HumanMessage(content=question)]
    
    route = "codebase"
    try:
        response = llm.invoke(messages_to_llm)
        if "history" in response.content.strip().lower():
            route = "github_history"
    except Exception as e:
        print(f"Router LLM fallback notice: {e}")
        
    trace.append(f"Query routed to: {route}")
    return {"route": route, "trace": trace}

# ── Node 2A: Stage 1 Codebase Retrieval ──────────────────────────────────
def retrieve_code(state: GraphState):
    repo = normalize_repo_name(state.get("repo"))
    query = state["question"]
    trace = list(state.get("trace", []))
    docs, metas = [], []
    
    try:
        if repo:
            results = code_col.query(query_texts=[query], n_results=25, where={"repo": repo})
        else:
            results = code_col.query(query_texts=[query], n_results=25)
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
    except Exception as e:
        print(f"Code retrieval notice: {e}")

    if not docs:
        trace.append("Stage 1: Direct vector search in codebase_chunks returned 0 documents")
        return {"context": "", "context_found": False, "trace": trace}

    # Score all retrieved documents against the raw user query using Cross-Encoder
    all_scores = cross_encoder.predict([[query, d] for d in docs])
    max_score = float(max(all_scores)) if len(all_scores) > 0 else -10.0

    # Check if the query is a standard architectural/overview query or has strong semantic relevance
    is_broad_code_query = bool(CODE_PATTERN.search(query))
    # Cross-Encoder logits: > -3.5 indicates direct relevance; broad code queries on small repos may score around -5.0
    threshold = -6.0 if is_broad_code_query else -3.5

    if max_score < threshold:
        trace.append(f"Stage 1: Direct vector search had low relevance (best score: {max_score:.2f} < {threshold}) -> Triggering LLM keyword expansion")
        return {"context": "", "context_found": False, "trace": trace}

    # Stratified balancing between code chunks and documentation chunks
    code_candidates: List[Tuple[str, dict, float]] = []
    doc_candidates: List[Tuple[str, dict, float]] = []
    
    for i, d in enumerate(docs):
        m = metas[i] if i < len(metas) else {}
        s = float(all_scores[i])
        fname = m.get("file", "").lower()
        mtype = m.get("type", "").lower()
        if fname.endswith(".md") or mtype == "markdown":
            doc_candidates.append((d, m, s))
        else:
            fpath = fname
            is_test = "test" in fpath or "spec" in fpath
            query_wants_test = "test" in query.lower()
            adj_s = s if (is_test and query_wants_test) or not is_test else s - 2.0
            code_candidates.append((d, m, adj_s))

    selected_chunks = []
    if code_candidates:
        code_candidates.sort(key=lambda x: x[2], reverse=True)
        selected_chunks.extend([c[0] for c in code_candidates[:5]])
    if doc_candidates:
        doc_candidates.sort(key=lambda x: x[2], reverse=True)
        selected_chunks.extend([d[0] for d in doc_candidates[:2]])

    context = "\n\n".join(selected_chunks)
    trace.append(f"Stage 1: Retrieved & re-ranked {len(selected_chunks)} matching chunks (top score: {max_score:.2f})")
    return {"context": context, "context_found": True, "trace": trace}

# ── Node 2B: Stage 1 GitHub History Retrieval ────────────────────────────
def retrieve_history(state: GraphState):
    repo = normalize_repo_name(state.get("repo"))
    query = state["question"]
    trace = list(state.get("trace", []))
    docs = []
    
    try:
        if repo:
            results = history_col.query(query_texts=[query], n_results=15, where={"repo": repo})
        else:
            results = history_col.query(query_texts=[query], n_results=15)
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
    except Exception as e:
        print(f"History retrieval notice: {e}")

    if not docs:
        trace.append(f"Stage 1: No closed GitHub issues/PRs found in github_history for {repo} -> Triggering LLM keyword search")
        return {"context": "", "context_found": False, "trace": trace}

    scores = cross_encoder.predict([[query, d] for d in docs])
    max_score = float(max(scores)) if len(scores) > 0 else -10.0

    if max_score < -6.5:
        trace.append(f"Stage 1: GitHub history items had low relevance ({max_score:.2f}) -> Triggering LLM keyword search")
        return {"context": "", "context_found": False, "trace": trace}

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:5]]
    trace.append(f"Stage 1: Retrieved {len(top_docs)} relevant GitHub issues/PRs from ChromaDB")
    return {"context": "\n\n".join(top_docs), "context_found": True, "trace": trace}

# ── Node 3: Stage 2 LLM Keyword Extraction & Secondary Vector Search ─────
def keyword_fallback_search(state: GraphState):
    """
    Triggered when Stage 1 vector search returns no context or low-relevance context.
    Uses the LLM to analyze the user's prompt and extract important search keywords,
    then performs a secondary vector search across ChromaDB.
    """
    repo = normalize_repo_name(state.get("repo"))
    query = state["question"]
    trace = list(state.get("trace", []))
    repo_summary = get_repo_metadata_summary(repo)

    kw_prompt = f"""You are a search keyword extraction engine for a code repository RAG system.
Target Repository: {repo}
Repository Indexed Metadata ({repo_summary})

User Prompt: "{query}"

Direct vector search with the user's raw prompt did not return matching context.
Follow these rules strictly:
1. If the user's prompt is completely unrelated to software development, programming, documentation, issues/bugs, or anything a code repository could contain (for example: baking recipes, sports scores, celebrity gossip, random non-technical questions), reply with EXACTLY:
UNRELATED
2. Otherwise, analyze what the user is looking for and generate 5 to 8 high-signal search keywords, function/class names from the repository metadata above, or programming concepts to search the vector database.
Reply ONLY with 'UNRELATED' or a comma-separated list of keywords (no explanations)."""

    try:
        kw_resp = llm.invoke([SystemMessage(content=kw_prompt)])
        extracted = kw_resp.content.strip()
    except Exception as e:
        extracted = ""
        print(f"Keyword extraction error: {e}")

    if not extracted or "UNRELATED" in extracted.upper():
        trace.append("Stage 2: LLM analyzed prompt -> Determined prompt has no relation to repository codebase")
        return {
            "context": "",
            "context_found": False,
            "search_keywords": "UNRELATED",
            "trace": trace
        }

    trace.append(f"Stage 2: LLM extracted search keywords: [{extracted}]")

    # Perform secondary vector search using the extracted keywords
    search_queries = [extracted, f"{query} {extracted}"]
    candidate_docs = []

    try:
        if repo:
            c_res = code_col.query(query_texts=search_queries, n_results=15, where={"repo": repo})
            h_res = history_col.query(query_texts=search_queries, n_results=10, where={"repo": repo})
        else:
            c_res = code_col.query(query_texts=search_queries, n_results=15)
            h_res = history_col.query(query_texts=search_queries, n_results=10)

        seen = set()
        for res_obj in (h_res, c_res):
            if res_obj and res_obj.get("documents"):
                for doc_list in res_obj["documents"]:
                    for d in doc_list:
                        if d not in seen:
                            seen.add(d)
                            candidate_docs.append(d)
    except Exception as e:
        print(f"Secondary search notice: {e}")

    if not candidate_docs:
        trace.append("Stage 2: Secondary vector search with LLM keywords returned 0 chunks")
        return {
            "context": "",
            "context_found": False,
            "search_keywords": extracted,
            "trace": trace
        }

    # Rank candidate documents against combined query + keywords
    combined_query = f"{query} {extracted}"
    scores = cross_encoder.predict([[combined_query, d] for d in candidate_docs])
    scored = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
    max_score = float(scored[0][0]) if scored else -10.0

    # Verify if the secondary retrieved context can actually help answer the prompt
    top_candidates = [doc for _, doc in scored[:6]]
    candidate_context = "\n\n".join(top_candidates)

    verify_prompt = f"""You are a context verification assistant for the repository '{repo}'.
User Question: "{query}"
Extracted Search Keywords: "{extracted}"

Retrieved Chunks:
{candidate_context[:4000]}

Can these retrieved chunks help answer the user's question (either directly, or by analyzing the code for potential bugs/enhancements/structure if the user asked about issues/PRs/architecture)?
Reply ONLY with 'YES' or 'NO'."""

    is_helpful = False
    try:
        v_resp = llm.invoke([SystemMessage(content=verify_prompt)])
        is_helpful = "YES" in v_resp.content.strip().upper()
    except Exception:
        is_helpful = max_score > -5.5

    if not is_helpful:
        trace.append("Stage 2: Secondary search chunks did not contain relevant information for the prompt")
        return {
            "context": "",
            "context_found": False,
            "search_keywords": extracted,
            "trace": trace
        }

    trace.append(f"Stage 2: Secondary search with LLM keywords retrieved {len(top_candidates)} relevant chunks")
    return {
        "context": candidate_context,
        "context_found": True,
        "search_keywords": extracted,
        "trace": trace
    }

# ── Node 4: Answer Generation (Normal Flow or Fallback Message) ──────────
def generate_answer(state: GraphState):
    repo_name = normalize_repo_name(state.get("repo")) or "the codebase"
    context = state.get("context", "").strip()
    context_found = state.get("context_found", False)
    keywords = state.get("search_keywords")
    trace = list(state.get("trace", []))

    # Stage 3 Fallback: If both initial search AND LLM keyword search failed to produce context
    if not context_found or not context:
        trace.append("Stage 3: No relevant context found after keyword pipeline -> Returning fallback notice")
        if keywords and keywords != "UNRELATED":
            fallback_msg = (
                f"I'm sorry, but the provided context doesn't include any information to answer your question for the **{repo_name}** repository.\n\n"
                f"**Pipeline Trace:**\n"
                f"- **Step 1 (Direct Vector Search):** No direct context matched your prompt.\n"
                f"- **Step 2 (LLM Keyword Expansion):** Analyzed your prompt and searched ChromaDB with keywords: `{keywords}`.\n"
                f"- **Result:** No matching code or GitHub history records were found in **{repo_name}** for this topic."
            )
        else:
            fallback_msg = (
                f"I'm sorry, but your prompt does not appear to have any relevant context in the **{repo_name}** repository. "
                f"Please ask a question related to the **{repo_name}** codebase, its functions, architecture, or GitHub history."
            )
        return {
            "generation": fallback_msg,
            "trace": trace,
            "messages": [AIMessage(content=fallback_msg)]
        }

    # Normal / Keyword-Recovered Synthesis Flow
    trace.append("Synthesized answer with Groq LLM & SQLite thread memory")
    keyword_note = ""
    if keywords and keywords != "UNRELATED":
        keyword_note = (
            f"\nNote: Direct search returned no exact match, so LLM keyword expansion (`{keywords}`) "
            f"was used to retrieve the relevant codebase chunks below. If the user asked about closed GitHub issues/PRs "
            f"and no explicit issue records are in the context, state clearly that no closed GitHub issues/PRs were indexed for {repo_name}, "
            f"and then use the retrieved code chunks to highlight the codebase's existing error handling, potential bugs, or enhancements."
        )

    sys_prompt = f"""You are a senior developer onboarding assistant for {repo_name}.
Answer the user's question thoroughly and accurately based on the retrieved context and conversation history.
Always reference relevant files, classes, functions, and line numbers when discussing code.{keyword_note}

Context:
{context}"""

    messages_to_llm = [SystemMessage(content=sys_prompt)] + state.get("messages", []) + [HumanMessage(content=state["question"])]
    
    response = llm.invoke(messages_to_llm)
    answer = response.content
    
    return {
        "generation": answer,
        "trace": trace,
        "messages": [AIMessage(content=answer)]
    }

def route_to_retriever(state: GraphState):
    return state["route"]

def check_context_after_stage1(state: GraphState):
    if state.get("context_found") and state.get("context"):
        return "generate"
    return "keyword_fallback"

# ── Build the Execution Graph ────────────────────────────────────────────
workflow = StateGraph(GraphState)

workflow.add_node("router", route_question)
workflow.add_node("retrieve_code", retrieve_code)
workflow.add_node("retrieve_history", retrieve_history)
workflow.add_node("keyword_fallback", keyword_fallback_search)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_to_retriever, {
    "codebase": "retrieve_code",
    "github_history": "retrieve_history"
})
workflow.add_conditional_edges("retrieve_code", check_context_after_stage1, {
    "generate": "generate",
    "keyword_fallback": "keyword_fallback"
})
workflow.add_conditional_edges("retrieve_history", check_context_after_stage1, {
    "generate": "generate",
    "keyword_fallback": "keyword_fallback"
})
workflow.add_edge("keyword_fallback", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()