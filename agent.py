import os
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# 1. Setup ChromaDB Connection
class LocalFastEmbed(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self.model.embed(input)]

client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = LocalFastEmbed()
code_col = client.get_collection(name="codebase_chunks", embedding_function=embed_fn)
history_col = client.get_collection(name="github_history", embedding_function=embed_fn)

# 2. Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

# 3. Define Graph State
class GraphState(TypedDict):
    question: str
    route: str
    context: str
    generation: str

# 4. Define Agent Nodes
def route_question(state: GraphState):
    question = state["question"].lower()
    
    # 1. Deterministic Fast-Path
    # If obvious history/bug keywords are in the prompt, route immediately
    history_keywords = ["cve", "vulnerability", "pr", "issue", "bug", "pull request"]
    if any(kw in question for kw in history_keywords):
        return {"route": "github_history"}
        
    # 2. LLM Semantic Routing for ambiguous questions
    system = """You are a strict routing assistant. 
    If the user asks about a vulnerability, CVE, bug, pull request, or historical discussion, reply ONLY with 'github_history'. 
    If the user asks about code implementation, functions, or how a specific class/method works, reply ONLY with 'codebase'."""
    
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=state["question"])])
    route = response.content.strip().lower()
    
    if "history" in route:
        return {"route": "github_history"}
    return {"route": "codebase"}

def retrieve_code(state: GraphState):
    results = code_col.query(query_texts=[state["question"]], n_results=3)
    context = "\n\n".join(results["documents"][0])
    return {"context": context}

def retrieve_history(state: GraphState):
    results = history_col.query(query_texts=[state["question"]], n_results=3)
    context = "\n\n".join(results["documents"][0])
    return {"context": context}

def generate_answer(state: GraphState):
    sys_prompt = "You are a senior developer. Answer the question based ONLY on the provided context. If the context does not contain the answer, say you don't know."
    user_prompt = f"Context:\n{state['context']}\n\nQuestion: {state['question']}"
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    return {"generation": response.content}

def route_to_retriever(state: GraphState):
    return state["route"]

# 5. Build the Execution Graph
workflow = StateGraph(GraphState)

workflow.add_node("router", route_question)
workflow.add_node("retrieve_code", retrieve_code)
workflow.add_node("retrieve_history", retrieve_history)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_to_retriever, {
    "codebase": "retrieve_code",
    "github_history": "retrieve_history"
})
workflow.add_edge("retrieve_code", "generate")
workflow.add_edge("retrieve_history", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 6. Test Execution
if __name__ == "__main__":
    print("Testing LangGraph Routing Pipeline...\n")
    
    q = "How does the PromptTemplate formatting vulnerability bypass work?"
    print(f"Question: {q}")
    
    result = app.invoke({"question": q})
    
    print(f"\n[Router Decision]: {result['route']}")
    print(f"\n[Generated Answer]:\n{result['generation']}")