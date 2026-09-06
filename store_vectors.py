import os
import glob
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding
from github import Github, Auth
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# --- NEW: Custom Wrapper for FastEmbed ---
class LocalFastEmbed(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        # This downloads the lightweight model locally on the first run
        self.model = TextEmbedding(model_name=model_name)
        
    def __call__(self, input: Documents) -> Embeddings:
        # FastEmbed generates numpy arrays; Chroma expects standard Python lists
        return [e.tolist() for e in self.model.embed(input)]

# 1. Initialize Local ChromaDB & FastEmbed
print("Initializing ChromaDB and Local Embedding Function...")
client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = LocalFastEmbed()

code_col = client.get_or_create_collection(
    name="codebase_chunks",
    embedding_function=embed_fn
)
history_col = client.get_or_create_collection(
    name="github_history",
    embedding_function=embed_fn
)

# 2. Setup Tree-sitter
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def extract_blocks_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        code_bytes = content.encode("utf-8")
    except Exception:
        return []

    tree = parser.parse(code_bytes)
    chunks = []

    def traverse(node):
        if node.type in ['function_definition', 'class_definition']:
            chunk_bytes = code_bytes[node.start_byte:node.end_byte]
            name_node = node.child_by_field_name('name')
            name = name_node.text.decode('utf-8') if name_node else "unknown"
            chunks.append({
                "type": node.type,
                "name": name,
                "code": chunk_bytes.decode('utf-8', errors="ignore"),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "file": filepath
            })
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return chunks

# 3. Ingest Code Chunks (Indexing 25 files for testing)
print("Extracting code blocks from repo_data...")
python_files = glob.glob("./repo_data/**/*.py", recursive=True)[:25]

code_docs, code_metas, code_ids = [], [], []
id_counter = 0

for py_file in python_files:
    blocks = extract_blocks_from_file(py_file)
    for block in blocks:
        code_docs.append(block["code"])
        code_metas.append({
            "file": block["file"],
            "name": block["name"],
            "type": block["type"],
            "start_line": block["start_line"],
            "end_line": block["end_line"]
        })
        code_ids.append(f"code_{id_counter}")
        id_counter += 1

if code_docs:
    print(f"Upserting {len(code_docs)} code chunks to ChromaDB...")
    code_col.upsert(documents=code_docs, metadatas=code_metas, ids=code_ids)
    print("Code chunks indexed successfully.")

# 4. Ingest GitHub History (Issues & PRs)
token = os.environ.get("GITHUB_TOKEN")
if token:
    print("Fetching GitHub issues/PRs...")
    g = Github(auth=Auth.Token(token))
    repo = g.get_repo("langchain-ai/langchain")
    issues = repo.get_issues(state="closed")[:15]

    hist_docs, hist_metas, hist_ids = [], [], []
    for issue in issues:
        body = issue.body or ""
        text = f"Title: {issue.title}\nType: {'PR' if issue.pull_request else 'Issue'}\nBody: {body[:800]}"
        hist_docs.append(text)
        hist_metas.append({
            "number": issue.number,
            "title": issue.title,
            "is_pr": bool(issue.pull_request)
        })
        hist_ids.append(f"gh_{issue.number}")

    if hist_docs:
        print(f"Upserting {len(hist_docs)} GitHub records to ChromaDB...")
        history_col.upsert(documents=hist_docs, metadatas=hist_metas, ids=hist_ids)
        print("GitHub history indexed successfully.")

print("\nDatabase persistence complete! Stored in ./chroma_db")