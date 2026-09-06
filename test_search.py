import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

class LocalFastEmbed(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        
    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self.model.embed(input)]

# 1. Connect to the existing local database
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = LocalFastEmbed()

code_col = client.get_collection(name="codebase_chunks", embedding_function=embed_fn)
history_col = client.get_collection(name="github_history", embedding_function=embed_fn)

def search_code(query, n_results=2):
    print(f"\n--- Searching Codebase for: '{query}' ---")
    results = code_col.query(query_texts=[query], n_results=n_results)
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        print(f"\nMatch {i+1} (Distance: {dist:.4f})")
        print(f"File: {meta['file']}")
        print(f"Function/Class: {meta['name']} (Lines {meta['start_line']}-{meta['end_line']})")

def search_history(query, n_results=2):
    print(f"\n--- Searching GitHub History for: '{query}' ---")
    results = history_col.query(query_texts=[query], n_results=n_results)
    
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        doc = results['documents'][0][i]
        print(f"\nMatch {i+1} (Distance: {dist:.4f})")
        print(f"Type: {'PR' if meta['is_pr'] else 'Issue'} #{meta['number']} - {meta['title']}")
        # Print just the first 150 characters of the body
        print(f"Preview: {doc.split('Body: ')[-1][:150]}...")

# 2. Run test queries
if __name__ == "__main__":
    search_code("How does the text splitter or chunking logic work?")
    search_history("Prompt template formatting vulnerability or CVE")