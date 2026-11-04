import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

class LocalFastEmbed(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self.model.embed(input)]

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_db"))
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)
embed_fn = LocalFastEmbed()

code_col = client.get_or_create_collection(
    name="codebase_chunks",
    embedding_function=embed_fn
)
history_col = client.get_or_create_collection(
    name="github_history",
    embedding_function=embed_fn
)