import chromadb.utils.embedding_functions as ef

# Local embedding using ChromaDB's built-in all-MiniLM-L6-v2 (no API key needed)
_local_embedding_fn = ef.DefaultEmbeddingFunction()

def get_embedding(text: str) -> list[float]:
    """
    Generates embedding for a single text using local ChromaDB embedding model.
    No API key required.
    """
    return _local_embedding_fn([text])[0]

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts using local ChromaDB embedding model.
    No API key required.
    """
    if not texts:
        return []
    return _local_embedding_fn(texts)
