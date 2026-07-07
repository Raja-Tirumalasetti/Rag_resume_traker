import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
import embeddeing

# Database directory
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Wrap our local embedding function in LangChain's Embeddings interface
class LocalChromaEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embeddeing.get_embeddings(texts)
        
    def embed_query(self, text: str) -> list[float]:
        return embeddeing.get_embedding(text)

def get_chroma_db(collection_name: str = "resumes_collection"):
    """
    Returns a LangChain Chroma vector store instance.
    """
    return Chroma(
        collection_name=collection_name,
        persist_directory=DB_DIR,
        embedding_function=LocalChromaEmbeddings()
    )

def add_chunks_to_db(chunks: list[dict], collection_name: str = "resumes_collection"):
    """
    Adds chunks as LangChain Documents to ChromaDB.
    """
    if not chunks:
        return
        
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source":    chunk["source"],
                "candidate": chunk.get("candidate", ""),
            }
        )
        for chunk in chunks
    ]
    
    db = get_chroma_db(collection_name)
    db.add_documents(documents)

def delete_collection(collection_name: str = "resumes_collection"):
    """
    Clears the ChromaDB persistent directory to delete the collection.
    """
    db = get_chroma_db(collection_name)
    try:
        db.delete_collection()
    except Exception as e:
        print(f"Error deleting collection: {e}")
