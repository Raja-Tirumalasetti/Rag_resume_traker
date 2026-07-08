import os
import re
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

# ── Keyword & Hybrid Search Helpers ────────────────────────────────────────────

STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    'can', 'cant', 'cannot', 'could', 'couldnt',
    'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during',
    'each',
    'few', 'for', 'from', 'further',
    'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows',
    'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself',
    'lets',
    'me', 'more', 'most', 'mustnt', 'my', 'myself',
    'no', 'nor', 'not',
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such',
    'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
    'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt',
    'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves',
    # Resume/search-specific words
    'resume', 'resumes', 'profile', 'profiles', 'candidate', 'candidates', 'skills', 'experience', 'projects', 'details', 'find', 'show', 'list', 'who', 'whose', 'tell', 'knows', 'has', 'have', 'working', 'worked', 'know', 'get', 'give'
}

def extract_keywords(query: str) -> list[str]:
    """Extracts non-stopword, alphanumeric keywords from query."""
    cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
    words = cleaned.split()
    return [w for w in words if w not in STOP_WORDS and len(w) >= 2]

def keyword_search(all_docs: list[Document], query: str) -> list[Document]:
    """
    Ranks documents from Chroma by occurrence of query keywords in:
    - candidate metadata (high boost)
    - source metadata (medium boost)
    - page_content (normal frequency count)
    """
    keywords = extract_keywords(query)
    if not keywords:
        return []
        
    scored_docs = []
    for doc in all_docs:
        score = 0.0
        content_lower = doc.page_content.lower()
        candidate_lower = doc.metadata.get("candidate", "").lower()
        source_lower = doc.metadata.get("source", "").lower()
        
        for kw in keywords:
            # Frequency count in content
            content_count = content_lower.count(kw)
            score += content_count * 1.0
            
            # Substantial boost if keyword matches candidate name
            if kw in candidate_lower:
                score += 100.0
                
            # Boost if keyword matches the filename
            if kw in source_lower:
                score += 50.0
                
        if score > 0:
            scored_docs.append((score, doc))
            
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]

def hybrid_search(db: Chroma, query: str, k: int = 24) -> list[Document]:
    """
    Performs keyword search, then similarity search, and merges the results.
    Keyword matches are placed first, followed by semantic matches.
    """
    # 1. Fetch all docs for keyword filtering
    try:
        data = db.get()
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        all_docs = []
        for doc_id, text, meta in zip(ids, documents, metadatas):
            meta = meta.copy() if meta else {}
            if "chunk_id" not in meta:
                meta["chunk_id"] = doc_id
            all_docs.append(Document(page_content=text, metadata=meta))
    except Exception as e:
        print(f"[hybrid_search] Error fetching all docs for keyword matching: {e}")
        all_docs = []

    # 2. Perform keyword search
    kw_docs = keyword_search(all_docs, query)
    
    # 3. Perform semantic similarity search
    try:
        sem_docs = db.similarity_search(query, k=k)
    except Exception as e:
        print(f"[hybrid_search] Error during similarity search: {e}")
        sem_docs = []
        
    # 4. Merge results (keyword matches first, deduplicated)
    merged_docs = []
    seen_ids = set()
    
    import hashlib
    def get_doc_id(doc):
        source = doc.metadata.get("source", "")
        candidate = doc.metadata.get("candidate", "")
        content_utf8 = doc.page_content.encode('utf-8', errors='ignore')
        content_hash = hashlib.md5(content_utf8).hexdigest()
        return f"{source}_{candidate}_{content_hash}"
        
    for doc in kw_docs:
        doc_id = get_doc_id(doc)
        if doc_id not in seen_ids:
            merged_docs.append(doc)
            seen_ids.add(doc_id)
            
    for doc in sem_docs:
        doc_id = get_doc_id(doc)
        if doc_id not in seen_ids:
            merged_docs.append(doc)
            seen_ids.add(doc_id)
            
    return merged_docs[:k]
