"""
retrieval.py — Semantic Search and Retrieval

Searches the Chroma vector database to find the most relevant document
chunks for a given user question.

Supports both Chroma collections:
    - fixed_size_chunks
    - sentence_chunks

Uses the SAME local SentenceTransformer model (all-MiniLM-L6-v2) from Phase 4.
No external API is required.

Cosine similarity note:
    Chroma collections are configured with hnsw:space="cosine".
    Chroma returns cosine DISTANCE (0 = identical, 2 = opposite).
    Cosine similarity = 1 - cosine distance.
"""

import os
import chromadb
from rag.embeddings import load_model, MODEL_NAME

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DEFAULT_TOP_K = 3
_CACHED_MODEL = None


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve(query, collection_name="sentence_chunks", top_k=DEFAULT_TOP_K,
             chroma_dir=CHROMA_DIR, model=None):
    """
    Retrieve the top-k most similar chunks for a query.

    Args:
        query (str): The user's question.
        collection_name (str): Which Chroma collection to search.
            "fixed_size_chunks" or "sentence_chunks".
        top_k (int): Number of results to return.
        chroma_dir (str): Path to the persistent Chroma database.
        model: A preloaded SentenceTransformer model. If None, cached one is used.

    Returns:
        list[dict]: Top-k results, each containing:
            - "text": the chunk text
            - "source": parent document filename
            - "cosine_similarity": float between -1 and 1 (higher = more similar)
            - "metadata": full metadata dict from Chroma
    """
    global _CACHED_MODEL
    # Load and cache model if not provided
    if model is None:
        if _CACHED_MODEL is None:
            _CACHED_MODEL = load_model()
        model = _CACHED_MODEL

    # Embed the query using the same model used during indexing
    query_embedding = model.encode([query]).tolist()

    # Connect to Chroma and get the collection
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name)

    # Query Chroma — returns distances (cosine distance when space="cosine")
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Build structured results
    retrieved = []
    for i in range(len(results["ids"][0])):
        cosine_distance = results["distances"][0][i]
        cosine_similarity = 1.0 - cosine_distance  # Convert distance → similarity

        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "cosine_similarity": round(cosine_similarity, 4),
            "metadata": results["metadatas"][0][i],
        })

    return retrieved


# ---------------------------------------------------------------------------
# Threshold-based retrieval with fallback
# ---------------------------------------------------------------------------

def retrieve_with_threshold(query, threshold, collection_name="sentence_chunks",
                            top_k=DEFAULT_TOP_K, chroma_dir=CHROMA_DIR, model=None):
    """
    Retrieve top-k results and apply a cosine-similarity threshold.

    If the top-1 result's cosine similarity is below the threshold,
    the query is considered out-of-scope and a fallback is returned.

    Under MOCK_LLM mode, this threshold is the ONLY signal used to decide
    whether a query is supported — no keyword matching, no LLM classification.

    Args:
        query (str): The user's question.
        threshold (float): Minimum top-1 cosine similarity to accept retrieval.
        collection_name (str): Chroma collection to search.
        top_k (int): Number of results to return.
        chroma_dir (str): Path to Chroma database.
        model: Preloaded SentenceTransformer model.

    Returns:
        dict with keys:
            - "query": the original query
            - "is_supported": True if top-1 similarity >= threshold
            - "top1_similarity": float
            - "threshold": float
            - "results": list of top-k result dicts (empty if unsupported)
            - "answer": fallback message if unsupported, else None
    """
    results = retrieve(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        chroma_dir=chroma_dir,
        model=model,
    )

    top1_similarity = results[0]["cosine_similarity"] if results else 0.0
    is_supported = top1_similarity >= threshold

    return {
        "query": query,
        "is_supported": is_supported,
        "top1_similarity": top1_similarity,
        "threshold": threshold,
        "results": results if is_supported else [],
        "answer": None if is_supported else (
            "I don't have enough information to answer that question. "
            "Please contact HR directly for assistance."
        ),
    }
