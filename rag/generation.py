"""
generation.py — Grounded Answer Generation & Groundedness Verification

Implements Phase 6 of the Naukri.com Domain Support Agent capstone:
1. Deterministic MOCK_LLM answer generation based ONLY on retrieved context.
2. Rule-based, deterministic groundedness check (no secondary LLM call required).
3. Two-layer defense:
   - Layer 1: Retrieval similarity threshold (< 0.28 -> "I don't know").
   - Layer 2: Groundedness check failure -> "I don't know".
"""

import os
import re
from rag.retrieval import retrieve_with_threshold, DEFAULT_TOP_K

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.28
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.80

FALLBACK_RESPONSE = (
    "I don't have enough information to answer that question. "
    "Please contact HR directly for assistance."
)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves", "may", "must",
    "shall", "also", "based", "according", "policy", "company"
}


def is_mock_mode():
    """Return True if running in MOCK_LLM mode (default: True)."""
    return os.getenv("MOCK_LLM", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Groundedness Verification (Deterministic, No External LLM)
# ---------------------------------------------------------------------------

def extract_content_words(text):
    """Extract lowercase words with length >= 3, excluding stopwords."""
    words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def check_groundedness(answer, context_chunks, threshold=DEFAULT_GROUNDEDNESS_THRESHOLD):
    """
    Verify whether the answer is substantiated by the retrieved context.

    Algorithm:
    1. Concatenate the text of all retrieved chunks into a context corpus.
    2. Extract content words (non-stopwords) from the answer and the context.
    3. Calculate the ratio of answer content words that appear in the context.
    4. If the ratio meets or exceeds `threshold`, the answer is grounded.

    Args:
        answer (str): The candidate answer string.
        context_chunks (list[dict]): Retrieved chunk dictionaries.
        threshold (float): Minimum supported token ratio (default: 0.80).

    Returns:
        tuple[bool, float, dict]:
            - is_grounded (bool)
            - groundedness_score (float, 0.0 to 1.0)
            - details (dict with token counts and unsupported terms)
    """
    if not answer or not context_chunks:
        return False, 0.0, {"reason": "Missing answer or context"}

    # Never treat the fallback refusal as grounded content
    if answer == FALLBACK_RESPONSE:
        return False, 0.0, {"reason": "Refusal fallback"}

    # Concatenate all retrieved chunk texts
    context_text = " ".join(c.get("text", "") for c in context_chunks).lower()
    context_words = set(extract_content_words(context_text))

    # Extract answer content words
    answer_words = extract_content_words(answer)

    if not answer_words:
        # If the answer has no content words (e.g. only stopwords), it cannot be validated
        return False, 0.0, {"reason": "No substantive content words in answer"}

    supported = [w for w in answer_words if w in context_words]
    unsupported = [w for w in answer_words if w not in context_words]

    score = len(supported) / len(answer_words)
    is_grounded = score >= threshold

    details = {
        "total_content_words": len(answer_words),
        "supported_words_count": len(supported),
        "unsupported_words_count": len(unsupported),
        "unsupported_samples": unsupported[:5],
        "score": round(score, 4),
        "threshold": threshold,
    }

    return is_grounded, round(score, 4), details


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------

def generate_answer(query, context_chunks, mode=None):
    """
    Generate an answer based ONLY on the retrieved context chunks.

    Args:
        query (str): The user's question.
        context_chunks (list[dict]): Top retrieved chunks from Chroma.
        mode (str, optional): "mock" or "real". Defaults to MOCK_LLM setting.

    Returns:
        str: Grounded answer text.
    """
    if not context_chunks:
        return FALLBACK_RESPONSE

    if mode is None:
        mode = "mock" if is_mock_mode() else "real"

    if mode == "mock":
        # Deterministic extraction directly from authoritative retrieved text
        # Select distinct sentences from top chunks
        seen_sentences = set()
        extracted = []

        for chunk in context_chunks:
            text = chunk.get("text", "").strip()
            # Split into sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            for s in sentences:
                s_key = s.lower()
                # Skip disclaimer sentences to keep answers concise and focused
                if "fictional project policy created for educational purposes" in s_key:
                    continue
                if s_key not in seen_sentences and len(s) > 15:
                    seen_sentences.add(s_key)
                    extracted.append(s)

        if not extracted:
            # Fallback to raw first chunk text if sentence split was empty
            first_text = context_chunks[0].get("text", "").strip()
            return first_text

        # Return the top coherent policy sentences directly from context
        return " ".join(extracted[:2])

    else:
        # Reserved for future Real LLM integration (Phase 7+)
        raise NotImplementedError("Real LLM mode is not configured for Phase 6.")


# ---------------------------------------------------------------------------
# End-to-End RAG Pipeline with Dual Guardrails
# ---------------------------------------------------------------------------

def answer_query(query, threshold=DEFAULT_THRESHOLD, collection_name="sentence_chunks",
                 top_k=DEFAULT_TOP_K, forced_candidate_answer=None):
    """
    Execute full RAG generation pipeline:
    1. Retrieval with Cosine Similarity Threshold (Phase 5).
    2. Grounded Answer Generation from Context (Phase 6).
    3. Groundedness Verification Guardrail (Phase 6).

    Args:
        query (str): The user's question.
        threshold (float): Retrieval cosine similarity threshold (default: 0.28).
        collection_name (str): Chroma collection to search.
        top_k (int): Number of context chunks to retrieve.
        forced_candidate_answer (str, optional): Used to test deliberate
            groundedness failures in demonstration scripts.

    Returns:
        dict: Complete structured response.
    """
    # 1. Retrieval & similarity threshold check
    retrieval_res = retrieve_with_threshold(
        query=query,
        threshold=threshold,
        collection_name=collection_name,
        top_k=top_k,
    )

    top1_sim = retrieval_res["top1_similarity"]
    is_supported = retrieval_res["is_supported"]

    # Guardrail 1: Retrieval Threshold Rejection
    if not is_supported:
        return {
            "query": query,
            "answer": FALLBACK_RESPONSE,
            "is_supported": False,
            "rejection_reason": "retrieval_threshold",
            "top1_similarity": top1_sim,
            "threshold": threshold,
            "sources": [],
            "context_chunks": [],
            "is_grounded": False,
            "groundedness_score": 0.0,
            "mock_mode": is_mock_mode(),
        }

    context_chunks = retrieval_res["results"]
    sources = list(dict.fromkeys(c.get("source", "unknown") for c in context_chunks))

    # 2. Answer generation
    if forced_candidate_answer is not None:
        candidate_answer = forced_candidate_answer
    else:
        candidate_answer = generate_answer(query, context_chunks)

    # 3. Guardrail 2: Groundedness Verification
    is_grounded, g_score, g_details = check_groundedness(candidate_answer, context_chunks)

    if not is_grounded:
        return {
            "query": query,
            "answer": FALLBACK_RESPONSE,
            "is_supported": True,
            "rejection_reason": "groundedness_failure",
            "unsupported_candidate": candidate_answer,
            "top1_similarity": top1_sim,
            "threshold": threshold,
            "sources": sources,
            "context_chunks": context_chunks,
            "is_grounded": False,
            "groundedness_score": g_score,
            "groundedness_details": g_details,
            "mock_mode": is_mock_mode(),
        }

    # Successful grounded response
    return {
        "query": query,
        "answer": candidate_answer,
        "is_supported": True,
        "rejection_reason": None,
        "top1_similarity": top1_sim,
        "threshold": threshold,
        "sources": sources,
        "context_chunks": context_chunks,
        "is_grounded": True,
        "groundedness_score": g_score,
        "groundedness_details": g_details,
        "mock_mode": is_mock_mode(),
    }
