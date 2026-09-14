"""
evaluate_agent.py — RAG Triad Evaluation at Scale (Capstone Task 13)

Evaluates the agent's RAG pipeline using 15 queries under MOCK_LLM mode.
Metrics (RAG Triad):
  1. Context Relevance  — top1_similarity from retrieval (is the context relevant to the query?)
  2. Groundedness       — groundedness_score from check_groundedness (is the answer supported by context?)
  3. Answer Relevance   — deterministic token overlap between query and answer (is the answer relevant to the query?)

Produces per-query scores, aggregate averages, and saves transcript to
transcripts/task13_rag_triad.json.
"""

import os
import sys
import re
import json
from datetime import datetime

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("MOCK_LLM", "true")

from evaluation.eval_set import EVALUATION_SET
from rag.generation import answer_query, FALLBACK_RESPONSE, DEFAULT_THRESHOLD

# ---------------------------------------------------------------------------
# Extended Evaluation Set (15 queries: 12 KB + 3 out-of-scope)
# ---------------------------------------------------------------------------

EXTENDED_EVAL_SET = list(EVALUATION_SET) + [
    {
        "id": "EVAL-13",
        "topic": "out_of_scope",
        "query": "What is the weather forecast in Mumbai today?",
        "expected_documents": [],
    },
    {
        "id": "EVAL-14",
        "topic": "out_of_scope",
        "query": "Can you tell me a funny joke about engineers?",
        "expected_documents": [],
    },
    {
        "id": "EVAL-15",
        "topic": "out_of_scope",
        "query": "What is the capital of France and its population?",
        "expected_documents": [],
    },
]


# ---------------------------------------------------------------------------
# Answer Relevance Scorer (Deterministic, MOCK_LLM compatible)
# ---------------------------------------------------------------------------

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
    "shall", "also", "based", "according", "policy", "company",
}


def extract_content_words(text):
    """Extract lowercase content words (len >= 3, not stopwords)."""
    words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def score_answer_relevance(query, answer):
    """
    Deterministic answer relevance: token overlap between query and answer.

    Measures what fraction of query content words appear in the answer.
    If the answer is the fallback refusal, score is 0.0 (correctly refused).

    Returns:
        float: Score between 0.0 and 1.0
    """
    if not query or not answer:
        return 0.0
    if answer == FALLBACK_RESPONSE:
        return 0.0

    query_words = set(extract_content_words(query))
    answer_words = set(extract_content_words(answer))

    if not query_words:
        return 0.0

    overlap = query_words & answer_words
    return round(len(overlap) / len(query_words), 4)


# ---------------------------------------------------------------------------
# Main Evaluation Runner
# ---------------------------------------------------------------------------

def run_rag_triad_evaluation():
    """Run RAG Triad evaluation on 15 queries and produce transcript."""
    print("=" * 75)
    print("TASK 13 — RAG TRIAD EVALUATION AT SCALE")
    print("=" * 75)
    print(f"Total Evaluation Queries: {len(EXTENDED_EVAL_SET)}")
    print(f"Metrics: Context Relevance, Groundedness, Answer Relevance")
    print(f"Mode: MOCK_LLM = {os.getenv('MOCK_LLM', 'true')}")
    print("=" * 75)

    results = []

    for item in EXTENDED_EVAL_SET:
        query_id = item["id"]
        topic = item["topic"]
        query = item["query"]
        expected_docs = item["expected_documents"]

        # Run through RAG pipeline
        rag_result = answer_query(query, threshold=DEFAULT_THRESHOLD)

        answer = rag_result.get("answer", "")
        top1_sim = rag_result.get("top1_similarity", 0.0)
        is_supported = rag_result.get("is_supported", False)
        is_grounded = rag_result.get("is_grounded", False)
        groundedness_score = rag_result.get("groundedness_score", 0.0)
        rejection_reason = rag_result.get("rejection_reason")
        sources = rag_result.get("sources", [])

        # RAG Triad Metrics
        context_relevance = round(top1_sim, 4)
        groundedness = round(groundedness_score, 4)
        answer_relevance = score_answer_relevance(query, answer)

        # For out-of-scope queries that are correctly refused,
        # context_relevance reflects the low similarity (correct behavior)
        is_out_of_scope = topic == "out_of_scope"
        correctly_refused = is_out_of_scope and rejection_reason is not None

        result = {
            "query_id": query_id,
            "topic": topic,
            "query": query,
            "expected_documents": expected_docs,
            "answer_preview": answer[:120] + ("..." if len(answer) > 120 else ""),
            "sources": sources,
            "is_supported": is_supported,
            "is_grounded": is_grounded,
            "rejection_reason": rejection_reason,
            "is_out_of_scope": is_out_of_scope,
            "correctly_refused": correctly_refused,
            "context_relevance": context_relevance,
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
        }
        results.append(result)

        # Print per-query result
        status = "REFUSED (correct)" if correctly_refused else (
            "REFUSED (unexpected)" if rejection_reason and not is_out_of_scope else "ANSWERED"
        )
        print(f"\n[{query_id}] {topic}")
        print(f"  Query:             \"{query[:70]}...\"" if len(query) > 70 else f"  Query:             \"{query}\"")
        print(f"  Status:            {status}")
        print(f"  Context Relevance: {context_relevance:.4f}")
        print(f"  Groundedness:      {groundedness:.4f}")
        print(f"  Answer Relevance:  {answer_relevance:.4f}")

    # Aggregate Averages
    n = len(results)
    avg_context = round(sum(r["context_relevance"] for r in results) / n, 4)
    avg_grounded = round(sum(r["groundedness"] for r in results) / n, 4)
    avg_answer = round(sum(r["answer_relevance"] for r in results) / n, 4)

    # In-scope averages (12 KB topic queries only)
    in_scope = [r for r in results if not r["is_out_of_scope"]]
    n_in = len(in_scope)
    avg_context_in = round(sum(r["context_relevance"] for r in in_scope) / n_in, 4) if n_in else 0.0
    avg_grounded_in = round(sum(r["groundedness"] for r in in_scope) / n_in, 4) if n_in else 0.0
    avg_answer_in = round(sum(r["answer_relevance"] for r in in_scope) / n_in, 4) if n_in else 0.0

    # Out-of-scope summary
    oos = [r for r in results if r["is_out_of_scope"]]
    oos_refused_count = sum(1 for r in oos if r["correctly_refused"])

    print("\n\n" + "=" * 75)
    print("AGGREGATE RESULTS")
    print("=" * 75)
    print(f"\n{'Metric':<25} {'All 15 Queries':<18} {'In-Scope (12)':<18}")
    print("-" * 61)
    print(f"{'Context Relevance':<25} {avg_context:<18.4f} {avg_context_in:<18.4f}")
    print(f"{'Groundedness':<25} {avg_grounded:<18.4f} {avg_grounded_in:<18.4f}")
    print(f"{'Answer Relevance':<25} {avg_answer:<18.4f} {avg_answer_in:<18.4f}")
    print("-" * 61)
    print(f"\nOut-of-Scope Queries: {len(oos)} total, {oos_refused_count} correctly refused")
    print("=" * 75)

    # Build and save transcript
    transcript = {
        "task": "Task 13 — RAG Triad Evaluation at Scale",
        "timestamp": datetime.now().isoformat(),
        "mock_mode": True,
        "total_queries": n,
        "in_scope_queries": n_in,
        "out_of_scope_queries": len(oos),
        "out_of_scope_correctly_refused": oos_refused_count,
        "aggregate_averages": {
            "all_queries": {
                "context_relevance": avg_context,
                "groundedness": avg_grounded,
                "answer_relevance": avg_answer,
            },
            "in_scope_only": {
                "context_relevance": avg_context_in,
                "groundedness": avg_grounded_in,
                "answer_relevance": avg_answer_in,
            },
        },
        "per_query_results": results,
    }

    transcript_path = os.path.join(ROOT_DIR, "transcripts", "task13_rag_triad.json")
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript saved to: {transcript_path}")
    return transcript


if __name__ == "__main__":
    run_rag_triad_evaluation()
