"""
calibrate_threshold.py — Empirical Threshold Calibration for Phase 5

Tests in-scope and out-of-scope queries against both Chroma collections,
records top-1 cosine similarity for each, and recommends a threshold.

Usage:
    python calibrate_threshold.py
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from rag.embeddings import load_model
from rag.retrieval import retrieve, retrieve_with_threshold

# ---------------------------------------------------------------------------
# Calibration queries
# ---------------------------------------------------------------------------

# In-scope queries — topics covered by the 12 knowledge base documents
IN_SCOPE_QUERIES = [
    {
        "query": "Who is eligible to apply for a job?",
        "expected_topic": "job_application_eligibility",
    },
    {
        "query": "How are interviews scheduled and can I reschedule?",
        "expected_topic": "interview_scheduling",
    },
    {
        "query": "Can I negotiate my salary after receiving an offer?",
        "expected_topic": "offer_negotiation",
    },
    {
        "query": "What happens during the probation period?",
        "expected_topic": "probation_period",
    },
    {
        "query": "Am I eligible for remote work?",
        "expected_topic": "remote_work",
    },
    {
        "query": "How long is applicant data retained?",
        "expected_topic": "applicant_data_retention",
    },
    {
        "query": "What is the referral bonus eligibility?",
        "expected_topic": "referral_bonus",
    },
]

# Out-of-scope queries — topics NOT in the knowledge base
OUT_OF_SCOPE_QUERIES = [
    {
        "query": "What is the weather in Mumbai today?",
        "expected_topic": "NONE",
    },
    {
        "query": "How do I cook chicken biryani?",
        "expected_topic": "NONE",
    },
    {
        "query": "What is the capital of France?",
        "expected_topic": "NONE",
    },
]

COLLECTIONS = ["sentence_chunks", "fixed_size_chunks"]


def run_calibration():
    """Run calibration queries and report results."""
    print("=" * 70)
    print("PHASE 5 — THRESHOLD CALIBRATION EXPERIMENT")
    print("=" * 70)
    print()

    model = load_model()
    all_results = {}

    for collection_name in COLLECTIONS:
        print(f"\n{'=' * 70}")
        print(f"COLLECTION: {collection_name}")
        print(f"{'=' * 70}")

        in_scope_scores = []
        out_scope_scores = []
        collection_results = []

        # --- In-scope queries ---
        print(f"\n--- IN-SCOPE QUERIES ---")
        for item in IN_SCOPE_QUERIES:
            results = retrieve(
                query=item["query"],
                collection_name=collection_name,
                top_k=3,
                model=model,
            )
            top1 = results[0]
            score = top1["cosine_similarity"]
            in_scope_scores.append(score)

            print(f"  Query:      {item['query']}")
            print(f"  Top-1 doc:  {top1['source']}")
            print(f"  Similarity: {score:.4f}")
            print(f"  Expected:   {item['expected_topic']}")
            match = item["expected_topic"] in top1["source"]
            print(f"  Correct:    {'YES' if match else 'NO'}")
            print()

            collection_results.append({
                "query": item["query"],
                "type": "in_scope",
                "expected_topic": item["expected_topic"],
                "top1_source": top1["source"],
                "top1_similarity": score,
                "top1_text": top1["text"][:100],
                "correct_retrieval": match,
            })

        # --- Out-of-scope queries ---
        print(f"--- OUT-OF-SCOPE QUERIES ---")
        for item in OUT_OF_SCOPE_QUERIES:
            results = retrieve(
                query=item["query"],
                collection_name=collection_name,
                top_k=3,
                model=model,
            )
            top1 = results[0]
            score = top1["cosine_similarity"]
            out_scope_scores.append(score)

            print(f"  Query:      {item['query']}")
            print(f"  Top-1 doc:  {top1['source']}")
            print(f"  Similarity: {score:.4f}")
            print()

            collection_results.append({
                "query": item["query"],
                "type": "out_of_scope",
                "expected_topic": "NONE",
                "top1_source": top1["source"],
                "top1_similarity": score,
                "top1_text": top1["text"][:100],
            })

        # --- Analysis ---
        min_in = min(in_scope_scores)
        max_in = max(in_scope_scores)
        avg_in = sum(in_scope_scores) / len(in_scope_scores)
        min_out = min(out_scope_scores)
        max_out = max(out_scope_scores)
        avg_out = sum(out_scope_scores) / len(out_scope_scores)

        print(f"--- SCORE ANALYSIS for {collection_name} ---")
        print(f"  In-scope  scores: min={min_in:.4f}  max={max_in:.4f}  avg={avg_in:.4f}")
        print(f"  Out-scope scores: min={min_out:.4f}  max={max_out:.4f}  avg={avg_out:.4f}")
        print(f"  Gap: lowest in-scope ({min_in:.4f}) vs highest out-scope ({max_out:.4f})")

        if min_in > max_out:
            suggested = round((min_in + max_out) / 2, 2)
            print(f"  [OK] Clear separation exists!")
            print(f"  Suggested threshold: {suggested}")
        else:
            suggested = round((avg_in + avg_out) / 2, 2)
            print(f"  [WARN] Some overlap — using midpoint of averages")
            print(f"  Suggested threshold: {suggested}")

        all_results[collection_name] = {
            "in_scope_scores": in_scope_scores,
            "out_of_scope_scores": out_scope_scores,
            "min_in_scope": min_in,
            "max_out_scope": max_out,
            "suggested_threshold": suggested,
            "details": collection_results,
        }

    return all_results


def run_demonstration(threshold, model):
    """Run the final demonstration with the calibrated threshold."""
    print()
    print("=" * 70)
    print(f"PHASE 5 — DEMONSTRATION (threshold = {threshold})")
    print("=" * 70)

    demo_queries = [
        # 5 in-scope
        "Who can apply for a job at the company?",
        "How do I reschedule my interview?",
        "What is verified during background checks?",
        "How long is the notice period?",
        "What is the policy on diversity hiring?",
        # 1 out-of-scope
        "What is the best programming language to learn?",
    ]

    demo_results = []

    for collection_name in COLLECTIONS:
        print(f"\n--- Collection: {collection_name} ---\n")

        for query in demo_queries:
            result = retrieve_with_threshold(
                query=query,
                threshold=threshold,
                collection_name=collection_name,
                model=model,
            )

            status = "ACCEPTED" if result["is_supported"] else "REJECTED -> I don't know"
            top_source = result["results"][0]["source"] if result["results"] else "N/A"

            print(f"  Query:       {query}")
            print(f"  Top-1 sim:   {result['top1_similarity']:.4f}")
            print(f"  Threshold:   {threshold}")
            print(f"  Decision:    {status}")
            if result["results"]:
                print(f"  Source:      {top_source}")
            if result["answer"]:
                print(f"  Fallback:    {result['answer']}")
            print()

            demo_results.append({
                "collection": collection_name,
                "query": query,
                "top1_similarity": result["top1_similarity"],
                "threshold": threshold,
                "is_supported": result["is_supported"],
                "top_source": top_source,
                "answer": result["answer"],
            })

    return demo_results


def main():
    """Run the full calibration + demonstration pipeline."""
    # Step 1: Calibration
    cal_results = run_calibration()

    # Step 2: Choose threshold from sentence_chunks (primary collection)
    primary = cal_results["sentence_chunks"]
    threshold = primary["suggested_threshold"]

    print()
    print("=" * 70)
    print("THRESHOLD SELECTION")
    print("=" * 70)
    print(f"  Primary collection: sentence_chunks")
    print(f"  Lowest in-scope similarity:   {primary['min_in_scope']:.4f}")
    print(f"  Highest out-of-scope similarity: {primary['max_out_scope']:.4f}")
    print(f"  Selected threshold: {threshold}")
    print("=" * 70)

    # Step 3: Demonstration
    model = load_model()
    demo_results = run_demonstration(threshold, model)

    # Step 4: Save transcript
    transcript = {
        "phase": "Phase 5 — Retrieval and Threshold Calibration",
        "timestamp": datetime.now().isoformat(),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "similarity_metric": "cosine_similarity (1 - chroma_cosine_distance)",
        "top_k": 3,
        "calibration": {},
        "selected_threshold": threshold,
        "demonstration": demo_results,
    }

    for cname in COLLECTIONS:
        cr = cal_results[cname]
        transcript["calibration"][cname] = {
            "in_scope_scores": [round(s, 4) for s in cr["in_scope_scores"]],
            "out_of_scope_scores": [round(s, 4) for s in cr["out_of_scope_scores"]],
            "min_in_scope": round(cr["min_in_scope"], 4),
            "max_out_scope": round(cr["max_out_scope"], 4),
            "suggested_threshold": cr["suggested_threshold"],
            "details": cr["details"],
        }

    transcript_path = os.path.join(
        os.path.dirname(__file__), "transcripts", "phase5_calibration.json"
    )
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\n  Transcript saved to: {transcript_path}")

    return threshold, cal_results


if __name__ == "__main__":
    main()
