"""
demo_phase6.py — Phase 6 Demonstration & Transcript Generator

Executes the four core Phase 6 scenarios:
  Case A: IN-SCOPE + GROUNDED ("What is the probation period?")
  Case B: IN-SCOPE + GROUNDED ("Can I negotiate my salary after receiving an offer?")
  Case C: OUT-OF-SCOPE (Threshold Rejection: "What is the capital of France?")
  Case D: CONTROLLED GROUNDEDNESS FAILURE (Unsupported Answer Rejection)

Saves the complete execution transcript to:
  transcripts/phase6_generation.json
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from rag.generation import answer_query, FALLBACK_RESPONSE, DEFAULT_THRESHOLD
from rag.retrieval import retrieve

TRANSCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "transcripts", "phase6_generation.json"
)


def run_demonstration():
    print("=" * 70)
    print("PHASE 6 GROUNDED GENERATION DEMONSTRATION")
    print("=" * 70)
    print()

    transcript = {
        "phase": "Phase 6 — Grounded Answer Generation & Groundedness Verification",
        "timestamp": datetime.now().isoformat(),
        "mock_mode": True,
        "calibrated_threshold": DEFAULT_THRESHOLD,
        "demonstrations": []
    }

    # ------------------------------------------------------------------
    # Case A: In-Scope 1
    # ------------------------------------------------------------------
    q_a = "What is the probation period?"
    print(f"[CASE A] IN-SCOPE 1: '{q_a}'")
    res_a = answer_query(q_a)
    print(f"  Top-1 Similarity : {res_a['top1_similarity']} (Threshold: {res_a['threshold']})")
    print(f"  Retrieval Status : {'ACCEPTED' if res_a['is_supported'] else 'REJECTED'}")
    print(f"  Primary Source   : {res_a['sources'][0] if res_a['sources'] else 'None'}")
    print(f"  Groundedness     : Score = {res_a['groundedness_score']} (Grounded: {res_a['is_grounded']})")
    print(f"  Final Answer     : {res_a['answer']}")
    print()

    transcript["demonstrations"].append({
        "case_id": "CASE_A_IN_SCOPE_1",
        "scenario": "In-Scope Question (Probation Period)",
        "query": q_a,
        "expected_result": "ACCEPTED_AND_GROUNDED",
        "top1_similarity": res_a["top1_similarity"],
        "retrieval_accepted": res_a["is_supported"],
        "primary_source": res_a["sources"][0] if res_a["sources"] else None,
        "generated_answer": res_a["answer"],
        "groundedness_score": res_a["groundedness_score"],
        "is_grounded": res_a["is_grounded"],
        "final_response": res_a["answer"],
        "fallback_triggered": res_a["answer"] == FALLBACK_RESPONSE
    })

    # ------------------------------------------------------------------
    # Case B: In-Scope 2
    # ------------------------------------------------------------------
    q_b = "Can I negotiate my salary after receiving an offer?"
    print(f"[CASE B] IN-SCOPE 2: '{q_b}'")
    res_b = answer_query(q_b)
    print(f"  Top-1 Similarity : {res_b['top1_similarity']} (Threshold: {res_b['threshold']})")
    print(f"  Retrieval Status : {'ACCEPTED' if res_b['is_supported'] else 'REJECTED'}")
    print(f"  Primary Source   : {res_b['sources'][0] if res_b['sources'] else 'None'}")
    print(f"  Groundedness     : Score = {res_b['groundedness_score']} (Grounded: {res_b['is_grounded']})")
    print(f"  Final Answer     : {res_b['answer']}")
    print()

    transcript["demonstrations"].append({
        "case_id": "CASE_B_IN_SCOPE_2",
        "scenario": "In-Scope Question (Offer Negotiation)",
        "query": q_b,
        "expected_result": "ACCEPTED_AND_GROUNDED",
        "top1_similarity": res_b["top1_similarity"],
        "retrieval_accepted": res_b["is_supported"],
        "primary_source": res_b["sources"][0] if res_b["sources"] else None,
        "generated_answer": res_b["answer"],
        "groundedness_score": res_b["groundedness_score"],
        "is_grounded": res_b["is_grounded"],
        "final_response": res_b["answer"],
        "fallback_triggered": res_b["answer"] == FALLBACK_RESPONSE
    })

    # ------------------------------------------------------------------
    # Case C: Out-of-Scope (Threshold Guardrail Rejection)
    # ------------------------------------------------------------------
    q_c = "What is the capital of France?"
    print(f"[CASE C] OUT-OF-SCOPE: '{q_c}'")
    res_c = answer_query(q_c)
    print(f"  Top-1 Similarity : {res_c['top1_similarity']} (Threshold: {res_c['threshold']})")
    print(f"  Retrieval Status : {'ACCEPTED' if res_c['is_supported'] else 'REJECTED (Below Threshold)'}")
    print(f"  Rejection Reason : {res_c.get('rejection_reason')}")
    print(f"  Final Answer     : {res_c['answer']}")
    print()

    transcript["demonstrations"].append({
        "case_id": "CASE_C_OUT_OF_SCOPE",
        "scenario": "Out-of-Scope Question (Guardrail 1: Retrieval Threshold)",
        "query": q_c,
        "expected_result": "REJECTED_BY_THRESHOLD",
        "top1_similarity": res_c["top1_similarity"],
        "retrieval_accepted": res_c["is_supported"],
        "rejection_reason": res_c.get("rejection_reason"),
        "primary_source": None,
        "generated_answer": None,
        "groundedness_score": 0.0,
        "is_grounded": False,
        "final_response": res_c["answer"],
        "fallback_triggered": res_c["answer"] == FALLBACK_RESPONSE
    })

    # ------------------------------------------------------------------
    # Case D: Controlled Groundedness Failure (Guardrail 2 Rejection)
    # ------------------------------------------------------------------
    q_d = "What happens during the probation period?"
    invented_hallucination = (
        "Employees on probation are entitled to 45 days of paid executive vacation, "
        "first-class international airline tickets, and complimentary private gourmet catering."
    )
    print(f"[CASE D] CONTROLLED GROUNDEDNESS FAILURE:")
    print(f"  Query            : '{q_d}'")
    print(f"  Simulated Answer : '{invented_hallucination}'")
    res_d = answer_query(q_d, forced_candidate_answer=invented_hallucination)
    print(f"  Retrieval Status : {'ACCEPTED' if res_d['is_supported'] else 'REJECTED'}")
    print(f"  Groundedness     : Score = {res_d['groundedness_score']} (Grounded: {res_d['is_grounded']})")
    print(f"  Rejection Reason : {res_d.get('rejection_reason')}")
    print(f"  Unsupported Words: {res_d.get('groundedness_details', {}).get('unsupported_samples')}")
    print(f"  Final Answer     : {res_d['answer']}")
    print()

    transcript["demonstrations"].append({
        "case_id": "CASE_D_GROUNDEDNESS_FAILURE",
        "scenario": "Unsupported Candidate Answer (Guardrail 2: Groundedness Check)",
        "query": q_d,
        "expected_result": "REJECTED_BY_GROUNDEDNESS",
        "top1_similarity": res_d["top1_similarity"],
        "retrieval_accepted": res_d["is_supported"],
        "rejection_reason": res_d.get("rejection_reason"),
        "simulated_unsupported_answer": invented_hallucination,
        "groundedness_score": res_d["groundedness_score"],
        "is_grounded": res_d["is_grounded"],
        "groundedness_details": res_d.get("groundedness_details"),
        "final_response": res_d["answer"],
        "fallback_triggered": res_d["answer"] == FALLBACK_RESPONSE
    })

    # Save transcript
    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2)

    print(f"Transcript written to: {TRANSCRIPT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
