"""
validate_phase6.py — Phase 6 Validation Script

Verifies grounded answer generation, deterministic MOCK_LLM execution,
and dual-layer guardrail defense against the Capstone requirements.

Usage:
    python validate_phase6.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from rag.generation import (
    generate_answer,
    check_groundedness,
    answer_query,
    is_mock_mode,
    FALLBACK_RESPONSE,
    DEFAULT_THRESHOLD,
)
from rag.retrieval import retrieve

ROOT_DIR = os.path.dirname(__file__)
TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "phase6_generation.json")
README_PATH = os.path.join(ROOT_DIR, "README.md")


def validate():
    print("=" * 65)
    print("PHASE 6 VALIDATION REPORT -- GROUNDED GENERATION & GUARDRAILS")
    print("=" * 65)
    print()

    all_passed = True

    def check(name, condition, details=""):
        nonlocal all_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_passed = False
        msg = f"[{status}] {name}"
        if details:
            msg += f" ({details})"
        print(msg)

    # ------------------------------------------------------------------
    # 1. MOCK_LLM Mode & Zero External Dependencies
    # ------------------------------------------------------------------
    check("MOCK_LLM mode is active by default", is_mock_mode() is True)

    import rag.generation as gen_mod
    gen_source = open(gen_mod.__file__, "r", encoding="utf-8").read()
    import re
    has_external_api = bool(re.search(r'\b(import\s+openai|import\s+anthropic|import\s+cohere|import\s+google\.generativeai)\b', gen_source, re.I))
    check("No external LLM API dependencies in rag/generation.py", not has_external_api)
    check("No API key required to run generation", "OPENAI_API_KEY" not in os.environ)

    # ------------------------------------------------------------------
    # 2. Retrieved Context is Passed to Generation
    # ------------------------------------------------------------------
    sample_context = retrieve("What is the probation period?", collection_name="sentence_chunks", top_k=2)
    check("Phase 5 retrieval returns context chunks", len(sample_context) > 0)

    gen_out = generate_answer("What is the probation period?", sample_context)
    check("Generation produces non-empty output from context", bool(gen_out))

    # ------------------------------------------------------------------
    # 3. Answer is Based Only on Retrieved Context
    # ------------------------------------------------------------------
    is_g, g_score, g_details = check_groundedness(gen_out, sample_context)
    check("Answer is substantiated by retrieved context (score >= 0.8)", is_g, f"score={g_score}")

    # ------------------------------------------------------------------
    # 4. In-Scope Question Gets Grounded Answer
    # ------------------------------------------------------------------
    res_in = answer_query("What is the probation period?")
    check("In-scope question is accepted by retrieval", res_in["is_supported"] is True)
    check("In-scope question receives legitimate policy answer", res_in["answer"] != FALLBACK_RESPONSE)
    check("In-scope response identifies correct source document", "probation_period.txt" in res_in["sources"])
    check("In-scope response passes groundedness check", res_in["is_grounded"] is True)

    # ------------------------------------------------------------------
    # 5. Out-of-Scope Question Triggers Fallback via Threshold
    # ------------------------------------------------------------------
    res_out = answer_query("What is the capital of France?")
    check("Out-of-scope question is rejected by threshold", res_out["is_supported"] is False)
    check("Out-of-scope question returns standard fallback response", res_out["answer"] == FALLBACK_RESPONSE)
    check("Rejection reason is correctly attributed to retrieval threshold", res_out["rejection_reason"] == "retrieval_threshold")

    # ------------------------------------------------------------------
    # 6. Groundedness Check Functionality & Failure Suppression
    # ------------------------------------------------------------------
    check("check_groundedness function is callable", callable(check_groundedness))

    unsupported_text = (
        "Employees receive 45 days of paid annual vacation, free international flights, "
        "and daily gourmet buffet lunches."
    )
    is_unsupported_g, unsupp_score, _ = check_groundedness(unsupported_text, sample_context)
    check("Unsupported text fails groundedness check", is_unsupported_g is False, f"score={unsupp_score}")

    # Test that answer_query suppresses ungrounded answers and returns fallback
    res_unsupp = answer_query("What is the probation period?", forced_candidate_answer=unsupported_text)
    check("Groundedness failure suppresses candidate answer", res_unsupp["answer"] == FALLBACK_RESPONSE)
    check("Rejection reason is correctly attributed to groundedness failure", res_unsupp["rejection_reason"] == "groundedness_failure")
    check("Ungrounded candidate answer is recorded in response for auditability", res_unsupp.get("unsupported_candidate") == unsupported_text)

    # ------------------------------------------------------------------
    # 7. Determinism Check
    # ------------------------------------------------------------------
    run_1 = answer_query("Can I negotiate my salary after receiving an offer?")
    run_2 = answer_query("Can I negotiate my salary after receiving an offer?")
    check("MOCK_LLM generation is 100% deterministic", run_1["answer"] == run_2["answer"] and run_1["groundedness_score"] == run_2["groundedness_score"])

    # ------------------------------------------------------------------
    # 8. Transcript & Documentation Check
    # ------------------------------------------------------------------
    transcript_exists = os.path.isfile(TRANSCRIPT_PATH)
    check("transcripts/phase6_generation.json exists", transcript_exists)

    if transcript_exists:
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        demos = t_data.get("demonstrations", [])
        check("Transcript contains at least 4 demonstration cases", len(demos) >= 4, f"found {len(demos)}")
        has_failure_demo = any(d.get("case_id") == "CASE_D_GROUNDEDNESS_FAILURE" for d in demos)
        check("Transcript records deliberate groundedness failure demonstration", has_failure_demo)

    readme_content = open(README_PATH, "r", encoding="utf-8").read() if os.path.isfile(README_PATH) else ""
    check("README.md contains Phase 6 documentation", "Phase 6" in readme_content)

    print()
    print("=" * 65)
    if all_passed:
        print("RESULT: ALL PHASE 6 CHECKS PASSED!")
    else:
        print("RESULT: SOME CHECKS FAILED - REVIEW ABOVE.")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
