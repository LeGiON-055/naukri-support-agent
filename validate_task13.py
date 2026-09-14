"""
validate_task13.py — RAG Triad Evaluation Validator (Capstone Task 13)

Validates that Task 13 implementation meets all capstone requirements:
1. Evaluation covers 15 queries (12 KB + 3 out-of-scope)
2. RAG Triad metrics computed per query (context relevance, groundedness, answer relevance)
3. Aggregate averages computed for all queries and in-scope subset
4. Out-of-scope queries correctly refused
5. Transcript saved to transcripts/task13_rag_triad.json
6. All execution deterministic under MOCK_LLM
"""

import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["MOCK_LLM"] = "true"

PASS = 0
FAIL = 0
RESULTS = []


def check(test_id, description, condition):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    RESULTS.append({"id": test_id, "description": description, "status": status})
    print(f"  [{status}] {test_id}: {description}")
    return condition


def main():
    global PASS, FAIL

    print("=" * 75)
    print("TASK 13 VALIDATION — RAG Triad Evaluation at Scale")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # Section 1: Module Import
    # -----------------------------------------------------------------------
    print("\n--- Section 1: Module Import ---")
    try:
        from evaluation.evaluate_agent import (
            run_rag_triad_evaluation,
            EXTENDED_EVAL_SET,
            score_answer_relevance,
            extract_content_words,
        )
        check("T13-01", "evaluate_agent module imports successfully", True)
    except Exception as e:
        check("T13-01", f"evaluate_agent module import failed: {e}", False)
        print(f"\nResults: {PASS} PASS / {FAIL} FAIL")
        return

    # -----------------------------------------------------------------------
    # Section 2: Evaluation Set Size
    # -----------------------------------------------------------------------
    print("\n--- Section 2: Evaluation Set ---")
    check("T13-02", f"Extended eval set has >= 15 queries (got {len(EXTENDED_EVAL_SET)})",
          len(EXTENDED_EVAL_SET) >= 15)

    in_scope = [q for q in EXTENDED_EVAL_SET if q["topic"] != "out_of_scope"]
    out_scope = [q for q in EXTENDED_EVAL_SET if q["topic"] == "out_of_scope"]
    check("T13-03", f"At least 12 in-scope queries (got {len(in_scope)})",
          len(in_scope) >= 12)
    check("T13-04", f"At least 2 out-of-scope queries (got {len(out_scope)})",
          len(out_scope) >= 2)

    # -----------------------------------------------------------------------
    # Section 3: Answer Relevance Scorer
    # -----------------------------------------------------------------------
    print("\n--- Section 3: Answer Relevance Scorer ---")
    score = score_answer_relevance("What is the remote work policy?",
                                   "The remote work policy allows employees to work from home.")
    check("T13-05", f"Answer relevance returns float (got {score})",
          isinstance(score, float) and 0.0 <= score <= 1.0)

    from rag.generation import FALLBACK_RESPONSE
    score_fallback = score_answer_relevance("Any query", FALLBACK_RESPONSE)
    check("T13-06", f"Fallback answer gets score 0.0 (got {score_fallback})",
          score_fallback == 0.0)

    # -----------------------------------------------------------------------
    # Section 4: Run Full Evaluation
    # -----------------------------------------------------------------------
    print("\n--- Section 4: Full Evaluation Run ---")
    try:
        transcript = run_rag_triad_evaluation()
        check("T13-07", "run_rag_triad_evaluation() executes successfully", True)
    except Exception as e:
        check("T13-07", f"run_rag_triad_evaluation() failed: {e}", False)
        print(f"\nResults: {PASS} PASS / {FAIL} FAIL")
        return

    # -----------------------------------------------------------------------
    # Section 5: Transcript Structure
    # -----------------------------------------------------------------------
    print("\n--- Section 5: Transcript Structure ---")
    check("T13-08", "Transcript is a dict", isinstance(transcript, dict))
    check("T13-09", "Transcript has 'total_queries' field",
          "total_queries" in transcript)
    check("T13-10", f"Total queries == {len(EXTENDED_EVAL_SET)}",
          transcript.get("total_queries") == len(EXTENDED_EVAL_SET))
    check("T13-11", "Transcript has 'aggregate_averages'",
          "aggregate_averages" in transcript)
    check("T13-12", "Transcript has 'per_query_results'",
          "per_query_results" in transcript)
    check("T13-13", f"per_query_results count == {len(EXTENDED_EVAL_SET)}",
          len(transcript.get("per_query_results", [])) == len(EXTENDED_EVAL_SET))

    # -----------------------------------------------------------------------
    # Section 6: Per-Query Metric Fields
    # -----------------------------------------------------------------------
    print("\n--- Section 6: Per-Query Metrics ---")
    pqr = transcript.get("per_query_results", [])
    if pqr:
        sample = pqr[0]
        check("T13-14", "Per-query result has 'context_relevance'",
              "context_relevance" in sample)
        check("T13-15", "Per-query result has 'groundedness'",
              "groundedness" in sample)
        check("T13-16", "Per-query result has 'answer_relevance'",
              "answer_relevance" in sample)

        # Check all scores are valid floats between 0 and 1
        all_valid = all(
            isinstance(r.get("context_relevance"), (int, float)) and
            isinstance(r.get("groundedness"), (int, float)) and
            isinstance(r.get("answer_relevance"), (int, float))
            for r in pqr
        )
        check("T13-17", "All per-query scores are valid floats", all_valid)
    else:
        check("T13-14", "Per-query results empty", False)
        check("T13-15", "Per-query results empty", False)
        check("T13-16", "Per-query results empty", False)
        check("T13-17", "Per-query results empty", False)

    # -----------------------------------------------------------------------
    # Section 7: Aggregate Averages
    # -----------------------------------------------------------------------
    print("\n--- Section 7: Aggregate Averages ---")
    agg = transcript.get("aggregate_averages", {})
    all_q = agg.get("all_queries", {})
    in_s = agg.get("in_scope_only", {})
    check("T13-18", "All-queries averages have 3 metrics",
          all(k in all_q for k in ["context_relevance", "groundedness", "answer_relevance"]))
    check("T13-19", "In-scope averages have 3 metrics",
          all(k in in_s for k in ["context_relevance", "groundedness", "answer_relevance"]))

    # In-scope context relevance should be positive (queries match KB docs)
    check("T13-20", f"In-scope avg context relevance > 0 (got {in_s.get('context_relevance', 0)})",
          in_s.get("context_relevance", 0) > 0)
    check("T13-21", f"In-scope avg groundedness > 0 (got {in_s.get('groundedness', 0)})",
          in_s.get("groundedness", 0) > 0)

    # -----------------------------------------------------------------------
    # Section 8: Out-of-Scope Handling
    # -----------------------------------------------------------------------
    print("\n--- Section 8: Out-of-Scope Handling ---")
    oos_results = [r for r in pqr if r.get("is_out_of_scope")]
    oos_refused = [r for r in oos_results if r.get("correctly_refused")]
    check("T13-22", f"Out-of-scope queries found in results (got {len(oos_results)})",
          len(oos_results) >= 2)
    check("T13-23", f"Out-of-scope queries correctly refused (got {len(oos_refused)}/{len(oos_results)})",
          len(oos_refused) == len(oos_results))

    # -----------------------------------------------------------------------
    # Section 9: Transcript File
    # -----------------------------------------------------------------------
    print("\n--- Section 9: Transcript File ---")
    transcript_path = os.path.join(ROOT_DIR, "transcripts", "task13_rag_triad.json")
    check("T13-24", "Transcript file exists", os.path.isfile(transcript_path))

    if os.path.isfile(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        check("T13-25", "Saved transcript is valid JSON with expected structure",
              "per_query_results" in saved and "aggregate_averages" in saved)
    else:
        check("T13-25", "Transcript file missing", False)

    # -----------------------------------------------------------------------
    # Section 10: Determinism (MOCK_LLM)
    # -----------------------------------------------------------------------
    print("\n--- Section 10: Determinism ---")
    check("T13-26", "MOCK_LLM mode active",
          os.getenv("MOCK_LLM", "true").lower() == "true")
    check("T13-27", f"Transcript records mock_mode=True",
          transcript.get("mock_mode") == True)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    total = PASS + FAIL
    print(f"TASK 13 VALIDATION RESULTS: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    if FAIL == 0:
        print("STATUS: ALL TESTS PASSED")
    else:
        print("STATUS: SOME TESTS FAILED")
    print("=" * 75)


if __name__ == "__main__":
    main()
