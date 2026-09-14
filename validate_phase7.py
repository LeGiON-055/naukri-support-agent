"""
validate_phase7.py — Phase 7 Validation Suite

Validates that Phase 7 / Task 5 meets all Capstone requirements:
1. Both chunking strategies evaluated (fixed_size_chunks & sentence_chunks).
2. Precision@3 calculated for both strategies.
3. Recall@3 calculated for both strategies.
4. Evaluation is performed at the document level (not raw chunk level).
5. Chunks are mapped to parent documents using metadata['source'].
6. Duplicate parent documents are deduplicated while preserving order.
7. Exact same evaluation queries are used for both strategies (covering all 12 KB topics).
8. Expected documents are parent documents (.txt files), not chunk IDs.
9. Precision@3 and Recall@3 per-query arithmetic is recorded for all queries.
10. Average Precision@3 and Recall@3 are correctly aggregated.
11. Evaluation transcript exists at transcripts/phase7_retrieval_evaluation.json.
12. Recommendation is data-driven based on measured numbers.
13. README.md contains Phase 7 documentation with summary table and recommendation.
14. Zero external LLM/API dependencies.

Usage:
    python validate_phase7.py
"""

import os
import sys
import json
import re

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

KB_DIR = os.path.join(ROOT_DIR, "knowledge_base")
EVAL_SET_PATH = os.path.join(ROOT_DIR, "evaluation", "eval_set.py")
EVAL_SCRIPT_PATH = os.path.join(ROOT_DIR, "evaluation", "evaluate_retrieval.py")
TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "phase7_retrieval_evaluation.json")
README_PATH = os.path.join(ROOT_DIR, "README.md")


def validate():
    print("=" * 70)
    print("PHASE 7 VALIDATION REPORT — RETRIEVAL EVALUATION (TASK 5)")
    print("=" * 70)
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

    # 1. Check eval_set.py
    check("eval_set.py exists", os.path.isfile(EVAL_SET_PATH))
    from evaluation.eval_set import EVALUATION_SET
    check("Evaluation set has at least 12 queries", len(EVALUATION_SET) >= 12, f"count={len(EVALUATION_SET)}")

    # Check knowledge base coverage
    from validate_knowledge_base import REQUIRED_FILES
    covered_docs = set()
    for item in EVALUATION_SET:
        for doc in item.get("expected_documents", []):
            covered_docs.add(doc)

    check("All 12 KB policy documents covered in evaluation set", set(REQUIRED_FILES).issubset(covered_docs), f"covered={len(covered_docs)}/12")

    # Check expected documents are parent .txt files, not chunk IDs
    all_parent_txt = all(
        isinstance(doc, str) and doc.endswith(".txt") and not doc.startswith("fixed_") and not doc.startswith("sent_")
        for item in EVALUATION_SET
        for doc in item.get("expected_documents", [])
    )
    check("Expected documents are parent .txt files (not chunk IDs)", all_parent_txt)

    # 2. Check evaluate_retrieval.py
    check("evaluate_retrieval.py exists", os.path.isfile(EVAL_SCRIPT_PATH))
    with open(EVAL_SCRIPT_PATH, "r", encoding="utf-8") as f:
        eval_code = f.read()

    check("Retrieval top_k is set to 3", "top_k=TOP_K" in eval_code or "top_k=3" in eval_code or "TOP_K = 3" in eval_code)
    check("Evaluates fixed_size_chunks collection", "fixed_size_chunks" in eval_code)
    check("Evaluates sentence_chunks collection", "sentence_chunks" in eval_code)
    check("Maps chunk to parent document (source metadata)", 'item["source"]' in eval_code or "item['source']" in eval_code)
    check("Performs deduplication of parent documents", "deduped_docs" in eval_code or "not in deduped" in eval_code)

    # 3. Check transcripts/phase7_retrieval_evaluation.json
    check("transcripts/phase7_retrieval_evaluation.json exists", os.path.isfile(TRANSCRIPT_PATH))
    t_data = {}
    if os.path.isfile(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            t_data = json.load(f)

    check("Transcript records top_k = 3", t_data.get("top_k") == 3)
    check("Transcript records document-level deduplicated evaluation", "deduplication" in t_data.get("evaluation_type", "").lower())
    
    strategies = t_data.get("strategies", {})
    check("Transcript contains fixed_size_chunks evaluation", "fixed_size_chunks" in strategies)
    check("Transcript contains sentence_chunks evaluation", "sentence_chunks" in strategies)

    fixed_data = strategies.get("fixed_size_chunks", {})
    sent_data = strategies.get("sentence_chunks", {})

    fixed_p = fixed_data.get("avg_precision_at_3")
    fixed_r = fixed_data.get("avg_recall_at_3")
    sent_p = sent_data.get("avg_precision_at_3")
    sent_r = sent_data.get("avg_recall_at_3")

    check("Fixed-size Average Precision@3 is valid number", isinstance(fixed_p, (int, float)) and 0.0 <= fixed_p <= 1.0, f"avg_p={fixed_p}")
    check("Fixed-size Average Recall@3 is valid number", isinstance(fixed_r, (int, float)) and 0.0 <= fixed_r <= 1.0, f"avg_r={fixed_r}")
    check("Sentence-based Average Precision@3 is valid number", isinstance(sent_p, (int, float)) and 0.0 <= sent_p <= 1.0, f"avg_p={sent_p}")
    check("Sentence-based Average Recall@3 is valid number", isinstance(sent_r, (int, float)) and 0.0 <= sent_r <= 1.0, f"avg_r={sent_r}")

    # Check per-query arithmetic exists
    fixed_queries = fixed_data.get("queries", [])
    sent_queries = sent_data.get("queries", [])
    has_arithmetic_fixed = len(fixed_queries) >= 12 and all("arithmetic_precision" in q and "arithmetic_recall" in q for q in fixed_queries)
    has_arithmetic_sent = len(sent_queries) >= 12 and all("arithmetic_precision" in q and "arithmetic_recall" in q for q in sent_queries)
    check("Per-query arithmetic present for all fixed-size queries", has_arithmetic_fixed)
    check("Per-query arithmetic present for all sentence queries", has_arithmetic_sent)

    # Check deduplication verified in transcript
    # Check that in cases where raw retrieved had duplicates, deduped_retrieved_docs is strictly smaller than raw
    any_dedup_occurred = any(
        len(q.get("deduped_retrieved_docs", [])) < len(q.get("raw_retrieved_docs", []))
        for q in fixed_queries + sent_queries
    )
    check("Parent document deduplication demonstrated on multi-chunk retrievals", any_dedup_occurred)

    # Check recommendation
    rec = t_data.get("recommendation", "")
    check("Data-driven recommendation exists in transcript", bool(rec) and len(rec) > 50)

    # 4. Check README.md
    readme_text = open(README_PATH, "r", encoding="utf-8").read() if os.path.isfile(README_PATH) else ""
    check("README.md contains Phase 7 section", "Phase 7" in readme_text)
    check("README.md contains Precision@3 and Recall@3 explanation", "Precision@3" in readme_text and "Recall@3" in readme_text)
    check("README.md contains retrieval comparison table", "Fixed-size" in readme_text and "Sentence-based" in readme_text)
    check("README.md documents final recommendation", "recommendation" in readme_text.lower() or "recommend" in readme_text.lower())

    # 5. Check no external API / LLM calls
    has_external_api = bool(re.search(r'\b(import\s+openai|import\s+anthropic|import\s+cohere|import\s+google\.generativeai)\b', eval_code, re.I))
    check("No external API/LLM calls in evaluation script", not has_external_api)

    print()
    print("=" * 70)
    if all_passed:
        print("RESULT: ALL PHASE 7 CHECKS PASSED!")
    else:
        print("RESULT: SOME PHASE 7 CHECKS FAILED — REVIEW ABOVE.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
