"""
validate_phase5.py — Phase 5 Validation Script

Verifies that the RAG retrieval pipeline and empirical threshold calibration
meet all Capstone Phase 5 requirements.

Usage:
    python validate_phase5.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from rag.retrieval import retrieve, retrieve_with_threshold, CHROMA_DIR
from rag.embeddings import MODEL_NAME

ROOT_DIR = os.path.dirname(__file__)
TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "phase5_calibration.json")
README_PATH = os.path.join(ROOT_DIR, "README.md")


def validate():
    print("=" * 60)
    print("PHASE 5 VALIDATION REPORT")
    print("=" * 60)
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

    # 1. Retrieval module and functions exist
    import rag.retrieval as ret_mod
    check("rag/retrieval.py exists", True)
    check("retrieve() function exists and callable", callable(getattr(ret_mod, "retrieve", None)))
    check("retrieve_with_threshold() exists and callable", callable(getattr(ret_mod, "retrieve_with_threshold", None)))

    # 2. Local model & no external API imports
    source_code = open(ret_mod.__file__, "r", encoding="utf-8").read()
    has_external_api = any(pkg in source_code for pkg in ["openai", "anthropic", "cohere", "gemini"])
    check("No external API dependency in retrieval.py", not has_external_api)
    check("Uses same local embedding model (all-MiniLM-L6-v2)", MODEL_NAME == "all-MiniLM-L6-v2")

    # 3. Chroma DB collections and cosine distance space
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    cols = [c.name for c in client.list_collections()]
    check("Both collections exist in Chroma", "fixed_size_chunks" in cols and "sentence_chunks" in cols, f"found: {cols}")

    sent_col = client.get_collection("sentence_chunks")
    fixed_col = client.get_collection("fixed_size_chunks")
    check("sentence_chunks configured with cosine space", sent_col.metadata.get("hnsw:space") == "cosine")
    check("fixed_size_chunks configured with cosine space", fixed_col.metadata.get("hnsw:space") == "cosine")

    # 4. Retrieval execution on both collections
    test_q = "What is the probation period?"
    res_sent = retrieve(test_q, collection_name="sentence_chunks", top_k=3)
    res_fixed = retrieve(test_q, collection_name="fixed_size_chunks", top_k=3)

    check("sentence_chunks retrieval returns top-3", len(res_sent) == 3, f"got {len(res_sent)}")
    check("fixed_size_chunks retrieval returns top-3", len(res_fixed) == 3, f"got {len(res_fixed)}")

    # 5. Metadata and cosine similarity properties
    sent_top = res_sent[0]
    check("Source metadata preserved in retrieved results", "source" in sent_top and sent_top["source"].endswith(".txt"), sent_top.get("source"))
    check("Cosine similarity is properly bounded [0.0, 1.0]", 0.0 <= sent_top["cosine_similarity"] <= 1.0, f"score={sent_top['cosine_similarity']}")

    # 6. Fallback and threshold logic
    # In-scope query should pass threshold
    in_scope_eval = retrieve_with_threshold("What is the probation period?", threshold=0.28, collection_name="sentence_chunks")
    check("In-scope query accepted by threshold", in_scope_eval["is_supported"] is True, f"similarity={in_scope_eval['top1_similarity']}")

    # Out-of-scope query should be rejected by threshold
    out_scope_eval = retrieve_with_threshold("What is the weather in Mumbai?", threshold=0.28, collection_name="sentence_chunks")
    check("Out-of-scope query rejected by threshold", out_scope_eval["is_supported"] is False, f"similarity={out_scope_eval['top1_similarity']}")
    check("Out-of-scope returns 'I don't know' fallback message", out_scope_eval["answer"] is not None and "don't have enough information" in out_scope_eval["answer"])

    # 7. Calibration transcript verification
    transcript_exists = os.path.isfile(TRANSCRIPT_PATH)
    check("transcripts/phase5_calibration.json exists", transcript_exists)

    if transcript_exists:
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        calib = data.get("calibration", {})
        sent_cal = calib.get("sentence_chunks", {})
        in_scores = sent_cal.get("in_scope_scores", [])
        out_scores = sent_cal.get("out_of_scope_scores", [])

        check("At least 3 in-scope calibration queries", len(in_scores) >= 3, f"found {len(in_scores)}")
        check("At least 2 out-of-scope calibration queries", len(out_scores) >= 2, f"found {len(out_scores)}")

        min_in = sent_cal.get("min_in_scope", 0)
        max_out = sent_cal.get("max_out_scope", 1)
        check("Clear separation exists (min_in_scope > max_out_scope)", min_in > max_out, f"min_in={min_in}, max_out={max_out}")

        sel_thresh = data.get("selected_threshold", 0)
        check("Selected threshold lies strictly between groups", max_out < sel_thresh < min_in, f"threshold={sel_thresh}")

        demo = data.get("demonstration", [])
        demo_sent = [d for d in demo if d.get("collection") == "sentence_chunks"]
        demo_in = [d for d in demo_sent if d.get("is_supported")]
        demo_out = [d for d in demo_sent if not d.get("is_supported")]

        check("Demonstration contains at least 5 in-scope queries", len(demo_in) >= 5, f"found {len(demo_in)}")
        check("Demonstration contains at least 1 out-of-scope fallback", len(demo_out) >= 1, f"found {len(demo_out)}")

    # 8. README.md documentation check
    readme_content = open(README_PATH, "r", encoding="utf-8").read() if os.path.isfile(README_PATH) else ""
    check("README.md contains Phase 5 documentation section", "Phase 5" in readme_content)
    check("README.md contains calibrated threshold value", "0.28" in readme_content)

    print()
    print("=" * 60)
    if all_passed:
        print("RESULT: ALL PHASE 5 CHECKS PASSED!")
    else:
        print("RESULT: SOME CHECKS FAILED - REVIEW ABOVE.")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
