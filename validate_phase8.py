"""
validate_phase8.py — Phase 8 Validation Suite (Task 6)

Verifies all Capstone Task 6 requirements:
1. Tool function check_job_application_status(record_id) exists and is callable.
2. Uses deterministic dataset from dataset.py (JOB_APPLICATIONS).
3. Returns status and expected_salary_inr.
4. Calculates escalation_score strictly bounded in [0.0, 1.0].
5. Combines flagged_priority_review and normalized days_since_created.
6. Exact formula is documented in code and README.
7. Escalation threshold is explicitly defined (0.45).
8. Escalation threshold is justified using empirical dataset distribution (80th percentile).
9. Invalid/nonexistent record IDs are handled gracefully with found=False.
10. Demonstrations cover high-urgency, standard, and invalid cases.
11. Transcript exists at transcripts/phase8_application_tool.json.
12. README.md contains complete Phase 8 documentation.

Usage:
    python validate_phase8.py
"""

import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from dataset import JOB_APPLICATIONS
from agent.tools import (
    check_job_application_status,
    calculate_escalation_score,
    ESCALATION_THRESHOLD,
    get_all_records,
)

TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "phase8_application_tool.json")
README_PATH = os.path.join(ROOT_DIR, "README.md")


def validate():
    print("=" * 70)
    print("PHASE 8 VALIDATION REPORT — CUSTOM APPLICATION TOOL (TASK 6)")
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

    # 1. Tool existence and callability
    check("check_job_application_status function exists and is callable", callable(check_job_application_status))
    check("calculate_escalation_score function exists and is callable", callable(calculate_escalation_score))

    # 2. Lookup on valid record
    test_id = "APP-001"
    res1 = check_job_application_status(test_id)
    check("Valid ID returns found=True", res1.get("found") is True)
    check("Returns status field", res1.get("status") in ["Applied", "Screening", "Interview Scheduled", "Offered", "Rejected"])
    check("Returns expected_salary_inr field as int", isinstance(res1.get("expected_salary_inr"), int))
    check("Returns days_since_created field", isinstance(res1.get("days_since_created"), int))
    check("Returns flagged_priority_review field", isinstance(res1.get("flagged_priority_review"), bool))
    check("Returns escalation_score as float", isinstance(res1.get("escalation_score"), float))
    check("Returns escalated boolean", isinstance(res1.get("escalated"), bool))
    check("Returns escalation_threshold", res1.get("escalation_threshold") == ESCALATION_THRESHOLD)

    # 3. Verify accuracy against underlying dataset
    source_rec = next((r for r in JOB_APPLICATIONS if r["record_id"] == test_id), None)
    check("Underlying record found in dataset", source_rec is not None)
    if source_rec:
        check("Status matches dataset record", res1["status"] == source_rec["status"])
        check("Salary matches dataset record", res1["expected_salary_inr"] == source_rec["expected_salary_inr"])
        check("Days match dataset record", res1["days_since_created"] == source_rec["days_since_created"])
        check("Priority flag matches dataset record", res1["flagged_priority_review"] == source_rec["flagged_priority_review"])

    # 4. Range and bounds verification across all 45 records
    all_scores = [calculate_escalation_score(r) for r in JOB_APPLICATIONS]
    all_bounded = all(0.0 <= s <= 1.0 for s in all_scores)
    check("Escalation scores strictly bounded in [0.0, 1.0] across all 45 records", all_bounded, f"min={min(all_scores):.4f}, max={max(all_scores):.4f}")

    # 5. Sensitivity checks for both formula terms
    dummy_flag_only = {"flagged_priority_review": True, "days_since_created": 0}
    dummy_days_only = {"flagged_priority_review": False, "days_since_created": 30}
    dummy_zero = {"flagged_priority_review": False, "days_since_created": 0}

    s_flag = calculate_escalation_score(dummy_flag_only)
    s_days = calculate_escalation_score(dummy_days_only)
    s_zero = calculate_escalation_score(dummy_zero)

    check("Priority flag contributes positively to score", s_flag > s_zero, f"diff={s_flag - s_zero:.4f}")
    check("Recency (days) contributes positively to score", s_days > s_zero, f"diff={s_days - s_zero:.4f}")
    check("Zero inputs produce exactly 0.0", s_zero == 0.0)

    # 6. Empirical threshold justification
    scores_sorted = sorted(all_scores)
    p80_val = scores_sorted[int(0.80 * len(scores_sorted))]
    check("Escalation threshold is explicitly set to 0.45", ESCALATION_THRESHOLD == 0.45)
    check("Threshold aligns with dataset P80 distribution", abs(ESCALATION_THRESHOLD - p80_val) <= 0.05, f"P80={p80_val:.4f}, threshold={ESCALATION_THRESHOLD}")

    # 7. Invalid record ID handling
    invalid_id = "APP-999"
    res_inv = check_job_application_status(invalid_id)
    check("Invalid record ID returns found=False", res_inv.get("found") is False)
    check("Invalid record returns escalated=False", res_inv.get("escalated") is False)
    check("Invalid record returns descriptive message", "not found" in res_inv.get("message", "").lower())
    check("None/empty ID handled without crashing", check_job_application_status("").get("found") is False)

    # 8. Determinism check
    res_a = check_job_application_status("APP-035")
    res_b = check_job_application_status("APP-035")
    check("Tool execution is 100% deterministic", res_a == res_b)

    # 9. Transcript check
    check("transcripts/phase8_application_tool.json exists", os.path.isfile(TRANSCRIPT_PATH))
    if os.path.isfile(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        demos = t_data.get("demonstrations", [])
        check("Transcript records at least 5 demonstration cases", len(demos) >= 5, f"found {len(demos)}")
        has_invalid = any(d.get("query_record_id") == "APP-999" for d in demos)
        check("Transcript records invalid ID test case", has_invalid)
        has_escalated = any(d.get("result", {}).get("escalated") is True for d in demos)
        check("Transcript records escalated case", has_escalated)

    # 10. README.md check
    readme_text = open(README_PATH, "r", encoding="utf-8").read() if os.path.isfile(README_PATH) else ""
    check("README.md contains Phase 8 section", "Phase 8" in readme_text)
    check("README.md documents escalation formula", "0.50" in readme_text and "priority_flag" in readme_text)
    check("README.md documents escalation threshold (0.45)", "0.45" in readme_text)
    check("README.md documents 80th percentile justification", "80th percentile" in readme_text.lower() or "p80" in readme_text.lower())

    print()
    print("=" * 70)
    if all_passed:
        print("RESULT: ALL PHASE 8 CHECKS PASSED!")
    else:
        print("RESULT: SOME PHASE 8 CHECKS FAILED — REVIEW ABOVE.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
