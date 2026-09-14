"""
demo_phase8.py — Phase 8 Demonstration & Transcript Generator

Demonstrates check_job_application_status() across key business scenarios:
  Case 1: High-urgency priority ticket (APP-035: Priority True, Days 29 -> Escalated)
  Case 2: Standard routine ticket (APP-002: Priority False, Days 7 -> Normal)
  Case 3: Brand-new priority ticket (APP-005: Priority True, Days 0 -> Escalated)
  Case 4: Long-wait unflagged ticket (APP-032: Priority False, Days 28 -> Escalated)
  Case 5: Invalid/Nonexistent ID (APP-999 -> Graceful refusal)

Saves complete execution transcript to:
  transcripts/phase8_application_tool.json
"""

import os
import sys
import json
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.tools import (
    check_job_application_status,
    calculate_escalation_score,
    ESCALATION_THRESHOLD,
    get_all_records,
)

TRANSCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "transcripts",
    "phase8_application_tool.json"
)


def run_demonstration():
    print("=" * 75)
    print("PHASE 8 — CUSTOM TOOL & ESCALATION SCORING DEMONSTRATION")
    print("=" * 75)
    print(f"Tool Function:         check_job_application_status(record_id)")
    print(f"Escalation Formula:    0.50 * priority_flag + 0.50 * (days_since_created / 30.0)")
    print(f"Escalation Threshold:  {ESCALATION_THRESHOLD} (Empirical P80 = 0.4667)")
    print("=" * 75)
    print()

    # Dataset distribution summary
    all_recs = get_all_records()
    all_scores = [calculate_escalation_score(r) for r in all_recs]
    all_scores_sorted = sorted(all_scores)

    min_s = min(all_scores)
    max_s = max(all_scores)
    median_s = all_scores_sorted[len(all_scores_sorted) // 2]
    p80_index = int(0.80 * len(all_scores_sorted))
    p80_val = all_scores_sorted[p80_index]

    escalated_total = sum(1 for s in all_scores if s >= ESCALATION_THRESHOLD)

    print(f"Dataset Distribution ({len(all_recs)} records):")
    print(f"  Min Score:    {min_s:.4f}")
    print(f"  Max Score:    {max_s:.4f}")
    print(f"  Median (P50): {median_s:.4f}")
    print(f"  P80 Score:    {p80_val:.4f}")
    print(f"  Escalated:    {escalated_total}/{len(all_recs)} ({escalated_total/len(all_recs)*100:.1f}%)")
    print("-" * 75)
    print()

    test_cases = [
        {
            "case_id": "CASE_1_HIGH_URGENCY_PRIORITY",
            "description": "High-urgency priority application with long waiting time",
            "record_id": "APP-035",
            "expected_escalated": True,
        },
        {
            "case_id": "CASE_2_STANDARD_ROUTINE",
            "description": "Standard low-latency application without priority flag",
            "record_id": "APP-002",
            "expected_escalated": False,
        },
        {
            "case_id": "CASE_3_BRAND_NEW_PRIORITY",
            "description": "Newly created application with priority review flag",
            "record_id": "APP-005",
            "expected_escalated": True,
        },
        {
            "case_id": "CASE_4_LONG_WAIT_UNFLAGGED",
            "description": "Long-pending unflagged application exceeding latency threshold",
            "record_id": "APP-032",
            "expected_escalated": True,
        },
        {
            "case_id": "CASE_5_INVALID_RECORD_ID",
            "description": "Invalid/nonexistent application ID",
            "record_id": "APP-999",
            "expected_escalated": False,
        },
    ]

    demo_results = []

    for tc in test_cases:
        cid = tc["case_id"]
        rid = tc["record_id"]
        desc = tc["description"]

        print(f"[{cid}] ID: '{rid}' -- {desc}")
        res = check_job_application_status(rid)

        if res["found"]:
            print(f"  Status:           {res['status']}")
            print(f"  Category:         {res['category']}")
            print(f"  Salary (INR):     INR {res['expected_salary_inr']:,}")
            print(f"  Days Since Sent:  {res['days_since_created']}")
            print(f"  Priority Flag:    {res['flagged_priority_review']}")
            print(f"  Escalation Score: {res['escalation_score']:.4f}")
            print(f"  Escalation Gate:  {'ESCALATED' if res['escalated'] else 'NORMAL PROGRESSION'}")
            print(f"  Agent Message:    {res['message']}")
        else:
            print(f"  Record Found:     False")
            print(f"  Agent Message:    {res['message']}")

        print()

        demo_results.append({
            "case_id": cid,
            "description": desc,
            "query_record_id": rid,
            "result": res,
        })

    transcript = {
        "phase": "Phase 8 — Custom Tool & Escalation Logic",
        "timestamp": datetime.now().isoformat(),
        "tool_name": "check_job_application_status",
        "formula": "escalation_score = 0.50 * priority_flag + 0.50 * (days_since_created / 30.0)",
        "escalation_threshold": ESCALATION_THRESHOLD,
        "dataset_statistics": {
            "total_records": len(all_recs),
            "min_score": min_s,
            "max_score": max_s,
            "median_score": median_s,
            "p80_score": p80_val,
            "escalated_count": escalated_total,
            "escalated_percentage": round(escalated_total / len(all_recs) * 100, 2),
        },
        "demonstrations": demo_results,
    }

    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print("=" * 75)
    print(f"Phase 8 demonstration transcript saved successfully to:\n  {TRANSCRIPT_PATH}")
    print("=" * 75)
    return transcript


if __name__ == "__main__":
    run_demonstration()
