"""
demo_phase9.py — Phase 9 Demonstration & Transcript Generator (Task 9)

Demonstrates the unified AgentResponse schema and validate_agent_response() helper:
1. Positive Validation Tests (All 5 Domain Scenarios):
   - Case 1: In-scope policy inquiry with citations
   - Case 2: Standard application status lookup (normal progression)
   - Case 3: Escalated application status lookup (recruiter escalation)
   - Case 4: Out-of-scope fallback (retrieval threshold rejection)
   - Case 5: Security violation refusal (prompt injection detection)

2. Negative Validation Tests (Deliberate Malformed Payloads):
   - Neg 1: Invalid intent literal
   - Neg 2: Empty answer string
   - Neg 3: Wrong expected salary data type
   - Neg 4: Escalation score out of [0.0, 1.0] bounds

3. Exports the full OpenAPI / JSON Schema.
4. Saves transcript to transcripts/phase9_schemas.json.
"""

import os
import sys
import json
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.schemas import (
    AgentResponse,
    ApplicationDetails,
    RefusalInfo,
    validate_agent_response,
    build_policy_response,
    build_status_response,
    build_refusal_response,
    ALLOWED_INTENTS,
)

TRANSCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "transcripts",
    "phase9_schemas.json"
)


def run_demonstration():
    print("=" * 75)
    print("PHASE 9 — STRUCTURED OUTPUT SCHEMAS DEMONSTRATION (TASK 9)")
    print("=" * 75)
    print(f"Allowed Intents:      {ALLOWED_INTENTS}")
    print(f"Validation Engine:    Pydantic V2 (validate_agent_response)")
    print("=" * 75)
    print()

    transcript = {
        "phase": "Phase 9 — Structured Output Schemas & Response Standardization",
        "timestamp": datetime.now().isoformat(),
        "json_schema": AgentResponse.model_json_schema(),
        "positive_tests": [],
        "negative_tests": [],
    }

    # =========================================================================
    # PART 1: POSITIVE VALIDATION TESTS (All 5 Domain Scenarios)
    # =========================================================================
    print("-" * 75)
    print("PART 1: POSITIVE VALIDATION TESTS (All 5 Domain Scenarios)")
    print("-" * 75)

    positive_cases = [
        {
            "case_id": "POS_1_POLICY_INQUIRY",
            "scenario": "In-Scope Policy Inquiry with Source Citations",
            "payload": {
                "intent": "policy_inquiry",
                "answer": (
                    "All new employees are placed on a probation period of 6 months "
                    "from their date of joining. Performance reviews determine confirmation."
                ),
                "sources": ["probation_period.txt"],
                "is_escalated": False,
                "application_details": None,
                "refusal_info": None,
            },
        },
        {
            "case_id": "POS_2_STATUS_ROUTINE",
            "scenario": "Standard Routine Application Status (Normal Progression)",
            "payload": {
                "intent": "application_status",
                "answer": (
                    "Application 'APP-002' for Data Analyst is currently in 'Screening' status "
                    "with expected salary of INR 980,000. It is progressing normally."
                ),
                "sources": [],
                "is_escalated": False,
                "application_details": {
                    "record_id": "APP-002",
                    "category": "Data Analyst",
                    "status": "Screening",
                    "expected_salary_inr": 980000,
                    "days_since_created": 7,
                    "flagged_priority_review": False,
                    "escalation_score": 0.1167,
                    "escalation_threshold": 0.45,
                },
                "refusal_info": None,
            },
        },
        {
            "case_id": "POS_3_STATUS_ESCALATED",
            "scenario": "Escalated Application Status (Urgent Recruiter Intervention)",
            "payload": {
                "intent": "application_status",
                "answer": (
                    "Application 'APP-035' for Sales Associate is in 'Screening' status. "
                    "This case has been ESCALATED (score: 0.9833 >= 0.45) for priority review."
                ),
                "sources": [],
                "is_escalated": True,
                "application_details": {
                    "record_id": "APP-035",
                    "category": "Sales Associate",
                    "status": "Screening",
                    "expected_salary_inr": 780000,
                    "days_since_created": 29,
                    "flagged_priority_review": True,
                    "escalation_score": 0.9833,
                    "escalation_threshold": 0.45,
                },
                "refusal_info": None,
            },
        },
        {
            "case_id": "POS_4_OUT_OF_SCOPE",
            "scenario": "Out-of-Scope Fallback (Retrieval Threshold Rejection)",
            "payload": {
                "intent": "out_of_scope",
                "answer": (
                    "I don't have enough information to answer that question. "
                    "Please contact HR directly for assistance."
                ),
                "sources": [],
                "is_escalated": False,
                "application_details": None,
                "refusal_info": {
                    "reason": "retrieval_threshold",
                    "details": "Query similarity 0.0123 below threshold 0.28.",
                    "similarity_score": 0.0123,
                    "threshold": 0.28,
                },
            },
        },
        {
            "case_id": "POS_5_SECURITY_VIOLATION",
            "scenario": "Security Violation Refusal (Prompt Injection Attempt)",
            "payload": {
                "intent": "security_violation",
                "answer": (
                    "I cannot comply with requests that attempt to override system rules "
                    "or reveal confidential system instructions."
                ),
                "sources": [],
                "is_escalated": False,
                "application_details": None,
                "refusal_info": {
                    "reason": "prompt_injection",
                    "details": "Matched disallowed prompt override pattern.",
                    "similarity_score": None,
                    "threshold": None,
                },
            },
        },
    ]

    for tc in positive_cases:
        cid = tc["case_id"]
        scen = tc["scenario"]
        payload = tc["payload"]

        is_valid, model_inst, err = validate_agent_response(payload)
        status_str = "VALID" if is_valid else "INVALID (ERROR)"

        print(f"[{cid}] {scen}")
        print(f"  Validation Result:  {status_str}")
        print(f"  Intent:             {payload['intent']}")
        print(f"  Escalated:          {payload['is_escalated']}")
        print(f"  Sources:            {payload['sources']}")
        print(f"  Answer Preview:     {payload['answer'][:70]}...")
        if payload["application_details"]:
            print(f"  App Details ID:     {payload['application_details']['record_id']} (Score: {payload['application_details']['escalation_score']})")
        if payload["refusal_info"]:
            print(f"  Refusal Reason:     {payload['refusal_info']['reason']}")
        print()

        transcript["positive_tests"].append({
            "case_id": cid,
            "scenario": scen,
            "is_valid": is_valid,
            "error": err,
            "validated_payload": model_inst.model_dump() if model_inst else None,
        })

    # =========================================================================
    # PART 2: NEGATIVE VALIDATION TESTS (Deliberate Malformed Payloads)
    # =========================================================================
    print("-" * 75)
    print("PART 2: NEGATIVE VALIDATION TESTS (Deliberate Malformed Payloads)")
    print("-" * 75)

    negative_cases = [
        {
            "case_id": "NEG_1_INVALID_INTENT",
            "scenario": "Invalid intent literal ('pizza_order')",
            "payload": {
                "intent": "pizza_order",
                "answer": "Here is your pizza order.",
                "sources": [],
                "is_escalated": False,
            },
            "expected_error_keyword": "intent",
        },
        {
            "case_id": "NEG_2_EMPTY_ANSWER",
            "scenario": "Empty answer string (violates min_length=1)",
            "payload": {
                "intent": "policy_inquiry",
                "answer": "",
                "sources": ["probation_period.txt"],
                "is_escalated": False,
            },
            "expected_error_keyword": "answer",
        },
        {
            "case_id": "NEG_3_WRONG_SALARY_TYPE",
            "scenario": "Wrong expected_salary data type (string instead of integer)",
            "payload": {
                "intent": "application_status",
                "answer": "Application details retrieved.",
                "sources": [],
                "is_escalated": False,
                "application_details": {
                    "record_id": "APP-001",
                    "expected_salary_inr": "not_an_integer",
                },
            },
            "expected_error_keyword": "expected_salary_inr",
        },
        {
            "case_id": "NEG_4_SCORE_OUT_OF_BOUNDS",
            "scenario": "Escalation score exceeds 1.0 (violates le=1.0)",
            "payload": {
                "intent": "application_status",
                "answer": "Application escalated.",
                "sources": [],
                "is_escalated": True,
                "application_details": {
                    "record_id": "APP-035",
                    "escalation_score": 1.85,
                },
            },
            "expected_error_keyword": "escalation_score",
        },
    ]

    for tc in negative_cases:
        cid = tc["case_id"]
        scen = tc["scenario"]
        payload = tc["payload"]
        exp_kw = tc["expected_error_keyword"]

        is_valid, model_inst, err = validate_agent_response(payload)
        caught_expected = (not is_valid) and (err is not None) and (exp_kw in err.lower())

        print(f"[{cid}] {scen}")
        print(f"  Valid:              {is_valid} (Correctly rejected: {caught_expected})")
        print(f"  Caught Error:       {err.splitlines()[0] if err else 'None'}")
        print()

        transcript["negative_tests"].append({
            "case_id": cid,
            "scenario": scen,
            "is_valid": is_valid,
            "caught_expected_error": caught_expected,
            "validation_error_message": err,
        })

    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print("=" * 75)
    print(f"Phase 9 demonstration transcript successfully saved to:\n  {TRANSCRIPT_PATH}")
    print("=" * 75)
    return transcript


if __name__ == "__main__":
    run_demonstration()
