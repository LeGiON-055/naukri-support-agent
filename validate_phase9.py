"""
validate_phase9.py — Phase 9 Validation Suite (Task 9)

Verifies all Capstone Task 9 requirements:
1. Structured response schema (AgentResponse) defined using Pydantic.
2. Nested models ApplicationDetails and RefusalInfo exist and enforce types.
3. Allowed intent literals enforced (policy_inquiry, application_status, out_of_scope, security_violation).
4. In-code validation helper validate_agent_response() exists and functions.
5. All 5 positive domain scenarios pass schema validation.
6. Malformed payloads (invalid intent, empty answer, wrong salary type, score > 1.0) are rejected.
7. Generates compliant JSON Schema (OpenAPI compatible).
8. Transcript exists at transcripts/phase9_schemas.json.
9. README.md contains complete Phase 9 documentation.

Usage:
    python validate_phase9.py
"""

import os
import sys
import json
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from agent.schemas import (
    AgentResponse,
    ApplicationDetails,
    RefusalInfo,
    validate_agent_response,
    ALLOWED_INTENTS,
)

TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "phase9_schemas.json")
README_PATH = os.path.join(ROOT_DIR, "README.md")


def validate():
    print("=" * 70)
    print("PHASE 9 VALIDATION REPORT — STRUCTURED OUTPUT SCHEMAS (TASK 9)")
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

    # 1. Models existence and inheritance
    check("AgentResponse class exists and inherits from BaseModel", issubclass(AgentResponse, BaseModel))
    check("ApplicationDetails class exists and inherits from BaseModel", issubclass(ApplicationDetails, BaseModel))
    check("RefusalInfo class exists and inherits from BaseModel", issubclass(RefusalInfo, BaseModel))
    check("validate_agent_response helper exists and is callable", callable(validate_agent_response))

    # 2. JSON Schema generation
    json_schema = AgentResponse.model_json_schema()
    check("AgentResponse generates valid JSON Schema dictionary", isinstance(json_schema, dict) and "properties" in json_schema)
    check("JSON Schema defines required properties", "required" in json_schema and "intent" in json_schema["required"] and "answer" in json_schema["required"])

    # 3. Intent constraints
    expected_intents = {"policy_inquiry", "application_status", "out_of_scope", "security_violation"}
    check("All 4 standard intents present in ALLOWED_INTENTS", set(ALLOWED_INTENTS) == expected_intents)

    # 4. Positive Validation Tests (All 5 Domain Scenarios)
    # Case 1: Policy inquiry
    p_policy = {
        "intent": "policy_inquiry",
        "answer": "Probation period is 6 months.",
        "sources": ["probation_period.txt"],
        "is_escalated": False,
    }
    v_pol, m_pol, _ = validate_agent_response(p_policy)
    check("Policy inquiry payload passes validation", v_pol is True)

    # Case 2: Routine application status
    p_status_norm = {
        "intent": "application_status",
        "answer": "Application is in Screening status.",
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
    }
    v_stat_n, m_stat_n, _ = validate_agent_response(p_status_norm)
    check("Routine application status payload passes validation", v_stat_n is True)

    # Case 3: Escalated application status
    p_status_esc = {
        "intent": "application_status",
        "answer": "Application is escalated to recruiter.",
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
    }
    v_stat_e, m_stat_e, _ = validate_agent_response(p_status_esc)
    check("Escalated application status payload passes validation", v_stat_e is True)

    # Case 4: Out-of-scope fallback
    p_out = {
        "intent": "out_of_scope",
        "answer": "I don't have enough information.",
        "sources": [],
        "is_escalated": False,
        "refusal_info": {
            "reason": "retrieval_threshold",
            "details": "Cosine similarity 0.0123 < 0.28.",
            "similarity_score": 0.0123,
            "threshold": 0.28,
        },
    }
    v_out, m_out, _ = validate_agent_response(p_out)
    check("Out-of-scope fallback payload passes validation", v_out is True)

    # Case 5: Security violation
    p_sec = {
        "intent": "security_violation",
        "answer": "I cannot fulfill this request.",
        "sources": [],
        "is_escalated": False,
        "refusal_info": {
            "reason": "prompt_injection",
            "details": "Disallowed system prompt override.",
        },
    }
    v_sec, m_sec, _ = validate_agent_response(p_sec)
    check("Security violation refusal payload passes validation", v_sec is True)

    # 5. Negative Rejection Tests (Deliberate Malformed Payloads)
    # Neg 1: Invalid intent
    v_bad_int, _, err_int = validate_agent_response({"intent": "invalid_intent", "answer": "text"})
    check("Rejects invalid intent literal", v_bad_int is False and "intent" in err_int.lower())

    # Neg 2: Empty answer
    v_empty_ans, _, err_ans = validate_agent_response({"intent": "policy_inquiry", "answer": ""})
    check("Rejects empty answer string (min_length=1)", v_empty_ans is False and "answer" in err_ans.lower())

    # Neg 3: Wrong salary type (string instead of int)
    bad_salary = {
        "intent": "application_status",
        "answer": "text",
        "application_details": {"record_id": "APP-001", "expected_salary_inr": "not_an_int"},
    }
    v_bad_sal, _, err_sal = validate_agent_response(bad_salary)
    check("Rejects non-integer expected_salary_inr", v_bad_sal is False and "expected_salary_inr" in err_sal.lower())

    # Neg 4: Escalation score > 1.0
    bad_score = {
        "intent": "application_status",
        "answer": "text",
        "application_details": {"record_id": "APP-001", "escalation_score": 1.75},
    }
    v_bad_score, _, err_score = validate_agent_response(bad_score)
    check("Rejects escalation_score > 1.0 (le=1.0)", v_bad_score is False and "escalation_score" in err_score.lower())

    # 6. Transcript verification
    check("transcripts/phase9_schemas.json exists", os.path.isfile(TRANSCRIPT_PATH))
    if os.path.isfile(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        check("Transcript records JSON Schema", "json_schema" in t_data)
        check("Transcript records at least 5 positive tests", len(t_data.get("positive_tests", [])) >= 5)
        check("Transcript records at least 4 negative tests", len(t_data.get("negative_tests", [])) >= 4)
        all_neg_caught = all(tc.get("caught_expected_error") is True for tc in t_data.get("negative_tests", []))
        check("Transcript records that all negative tests were correctly caught", all_neg_caught)

    # 7. README documentation check
    readme_text = open(README_PATH, "r", encoding="utf-8").read() if os.path.isfile(README_PATH) else ""
    check("README.md contains Phase 9 section", "Phase 9" in readme_text)
    check("README.md documents AgentResponse fields", "AgentResponse" in readme_text and "application_details" in readme_text)
    check("README.md documents all 4 allowed intents", all(i in readme_text for i in ALLOWED_INTENTS))
    check("README.md documents validate_agent_response", "validate_agent_response" in readme_text)

    print()
    print("=" * 70)
    if all_passed:
        print("RESULT: ALL PHASE 9 CHECKS PASSED!")
    else:
        print("RESULT: SOME PHASE 9 CHECKS FAILED — REVIEW ABOVE.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
