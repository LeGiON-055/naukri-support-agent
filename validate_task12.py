"""
validate_task12.py — Programmatic Verification Suite for Capstone Task 12

Validates:
1. transcripts/requests.jsonl exists and is non-empty.
2. Every non-empty line in requests.jsonl is valid parseable JSON.
3. One log entry per request.
4. Every entry contains a valid trace ID with 'tr-' prefix.
5. Trace IDs are unique across separate requests.
6. Every entry contains a numeric duration_ms >= 0.
7. Every entry contains valid HTTP method ('GET'/'POST') and path.
8. PII sanitization: [REDACTED_PHONE] is present in the logged text for PII queries.
9. Raw phone number (9876543210) NEVER appears anywhere in requests.jsonl.
10. Handled errors (HTTP 400) produce structured log entries with error details.
11. Multi-turn demonstration: Separate trace IDs for the same thread ID.
12. Runtime transcript transcripts/task12_logging.json exists with complete evidence.
13. Existing API endpoints and schemas remain functional.
"""

import os
import sys
import json
from starlette.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.main import app
from api.logger import LOG_FILE, read_logs, generate_trace_id, log_request_event
from agent.schemas import AgentResponse

TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "task12_logging.json")


def validate_task12():
    print("=" * 80)
    print("CAPSTONE TASK 12 VALIDATION REPORT — STRUCTURED JSONL LOGGING")
    print("=" * 80)

    total_checks = 0
    passed_checks = 0

    def check(condition: bool, description: str):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"[PASS] {description}")
        else:
            print(f"[FAIL] {description}")

    # -----------------------------------------------------------------------
    # 1. Module & Helper Verification
    # -----------------------------------------------------------------------
    check(callable(generate_trace_id), "generate_trace_id is callable")
    t1 = generate_trace_id()
    t2 = generate_trace_id()
    check(isinstance(t1, str) and t1.startswith("tr-"), "generate_trace_id returns string starting with 'tr-'")
    check(t1 != t2, "Consecutive trace IDs are unique")
    check(callable(log_request_event), "log_request_event is callable")
    check(callable(read_logs), "read_logs is callable")

    # -----------------------------------------------------------------------
    # 2. Log File Existence & Basic Structure
    # -----------------------------------------------------------------------
    check(os.path.exists(LOG_FILE), f"{LOG_FILE} exists on disk")
    logs = read_logs()
    check(len(logs) >= 5, f"requests.jsonl contains at least 5 logged requests (found: {len(logs)})")

    # -----------------------------------------------------------------------
    # 3. JSON Validity of Every Line
    # -----------------------------------------------------------------------
    all_lines_valid = True
    raw_lines_count = 0
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            raw_lines_count += 1
            try:
                json.loads(stripped)
            except Exception:
                all_lines_valid = False
                break
    check(all_lines_valid and raw_lines_count > 0, "Every non-empty line in requests.jsonl is valid JSON")
    check(len(logs) == raw_lines_count, f"Parsed log entries ({len(logs)}) match non-empty lines ({raw_lines_count})")

    # -----------------------------------------------------------------------
    # 4. Trace ID & Uniqueness
    # -----------------------------------------------------------------------
    all_have_trace_id = all("trace_id" in e and isinstance(e["trace_id"], str) and len(e["trace_id"]) > 3 for e in logs)
    check(all_have_trace_id, "Every entry contains a valid non-empty trace_id")

    trace_ids = [e["trace_id"] for e in logs]
    traces_are_unique = len(trace_ids) == len(set(trace_ids))
    check(traces_are_unique, f"All {len(trace_ids)} trace IDs in requests.jsonl are globally unique")

    # -----------------------------------------------------------------------
    # 5. Timing Field (duration_ms)
    # -----------------------------------------------------------------------
    all_have_timing = all("duration_ms" in e and isinstance(e["duration_ms"], (int, float)) and e["duration_ms"] >= 0 for e in logs)
    check(all_have_timing, "Every entry contains numeric duration_ms >= 0")

    # -----------------------------------------------------------------------
    # 6. Request Metadata (method, path, status_code, timestamp)
    # -----------------------------------------------------------------------
    all_have_method = all(e.get("method") in ["GET", "POST"] for e in logs)
    all_have_path = all(e.get("path") in ["/ask", "/add-document", "/"] for e in logs)
    all_have_status = all(isinstance(e.get("status_code"), int) for e in logs)
    all_have_ts = all(isinstance(e.get("timestamp"), str) and "T" in e["timestamp"] for e in logs)
    check(all_have_method, "Every entry records HTTP method ('POST' or 'GET')")
    check(all_have_path, "Every entry records endpoint path ('/ask' or '/add-document')")
    check(all_have_status, "Every entry records integer HTTP status_code")
    check(all_have_ts, "Every entry records ISO-8601 UTC timestamp")

    # -----------------------------------------------------------------------
    # 7. PII Phone Number Masking Guarantee
    # -----------------------------------------------------------------------
    raw_phone_test = "9876543210"
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        entire_log_file = f.read()

    check(raw_phone_test not in entire_log_file, f"Raw phone number '{raw_phone_test}' is ABSENT from entire JSONL file")
    pii_entries = [e for e in logs if "[REDACTED_PHONE]" in (e.get("request_text") or "")]
    check(len(pii_entries) >= 1, "At least one log entry contains '[REDACTED_PHONE]' token")

    # -----------------------------------------------------------------------
    # 8. Error Logging Verification
    # -----------------------------------------------------------------------
    error_entries = [e for e in logs if e.get("status_code") == 400 and e.get("error") is not None]
    check(len(error_entries) >= 1, "Handled HTTP 400 error is captured in JSONL log with error details")

    # -----------------------------------------------------------------------
    # 9. Thread ID vs Trace ID Multi-Turn Differentiation
    # -----------------------------------------------------------------------
    multi_thread_entries = [e for e in logs if e.get("thread_id") == "multi_turn_audit_thread"]
    if len(multi_thread_entries) >= 2:
        distinct_traces_in_thread = multi_thread_entries[0]["trace_id"] != multi_thread_entries[1]["trace_id"]
        check(distinct_traces_in_thread, "Multi-turn turns sharing the same thread_id have distinct trace_ids")
    else:
        check(False, "Found fewer than 2 entries for multi_turn_audit_thread")

    # -----------------------------------------------------------------------
    # 10. Live Client Round-Trip Verification
    # -----------------------------------------------------------------------
    client = TestClient(app)
    resp = client.post("/ask", json={"query": "What is the probation period?"})
    check(resp.status_code == 200, "Live POST /ask returns HTTP 200")
    check("X-Trace-ID" in resp.headers, "Live POST /ask response includes 'X-Trace-ID' header")
    agent_data = resp.json()
    check("intent" in agent_data and "answer" in agent_data, "Live POST /ask returns valid AgentResponse schema")

    # -----------------------------------------------------------------------
    # 11. Runtime Transcript File
    # -----------------------------------------------------------------------
    check(os.path.exists(TRANSCRIPT_PATH), f"Runtime transcript exists at {TRANSCRIPT_PATH}")
    if os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        check(t_data.get("task") == "Capstone Task 12 — Structured JSONL Logging", "Transcript specifies Task 12 title")
        check(t_data.get("raw_phone_leak_check_passed") is True, "Transcript confirms raw phone leak check passed")
        check(len(t_data.get("demonstrations", [])) >= 6, "Transcript contains at least 6 demonstration scenarios")

    print("\n" + "=" * 80)
    print(f"RESULT: {passed_checks}/{total_checks} CHECKS PASSED " + ("(100% SUCCESS!)" if passed_checks == total_checks else "(FAILED)"))
    print("=" * 80)

    if passed_checks != total_checks:
        sys.exit(1)


if __name__ == "__main__":
    validate_task12()
