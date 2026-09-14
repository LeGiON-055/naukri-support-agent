"""
validate_task14.py — MCP Tool Lookup Validator (Capstone Task 14)

Validates that Task 14 implementation meets all capstone requirements:
1. MCP server exposes at least 1 tool via FastMCP
2. MCP client discovers tools and calls them
3. At least 2 record IDs tested (found + not-found)
4. Transcript saved to transcripts/task14_mcp.json
"""

import os
import sys
import json
import asyncio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    print("TASK 14 VALIDATION - MCP Tool Lookup")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # Section 1: Module Import
    # -----------------------------------------------------------------------
    print("\n--- Section 1: Module Import ---")
    lookup_application_status = None
    try:
        from mcp_tools.server import mcp as mcp_server, lookup_application_status
        check("T14-01", "MCP server module imports successfully", True)
    except Exception as e:
        check("T14-01", f"MCP server import failed: {e}", False)
        print(f"\nResults: {PASS} PASS / {FAIL} FAIL")
        return

    try:
        from mcp_tools.client import run_mcp_client, TEST_RECORDS
        check("T14-02", "MCP client module imports successfully", True)
    except Exception as e:
        check("T14-02", f"MCP client import failed: {e}", False)
        print(f"\nResults: {PASS} PASS / {FAIL} FAIL")
        return

    # -----------------------------------------------------------------------
    # Section 2: Server Tool Registration
    # -----------------------------------------------------------------------
    print("\n--- Section 2: Server Tool ---")
    check("T14-03", "lookup_application_status is callable",
          callable(lookup_application_status))

    # Direct call test (bypasses MCP transport)
    result = lookup_application_status("APP-001")
    check("T14-04", "Direct tool call returns dict", isinstance(result, dict))
    check("T14-05", "Direct tool call finds APP-001", result.get("found") == True)

    result_missing = lookup_application_status("APP-999")
    check("T14-06", "Direct tool call handles missing record", result_missing.get("found") == False)

    # -----------------------------------------------------------------------
    # Section 3: Client Test Records
    # -----------------------------------------------------------------------
    print("\n--- Section 3: Client Configuration ---")
    check("T14-07", f"Client has >= 2 test records (got {len(TEST_RECORDS)})",
          len(TEST_RECORDS) >= 2)

    has_valid = any("001" in r["record_id"] or "010" in r["record_id"] for r in TEST_RECORDS)
    has_invalid = any("999" in r["record_id"] for r in TEST_RECORDS)
    check("T14-08", "Client tests include a valid record ID", has_valid)
    check("T14-09", "Client tests include an invalid record ID", has_invalid)

    # -----------------------------------------------------------------------
    # Section 4: MCP Client-Server Integration
    # -----------------------------------------------------------------------
    print("\n--- Section 4: MCP Client-Server Integration ---")
    try:
        transcript = asyncio.run(run_mcp_client())
        check("T14-10", "MCP client-server integration runs successfully", True)
    except Exception as e:
        check("T14-10", f"MCP client-server integration failed: {e}", False)
        transcript = None

    if transcript:
        check("T14-11", "Transcript has tools_discovered",
              "tools_discovered" in transcript)
        check("T14-12", f"At least 1 tool discovered (got {transcript.get('tool_count', 0)})",
              transcript.get("tool_count", 0) >= 1)
        check("T14-13", f"At least 2 tool calls made (got {transcript.get('total_calls', 0)})",
              transcript.get("total_calls", 0) >= 2)
        check("T14-14", f"All calls successful ({transcript.get('successful_calls', 0)}/{transcript.get('total_calls', 0)})",
              transcript.get("successful_calls", 0) == transcript.get("total_calls", 0))

        # Check test invocations
        invocations = transcript.get("test_invocations", [])
        found_results = [i for i in invocations if i.get("result", {}).get("found") == True]
        not_found_results = [i for i in invocations if i.get("result", {}).get("found") == False]
        check("T14-15", f"At least 1 found result (got {len(found_results)})",
              len(found_results) >= 1)
        check("T14-16", f"At least 1 not-found result (got {len(not_found_results)})",
              len(not_found_results) >= 1)
    else:
        for tid in range(11, 17):
            check(f"T14-{tid}", "Skipped (integration failed)", False)

    # -----------------------------------------------------------------------
    # Section 5: Transcript File
    # -----------------------------------------------------------------------
    print("\n--- Section 5: Transcript File ---")
    transcript_path = os.path.join(ROOT_DIR, "transcripts", "task14_mcp.json")
    check("T14-17", "Transcript file exists", os.path.isfile(transcript_path))

    if os.path.isfile(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        check("T14-18", "Saved transcript has expected structure",
              "tools_discovered" in saved and "test_invocations" in saved)
    else:
        check("T14-18", "Transcript file missing", False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    total = PASS + FAIL
    print(f"TASK 14 VALIDATION RESULTS: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    if FAIL == 0:
        print("STATUS: ALL TESTS PASSED")
    else:
        print("STATUS: SOME TESTS FAILED")
    print("=" * 75)


if __name__ == "__main__":
    main()
