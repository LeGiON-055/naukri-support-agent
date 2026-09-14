"""
validate_task15.py — SQLite Checkpointing Validator (Capstone Task 15)

Validates that Task 15 implementation meets all capstone requirements:
1. Graph compiled with SqliteSaver checkpointer
2. Queries run with thread_id -> state saved to SQLite
3. Checkpoint data persisted in SQLite database
4. State can be resumed from checkpoint
5. State continuity verified (node_history preserved)
6. Transcript saved
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


def check(test_id, description, condition):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {test_id}: {description}")
    return condition


def main():
    global PASS, FAIL

    print("=" * 75)
    print("TASK 15 VALIDATION - SQLite Checkpointing")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # Section 1: Module Import
    # -----------------------------------------------------------------------
    print("\n--- Section 1: Module Import ---")
    try:
        from resilience.checkpointing import (
            build_checkpointed_graph,
            run_checkpointing_demo,
            CHECKPOINT_DB_PATH,
        )
        check("T15-01", "checkpointing module imports successfully", True)
    except Exception as e:
        check("T15-01", f"Import failed: {e}", False)
        print(f"\nResults: {PASS} PASS / {FAIL} FAIL")
        return

    # -----------------------------------------------------------------------
    # Section 2: Build Checkpointed Graph
    # -----------------------------------------------------------------------
    print("\n--- Section 2: Build Checkpointed Graph ---")
    import sqlite3
    # Clean up
    test_db = os.path.join(ROOT_DIR, "resilience", "test_checkpoints.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    try:
        app, conn, checkpointer = build_checkpointed_graph(test_db)
        check("T15-02", "build_checkpointed_graph returns compiled graph", app is not None)
        check("T15-03", "Checkpointer is SqliteSaver",
              "SqliteSaver" in type(checkpointer).__name__)
    except Exception as e:
        check("T15-02", f"Build failed: {e}", False)
        check("T15-03", f"Build failed: {e}", False)
        conn = None

    # -----------------------------------------------------------------------
    # Section 3: Run with Checkpointing
    # -----------------------------------------------------------------------
    print("\n--- Section 3: Run with Checkpointing ---")
    if app:
        thread_config = {"configurable": {"thread_id": "test-thread-1"}}
        initial_state = {
            "raw_query": "What is the probation period?",
            "masked_query": "What is the probation period?",
            "pii_detected": False,
            "is_injection": False,
            "matched_pattern": None,
            "intent": None,
            "record_id": None,
            "tool_invoked": False,
            "rag_result": None,
            "status_result": None,
            "response": None,
            "is_valid_schema": False,
            "node_history": [],
            "thread_id": "test-thread-1",
            "history": [],
        }
        try:
            final_state = app.invoke(initial_state, config=thread_config)
            check("T15-04", "Graph executes with thread_id", True)
            check("T15-05", "Final state has node_history",
                  len(final_state.get("node_history", [])) > 0)
            check("T15-06", "Final state has intent",
                  final_state.get("intent") is not None)
            check("T15-07", "Final state has response",
                  final_state.get("response") is not None)
        except Exception as e:
            check("T15-04", f"Execution failed: {e}", False)
            check("T15-05", "Skipped", False)
            check("T15-06", "Skipped", False)
            check("T15-07", "Skipped", False)
            final_state = None

        # -----------------------------------------------------------------------
        # Section 4: Verify Checkpoint in SQLite
        # -----------------------------------------------------------------------
        print("\n--- Section 4: SQLite Verification ---")
        check("T15-08", "SQLite DB file exists", os.path.isfile(test_db))
        db_size = os.path.getsize(test_db) if os.path.isfile(test_db) else 0
        check("T15-09", f"SQLite DB has data ({db_size} bytes)", db_size > 0)

        verify_conn = sqlite3.connect(test_db)
        cursor = verify_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        check("T15-10", f"SQLite has checkpoint tables ({len(tables)} tables)", len(tables) > 0)
        verify_conn.close()

        # -----------------------------------------------------------------------
        # Section 5: State Resume
        # -----------------------------------------------------------------------
        print("\n--- Section 5: State Resume ---")
        if final_state:
            try:
                saved_state = app.get_state(thread_config)
                resumed = saved_state.values if saved_state else {}
                check("T15-11", "get_state returns saved values",
                      len(resumed) > 0)
                check("T15-12", "Resumed node_history matches original",
                      resumed.get("node_history") == final_state.get("node_history"))
                check("T15-13", "Resumed intent matches original",
                      resumed.get("intent") == final_state.get("intent"))
            except Exception as e:
                check("T15-11", f"get_state failed: {e}", False)
                check("T15-12", "Skipped", False)
                check("T15-13", "Skipped", False)
        else:
            check("T15-11", "Skipped (no state)", False)
            check("T15-12", "Skipped", False)
            check("T15-13", "Skipped", False)

        if conn:
            conn.close()
    else:
        for i in range(4, 14):
            check(f"T15-{i:02d}", "Skipped (graph build failed)", False)

    # Clean up test DB
    if os.path.exists(test_db):
        os.remove(test_db)

    # -----------------------------------------------------------------------
    # Section 6: Full Demo Run
    # -----------------------------------------------------------------------
    print("\n--- Section 6: Full Demo Run ---")
    try:
        transcript = run_checkpointing_demo()
        check("T15-14", "run_checkpointing_demo executes successfully", True)
        check("T15-15", "Transcript has state_continuity_verified",
              "state_continuity_verified" in transcript)
        check("T15-16", "State continuity verified is True",
              transcript.get("state_continuity_verified") == True)
    except Exception as e:
        check("T15-14", f"Demo failed: {e}", False)
        check("T15-15", "Skipped", False)
        check("T15-16", "Skipped", False)

    # -----------------------------------------------------------------------
    # Section 7: Transcript File
    # -----------------------------------------------------------------------
    print("\n--- Section 7: Transcript File ---")
    transcript_path = os.path.join(ROOT_DIR, "transcripts", "task15_checkpointing.json")
    check("T15-17", "Transcript file exists", os.path.isfile(transcript_path))

    if os.path.isfile(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        check("T15-18", "Saved transcript has expected structure",
              "demo_steps" in saved and "state_continuity_verified" in saved)
    else:
        check("T15-18", "Transcript file missing", False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    total = PASS + FAIL
    print(f"TASK 15 VALIDATION RESULTS: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    if FAIL == 0:
        print("STATUS: ALL TESTS PASSED")
    else:
        print("STATUS: SOME TESTS FAILED")
    print("=" * 75)


if __name__ == "__main__":
    main()
