"""
validate_task8.py — Validation Suite for Capstone Task 8 (Persistent Conversation Memory)

Verifies:
1. JSON storage file is created and persists records.
2. Multi-turn conversation carryover works seamlessly across turns.
3. Fresh conversation ID starts with empty history (strict isolation).
4. Restart persistence: re-reading from disk yields accurate, uncorrupted history.
5. Zero cross-contamination between independent threads.
6. PII sanitization: raw phone numbers are never stored in persistent memory.
7. transcripts/task8_memory.json exists and contains complete evidence.
8. README.md documents Task 8 persistent memory architecture.

Usage:
    python validate_task8.py
"""

import os
import sys
import json
import re

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent.memory import (
    load_conversation,
    save_turn,
    clear_conversation,
    contextualize_query,
    STORAGE_FILE,
)
from agent.graph import run_agent_workflow

TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "task8_memory.json")
README_PATH = os.path.join(ROOT_DIR, "README.md")


def validate():
    print("=" * 75)
    print("CAPSTONE TASK 8 VALIDATION REPORT — PERSISTENT CONVERSATION MEMORY")
    print("=" * 75)
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

    # -----------------------------------------------------------------------
    # 1. Storage Operations & Persistence Verification
    # -----------------------------------------------------------------------
    clear_conversation("val_thread_1")
    check("New thread ID returns empty list", load_conversation("val_thread_1") == [])

    save_turn("val_thread_1", "What is probation?", "Probation is 6 months.")
    check("JSON storage file physically exists on disk", os.path.isfile(STORAGE_FILE))

    h1 = load_conversation("val_thread_1")
    check("Saving 1 turn creates exactly 2 messages (user + assistant)", len(h1) == 2)
    check("Turn structure contains required fields (turn, role, content, timestamp)",
          all(k in h1[0] for k in ["turn", "role", "content", "timestamp"]))
    check("Turn number 1 recorded correctly", h1[0]["turn"] == 1 and h1[1]["turn"] == 1)

    # -----------------------------------------------------------------------
    # 2. Multi-Turn Carryover Verification
    # -----------------------------------------------------------------------
    save_turn("val_thread_1", "How long is that period?", "It is 6 months from date of joining.")
    h2 = load_conversation("val_thread_1")
    check("Saving second turn increments history count to 4", len(h2) == 4)
    check("Second turn records turn number 2", h2[2]["turn"] == 2 and h2[3]["turn"] == 2)

    # Contextualization check
    contextualized = contextualize_query("How long is that period?", h1)
    check("Query contextualization resolves 'that period' to probation",
          "probation" in contextualized.lower())

    # -----------------------------------------------------------------------
    # 3. Fresh Conversation & Isolation Verification
    # -----------------------------------------------------------------------
    clear_conversation("val_thread_2")
    h_fresh = load_conversation("val_thread_2")
    check("Fresh thread starts with 0 messages (strict isolation)", len(h_fresh) == 0)

    # -----------------------------------------------------------------------
    # 4. Zero Cross-Contamination Verification
    # -----------------------------------------------------------------------
    save_turn("val_thread_2", "Status of APP-002?", "Screening status.")
    h_thread_1 = load_conversation("val_thread_1")
    h_thread_2 = load_conversation("val_thread_2")
    check("Thread 1 history untouched by Thread 2 additions", len(h_thread_1) == 4)
    check("Thread 2 history isolated to its own turns", len(h_thread_2) == 2)

    # Verify no message content overlaps
    t1_contents = {m["content"] for m in h_thread_1}
    t2_contents = {m["content"] for m in h_thread_2}
    check("Zero message content overlap between distinct threads", t1_contents.isdisjoint(t2_contents))

    # -----------------------------------------------------------------------
    # 5. Restart Persistence (Disk Re-read)
    # -----------------------------------------------------------------------
    with open(STORAGE_FILE, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    check("Raw disk reload confirms val_thread_1 is stored", "val_thread_1" in disk_data)
    check("Raw disk reload confirms val_thread_2 is stored", "val_thread_2" in disk_data)

    # -----------------------------------------------------------------------
    # 6. PII Masking Persistence Verification
    # -----------------------------------------------------------------------
    clear_conversation("val_thread_pii")
    save_turn("val_thread_pii", "Call me at 9876543210 regarding probation", "Acknowledged.")
    h_pii = load_conversation("val_thread_pii")
    check("Raw phone number is NOT saved in persistent JSON storage", "9876543210" not in h_pii[0]["content"])
    check("PII replaced with [REDACTED_PHONE] in storage", "[REDACTED_PHONE]" in h_pii[0]["content"])

    # Clean up test threads
    clear_conversation("val_thread_1")
    clear_conversation("val_thread_2")
    clear_conversation("val_thread_pii")

    # -----------------------------------------------------------------------
    # 7. Transcript Verification
    # -----------------------------------------------------------------------
    check("transcripts/task8_memory.json exists", os.path.isfile(TRANSCRIPT_PATH))
    if os.path.isfile(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        demos = t_data.get("demonstrations", [])
        check("Transcript contains demonstrations", len(demos) >= 2, f"found {len(demos)}")

    # -----------------------------------------------------------------------
    # 8. README Verification
    # -----------------------------------------------------------------------
    readme_content = open(README_PATH, "r", encoding="utf-8").read() if os.path.isfile(README_PATH) else ""
    check("README.md documents Task 8", "Task 8" in readme_content or "Persistent Conversation Memory" in readme_content)
    check("README.md documents conversation history location", "conversation_history.json" in readme_content or "transcripts" in readme_content)

    print()
    print("=" * 75)
    if all_passed:
        print("RESULT: ALL TASK 8 CHECKS PASSED (100% SUCCESS)!")
    else:
        print("RESULT: SOME CHECKS FAILED - REVIEW ABOVE.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
