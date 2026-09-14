"""
demo_task15.py — SQLite Checkpointing Demo (Capstone Task 15)

Demonstrates:
1. Building LangGraph workflow with SqliteSaver checkpointer
2. Running queries on different threads (state checkpointed to SQLite)
3. Verifying checkpoint persistence in SQLite database
4. Resuming state from checkpoint (simulating crash recovery)
5. Proving state continuity across save/resume cycle
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["MOCK_LLM"] = "true"

from resilience.checkpointing import run_checkpointing_demo


def main():
    print("=" * 75)
    print("CAPSTONE TASK 15 - SQLITE CHECKPOINTING DEMO")
    print("=" * 75)
    print()

    transcript = run_checkpointing_demo()

    print()
    print("=" * 75)
    print("DEMO COMPLETE")
    print("=" * 75)
    print(f"  State continuity verified: {transcript['state_continuity_verified']}")
    print(f"  Transcript: transcripts/task15_checkpointing.json")


if __name__ == "__main__":
    main()
