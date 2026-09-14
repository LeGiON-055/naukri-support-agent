"""
demo_task16.py — Resilience Demo (Capstone Task 16)

Demonstrates:
1. Retry with exponential backoff and jitter (transient failure recovery)
2. Max retry exhaustion (permanent failure handling)
3. Per-node timeout enforcement
4. Global workflow timeout budget
5. Saves transcript to transcripts/task16_resilience.json
"""

import os
import sys
import json
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["MOCK_LLM"] = "true"

from resilience.retry import run_retry_demo
from resilience.timeout import run_timeout_demo


TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "task16_resilience.json")


def main():
    print("=" * 75)
    print("CAPSTONE TASK 16 - RESILIENCE DEMO")
    print("=" * 75)
    print()

    # Part 1: Retry with backoff
    retry_results = run_retry_demo()

    print()

    # Part 2: Timeout handling
    timeout_results = run_timeout_demo()

    # Build combined transcript
    transcript = {
        "task": "Task 16 - Resilience (Retry + Timeout)",
        "timestamp": datetime.now().isoformat(),
        "mock_mode": True,
        "retry_results": retry_results,
        "timeout_results": timeout_results,
        "retry_scenarios_count": len(retry_results),
        "timeout_scenarios_count": len(timeout_results),
    }

    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 75)
    print("DEMO COMPLETE")
    print("=" * 75)
    print(f"  Retry scenarios: {len(retry_results)}")
    print(f"  Timeout scenarios: {len(timeout_results)}")
    print(f"  Transcript: transcripts/task16_resilience.json")


if __name__ == "__main__":
    main()
