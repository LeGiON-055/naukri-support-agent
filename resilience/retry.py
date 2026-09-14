"""
retry.py — Exponential Backoff Retry Demo (Capstone Task 16)

Demonstrates retrying a simulated transient failure with exponential
backoff and jitter. Shows the agent can handle temporary failures
gracefully without crashing.

Demonstrates:
1. A function that fails N times then succeeds
2. Retry decorator with exponential backoff + jitter
3. Successful recovery after transient failures
4. Failure case when max retries exceeded
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from functools import wraps

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "task16_resilience.json")


# ---------------------------------------------------------------------------
# Retry Decorator with Exponential Backoff + Jitter
# ---------------------------------------------------------------------------

def retry_with_backoff(max_retries=3, base_delay=0.1, max_delay=2.0, jitter=True):
    """
    Decorator that retries a function on exception with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts.
        base_delay (float): Initial delay in seconds.
        max_delay (float): Maximum delay cap in seconds.
        jitter (bool): Whether to add random jitter to delay.

    Returns:
        Decorated function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt_log = []
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    start = time.time()
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    attempt_log.append({
                        "attempt": attempt + 1,
                        "status": "SUCCESS",
                        "elapsed_ms": round(elapsed * 1000, 2),
                    })
                    return result, attempt_log
                except Exception as e:
                    elapsed = time.time() - start
                    last_exception = e

                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        if jitter:
                            delay = delay * (0.5 + random.random() * 0.5)
                        attempt_log.append({
                            "attempt": attempt + 1,
                            "status": "FAILED",
                            "error": str(e),
                            "elapsed_ms": round(elapsed * 1000, 2),
                            "retry_delay_ms": round(delay * 1000, 2),
                        })
                        time.sleep(delay)
                    else:
                        attempt_log.append({
                            "attempt": attempt + 1,
                            "status": "FAILED (final)",
                            "error": str(e),
                            "elapsed_ms": round(elapsed * 1000, 2),
                        })

            raise last_exception

        wrapper._attempt_log = []
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Simulated Transient Failure Function
# ---------------------------------------------------------------------------

class TransientError(Exception):
    """Simulated transient error (e.g., network timeout, rate limit)."""
    pass


def make_flaky_function(fail_count=2):
    """
    Create a function that fails `fail_count` times then succeeds.

    Args:
        fail_count (int): Number of times to fail before succeeding.

    Returns:
        Callable that simulates transient failures.
    """
    call_counter = {"count": 0}

    @retry_with_backoff(max_retries=5, base_delay=0.05, max_delay=0.5, jitter=True)
    def flaky_lookup(record_id):
        call_counter["count"] += 1
        if call_counter["count"] <= fail_count:
            raise TransientError(
                f"Simulated transient failure (attempt {call_counter['count']}/{fail_count + 1})"
            )
        # Success on attempt fail_count + 1
        from agent.tools import check_job_application_status
        return check_job_application_status(record_id)

    return flaky_lookup, call_counter


def make_always_failing_function():
    """Create a function that always fails (exceeds max retries)."""

    @retry_with_backoff(max_retries=3, base_delay=0.05, max_delay=0.5, jitter=True)
    def always_fail(record_id):
        raise TransientError("Permanent simulated failure")

    return always_fail


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------

def run_retry_demo():
    """Demonstrate retry with exponential backoff."""
    print("=" * 75)
    print("TASK 16 - RETRY WITH EXPONENTIAL BACKOFF DEMO")
    print("=" * 75)

    results = []

    # Scenario 1: Fails 2 times, succeeds on 3rd attempt
    print("\n--- Scenario 1: Transient Failure (recovers after 2 failures) ---")
    flaky_fn, counter = make_flaky_function(fail_count=2)
    start = time.time()
    try:
        result, attempt_log = flaky_fn("APP-001")
        total_time = time.time() - start
        print(f"  Result: SUCCESS after {len(attempt_log)} attempts")
        print(f"  Record found: {result.get('found')}")
        print(f"  Total time: {total_time * 1000:.1f}ms")
        for log in attempt_log:
            print(f"    Attempt {log['attempt']}: {log['status']} ({log['elapsed_ms']}ms)")

        results.append({
            "scenario": "transient_failure_recovery",
            "description": "Fails 2 times, succeeds on 3rd",
            "max_retries": 5,
            "success": True,
            "total_attempts": len(attempt_log),
            "total_time_ms": round(total_time * 1000, 2),
            "attempt_log": attempt_log,
            "result_found": result.get("found"),
        })
    except TransientError as e:
        total_time = time.time() - start
        print(f"  Result: FAILED - {e}")
        results.append({
            "scenario": "transient_failure_recovery",
            "success": False,
            "error": str(e),
        })

    # Scenario 2: Fails 4 times, succeeds on 5th (tests deeper backoff)
    print("\n--- Scenario 2: Deeper Backoff (recovers after 4 failures) ---")
    flaky_fn2, counter2 = make_flaky_function(fail_count=4)
    start = time.time()
    try:
        result2, attempt_log2 = flaky_fn2("APP-005")
        total_time = time.time() - start
        print(f"  Result: SUCCESS after {len(attempt_log2)} attempts")
        print(f"  Record found: {result2.get('found')}")
        print(f"  Total time: {total_time * 1000:.1f}ms")
        for log in attempt_log2:
            print(f"    Attempt {log['attempt']}: {log['status']} ({log['elapsed_ms']}ms)")

        results.append({
            "scenario": "deep_backoff_recovery",
            "description": "Fails 4 times, succeeds on 5th",
            "max_retries": 5,
            "success": True,
            "total_attempts": len(attempt_log2),
            "total_time_ms": round(total_time * 1000, 2),
            "attempt_log": attempt_log2,
            "result_found": result2.get("found"),
        })
    except TransientError as e:
        total_time = time.time() - start
        print(f"  Result: FAILED - {e}")
        results.append({
            "scenario": "deep_backoff_recovery",
            "success": False,
            "error": str(e),
        })

    # Scenario 3: Always fails -> exceeds max retries
    print("\n--- Scenario 3: Permanent Failure (exceeds max retries) ---")
    always_fail = make_always_failing_function()
    start = time.time()
    attempt_log3 = []
    try:
        result3, attempt_log3 = always_fail("APP-001")
        total_time = time.time() - start
        print(f"  Result: Unexpected success")
        results.append({
            "scenario": "permanent_failure",
            "success": True,
            "note": "Unexpected success",
        })
    except TransientError as e:
        total_time = time.time() - start
        print(f"  Result: CORRECTLY EXHAUSTED retries")
        print(f"  Final error: {e}")
        print(f"  Total time: {total_time * 1000:.1f}ms")

        results.append({
            "scenario": "permanent_failure",
            "description": "Always fails, exhausts all retries",
            "max_retries": 3,
            "success": False,
            "correctly_exhausted": True,
            "final_error": str(e),
            "total_time_ms": round(total_time * 1000, 2),
        })

    return results


if __name__ == "__main__":
    run_retry_demo()
