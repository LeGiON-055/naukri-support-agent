"""
timeout.py — Per-Node and Global Timeout Demo (Capstone Task 16)

Demonstrates:
1. A node that exceeds its per-node timeout and fails cleanly
2. A workflow that completes within its timeout budget
3. Graceful timeout handling without crashing the agent

Uses threading-based timeouts (no async) for MOCK_LLM compatibility.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# Timeout Decorator (Thread-Based)
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    """Raised when a function exceeds its allotted timeout."""
    pass


def with_timeout(timeout_seconds):
    """
    Decorator that enforces a timeout on a function.

    Uses a separate thread to run the function. If the function
    does not complete within `timeout_seconds`, raises TimeoutError.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                # Thread is still running -> timeout
                raise TimeoutError(
                    f"Function '{func.__name__}' exceeded timeout of {timeout_seconds}s"
                )
            if error[0]:
                raise error[0]
            return result[0]

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Simulated Operations
# ---------------------------------------------------------------------------

def fast_operation(record_id):
    """A fast operation that completes well within timeout."""
    from agent.tools import check_job_application_status
    time.sleep(0.05)  # 50ms - well within any timeout
    return check_job_application_status(record_id)


def slow_operation(record_id):
    """A slow operation that simulates a long-running computation."""
    time.sleep(3.0)  # 3 seconds - will exceed most timeouts
    from agent.tools import check_job_application_status
    return check_job_application_status(record_id)


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------

def run_timeout_demo():
    """Demonstrate per-node and global timeout handling."""
    print("=" * 75)
    print("TASK 16 - TIMEOUT HANDLING DEMO")
    print("=" * 75)

    results = []

    # Scenario 1: Fast operation completes within timeout
    print("\n--- Scenario 1: Operation Within Timeout (1.0s budget) ---")
    timed_fast = with_timeout(1.0)(fast_operation)
    start = time.time()
    try:
        result = timed_fast("APP-001")
        elapsed = time.time() - start
        print(f"  Result: SUCCESS in {elapsed * 1000:.1f}ms")
        print(f"  Record found: {result.get('found')}")
        print(f"  Status: {result.get('status')}")
        results.append({
            "scenario": "within_timeout",
            "description": "Fast operation completes within 1.0s budget",
            "timeout_seconds": 1.0,
            "success": True,
            "elapsed_ms": round(elapsed * 1000, 2),
            "record_found": result.get("found"),
        })
    except TimeoutError as e:
        elapsed = time.time() - start
        print(f"  Result: UNEXPECTED TIMEOUT - {e}")
        results.append({
            "scenario": "within_timeout",
            "success": False,
            "error": str(e),
        })

    # Scenario 2: Slow operation exceeds per-node timeout
    print("\n--- Scenario 2: Operation Exceeds Timeout (0.5s budget) ---")
    timed_slow = with_timeout(0.5)(slow_operation)
    start = time.time()
    try:
        result = timed_slow("APP-001")
        elapsed = time.time() - start
        print(f"  Result: UNEXPECTED SUCCESS")
        results.append({
            "scenario": "exceeds_timeout",
            "success": True,
            "note": "Unexpected success",
        })
    except TimeoutError as e:
        elapsed = time.time() - start
        print(f"  Result: CORRECTLY TIMED OUT in {elapsed * 1000:.1f}ms")
        print(f"  Error: {e}")
        print(f"  Agent remains operational (no crash)")
        results.append({
            "scenario": "exceeds_timeout",
            "description": "Slow operation exceeds 0.5s budget",
            "timeout_seconds": 0.5,
            "success": False,
            "correctly_timed_out": True,
            "elapsed_ms": round(elapsed * 1000, 2),
            "error": str(e),
        })

    # Scenario 3: Global workflow timeout (multiple operations)
    print("\n--- Scenario 3: Global Workflow Timeout Budget (2.0s) ---")
    global_timeout = 2.0
    start = time.time()
    operations = [
        ("APP-001", "fast"),
        ("APP-005", "fast"),
        ("APP-010", "fast"),
    ]
    completed = []
    timed_out = False

    for record_id, speed in operations:
        elapsed_so_far = time.time() - start
        remaining = global_timeout - elapsed_so_far

        if remaining <= 0:
            timed_out = True
            print(f"  Global timeout reached after {elapsed_so_far * 1000:.1f}ms")
            break

        try:
            timed_op = with_timeout(min(remaining, 1.0))(fast_operation)
            result = timed_op(record_id)
            op_time = time.time() - start
            completed.append({
                "record_id": record_id,
                "status": "SUCCESS",
                "cumulative_ms": round(op_time * 1000, 2),
            })
            print(f"  {record_id}: SUCCESS ({op_time * 1000:.1f}ms cumulative)")
        except TimeoutError as e:
            timed_out = True
            print(f"  {record_id}: TIMED OUT")
            break

    total_elapsed = time.time() - start
    print(f"  Total: {len(completed)}/{len(operations)} completed in {total_elapsed * 1000:.1f}ms")

    results.append({
        "scenario": "global_workflow_timeout",
        "description": f"3 operations under {global_timeout}s global budget",
        "global_timeout_seconds": global_timeout,
        "operations_completed": len(completed),
        "operations_total": len(operations),
        "timed_out": timed_out,
        "total_elapsed_ms": round(total_elapsed * 1000, 2),
        "completed_operations": completed,
    })

    return results


if __name__ == "__main__":
    run_timeout_demo()
