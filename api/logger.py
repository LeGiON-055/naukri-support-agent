"""
logger.py — Structured JSONL Request Logger (Capstone Task 12)

Provides:
1. Unique trace ID generation (UUID-based).
2. PII phone-number sanitization via agent.guardrails.mask_phone_numbers.
3. Atomic single-line JSONL logging to transcripts/requests.jsonl.
4. Log reader utility for verification and analytics.
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent.guardrails import mask_phone_numbers

LOG_FILE = os.path.join(ROOT_DIR, "transcripts", "requests.jsonl")


def generate_trace_id() -> str:
    """Generate a unique request trace ID with 'tr-' prefix."""
    return f"tr-{uuid.uuid4().hex[:12]}"


def log_request_event(
    trace_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_text: Optional[str] = None,
    thread_id: Optional[str] = None,
    intent: Optional[str] = None,
    error: Optional[str] = None,
    log_file: str = LOG_FILE,
) -> Dict[str, Any]:
    """
    Construct a structured log entry, sanitize PII, and append one JSONL line.

    Args:
        trace_id: Unique request identifier.
        method: HTTP verb ('GET', 'POST', etc.).
        path: HTTP endpoint path (e.g. '/ask').
        status_code: HTTP response status code.
        duration_ms: Request processing duration in milliseconds.
        request_text: Raw query or input text (will be sanitized for PII).
        thread_id: Conversation session identifier if available.
        intent: Classified domain intent if available.
        error: Error message/detail if request failed, else None.
        log_file: Destination JSONL file path.

    Returns:
        Dict[str, Any]: The structured log entry dictionary.
    """
    # 1. Sanitize request text using Phase 10 single source of truth
    sanitized_text = None
    if request_text is not None:
        sanitized_text, _, _ = mask_phone_numbers(request_text)

    # 2. Build structured dictionary
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "method": method.upper(),
        "path": path,
        "status_code": int(status_code),
        "duration_ms": round(float(duration_ms), 2),
        "thread_id": thread_id,
        "request_text": sanitized_text,
        "intent": intent,
        "error": error,
    }

    # 3. Ensure parent directory exists and append line
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def read_logs(log_file: str = LOG_FILE) -> List[Dict[str, Any]]:
    """
    Read all valid JSON lines from the specified JSONL log file.
    Returns empty list if file does not exist.
    """
    if not os.path.exists(log_file):
        return []

    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                entries.append(json.loads(line_str))
            except json.JSONDecodeError as err:
                print(f"Warning: line {line_num} in {log_file} is not valid JSON: {err}")
    return entries


def clear_logs(log_file: str = LOG_FILE) -> None:
    """Reset or truncate the JSONL log file."""
    if os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            f.truncate(0)
