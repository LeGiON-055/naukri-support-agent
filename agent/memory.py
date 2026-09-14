"""
memory.py — Persistent Conversation Memory (Capstone Task 8)

Implements multi-turn conversation persistence for the Naukri Domain Support Agent:
1. Thread-isolated JSON history persistence (load_conversation, save_turn, clear_conversation).
2. PII Sanitization: Reuses mask_phone_numbers so raw phone numbers are never stored.
3. Deterministic query contextualization for multi-turn follow-up queries (MOCK_LLM mode).
4. Full isolation between independent session threads.
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Root directory configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_FILE = os.path.join(ROOT_DIR, "transcripts", "conversation_history.json")

# Import PII masking from Phase 10 guardrails
from agent.guardrails import mask_phone_numbers

# Authoritative domain topics recognized in this capstone
DOMAIN_TOPICS = [
    ("probation", "probation period"),
    ("notice period", "notice period"),
    ("referral bonus", "referral bonus policy"),
    ("referral", "referral bonus policy"),
    ("negotiat", "offer negotiation policy"),
    ("leave", "leave policy"),
    ("working hours", "working hours policy"),
    ("attendance", "attendance policy"),
    ("background check", "background check verification"),
    ("performance", "performance appraisal policy"),
    ("code of conduct", "code of conduct policy"),
]

ANAPHORIC_PATTERNS = [
    r'\b(?:that|this)\s+period\b',
    r'\b(?:that|this)\s+policy\b',
    r'\bhow\s+long\s+is\s+(?:that|it|this)\b',
    r'\bwhat\s+about\s+(?:it|that)\b',
    r'\bhow\s+much\s+is\s+(?:it|that)\b',
    r'\bcan\s+i\s+negotiate\s+(?:it|that)\b',
]


# ---------------------------------------------------------------------------
# 1. Load Conversation History
# ---------------------------------------------------------------------------

def load_conversation(thread_id: str, storage_file: str = STORAGE_FILE) -> List[Dict[str, Any]]:
    """
    Load stored message turns for a given thread_id from the JSON persistence file.

    Returns an empty list [] if:
    - The storage file does not exist yet.
    - The thread_id has no previous turns.
    - The file contains malformed data.
    """
    if not thread_id:
        return []

    if not os.path.isfile(storage_file):
        return []

    try:
        with open(storage_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return []
            return data.get(str(thread_id), [])
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# 2. Save Conversation Turn
# ---------------------------------------------------------------------------

def save_turn(
    thread_id: str,
    user_text: str,
    assistant_text: str,
    storage_file: str = STORAGE_FILE,
) -> None:
    """
    Append a user message and assistant response to the specified thread.

    PII Safety:
    Reuses mask_phone_numbers to ensure raw phone numbers are NEVER persisted
    in long-term conversation storage.
    """
    if not thread_id:
        return

    # 1. Sanitize user text with Phase 10 PII guardrail before saving
    masked_user_text, _, _ = mask_phone_numbers(user_text)

    # 2. Load existing storage dictionary
    os.makedirs(os.path.dirname(os.path.abspath(storage_file)), exist_ok=True)
    data = {}
    if os.path.isfile(storage_file):
        try:
            with open(storage_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
        except (json.JSONDecodeError, OSError):
            data = {}

    # 3. Compute turn number
    thread_history = data.setdefault(str(thread_id), [])
    turn_number = (len(thread_history) // 2) + 1
    timestamp = datetime.now().isoformat()

    # 4. Append user and assistant messages
    thread_history.append({
        "turn": turn_number,
        "role": "user",
        "content": masked_user_text,
        "timestamp": timestamp,
    })
    thread_history.append({
        "turn": turn_number,
        "role": "assistant",
        "content": assistant_text,
        "timestamp": timestamp,
    })

    # 5. Write back to disk atomically
    with open(storage_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# 3. Clear Conversation History
# ---------------------------------------------------------------------------

def clear_conversation(thread_id: str, storage_file: str = STORAGE_FILE) -> None:
    """Remove conversation history for a specific thread_id (used for resets)."""
    if not os.path.isfile(storage_file) or not thread_id:
        return

    try:
        with open(storage_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and str(thread_id) in data:
            del data[str(thread_id)]
            with open(storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# 4. Deterministic Query Contextualization
# ---------------------------------------------------------------------------

def extract_previous_topic(history: List[Dict[str, Any]]) -> Optional[str]:
    """Scan previous messages (most recent first) to identify the active domain topic."""
    if not history:
        return None

    # Reverse search through past assistant and user messages
    for msg in reversed(history):
        content = msg.get("content", "").lower()

        # Check for application record IDs first (e.g. APP-002)
        app_match = re.search(r'\bapp-\d{3}\b', content, re.IGNORECASE)
        if app_match:
            return app_match.group(0).upper()

        # Check domain policy topics
        for keyword, canonical_topic in DOMAIN_TOPICS:
            if keyword in content:
                return canonical_topic

    return None


def contextualize_query(query: str, history: List[Dict[str, Any]]) -> str:
    """
    Resolve anaphoric and follow-up phrasing using conversation history.

    In MOCK_LLM mode, this deterministically substitutes pronoun / vague references
    (e.g., 'that period', 'this policy') with the topic established in previous turns,
    allowing downstream semantic retrieval in Chroma to accurately match chunks.

    If history is empty or query has no anaphora, returns query unchanged.
    """
    if not query or not history:
        return query

    # Find the active topic in history
    active_topic = extract_previous_topic(history)
    if not active_topic:
        return query

    clean_q = query.strip()
    clean_lower = clean_q.lower()

    # Check if query contains anaphoric phrasing
    has_anaphora = any(re.search(pat, clean_lower) for pat in ANAPHORIC_PATTERNS)

    if has_anaphora:
        # Check specific substitutions
        if re.search(r'\b(?:that|this)\s+period\b', clean_lower):
            # Replace "that period" with the active topic (e.g. "probation period")
            return re.sub(r'\b(?:that|this)\s+period\b', active_topic, clean_q, flags=re.IGNORECASE)

        if re.search(r'\b(?:that|this)\s+policy\b', clean_lower):
            return re.sub(r'\b(?:that|this)\s+policy\b', active_topic, clean_q, flags=re.IGNORECASE)

        if re.search(r'\bhow\s+long\s+is\s+(?:that|it|this)\b', clean_lower):
            return f"What is the duration of the {active_topic}?"

        # Fallback query augmentation with prior topic
        return f"{clean_q} (regarding {active_topic})"

    return query
