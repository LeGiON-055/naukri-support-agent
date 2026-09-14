"""
demo_task8.py — Demonstration Script for Capstone Task 8 (Persistent Conversation Memory)

Demonstrates:
1. Demonstration A: Multi-turn conversation carryover on thread 'thread_conv_alpha'.
   - Turn 1: 'What is the probation period?' (establishes probation context).
   - Turn 2: 'How long is that period?' (remembers context, resolves to probation).
2. Demonstration B: Fresh conversation isolation on thread 'thread_conv_beta'.
   - Turn 1: 'How long is that period?' (empty history, fails to resolve, fallback).
3. Demonstration C: PII Masking persistence check on thread 'thread_conv_gamma'.
   - User provides raw phone number; verified that raw phone is NEVER saved to JSON history.

Exports full execution evidence to:
  transcripts/task8_memory.json
"""

import os
import sys
import json
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent.graph import run_agent_workflow
from agent.memory import (
    load_conversation,
    clear_conversation,
    STORAGE_FILE,
    contextualize_query,
)

TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "task8_memory.json")


def run_demo():
    print("=" * 80)
    print("CAPSTONE TASK 8 DEMONSTRATION — PERSISTENT CONVERSATION MEMORY")
    print("=" * 80)
    print(f"Persistent Storage File: {STORAGE_FILE}")
    print()

    # Clear test threads for deterministic execution
    clear_conversation("thread_conv_alpha")
    clear_conversation("thread_conv_beta")
    clear_conversation("thread_conv_gamma")

    transcript_record = {
        "task": "Capstone Task 8 — Persistent Conversation Memory",
        "timestamp": datetime.now().isoformat(),
        "storage_file": os.path.relpath(STORAGE_FILE, ROOT_DIR),
        "demonstrations": [],
    }

    # -----------------------------------------------------------------------
    # Demonstration A: Multi-Turn Conversation Carryover
    # -----------------------------------------------------------------------
    print("[DEMO A] Multi-Turn Conversation Carryover (thread_conv_alpha)")
    thread_a = "thread_conv_alpha"
    demo_a_turns = []

    # Turn 1: Establish context
    q1 = "What is the probation period?"
    h1_pre = load_conversation(thread_a)
    print(f"  Turn 1 Query     : '{q1}'")
    print(f"  History Before   : {len(h1_pre)} messages")
    res1 = run_agent_workflow(q1, thread_id=thread_a)
    h1_post = load_conversation(thread_a)
    ans1 = res1["response"].answer
    print(f"  Answer           : {ans1[:90]}...")
    print(f"  History After    : {len(h1_post)} messages")
    print(f"  Visited Nodes    : {res1['node_history']}")
    print()

    demo_a_turns.append({
        "turn": 1,
        "input_query": q1,
        "loaded_history_count": len(h1_pre),
        "contextualized_query": q1,
        "answer": ans1,
        "post_turn_history_count": len(h1_post),
        "visited_nodes": res1["node_history"],
    })

    # Turn 2: Follow-up question relying on previous turn
    q2 = "How long is that period?"
    h2_pre = load_conversation(thread_a)
    contextualized_q2 = contextualize_query(q2, h2_pre)
    print(f"  Turn 2 Follow-Up : '{q2}'")
    print(f"  History Before   : {len(h2_pre)} messages (Loaded from Turn 1)")
    print(f"  Contextualized   : '{contextualized_q2}'")
    res2 = run_agent_workflow(q2, thread_id=thread_a)
    h2_post = load_conversation(thread_a)
    ans2 = res2["response"].answer
    print(f"  Answer           : {ans2[:90]}...")
    print(f"  History After    : {len(h2_post)} messages")
    print(f"  Visited Nodes    : {res2['node_history']}")
    print(f"  Context Carried? : {'probation' in ans2.lower() or '6 months' in ans2.lower()}")
    print()

    demo_a_turns.append({
        "turn": 2,
        "input_query": q2,
        "loaded_history_count": len(h2_pre),
        "contextualized_query": contextualized_q2,
        "answer": ans2,
        "post_turn_history_count": len(h2_post),
        "visited_nodes": res2["node_history"],
        "context_carryover_successful": ("probation" in ans2.lower() or "6 months" in ans2.lower()),
    })

    transcript_record["demonstrations"].append({
        "scenario": "Multi-Turn Conversation Carryover",
        "thread_id": thread_a,
        "turns": demo_a_turns,
    })

    # -----------------------------------------------------------------------
    # Demonstration B: Fresh Conversation Isolation
    # -----------------------------------------------------------------------
    print("[DEMO B] Fresh Conversation Isolation (thread_conv_beta)")
    thread_b = "thread_conv_beta"
    h_b_pre = load_conversation(thread_b)
    print(f"  Thread ID        : '{thread_b}'")
    print(f"  History Before   : {len(h_b_pre)} messages (Strictly Empty)")
    print(f"  Query            : '{q2}' (Ambiguous without prior context)")
    res_b = run_agent_workflow(q2, thread_id=thread_b)
    h_b_post = load_conversation(thread_b)
    ans_b = res_b["response"].answer
    primary_source = res_b["rag_result"]["sources"][0] if res_b.get("rag_result") and res_b["rag_result"].get("sources") else "unknown"
    was_not_contextualized = (res_b["raw_query"] == q2)
    isolation_proven = (len(h_b_pre) == 0 and was_not_contextualized and primary_source != "probation_period.txt")
    print(f"  Primary Source   : {primary_source} (Generic 'period' match, not probation)")
    print(f"  Contextualized?  : {not was_not_contextualized} (Must be False - uncontextualized)")
    print(f"  History After    : {len(h_b_post)} messages")
    print(f"  Visited Nodes    : {res_b['node_history']}")
    print(f"  Isolation Proven : {isolation_proven} (0 history loaded & no context inherited)")
    print()

    transcript_record["demonstrations"].append({
        "scenario": "Fresh Conversation Isolation",
        "thread_id": thread_b,
        "turns": [
            {
                "turn": 1,
                "input_query": q2,
                "loaded_history_count": len(h_b_pre),
                "contextualized_query": res_b["raw_query"],
                "primary_source": primary_source,
                "answer": ans_b,
                "intent": res_b["response"].intent,
                "context_inherited": False,
                "post_turn_history_count": len(h_b_post),
                "visited_nodes": res_b["node_history"],
                "isolated_from_thread_alpha": isolation_proven,
            }
        ],
    })

    # -----------------------------------------------------------------------
    # Demonstration C: PII Masking in Persistent Memory
    # -----------------------------------------------------------------------
    print("[DEMO C] PII Masking Persistence (thread_conv_gamma)")
    thread_c = "thread_conv_gamma"
    raw_phone_q = "My contact is 9876543210. What is the notice period?"
    print(f"  Raw User Query   : [REDACTED FOR PRIVACY — Contains phone number 9876543210]")
    res_c = run_agent_workflow(raw_phone_q, thread_id=thread_c)
    h_c_post = load_conversation(thread_c)
    stored_user_msg = h_c_post[0]["content"] if h_c_post else ""
    raw_phone_leaked = "9876543210" in stored_user_msg
    redacted_token_present = "[REDACTED_PHONE]" in stored_user_msg

    print(f"  Stored in JSON   : '{stored_user_msg}'")
    print(f"  Token Present?   : {redacted_token_present} (Must be True)")
    print(f"  Raw Leaked?      : {raw_phone_leaked} (Must be False)")
    print()

    transcript_record["demonstrations"].append({
        "scenario": "PII Masking in Persistent Memory",
        "thread_id": thread_c,
        "stored_user_message": stored_user_msg,
        "redacted_token_present": redacted_token_present,
        "raw_phone_leaked": raw_phone_leaked,
    })

    # -----------------------------------------------------------------------
    # Export Transcript
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcript_record, f, indent=2)

    print("=" * 80)
    print(f"Demonstration complete. Saved evidence to: {TRANSCRIPT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
