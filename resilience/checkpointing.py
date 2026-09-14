"""
checkpointing.py — SQLite Checkpointing Demo (Capstone Task 15)

Demonstrates saving and resuming a LangGraph workflow using
langgraph-checkpoint-sqlite. Like saving a game, the graph state
is persisted to SQLite so it can be recovered after interruption.

Demonstrates:
1. Build graph with SqliteSaver checkpointer
2. Run a query with thread_id -> state saved to SQLite
3. Verify checkpoint exists in SQLite
4. Resume from checkpoint (simulating recovery after crash)
5. Prove state continuity: node_history preserved across save/resume
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("MOCK_LLM", "true")

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import (
    AgentState,
    guardrail_node,
    intent_router_node,
    route_intent,
    rag_policy_node,
    status_tool_node,
    output_validation_node,
)

# ---------------------------------------------------------------------------
# SQLite Checkpoint Database Path
# ---------------------------------------------------------------------------

CHECKPOINT_DB_PATH = os.path.join(ROOT_DIR, "resilience", "checkpoints.db")
TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "task15_checkpointing.json")


# ---------------------------------------------------------------------------
# Build Graph with SQLite Checkpointer
# ---------------------------------------------------------------------------

def build_checkpointed_graph(db_path: str = CHECKPOINT_DB_PATH, interrupt_before: list = None):
    """Build the 5-node LangGraph with SqliteSaver checkpointer and optional interrupt points."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    workflow = StateGraph(AgentState)

    # Register all 5 nodes (same as agent/graph.py)
    workflow.add_node("guardrail_node", guardrail_node)
    workflow.add_node("intent_router_node", intent_router_node)
    workflow.add_node("rag_policy_node", rag_policy_node)
    workflow.add_node("status_tool_node", status_tool_node)
    workflow.add_node("output_validation_node", output_validation_node)

    # Edges (same as agent/graph.py)
    workflow.add_edge(START, "guardrail_node")
    workflow.add_edge("guardrail_node", "intent_router_node")
    workflow.add_edge("rag_policy_node", "output_validation_node")
    workflow.add_edge("status_tool_node", "output_validation_node")
    workflow.add_edge("output_validation_node", END)

    # Conditional routing
    workflow.add_conditional_edges(
        "intent_router_node",
        route_intent,
        {
            "rag_policy_node": "rag_policy_node",
            "status_tool_node": "status_tool_node",
            "output_validation_node": "output_validation_node",
        },
    )

    compile_kwargs = {"checkpointer": checkpointer}
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    compiled = workflow.compile(**compile_kwargs)
    return compiled, conn, checkpointer


# ---------------------------------------------------------------------------
# Demo Runner
# ---------------------------------------------------------------------------

def run_checkpointing_demo():
    """Demonstrate SQLite checkpointing: interrupt, save, resume, and verify."""
    print("=" * 75)
    print("TASK 15 - SQLITE CHECKPOINTING & INTERRUPTION/RESUME DEMO")
    print("=" * 75)

    # Clean up any previous checkpoint DB
    if os.path.exists(CHECKPOINT_DB_PATH):
        try:
            os.remove(CHECKPOINT_DB_PATH)
        except Exception:
            pass

    rel_db_path = os.path.relpath(CHECKPOINT_DB_PATH, ROOT_DIR).replace("\\", "/")
    results = []

    # Step 1: Build graph with checkpointer and interrupt_before=["status_tool_node"]
    print("\n--- Step 1: Build Checkpointed Graph with Interruption Point ---")
    app, conn, checkpointer = build_checkpointed_graph(
        CHECKPOINT_DB_PATH, interrupt_before=["status_tool_node"]
    )
    print(f"  SQLite DB: {rel_db_path}")
    print(f"  Checkpointer: {type(checkpointer).__name__}")
    print("  Interrupt point: ['status_tool_node']")
    results.append({
        "step": "build_graph",
        "checkpointer_type": type(checkpointer).__name__,
        "db_path": rel_db_path,
        "interrupt_before": ["status_tool_node"],
    })

    # Step 2: Run 1 (Interrupted Run)
    print("\n--- Step 2: Run 1 — Execute until Interrupt (Thread: chk-interrupt-demo) ---")
    thread_id = "chk-interrupt-demo"
    thread_config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "raw_query": "What is the status of APP-005?",
        "masked_query": "What is the status of APP-005?",
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
        "thread_id": thread_id,
        "history": [],
    }

    print(f"  Input query: {initial_state['raw_query']}")
    app.invoke(initial_state, config=thread_config)
    state_checkpoint_1 = app.get_state(thread_config)
    run1_nodes = state_checkpoint_1.values.get("node_history", [])
    next_nodes = list(state_checkpoint_1.next)

    print(f"  Run 1 Halted at Interrupt Point!")
    print(f"  Completed Nodes: {run1_nodes}")
    print(f"  Next Pending Node(s): {next_nodes}")
    print(f"  Intent Determined: {state_checkpoint_1.values.get('intent')}")
    print(f"  Response yet created: {state_checkpoint_1.values.get('response') is not None}")

    results.append({
        "step": "run_1_interrupted",
        "thread_id": thread_id,
        "query": initial_state["raw_query"],
        "intent": state_checkpoint_1.values.get("intent"),
        "nodes_executed": run1_nodes,
        "interrupted_before": next_nodes,
        "checkpoint_saved": len(next_nodes) > 0,
    })

    # Step 3: Verify Checkpoint in SQLite
    print("\n--- Step 3: Verify Checkpoint in SQLite ---")
    db_exists = os.path.isfile(CHECKPOINT_DB_PATH)
    db_size = os.path.getsize(CHECKPOINT_DB_PATH) if db_exists else 0
    print(f"  DB exists: {db_exists}")
    print(f"  DB size: {db_size} bytes")

    verify_conn = sqlite3.connect(CHECKPOINT_DB_PATH)
    cursor = verify_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"  Tables: {tables}")

    checkpoint_count = 0
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM \"{table}\"")
        count = cursor.fetchone()[0]
        print(f"    {table}: {count} rows")
        checkpoint_count += count
    verify_conn.close()

    results.append({
        "step": "verify_checkpoint",
        "db_exists": db_exists,
        "db_size_bytes": db_size,
        "tables": tables,
        "total_rows": checkpoint_count,
    })

    # Step 4: Run 2 (Resume from Checkpoint on same thread)
    print("\n--- Step 4: Run 2 — Resume Execution from SQLite Checkpoint ---")
    print("  Resuming same thread with app.invoke(None)...")
    final_state = app.invoke(None, config=thread_config)
    state_checkpoint_2 = app.get_state(thread_config)
    run2_all_nodes = state_checkpoint_2.values.get("node_history", [])
    run2_next = list(state_checkpoint_2.next)
    resp = final_state.get("response")
    answer = resp.answer if hasattr(resp, "answer") else str(resp)

    run1_count = len(run1_nodes)
    resumed_new_nodes = run2_all_nodes[run1_count:]
    interruption_resumed = (
        len(run1_nodes) == 2
        and "status_tool_node" in resumed_new_nodes
        and len(run2_next) == 0
    )

    print(f"  Run 2 Completed Successfully!")
    print(f"  Previously completed nodes skipped: {run1_nodes}")
    print(f"  Nodes executed upon resume: {resumed_new_nodes}")
    print(f"  Full Node History: {run2_all_nodes}")
    print(f"  Final Answer: {answer[:100]}...")
    print(f"  Schema Valid: {final_state.get('is_valid_schema')}")
    print(f"  Resume Verified: {interruption_resumed}")

    results.append({
        "step": "run_2_resumed",
        "thread_id": thread_id,
        "previously_completed_nodes_skipped": run1_nodes,
        "resumed_executed_nodes": resumed_new_nodes,
        "full_node_history": run2_all_nodes,
        "answer_preview": answer[:120],
        "is_valid_schema": final_state.get("is_valid_schema"),
        "interruption_resumed_successfully": interruption_resumed,
    })

    # Step 5: Multi-thread Isolation Test (Run query on thread 2)
    print("\n--- Step 5: Multi-Thread Isolation (Thread: chk-thread-2) ---")
    app_uninterrupted, conn_unint, _ = build_checkpointed_graph(
        CHECKPOINT_DB_PATH, interrupt_before=None
    )
    thread_config_2 = {"configurable": {"thread_id": "chk-thread-2"}}
    initial_state_2 = {
        "raw_query": "What is the remote work policy?",
        "masked_query": "What is the remote work policy?",
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
        "thread_id": "chk-thread-2",
        "history": [],
    }

    final_state_2 = app_uninterrupted.invoke(initial_state_2, config=thread_config_2)
    resp_2 = final_state_2.get("response")
    answer_2 = resp_2.answer if hasattr(resp_2, "answer") else str(resp_2)
    nodes_2 = final_state_2.get("node_history", [])

    print(f"  Query: {initial_state_2['raw_query']}")
    print(f"  Intent: {final_state_2.get('intent')}")
    print(f"  Node History: {nodes_2}")
    print(f"  Answer: {answer_2[:100]}...")

    results.append({
        "step": "run_query_thread_2",
        "thread_id": "chk-thread-2",
        "query": initial_state_2["raw_query"],
        "intent": final_state_2.get("intent"),
        "node_history": nodes_2,
        "answer_preview": answer_2[:120],
        "is_valid_schema": final_state_2.get("is_valid_schema"),
    })

    # Step 6: Verify State Continuity across all threads
    print("\n--- Step 6: Verify State Continuity Across Threads ---")
    saved_state_1 = app.get_state(thread_config)
    saved_state_2 = app_uninterrupted.get_state(thread_config_2)

    thread_1_match = (saved_state_1.values.get("node_history") == run2_all_nodes)
    thread_2_match = (saved_state_2.values.get("node_history") == nodes_2)
    state_continuity = thread_1_match and thread_2_match and interruption_resumed

    print(f"  Thread 1 final state matches SQLite: {thread_1_match}")
    print(f"  Thread 2 final state matches SQLite: {thread_2_match}")
    print(f"  Interruption & resume verified: {interruption_resumed}")
    print(f"  Overall state continuity verified: {state_continuity}")

    results.append({
        "step": "resume_checkpoint",
        "thread_1_state_match": thread_1_match,
        "thread_1_resumed_nodes": saved_state_1.values.get("node_history"),
        "thread_2_state_match": thread_2_match,
        "thread_2_resumed_nodes": saved_state_2.values.get("node_history"),
        "interruption_resumed_verified": interruption_resumed,
        "state_continuity_verified": state_continuity,
    })

    conn.close()
    conn_unint.close()

    # Save transcript
    transcript = {
        "task": "Task 15 - SQLite Checkpointing",
        "timestamp": datetime.now().isoformat(),
        "mock_mode": True,
        "db_path": rel_db_path,
        "interruption_demo": {
            "run_1_halted_at": next_nodes,
            "run_1_completed_nodes": run1_nodes,
            "run_2_resumed_nodes": resumed_new_nodes,
            "run_2_final_nodes": run2_all_nodes,
            "interruption_resume_verified": interruption_resumed,
        },
        "demo_steps": results,
        "state_continuity_verified": state_continuity,
    }

    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nTranscript saved to: {TRANSCRIPT_PATH}")
    print("=" * 75)

    return transcript


if __name__ == "__main__":
    run_checkpointing_demo()
