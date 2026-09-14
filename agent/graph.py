"""
graph.py — LangGraph Workflow Orchestration (Capstone Task 7 & Task 8)

Defines the 5-node LangGraph state graph for the Naukri Domain Support Agent:
1. guardrail_node: Applies Phase 10 PII masking and prompt injection detection.
2. intent_router_node: Deterministically classifies intent (policy vs status).
3. rag_policy_node: Executes Phase 6 grounded retrieval over policy knowledge base.
4. status_tool_node: Executes Phase 8 application status lookup with escalation scoring.
5. output_validation_node: Validates outgoing response against Phase 9 AgentResponse schema.

Integrated with:
- Genuine LangGraph conditional edge routing via route_intent.
- Capstone Task 8 persistent conversation memory (load_conversation, save_turn, contextualize_query).
"""

import os
import re
import sys
from typing import TypedDict, Optional, List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from langgraph.graph import StateGraph, START, END

# Import schemas, tools, guardrails, and memory
from agent.schemas import (
    AgentResponse,
    build_policy_response,
    build_status_response,
    build_refusal_response,
    validate_agent_response,
)
from agent.guardrails import (
    mask_phone_numbers,
    detect_prompt_injection,
    DEFAULT_THRESHOLD,
    DEFAULT_GROUNDEDNESS_THRESHOLD,
    FALLBACK_RESPONSE,
)
from agent.tools import check_job_application_status
from rag.generation import answer_query
from agent.memory import (
    load_conversation,
    save_turn,
    contextualize_query,
)


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    raw_query: str                          # Raw input query
    masked_query: str                      # Sanitized query (phone replaced with [REDACTED_PHONE])
    pii_detected: bool                     # Whether a phone number was detected
    is_injection: bool                     # Whether prompt injection was detected
    matched_pattern: Optional[str]         # Matched injection pattern if any
    intent: Optional[str]                  # "policy_inquiry" | "application_status" | "security_violation"
    record_id: Optional[str]               # Extracted candidate application ID (e.g. "APP-002")
    tool_invoked: bool                     # True if RAG or Status tool was called
    rag_result: Optional[Dict[str, Any]]   # Raw RAG telemetry bundle
    status_result: Optional[Dict[str, Any]]# Raw status tool telemetry bundle
    response: Optional[AgentResponse]      # Structured output model
    is_valid_schema: bool                  # Result of schema validation
    node_history: List[str]                # Chronological trace of visited nodes
    thread_id: Optional[str]               # Active session / thread identifier
    history: List[Dict[str, Any]]          # Prior conversation messages loaded from memory


# ---------------------------------------------------------------------------
# 2. Graph Node Functions
# ---------------------------------------------------------------------------

def guardrail_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Input Guardrail Node
    Applies PII masking and prompt injection detection before any downstream execution.
    """
    history = list(state.get("node_history", []))
    history.append("guardrail_node")

    raw_q = state.get("raw_query", "")
    masked_q, has_pii, detected_phones = mask_phone_numbers(raw_q)
    is_injection, matched_pat = detect_prompt_injection(masked_q)

    response = None
    if is_injection:
        response = build_refusal_response(
            intent="security_violation",
            answer=(
                "Security violation detected: The input contains prohibited prompt override "
                "or injection patterns. Request has been blocked."
            ),
            reason="prompt_injection",
            details=f"Matched prohibited override pattern: '{matched_pat}'",
        )

    return {
        "masked_query": masked_q,
        "pii_detected": has_pii,
        "is_injection": is_injection,
        "matched_pattern": matched_pat,
        "response": response,
        "node_history": history,
    }


def intent_router_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Intent Router Node
    Classifies intent deterministically and extracts candidate application IDs.
    """
    history = list(state.get("node_history", []))
    history.append("intent_router_node")

    # If injection detected at perimeter, preserve security_violation
    if state.get("is_injection"):
        return {
            "intent": "security_violation",
            "record_id": None,
            "node_history": history,
        }

    masked_q = state.get("masked_query", "")

    # Check for application record ID (e.g. APP-001)
    app_match = re.search(r'\bAPP-\d{3}\b', masked_q, re.IGNORECASE)
    if app_match:
        return {
            "intent": "application_status",
            "record_id": app_match.group(0).upper(),
            "node_history": history,
        }

    # Default to policy inquiry
    return {
        "intent": "policy_inquiry",
        "record_id": None,
        "node_history": history,
    }


def route_intent(state: AgentState) -> str:
    """
    Conditional Routing Function
    Routes flow dynamically based on classified intent:
      - 'security_violation' -> 'output_validation_node' (tools bypassed!)
      - 'application_status' -> 'status_tool_node'
      - 'policy_inquiry'     -> 'rag_policy_node'
    """
    intent = state.get("intent")
    if intent == "security_violation":
        return "output_validation_node"
    elif intent == "application_status":
        return "status_tool_node"
    else:
        return "rag_policy_node"


def rag_policy_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Policy RAG Node
    Executes semantic search over policy documents with dual-layer defense.
    """
    history = list(state.get("node_history", []))
    history.append("rag_policy_node")

    query = state.get("masked_query", "")
    rag_result = answer_query(query, threshold=DEFAULT_THRESHOLD)

    rejection_reason = rag_result.get("rejection_reason")
    top1_sim = rag_result.get("top1_similarity", 0.0)
    groundedness_score = rag_result.get("groundedness_score", 0.0)

    if rejection_reason == "retrieval_threshold":
        response = build_refusal_response(
            intent="out_of_scope",
            answer=FALLBACK_RESPONSE,
            reason="retrieval_threshold",
            details=f"Query similarity {top1_sim:.4f} below calibrated threshold {DEFAULT_THRESHOLD}.",
            similarity_score=top1_sim,
            threshold=DEFAULT_THRESHOLD,
        )
    elif rejection_reason == "groundedness_failure":
        response = build_refusal_response(
            intent="out_of_scope",
            answer=FALLBACK_RESPONSE,
            reason="groundedness_failure",
            details=f"Candidate answer failed groundedness verification (score {groundedness_score:.4f} < {DEFAULT_GROUNDEDNESS_THRESHOLD}).",
            similarity_score=top1_sim,
            threshold=DEFAULT_GROUNDEDNESS_THRESHOLD,
        )
    else:
        response = build_policy_response(
            answer=rag_result["answer"],
            sources=rag_result["sources"],
        )

    return {
        "rag_result": rag_result,
        "tool_invoked": True,
        "response": response,
        "node_history": history,
    }


def status_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Application Status Tool Node
    Queries candidate records and computes escalation scoring.
    """
    history = list(state.get("node_history", []))
    history.append("status_tool_node")

    record_id = state.get("record_id", "")
    status_data = check_job_application_status(record_id)

    response = build_status_response(
        answer=status_data["message"],
        application_details=status_data,
        is_escalated=status_data["escalated"],
    )

    return {
        "status_result": status_data,
        "tool_invoked": True,
        "response": response,
        "node_history": history,
    }


def output_validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Output Validation Node
    Guarantees every response complies with the Phase 9 AgentResponse JSON Schema.
    """
    history = list(state.get("node_history", []))
    history.append("output_validation_node")

    resp = state.get("response")
    is_valid, validated_model, _ = validate_agent_response(resp)

    return {
        "response": validated_model if is_valid else resp,
        "is_valid_schema": is_valid,
        "node_history": history,
    }


# ---------------------------------------------------------------------------
# 3. Build & Compile StateGraph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    workflow = StateGraph(AgentState)

    # Register all 5 discrete nodes
    workflow.add_node("guardrail_node", guardrail_node)
    workflow.add_node("intent_router_node", intent_router_node)
    workflow.add_node("rag_policy_node", rag_policy_node)
    workflow.add_node("status_tool_node", status_tool_node)
    workflow.add_node("output_validation_node", output_validation_node)

    # Standard sequential edges
    workflow.add_edge(START, "guardrail_node")
    workflow.add_edge("guardrail_node", "intent_router_node")
    workflow.add_edge("rag_policy_node", "output_validation_node")
    workflow.add_edge("status_tool_node", "output_validation_node")
    workflow.add_edge("output_validation_node", END)

    # Genuine LangGraph conditional routing edge
    workflow.add_conditional_edges(
        "intent_router_node",
        route_intent,
        {
            "rag_policy_node": "rag_policy_node",
            "status_tool_node": "status_tool_node",
            "output_validation_node": "output_validation_node",
        },
    )

    return workflow.compile()


# Compile global workflow runner
app = build_graph()


# ---------------------------------------------------------------------------
# 4. Public Agent Runner (with Persistent Memory Integration)
# ---------------------------------------------------------------------------

def run_agent_workflow(query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the agent workflow for a query with optional conversation memory.

    Lifecycle:
    1. Memory retrieval: Load previous turns for thread_id.
    2. Query Contextualization: Resolve follow-up references using memory.
    3. LangGraph Execution: Process through the 5-node StateGraph.
    4. Memory Persistence: Save turn (user query + assistant answer) to JSON.
    """
    history: List[Dict[str, Any]] = []
    effective_query = query

    # Pre-execution: load memory and contextualize query
    if thread_id:
        history = load_conversation(thread_id)
        effective_query = contextualize_query(query, history)

    initial_state: AgentState = {
        "raw_query": effective_query,
        "masked_query": effective_query,
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
        "history": history,
    }

    # Execute compiled LangGraph
    final_state = app.invoke(initial_state)

    # Post-execution: save turn to persistent JSON history
    if thread_id and final_state.get("response"):
        resp_obj = final_state["response"]
        answer_text = resp_obj.answer if hasattr(resp_obj, "answer") else str(resp_obj)
        save_turn(thread_id, query, answer_text)

    return final_state
