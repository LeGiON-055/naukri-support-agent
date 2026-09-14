"""
main.py — FastAPI Web Interface for Naukri Domain Support Agent (Capstone Task 11)

Exposes RESTful endpoints using Pydantic request and response models:
1. POST /ask:
   - Accepts user query and optional thread_id.
   - Invokes run_agent_workflow() (LangGraph orchestration + persistent memory).
   - Returns validated AgentResponse model conforming to Phase 9 schema.
2. POST /add-document:
   - Ingests new policy text into knowledge_base/ and indexes chunks into ChromaDB.
   - Reuses existing chunking and embedding pipelines.
3. GET /docs:
   - OpenAPI Swagger UI documentation automatically enabled by FastAPI.
"""

import os
import sys
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field
import chromadb

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent.schemas import AgentResponse
from agent.graph import run_agent_workflow
from rag.chunking import sentence_chunks
from rag.embeddings import load_model, generate_embeddings
from api.logger import generate_trace_id, log_request_event

# Paths
KB_DIR = os.path.join(ROOT_DIR, "knowledge_base")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")

# Initialize FastAPI application
app = FastAPI(
    title="Naukri Domain Support Agent API",
    description=(
        "Production-grade FastAPI deployment of the Naukri.com recruitment and "
        "HR policy support agent. Features 5-node LangGraph orchestration, persistent "
        "conversation memory, dual-layer guardrails, and structured response schemas."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The question or prompt to send to the support agent",
        examples=["What is the probation period?"]
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional conversation thread ID to maintain multi-turn memory",
        examples=["api_session_001"]
    )


class AddDocumentRequest(BaseModel):
    filename: str = Field(
        ...,
        min_length=1,
        description="Target filename inside knowledge_base (e.g. 'remote_work.txt')",
        examples=["remote_work_policy.txt"]
    )
    content: str = Field(
        ...,
        min_length=10,
        description="Full text of the recruitment policy document",
        examples=["Remote work is permitted for senior engineers with manager approval."]
    )


class AddDocumentResponse(BaseModel):
    status: str = Field(..., description="Ingestion status ('success' or 'error')")
    filename: str = Field(..., description="Name of the ingested file")
    chunks_added: int = Field(..., description="Number of sentence chunks added to ChromaDB")
    message: str = Field(..., description="Human-readable status summary")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint confirming API service is active."""
    return {
        "service": "Naukri Domain Support Agent API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": ["/ask", "/add-document", "/docs", "/redoc"]
    }


@app.post(
    "/ask",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Agent"],
    summary="Submit a query to the support agent",
)
def ask(request: AskRequest, response: Response):
    """
    Submit a policy inquiry or job application status question.

    - Uses the 5-node LangGraph orchestration workflow.
    - Preserves Task 8 persistent conversation memory when thread_id is supplied.
    - Applies Phase 10 dual-layer safety guardrails (phone masking, prompt injection block).
    - Returns a strict, validated Phase 9 AgentResponse JSON schema.
    - Emits structured JSONL audit logs with trace ID and timing.
    """
    trace_id = generate_trace_id()
    response.headers["X-Trace-ID"] = trace_id
    start_time = time.perf_counter()

    clean_query = request.query.strip()
    if not clean_query:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_request_event(
            trace_id=trace_id,
            method="POST",
            path="/ask",
            status_code=status.HTTP_400_BAD_REQUEST,
            duration_ms=duration_ms,
            request_text=request.query,
            thread_id=request.thread_id,
            error="Query cannot be empty or contain only whitespace.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or contain only whitespace.",
            headers={"X-Trace-ID": trace_id},
        )

    try:
        # Run agent workflow through LangGraph with memory
        result = run_agent_workflow(clean_query, thread_id=request.thread_id)
        response_model = result.get("response")

        if response_model is None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_request_event(
                trace_id=trace_id,
                method="POST",
                path="/ask",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                duration_ms=duration_ms,
                request_text=clean_query,
                thread_id=request.thread_id,
                error="Agent workflow finished without producing a valid AgentResponse.",
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent workflow finished without producing a valid AgentResponse.",
                headers={"X-Trace-ID": trace_id},
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        log_request_event(
            trace_id=trace_id,
            method="POST",
            path="/ask",
            status_code=status.HTTP_200_OK,
            duration_ms=duration_ms,
            request_text=clean_query,
            thread_id=request.thread_id,
            intent=response_model.intent,
        )
        return response_model

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_request_event(
            trace_id=trace_id,
            method="POST",
            path="/ask",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            duration_ms=duration_ms,
            request_text=clean_query,
            thread_id=request.thread_id,
            error=f"Internal agent execution error: {str(e)}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal agent execution error: {str(e)}",
            headers={"X-Trace-ID": trace_id},
        )


@app.post(
    "/add-document",
    response_model=AddDocumentResponse,
    status_code=status.HTTP_200_OK,
    tags=["Knowledge Base"],
    summary="Add and index a new policy document into the knowledge base",
)
def add_document(request: AddDocumentRequest, response: Response):
    """
    Add a new text policy document to knowledge_base/ and index sentence chunks into ChromaDB.
    """
    trace_id = generate_trace_id()
    response.headers["X-Trace-ID"] = trace_id
    start_time = time.perf_counter()

    filename = request.filename.strip()
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    content = request.content.strip()
    summary_text = f"Filename: {filename} (length: {len(content)})"
    if len(content) < 10:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_request_event(
            trace_id=trace_id,
            method="POST",
            path="/add-document",
            status_code=status.HTTP_400_BAD_REQUEST,
            duration_ms=duration_ms,
            request_text=summary_text,
            error="Document content must be at least 10 characters.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document content must be at least 10 characters.",
            headers={"X-Trace-ID": trace_id},
        )

    try:
        # 1. Save document to knowledge_base/
        os.makedirs(KB_DIR, exist_ok=True)
        doc_path = os.path.join(KB_DIR, filename)
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)

        # 2. Chunk document into sentence chunks
        chunks = sentence_chunks(content, filename)
        if not chunks:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_request_event(
                trace_id=trace_id,
                method="POST",
                path="/add-document",
                status_code=status.HTTP_400_BAD_REQUEST,
                duration_ms=duration_ms,
                request_text=summary_text,
                error="Document content could not be partitioned into valid sentence chunks.",
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document content could not be partitioned into valid sentence chunks.",
                headers={"X-Trace-ID": trace_id},
            )

        # 3. Generate embeddings
        model = load_model()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(model, chunk_texts)

        # 4. Store in Chroma collection 'sentence_chunks'
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name="sentence_chunks")

        timestamp_id = int(time.time())
        ids = [f"{filename}_{i}_{timestamp_id}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        log_request_event(
            trace_id=trace_id,
            method="POST",
            path="/add-document",
            status_code=status.HTTP_200_OK,
            duration_ms=duration_ms,
            request_text=summary_text,
            intent="document_ingestion",
        )

        return AddDocumentResponse(
            status="success",
            filename=filename,
            chunks_added=len(chunks),
            message=f"Successfully indexed {len(chunks)} chunks from '{filename}' into knowledge base.",
        )

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_request_event(
            trace_id=trace_id,
            method="POST",
            path="/add-document",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            duration_ms=duration_ms,
            request_text=summary_text,
            error=f"Failed to add document to knowledge base: {str(e)}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document to knowledge base: {str(e)}",
            headers={"X-Trace-ID": trace_id},
        )
