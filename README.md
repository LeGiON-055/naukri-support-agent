# Naukri.com Domain Support Agent

A recruitment and HR support agent for a fictional Naukri.com employer-support scenario, built as a Masai School Final Capstone project.

> **Status: Under Development** — The project structure is in place. Features are being implemented phase by phase.

## What This Agent Will Do

When complete, this agent will be able to:

- **Answer recruitment-policy questions** using RAG (Retrieval-Augmented Generation) over a local knowledge base
- **Check job application status** using a generated dataset of 40+ applications
- **Route questions intelligently** using a LangGraph workflow with conditional routing
- **Remember conversation context** across multiple turns within a session
- **Apply safety guardrails** — mask phone numbers, detect prompt injection, and refuse unsupported answers
- **Expose a web API** using FastAPI (`POST /ask`, `POST /add-document`)
- **Expose the status tool via MCP** (Model Context Protocol) for standard tool discovery
- **Demonstrate resilience** with SQLite checkpointing, exponential-backoff retries, and timeouts
- **Evaluate RAG quality** using context relevance, groundedness, and answer relevance metrics
- **Work offline** using MOCK_LLM mode (no API keys or network access required)

## Tech Stack

- Python
- LangGraph (workflow orchestration)
- LangChain (LLM integration)
- ChromaDB (vector database)
- SentenceTransformers (local embeddings)
- FastAPI (web API)
- FastMCP (MCP server)
- SQLite (checkpointing)
- Pydantic (data validation)

## Project Structure

```
naukri-support-agent/
│
├── dataset.py              # Generates 40+ fake job applications (deterministic seed)
├── knowledge_base/         # 12 recruitment policy documents (plain text)
│
├── rag/                    # Retrieval-Augmented Generation pipeline
│   ├── chunking.py         #   Split documents into searchable chunks
│   ├── embeddings.py       #   Convert text into numerical vectors
│   ├── retrieval.py        #   Find the most relevant chunks for a question
│   └── generation.py       #   Generate an answer from retrieved context
│
├── agent/                  # LangGraph agent and supporting logic
│   ├── graph.py            #   Workflow definition (nodes, edges, routing)
│   ├── tools.py            #   Functions the agent can call (RAG, status check)
│   ├── memory.py           #   Conversation history (per-session persistence)
│   ├── schemas.py          #   Structured response format (JSON schema)
│   └── guardrails.py       #   Input sanitization and output validation
│
├── evaluation/             # Quality measurement scripts
│   ├── eval_set.py         #   15 test queries with expected results
│   ├── evaluate_retrieval.py  # Precision@3 and Recall@3 comparison
│   └── evaluate_agent.py   #   RAG triad evaluation (context, ground, answer)
│
├── resilience/             # Robustness demonstrations
│   ├── checkpointing.py    #   Save/resume workflows with SQLite
│   ├── retry.py            #   Exponential backoff for transient failures
│   └── timeout.py          #   Per-node and global timeout handling
│
├── mcp/                    # Model Context Protocol integration
│   ├── server.py           #   Publishes the status-check tool
│   └── client.py           #   Connects to the server and calls the tool
│
├── api/                    # Web interface
│   └── main.py             #   FastAPI endpoints (POST /ask, POST /add-document)
│
├── transcripts/            # Stored conversation histories (JSON)
│
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template (no real secrets)
├── .gitignore              # Files excluded from version control
└── README.md               # This file
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/naukri-support-agent.git
cd naukri-support-agent

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy the environment template
cp .env.example .env
```

## Phase 5 — Retrieval & Empirical Threshold Calibration

### 1. Retrieval Architecture
The retrieval layer (`rag/retrieval.py`) searches ChromaDB vector store for relevant knowledge base policy chunks:
- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers, 384 dimensions, local execution, 0 external API calls).
- **Supported Collections**:
  - `sentence_chunks` (Sentence-based chunking)
  - `fixed_size_chunks` (Fixed 200-char with 50-char overlap)
- **Top-k**: Default `k = 3`.
- **Similarity Metric**: Cosine similarity. Chroma collections are initialized with `metadata={"hnsw:space": "cosine"}`, which causes Chroma to return cosine distance ($d \in [0, 2]$). Retrieval converts this to cosine similarity:
  $$\text{cosine\_similarity} = 1.0 - \text{cosine\_distance}$$
- **Metadata**: Preserves `source` document filename, `chunk_index`, and `strategy` for traceability.

### 2. Empirical Calibration Results (`calibrate_threshold.py`)
Rather than picking an arbitrary threshold (e.g. 0.5 or 0.7), the threshold was empirically calibrated using 7 in-scope queries (covering core policies) and 3 out-of-scope queries (general knowledge/weather/recipes):

| Collection | In-Scope Range (Top-1) | Out-of-Scope Range (Top-1) | Score Margin | Calibrated Threshold |
|------------|------------------------|----------------------------|--------------|----------------------|
| `sentence_chunks` | 0.4683 – 0.8520 (mean: 0.6950) | 0.0123 – 0.0977 (mean: 0.0683) | +0.3706 | **0.28** |
| `fixed_size_chunks` | 0.5554 – 0.8544 (mean: 0.7055) | 0.0499 – 0.1141 (mean: 0.0836) | +0.4413 | **0.28** |

**Rationale for Chosen Threshold (0.28)**:
The empirical boundary shows that the highest out-of-scope score observed is 0.0977 (`sentence_chunks`) and 0.1141 (`fixed_size_chunks`), while the lowest in-scope score is 0.4683. The midpoint between the closest clusters $(0.4683 + 0.0977) / 2 = 0.2830$ rounds to **0.28**. This creates a balanced guardrail with a >0.18 safety margin against both false positives and false negatives.

### 3. Fallback Behavior
Under `MOCK_LLM` mode, cosine similarity against the knowledge base is the sole signal used for decision-making (no keyword filtering or LLM classification).
- If $\text{top-1 similarity} \ge 0.28$: Query is accepted; retrieved chunks are forwarded as context.
- If $\text{top-1 similarity} < 0.28$: Query is marked unsupported and rejected with fallback message:
  > *"I don't have enough information to answer that question. Please contact HR directly for assistance."*

### 4. Verification & Demonstration
The pipeline was verified with 5 in-scope queries and 1 out-of-scope query across both collections:
- **In-scope (Accepted)**:
  - *"Who can apply for a job at the company?"* (Sim: 0.6317 / 0.6380) → Accepted
  - *"How do I reschedule my interview?"* (Sim: 0.7113 / 0.6816) → Accepted
  - *"What is verified during background checks?"* (Sim: 0.7054 / 0.7347) → Accepted
  - *"How long is the notice period?"* (Sim: 0.6919 / 0.6953) → Accepted
  - *"What is the policy on diversity hiring?"* (Sim: 0.6011 / 0.6068) → Accepted
- **Out-of-scope (Fallback Triggered)**:
  - *"What is the best programming language to learn?"* (Sim: 0.0571 / 0.1120) → **Rejected $\rightarrow$ Fallback Triggered**

Evidence is permanently logged in [`transcripts/phase5_calibration.json`](file:///c:/Users/ACER/Capstone%20Project/naukri-support-agent/transcripts/phase5_calibration.json).

## How to Run

```bash
# 1. Build vector stores (if not already done)
python build_vector_store.py

# 2. Run empirical threshold calibration
python calibrate_threshold.py

# 3. Validate Phase 5
python validate_phase5.py
```

## Limitations

- This is a learning project, not a production system.
- The knowledge base is small and fictional.
- MOCK_LLM mode produces deterministic but simplified answers.
- The dataset is synthetically generated, not from real Naukri.com data.

