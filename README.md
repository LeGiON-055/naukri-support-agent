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

## How to Run

> Coming soon — features are being implemented phase by phase.

## Limitations

- This is a learning project, not a production system.
- The knowledge base is small and fictional.
- MOCK_LLM mode produces deterministic but simplified answers.
- The dataset is synthetically generated, not from real Naukri.com data.
