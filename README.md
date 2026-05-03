# Helix SROP — Himanshu Wagh

A stateful, agentic RAG orchestration pipeline built with **Google ADK**, **FastAPI**, and **SQLAlchemy**. This system provides a support concierge that can answer product questions, manage user builds, and escalate issues to human support, all while maintaining context across restarts.

## Setup

```bash
git clone <your-repo>
cd helix-srop-assignment
uv sync
cp .env.example .env  # fill in GOOGLE_API_KEY
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload
```

## Quick Test

```bash
# 1. Create a session
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

# 2. Ask a product question (Knowledge Agent)
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .

# 3. Ask an account question (Account Agent - relies on session state)
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "Show me my recent builds"}' | jq .
```

## Architecture

```text
       ┌──────────────────────────────────────────────────────────┐
       │                 API Request (POST /chat)                 │
       └────────────────────────────┬─────────────────────────────┘
                                    │
           ┌────────────────────────▼────────────────────────┐
           │  Pipeline Entry (E1: Idempotency & E5: Redaction) │
           └────────────────────────┬────────────────────────┘
                                    │
           ┌────────────────────────▼────────────────────────┐
           │        Deterministic Guard (State Lookups)       │
           └───────────┬────────────────────────────┬────────┘
                       │ (Match)                    │ (No Match)
                       ▼                            ▼
           ┌───────────────────────┐    ┌───────────────────────────┐
           │    Direct Response    │    │  Agentic Orchestration    │
           │     (Smalltalk)       │    │      (Google ADK)         │
           └───────────┬───────────┘    └────────────┬──────────────┘
                       │                             │
                       │             ┌───────────────┼───────────────┐
                       │             │               │               │
                       │      ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                       │      │ Knowledge   │ │ Account     │ │ Escalation  │
                       │      │ Agent (RAG) │ │ Agent (DB)  │ │ Agent (E2)  │
                       │      └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                       │             │               │               │
                       └─────────────┴───────┬───────┴───────────────┘
                                             │
                       ┌─────────────────────▼───────────────────────┐
                       │   Persistence Layer (SQLite + ChromaDB)     │
                       │   - Messages, SessionState, AgentTraces     │
                       └─────────────────────┬───────────────────────┘
                                             │
       ┌─────────────────────────────────────▼───────────────────────┐
       │                API Response (PipelineResult)                │
       └─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### State persistence (Pattern 3)
I used **Pattern 3 (In-memory runner with DB re-hydration)**. Every turn, the session state is loaded from SQLite and injected into the agent's system prompt. This ensures that the agent survives server restarts and scales horizontally without losing context, while avoiding the complexity of a persistent WebSocket-based runner.

### Chunking strategy
I used **heading-aware chunking** for the documentation. Technical documents (Markdown) are naturally segmented by headers. This strategy ensures that related instructions stay within the same context window, improving the relevance of retrieval for "How-to" queries compared to fixed-size splitting.

### Vector store choice
I chose **Chroma** because it provides robust local persistence, supports metadata filtering (useful for future product-area scoped searches), and integrates seamlessly with the Google Generative AI embeddings used in this project.

## Known Limitations

- **Rate Limits**: The ingestion script can hit Gemini free-tier rate limits if the `docs/` folder is very large.
- **Routing Ambiguity**: In very short queries (e.g., "help"), the root agent may occasionally default to `smalltalk` instead of asking clarifying questions for escalation.
- **Trace Visualization**: While traces are captured in SQLite, there is no built-in UI for viewing the graph; they must be queried via the `/v1/traces/` endpoint.

## What I'd Do With More Time

- **E3 (Streaming)**: Implement Server-Sent Events (SSE) to improve perceived latency.
- **Reranking**: Add a cross-encoder reranking step after vector retrieval to further improve RAG accuracy for ambiguous queries.
- **Conversation UI**: Build a frontend dashboard to visualize the agentic event loop and the citation chunks in real-time.

## Time Spent

| Phase | Time |
|-------|------|
| Setup + DB + FastAPI boilerplate | 45 min |
| RAG ingest + search_docs | 60 min |
| ADK agents + Orchestration | 90 min |
| pipeline.py + state persistence | 60 min |
| Extensions (E1, E2, E5, E7) | 120 min |
| README + Demo Prep | 30 min |
| **Total** | **~7h 45m** |

## Extensions Completed

- [x] **E1: Idempotency** — Supports `Idempotency-Key` header to prevent duplicate processing.
- [x] **E2: Escalation agent** — Handles user frustration and creates support tickets.
- [x] **E5: Guardrails** — Refuses out-of-scope requests and redacts PII in logs.
- [x] **E7: Eval harness** — Automated routing accuracy report (Current Accuracy: **85.7%**).
