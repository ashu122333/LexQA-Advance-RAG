# LexQA — Transparent Retrieval-Augmented Generation

A portfolio-grade Retrieval-Augmented Generation system that answers questions about a curated corpus of foundational RAG research papers — and lets you **watch every stage of the pipeline run in real time**.

LexQA is built around one principle: a production RAG system is not one model call, it's a pipeline of tightly orchestrated components. The frontend makes every step of that pipeline observable, so the engineering depth behind the answer is as visible as the answer itself.

---

## What's inside

```
LexQA/
├── backend/        FastAPI service — the full RAG pipeline
│   ├── main.py     /query · /md · /health endpoints
│   └── rag/        pipeline stages (validation, retrieval, generation, audit)
├── frontend/       Streamlit UI — chat + live pipeline visualization
│   └── app.py
├── datasource/     8 source PDFs (foundational RAG papers)
└── README.md       this file
```

---

## Pipeline

Every query passes through three phases:

```
                 ┌──────────────────────────────────────────────────┐
 user question → │ 1 · understand   validate → rewrite → plan hops  │
                 ├──────────────────────────────────────────────────┤
                 │ 2 · retrieve     dense (pgvector) ─┐              │
                 │                                    ├─→ RRF + rerank
                 │                  BM25 (tsvector) ──┘              │
                 ├──────────────────────────────────────────────────┤
                 │ 3 · generate     generate → audit → format       │
                 └──────────────────────────────────────────────────┘
                                                                  ↓
                                                       grounded answer
```

| Phase | Stage | What it does |
|---|---|---|
| **1. understand** | Input validation | LLM guardrail rejects malformed, unsafe, or off-domain queries before any retrieval cost |
|  | Query rewrite   | Rewrites the question into a clear, self-contained form |
|  | Hop planner     | Classifies the query as single-hop or multi-hop; multi-hop questions are decomposed into sub-questions |
| **2. retrieve** | Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2` embeddings in `pgvector` (cosine) |
|  | BM25 retrieval  | Postgres `tsvector` lexical index — catches acronyms, code, exact names |
|  | RRF fusion      | Reciprocal Rank Fusion combines both into a top-10 candidate set |
|  | Reranker        | Cohere cross-encoder (via OpenRouter) re-scores the candidates pairwise → top-5 |
| **3. generate** | LLM generation  | Generator produces an answer conditioned strictly on the assembled context |
|  | Output validator| Second LLM pass scores **groundedness** (no unsupported claims) and **relevance** |
|  | Format          | Markdown-rendered response with timings and validation flags |

---

## Quickstart

### Prerequisites
- Python 3.11+
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed
- An [OpenRouter](https://openrouter.ai) API key

### Environment

Create `backend/.env`:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/lexqa
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_REASONING_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_GENERATION_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_RERANK_MODEL=cohere/rerank-english-v3.0
```

### Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Healthcheck: <http://localhost:8000/health> → `{"status": "ok"}`

### Run the frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. The sidebar lets you change the backend URL at runtime.

---

## API reference

### `POST /query`

```json
{ "question": "How does Self-RAG decide when to retrieve?" }
```

Response on success:

```json
{
  "response": "Self-RAG uses reflection tokens …",
  "timings": {
    "input_preprocessing_ms": 812,
    "retrieval_ms": 1340,
    "generation_ms": 2105
  },
  "validation": {
    "grounded": true,
    "relevant": true,
    "grounded_unsupported_claims": [],
    "relevant_explanation": ""
  }
}
```

Response on guarded rejection:

```json
{ "error": "Invalid query", "reason": "Query is off-domain." }
```

### `POST /md`

Takes a raw answer string and returns a Markdown-formatted version.

### `GET /health`

Liveness probe.

---

## Knowledge base — 8 foundational papers

| arXiv | Paper | Category |
|---|---|---|
| [2002.08909](https://arxiv.org/abs/2002.08909) | REALM — Retrieval-Augmented LM Pre-training | Foundations |
| [2004.04906](https://arxiv.org/abs/2004.04906) | Dense Passage Retrieval (DPR) | Retrieval |
| [2005.11401](https://arxiv.org/abs/2005.11401) | RAG — Retrieval-Augmented Generation (Lewis et al.) | Foundations |
| [2305.06983](https://arxiv.org/abs/2305.06983) | FLARE — Active Retrieval Augmented Generation | Adaptive RAG |
| [2306.04136](https://arxiv.org/abs/2306.04136) | Chain-of-Note — Robust RAG via Reading Notes | Reasoning |
| [2310.11511](https://arxiv.org/abs/2310.11511) | Self-RAG — Self-Reflective Retrieval-Augmented Generation | Adaptive RAG |
| [2312.10997](https://arxiv.org/abs/2312.10997) | RAG for LLMs — A Survey | Survey |
| [2407.01219](https://arxiv.org/abs/2407.01219) | Searching for Best Practices in RAG | Engineering |

The PDFs live in [`datasource/`](datasource/) and are chunked, summarised, embedded, and indexed into Postgres before the backend serves any query.

---

## Tech stack

| Layer | Tooling |
|---|---|
| Frontend     | Streamlit + custom CSS, Mermaid (architecture diagram), threaded API client (live pipeline animation) |
| Backend      | FastAPI · Uvicorn · Pydantic |
| Vector store | PostgreSQL + `pgvector` (cosine similarity) |
| Lexical index| Postgres `tsvector` full-text search |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker     | Cohere `rerank-english-v3.0` via OpenRouter |
| LLMs         | OpenRouter-hosted models for validation, rewriting, planning, generation, and auditing |

---

## Engineering highlights

- **Hybrid retrieval out of the box.** Dense + lexical fused with RRF — robust to both semantic and keyword-heavy queries.
- **Cross-encoder reranking.** The single highest-leverage step in real-world RAG, included by default.
- **Two-sided LLM guardrails.** An input validator blocks bad queries cheaply; an output validator catches ungrounded or irrelevant answers before they reach the user.
- **Multi-hop planning.** Complex questions are decomposed into sub-questions, each retrieved independently, then merged into a single grounded context.
- **Observable pipeline.** The frontend animates each stage live and surfaces real per-stage timings — no black box.

---

## Project status

LexQA is a working demo, not a production service. Known limitations:

- The ingestion pipeline (PDF parsing, chunking, summarisation, embedding) lives outside this repo. The backend assumes a populated `chunks` table.
- API keys and DB URLs are read from `backend/.env`; no secrets manager.
- No auth, rate limiting, caching, or observability stack — out of scope for the demo.

Roadmap ideas:
- Streaming pipeline updates over Server-Sent Events instead of post-hoc timing reconciliation.
- Inline source citations in the answer (chunk → arXiv ID anchors).
- Eval harness over a held-out QA set with groundedness/relevance scoring.
