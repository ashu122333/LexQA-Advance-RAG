# LexQA — Streamlit Frontend

A polished, recruiter-friendly Streamlit UI for the **LexQA** Retrieval-Augmented Generation system.

The interface is designed around one core idea: **transparency**. Every stage of the RAG pipeline lights up live as your query travels through it, so anyone watching can see exactly what the system is doing — from input validation to final, grounded output.

## Features

- **Live pipeline visualization** — 10 stages animate in real time while the backend processes the query, then reconcile to real per-stage timings.
- **Two-column chat layout** — markdown-rendered answers on the left, pipeline trace on the right.
- **System architecture page** — Mermaid flowchart of the full system + component responsibilities table.
- **Knowledge base sidebar** — all 8 source papers with title, authors, year, category, and a one-line summary.
- **Inline metrics** — total latency, per-bucket timing breakdown, groundedness and relevance flags from the output validator.
- **Example prompts & query history** — one-click examples, persistent history within the session.
- **Graceful error handling** — surfaces backend rejections (invalid input, ungrounded output) as warnings, marks the failing stage in the pipeline.

## Run it

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

By default the frontend talks to `http://localhost:8000`. You can change the backend URL from the sidebar at runtime (no restart needed).

Make sure the FastAPI backend is up first:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Entire app — Chat, Architecture, and About pages |
| `.streamlit/config.toml` | Dark theme & server config |
| `requirements.txt` | `streamlit`, `requests` |
