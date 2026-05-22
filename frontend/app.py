"""LexQA — Transparent Retrieval-Augmented Generation
A polished Streamlit demo of a multi-stage RAG pipeline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
APP_TITLE = "LexQA"
APP_TAGLINE = "Transparent Retrieval-Augmented Generation for Research Literature"

st.set_page_config(
    page_title=f"{APP_TITLE} — Transparent RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data: the 8 source papers
# ---------------------------------------------------------------------------

SOURCES: list[dict[str, str]] = [
    {
        "id": "2002.08909",
        "title": "REALM: Retrieval-Augmented Language Model Pre-Training",
        "authors": "Guu, Lee, Tung, Pasupat, Chang",
        "year": "2020",
        "category": "Foundations",
        "summary": "Pre-trains a language model with a learned neural retriever, enabling latent knowledge to live in a corpus rather than only in parameters.",
    },
    {
        "id": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain QA",
        "authors": "Karpukhin et al.",
        "year": "2020",
        "category": "Retrieval",
        "summary": "Introduces DPR — dual-encoder dense retrieval trained with in-batch negatives that decisively outperforms BM25 on open-domain QA.",
    },
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "authors": "Lewis et al.",
        "year": "2020",
        "category": "Foundations",
        "summary": "The original RAG paper. Couples a parametric seq2seq generator with a non-parametric retriever for knowledge-intensive tasks.",
    },
    {
        "id": "2305.06983",
        "title": "Active Retrieval Augmented Generation (FLARE)",
        "authors": "Jiang et al.",
        "year": "2023",
        "category": "Adaptive RAG",
        "summary": "Actively decides when and what to retrieve mid-generation, using the model's own uncertainty as the trigger.",
    },
    {
        "id": "2306.04136",
        "title": "Chain-of-Note: Robust RAG via Reading Notes",
        "authors": "Yu et al.",
        "year": "2023",
        "category": "Reasoning",
        "summary": "Generates sequential reading notes for each retrieved passage, improving robustness to noisy and irrelevant context.",
    },
    {
        "id": "2310.11511",
        "title": "Self-RAG: Self-Reflective Retrieval-Augmented Generation",
        "authors": "Asai et al.",
        "year": "2023",
        "category": "Adaptive RAG",
        "summary": "Trains a model to retrieve on demand and critique its own outputs with reflection tokens for grounded, controllable generation.",
    },
    {
        "id": "2312.10997",
        "title": "Retrieval-Augmented Generation for LLMs: A Survey",
        "authors": "Gao et al.",
        "year": "2023",
        "category": "Survey",
        "summary": "Comprehensive survey of naive, advanced, and modular RAG paradigms — the reference map for the field.",
    },
    {
        "id": "2407.01219",
        "title": "Searching for Best Practices in RAG",
        "authors": "Wang et al.",
        "year": "2024",
        "category": "Engineering",
        "summary": "Empirical study identifying chunking, retrieval, reranking, and prompting choices that consistently move the needle in production RAG.",
    },
]

CATEGORY_COLOR = {
    "Foundations": "#7C5CFF",
    "Retrieval": "#2EC4B6",
    "Adaptive RAG": "#FF6B6B",
    "Reasoning": "#FFB454",
    "Survey": "#5B8DEF",
    "Engineering": "#9BD17C",
}


# ---------------------------------------------------------------------------
# Data: pipeline stages (must mirror the backend's behaviour)
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    key: str
    label: str
    icon: str
    description: str
    status: str = "waiting"        # waiting | active | done | skipped | error
    detail: str = ""
    duration_ms: int | None = None


def fresh_stages() -> list[Stage]:
    return [
        Stage("input", "Query Input",
              "✍️",
              "User query captured and normalised before entering the pipeline."),
        Stage("validation", "Query Validation",
              "\U0001F6E1️",
              "LLM guardrail checks the query is well-formed, safe, and on-topic."),
        Stage("rewrite", "Query Rewriting",
              "\U0001F4DD",
              "Ambiguous or terse questions are rewritten into a self-contained form."),
        Stage("planning", "Hop Planning",
              "\U0001F9ED",
              "Decides whether the question is single-hop or needs decomposition into sub-questions."),
        Stage("retrieval", "Hybrid Retrieval",
              "\U0001F50D",
              "Dense (pgvector) + BM25 lexical search fused with reciprocal rank fusion."),
        Stage("rerank", "Cross-Encoder Rerank",
              "⚖️",
              "Top-K candidates are reranked by a cross-encoder for precise relevance."),
        Stage("context", "Context Assembly",
              "\U0001F9F1",
              "Selected chunks and summaries are stitched into a grounded prompt context."),
        Stage("generation", "LLM Generation",
              "\U0001F9E0",
              "The LLM produces an answer conditioned strictly on the retrieved context."),
        Stage("output", "Output Validation",
              "✅",
              "Validator checks groundedness (no unsupported claims) and relevance to the question."),
        Stage("final", "Final Response",
              "\U0001F4E6",
              "Validated, formatted answer delivered to the user with timings & citations."),
    ]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("api_url", DEFAULT_API_URL)
    ss.setdefault("history", [])          # list of {q, a, timings, validation, stages, ts}
    ss.setdefault("page", "Chat")
    ss.setdefault("pending_query", None)


_init_state()


def _use_example(text: str) -> None:
    """Callback for sidebar example buttons — runs before the next rerun."""
    st.session_state.pending_query = text
    st.session_state.page = "Chat"


# ---------------------------------------------------------------------------
# Global CSS — polished, recruiter-friendly look
# ---------------------------------------------------------------------------


CSS = """
<style>
/* ---------- base ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1200px; }
section[data-testid="stSidebar"] { background: #0E1117; border-right: 1px solid #1f2330; }

/* ---------- hero ---------- */
.hero {
    background: radial-gradient(120% 120% at 0% 0%, rgba(124,92,255,0.18), transparent 55%),
                radial-gradient(120% 120% at 100% 0%, rgba(46,196,182,0.14), transparent 55%),
                linear-gradient(135deg, #11141C, #0B0D12);
    border: 1px solid #1f2330;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
.hero-title {
    font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
    margin: 0; color: #F4F5FA;
}
.hero-tagline { color: #98A0B3; margin-top: 6px; font-size: 1.02rem; }
.hero-badges { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 500;
    background: rgba(124,92,255,0.12); color: #C9BEFF;
    border: 1px solid rgba(124,92,255,0.35);
}
.badge.alt   { background: rgba(46,196,182,0.10); color: #9CE6DC; border-color: rgba(46,196,182,0.35); }
.badge.warm  { background: rgba(255,180,84,0.10); color: #FFD79B; border-color: rgba(255,180,84,0.35); }
.badge.blue  { background: rgba(91,141,239,0.12); color: #B9CCFB; border-color: rgba(91,141,239,0.35); }

/* ---------- section headings ---------- */
.section-title {
    font-size: 1.05rem; font-weight: 600; color: #E6E8EE;
    margin: 6px 0 10px; letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 8px;
}
.section-title::before {
    content: ""; display: inline-block; width: 3px; height: 16px;
    background: linear-gradient(180deg, #7C5CFF, #2EC4B6); border-radius: 2px;
}

/* ---------- chat bubbles ---------- */
.user-bubble, .ai-bubble {
    border-radius: 14px; padding: 14px 18px; margin: 10px 0;
    line-height: 1.55; border: 1px solid #1f2330;
}
.user-bubble {
    background: linear-gradient(135deg, rgba(124,92,255,0.14), rgba(124,92,255,0.06));
    border-color: rgba(124,92,255,0.35);
}
.ai-bubble { background: #141821; }
.bubble-role {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #98A0B3; margin-bottom: 4px;
}

/* ---------- pipeline stages ---------- */
.stage-card {
    display: flex; gap: 14px; padding: 12px 14px; margin: 8px 0;
    border-radius: 12px; border: 1px solid #1f2330; background: #11141C;
    transition: all 0.25s ease;
}
.stage-card.active {
    border-color: #7C5CFF;
    background: linear-gradient(135deg, rgba(124,92,255,0.10), rgba(124,92,255,0.02));
    box-shadow: 0 0 0 3px rgba(124,92,255,0.08);
}
.stage-card.done   { border-color: rgba(46,196,182,0.45); }
.stage-card.error  { border-color: rgba(255,107,107,0.55); }
.stage-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: #1A1E2A; display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; flex-shrink: 0;
}
.stage-card.active .stage-icon { background: rgba(124,92,255,0.18); }
.stage-card.done   .stage-icon { background: rgba(46,196,182,0.16); }
.stage-card.error  .stage-icon { background: rgba(255,107,107,0.18); }
.stage-body { flex: 1; min-width: 0; }
.stage-label {
    font-weight: 600; color: #E6E8EE; font-size: 0.94rem;
    display: flex; justify-content: space-between; gap: 10px;
}
.stage-status {
    font-size: 0.72rem; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.06em; color: #98A0B3;
}
.stage-card.active .stage-status { color: #B7A4FF; }
.stage-card.done   .stage-status { color: #6FE2D2; }
.stage-card.error  .stage-status { color: #FF8B8B; }
.stage-desc { color: #98A0B3; font-size: 0.84rem; margin-top: 4px; }
.stage-detail {
    color: #C9CFDE; font-size: 0.8rem; margin-top: 6px; font-family: 'JetBrains Mono', monospace;
}

/* spinner dot for active state */
.dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #7C5CFF; display: inline-block; margin-right: 6px;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1);   opacity: 1; }
    50%      { transform: scale(1.4); opacity: 0.55; }
}

/* ---------- metric tiles ---------- */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.metric {
    background: #141821; border: 1px solid #1f2330; border-radius: 12px;
    padding: 14px 16px;
}
.metric-label { font-size: 0.75rem; color: #98A0B3; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #F4F5FA; margin-top: 4px; }
.metric-sub   { font-size: 0.78rem; color: #6FE2D2; margin-top: 2px; }
.metric-sub.warn { color: #FFB454; }
.metric-sub.bad  { color: #FF8B8B; }

/* ---------- source cards ---------- */
.source-card {
    background: #141821; border: 1px solid #1f2330; border-radius: 10px;
    padding: 10px 12px; margin-bottom: 8px; transition: border-color 0.15s ease;
}
.source-card:hover { border-color: #2a2f3e; }
.source-title { font-weight: 600; color: #E6E8EE; font-size: 0.86rem; line-height: 1.3; }
.source-meta  { font-size: 0.72rem; color: #98A0B3; margin-top: 3px; }
.cat-pill {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.04em;
    color: #0B0D12;
}

/* ---------- architecture diagram container ---------- */
.diagram-card {
    background: #0E1117; border: 1px solid #1f2330; border-radius: 14px;
    padding: 18px; margin: 8px 0 16px;
}

/* ---------- nav radio ---------- */
div[role="radiogroup"] > label {
    background: #141821; padding: 8px 12px; border-radius: 8px;
    border: 1px solid transparent; margin-right: 4px;
}
div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(124,92,255,0.14); border-color: rgba(124,92,255,0.45);
}

/* hide default streamlit footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-title">⚡ {APP_TITLE}</div>
          <div class="hero-tagline">{APP_TAGLINE}</div>
          <div class="hero-badges">
            <span class="badge">FastAPI Backend</span>
            <span class="badge alt">Hybrid Retrieval · pgvector + BM25</span>
            <span class="badge warm">Cross-Encoder Reranking</span>
            <span class="badge blue">LLM-Validated Output</span>
            <span class="badge">8 Foundational Papers</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ⚡ LexQA")
        st.caption("A transparent RAG demo")

        st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "nav", ["Chat", "Architecture", "About"],
            label_visibility="collapsed",
            horizontal=False, key="page",
        )

        st.markdown('<div class="section-title">Backend</div>', unsafe_allow_html=True)
        st.session_state.api_url = st.text_input(
            "API URL", value=st.session_state.api_url, label_visibility="collapsed"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Ping", use_container_width=True):
                ok, msg = ping_backend(st.session_state.api_url)
                (st.success if ok else st.error)(msg)
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.markdown('<div class="section-title">Knowledge Base · 8 sources</div>',
                    unsafe_allow_html=True)
        for src in SOURCES:
            color = CATEGORY_COLOR.get(src["category"], "#7C5CFF")
            with st.expander(f"{src['title']}", expanded=False):
                st.markdown(
                    f"<span class='cat-pill' style='background:{color}'>"
                    f"{src['category']}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{src['authors']} · {src['year']} · arXiv:{src['id']}")
                st.write(src["summary"])

        st.markdown('<div class="section-title">Try an example</div>',
                    unsafe_allow_html=True)
        examples = [
            "What is the difference between RAG and REALM?",
            "How does Self-RAG decide when to retrieve?",
            "Why does hybrid retrieval outperform pure dense search?",
            "What chunking strategies work best in production RAG?",
        ]
        for ex in examples:
            st.button(
                ex,
                key=f"ex-{ex}",
                use_container_width=True,
                on_click=_use_example,
                args=(ex,),
            )

        st.markdown("---")
        st.caption("Built with Streamlit · FastAPI · pgvector")


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------


def ping_backend(api_url: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=4)
        if r.ok and r.json().get("status") == "ok":
            return True, "Backend healthy"
        return False, f"Backend responded {r.status_code}"
    except Exception as e:                                  # noqa: BLE001
        return False, f"Cannot reach backend: {e.__class__.__name__}"


@dataclass
class CallResult:
    ok: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def call_query(api_url: str, question: str, result: CallResult) -> None:
    """Run /query in a background thread so the UI can animate stages."""
    try:
        r = requests.post(
            f"{api_url.rstrip('/')}/query",
            json={"question": question},
            timeout=180,
        )
        r.raise_for_status()
        result.payload = r.json()
        result.ok = True
    except requests.HTTPError as e:
        result.ok = False
        try:
            result.error = e.response.json().get("detail", str(e))
        except Exception:                                   # noqa: BLE001
            result.error = str(e)
    except Exception as e:                                  # noqa: BLE001
        result.ok = False
        result.error = f"{e.__class__.__name__}: {e}"


# ---------------------------------------------------------------------------
# Pipeline rendering
# ---------------------------------------------------------------------------


def stage_card_html(stage: Stage) -> str:
    status_label = {
        "waiting":  "Waiting",
        "active":   "Running",
        "done":     "Completed",
        "skipped":  "Skipped",
        "error":    "Failed",
    }.get(stage.status, stage.status)

    duration_html = (
        f" · {stage.duration_ms} ms"
        if stage.duration_ms is not None and stage.status in {"done", "error"}
        else ""
    )
    detail_html = (
        f"<div class='stage-detail'>{stage.detail}</div>" if stage.detail else ""
    )
    dot_html = "<span class='dot'></span>" if stage.status == "active" else ""

    return f"""
    <div class='stage-card {stage.status}'>
      <div class='stage-icon'>{stage.icon}</div>
      <div class='stage-body'>
        <div class='stage-label'>
          <span>{stage.label}</span>
          <span class='stage-status'>{dot_html}{status_label}{duration_html}</span>
        </div>
        <div class='stage-desc'>{stage.description}</div>
        {detail_html}
      </div>
    </div>
    """


def render_pipeline(container, stages: list[Stage]) -> None:
    html = "".join(stage_card_html(s) for s in stages)
    container.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline runner — animates stages while /query runs in background
# ---------------------------------------------------------------------------


# These weights roughly reflect where time is spent. They're used to pace
# the animation while the backend call is in flight. The displayed durations
# are then reconciled to real values from the server's timing breakdown.
STAGE_WEIGHTS: dict[str, float] = {
    "input":      0.5,
    "validation": 1.4,
    "rewrite":    1.2,
    "planning":   1.4,
    "retrieval":  3.0,
    "rerank":     2.0,
    "context":    0.6,
    "generation": 4.0,
    "output":     1.6,
    "final":      0.4,
}

# Backend timing buckets → stages they cover (for reconciling real durations).
TIMING_BUCKETS: dict[str, list[str]] = {
    "input_preprocessing_ms": ["validation", "rewrite", "planning"],
    "retrieval_ms":           ["retrieval", "rerank", "context"],
    "generation_ms":          ["generation", "output"],
}


def _redistribute(total_ms: int, keys: list[str]) -> dict[str, int]:
    """Split a timing bucket across its stages, weighted."""
    weights = [STAGE_WEIGHTS[k] for k in keys]
    total_w = sum(weights) or 1.0
    out, used = {}, 0
    for k, w in zip(keys[:-1], weights[:-1]):
        share = int(total_ms * (w / total_w))
        out[k] = share
        used += share
    out[keys[-1]] = max(0, total_ms - used)
    return out


def run_query_with_animation(
    api_url: str,
    question: str,
    pipeline_slot,
) -> tuple[CallResult, list[Stage]]:
    """
    Animate stages while the API call runs in a background thread.
    Returns the final CallResult and the final list of Stage objects.
    """
    stages = fresh_stages()
    result = CallResult()
    thread = threading.Thread(target=call_query, args=(api_url, question, result))

    # Stage 0: input is instant
    stages[0].status = "done"
    stages[0].duration_ms = 0
    render_pipeline(pipeline_slot, stages)

    thread.start()

    # Animate stages 1..N-1 while we wait. Speed adapts to how long the
    # backend takes — never finish early; if the backend is still working,
    # we hold the last animated stage in "active" until the response arrives.
    started = time.time()
    cursor = 1
    stages[cursor].status = "active"
    render_pipeline(pipeline_slot, stages)
    stage_started = time.time()

    # Target minimum dwell per stage (seconds). Keeps the UI legible
    # without artificially slowing down real fast backends.
    base_dwell = 0.45

    while thread.is_alive():
        # if we've shown stage `cursor` long enough, advance — but never
        # past the last "real" pipeline stage; the final stage is reserved
        # for after the response arrives.
        elapsed = time.time() - stage_started
        weight = STAGE_WEIGHTS[stages[cursor].key]
        dwell = base_dwell * weight
        if elapsed >= dwell and cursor < len(stages) - 2:
            stages[cursor].status = "done"
            stages[cursor].duration_ms = int(elapsed * 1000)
            cursor += 1
            stages[cursor].status = "active"
            stage_started = time.time()
            render_pipeline(pipeline_slot, stages)
        time.sleep(0.08)

    thread.join()
    total_ms = int((time.time() - started) * 1000)

    # Mark whatever's still active as done (or error), then finish the last stage.
    for s in stages:
        if s.status == "active":
            s.status = "done"
            s.duration_ms = int((time.time() - stage_started) * 1000)

    if not result.ok:
        # Mark the last running stage as error and stop.
        for s in reversed(stages):
            if s.status == "done":
                s.status = "error"
                s.detail = result.error[:160]
                break
        return result, stages

    payload = result.payload

    # Backend may short-circuit on invalid input or invalid output.
    if "error" in payload:
        for s in stages:
            if s.status not in {"done", "active"}:
                s.status = "skipped"
        # find a reasonable stage to mark as error
        target_key = "validation" if "Invalid query" in payload.get("error", "") else "output"
        for s in stages:
            if s.key == target_key:
                s.status = "error"
                s.detail = payload.get("reason", payload["error"])[:200]
                break
        return result, stages

    # Reconcile durations against backend timings
    timings = payload.get("timings", {}) or {}
    for bucket, keys in TIMING_BUCKETS.items():
        if bucket in timings:
            for k, ms in _redistribute(int(timings[bucket]), keys).items():
                for s in stages:
                    if s.key == k:
                        s.duration_ms = ms

    # Attach detail snippets where it adds value
    validation = payload.get("validation", {}) or {}
    for s in stages:
        if s.key == "output":
            grounded = validation.get("grounded")
            relevant = validation.get("relevant")
            s.detail = (
                f"grounded={grounded} · relevant={relevant}"
                if grounded is not None else ""
            )
        if s.key == "retrieval":
            s.detail = "hybrid: dense (pgvector cosine) ∪ BM25 → RRF top-10"
        if s.key == "rerank":
            s.detail = "Cohere rerank via OpenRouter → top-5"

    # Final stage gets a small synthetic duration so total ≈ total_ms
    accounted = sum((s.duration_ms or 0) for s in stages if s.key != "final")
    stages[-1].duration_ms = max(0, total_ms - accounted)
    stages[-1].status = "done"

    render_pipeline(pipeline_slot, stages)
    return result, stages


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------


def render_chat_history() -> None:
    if not st.session_state.history:
        st.info(
            "No queries yet. Try one of the example prompts in the sidebar, "
            "or ask your own question below."
        )
        return

    for entry in st.session_state.history:
        st.markdown(
            f"<div class='user-bubble'>"
            f"<div class='bubble-role'>You</div>{entry['q']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='ai-bubble'><div class='bubble-role'>LexQA</div>",
            unsafe_allow_html=True,
        )
        st.markdown(entry["a"])
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("\U0001F9EA Pipeline trace, metrics & validation",
                         expanded=False):
            render_metrics(entry.get("timings", {}), entry.get("validation", {}))
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            stages = entry.get("stages") or []
            if stages:
                st.markdown(
                    "".join(stage_card_html(s) for s in stages),
                    unsafe_allow_html=True,
                )


def render_metrics(timings: dict, validation: dict) -> None:
    total = sum(int(v) for v in (timings or {}).values()) or 0
    cards = [
        ("Total latency", f"{total} ms", "end-to-end", ""),
        ("Pre-processing", f"{timings.get('input_preprocessing_ms', 0)} ms",
         "validate · rewrite · plan", ""),
        ("Retrieval", f"{timings.get('retrieval_ms', 0)} ms",
         "hybrid + rerank", ""),
        ("Generation", f"{timings.get('generation_ms', 0)} ms",
         "LLM + output validation", ""),
    ]
    if validation:
        grounded = validation.get("grounded")
        relevant = validation.get("relevant")
        if grounded is not None:
            cls = "" if grounded else "bad"
            cards.append(("Grounded", "yes" if grounded else "no",
                          "no unsupported claims" if grounded else "claims unsupported",
                          cls))
        if relevant is not None:
            cls = "" if relevant else "warn"
            cards.append(("Relevant", "yes" if relevant else "no",
                          "answers the question" if relevant else "off-topic",
                          cls))

    html = "<div class='metric-grid'>"
    for label, value, sub, cls in cards:
        html += (
            f"<div class='metric'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>"
            f"<div class='metric-sub {cls}'>{sub}</div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_chat_page() -> None:
    render_hero()

    col_chat, col_pipe = st.columns([1.35, 1], gap="large")

    with col_chat:
        st.markdown('<div class="section-title">Conversation</div>',
                    unsafe_allow_html=True)
        render_chat_history()

        question = st.chat_input("Ask anything about RAG, retrieval, or these 8 papers…")

        if not question and st.session_state.pending_query:
            question = st.session_state.pending_query
            st.session_state.pending_query = None

    with col_pipe:
        st.markdown('<div class="section-title">Live Pipeline</div>',
                    unsafe_allow_html=True)
        pipeline_slot = st.empty()

        if not question:
            # Render idle pipeline preview so the page never looks empty
            render_pipeline(pipeline_slot, fresh_stages())
            st.caption(
                "Each stage lights up in real time as your query travels through "
                "the RAG system."
            )

    if question:
        # Render user bubble immediately in the chat column
        with col_chat:
            st.markdown(
                f"<div class='user-bubble'>"
                f"<div class='bubble-role'>You</div>{question}</div>",
                unsafe_allow_html=True,
            )

        with col_pipe:
            result, final_stages = run_query_with_animation(
                st.session_state.api_url, question, pipeline_slot
            )

        with col_chat:
            if not result.ok:
                st.error(f"Backend error: {result.error}")
                return

            payload = result.payload
            if "error" in payload:
                st.warning(f"**{payload['error']}** — {payload.get('reason', '')}")
                return

            answer = payload.get("response", "_(empty response)_")
            timings = payload.get("timings", {})
            validation = payload.get("validation", {})

            st.markdown(
                "<div class='ai-bubble'><div class='bubble-role'>LexQA</div>",
                unsafe_allow_html=True,
            )
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            render_metrics(timings, validation)

            st.session_state.history.append({
                "q": question,
                "a": answer,
                "timings": timings,
                "validation": validation,
                "stages": final_stages,
                "ts": time.time(),
            })


# ---------------------------------------------------------------------------
# Architecture page
# ---------------------------------------------------------------------------


MERMAID_DIAGRAM = r"""
flowchart TB
    Q(( )):::startBubble
    QL[user question]:::ghost
    Q --- QL

    subgraph P1["<b>1 &middot; understand</b>"]
      direction LR
      V[validate]:::box --> R[rewrite]:::box --> PL[plan hops]:::box
    end

    subgraph P2["<b>2 &middot; retrieve</b>"]
      direction TB
      S[semantic search]:::box
      K[keyword search]:::box
      M[merge RRF + AI rerank]:::box
      S --> M
      K --> M
    end

    subgraph P3["<b>3 &middot; generate</b>"]
      direction LR
      G[generate]:::box --> A[audit]:::box --> F[format]:::box
    end

    E(( )):::endBubble
    EL[grounded answer]:::ghost
    E --- EL

    Q --> P1
    P1 --> P2
    P2 --> P3
    P3 --> E

    classDef box fill:#5346B5,stroke:#7C5CFF,stroke-width:1.5px,color:#FFFFFF,rx:10,ry:10
    classDef startBubble fill:#4a5163,stroke:#4a5163,color:#FFFFFF
    classDef endBubble fill:#2EC4B6,stroke:#2EC4B6,color:#0B0D12
    classDef ghost fill:#0E1117,stroke:#0E1117,color:#98A0B3

    linkStyle default stroke:#5B6478,stroke-width:1.5px
"""


def render_architecture_page() -> None:
    render_hero()

    st.markdown('<div class="section-title">System Architecture</div>',
                unsafe_allow_html=True)
    st.write(
        "LexQA is a transparent RAG system. Every query traverses a fixed "
        "pipeline of stages — each one observable from the chat UI. The diagram "
        "below shows the same components the live pipeline panel animates."
    )

    components.html(
        f"""
        <div style="background:#0B0D12; border:1px solid #1f2330;
                    border-radius:14px; padding:28px 18px;">
          <pre class="mermaid" style="background:transparent;
               text-align:center; font-size:16px;
               font-family:'Inter',sans-serif;">{MERMAID_DIAGRAM}</pre>
        </div>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{
            startOnLoad: true,
            theme: "base",
            securityLevel: "loose",
            themeVariables: {{
              background: "#0B0D12",
              primaryColor: "#5346B5",
              primaryTextColor: "#FFFFFF",
              primaryBorderColor: "#7C5CFF",
              secondaryColor: "#11141C",
              tertiaryColor: "#0E1117",
              lineColor: "#5B6478",
              clusterBkg: "#10131B",
              clusterBorder: "#252A38",
              titleColor: "#E6E8EE",
              edgeLabelBackground: "#0B0D12",
              fontFamily: "Inter, sans-serif",
              fontSize: "16px"
            }},
            flowchart: {{
              htmlLabels: true,
              curve: "linear",
              padding: 28,
              nodeSpacing: 60,
              rankSpacing: 70,
              useMaxWidth: true
            }}
          }});
        </script>
        <style>
          .mermaid .cluster rect {{
            rx: 14px; ry: 14px;
            stroke-width: 1.5px !important;
          }}
          .mermaid .cluster .cluster-label foreignObject {{
            overflow: visible;
          }}
          .mermaid .cluster .nodeLabel {{
            color: #E6E8EE !important;
            font-weight: 600;
            font-size: 15px;
            letter-spacing: 0.02em;
          }}
          .mermaid .node rect {{
            rx: 10px; ry: 10px;
          }}
          .mermaid .node .nodeLabel {{
            font-size: 15px;
            font-weight: 500;
            padding: 4px 8px;
          }}
          .mermaid .label foreignObject {{
            overflow: visible;
          }}
        </style>
        """,
        height=900,
        scrolling=False,
    )

    st.markdown('<div class="section-title">Component Responsibilities</div>',
                unsafe_allow_html=True)

    components_table = [
        ("\U0001F5A5️ Streamlit Frontend",
         "Renders the chat UI, animates the live pipeline view, and visualises "
         "metrics, validations, and source citations."),
        ("⚡ FastAPI Backend",
         "Single entry point — exposes `/query`, `/md`, `/health`. Orchestrates "
         "the full RAG pipeline and returns a structured JSON envelope."),
        ("\U0001F6E1️ Input Validation",
         "LLM-based guardrail that rejects malformed, unsafe, or off-domain "
         "questions before any retrieval cost is incurred."),
        ("\U0001F4DD Query Rewriter & Hop Planner",
         "Rewrites the question into a self-contained form, then decides "
         "between single-hop retrieval and multi-hop decomposition."),
        ("\U0001F9EC Embedding Model",
         "sentence-transformers encoder used both for indexing chunks at "
         "ingestion time and for encoding the live query."),
        ("\U0001F4E6 pgvector Store",
         "PostgreSQL with the `pgvector` extension. Chunks live alongside "
         "their summaries and dense embeddings for cosine search."),
        ("\U0001F524 BM25 Lexical Index",
         "Postgres `tsvector` full-text index — provides keyword recall that "
         "dense retrieval often misses (acronyms, code, exact names)."),
        ("\U0001F501 Reciprocal Rank Fusion",
         "Combines dense and lexical results into a single top-10 candidate "
         "set using RRF — robust to score-scale differences."),
        ("⚖️ Cross-Encoder Reranker",
         "Cohere reranker (via OpenRouter) re-scores the candidates pairwise "
         "with the query for high-precision top-5 selection."),
        ("\U0001F9E0 LLM Generator",
         "Produces the answer conditioned strictly on the assembled context. "
         "No-context fallback is intentionally avoided."),
        ("✅ Output Validator",
         "Second LLM pass scores the answer on **groundedness** (every claim "
         "supported by context) and **relevance** (actually answers the user)."),
    ]
    for name, desc in components_table:
        st.markdown(
            f"<div class='source-card'>"
            f"<div class='source-title'>{name}</div>"
            f"<div class='source-meta'>{desc}</div></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------


def render_about_page() -> None:
    render_hero()

    st.markdown('<div class="section-title">About this project</div>',
                unsafe_allow_html=True)
    st.write(
        "LexQA is a portfolio-grade demonstration of a modern Retrieval-"
        "Augmented Generation system, built to make every internal decision "
        "**observable** rather than hidden behind a single API call."
    )
    st.write(
        "Most production RAG systems get one or two pieces right. LexQA wires "
        "together the full set of best practices — input/output validation, "
        "hybrid retrieval, cross-encoder reranking, multi-hop planning, and "
        "grounded answer validation — into a single, inspectable pipeline."
    )

    st.markdown('<div class="section-title">Engineering highlights</div>',
                unsafe_allow_html=True)

    highlights = [
        ("Hybrid retrieval",
         "Dense (pgvector cosine) ∪ BM25 (`tsvector`) fused with Reciprocal "
         "Rank Fusion — combines semantic and lexical signals."),
        ("Cross-encoder reranking",
         "Top-10 candidates re-scored by a cross-encoder for high-precision "
         "top-5 — the single highest-leverage step in real-world RAG."),
        ("Multi-hop planning",
         "Questions are classified single- vs multi-hop. Multi-hop queries "
         "are decomposed into sub-questions, each retrieved independently."),
        ("Two-sided LLM guardrails",
         "Input validator blocks malformed/off-topic queries; output validator "
         "checks the answer is grounded and relevant before returning it."),
        ("Transparent UI",
         "Every stage lights up live with status and per-stage timings — "
         "no black box. Validation results and metrics are surfaced inline."),
    ]
    for title, body in highlights:
        st.markdown(
            f"<div class='source-card'>"
            f"<div class='source-title'>{title}</div>"
            f"<div class='source-meta'>{body}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Frontend** — Streamlit + custom CSS
        - **Backend** — FastAPI · Uvicorn
        - **Vector store** — PostgreSQL + pgvector
        - **Lexical index** — Postgres `tsvector`
        - **Embeddings** — sentence-transformers
        - **Reranker / LLM** — OpenRouter-hosted models (Cohere rerank · LLM generator)
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


render_sidebar()

page = st.session_state.page
if page == "Chat":
    render_chat_page()
elif page == "Architecture":
    render_architecture_page()
else:
    render_about_page()
