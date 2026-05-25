# ── LexQA backend ──────────────────────────────────────────────────────────
# The embedding model (all-MiniLM-L6-v2, ~90 MB) is downloaded ONCE at
# build time and baked into the image layer.  On Render's free tier the
# disk resets on every cold start, but the image layer is preserved, so
# the model is already present every time the container wakes up.
# ───────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Tell HuggingFace / sentence-transformers where to cache models.
# This must be the same path at build time and at runtime.
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence_transformers

WORKDIR /app

# Install dependencies first (layer is cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model ───────────────────────────────────────
# Running this at build time bakes the weights into the image.
# The model is ~90 MB; the layer is cached by Render's build system so
# rebuilds only re-download if you change the model name.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model cached successfully.')"

# Copy application code (after the model download so that code changes
# don't invalidate the heavy model-download layer)
COPY . .

# Shell form so $PORT is expanded at runtime.
# Render injects $PORT automatically; fallback to 8000 for local dev.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
