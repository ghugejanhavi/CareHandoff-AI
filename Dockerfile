# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --prefer-offline
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python base (deps + source, shared by stages 3 & 4) ──────────────
FROM python:3.11-slim AS python-base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# ── Stage 3: Pre-build vector store ───────────────────────────────────────────
# Ingestion runs here at build time so cold starts are instant.
# The API key is confined to this intermediate layer and does NOT appear
# in the final runtime image.
FROM python-base AS ingestion-builder
ARG OPENAI_EMBEDDING_KEY
ARG OPENAI_API_KEY
RUN mkdir -p chromadb_data data/raw/mimic && \
    OPENAI_EMBEDDING_KEY="${OPENAI_EMBEDDING_KEY}" \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    python -c "\
from orchestrator import AgenticRAGOrchestrator; \
o = AgenticRAGOrchestrator(); \
result = o.ensure_data_ingested(); \
print('Pre-ingestion complete:', result)"

# ── Stage 4: Final runtime image ──────────────────────────────────────────────
FROM python-base AS runtime
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
# Bake pre-built vector store — API key NOT present in this stage
COPY --from=ingestion-builder /app/chromadb_data ./chromadb_data
RUN mkdir -p data/raw/mimic
EXPOSE 8000
# Render injects $PORT at runtime; fall back to 8000 for local dev.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
