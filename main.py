"""
CareHandoff AI — FastAPI application entry point.
Serves:
  - REST API at /api/*
  - Gradio Clinical Q&A at /qa
  - React SPA at all other paths (from frontend/dist)
"""
import asyncio
import contextlib
import logging
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.state import get_state, initialize_state
from gradio_qa_app import create_qa_blocks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_DIST = Path(__file__).parent / "frontend" / "dist"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    api_key  = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE") or "(standard OpenAI)"
    logger.info("[Startup] OPENAI_API_KEY present: %s", bool(api_key))
    logger.info("[Startup] OPENAI_API_BASE: %s", api_base)
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Go to Render → your service → Environment tab and add it."
        )
    logger.info("[Startup] Initialising CareHandoff AI…")
    # Run in a thread so the RxNorm connectivity ping (5 s timeout) inside
    # AgenticRAGOrchestrator.__init__ doesn't block the event loop and cause
    # Render's health check to time out.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_state)

    async def _ingest_background() -> None:
        """Embed and persist all data after the server is already up."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, get_state().orchestrator.ensure_data_ingested
            )
            if result.get("notes_ingested"):
                logger.info("[Ingest] Discharge notes ingested.")
            if result.get("guidelines_ingested"):
                logger.info("[Ingest] Guidelines ingested.")
            logger.info("[Ingest] Background ingestion complete.")
        except Exception as exc:
            logger.error("[Ingest] Background ingestion failed: %s", exc)

    asyncio.create_task(_ingest_background())
    logger.info("[Startup] Ready. Data ingestion running in background…")
    yield
    logger.info("[Shutdown] Done.")


app = FastAPI(
    title="CareHandoff AI",
    description="Agentic RAG system for clinical discharge documentation.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API
app.include_router(router, prefix="/api")

# /qa (no trailing slash) → redirect to /qa/ so Starlette's Mount matches correctly.
# Mount("/qa", ...) only gives a full match for "/qa/" and "/qa/*", not bare "/qa".
@app.get("/qa", include_in_schema=False)
async def redirect_qa():
    return RedirectResponse(url="/qa/", status_code=301)

# Explicit root handler registered before Gradio mount.
# /{full_path:path} does NOT match bare "/" in Starlette, so without this
# Render's HEAD / port probe returns 405 and the deploy is killed.
@app.get("/", include_in_schema=False)
async def serve_root():
    if _DIST.exists():
        return FileResponse(str(_DIST / "index.html"))
    return {"status": "ok"}

# Gradio Clinical Q&A — blocks created lazily (state resolved at request time)
_qa_blocks = create_qa_blocks()
app = gr.mount_gradio_app(app, _qa_blocks, path="/qa")

# React SPA static files (production build)
if _DIST.exists():
    _assets = _DIST / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(str(_DIST / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def dev_root():
        return {
            "message": (
                "API is running. "
                "Start the Vite dev server: cd frontend && npm run dev"
            )
        }
