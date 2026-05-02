import sys
import os
import logging
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import processDocuments, processChats, agent

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: verify Neo4j connection then bootstrap schema.
    Shutdown: cleanly close the Neo4j driver.
    """
    try:
        from utils.neo4j_client import verify_connection, bootstrap_schema, close_neo4j
        ok = await verify_connection(timeout=15.0)
        if ok:
            await bootstrap_schema()
            log.info("[startup] Neo4j schema ready")
        else:
            log.warning(
                "[startup] Neo4j unreachable — graph features disabled until fixed.\n"
                "  Vector search (pgvector) is unaffected and fully operational."
            )
    except Exception as e:
        log.warning(f"[startup] Neo4j bootstrap failed (non-fatal): {e}")

    yield   # ← server runs here

    try:
        from utils.neo4j_client import close_neo4j
        await close_neo4j()
        log.info("[shutdown] Neo4j driver closed")
    except Exception:
        pass


app = FastAPI(
    title="RAG Processing Server",
    description="Handles file parsing, chunking, embedding and agentic RAG",
    version="2.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your actual domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API/Routers ───────────────────────────────────────────────────────────────────
app.include_router(processDocuments.router)
app.include_router(agent.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"[server] Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ── Entry point for local dev ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)