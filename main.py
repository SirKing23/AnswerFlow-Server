import sys
import os

# Ensure project root is in Python path so all module imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import ingest

app = FastAPI(
    title="RAG Processing Server",
    description="Handles file parsing, chunking, and embedding for Supabase RAG pipeline",
    version="1.0.0",
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Render uses this endpoint to verify the service is running.
    Configure in Render dashboard: Health Check Path = /health
    """
    return {"status": "ok"}


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[server] Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ── Entry point for local dev ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
