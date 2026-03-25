import sys
import os

# Ensure project root is in Python path so all module imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import processDocuments, processChats, agent

app = FastAPI(
    title="RAG Processing Server",
    description="Handles file parsing, chunking, embedding and agentic RAG",
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this to your actual domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(processDocuments.router)
#app.include_router(processChats.router)
app.include_router(agent.router)


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
