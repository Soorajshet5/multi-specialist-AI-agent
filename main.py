from __future__ import annotations
"""
main.py
FastAPI entry point for the Elite AI Agent Service.

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /run-agent          — main agent invocation
    GET  /health             — liveness probe
    GET  /docs               — Swagger UI (auto-generated)
"""
import os
from dotenv import load_dotenv

load_dotenv()  # This looks for your .env file and loads the keys

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from graph import langgraph_app
from models import UserRequest

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Elite AI Agent Service starting up...")
    yield
    logger.info("Elite AI Agent Service shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Elite AI Agent Service",
    description=(
        "Production-grade multi-specialist LangGraph agent with "
        "Groq LLM, Alpha Vantage market data, and circuit-breaker self-healing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.1f}"
    return response


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health_check():
    """Liveness probe — returns 200 if the service is up."""
    return {"status": "ok"}


@app.post("/run-agent", tags=["agent"])
async def run_agent(request: UserRequest):
    """
    Main agent invocation endpoint.
    LangGraph ainvoke returns a plain dict — accessed with .get()
    """
    logger.info("Received task from user '%s': %s", request.user_id, request.task)

    from models import AgentState
    initial_state = AgentState(task=request.task)

    config = {"configurable": {"thread_id": request.user_id}}

    try:
        final_state: dict = await langgraph_app.ainvoke(
            initial_state,
            config=config,
        )
    except Exception as exc:
        logger.error("LangGraph invocation failed for user '%s': %s", request.user_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")

    circuit_broken = final_state.get("circuit_broken", False)
    status = "fallback" if circuit_broken else "success"

    logger.info(
        "Task completed | user=%s | status=%s | errors=%d",
        request.user_id,
        status,
        final_state.get("error_count", 0),
    )

    return {
        "status": status,
        "result": final_state.get("final_output"),
        "error_count": final_state.get("error_count", 0),
        "circuit_broken": circuit_broken,
        "user_id": request.user_id,
    }