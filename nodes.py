"""
nodes.py
All LangGraph node functions for the Elite AI Agent.

Node contract: every function receives AgentState, mutates it in place,
and returns the same state object. LangGraph handles serialisation.
"""

from __future__ import annotations

import logging
import os
import io

import instructor
import polars as pl
import requests
from groq import Groq

from models import AgentState, ReasoningPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client initialisation (done once at import time)
# ---------------------------------------------------------------------------

_groq_raw = Groq(api_key=os.environ["GROQ_API_KEY"])
_client = instructor.from_groq(_groq_raw, mode=instructor.Mode.TOOLS)

_GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Supervisor node — routes to the right specialist
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> AgentState:
    """
    Reads the task and sets a routing hint in the plan.
    Real routing is done via add_conditional_edges in graph.py,
    but this node is also the place to do auth checks, rate-limit guards, etc.
    """
    task_lower = state.task.lower()
    logger.info("Supervisor received task: %s", state.task)

    if state.circuit_broken:
        logger.warning("Circuit already broken — short-circuiting supervisor.")
        return state

    # Inject a routing tag at position 0 so nodes can inspect intent if needed
    if "stock" in task_lower:
        state.plan = ["__route:market_analyst__"] + state.plan
    elif "tax" in task_lower or "compliance" in task_lower:
        state.plan = ["__route:compliance_expert__"] + state.plan
    else:
        state.plan = ["__route:general_assistant__"] + state.plan

    return state


def supervisor_router(state: AgentState) -> str:
    """
    Conditional-edge function called by LangGraph after supervisor_node.
    Returns the name of the next node as a string.
    """
    if state.circuit_broken:
        return "__end__"
    if state.plan and state.plan[0].startswith("__route:"):
        route = state.plan.pop(0).replace("__route:", "").replace("__", "")
        return route
    return "general_assistant"


# ---------------------------------------------------------------------------
# Reasoning node — generates a step-by-step plan via Groq + Instructor
# ---------------------------------------------------------------------------

def reasoning_node(state: AgentState) -> AgentState:
    """
    Calls Groq with structured output (ReasoningPlan) to decompose the task
    into a concrete list of executable steps.
    """
    if state.circuit_broken:
        return state

    logger.info("Reasoning node: planning for task '%s'", state.task)

    try:
        response: ReasoningPlan = _client.chat.completions.create(
            model=_GROQ_MODEL,
            response_model=ReasoningPlan,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite AI planner. Break the user's task into "
                        "3-5 concrete, executable steps. Be specific and brief. "
                        "Each step must be a single actionable sentence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Plan the steps for: {state.task}",
                },
            ],
        )
        state.plan = response.steps
        logger.info(
            "Plan generated (%d steps, ~%s ms): %s",
            len(state.plan),
            response.estimated_time_ms or "unknown",
            state.plan,
        )
    except Exception as exc:
        logger.error("Reasoning node failed: %s", exc)
        state.final_output = f"Error in reasoning: {exc}"

    return state


# ---------------------------------------------------------------------------
# Market data helper
# ---------------------------------------------------------------------------

def _fetch_market_data(ticker: str) -> dict:
    """
    Fetches daily OHLCV data from Alpha Vantage and computes the 50-day MA
    using Polars lazy evaluation (Rust-layer optimisation).
    Returns the most recent row as a dict.
    """
    api_key = os.environ["ALPHA_VANTAGE_KEY"]
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        f"&apikey={api_key}"
        f"&datatype=csv"
        f"&outputsize=compact"  # last 100 days — enough for MA50
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # Polars lazy scan from in-memory bytes (no temp file needed)
    csv_bytes = io.BytesIO(response.content)
    df = pl.read_csv(csv_bytes)

    # Validate expected columns exist
    required = {"timestamp", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Alpha Vantage response missing columns: {missing}. Got: {df.columns}")

    analysis = (
        df.lazy()
        .sort("timestamp", descending=True)          # newest first
        .with_columns(
            pl.col("close")
            .cast(pl.Float64)
            .rolling_mean(window_size=50)
            .alias("MA50")
        )
        .limit(1)
        .collect()
    )

    return analysis.to_dicts()[0]


def _extract_ticker(step: str, task: str) -> str:
    """
    Lightweight ticker extractor: ask Groq to identify the ticker symbol
    from the current step or original task. Falls back to SBIN.BSE.
    """
    try:
        response = _groq_raw.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial data assistant. Extract the stock ticker symbol "
                        "from the user message. Return ONLY the ticker (e.g. SBIN.BSE, RELIANCE.BSE, INFY.NSE). "
                        "If you cannot identify one, return SBIN.BSE."
                    ),
                },
                {"role": "user", "content": f"Step: {step}\nTask: {task}"},
            ],
            max_tokens=20,
        )
        ticker = response.choices[0].message.content.strip().upper()
        # Sanitise: only alphanumeric + dot
        ticker = "".join(c for c in ticker if c.isalnum() or c == ".")
        return ticker or "SBIN.BSE"
    except Exception:
        return "SBIN.BSE"


# ---------------------------------------------------------------------------
# Market analyst execution node
# ---------------------------------------------------------------------------

def market_analyst_node(state: AgentState) -> AgentState:
    """
    Specialist node: fetches real market data and computes the MA50.
    Called when supervisor routes to 'market_analyst'.
    """
    if state.circuit_broken or not state.plan:
        return state

    current_step = state.plan.pop(0) if state.plan else state.task

    try:
        ticker = _extract_ticker(current_step, state.task)
        logger.info("Market analyst: fetching data for %s", ticker)
        data = _fetch_market_data(ticker)
        state.market_data = data

        close = data.get("close", "N/A")
        ma50 = data.get("MA50")
        ma50_str = f"{ma50:.2f}" if ma50 is not None else "N/A (< 50 days data)"

        state.final_output = (
            f"[Market Analyst] {ticker} | "
            f"Latest close: {close} | "
            f"50-day MA: {ma50_str} | "
            f"Signal: {'ABOVE MA50' if ma50 and float(close) > ma50 else 'BELOW MA50'}"
        )
        logger.info("Market data result: %s", state.final_output)

    except Exception as exc:
        logger.error("Market analyst node failed: %s", exc)
        state.final_output = f"Error fetching market data: {exc}"

    return state


# ---------------------------------------------------------------------------
# Compliance expert node (stub — extend with real rules engine)
# ---------------------------------------------------------------------------

def compliance_expert_node(state: AgentState) -> AgentState:
    """
    Specialist node for tax / compliance queries.
    Currently a structured stub — extend with a rules engine or RAG retrieval.
    """
    if state.circuit_broken or not state.plan:
        return state

    current_step = state.plan.pop(0) if state.plan else state.task

    try:
        response = _groq_raw.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Indian tax and compliance expert. "
                        "Provide concise, accurate guidance. "
                        "Always add: 'Consult a CA for personalised advice.'"
                    ),
                },
                {"role": "user", "content": f"Step: {current_step}\nTask: {state.task}"},
            ],
            max_tokens=300,
        )
        state.final_output = (
            f"[Compliance Expert] {response.choices[0].message.content.strip()}"
        )
    except Exception as exc:
        logger.error("Compliance node failed: %s", exc)
        state.final_output = f"Error in compliance node: {exc}"

    return state


# ---------------------------------------------------------------------------
# General assistant execution node
# ---------------------------------------------------------------------------

def general_assistant_node(state: AgentState) -> AgentState:
    """
    Fallback execution node for all non-specialist tasks.
    Executes the first step in the plan using Groq.
    """
    if state.circuit_broken or not state.plan:
        return state

    current_step = state.plan.pop(0)
    logger.info("General assistant executing step: %s", current_step)

    try:
        response = _groq_raw.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an elite AI assistant. Execute the given step precisely and concisely.",
                },
                {"role": "user", "content": current_step},
            ],
            max_tokens=500,
        )
        result = response.choices[0].message.content.strip()
        state.final_output = f"[General] {result}"
    except Exception as exc:
        logger.error("General assistant node failed: %s", exc)
        state.final_output = f"Error during execution: {exc}"

    return state


# ---------------------------------------------------------------------------
# Validation node — circuit breaker
# ---------------------------------------------------------------------------

def validation_node(state: AgentState) -> AgentState:
    """
    Checks the output of the previous node.
    Increments error_count on failure; trips circuit_broken at 3 errors.
    Pydantic's ge=0 / le=3 constraint on error_count acts as a hard guardrail.
    """
    if state.final_output and "error" in state.final_output.lower():
        try:
            state.error_count = state.error_count + 1  # triggers Pydantic validation
        except Exception:
            # If le=3 constraint fires, force the circuit open
            state.circuit_broken = True

    if state.error_count >= 3:
        state.circuit_broken = True
        logger.warning("Circuit breaker OPEN after %d errors.", state.error_count)

    return state


# ---------------------------------------------------------------------------
# Conditional edge router after validation
# ---------------------------------------------------------------------------

def validation_router(state: AgentState) -> str:
    """
    Decides what happens after validation:
    - Circuit broken → end
    - Success → end
    - Remaining steps → back to reasoning for a new plan
    """
    if state.circuit_broken:
        return "__end__"
    if state.final_output and "error" not in state.final_output.lower():
        return "__end__"
    return "reasoning"