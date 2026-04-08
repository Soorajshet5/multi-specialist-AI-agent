"""
graph.py
Builds and compiles the LangGraph StateGraph.
Imported by main.py — compiled once at startup, reused across all requests.

Graph topology:
    START
      └─► supervisor ──(conditional)──► market_analyst ─┐
                                    └─► compliance_expert─┤
                                    └─► general_assistant ┤
                                                          │
                                                    reasoning ◄─(retry loop)
                                                          │
                                                    validation
                                                          │
                                              (conditional)──► END
                                                          └──► reasoning (retry)
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from models import AgentState
from nodes import (
    compliance_expert_node,
    general_assistant_node,
    market_analyst_node,
    reasoning_node,
    supervisor_node,
    supervisor_router,
    validation_node,
    validation_router,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Constructs the full agent graph with:
    - Multi-specialist routing via supervisor
    - Self-healing retry loop via validation
    - Circuit breaker (3-strike, then END)
    - MemorySaver checkpointing (per thread_id / user_id)
    """
    workflow = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node("compliance_expert", compliance_expert_node)
    workflow.add_node("general_assistant", general_assistant_node)
    workflow.add_node("validation", validation_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.add_edge(START, "supervisor")

    # ── Supervisor → specialist (conditional) ─────────────────────────────────
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "market_analyst": "market_analyst",
            "compliance_expert": "compliance_expert",
            "general_assistant": "reasoning",   # general path still uses planner
            "__end__": END,
        },
    )

    # ── Specialists → validation ──────────────────────────────────────────────
    workflow.add_edge("market_analyst", "validation")
    workflow.add_edge("compliance_expert", "validation")

    # ── Reasoning → general_assistant → validation ───────────────────────────
    workflow.add_edge("reasoning", "general_assistant")
    workflow.add_edge("general_assistant", "validation")

    # ── Validation → retry loop or END (conditional) ─────────────────────────
    workflow.add_conditional_edges(
        "validation",
        validation_router,
        {
            "reasoning": "reasoning",
            "__end__": END,
        },
    )

    # ── Compile with in-memory checkpointing ─────────────────────────────────
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled successfully.")
    return app


# Module-level singleton — built once, shared across all FastAPI requests
langgraph_app = build_graph()