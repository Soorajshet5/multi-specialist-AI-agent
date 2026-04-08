"""
models.py
Pydantic state & schema definitions for the Elite AI Agent.
All models use validate_assignment=True so LangGraph node mutations work.
"""

from __future__ import annotations

from typing import Annotated, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """
    Central state object threaded through every LangGraph node.
    Immutability is disabled so nodes can mutate fields directly.
    """

    model_config = ConfigDict(validate_assignment=True)

    task: str
    plan: List[str] = []
    error_count: Annotated[int, Field(ge=0, le=3)] = 0
    final_output: Optional[str] = None
    circuit_broken: bool = False
    # Stores raw market data for downstream nodes
    market_data: Optional[dict] = None


class ReasoningPlan(BaseModel):
    """
    Structured output returned by the Groq/Instructor reasoning call.
    'steps' is a list of concrete action strings the execution node will pop off.
    'estimated_time_ms' is optional metadata — logged but not acted on.
    """

    model_config = ConfigDict(validate_assignment=True)

    steps: List[str]
    estimated_time_ms: Optional[int] = None


class UserRequest(BaseModel):
    """FastAPI request body schema with validation."""

    task: str = Field(..., min_length=3, max_length=500)
    user_id: str = Field(..., min_length=1, max_length=64)