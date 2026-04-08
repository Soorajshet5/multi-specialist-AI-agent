# 🤖 Multi-Specialist AI Agent System

A **production-grade multi-agent orchestration system** built with LangGraph, FastAPI, and Groq. A supervisor routes incoming requests to specialized agents — Market Analyst, Code Assistant, or General AI — each backed by structured Pydantic outputs and real-time NSE/BSE data.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.x-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-async-green?logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🏗️ Architecture

```
User Request
     │
     ▼
┌─────────────────┐
│   FastAPI API   │  ← Async endpoints, Pydantic v2 validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Supervisor   │  ← LangGraph StateGraph node
│  (Task Router)  │    Routes based on intent classification
└────────┬────────┘
         │
   ┌─────┼─────┐
   ▼     ▼     ▼
┌──────┐ ┌──────┐ ┌──────┐
│Market│ │Code  │ │General│
│Analyst│ │Asst  │ │  AI  │
└──┬───┘ └──┬───┘ └──┬───┘
   │         │        │
   ▼         ▼        ▼
Alpha      Groq     Groq
Vantage  (LLaMA   (LLaMA
  API    3.3-70b) 3.3-70b)
   │
   ▼
Polars (lazy evaluation)
NSE/BSE live price data

         │
         ▼
┌─────────────────┐
│ Circuit Breaker │  ← Self-heals after 3 failures, re-routes
│  + Checkpointer │  ← MemorySaver per user session
└─────────────────┘
```

**Key design decisions:**
- `StateGraph` with 5 nodes: supervisor → specialist → reasoning → execution → validation
- Circuit breaker catches node failures, increments error count, re-routes without crashing the API
- `MemorySaver` checkpointing maintains per-user conversation state across requests
- `Instructor` library enforces structured Pydantic outputs from LLM responses

---

## 📊 Real Benchmark Results

| Task | Agent | Latency |
|---|---|---|
| Fetch SBI live price (NSE) | Market Analyst | **2.7s** |
| Explain Python decorators with code | Code Assistant | **1.4s** |
| Full pipeline (supervisor → execution → validation) | All nodes | **~4.1s** |

Live price fetched: **SBI ₹1019.45** via Alpha Vantage → Polars pipeline

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph `StateGraph` |
| LLM backbone | Groq API — `llama-3.3-70b-versatile` |
| Structured outputs | Instructor + Pydantic v2 |
| Market data | Alpha Vantage API (NSE/BSE) |
| Data processing | Polars (lazy evaluation via BytesIO) |
| API layer | FastAPI (async) |
| Validation | Pydantic v2 models |

---

## 🐛 Hardest Bugs Fixed

**1. Pydantic v2 immutability blocking state mutations**
LangGraph's `StateGraph` passes state as a frozen Pydantic model. Direct attribute assignment raises `ValidationError`. Fix: use `.model_copy(update={...})` pattern instead of mutating in place.

**2. `ainvoke` returning `dict` not `AgentState`**
`graph.ainvoke()` returns a raw dict, not your typed `AgentState`. Wrapping the return in `AgentState(**result)` breaks if extra keys exist. Fix: use `TypedDict` for state instead of Pydantic BaseModel — LangGraph merges cleanly without type conflicts.

**3. Polars can't lazy-scan URLs directly**
`pl.scan_csv("https://...")` fails silently. Fix: fetch raw bytes with `requests`, wrap in `BytesIO`, then pass to `pl.read_csv()` — then convert to lazy frame for downstream operations.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Alpha Vantage API key (free at [alphavantage.co](https://www.alphavantage.co))

### Install

```bash
git clone https://github.com/Soorajshet5/multi-specialist-AI-agent.git
cd multi-specialist-AI-agent

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Add your API keys to .env
```

```env
GROQ_API_KEY=your_groq_key_here
ALPHA_VANTAGE_API_KEY=your_av_key_here
```

### Run

```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 📁 Project Structure

```
multi-specialist-AI-agent/
├── main.py          # FastAPI app, async endpoints
├── graph.py         # LangGraph StateGraph definition
├── nodes.py         # Supervisor + specialist agent nodes
├── models.py        # Pydantic v2 state and response models
├── index.html       # Frontend UI
├── requirements.txt
└── .env.example
```

---

## 💡 How It Works

1. **Request comes in** via FastAPI POST endpoint
2. **Supervisor node** classifies intent → routes to Market Analyst, Code Assistant, or General AI
3. **Specialist node** executes — fetches live data or calls Groq LLM
4. **Validation node** checks output against Pydantic schema
5. **Circuit breaker** intercepts failures — after 3 errors, re-routes and logs without crashing
6. **MemorySaver** checkpoints the session state — context persists across turns per user

---

## 🛠️ Self-Healing Circuit Breaker

The most interesting part of this system. When any node fails:

```python
if agent_state.error_count >= 3:
    # re-route to fallback node
    return route_to_fallback(state)
else:
    state = state.model_copy(update={"error_count": state.error_count + 1})
    raise NodeException(...)
```

The API never returns a 500. Failures are caught, counted, and handled — exactly how production systems behave.

---

## 📈 Built For Production, Not Demos

This isn't a notebook demo. It's designed around real failure modes:
- State immutability in multi-node graphs
- Async API calls with proper error boundaries
- Lazy data evaluation to avoid loading full datasets into memory
- Structured LLM outputs that don't hallucinate schema

---

## 🤝 Connect

Built by **Sooraj** — ML Engineer in training, building in public.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Soorajshet5)

---

*Star ⭐ the repo if this helped you understand LangGraph state management or multi-agent design.*
