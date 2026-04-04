"""
FastAPI application for the Nawaloka Health Assistant Multi-Agent System.

Async endpoints use LangGraph's ainvoke() / astream() so the event loop
is never blocked during LLM calls — multiple concurrent requests are
handled efficiently.

Start:
    cd src && uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs   (Swagger)
    http://localhost:8000/redoc  (ReDoc)
"""

import asyncio
import json
import time
import sys
import os

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.schemas import (
    ChatRequest, ChatResponse,
    HealthResponse, ToolStatus,
    MemoryResponse, FactItem,
    MemoryClearRequest, MemoryClearResponse,
    GraphResponse, GraphEdge,
    ErrorResponse,
)


# ── Lifespan — build agent once at startup ────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise the agent on startup (expensive: DB connections, LLM clients,
    Qdrant vectors).  Runs build_agent() in a thread so it doesn't block
    the event loop.
    """
    logger.info("Starting Nawaloka Health Assistant API...")

    from agents.orchestrator import build_agent

    # build_agent() is sync (DB connections, Qdrant init) — run in thread pool
    agent = await asyncio.to_thread(
        build_agent, enable_crm=True, enable_rag=True, enable_web=True
    )

    app.state.agent = agent
    logger.success("Agent ready — API is live")

    yield  # app runs here

    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Nawaloka Health Assistant API",
    description=(
        "Multi-Agent LangGraph system for Nawaloka Hospitals.\n\n"
        "- **POST /chat** — send a message, get an AI response\n"
        "- **POST /chat/stream** — same, but streamed node-by-node (SSE)\n"
        "- **GET /health** — liveness check\n"
        "- **GET /graph** — LangGraph topology (Mermaid)\n"
        "- **GET /memory/{user_id}** — inspect stored facts\n"
        "- **POST /memory/clear** — clear session memory"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # teaching project — open CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────

def _get_agent():
    """Retrieve the agent from app state, or raise 503."""
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")
    return agent


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the agent and get a response.

    Uses ``graph.ainvoke()`` internally so this endpoint is fully async —
    multiple concurrent requests are handled without blocking.
    """
    agent = _get_agent()

    try:
        resp = await agent.achat(
            user_message=request.user_message,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        return ChatResponse(
            answer=resp.answer,
            route=resp.route,
            routes=resp.routes,
            action=resp.action,
            tool_output=resp.tool_output,
            memory_context=resp.memory_context,
            latency_ms=resp.latency_ms,
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream agent responses node-by-node via Server-Sent Events (SSE).

    Uses ``graph.astream()`` — each LangGraph node emits a state diff
    as it completes.  The client receives real-time updates:

    ```
    data: {"node": "recall", "status": "done", "data": {"memory_lines": 7}}
    data: {"node": "supervisor", "status": "done", "data": {"routes": ["rag"]}}
    data: {"node": "clinical_agent", "status": "done", "data": {"answer_preview": "..."}}
    data: {"node": "complete", "status": "done", "data": {"answer": "...", "latency_ms": 1234}}
    ```
    """
    agent = _get_agent()

    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=request.user_message)],
        "user_id": request.user_id,
        "session_id": request.session_id,
        "agent_outputs": [],
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.time()

        try:
            async for step in agent.graph.astream(initial_state):
                for node_name, node_output in step.items():
                    event_data = {"node": node_name, "status": "done", "data": {}}

                    if node_output is None:
                        event_data["data"] = {"info": "pass-through"}
                    else:
                        # Summarise each node's output (don't send raw state)
                        for key, val in node_output.items():
                            if val is None:
                                continue
                            if isinstance(val, str):
                                event_data["data"][key] = val[:200]
                            elif isinstance(val, list):
                                event_data["data"][key] = f"[{len(val)} items]"
                            elif isinstance(val, dict):
                                event_data["data"][key] = str(val)[:200]
                            else:
                                event_data["data"][key] = str(val)[:100]

                    yield f"data: {json.dumps(event_data)}\n\n"

            # Final completion event
            latency = int((time.time() - t0) * 1000)
            complete = {
                "node": "complete",
                "status": "done",
                "data": {"latency_ms": latency},
            }
            yield f"data: {json.dumps(complete)}\n\n"

        except Exception as e:
            error_event = {"node": "error", "status": "failed", "data": {"detail": str(e)}}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness and readiness check — reports which tools are enabled."""
    agent = getattr(app.state, "agent", None)

    if agent is None:
        return HealthResponse(status="starting", tools=ToolStatus())

    return HealthResponse(
        status="ok",
        tools=ToolStatus(
            crm=agent.crm_tool is not None,
            rag=agent.rag_tool is not None,
            web_search=agent.web_tool is not None,
        ),
    )


@app.get("/graph", response_model=GraphResponse, tags=["System"])
async def graph_topology():
    """
    Return the LangGraph topology as a Mermaid diagram and structured
    nodes/edges list.
    """
    agent = _get_agent()

    g = agent.graph.get_graph()

    # Mermaid text
    mermaid = g.draw_mermaid()

    # Structured nodes & edges
    nodes = list(g.nodes.keys())
    edges = []
    for edge in g.edges:
        edges.append(GraphEdge(
            source=edge.source,
            target=edge.target,
            conditional=edge.conditional,
            label=edge.data if isinstance(edge.data, str) else None,
        ))

    return GraphResponse(mermaid=mermaid, nodes=nodes, edges=edges)


@app.get("/memory/{user_id}", response_model=MemoryResponse, tags=["Memory"])
async def get_memory(user_id: str):
    """
    Inspect a user's stored long-term facts.

    Runs the DB query in a thread pool so it doesn't block the event loop.
    """
    agent = _get_agent()

    try:
        # lt_store.get_all_facts is sync — run in thread pool
        facts = await asyncio.to_thread(
            agent.lt_store.get_all_facts, user_id
        )

        fact_items = [
            FactItem(
                text=f.text if hasattr(f, "text") else str(f),
                tags=f.tags if hasattr(f, "tags") else [],
                score=f.score if hasattr(f, "score") else 0.0,
            )
            for f in facts
        ]

        return MemoryResponse(
            user_id=user_id,
            fact_count=len(fact_items),
            facts=fact_items,
        )

    except Exception as e:
        logger.error(f"Memory lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear", response_model=MemoryClearResponse, tags=["Memory"])
async def clear_memory(request: MemoryClearRequest):
    """
    Clear a user's short-term memory for a specific session.

    Long-term facts are NOT deleted (they persist across sessions).
    Runs the DB operation in a thread pool.
    """
    agent = _get_agent()

    try:
        await asyncio.to_thread(
            agent.st_store.clear, request.user_id, request.session_id
        )

        return MemoryClearResponse(
            cleared=True,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    except Exception as e:
        logger.error(f"Memory clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
