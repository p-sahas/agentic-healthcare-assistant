"""
FastAPI application for the Nawaloka Health Assistant.

Architecture:
    Sync HTTP contract · async internals · CAG short-circuit for common
    queries · full LangGraph agent on cache miss · background cache writes.

Endpoints (mounted from ``routers/`` and ``routers/tools/``):

    POST  /chat                          — main conversational endpoint
    POST  /chat/reset                    — clear session ST memory
    GET   /sessions/{sid}/turns          — debug: dump recent ST turns
    GET   /health, /ready, /config       — system status

    POST  /tools/crm/*                   — 5 CRM actions
    POST  /tools/rag/search              — internal KB retrieval
    GET   /tools/rag/stats               — RAG cache stats
    POST  /tools/web_search              — Tavily real-time search
    POST  /tools/cag/{get,set,clear}     — semantic cache CRUD
    GET   /tools/cag/stats
    POST  /tools/memory/{recall,store_fact,distill}
    GET   /tools/memory/facts/{user_id}
    POST  /tools/crawl                   — Playwright BFS crawl

Start:
    cd src && uvicorn api.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Ensure src/ is on the path regardless of launch cwd
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

from api.middleware import install_middleware
from api.routers import chat as chat_router
from api.routers import chat_sessions as chat_sessions_router
from api.routers import health as health_router
from api.routers import patients as patients_router
from api.routers.tools import cag as cag_router
from api.routers.tools import crawl as crawl_router
from api.routers.tools import crm as crm_router
from api.routers.tools import memory as memory_router
from api.routers.tools import rag as rag_router
from api.routers.tools import web as web_router


# ── Lifespan — warm heavy singletons before accepting traffic ────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the agent (LLM clients, DB pools, Qdrant, embedder) once at
    startup. Runs in a worker thread to keep the event loop responsive.

    Warmup steps performed before accepting traffic:
      1. Pre-fetch every Langfuse prompt → eliminates per-request hops.
      2. Run a one-token LLM call against the router model → warms the
         Groq HTTP pool / TCP / TLS, removing cold-start tax on the first
         user request.
      3. Run a one-vector embed → warms OpenAI embeddings client.
    """
    logger.info("Starting Nawaloka Health Assistant API...")

    from agents.orchestrator import build_agent
    from agents.prompts.agent_prompts import LANGFUSE_PROMPT_NAMES as AGENT_PROMPTS
    from memory.prompts import LANGFUSE_PROMPT_NAMES as MEMORY_PROMPTS
    from infrastructure.llm.embeddings import get_default_embeddings, get_local_embedder
    from infrastructure.observability import prefetch_prompts
    from services.chat_service.cag_cache import CAGCache

    # Every Langfuse prompt name used anywhere in the codebase. Listing
    # them here means none ever incurs a network hop on the request path
    # — the per-call ``fetch_prompt()`` always finds it in cache.
    ALL_PROMPT_NAMES = list(AGENT_PROMPTS.values()) + list(MEMORY_PROMPTS.values())

    # Heavy, mostly I/O-bound — off the event loop
    agent = await asyncio.to_thread(
        build_agent, enable_crm=True, enable_rag=True, enable_web=True,
    )
    embedder = agent.rag_tool.embedder if agent.rag_tool else get_default_embeddings()

    # ── Local-embedded CAG cache ─────────────────────────────────
    # The chat hot path's CAG short-circuit lookup runs on a local
    # sentence-transformers model (~30 ms, no network) instead of
    # OpenAI (~700-1500 ms from Sri Lanka). Different collection
    # name (cag_cache_local, dim=384) so it doesn't clash with the
    # legacy OpenAI-embedded cag_cache (dim=1536).
    local_embedder = await asyncio.to_thread(get_local_embedder)

    # Drop any stale cag_cache_local from the previous container run
    # before warmup. The collection lives on Qdrant Cloud (persistent),
    # so old "I apologize / not available" answers from earlier code
    # paths out-rank the real FAQ entries we're about to load. A
    # clean recreate every startup is cheap (collection is small) and
    # makes FAQ correctness deterministic.
    try:
        from infrastructure.db.qdrant_client import get_qdrant_client, collection_exists
        _qc = get_qdrant_client()
        if collection_exists("cag_cache_local"):
            await asyncio.to_thread(_qc.delete_collection, "cag_cache_local")
            logger.info("CAG cache_local: dropped stale collection for clean warmup")
    except Exception as _exc:
        logger.warning("CAG cache_local purge skipped: {}", _exc)

    cag_cache = await asyncio.to_thread(
        CAGCache,
        local_embedder,
        "cag_cache_local",
        local_embedder.dim,
    )

    # Make the RAG tool's internal cache point at the same local cache —
    # this keeps the /chat short-circuit and the RAG pipeline coherent
    # (a query the agent answered via RAG can still hit the cache on a
    # subsequent /chat call).
    if agent.rag_tool is not None:
        agent.rag_tool._cache = cag_cache
        cag_service = getattr(agent.rag_tool, "_cag_service", None)
        if cag_service is not None:
            cag_service.cache = cag_cache

    # Attach the cache to the orchestrator so the decision LangGraph's
    # cag_node (which reads ``agent.cag_cache`` via a getter closure)
    # can perform semantic lookups during the parallel classification
    # phase. The graph itself was compiled at orchestrator-build time
    # before this cache existed; the late binding is intentional.
    agent.cag_cache = cag_cache

    app.state.agent = agent
    app.state.embedder = embedder              # remote, for LT memory + RAG index
    app.state.local_embedder = local_embedder  # local, for CAG short-circuit
    app.state.cag_cache = cag_cache

    # ── Per-session warm cache ──────────────────────────────────
    # Keyed by ``(patient_id, session_id)``. Populated by the
    # ``/sessions/warmup`` endpoint at login (and on session switch),
    # then maintained write-through by the chat hot path.
    #
    # Holds:
    #   patient:  full Patient row as a dict
    #   st_turns: most recent ConversationTurns
    #
    # When the chat endpoint finds an entry here it can skip both the
    # patient PK lookup and the ST recall round-trip — those happen
    # during the dead time between login and first message.
    app.state.session_cache = {}

    # ── Warmup: prompts + LLM pools + embedder ──────────────────
    async def _warmup_prompts():
        try:
            await asyncio.to_thread(prefetch_prompts, ALL_PROMPT_NAMES)
        except Exception as exc:
            logger.warning("Prompt prefetch failed: {}", exc)

    async def _warmup_router():
        try:
            await agent.router.aroute("ping", "")
        except Exception as exc:
            logger.debug("Router warmup error (non-fatal): {}", exc)

    async def _warmup_fast_llm():
        try:
            await agent.llm_fast.ainvoke("hi")
        except Exception as exc:
            logger.debug("Fast LLM warmup error (non-fatal): {}", exc)

    async def _warmup_embedder():
        try:
            await asyncio.to_thread(embedder.embed_query, "warmup")
        except Exception as exc:
            logger.debug("Embedder warmup error (non-fatal): {}", exc)

    async def _warmup_faqs_cache():
        """
        Seed CAG with the curated patient FAQs from ``config/faqs.yaml``.
        Each entry already has a hand-written ``answer`` so we bypass
        CRAG and write directly into the cache — sub-200 ms hits for
        common patient questions without any LLM call.
        """
        if cag_cache is None:
            return
        try:
            from infrastructure.config import KNOWN_FAQS
        except Exception as exc:
            logger.debug("FAQ load skipped: {}", exc)
            return

        faqs = [
            f for f in (KNOWN_FAQS or [])
            if isinstance(f, dict) and f.get("query") and f.get("answer")
        ]
        if not faqs:
            return

        cached = 0
        for entry in faqs:
            try:
                await asyncio.to_thread(
                    cag_cache.set,
                    str(entry["query"]),
                    {"answer": str(entry["answer"]), "evidence_urls": []},
                )
                cached += 1
            except Exception as exc:
                logger.debug("CAG FAQ warmup failed for '{}': {}",
                             entry.get("query", "?")[:60], exc)
        if cached:
            logger.info("Pre-warmed CAG with {}/{} FAQ entries from faqs.yaml",
                        cached, len(faqs))

    async def _warmup_reference_cache():
        """
        Pre-seed the CAG cache with patient-agnostic hospital reference
        answers (departments, locations). The first user to ask "what
        departments do you have" then gets a sub-300 ms cache hit
        instead of paying for routing + CRM lookup + synthesis.

        Tool output is wrapped with a brief conversational frame so the
        cache hit reads naturally without going through the synth.
        """
        crm = getattr(agent, "crm_tool", None)
        if crm is None or cag_cache is None:
            return
        try:
            specialties_md = await asyncio.to_thread(crm.list_specialties)
            locations_md = await asyncio.to_thread(crm.list_locations)
        except Exception as exc:
            logger.warning("Reference data fetch failed: {}", exc)
            return

        spec_answer = (
            "Here are the departments offered at Nawaloka Hospitals:\n\n"
            f"{specialties_md}\n\n"
            "Let me know if you'd like to see doctors in any specialty."
        )
        loc_answer = (
            "Here are our active branches and clinics:\n\n"
            f"{locations_md}\n\n"
            "Let me know if you'd like directions or details for any of them."
        )

        # Multiple natural phrasings of the same intent — CAG's cosine
        # similarity covers paraphrases, but seeding several anchors
        # reduces near-miss latency.
        seeds = [
            ("What departments do you have?", spec_answer),
            ("What specialties does Nawaloka offer?", spec_answer),
            ("List the departments at the hospital", spec_answer),
            ("How many specialties are there?", spec_answer),
            ("Where are you located?", loc_answer),
            ("What branches / hospitals / clinics do you have?", loc_answer),
            ("List your hospital locations", loc_answer),
        ]

        cached = 0
        for query, answer in seeds:
            try:
                await asyncio.to_thread(
                    cag_cache.set, query,
                    {"answer": answer, "evidence_urls": []},
                )
                cached += 1
            except Exception as exc:
                logger.debug("CAG warmup failed for '{}': {}", query, exc)
        if cached:
            logger.info("Pre-warmed CAG with {}/{} reference Q&A seeds", cached, len(seeds))

    # Run all warmups concurrently — total time is max(individual), not sum
    await asyncio.gather(
        _warmup_prompts(),
        _warmup_router(),
        _warmup_fast_llm(),
        _warmup_embedder(),
        _warmup_reference_cache(),
        _warmup_faqs_cache(),
    )

    logger.success("API ready — agent + CAG cache online (warmup complete)")

    try:
        yield
    finally:
        logger.info("Shutting down API...")
        mcp_client = getattr(agent, "mcp_client", None)
        if mcp_client is not None:
            try:
                close = getattr(mcp_client, "aclose", None) or getattr(mcp_client, "close", None)
                if close:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as exc:
                logger.warning("MCP client shutdown raised: {}", exc)


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nawaloka Health Assistant API",
    description=(
        "Sync chat API backed by a LangGraph multi-agent system. "
        "Hot path short-circuits via a Qdrant semantic cache; full agent "
        "runs only on cache miss. All MCP-wrapped tools are also exposed "
        "as direct REST endpoints under ``/tools/*``."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # teaching project — tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_middleware(app)


# ── Routers ──────────────────────────────────────────────────────────

app.include_router(health_router.router)
app.include_router(patients_router.router)
app.include_router(chat_sessions_router.router)
app.include_router(chat_router.router)
app.include_router(crm_router.router)
app.include_router(rag_router.router)
app.include_router(web_router.router)
app.include_router(cag_router.router)
app.include_router(memory_router.router)
app.include_router(crawl_router.router)


@app.get("/", tags=["System"])
async def root():
    """Friendly landing page pointer."""
    return {
        "service": "Nawaloka Health Assistant API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
    }
