"""
Agent Orchestrator - main execution loop.

Flow:
  1. Recall memory (short-term turns + long-term facts).
  2. Route the user query (LLM → RouteDecision).
  3. Dispatch to the selected tool (CRM / RAG / Web Search / direct).
  4. Synthesise final answer (LLM merges tool output + memory).
  5. Store the new conversation turn in short-term memory.
  6. Trigger distillation if needed (extract long-term facts).

Every step is traced via LangFuse '@observe' for full observability.
"""

from loguru import logger
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from memory.schemas import ConversationTurn
from agents.prompts.agent_prompts import build_synthesiser_prompt
from agents.router import QueryRouter, RouteDecision
from infrastructure.observability import (
    observe,
    update_current_trace,
    update_current_observation,
    flush,
)


@dataclass
class AgentResponse:
    """
    Complete agent response with metadata.

    Attributes:
        answer: The final text response to the user.
        route: Which route was selected.
        action: CRM sub-action (if route == crm).
        tool_output: Raw tool output (for debugging).
        memory_context: Memory context that was used.
        latency_ms: End-to-end processing time.
    """

    answer: str
    route: str = "direct"
    action: Optional[str] = None
    tool_output: str = ""
    memory_context: str = ""
    latency_ms: int = 0


class AgentOrchestrator:
    """
    The main agent loop that ties routing, tools, memory, and synthesis together.

    Dependencies (injected via '__init__'):
        llm          - LangChain ChatOpenAI
        st_store     - ShortTermMemoryStore
        lt_store     - LongTermMemoryStore
        recaller     - MemoryRecaller
        distiller    - MemoryDistiller
        crm_tool     - CRMTool   (optional - None if CRM is unavailable)
        rag_tool     - RAGTool   (optional - None if KB is empty)
        web_tool     - WebSearchTool (optional - None if Tavily key is missing)
    """

    def __init__(
        self,
        llm_chat: Any,
        llm_router: Any,
        st_store: Any,
        lt_store: Any,
        recaller: Any,
        distiller: Any,
        crm_tool: Optional[Any] = None,
        rag_tool: Optional[Any] = None,
        web_tool: Optional[Any] = None,
    ) -> None:
        self.llm_chat = llm_chat
        self.st_store = st_store
        self.lt_store = lt_store
        self.recaller = recaller
        self.distiller = distiller

        self.crm_tool = crm_tool
        self.rag_tool = rag_tool
        self.web_tool = web_tool

        self.router = QueryRouter(llm_router)

    # public entry point

    @observe(name="agent_chat")
    def chat(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
    ) -> AgentResponse:
        """
        Process a single user message through the full agent pipeline.

        This is the **top-level LangFuse trace** - every sub-step
        (recall, route, tool, synthesise) appears as a nested span.
        """
        t0 = time.time()

        # Tag the trace with user / session for dashboard filtering
        update_current_trace(
            user_id=user_id,
            session_id=session_id,
            tags=["agent"],
        )

        #  Step 1: Recall memory
        memory_context = self._recall_memory(user_id, session_id, user_message)

        #  Step 2: Route
        decision = self.router.route(user_message, memory_context)
        logger.info(
            "Route: {} (action={}, conf={:.2f}) - {}",
            decision.route,
            decision.action,
            decision.confidence,
            decision.reasoning,
        )

        # Step 3: Dispatch to tool
        tool_output = self._dispatch(decision)

        # Step 4: Synthesise final answer
        answer = self._synthesise(
            user_message=user_message,
            memory_context=memory_context,
            route=decision.route,
            tool_output=tool_output,
        )

        #  Step 5: Store turns in ST memory
        self._store_turns(user_id, session_id, user_message, answer)

        # Step 6: Trigger distillation (background-safe)
        self._maybe_distill(user_id, session_id)

        latency_ms = int((time.time() - t0) * 1000)

        # Attach final metadata to the trace
        update_current_trace(
            metadata={
                "route": decision.route,
                "action": decision.action,
                "confidence": decision.confidence,
                "latency_ms": latency_ms,
            },
        )

        return AgentResponse(
            answer=answer,
            route=decision.route,
            action=decision.action,
            tool_output=tool_output,
            memory_context=memory_context,
            latency_ms=latency_ms,
        )

    # internal steps

    @observe(name="memory_recall")
    def _recall_memory(
        self,
        user_id: str,
        session_id: str,
        query: str,
    ) -> str:
        """Recall ST + LT memory and format as a context string."""
        try:
            st_turns, lt_facts = self.recaller.recall(
                user_id=user_id,
                session_id=session_id,
                query=query,
            )
            context = self.recaller.format_context(st_turns, lt_facts)
            update_current_observation(
                output=context[:500],
                metadata={
                    "st_turns": len(st_turns),
                    "lt_facts": len(lt_facts),
                },
            )
            return context
        except Exception as exc:
            logger.warning("Memory recall failed: {}", exc)
            return "(memory unavailable)"

    @observe(name="tool_dispatch")
    def _dispatch(self, decision: RouteDecision) -> str:
        """Dispatch to the selected tool and return its output."""
        route = decision.route
        params = decision.params or {}

        update_current_observation(
            input=f"route={route} action={decision.action} params={params}",
        )

        if route == "crm":
            if not self.crm_tool:
                return "CRM tool is not available."
            action = decision.action or "lookup_patient"
            logger.info("Dispatching CRM action: {} params={}", action, params)
            result = self.crm_tool.dispatch(action, params)
            update_current_observation(output=result[:500])
            return result

        if route == "rag":
            if not self.rag_tool:
                return "Internal knowledge base is not available."
            query = params.get("query", "")
            if not query:
                return "No query provided for knowledge base search."
            logger.info("Dispatching RAG search: {}", query[:80])
            result = self.rag_tool.dispatch("search", {"query": query})
            update_current_observation(output=result[:500])
            return result

        if route == "web_search":
            if not self.web_tool:
                return "Web search tool is not available (TAVILY_API_KEY not set)."
            query = params.get("query", "")
            if not query:
                return "No query provided for web search."
            logger.info("Dispatching web search: {}", query[:80])
            result = self.web_tool.dispatch("search", {"query": query})
            update_current_observation(output=result[:500])
            return result

        # direct - no tool needed
        return ""

    @observe(name="synthesiser", as_type="generation")
    def _synthesise(
        self,
        user_message: str,
        memory_context: str,
        route: str,
        tool_output: str,
    ) -> str:
        """Run the synthesiser LLM to produce the final answer."""
        system_prompt, user_prompt = build_synthesiser_prompt(
            user_message=user_message,
            memory_context=memory_context,
            route=route,
            tool_output=tool_output,
        )

        update_current_observation(
            input=user_prompt[:1000],
            model=self._model_name(),
        )

        try:
            response = self.llm_chat.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            # Extract token usage if available
            usage = {}
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                token_usage = meta.get("token_usage") or meta.get("usage", {})
                if token_usage:
                    usage = {
                        "input": token_usage.get("prompt_tokens", 0),
                        "output": token_usage.get("completion_tokens", 0),
                        "total": token_usage.get("total_tokens", 0),
                    }

            update_current_observation(
                output=content.strip()[:1000],
                usage=usage if usage else None,
            )

            return content.strip()
        except Exception as exc:
            logger.error("Synthesiser LLM failed: {}", exc)
            if tool_output:
                return f"Here's what I found:\n{tool_output}"
            return "I'm sorry, I encountered an error processing your request."

    @observe(name="memory_store")
    def _store_turns(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        answer: str,
    ) -> None:
        """Store user + assistant turns in short-term memory."""
        now = time.time()
        self.st_store.add(
            user_id,
            session_id,
            ConversationTurn(
                user_id=user_id,
                session_id=session_id,
                role="user",
                content=user_message,
                ts=now,
            ),
        )
        self.st_store.add(
            user_id,
            session_id,
            ConversationTurn(
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                content=answer,
                ts=now,
            ),
        )

    @observe(name="memory_distill")
    def _maybe_distill(self, user_id: str, session_id: str) -> None:
        """Trigger memory distillation if the policy says so."""
        try:
            recent = self.st_store.recent(user_id, session_id, k=10)
            if self.distiller.should_distill(recent):
                logger.info("Triggering memory distillation for {}", user_id)
                self.distiller.distill(user_id, recent)
                update_current_observation(
                    metadata={"distillation_triggered": True},
                )
        except Exception as exc:
            logger.warning("Distillation skipped: {}", exc)

    #  helpers

    def _model_name(self) -> str:
        """Extract model name from the chat LLM for LangFuse metadata."""
        if hasattr(self.llm_chat, "model_name"):
            return self.llm_chat.model_name
        if hasattr(self.llm_chat, "model"):
            return self.llm_chat.model
        return "unknown"


# Factory: build a fully-wired orchestrator from config


def build_agent(
    enable_crm: bool = True,
    enable_rag: bool = True,
    enable_web: bool = True,
) -> AgentOrchestrator:
    """
    Convenience factory that constructs and wires all components.

    Reads config / env for API keys and database URLs.

    Args:
        enable_crm: Attach CRM tool.
        enable_rag: Attach RAG tool.
        enable_web: Attach Web Search tool.

    Returns:
        A fully initialised ``AgentOrchestrator``.
    """
    from dotenv import load_dotenv

    load_dotenv()

    # Eagerly init LangFuse so child spans are captured
    from infrastructure.observability import get_langfuse

    get_langfuse()

    from infrastructure.llm import (
        get_chat_llm,
        get_router_llm,
        get_extractor_llm,
        get_default_embeddings,
    )
    from memory.st_store import ShortTermMemoryStore
    from memory.lt_store import LongTermMemoryStore
    from memory.memory_ops import MemoryRecaller, MemoryDistiller

    # 3-model architecture
    # Gemini 2.0 Flash - synthesis
    llm_chat = get_chat_llm(temperature=0)
    llm_router = get_router_llm(temperature=0)       # GPT-4o-mini - routing
    # Llama 3.1 8B via Groq - extraction
    llm_extractor = get_extractor_llm(temperature=0)
    embedder = get_default_embeddings()

    logger.info("LLM models loaded:")
    logger.info("   Chat (synthesis) : {}", getattr(
        llm_chat, 'model_name', getattr(llm_chat, 'model', '?')))
    logger.info("   Router           : {}", getattr(
        llm_router, 'model_name', getattr(llm_router, 'model', '?')))
    logger.info("   Extractor        : {}", getattr(llm_extractor,
                'model_name', getattr(llm_extractor, 'model', '?')))

    # Memory stores
    st_store = ShortTermMemoryStore()
    lt_store = LongTermMemoryStore(embedder)
    recaller = MemoryRecaller(st_store, lt_store)
    distiller = MemoryDistiller(llm_extractor, lt_store)

    # Tools (each optional)
    crm_tool = None
    rag_tool = None
    web_tool = None

    if enable_crm:
        try:
            from agents.tools import CRMTool

            crm_tool = CRMTool()
            logger.info("✓ CRM tool loaded")
        except Exception as exc:
            logger.warning("CRM tool unavailable: {}", exc)

    if enable_rag:
        try:
            from agents.tools import RAGTool
            from infrastructure.config import KNOWN_FAQS
            from infrastructure.db.qdrant_client import ensure_kb_ingested

            # Auto-ingest KB if the collection is missing or empty
            ensure_kb_ingested()

            rag_tool = RAGTool(embedder=embedder, llm=llm_chat)
            logger.info(" RAG tool loaded (CAG-enabled)")

            # Warm CAG cache with known FAQs from config/faqs.yaml
            if KNOWN_FAQS:
                warmed = rag_tool.warm_cache(KNOWN_FAQS)
                if warmed:
                    logger.info(
                        "CAG cache warmed with {} new FAQ entries", warmed)
                else:
                    logger.info("CAG cache already warm ({} FAQs)",
                                len(KNOWN_FAQS))
        except Exception as exc:
            logger.warning("RAG tool unavailable: {}", exc)

    if enable_web:
        try:
            from agents.tools import WebSearchTool

            web_tool = WebSearchTool()
            logger.info(" Web search tool loaded")
        except Exception as exc:
            logger.warning("Web search tool unavailable: {}", exc)

    return AgentOrchestrator(
        llm_chat=llm_chat,
        llm_router=llm_router,
        st_store=st_store,
        lt_store=lt_store,
        recaller=recaller,
        distiller=distiller,
        crm_tool=crm_tool,
        rag_tool=rag_tool,
        web_tool=web_tool,
    )
