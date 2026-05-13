"""
LangGraph ↔ LiveKit adapter.

This file is the only place the voice layer touches the agent. Everything
else (STT, TTS, VAD, session) is generic LiveKit plumbing — this adapter
is what makes it *our* agent answering the call.

Architecture:

    LiveKit Agent  ──── audio ───▶ Deepgram STT
                                     │ transcript
                                     ▼
                              LangGraphLLMAdapter   ◀── this file
                                     │
                                     ▼
                  AgentOrchestrator.achat(text, user_id, session_id)
                                     │ AgentResponse.answer
                                     ▼
                               Deepgram TTS  ──── audio ───▶ user

Future improvement (Week 14+):
    Right now we call ``orchestrator.achat()``, which uses the legacy
    multi-agent graph (recall → supervisor → agents → merge → save).
    The HTTP chat endpoint (``api/routers/chat.py``) goes through the
    newer ``decision_graph`` first (parallel guardrail + router + CAG)
    before falling through to the same multi-agent graph. Voice users
    therefore skip the guardrail and CAG cache. To lift this, extract
    ``_run_chat_pipeline`` from chat.py into a shared service module
    and call it here. Kept simple for now to keep the voice module
    fully decoupled.

Compatible with **livekit-agents >= 1.5.0**.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger

from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    LLM,
    LLMStream,
    Tool,
)

from agents.orchestrator import AgentOrchestrator


# Adapter 

class LangGraphLLMAdapter(LLM):
    """Wrap ``AgentOrchestrator`` so it satisfies LiveKit's ``LLM`` interface.

    LiveKit's ``Agent`` calls ``llm.chat(chat_ctx=...)`` after STT
    finalises a transcript. This adapter hands that transcript to the
    orchestrator and returns the answer text as a single ``ChatChunk``
    for the TTS plugin to speak.

    Parameters
    ----------
    orchestrator : AgentOrchestrator
        The pre-built multi-agent graph (from ``build_agent()``).
    user_id : str
        Caller identity. Defaults to ``"voice-user"`` when no
        participant identity is available; the agent factory
        overrides this with ``participant.identity`` per session.
    session_id : str
        Voice session identifier. Defaults to ``"voice-session"``;
        the agent factory uses ``f"voice-{room.name}"``.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        user_id: str = "voice-user",
        session_id: str = "voice-session",
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._user_id = user_id
        self._session_id = session_id
        self._current_task: Optional[asyncio.Task] = None

    #  LiveKit LLM interface (v1.5)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "LangGraphLLMStream":
        """Called by LiveKit after STT produces a final transcript."""
        return LangGraphLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            orchestrator=self._orchestrator,
            user_id=self._user_id,
            session_id=self._session_id,
        )

    # Cancellation hook (barge-in)

    def cancel_current(self) -> None:
        """Cancel any in-flight ``achat()`` task. Called on barge-in."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.info("Cancelled in-flight agent task (barge-in)")

    def update_identity(self, user_id: str, session_id: str) -> None:
        """Update caller identity (used when a new participant joins)."""
        self._user_id = user_id
        self._session_id = session_id
        logger.debug(f"Adapter identity → user={user_id} session={session_id}")


#  Stream 

class LangGraphLLMStream(LLMStream):
    """Async iterator that runs the agent and yields its answer.

    LiveKit's pipeline consumes this stream chunk-by-chunk and pipes
    each chunk's content into the TTS plugin. Since ``achat()`` returns
    a complete answer (not a token stream), we emit a single chunk
    containing the full response. The TTS plugin handles word-level
    streaming downstream.
    """

    def __init__(
        self,
        llm: LangGraphLLMAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        orchestrator: AgentOrchestrator,
        user_id: str,
        session_id: str,
    ) -> None:
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._orchestrator = orchestrator
        self._user_id = user_id
        self._session_id = session_id
        self._adapter = llm

    async def _run(self) -> None:
        # ── Extract the latest user message from the chat context ──
        user_text = ""
        for msg in reversed(self._chat_ctx.messages):
            if not (hasattr(msg, "role") and msg.role == "user"):
                continue
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        user_text = part
                        break
            if user_text:
                break

        if not user_text:
            logger.warning("No user text found in chat context — skipping agent")
            return

        preview = user_text[:80] + ("..." if len(user_text) > 80 else "")
        logger.info(f'Voice → Agent: "{preview}"')

        t0 = time.perf_counter()
        try:
            self._adapter._current_task = asyncio.current_task()
            response = await self._orchestrator.achat(
                user_message=user_text,
                user_id=self._user_id,
                session_id=self._session_id,
            )
            answer = response.answer or ""
            elapsed = int((time.perf_counter() - t0) * 1000)
            ans_preview = answer[:80] + ("..." if len(answer) > 80 else "")
            logger.success(
                f'Agent → TTS: "{ans_preview}" '
                f"[route={response.route}, {elapsed} ms]"
            )

            self._event_ch.send_nowait(
                ChatChunk(
                    id="lg-response",
                    delta=ChoiceDelta(role="assistant", content=answer),
                )
            )

        except asyncio.CancelledError:
            elapsed = int((time.perf_counter() - t0) * 1000)
            logger.info(f"Agent task cancelled after {elapsed} ms (barge-in)")
            raise

        except Exception:
            logger.exception("Agent processing failed")
            # Don't leave TTS hanging — emit a graceful fallback so the
            # user hears *something* instead of dead air.
            self._event_ch.send_nowait(
                ChatChunk(
                    id="lg-error",
                    delta=ChoiceDelta(
                        role="assistant",
                        content=(
                            "I'm sorry, I had a problem processing that. "
                            "Could you please try again?"
                        ),
                    ),
                )
            )

        finally:
            self._adapter._current_task = None
