"""
LiveKit voice agent factory.

Wires the four pieces (Silero VAD, Deepgram STT, LangGraph adapter,
Deepgram TTS) into a ``livekit.agents.voice.Agent`` and starts an
``AgentSession`` for each room.

This file owns the LiveKit-specific lifecycle. The orchestrator
itself never sees LiveKit.

Compatible with **livekit-agents >= 1.5.0**.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import silero

from agents.orchestrator import AgentOrchestrator, build_agent
from voice.adapter import LangGraphLLMAdapter
from voice.config import VoiceConfig, load_voice_config
from voice.pipeline import (
    SessionManager,
    VoiceSession,
    on_agent_speech_finished,
    on_agent_speech_interrupted,
    on_agent_speech_started,
    on_user_speech_committed,
    on_user_speech_started,
)
from voice.stt import make_stt
from voice.tts import make_tts


# Module-level singletons (one per worker process) 

_orchestrator: Optional[AgentOrchestrator] = None
_session_manager = SessionManager()


async def _get_orchestrator() -> AgentOrchestrator:
    """Lazy-init the LangGraph agent. Cached for the worker's lifetime."""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Building LangGraph agent (one-time, may take a few seconds)...")
        # build_agent() is sync (DB connections, model loads); shove it
        # off the event loop so we don't block other rooms during startup.
        _orchestrator = await asyncio.to_thread(build_agent)
        logger.success("LangGraph agent ready")
    return _orchestrator


async def warm_start(orchestrator: AgentOrchestrator) -> None:
    """Throwaway ``achat`` call to prime LLM connection pools.

    Eliminates the cold-start tax on the first real user turn.
    """
    try:
        logger.debug("Warming agent (throwaway 'hello')...")
        await orchestrator.achat(
            user_message="hello",
            user_id="__warmup__",
            session_id="__warmup__",
        )
        logger.debug("Warm-up done")
    except Exception:
        logger.warning("Warm-up failed (non-fatal)")


# Voice agent builder 

VOICE_INSTRUCTIONS = (
    "You are the Nawaloka Hospital voice assistant. "
    "You're on a phone call — keep replies short, warm, and professional. "
    "Aim for under three sentences. The patient is listening, not reading: "
    "no bullet points, no markdown, no asterisks. Use natural speech."
)


def build_voice_agent(
    *,
    participant: rtc.RemoteParticipant,
    room_name: str,
    cfg: VoiceConfig,
    orchestrator: AgentOrchestrator,
) -> tuple[Agent, VoiceSession]:
    """Build a configured LiveKit ``Agent`` for one participant.

    Returns the Agent (for the AgentSession to start) and the VoiceSession
    record (for event handlers to update).
    """
    user_id = participant.identity or "unknown-user"
    session = _session_manager.get_or_create(
        participant_id=participant.identity,
        user_id=user_id,
        room_name=room_name,
    )
    logger.info(f"Participant joined: {user_id} → {session.session_id}")

    # The bridge between voice and the agent.
    adapter = LangGraphLLMAdapter(
        orchestrator=orchestrator,
        user_id=user_id,
        session_id=session.session_id,
    )

    # Plugins.
    vad = silero.VAD.load(
        min_silence_duration=cfg.silence_threshold_ms / 1000.0,
        activation_threshold=cfg.vad_threshold,
        sample_rate=16000,   # Deepgram + Silero want 16 kHz
    )
    stt = make_stt(cfg)
    tts = make_tts(cfg)

    agent = Agent(
        instructions=VOICE_INSTRUCTIONS,
        stt=stt,
        llm=adapter,
        tts=tts,
        vad=vad,
        allow_interruptions=cfg.interruption_enabled,
        min_endpointing_delay=cfg.min_endpointing_delay,
    )

    logger.success(
        f"Voice agent ready — "
        f"STT={cfg.stt_provider}/{cfg.stt_model}, "
        f"TTS={cfg.tts_provider}/{cfg.tts_model}, "
        f"interruptions={cfg.interruption_enabled}"
    )
    # Stash the adapter on the agent for barge-in cancellation hooks.
    agent._lg_adapter = adapter  # type: ignore[attr-defined]
    return agent, session


# Worker entrypoint helper 

async def create_and_start_agent(ctx: JobContext) -> AgentSession:
    """Connect to the room, wait for a participant, build + start the agent.

    Called from ``voice.run.entrypoint``. Handles the full per-room
    lifecycle including event wiring and disconnect cleanup.
    """
    cfg = load_voice_config()
    orchestrator = await _get_orchestrator()

    # Connect (audio only — we don't care about video tracks).
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first remote participant.
    participant = await ctx.wait_for_participant()

    agent, session = build_voice_agent(
        participant=participant,
        room_name=ctx.room.name,
        cfg=cfg,
        orchestrator=orchestrator,
    )

    # Start a session in this room.
    agent_session = AgentSession()

    #  Wire LiveKit events to our session record 
    @agent_session.on("user_state_changed")
    def _on_user_state(ev):
        # ev.new_state ∈ {"speaking", "listening", "away", ...}
        new = getattr(ev, "new_state", None)
        if new == "speaking":
            on_user_speech_started(session)
        elif new in ("listening", "away") and session.is_user_speaking:
            # We don't have the transcript here — that arrives via the
            # adapter — so just clear the speaking flag.
            session.is_user_speaking = False

    @agent_session.on("agent_state_changed")
    def _on_agent_state(ev):
        new = getattr(ev, "new_state", None)
        if new == "speaking":
            on_agent_speech_started(session)
        elif new == "listening" and session.is_agent_speaking:
            on_agent_speech_finished(session)

    @agent_session.on("agent_false_interruption")
    def _on_false_interrupt(_ev):
        # Detected speech wasn't real — agent resumes. Nothing to do.
        pass

    @agent_session.on("conversation_item_added")
    def _on_item(ev):
        # Use this for the "user committed" event since user_speech_committed
        # was renamed/removed in 1.5.x.
        item = getattr(ev, "item", None)
        if item is not None and getattr(item, "role", None) == "user":
            text = getattr(item, "text_content", None) or ""
            if text:
                on_user_speech_committed(session, text)

    # Cancel in-flight agent task on barge-in.
    adapter: LangGraphLLMAdapter = agent._lg_adapter  # type: ignore[attr-defined]

    # Some 1.5.x builds emit a dedicated event; if not, the
    # agent_state_changed → speaking → listening flip on the user side
    # is what triggers barge-in. Hook both for safety.
    try:
        @agent_session.on("agent_speech_interrupted")  # type: ignore[arg-type]
        def _on_interrupted(_ev):
            on_agent_speech_interrupted(session)
            adapter.cancel_current()
    except Exception:
        # Event name not registered in this version — fall back to
        # state-change driven cancellation only.
        logger.debug("agent_speech_interrupted event not available; using state changes")

    # Clean up on disconnect.
    @ctx.room.on("participant_disconnected")
    def _on_disconnect(p: rtc.RemoteParticipant):
        if p.identity == participant.identity:
            _session_manager.end_session(p.identity)

    # Go live.
    await agent_session.start(agent, room=ctx.room)

    # Greet the user.
    await agent_session.say(
        "Hi! Welcome to Nawaloka Hospital. How can I help you today?",
        allow_interruptions=True,
    )

    return agent_session
