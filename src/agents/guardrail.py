"""
Domain Guardrail — keeps the assistant on-topic.

A binary classifier that decides whether the user message is within
the scope of a Nawaloka Hospital health assistant. Runs on Llama
3.1 8B Instant via Groq (~150 ms) and is invoked in parallel with
the router / CAG / memory recall, so its latency is hidden inside
the existing parallel batch.

When ``out_of_scope`` is returned, the chat pipeline short-circuits
to a templated polite refusal — no router classification, no tool
call, no synthesis LLM call. This is what stops the bot from
answering "who is the president of the USA?" or burning a Tavily
search on unrelated weather questions.

The guardrail fails *open* on any error: a transient Groq outage
should not block legitimate users, so we let the request continue
through the normal router path.
"""

from __future__ import annotations

from typing import Any, Literal
from loguru import logger

from infrastructure.observability import observe, update_current_observation


GuardrailVerdict = Literal["in_scope", "out_of_scope"]


_GUARDRAIL_SYSTEM = """\
You are a scope filter for Nawaloka Hospital's AI assistant.

Decide whether the user's message is within the assistant's domain.

IN-SCOPE — the assistant should help with:
  • Patient's own appointments / bookings / medical records
  • Hospital info (hours, location, parking, departments, doctors,
    services, pricing, insurance, contact)
  • Medical procedures, preparation, prescriptions, what to bring,
    fasting, recovery
  • The patient's own health questions, symptoms, follow-ups
  • Logistics tied to attending the hospital — directions, traffic
    to the hospital, transit, weather only when the user is asking
    about getting to or around the hospital
  • Greetings, small talk, thanks (these are still in-scope; the
    main assistant handles them)

OUT-OF-SCOPE — politely refuse:
  • General world knowledge (presidents, capitals, sports, history,
    celebrities, politics, science trivia)
  • Other businesses, brands, services, products unrelated to
    Nawaloka or healthcare
  • Generic weather, news, stock prices, sports scores
  • Coding help, math problems, jokes, riddles, role-play
  • Gibberish or random non-questions
  • Anything you can't confidently tie to a hospital / health /
    patient-care intent

Answer with ONE WORD ONLY: ``in_scope`` or ``out_of_scope``.
No explanation, no punctuation, no other tokens.
"""


# Few-shot examples baked into the user-prompt template — keeps the
# 8B model honest without burning a separate fine-tune.
_GUARDRAIL_EXAMPLES = """\
Examples:
  USER: "Do I have an appointment today?"           → in_scope
  USER: "What should I bring for a CT scan?"        → in_scope
  USER: "Hey there"                                 → in_scope
  USER: "Is the hospital open at night?"            → in_scope
  USER: "Is there traffic to the hospital now?"     → in_scope
  USER: "Who is the president of the USA?"          → out_of_scope
  USER: "What's the weather in Sri Lanka?"          → out_of_scope
  USER: "Write me a Python function"                → out_of_scope
  USER: "total guardrail"                           → out_of_scope
  USER: "What's the capital of France?"             → out_of_scope
"""


def _build_user_prompt(message: str) -> str:
    return f"{_GUARDRAIL_EXAMPLES}\n\nUSER: \"{(message or '').strip()}\"\n→"


class Guardrail:
    """Binary in_scope / out_of_scope classifier."""

    def __init__(self, llm: Any) -> None:
        """
        Args:
            llm: A LangChain ``ChatOpenAI``-compatible instance.
                Use the extractor LLM (Llama 3.1 8B on Groq) — it's
                cheap and fast enough that parallel guardrail latency
                is hidden behind the router (~800 ms) in every gather.
        """
        self.llm = llm

    @observe(name="guardrail", as_type="generation")
    async def aclassify(self, message: str) -> GuardrailVerdict:
        """Classify *message* as ``in_scope`` or ``out_of_scope``.

        Fails open: any LLM error returns ``in_scope`` so transient
        provider issues don't lock real users out of the assistant.
        """
        msgs = [
            {"role": "system", "content": _GUARDRAIL_SYSTEM},
            {"role": "user", "content": _build_user_prompt(message)},
        ]
        try:
            response = await self.llm.ainvoke(msgs)
        except Exception as exc:
            logger.warning("Guardrail LLM error (failing open): {}", exc)
            return "in_scope"

        raw = (
            response.content if hasattr(response, "content") else str(response)
        ).strip().lower()

        # Be permissive in parsing — the model occasionally adds quotes,
        # backticks, or trailing punctuation despite the instruction.
        verdict: GuardrailVerdict
        if "out_of_scope" in raw or "out-of-scope" in raw or "out of scope" in raw:
            verdict = "out_of_scope"
        elif "in_scope" in raw or "in-scope" in raw or "in scope" in raw:
            verdict = "in_scope"
        else:
            # Unrecognised response — safest default is to let the
            # normal pipeline handle it.
            logger.debug("Guardrail unparsable response {!r} → defaulting in_scope", raw[:50])
            verdict = "in_scope"

        update_current_observation(
            input=(message or "")[:200],
            output=verdict,
        )
        return verdict


# Templated refusal returned when the guardrail says out_of_scope.
# Kept short and friendly — the bot is a hospital concierge, not a
# rule enforcer. Single source of truth so we only edit it here.
OUT_OF_SCOPE_REPLY = (
    "I'm the Nawaloka Health Assistant — I can help with your "
    "appointments, doctors, hospital information, and your medical "
    "records. That's outside what I'm built for, but I'm happy to "
    "help with anything related to your visit. What can I do for you?"
)
