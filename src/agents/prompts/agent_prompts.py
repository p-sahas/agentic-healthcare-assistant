"""
Prompt templates for the routing-engine agent.

Prompts are fetched from **LangFuse Prompt Management** at runtime.
If a prompt hasn't been created in LangFuse yet, the local fallback
(defined below) is used instead - so the system works out-of-the-box.

To manage prompts via LangFuse Cloud:
  1. Open LangFuse → Prompts → + New Prompt
  2. Create prompts with the names listed in the LANGFUSE_PROMPT_NAMES dict
  3. Use {{variable}} (double-curly Mustache syntax) for template variables
  4. Set a version to "production" to make it active

Three prompt roles:
  1. ROUTER      - classifies user intent → route + params
  2. SYNTHESISER - merges tool output + memory context → final answer
  3. SYSTEM      - base persona injected into every LLM call
"""

from infrastructure.observability import fetch_prompt


# LangFuse prompt names → create these in your dashboard


LANGFUSE_PROMPT_NAMES = {
    "agent_system":       "nawaloka-agent-system",
    "router_system":      "nawaloka-router-system",
    "router_user":        "nawaloka-router-user",
    "synthesiser_system": "nawaloka-synthesiser-system",
    "synthesiser_user":   "nawaloka-synthesiser-user",
}


# 1. SYSTEM - Base agent persona (fallback)


_AGENT_SYSTEM_FALLBACK = """\
You are **Nawaloka Health Assistant**, a friendly and knowledgeable AI assistant
for Nawaloka Hospitals, Sri Lanka.

Your capabilities:
• Answer questions about Nawaloka services, departments, and policies (internal KB).
• Look up patient records, doctor availability, and appointment details (CRM).
• Book, cancel, or reschedule appointments for patients (CRM).
• Search the web for real-time information like hospital hours, directions, news.
• Remember patient preferences and past interactions across sessions.

Communication rules:
1. Be warm, professional, and concise.
2. Always confirm before making changes (booking, cancel, reschedule).
3. Never reveal internal system details or raw IDs to the patient.
4. If unsure, say so rather than guessing.
5. Use the patient's name when available.
6. Respond in the same language as the patient (Sinhala, Tamil, or English).
"""


# 2. ROUTER - Intent classification (fallback)


_ROUTER_SYSTEM_FALLBACK = """\
You are a query router for a healthcare AI system.

Given a user message AND memory context, classify the intent into
exactly ONE primary route (or DIRECT if no tool is needed).

ROUTES:
  crm        - Patient lookup, doctor search, booking, cancellation, rescheduling.
  rag        - Hospital policies, services, departments, procedures (internal KB).
  web_search - Real-time info: hours, directions, traffic, current announcements.
  direct     - Greeting, chitchat, follow-up, or answerable from memory alone.

For CRM you must also extract the sub-action:
  lookup_patient | search_doctors | create_booking | cancel_booking | reschedule_booking

OUTPUT FORMAT (strict JSON, no markdown fences):
{
  "route": "<crm|rag|web_search|direct>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one-sentence explanation>",
  "action": "<sub-action or null>",
  "params": { <extracted parameters or empty {}> }
}

PARAMETER EXTRACTION RULES:
• For lookup_patient  → extract phone, name, patient_id, external_user_id (any available).
• For search_doctors  → extract specialty, location, name (any available).
• For create_booking  → extract patient_id, doctor_id, location_id, start_time, reason.
• For cancel_booking  → extract booking_id.
• For reschedule_booking → extract booking_id, new_start_time.
• For rag search      → put the search query in params.query.
• For web_search      → put the search query in params.query.
• For direct          → params = {}.

If user intent is ambiguous, prefer crm > rag > web_search > direct.
"""

_ROUTER_USER_FALLBACK = """\
MEMORY CONTEXT:
{memory_context}

USER MESSAGE:
{user_message}

Classify and extract (JSON):"""


# 3. SYNTHESISER - Final response generation (fallback)


_SYNTHESISER_SYSTEM_FALLBACK = """\
You are the response synthesiser for a healthcare AI assistant.

You receive:
1. The original user message.
2. Memory context (recent conversation + remembered facts).
3. Tool output (from CRM / RAG / Web Search, or none for direct).
4. The route that was taken.

Your job: produce a **natural, helpful reply** that:
• Directly answers the user's question or confirms the action taken.
• Incorporates tool output seamlessly (don't dump raw data).
• Uses remembered facts and conversation history for personalisation.
• Follows the communication rules (warm, professional, concise).
• Never mentions internal route names, tool names, or system details.
• Asks clarifying questions when necessary information is missing.
"""

_SYNTHESISER_USER_FALLBACK = """\
MEMORY CONTEXT:
{memory_context}

ROUTE TAKEN: {route}
TOOL OUTPUT:
{tool_output}

USER MESSAGE:
{user_message}

Compose your reply:"""


# Prompt builders - fetch from LangFuse, fall back to local


def build_router_prompt(
    user_message: str,
    memory_context: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the router call."""
    system_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["router_system"],
        fallback=_ROUTER_SYSTEM_FALLBACK,
    )
    user_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["router_user"],
        fallback=_ROUTER_USER_FALLBACK,
        memory_context=memory_context or "(no memory context)",
        user_message=user_message,
    )
    return system_prompt, user_prompt


def build_synthesiser_prompt(
    user_message: str,
    memory_context: str,
    route: str,
    tool_output: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the synthesiser call."""
    agent_system = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["agent_system"],
        fallback=_AGENT_SYSTEM_FALLBACK,
    )
    synth_system = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["synthesiser_system"],
        fallback=_SYNTHESISER_SYSTEM_FALLBACK,
    )
    user_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["synthesiser_user"],
        fallback=_SYNTHESISER_USER_FALLBACK,
        memory_context=memory_context or "(no memory context)",
        route=route,
        tool_output=tool_output or "(no tool output - direct response)",
        user_message=user_message,
    )
    combined_system = agent_system + "\n\n" + synth_system
    return combined_system, user_prompt
