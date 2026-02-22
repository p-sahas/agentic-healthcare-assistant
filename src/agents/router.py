"""
Query Router — LLM-based intent classification.

Takes a user message + memory context and returns a ``RouteDecision``
that tells the orchestrator which tool to invoke and with what params.
"""

import json
from loguru import logger
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agents.prompts.agent_prompts import build_router_prompt
from infrastructure.observability import observe, update_current_observation
# Valid routes
VALID_ROUTES = {"crm", "rag", "web_search", "direct"}

# Valid CRM sub-actions
VALID_CRM_ACTIONS = {
    "lookup_patient",
    "search_doctors",
    "create_booking",
    "cancel_booking",
    "reschedule_booking",
}


@dataclass
class RouteDecision:
    """
    Output of the router LLM call.

    Attributes:
        route: Primary route (crm | rag | web_search | direct).
        confidence: Router's self-assessed confidence [0-1].
        reasoning: One-line explanation of the routing decision.
        action: CRM sub-action (only when route == crm).
        params: Extracted parameters for the tool.
    """

    route: str = "direct"
    confidence: float = 0.0
    reasoning: str = ""
    action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


class QueryRouter:
    """
    Routes user queries to the appropriate tool path.

    Uses an LLM call with structured JSON output to classify intent.
    Falls back to ``direct`` on parse errors.
    """

    def __init__(self, llm: Any) -> None:
        """
        Args:
            llm: A LangChain ``ChatOpenAI`` (or compatible) instance.
        """
        self.llm = llm

    @observe(name="router", as_type="generation")
    def route(
        self,
        user_message: str,
        memory_context: str = "",
    ) -> RouteDecision:
        """
        Classify user intent and extract parameters.

        Traced as a LangFuse **generation** so cost/tokens are captured.
        """
        system_prompt, user_prompt = build_router_prompt(
            user_message=user_message,
            memory_context=memory_context,
        )

        update_current_observation(
            input=user_prompt[:1000],
            model=self._model_name(),
        )

        try:
            response = self.llm.invoke(
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
                output=content[:500],
                usage=usage if usage else None,
            )

        except Exception as exc:
            logger.error("Router LLM call failed: {}", exc)
            return RouteDecision(
                route="direct",
                confidence=0.0,
                reasoning=f"Router LLM error: {exc}",
            )

        return self._parse_response(content)

    def _model_name(self) -> str:
        """Extract model name from the LLM for LangFuse metadata."""
        if hasattr(self.llm, "model_name"):
            return self.llm.model_name
        if hasattr(self.llm, "model"):
            return self.llm.model
        return "unknown"

    # ── parsing ───────────────────────────────────────────────

    def _parse_response(self, raw: str) -> RouteDecision:
        """
        Parse the JSON response from the router LLM.

        Handles markdown fences and partial JSON gracefully.
        """
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]  # drop first line
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Locate JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            logger.warning("Router output is not JSON; falling back to direct.")
            return RouteDecision(
                route="direct",
                confidence=0.0,
                reasoning="Failed to parse router output as JSON.",
            )

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("Router JSON parse error: {}", exc)
            return RouteDecision(
                route="direct",
                confidence=0.0,
                reasoning=f"JSON parse error: {exc}",
            )

        # Validate
        route = data.get("route", "direct")
        if route not in VALID_ROUTES:
            logger.warning("Invalid route '{}'; falling back to direct.", route)
            route = "direct"

        action = data.get("action")
        if route == "crm" and action not in VALID_CRM_ACTIONS:
            logger.warning(
                "Invalid CRM action '{}'; defaulting to lookup_patient.", action
            )
            action = "lookup_patient"

        return RouteDecision(
            route=route,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            action=action if route == "crm" else None,
            params=data.get("params", {}),
        )
