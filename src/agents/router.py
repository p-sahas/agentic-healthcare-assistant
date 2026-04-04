"""
Query Router — LLM-based intent classification.

Takes a user message + memory context and returns a ``MultiRouteDecision``
containing one or more ``RouteDecision`` objects.  When the user query
contains multiple independent intents (e.g. "Check my appointments AND
what is the infection control policy?") the router returns multiple
routes so the orchestrator can fan out to parallel agent nodes.
"""

import json
from loguru import logger
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

# Maximum routes per query (safety cap)
MAX_ROUTES = 3


@dataclass
class RouteDecision:
    """
    A single routing decision for one intent.

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


@dataclass
class MultiRouteDecision:
    """
    Container for one or more RouteDecision objects.

    Single-intent queries produce ``decisions`` with one element.
    Multi-intent queries (e.g. "book me an appointment AND tell me
    about infection control") produce multiple elements, enabling
    LangGraph fan-out to parallel agent nodes.
    """

    decisions: List[RouteDecision] = field(default_factory=list)

    @property
    def is_multi_route(self) -> bool:
        return len(self.decisions) > 1

    @property
    def primary(self) -> RouteDecision:
        """First (or only) decision — backward compatibility."""
        return self.decisions[0] if self.decisions else RouteDecision()


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
    ) -> MultiRouteDecision:
        """
        Classify user intent and extract parameters.

        Returns a ``MultiRouteDecision`` containing one or more
        ``RouteDecision`` objects.  For most queries only one route
        is returned; multi-route is triggered only when the user
        asks clearly separate questions in one message.

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
            return MultiRouteDecision(decisions=[
                RouteDecision(
                    route="direct",
                    confidence=0.0,
                    reasoning=f"Router LLM error: {exc}",
                )
            ])

        return self._parse_response(content)

    def _model_name(self) -> str:
        """Extract model name from the LLM for LangFuse metadata."""
        if hasattr(self.llm, "model_name"):
            return self.llm.model_name
        if hasattr(self.llm, "model"):
            return self.llm.model
        return "unknown"

    # ── parsing ───────────────────────────────────────────────

    def _parse_response(self, raw: str) -> MultiRouteDecision:
        """
        Parse the JSON response from the router LLM.

        Supports two formats:
          - Multi-route (new):  ``{"routes": [{...}, {...}]}``
          - Single-route (old): ``{"route": "crm", ...}``

        The old format is auto-wrapped into a single-element list
        for full backward compatibility.
        """
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Locate JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            logger.warning("Router output is not JSON; falling back to direct.")
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0,
                              reasoning="Failed to parse router output as JSON.")
            ])

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("Router JSON parse error: {}", exc)
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0,
                              reasoning=f"JSON parse error: {exc}")
            ])

        # ── Normalise to a list of route dicts ──────────────────
        if "routes" in data and isinstance(data["routes"], list):
            # New multi-route format
            route_dicts = data["routes"][:MAX_ROUTES]
        else:
            # Old single-route format — wrap in list
            route_dicts = [data]

        # ── Build RouteDecision objects ──────────────────────────
        decisions: List[RouteDecision] = []
        seen_routes: set = set()

        for rd in route_dicts:
            route = rd.get("route", "direct")
            if route not in VALID_ROUTES:
                logger.warning("Invalid route '{}'; skipping.", route)
                continue
            # Deduplicate (same route appearing twice)
            if route in seen_routes:
                continue
            seen_routes.add(route)

            action = rd.get("action")
            if route == "crm" and action not in VALID_CRM_ACTIONS:
                logger.warning(
                    "Invalid CRM action '{}'; defaulting to lookup_patient.", action
                )
                action = "lookup_patient"

            decisions.append(RouteDecision(
                route=route,
                confidence=float(rd.get("confidence", 0.5)),
                reasoning=rd.get("reasoning", ""),
                action=action if route == "crm" else None,
                params=rd.get("params", {}),
            ))

        # Fallback if nothing valid was parsed
        if not decisions:
            decisions = [RouteDecision(route="direct", confidence=0.0,
                                       reasoning="No valid routes parsed.")]

        return MultiRouteDecision(decisions=decisions)
