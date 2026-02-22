"""
Agentic Routing Engine — the core agent module.

Public API:
    build_agent()        → AgentOrchestrator (fully wired, ready to chat)
    AgentOrchestrator    → main orchestrator class
    AgentResponse        → response dataclass
    QueryRouter          → intent classifier
    RouteDecision        → routing result dataclass
"""

from .orchestrator import AgentOrchestrator, AgentResponse, build_agent
from .router import QueryRouter, RouteDecision

__all__ = [
    "AgentOrchestrator",
    "AgentResponse",
    "QueryRouter",
    "RouteDecision",
    "build_agent",
]
