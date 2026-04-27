"""
Pydantic request / response schemas for the Nawaloka Health Assistant API.

Organized by router: chat, health, and the per-tool groups (crm/rag/web/
cag/memory/crawl). All schemas are validated at the FastAPI boundary so
internal code can assume clean inputs.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# Chat
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """POST /chat"""
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, description="User's natural-language message")


class ChatResponse(BaseModel):
    """POST /chat response."""
    answer: str
    route: Literal["cag_hit", "crm", "rag", "web_search", "direct", "multi", "out_of_scope"]
    routes: List[str] = Field(default_factory=list, description="All routes taken for multi-intent queries")
    cached: bool = False
    latency_ms: int = 0
    trace_id: Optional[str] = None
    timings: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-node wall-clock latency in ms (cag, recall, route, tool, synth, save).",
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Which LLM produced the final answer (fast / chat).",
    )


class ChatResetRequest(BaseModel):
    """POST /chat/reset — clear ST memory for a session."""
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class ChatResetResponse(BaseModel):
    cleared: bool = True
    user_id: str
    session_id: str


class TurnItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    ts: float


class SessionTurnsResponse(BaseModel):
    user_id: str
    session_id: str
    turn_count: int
    turns: List[TurnItem] = Field(default_factory=list)


class SessionWarmupRequest(BaseModel):
    """POST /sessions/warmup — preload patient + ST turns into the server cache."""
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class SessionWarmupResponse(BaseModel):
    warmed: bool = True
    patient_loaded: bool = False
    st_turn_count: int = 0
    latency_ms: int = 0


# ── Chat sessions (ChatGPT-style sidebar) ──────────────────────────

class ChatSessionMeta(BaseModel):
    """One row in the patient's session list."""
    session_id: str
    patient_id: str
    title: str
    last_message_at: Optional[int] = None
    created_at: int
    updated_at: int
    archived: int = 0


class ChatSessionListResponse(BaseModel):
    sessions: List[ChatSessionMeta] = Field(default_factory=list)


class ChatSessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    title: Optional[str] = None             # backend supplies a default if absent
    session_id: Optional[str] = None        # client may suggest one; backend dedupes


class ChatSessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    archived: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════
# Health / System
# ═══════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "degraded"] = "ok"


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: Optional[str] = None


class ReadinessResponse(BaseModel):
    ready: bool
    checks: List[ReadinessCheck] = Field(default_factory=list)


class ConfigResponse(BaseModel):
    chat_model: str
    router_model: str
    extractor_model: str
    embedding_model: str
    provider: str
    tools_enabled: Dict[str, bool]


# ═══════════════════════════════════════════════════════════════════
# CRM tool
# ═══════════════════════════════════════════════════════════════════

class CRMLookupPatientRequest(BaseModel):
    phone: Optional[str] = None
    name: Optional[str] = None
    patient_id: Optional[str] = None
    external_user_id: Optional[str] = None


class CRMSearchDoctorsRequest(BaseModel):
    specialty: Optional[str] = None
    location: Optional[str] = None
    doctor_name: Optional[str] = None
    available_from: Optional[str] = None
    available_to: Optional[str] = None


class CRMCreateBookingRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    doctor_id: str = Field(..., min_length=1)
    start_time: str = Field(..., description="ISO-8601 datetime")
    duration_minutes: int = 30
    notes: Optional[str] = None


class CRMCancelBookingRequest(BaseModel):
    booking_id: str = Field(..., min_length=1)
    reason: Optional[str] = None


class CRMRescheduleBookingRequest(BaseModel):
    booking_id: str = Field(..., min_length=1)
    new_start_time: str = Field(..., min_length=1)


class CRMResponse(BaseModel):
    result: str
    latency_ms: int = 0


# ═══════════════════════════════════════════════════════════════════
# RAG tool
# ═══════════════════════════════════════════════════════════════════

class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 4
    threshold: float = 0.5
    use_cache: bool = True


class RAGResponse(BaseModel):
    result: str
    latency_ms: int = 0


class RAGStatsResponse(BaseModel):
    stats: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Web search tool
# ═══════════════════════════════════════════════════════════════════

class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = 5


class WebSearchResponse(BaseModel):
    result: str
    latency_ms: int = 0


# ═══════════════════════════════════════════════════════════════════
# CAG cache tool
# ═══════════════════════════════════════════════════════════════════

class CAGGetRequest(BaseModel):
    query: str = Field(..., min_length=1)


class CAGGetResponse(BaseModel):
    hit: bool
    query: str = ""
    answer: str = ""
    evidence_urls: List[str] = Field(default_factory=list)
    score: float = 0.0
    ts: float = 0.0


class CAGSetRequest(BaseModel):
    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    evidence_urls: List[str] = Field(default_factory=list)


class CAGSetResponse(BaseModel):
    cached: bool = True
    query: str


class CAGStatsResponse(BaseModel):
    stats: Dict[str, Any] = Field(default_factory=dict)


class CAGClearResponse(BaseModel):
    cleared: bool = True


# ═══════════════════════════════════════════════════════════════════
# Memory tool
# ═══════════════════════════════════════════════════════════════════

class MemoryRecallRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class FactItem(BaseModel):
    id: Optional[str] = None
    text: str
    tags: List[str] = Field(default_factory=list)
    score: float = 0.0


class MemoryRecallResponse(BaseModel):
    st_turns: List[TurnItem] = Field(default_factory=list)
    lt_facts: List[FactItem] = Field(default_factory=list)


class MemoryFactsResponse(BaseModel):
    user_id: str
    fact_count: int
    facts: List[FactItem] = Field(default_factory=list)


class MemoryStoreFactRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)


class MemoryStoreFactResponse(BaseModel):
    stored: bool = True
    fact_id: Optional[str] = None


class MemoryDistillRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class MemoryDistillResponse(BaseModel):
    distilled_count: int = 0
    triggered: bool


# ═══════════════════════════════════════════════════════════════════
# Crawler tool
# ═══════════════════════════════════════════════════════════════════

class CrawlRequest(BaseModel):
    start_urls: List[str] = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    max_depth: int = 2
    exclude_patterns: List[str] = Field(default_factory=list)
    request_delay: float = 2.0


class CrawledDoc(BaseModel):
    url: str
    title: str = ""
    headings: List[str] = Field(default_factory=list)
    content: str = ""
    depth_level: int = 0


class CrawlResponse(BaseModel):
    doc_count: int
    docs: List[CrawledDoc] = Field(default_factory=list)
    latency_ms: int = 0


# ═══════════════════════════════════════════════════════════════════
# Patients (phone-based identity, not auth)
# ═══════════════════════════════════════════════════════════════════

Gender = Literal["M", "F", "X"]


class PatientLookupRequest(BaseModel):
    """POST /patients/lookup — phone-based "who is this?"."""
    phone: str = Field(..., min_length=4, description="Any common phone format; normalized server-side")


class PatientRegisterRequest(BaseModel):
    """POST /patients/register — required fields at sign-up."""
    full_name: str = Field(..., min_length=1, max_length=200)
    phone: str = Field(..., min_length=4)
    dob: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    gender: Gender


class PatientUpdateRequest(BaseModel):
    """PUT /patients/{patient_id} — profile screen edits."""
    email: Optional[str] = Field(default=None, max_length=200)
    notes: Optional[str] = Field(default=None, max_length=2000)


class PatientResponse(BaseModel):
    """Canonical patient record returned by every /patients/* endpoint."""
    patient_id: str
    full_name: str
    phone: str                  # display form ("+94…")
    dob: Optional[str] = None
    gender: Optional[Gender] = None
    email: Optional[str] = None
    notes: Optional[str] = None
    active: int = 1
    created_at: int = 0
    updated_at: int = 0


# ═══════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
