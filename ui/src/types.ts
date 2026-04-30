/**
 * TypeScript mirror of `src/api/schemas.py`. Any schema change on the
 * backend should be reflected here — we don't codegen in v1.
 */

// ── Chat ─────────────────────────────────────────────────────────────

export type Route =
  | "cag_hit"
  | "crm"
  | "rag"
  | "web_search"
  | "direct"
  | "multi"
  | "out_of_scope";

export interface ChatRequest {
  user_id: string;
  session_id: string;
  message: string;
}

export interface ChatResponse {
  answer: string;
  route: Route;
  routes: string[];
  cached: boolean;
  latency_ms: number;
  trace_id: string | null;
  timings?: Record<string, number>;
  model_used?: string | null;
}

export interface TurnItem {
  role: "user" | "assistant";
  content: string;
  ts: number;
}

export interface SessionTurnsResponse {
  user_id: string;
  session_id: string;
  turn_count: number;
  turns: TurnItem[];
}

// ── Health ───────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "ok" | "starting" | "degraded";
}

export interface ReadinessCheck {
  name: string;
  ok: boolean;
  detail?: string | null;
}

export interface ReadinessResponse {
  ready: boolean;
  checks: ReadinessCheck[];
}

export interface ConfigResponse {
  chat_model: string;
  router_model: string;
  extractor_model: string;
  embedding_model: string;
  provider: string;
  tools_enabled: Record<string, boolean>;
}

// ── CRM ──────────────────────────────────────────────────────────────

export interface CRMResponse {
  result: string;
  latency_ms: number;
}

// ── RAG ──────────────────────────────────────────────────────────────

export interface RAGResponse {
  result: string;
  latency_ms: number;
}

export interface RAGStatsResponse {
  stats: Record<string, unknown>;
}

// ── Web ──────────────────────────────────────────────────────────────

export interface WebSearchResponse {
  result: string;
  latency_ms: number;
}

// ── CAG ──────────────────────────────────────────────────────────────

export interface CAGGetResponse {
  hit: boolean;
  query: string;
  answer: string;
  evidence_urls: string[];
  score: number;
  ts: number;
}

export interface CAGSetResponse {
  cached: boolean;
  query: string;
}

export interface CAGStatsResponse {
  stats: Record<string, unknown>;
}

// ── Memory ───────────────────────────────────────────────────────────

export interface FactItem {
  id?: string | null;
  text: string;
  tags: string[];
  score: number;
}

export interface MemoryRecallResponse {
  st_turns: TurnItem[];
  lt_facts: FactItem[];
}

export interface MemoryFactsResponse {
  user_id: string;
  fact_count: number;
  facts: FactItem[];
}

// ── Patients (phone-based identity, not auth) ────────────────────────

export type Gender = "M" | "F" | "X";

export interface Patient {
  patient_id: string;
  full_name: string;
  phone: string;
  dob: string | null;
  gender: Gender | null;
  email: string | null;
  notes: string | null;
  active: number;
  created_at: number;
  updated_at: number;
}

export interface PatientRegisterPayload {
  full_name: string;
  phone: string;
  dob: string;          // YYYY-MM-DD
  gender: Gender;
}

export interface PatientUpdatePayload {
  email?: string | null;
  notes?: string | null;
}

// ── Chat sessions (ChatGPT-style threads) ────────────────────────────

export interface ChatSessionMeta {
  session_id: string;
  patient_id: string;
  title: string;
  last_message_at: number | null;
  created_at: number;
  updated_at: number;
  archived: number;
}

// ── /chat/stream events ──────────────────────────────────────────────

export type StageId =
  | "cache" | "recall_st" | "recall_lt"
  | "patient" | "route" | "synth" | "save";

export interface StageStartEvent {
  type: "stage_start";
  stage: StageId;
  label: string;
  detail?: Record<string, unknown>;
}

export interface StageDoneEvent {
  type: "stage_done";
  stage: StageId;
  ms: number;
  detail?: Record<string, unknown>;
}

export interface ToolInvokeEvent {
  type: "tool_invoke";
  route: string;
  action: string | null;
  label: string;
}

export interface ToolDoneEvent {
  type: "tool_done";
  route: string;
  action: string | null;
  ms: number;
  summary?: string;
}

export interface FinalEvent extends ChatResponse {
  type: "final";
}

export interface ErrorEvent {
  type: "error";
  status?: number;
  message: string;
}

export type StreamEvent =
  | StageStartEvent | StageDoneEvent
  | ToolInvokeEvent | ToolDoneEvent
  | FinalEvent | ErrorEvent;

// ── UI-local types (not from backend) ────────────────────────────────

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts: number;
  meta?: {
    route: Route;
    routes: string[];
    cached: boolean;
    latency_ms: number;
    trace_id: string | null;
    timings?: Record<string, number>;
    model_used?: string | null;
  };
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  messages: UIMessage[];
}
