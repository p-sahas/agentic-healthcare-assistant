/**
 * Thin fetch wrapper for the FastAPI backend.
 *
 * All requests go through ``/api/...`` so the Vite dev proxy can forward
 * them to the real API host. In production, serve the UI behind the same
 * domain and the proxy becomes a no-op.
 */

import type {
  CAGGetResponse,
  CAGSetResponse,
  CAGStatsResponse,
  ChatRequest,
  ChatResponse,
  ChatSessionMeta,
  ConfigResponse,
  CRMResponse,
  FactItem,
  HealthResponse,
  MemoryFactsResponse,
  MemoryRecallResponse,
  Patient,
  PatientRegisterPayload,
  PatientUpdatePayload,
  RAGResponse,
  RAGStatsResponse,
  ReadinessResponse,
  SessionTurnsResponse,
  StreamEvent,
  WebSearchResponse,
} from "@/types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(public status: number, public body: unknown, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { json?: unknown } = {},
): Promise<T> {
  const { json, headers, ...rest } = init;
  const res = await fetch(`${BASE}${path}`, {
    ...rest,
    headers: {
      "content-type": "application/json",
      ...(headers || {}),
    },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });

  if (!res.ok) {
    let body: unknown = null;
    try { body = await res.json(); } catch { /* ignore */ }
    const msg = (body as { detail?: string })?.detail || res.statusText;
    throw new ApiError(res.status, body, msg);
  }

  const text = await res.text();
  return (text ? JSON.parse(text) : null) as T;
}

// ── Chat ─────────────────────────────────────────────────────────────

export const chatApi = {
  send: (req: ChatRequest) =>
    request<ChatResponse>("/chat", { method: "POST", json: req }),
  reset: (user_id: string, session_id: string) =>
    request<{ cleared: boolean }>("/chat/reset", {
      method: "POST",
      json: { user_id, session_id },
    }),
  sessionTurns: (session_id: string, user_id: string, limit = 20) =>
    request<SessionTurnsResponse>(
      `/sessions/${encodeURIComponent(session_id)}/turns?user_id=${encodeURIComponent(user_id)}&limit=${limit}`,
    ),

  /**
   * Stream the chain of thought via Server-Sent Events. Each pipeline
   * phase fires a callback so the UI can render a live timeline.
   *
   * Returns the final ChatResponse so callers can `await` the same way
   * they would `chatApi.send()`. If anything goes wrong mid-stream the
   * promise rejects with an ApiError; the caller can fall back to the
   * non-streaming endpoint.
   */
  stream: async (
    req: ChatRequest,
    onEvent: (event: StreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<ChatResponse> => {
    const res = await fetch(`${BASE}/chat/stream`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(req),
      signal,
    });
    if (!res.ok || !res.body) {
      let body: unknown = null;
      try { body = await res.json(); } catch { /* ignore */ }
      throw new ApiError(res.status, body, (body as { detail?: string })?.detail || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finalResponse: ChatResponse | null = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // Split SSE frames (separated by blank lines).
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!frame || frame.startsWith(":")) continue; // comments / keepalives
        // SSE allows multiple `data:` lines per frame, but we only emit one.
        const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        const json = dataLine.slice("data:".length).trim();
        if (!json) continue;
        let event: StreamEvent;
        try {
          event = JSON.parse(json) as StreamEvent;
        } catch {
          continue;
        }
        onEvent(event);
        if (event.type === "final") {
          // strip the discriminator before returning a ChatResponse-shaped object
          const { type: _t, ...rest } = event;
          finalResponse = rest as unknown as ChatResponse;
        } else if (event.type === "error") {
          throw new ApiError(event.status ?? 500, null, event.message);
        }
      }
    }

    if (!finalResponse) throw new ApiError(0, null, "Stream ended without a final event");
    return finalResponse;
  },
};

export const sessionApi = {
  /**
   * Preload patient profile + recent ST turns into the server-side cache.
   * Fire this after login and on every session switch so the first
   * /chat request skips two Supabase round-trips.
   */
  warmup: (user_id: string, session_id: string) =>
    request<{
      warmed: boolean;
      patient_loaded: boolean;
      st_turn_count: number;
      latency_ms: number;
    }>("/sessions/warmup", {
      method: "POST",
      json: { user_id, session_id },
    }),
};

export const chatSessionsApi = {
  /** List a patient's conversation threads, newest activity first. */
  list: (user_id: string, include_archived = false) =>
    request<{ sessions: ChatSessionMeta[] }>(
      `/chat_sessions?user_id=${encodeURIComponent(user_id)}&include_archived=${include_archived}`,
    ),

  create: (user_id: string, title?: string, session_id?: string) =>
    request<ChatSessionMeta>("/chat_sessions", {
      method: "POST",
      json: { user_id, title, session_id },
    }),

  rename: (session_id: string, title: string) =>
    request<ChatSessionMeta>(`/chat_sessions/${encodeURIComponent(session_id)}`, {
      method: "PATCH",
      json: { title },
    }),

  archive: (session_id: string, archived: boolean) =>
    request<ChatSessionMeta>(`/chat_sessions/${encodeURIComponent(session_id)}`, {
      method: "PATCH",
      json: { archived: archived ? 1 : 0 },
    }),

  remove: (session_id: string) =>
    request<{ deleted: boolean }>(`/chat_sessions/${encodeURIComponent(session_id)}`, {
      method: "DELETE",
    }),
};

// ── System ───────────────────────────────────────────────────────────

export const systemApi = {
  health: () => request<HealthResponse>("/health"),
  ready: () => request<ReadinessResponse>("/ready"),
  config: () => request<ConfigResponse>("/config"),
};

// ── CRM ──────────────────────────────────────────────────────────────

export const crmApi = {
  lookupPatient: (body: { phone?: string; name?: string; patient_id?: string }) =>
    request<CRMResponse>("/tools/crm/lookup_patient", { method: "POST", json: body }),
  searchDoctors: (body: { specialty?: string; location?: string; doctor_name?: string }) =>
    request<CRMResponse>("/tools/crm/search_doctors", { method: "POST", json: body }),
};

// ── RAG ──────────────────────────────────────────────────────────────

export const ragApi = {
  search: (body: { query: string; top_k?: number; use_cache?: boolean }) =>
    request<RAGResponse>("/tools/rag/search", { method: "POST", json: body }),
  stats: () => request<RAGStatsResponse>("/tools/rag/stats"),
};

// ── Web ──────────────────────────────────────────────────────────────

export const webApi = {
  search: (body: { query: string; max_results?: number }) =>
    request<WebSearchResponse>("/tools/web_search", { method: "POST", json: body }),
};

// ── CAG ──────────────────────────────────────────────────────────────

export const cagApi = {
  get: (query: string) =>
    request<CAGGetResponse>("/tools/cag/get", { method: "POST", json: { query } }),
  set: (query: string, answer: string, evidence_urls: string[] = []) =>
    request<CAGSetResponse>("/tools/cag/set", {
      method: "POST",
      json: { query, answer, evidence_urls },
    }),
  stats: () => request<CAGStatsResponse>("/tools/cag/stats"),
  clear: () => request<{ cleared: boolean }>("/tools/cag/clear", { method: "POST" }),
};

// ── Memory ───────────────────────────────────────────────────────────

export const memoryApi = {
  recall: (user_id: string, session_id: string, query: string) =>
    request<MemoryRecallResponse>("/tools/memory/recall", {
      method: "POST",
      json: { user_id, session_id, query },
    }),
  facts: (user_id: string) =>
    request<MemoryFactsResponse>(`/tools/memory/facts/${encodeURIComponent(user_id)}`),
  storeFact: (user_id: string, text: string, tags: string[] = []) =>
    request<{ stored: boolean; fact_id: string | null }>("/tools/memory/store_fact", {
      method: "POST",
      json: { user_id, text, tags },
    }),
  distill: (user_id: string, session_id: string) =>
    request<{ distilled_count: number; triggered: boolean }>(
      "/tools/memory/distill",
      { method: "POST", json: { user_id, session_id } },
    ),
};

// ── Patients (phone-based identity) ──────────────────────────────────

export const patientApi = {
  /**
   * Look up an existing patient by phone. Throws ApiError(404) if none.
   * The UI catches the 404 to switch to the registration form.
   */
  lookup: (phone: string) =>
    request<Patient>("/patients/lookup", { method: "POST", json: { phone } }),

  /**
   * Create a new patient. Throws ApiError(409) if the phone is taken.
   */
  register: (payload: PatientRegisterPayload) =>
    request<Patient>("/patients/register", { method: "POST", json: payload }),

  get: (patient_id: string) =>
    request<Patient>(`/patients/${encodeURIComponent(patient_id)}`),

  update: (patient_id: string, payload: PatientUpdatePayload) =>
    request<Patient>(`/patients/${encodeURIComponent(patient_id)}`, {
      method: "PUT",
      json: payload,
    }),
};

// Re-export types actually used by callers
export type { FactItem };
