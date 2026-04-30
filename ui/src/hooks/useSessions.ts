import { useCallback, useEffect, useState } from "react";
import { chatSessionsApi } from "@/api/client";
import type { ChatSessionMeta } from "@/types";

export type SessionMeta = ChatSessionMeta;

const LS_ACTIVE = "nawaloka.sessions.active";

/**
 * Server-backed session registry.
 *
 * Sessions live in Supabase (``chat_sessions`` table) so they persist
 * across browsers and survive ``localStorage`` clears. Only the
 * "currently selected session" id is cached locally so a refresh lands
 * the user back where they were.
 *
 * Loading: when ``patientId`` becomes known we fetch the patient's
 * sessions. Sidebar shows them newest-message-first.
 *
 * Empty state: if the patient has no sessions yet, we lazily create
 * the first one so the user always has somewhere to type.
 */
export function useSessions(patientId: string | undefined | null) {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeId, setActiveIdState] = useState<string>("");
  const [loaded, setLoaded] = useState(false);

  // Persist the chosen activeId so a refresh keeps you on the same chat.
  const setActiveId = useCallback((id: string) => {
    setActiveIdState(id);
    try { localStorage.setItem(LS_ACTIVE, id); } catch { /* ignore */ }
  }, []);

  const refresh = useCallback(async () => {
    if (!patientId) return;
    try {
      const { sessions: list } = await chatSessionsApi.list(patientId);
      setSessions(list);
      // If our cached active id is gone (deleted / new patient), pick first.
      const cachedActive = (() => {
        try { return localStorage.getItem(LS_ACTIVE) || ""; } catch { return ""; }
      })();
      const stillExists = list.some((s) => s.session_id === cachedActive);
      if (!stillExists) {
        if (list.length > 0) setActiveId(list[0].session_id);
        else setActiveIdState("");          // will trigger lazy-create below
      } else if (cachedActive) {
        setActiveIdState(cachedActive);
      }
    } catch (e) {
      // Non-fatal — sidebar can still operate offline-ish on stale state
      // eslint-disable-next-line no-console
      console.warn("Failed to fetch chat sessions:", e);
    } finally {
      // Set loaded=true AFTER the fetch resolves. The lazy-create
      // effect below gates on this flag, so flipping it earlier
      // (e.g. while refresh() is still in flight) caused a race
      // where every patient-id hydrate spawned a fresh empty
      // session before the existing-session list could populate.
      setLoaded(true);
    }
  }, [patientId, setActiveId]);

  // Initial load on patient change.
  useEffect(() => {
    if (!patientId) {
      setSessions([]);
      setActiveIdState("");
      setLoaded(true);
      return;
    }
    // Critical: reset loaded BEFORE kicking off the async refresh,
    // so the lazy-create effect (which gates on loaded) cannot fire
    // during the network round-trip. Without this, every refresh
    // hydrate cycle would spam-create empty sessions in the DB
    // because sessions=[] / activeId="" / loaded=true are all true
    // for the brief window between patientId arriving and the list
    // fetch resolving.
    setLoaded(false);
    void refresh();
  }, [patientId, refresh]);

  // Lazy-create a first session when the patient has none.
  useEffect(() => {
    if (!loaded || !patientId) return;
    if (sessions.length > 0) return;
    if (activeId) return;
    void chatSessionsApi.create(patientId).then((row) => {
      setSessions([row]);
      setActiveId(row.session_id);
    }).catch((e) => {
      // eslint-disable-next-line no-console
      console.warn("Failed to create initial chat session:", e);
    });
  }, [loaded, patientId, sessions.length, activeId, setActiveId]);

  const create = useCallback(async (title?: string) => {
    if (!patientId) return null;
    const row = await chatSessionsApi.create(patientId, title);
    setSessions((prev) => [row, ...prev.filter((s) => s.session_id !== row.session_id)]);
    setActiveId(row.session_id);
    return row;
  }, [patientId, setActiveId]);

  const remove = useCallback(async (session_id: string) => {
    if (!patientId) return;
    try {
      await chatSessionsApi.remove(session_id);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("Failed to delete session:", e);
      return;
    }
    setSessions((prev) => {
      const next = prev.filter((s) => s.session_id !== session_id);
      if (session_id === activeId) {
        setActiveId(next[0]?.session_id ?? "");
      }
      return next;
    });
  }, [patientId, activeId, setActiveId]);

  const rename = useCallback(async (session_id: string, title: string) => {
    try {
      const row = await chatSessionsApi.rename(session_id, title);
      setSessions((prev) => prev.map((s) => (s.session_id === session_id ? row : s)));
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn("Failed to rename session:", e);
    }
  }, []);

  return {
    sessions,
    activeId,
    setActiveId,
    create,
    remove,
    rename,
    refresh,
    loaded,
  };
}
