import { useCallback, useEffect, useRef, useState } from "react";
import { chatApi, ApiError } from "@/api/client";
import type { UIMessage } from "@/types";

/**
 * The API is sync — one request / one response. To give the user a
 * sense of what's happening inside the agent, we cycle through a
 * scripted set of "stages" while the call is in flight. On success the
 * real metadata (route, cached, latency) replaces the guesswork.
 */
export const THINKING_STAGES = [
  { id: "cache", label: "Checking semantic cache" },
  { id: "recall", label: "Recalling conversation memory" },
  { id: "route", label: "Classifying intent" },
  { id: "tool", label: "Invoking tools / retrieving context" },
  { id: "synth", label: "Synthesising response" },
] as const;

export type ThinkingStageId = (typeof THINKING_STAGES)[number]["id"];

interface UseChatArgs {
  userId: string;
  sessionId: string;
  onMessage?: (m: UIMessage) => void;
}

export function useChat({ userId, sessionId, onMessage }: UseChatArgs) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [stageIdx, setStageIdx] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const stageTimer = useRef<number | null>(null);

  // ── Reset + load history when user or session changes ─────────────
  // Without this, switching to a "new conversation" left the previous
  // session's messages on screen because the hook's state outlives the
  // ChatWindow component. Now: clear immediately, then fetch the
  // session's persisted ST turns so previous chats render naturally.
  useEffect(() => {
    let cancelled = false;
    setMessages([]);
    setError(null);
    if (!userId || !sessionId) return;

    void chatApi
      .sessionTurns(sessionId, userId, 50)
      .then((res) => {
        if (cancelled || !res?.turns) return;
        setMessages(
          res.turns.map((t) => ({
            id: crypto.randomUUID(),
            role: t.role as "user" | "assistant",
            content: t.content,
            ts: t.ts,
          })),
        );
      })
      .catch(() => { /* silent — empty history is fine */ });

    return () => { cancelled = true; };
  }, [userId, sessionId]);

  const startStageTicker = () => {
    setStageIdx(0);
    let i = 0;
    // Gently advance through the scripted stages while the API call is
    // pending. Don't auto-advance past the last — hold the final stage
    // until the response lands.
    stageTimer.current = window.setInterval(() => {
      i = Math.min(i + 1, THINKING_STAGES.length - 1);
      setStageIdx(i);
    }, 550);
  };

  const stopStageTicker = () => {
    if (stageTimer.current != null) {
      window.clearInterval(stageTimer.current);
      stageTimer.current = null;
    }
    setStageIdx(null);
  };

  const send = useCallback(async (text: string) => {
    if (!text.trim() || loading) return;
    setError(null);

    const userMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      ts: Date.now() / 1000,
    };
    setMessages((prev) => [...prev, userMsg]);
    onMessage?.(userMsg);

    setLoading(true);
    startStageTicker();

    try {
      const res = await chatApi.send({
        user_id: userId,
        session_id: sessionId,
        message: text,
      });
      const botMsg: UIMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: res.answer,
        ts: Date.now() / 1000,
        meta: {
          route: res.route,
          routes: res.routes,
          cached: res.cached,
          latency_ms: res.latency_ms,
          trace_id: res.trace_id,
          timings: res.timings,
          model_used: res.model_used,
        },
      };
      setMessages((prev) => [...prev, botMsg]);
      onMessage?.(botMsg);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e);
      setError(msg);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `⚠️  Request failed: ${msg}`,
          ts: Date.now() / 1000,
        },
      ]);
    } finally {
      stopStageTicker();
      setLoading(false);
    }
  }, [loading, userId, sessionId, onMessage]);

  const reset = useCallback(async () => {
    try {
      await chatApi.reset(userId, sessionId);
    } catch {
      /* ignore — just clearing client state too */
    }
    setMessages([]);
    setError(null);
  }, [userId, sessionId]);

  return {
    messages,
    setMessages,
    loading,
    error,
    stageIdx,
    send,
    reset,
  };
}
