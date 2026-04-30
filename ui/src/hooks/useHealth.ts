import { useEffect, useRef, useState } from "react";
import { systemApi } from "@/api/client";
import type { ConfigResponse, ReadinessResponse } from "@/types";

/**
 * Health/readiness polling for the status bar.
 *
 * - `/health` polls every ``intervalMs`` (default **30s**) — light request.
 * - `/config` is fetched once at mount.
 * - `/ready` is refetched only when the health status transitions
 *   (e.g. ok → offline) — not on every health tick.
 * - Polling pauses automatically when the tab is hidden (Page
 *   Visibility API) so a backgrounded tab does not keep hammering the
 *   server. Resumes immediately on visibility return.
 */
export function useHealth(intervalMs = 30_000) {
  const [status, setStatus] = useState<"unknown" | "ok" | "starting" | "degraded" | "offline">("unknown");
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const lastStatus = useRef<string>("");

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;

    const poll = async () => {
      if (document.hidden) return; // skip while tab not visible
      try {
        const h = await systemApi.health();
        if (!cancelled) setStatus(h.status);
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };

    const start = () => {
      if (timer != null) return;
      void poll();
      timer = window.setInterval(poll, intervalMs);
    };
    const stop = () => {
      if (timer != null) {
        window.clearInterval(timer);
        timer = null;
      }
    };

    const onVisibility = () => {
      if (document.hidden) stop();
      else start();
    };

    start();
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [intervalMs]);

  // /config once; /ready only on health-status transitions
  useEffect(() => {
    systemApi.config().then(setConfig).catch(() => { /* non-fatal */ });
  }, []);

  useEffect(() => {
    if (status === "unknown" || status === lastStatus.current) return;
    lastStatus.current = status;
    systemApi.ready().then(setReadiness).catch(() => { /* non-fatal */ });
  }, [status]);

  return { status, readiness, config };
}
