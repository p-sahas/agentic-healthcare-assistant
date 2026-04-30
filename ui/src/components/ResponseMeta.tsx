import { useState } from "react";
import clsx from "clsx";
import {
  ChevronRight,
  Database,
  Globe,
  HeartPulse,
  LayoutList,
  ShieldOff,
  Sparkles,
  Timer,
  Zap,
  type LucideIcon,
} from "lucide-react";
import type { Route, UIMessage } from "@/types";

interface Props {
  meta: NonNullable<UIMessage["meta"]>;
}

const ROUTE_LABELS: Record<Route, string> = {
  cag_hit: "CAG cache hit",
  crm: "CRM · admin",
  rag: "RAG · clinical KB",
  web_search: "Web · Tavily",
  direct: "Direct · no tool",
  multi: "Multi-intent fan-out",
  out_of_scope: "Guardrail · out of scope",
};

const ROUTE_ICONS: Record<Route, LucideIcon> = {
  cag_hit: Zap,
  crm: HeartPulse,
  rag: Database,
  web_search: Globe,
  direct: Sparkles,
  multi: LayoutList,
  out_of_scope: ShieldOff,
};

/**
 * Post-response metadata chip strip. Shows which route the agent chose,
 * whether the response came from the cache, and end-to-end latency.
 * Expands into a debug panel when clicked.
 */
export function ResponseMeta({ meta }: Props) {
  const [open, setOpen] = useState(false);
  const Icon = ROUTE_ICONS[meta.route] ?? Sparkles;
  const fast = meta.latency_ms < 300;
  const slow = meta.latency_ms > 2000;

  return (
    <div className="text-[11px] text-slate-400 flex flex-wrap gap-1.5 pl-1 items-center">
      <span className={clsx("chip", meta.cached
        ? "bg-success/15 text-success border border-success/30"
        : "bg-bg-soft border border-border text-slate-300")}
      >
        <Icon size={11} />
        {ROUTE_LABELS[meta.route] ?? meta.route}
      </span>

      {meta.routes.length > 1 && (
        <span className="chip bg-bg-soft border border-border text-slate-300">
          <LayoutList size={11} /> {meta.routes.length} routes
        </span>
      )}

      <span
        className={clsx(
          "chip bg-bg-soft border border-border",
          fast ? "text-success" : slow ? "text-warn" : "text-slate-300",
        )}
        title="End-to-end latency"
      >
        <Timer size={11} /> {meta.latency_ms} ms
      </span>

      {meta.cached && (
        <span className="chip bg-brand-500/10 border border-brand-500/30 text-brand-400">
          from cache
        </span>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="chip border border-border text-slate-400 hover:text-slate-200 hover:bg-bg-soft"
      >
        <ChevronRight
          size={11}
          className={clsx("transition-transform", open && "rotate-90")}
        />
        details
      </button>

      {/* Per-node timing chips — show only for non-cache responses */}
      {!meta.cached && meta.timings && Object.keys(meta.timings).length > 0 && (
        <div className="basis-full flex flex-wrap gap-1 pt-1">
          {Object.entries(meta.timings).map(([k, v]) => (
            <span
              key={k}
              className="chip bg-bg-soft border border-border text-slate-400 text-[10px]"
              title={`${k} stage latency`}
            >
              {k} <span className="text-slate-300 ml-0.5">{v}ms</span>
            </span>
          ))}
        </div>
      )}

      {open && (
        <div className="basis-full mt-2 card px-3 py-2 font-mono text-[11px] text-slate-400 space-y-1">
          <Row label="route" value={meta.route} />
          {meta.routes.length > 0 && <Row label="routes" value={meta.routes.join(", ")} />}
          <Row label="cached" value={String(meta.cached)} />
          <Row label="latency_ms" value={String(meta.latency_ms)} />
          {meta.model_used && <Row label="model_used" value={meta.model_used} />}
          {meta.trace_id && <Row label="trace_id" value={meta.trace_id} />}
          {meta.timings && Object.keys(meta.timings).length > 0 && (
            <>
              <div className="pt-1 text-slate-500">timings (ms)</div>
              {Object.entries(meta.timings).map(([k, v]) => (
                <Row key={k} label={`  ${k}`} value={String(v)} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-3">
      <span className="text-slate-500 w-24 shrink-0">{label}</span>
      <span className="text-slate-300 break-all">{value}</span>
    </div>
  );
}
