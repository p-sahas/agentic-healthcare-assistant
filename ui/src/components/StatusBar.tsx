import { useState } from "react";
import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, CircleDot, ChevronDown } from "lucide-react";
import type { ConfigResponse, ReadinessResponse } from "@/types";

interface Props {
  status: "unknown" | "ok" | "starting" | "degraded" | "offline";
  readiness: ReadinessResponse | null;
  config: ConfigResponse | null;
}

const STATUS_META: Record<Props["status"], { label: string; color: string }> = {
  ok: { label: "Healthy", color: "text-success" },
  starting: { label: "Starting", color: "text-warn" },
  degraded: { label: "Degraded", color: "text-warn" },
  offline: { label: "Offline", color: "text-danger" },
  unknown: { label: "Unknown", color: "text-slate-500" },
};

export function StatusBar({ status, readiness, config }: Props) {
  const [open, setOpen] = useState(false);
  const meta = STATUS_META[status];

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border text-xs hover:bg-bg-soft"
      >
        <CircleDot
          size={12}
          className={clsx(meta.color, status === "ok" && "animate-pulse-slow")}
        />
        <span className={meta.color}>{meta.label}</span>
        {config && (
          <span className="hidden sm:inline text-slate-500 border-l border-border pl-2">
            {config.chat_model.split("/").pop()}
          </span>
        )}
        <ChevronDown size={12} className={clsx("text-slate-500 transition-transform", open && "rotate-180")} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="absolute right-0 top-full mt-2 w-80 card p-3 z-20 space-y-3 text-xs"
          >
            {/* readiness checks */}
            <div>
              <div className="text-slate-400 mb-1 flex items-center gap-1">
                <Activity size={12} /> Readiness
              </div>
              <div className="space-y-1">
                {(readiness?.checks ?? []).map((c) => (
                  <div key={c.name} className="flex items-center gap-2">
                    <CircleDot size={10} className={c.ok ? "text-success" : "text-danger"} />
                    <span className="text-slate-300 capitalize">{c.name}</span>
                    <span className="text-slate-500 ml-auto truncate">{c.detail ?? ""}</span>
                  </div>
                ))}
                {!readiness && (
                  <div className="text-slate-500">fetching…</div>
                )}
              </div>
            </div>

            {/* config */}
            <div>
              <div className="text-slate-400 mb-1">Models</div>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[11px] text-slate-300">
                {config ? (
                  <>
                    <Row label="chat" value={config.chat_model} />
                    <Row label="router" value={config.router_model} />
                    <Row label="extract" value={config.extractor_model} />
                    <Row label="embed" value={config.embedding_model} />
                    <Row label="provider" value={config.provider} />
                  </>
                ) : (
                  <span className="text-slate-500 col-span-2">fetching…</span>
                )}
              </dl>
            </div>

            {/* tools enabled */}
            {config && (
              <div>
                <div className="text-slate-400 mb-1">Tools enabled</div>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(config.tools_enabled).map(([k, v]) => (
                    <span
                      key={k}
                      className={clsx(
                        "chip border",
                        v ? "bg-success/10 border-success/30 text-success" : "bg-bg-soft border-border text-slate-500",
                      )}
                    >
                      {k}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-slate-500">{label}</dt>
      <dd className="truncate" title={value}>{value}</dd>
    </>
  );
}
