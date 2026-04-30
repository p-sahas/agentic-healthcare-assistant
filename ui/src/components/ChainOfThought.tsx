import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import {
  Brain,
  Check,
  Database,
  Globe,
  HeartPulse,
  Loader2,
  Route as RouteIcon,
  Sparkles,
  Wrench,
  Zap,
  type LucideIcon,
} from "lucide-react";
import type { ThoughtItem } from "@/hooks/useChatStream";

interface Props {
  items: ThoughtItem[];
}

/**
 * Live chain-of-thought timeline.
 *
 * Each item from ``useChatStream.thoughts`` becomes a row. Rows
 * animate in as events arrive and tick over from "running" (spinner)
 * to "done" (check + ms). The component disappears once the final
 * answer renders — at that point all events have completed.
 */
export function ChainOfThought({ items }: Props) {
  if (items.length === 0) return null;

  return (
    <div className="flex gap-3">
      <div className="shrink-0 size-8 rounded-full flex items-center justify-center bg-bg-soft border border-border text-brand-400">
        <Brain size={16} />
      </div>
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="card px-3 py-3 text-sm space-y-1.5 flex-1 max-w-[85%]"
      >
        <div className="text-xs text-slate-400 mb-1">Thinking…</div>
        <AnimatePresence initial={false}>
          {items.map((item) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className={clsx(
                "flex items-start gap-2 text-sm",
                item.status === "done" ? "text-slate-300" : "text-slate-100",
              )}
            >
              <StatusBadge item={item} />
              <div className="flex-1 min-w-0">
                <div className="leading-snug">
                  <span className="text-slate-200">{item.label}</span>
                  {item.status === "done" && typeof item.ms === "number" && (
                    <span className="text-[10px] text-slate-500 ml-2">{item.ms} ms</span>
                  )}
                </div>
                {item.detail && item.status === "done" && (
                  <div className="text-[11px] text-slate-500 truncate">{item.detail}</div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}


function StatusBadge({ item }: { item: ThoughtItem }) {
  const Icon = pickIcon(item);
  const colour = item.status === "done"
    ? "bg-success/15 border-success/40 text-success"
    : item.status === "error"
      ? "bg-danger/15 border-danger/40 text-danger"
      : "bg-brand-500/15 border-brand-500/40 text-brand-400";

  return (
    <div
      className={clsx(
        "shrink-0 mt-0.5 size-5 rounded-md flex items-center justify-center border",
        colour,
      )}
    >
      {item.status === "running"
        ? <Loader2 size={12} className="animate-spin" />
        : item.status === "done"
          ? <Check size={12} />
          : <Icon size={12} />}
    </div>
  );
}


function pickIcon(item: ThoughtItem): LucideIcon {
  if (item.type === "tool") {
    if (item.matchKey.startsWith("tool:crm")) return HeartPulse;
    if (item.matchKey.startsWith("tool:rag")) return Database;
    if (item.matchKey.startsWith("tool:web")) return Globe;
    if (item.matchKey.startsWith("tool:cag")) return Zap;
    return Wrench;
  }
  // stage: pick by stage id
  if (item.matchKey === "stage:cache") return Zap;
  if (item.matchKey === "stage:recall_st") return Brain;
  if (item.matchKey === "stage:recall_lt") return Brain;
  if (item.matchKey === "stage:patient") return HeartPulse;
  if (item.matchKey === "stage:route") return RouteIcon;
  if (item.matchKey === "stage:synth") return Sparkles;
  return Sparkles;
}
