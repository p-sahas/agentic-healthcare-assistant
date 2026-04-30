import { motion } from "framer-motion";
import clsx from "clsx";
import { Check, Loader2, Database, Brain, Route as RouteIcon, Wrench, Sparkles, type LucideIcon } from "lucide-react";
import { THINKING_STAGES } from "@/hooks/useChat";

interface Props {
  activeIdx: number;
}

const ICONS: Record<(typeof THINKING_STAGES)[number]["id"], LucideIcon> = {
  cache: Database,
  recall: Brain,
  route: RouteIcon,
  tool: Wrench,
  synth: Sparkles,
};

/**
 * Animated "chain of thought" strip shown while the /chat request is in
 * flight. Stages before ``activeIdx`` are marked done; the active stage
 * shows a spinner; later stages are dim placeholders. This is a UX
 * simulation — the API is synchronous and does not stream real stages.
 */
export function ThinkingStages({ activeIdx }: Props) {
  return (
    <div className="flex gap-3">
      <div className="shrink-0 size-8 rounded-full flex items-center justify-center bg-bg-soft border border-border text-brand-400">
        <Brain size={16} />
      </div>
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="card px-3 py-3 text-sm text-slate-200 space-y-2 flex-1 max-w-[85%]"
      >
        <div className="text-xs text-slate-400 mb-1">Agent is working…</div>
        {THINKING_STAGES.map((s, i) => {
          const Icon = ICONS[s.id];
          const state: "done" | "active" | "pending" =
            i < activeIdx ? "done" : i === activeIdx ? "active" : "pending";
          return (
            <div
              key={s.id}
              className={clsx(
                "flex items-center gap-2 text-sm",
                state === "done" && "text-slate-300",
                state === "active" && "text-slate-100",
                state === "pending" && "text-slate-500",
              )}
            >
              <div
                className={clsx(
                  "size-5 rounded-md flex items-center justify-center border",
                  state === "done" && "bg-success/15 border-success/50 text-success",
                  state === "active" && "bg-brand-500/15 border-brand-500/50 text-brand-400",
                  state === "pending" && "bg-bg-soft border-border text-slate-500",
                )}
              >
                {state === "done" && <Check size={12} />}
                {state === "active" && <Loader2 size={12} className="animate-spin" />}
                {state === "pending" && <Icon size={12} />}
              </div>
              <span>{s.label}</span>
              {state === "active" && (
                <motion.span
                  animate={{ opacity: [0.4, 1, 0.4] }}
                  transition={{ duration: 1.2, repeat: Infinity }}
                  className="text-xs text-slate-500"
                >
                  …
                </motion.span>
              )}
            </div>
          );
        })}
      </motion.div>
    </div>
  );
}
