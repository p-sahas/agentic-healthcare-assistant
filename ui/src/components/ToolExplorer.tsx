import { useState, type ReactNode } from "react";
import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import {
  ChevronDown,
  Database,
  Globe,
  HeartPulse,
  Loader2,
  Zap,
  Brain,
  type LucideIcon,
} from "lucide-react";
import { cagApi, crmApi, memoryApi, ragApi, webApi } from "@/api/client";

interface Props {
  userId: string;
  sessionId: string;
}

/**
 * Collapsible explorer with one subpanel per backend tool. Each panel
 * exposes a tiny form whose fields match the corresponding REST
 * endpoint's request schema.
 */
export function ToolExplorer({ userId, sessionId }: Props) {
  return (
    <div className="space-y-2">
      <Panel icon={HeartPulse} label="CRM" accent="text-rose-400">
        <CrmPanel />
      </Panel>
      <Panel icon={Database} label="RAG · internal KB" accent="text-brand-400">
        <RagPanel />
      </Panel>
      <Panel icon={Globe} label="Web search · Tavily" accent="text-emerald-400">
        <WebPanel />
      </Panel>
      <Panel icon={Zap} label="CAG · semantic cache" accent="text-amber-400">
        <CagPanel />
      </Panel>
      <Panel icon={Brain} label="Memory" accent="text-violet-400">
        <MemoryPanel userId={userId} sessionId={sessionId} />
      </Panel>
    </div>
  );
}

// ── Collapsible panel wrapper ────────────────────────────────────────

function Panel({
  icon: Icon,
  label,
  accent,
  children,
}: {
  icon: LucideIcon;
  label: string;
  accent: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-bg-soft"
      >
        <Icon size={14} className={accent} />
        <span className="flex-1 text-left text-slate-200">{label}</span>
        <ChevronDown
          size={14}
          className={clsx("text-slate-500 transition-transform", open && "rotate-180")}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="border-t border-border"
          >
            <div className="p-3 space-y-2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Reusable output box + runner helper ─────────────────────────────

function useRunner<T>() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run(fn: () => Promise<T>) {
    setLoading(true);
    setErr(null);
    try {
      setResult(await fn());
    } catch (e) {
      setErr((e as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }
  return { loading, result, err, run, setResult };
}

function RunButton({ onClick, loading, label = "Run" }: { onClick: () => void; loading: boolean; label?: string }) {
  return (
    <button type="button" className="btn-primary w-full" onClick={onClick} disabled={loading}>
      {loading ? <Loader2 size={14} className="animate-spin" /> : null}
      {loading ? "Running…" : label}
    </button>
  );
}

function Output({ children, error, empty = "No result yet" }: { children?: ReactNode; error?: string | null; empty?: string }) {
  if (error) return <pre className="text-xs text-danger whitespace-pre-wrap break-words">{error}</pre>;
  if (!children) return <div className="text-xs text-slate-500">{empty}</div>;
  return (
    <pre className="text-xs text-slate-300 whitespace-pre-wrap break-words max-h-60 overflow-auto">
      {children}
    </pre>
  );
}

// ── CRM ─────────────────────────────────────────────────────────────

function CrmPanel() {
  const [tab, setTab] = useState<"patient" | "doctors">("patient");
  const [phone, setPhone] = useState("");
  const [name, setName] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [location, setLocation] = useState("");
  const r = useRunner<{ result: string; latency_ms: number }>();

  return (
    <>
      <div className="flex gap-1 text-xs">
        <TabBtn active={tab === "patient"} onClick={() => setTab("patient")}>Lookup patient</TabBtn>
        <TabBtn active={tab === "doctors"} onClick={() => setTab("doctors")}>Search doctors</TabBtn>
      </div>
      {tab === "patient" ? (
        <>
          <input className="input" placeholder="Phone (e.g. 078 103 0736)" value={phone} onChange={(e) => setPhone(e.target.value)} />
          <input className="input" placeholder="Or name" value={name} onChange={(e) => setName(e.target.value)} />
          <RunButton
            loading={r.loading}
            onClick={() => r.run(() => crmApi.lookupPatient({ phone: phone || undefined, name: name || undefined }))}
          />
        </>
      ) : (
        <>
          <input className="input" placeholder="Specialty (e.g. Cardiology)" value={specialty} onChange={(e) => setSpecialty(e.target.value)} />
          <input className="input" placeholder="Location (optional)" value={location} onChange={(e) => setLocation(e.target.value)} />
          <RunButton
            loading={r.loading}
            onClick={() => r.run(() => crmApi.searchDoctors({ specialty: specialty || undefined, location: location || undefined }))}
          />
        </>
      )}
      <Output error={r.err}>
        {r.result ? `${r.result.result}\n\n(${r.result.latency_ms} ms)` : null}
      </Output>
    </>
  );
}

// ── RAG ─────────────────────────────────────────────────────────────

function RagPanel() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(4);
  const [useCache, setUseCache] = useState(true);
  const r = useRunner<{ result: string; latency_ms: number }>();
  return (
    <>
      <input className="input" placeholder="Search internal KB…" value={query} onChange={(e) => setQuery(e.target.value)} />
      <div className="flex gap-2 items-center text-xs">
        <label className="text-slate-400">top_k</label>
        <input type="number" min={1} max={10} className="input w-20" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
        <label className="ml-2 inline-flex items-center gap-1 text-slate-400">
          <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} /> cache
        </label>
      </div>
      <RunButton
        loading={r.loading}
        onClick={() => r.run(() => ragApi.search({ query, top_k: topK, use_cache: useCache }))}
      />
      <Output error={r.err}>
        {r.result ? `${r.result.result}\n\n(${r.result.latency_ms} ms)` : null}
      </Output>
    </>
  );
}

// ── Web ─────────────────────────────────────────────────────────────

function WebPanel() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const r = useRunner<{ result: string; latency_ms: number }>();
  return (
    <>
      <input className="input" placeholder="Ask Tavily…" value={query} onChange={(e) => setQuery(e.target.value)} />
      <div className="flex gap-2 items-center text-xs">
        <label className="text-slate-400">max_results</label>
        <input type="number" min={1} max={10} className="input w-20" value={k} onChange={(e) => setK(Number(e.target.value))} />
      </div>
      <RunButton loading={r.loading} onClick={() => r.run(() => webApi.search({ query, max_results: k }))} />
      <Output error={r.err}>
        {r.result ? `${r.result.result}\n\n(${r.result.latency_ms} ms)` : null}
      </Output>
    </>
  );
}

// ── CAG ─────────────────────────────────────────────────────────────

function CagPanel() {
  const [tab, setTab] = useState<"get" | "set" | "stats">("get");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const r = useRunner<unknown>();

  return (
    <>
      <div className="flex gap-1 text-xs">
        <TabBtn active={tab === "get"} onClick={() => setTab("get")}>Get</TabBtn>
        <TabBtn active={tab === "set"} onClick={() => setTab("set")}>Set</TabBtn>
        <TabBtn active={tab === "stats"} onClick={() => setTab("stats")}>Stats</TabBtn>
      </div>
      {tab === "get" && (
        <>
          <input className="input" placeholder="Query to look up" value={query} onChange={(e) => setQuery(e.target.value)} />
          <RunButton loading={r.loading} onClick={() => r.run(() => cagApi.get(query))} />
        </>
      )}
      {tab === "set" && (
        <>
          <input className="input" placeholder="Query" value={query} onChange={(e) => setQuery(e.target.value)} />
          <textarea className="input min-h-[80px]" placeholder="Answer" value={answer} onChange={(e) => setAnswer(e.target.value)} />
          <RunButton loading={r.loading} onClick={() => r.run(() => cagApi.set(query, answer))} />
        </>
      )}
      {tab === "stats" && (
        <RunButton loading={r.loading} onClick={() => r.run(() => cagApi.stats())} label="Fetch stats" />
      )}
      <Output error={r.err}>
        {r.result ? JSON.stringify(r.result, null, 2) : null}
      </Output>
    </>
  );
}

// ── Memory ──────────────────────────────────────────────────────────

function MemoryPanel({ userId, sessionId }: { userId: string; sessionId: string }) {
  const [tab, setTab] = useState<"facts" | "recall" | "distill">("facts");
  const [query, setQuery] = useState("");
  const r = useRunner<unknown>();

  return (
    <>
      <div className="flex gap-1 text-xs flex-wrap">
        <TabBtn active={tab === "facts"} onClick={() => setTab("facts")}>List facts</TabBtn>
        <TabBtn active={tab === "recall"} onClick={() => setTab("recall")}>Recall</TabBtn>
        <TabBtn active={tab === "distill"} onClick={() => setTab("distill")}>Distill</TabBtn>
      </div>
      <div className="text-[11px] text-slate-500">
        user=<code className="text-slate-300">{userId}</code>{" "}
        session=<code className="text-slate-300">{sessionId}</code>
      </div>
      {tab === "facts" && (
        <RunButton loading={r.loading} onClick={() => r.run(() => memoryApi.facts(userId))} label="Fetch facts" />
      )}
      {tab === "recall" && (
        <>
          <input className="input" placeholder="Recall query" value={query} onChange={(e) => setQuery(e.target.value)} />
          <RunButton loading={r.loading} onClick={() => r.run(() => memoryApi.recall(userId, sessionId, query))} />
        </>
      )}
      {tab === "distill" && (
        <RunButton loading={r.loading} onClick={() => r.run(() => memoryApi.distill(userId, sessionId))} label="Trigger distillation" />
      )}
      <Output error={r.err}>
        {r.result ? JSON.stringify(r.result, null, 2) : null}
      </Output>
    </>
  );
}

function TabBtn({ active, children, onClick }: { active: boolean; children: ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "px-2 py-1 rounded-md border text-xs",
        active ? "bg-brand-500/15 border-brand-500/40 text-brand-400" : "bg-bg-soft border-border text-slate-400 hover:text-slate-200",
      )}
    >
      {children}
    </button>
  );
}
