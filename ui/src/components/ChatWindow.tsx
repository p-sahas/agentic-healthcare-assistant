import { useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import type { UIMessage } from "@/types";
import type { ThoughtItem } from "@/hooks/useChatStream";
import { ChainOfThought } from "./ChainOfThought";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: UIMessage[];
  loading: boolean;
  thoughts: ThoughtItem[];
  error: string | null;
}

export function ChatWindow({ messages, loading, thoughts, error }: Props) {
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, thoughts.length]);

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
      {messages.length === 0 && !loading && (
        <EmptyState />
      )}

      <div className="space-y-4 max-w-3xl mx-auto w-full">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}

        {loading && <ChainOfThought items={thoughts} />}

        {error && (
          <div className="text-xs text-danger px-2">
            Connection error: {error}
          </div>
        )}

        <div ref={end} />
      </div>
    </div>
  );
}

function EmptyState() {
  // Generic starter questions — kept neutral and broadly useful so
  // they don't mislead about the assistant's scope or look like
  // role-play prompts. One per route family: CRM, RAG/CAG, CRM-list,
  // RAG-clinical.
  const samples = [
    "What are the opening hours of the hospital?",
    "Do I have any appointments this week?",
    "What should I bring for my first appointment?",
    "Show me all the dermatology consultants.",
  ];
  return (
    <div className="max-w-2xl mx-auto text-center py-12 space-y-6 animate-fade-in">
      <div className="inline-flex items-center justify-center size-14 rounded-2xl bg-brand-500/10 border border-brand-500/30">
        <Sparkles className="text-brand-400" size={26} />
      </div>
      <div>
        <h2 className="text-xl font-semibold text-slate-100">Nawaloka Health Assistant</h2>
        <p className="text-sm text-slate-400 mt-1">
          Ask about appointments, doctors, policies, or procedures. The agent routes your query across
          a CRM, internal KB (RAG), semantic cache (CAG), and live web search.
        </p>
      </div>
      <div className="grid sm:grid-cols-2 gap-2 text-left">
        {samples.map((s) => (
          <div key={s} className="card px-3 py-2 text-sm text-slate-300">
            {s}
          </div>
        ))}
      </div>
    </div>
  );
}
