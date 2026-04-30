import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion } from "framer-motion";
import clsx from "clsx";
import { Bot, User } from "lucide-react";
import type { UIMessage } from "@/types";
import { ResponseMeta } from "./ResponseMeta";

interface Props {
  message: UIMessage;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={clsx("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      <div
        className={clsx(
          "shrink-0 size-8 rounded-full flex items-center justify-center border",
          isUser
            ? "bg-brand-500/15 border-brand-500/40 text-brand-400"
            : "bg-bg-soft border-border text-slate-300",
        )}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className={clsx("max-w-[85%] space-y-2", isUser && "items-end")}>
        <div
          className={clsx(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed prose prose-sm prose-invert max-w-none",
            isUser
              ? "bg-brand-500 text-slate-900 rounded-tr-sm prose-a:text-slate-800"
              : "bg-bg-card border border-border text-slate-100 rounded-tl-sm",
          )}
        >
          {isUser ? (
            <p className="m-0 whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content || "…"}
            </ReactMarkdown>
          )}
        </div>
        {message.meta && <ResponseMeta meta={message.meta} />}
      </div>
    </motion.div>
  );
}
