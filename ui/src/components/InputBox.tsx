import { useRef, useState, type KeyboardEvent } from "react";
import { SendHorizontal, RotateCcw } from "lucide-react";

interface Props {
  disabled?: boolean;
  onSend: (text: string) => void;
  onReset?: () => void;
  placeholder?: string;
}

export function InputBox({ disabled, onSend, onReset, placeholder }: Props) {
  const [text, setText] = useState("");
  const ta = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
    ta.current?.focus();
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="card p-2 flex items-end gap-2">
      <textarea
        ref={ta}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        disabled={disabled}
        rows={1}
        placeholder={placeholder || "Ask the hospital assistant anything…"}
        className="flex-1 resize-none bg-transparent outline-none text-sm placeholder-slate-500 py-1.5 px-2 min-h-[36px] max-h-[160px]"
        style={{ fieldSizing: "content" } as React.CSSProperties}
      />
      <div className="flex items-center gap-1">
        {onReset && (
          <button
            type="button"
            className="btn-ghost"
            onClick={onReset}
            title="Clear this session's short-term memory"
            disabled={disabled}
          >
            <RotateCcw size={16} />
          </button>
        )}
        <button
          type="button"
          className="btn-primary"
          onClick={submit}
          disabled={disabled || !text.trim()}
          title="Send (Enter)"
        >
          <SendHorizontal size={16} />
          <span className="hidden sm:inline">Send</span>
        </button>
      </div>
    </div>
  );
}
