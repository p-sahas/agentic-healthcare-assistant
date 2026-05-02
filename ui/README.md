# Nawaloka Health Assistant — Web UI

A single-page React app for demoing the FastAPI backend (`src/api/`). The
UI is UI-only: no auth, no persistence of its own, no business logic. It
drives the existing `/chat` endpoint and exposes every `/tools/*` route
through a collapsible explorer panel.

## Stack
- **Vite 5** + **React 18** + **TypeScript** — fast dev loop, no framework magic.
- **Tailwind CSS** for styling.
- **Framer Motion** for stage + panel animations.
- **lucide-react** for icons.
- **react-markdown** renders assistant answers.

No Redux, no react-query — the app is small enough that three custom
hooks (`useChat`, `useHealth`, `useSessions`) cover everything.

## Run

1. **Start the backend** (once, from the repo root):
   ```bash
   cd src && uvicorn api.main:app --reload --port 8000
   ```

2. **Start the UI**:
   ```bash
   cd ui
   npm install
   npm run dev
   # → http://localhost:5173
   ```

The UI calls `/api/*`; Vite's dev proxy forwards those to
`VITE_API_URL` (default `http://localhost:8000`). Override by copying
`.env.example` to `.env` and editing.

## What you see

| Area | Purpose |
|---|---|
| Top bar | App title + active session + status chip (health, model, readiness) |
| Sidebar · Sessions | Local-only session list (stored in `localStorage`). Each session maps to a `(user_id, session_id)` pair on the backend so ST memory stays scoped. |
| Sidebar · Tools | One collapsible panel per MCP-wrapped tool: **CRM · RAG · Web search · CAG cache · Memory**. Call each endpoint directly without the agent. |
| Chat window | Standard chat bubbles. User ID + session ID displayed under the input. |
| Thinking stages | Animated 5-stage pipeline shown during the request: cache → recall → route → tools → synthesis. *Simulated* because the API is synchronous — stages time-advance on a fixed schedule. |
| Response meta | After each reply: route chip (colour-coded per route), latency, cached flag, multi-route indicator, expandable debug JSON. |
| Status popover | Live readiness probes (Qdrant, Supabase, tools), active models per role, and which tools are enabled. |

## Architecture

```
ui/
├── index.html
├── vite.config.ts          # dev server + /api proxy
├── tailwind.config.ts      # custom dark palette (bg / brand / success …)
├── src/
│   ├── main.tsx            # React entry
│   ├── App.tsx             # layout (header + sidebar + chat + input)
│   ├── types.ts            # mirrors src/api/schemas.py
│   ├── api/client.ts       # fetch wrapper, one object per backend group
│   ├── hooks/
│   │   ├── useChat.ts      # send/reset + simulated thinking stages
│   │   ├── useHealth.ts    # /health poll + /config + /ready
│   │   └── useSessions.ts  # localStorage-backed session registry
│   └── components/
│       ├── ChatWindow.tsx
│       ├── MessageBubble.tsx
│       ├── InputBox.tsx
│       ├── ThinkingStages.tsx
│       ├── ResponseMeta.tsx
│       ├── ToolExplorer.tsx   # CRM / RAG / Web / CAG / Memory panels
│       ├── Sidebar.tsx
│       └── StatusBar.tsx
```

## Decisions & non-goals

- **No auth.** Deliberate. The backend `/chat` endpoint accepts any
  `user_id` / `session_id` string — we generate those client-side so the
  demo works without a login server.
- **No streaming.** The backend is sync-by-contract (see plan). The
  animated stages are UX simulation; the chip strip you see *after* the
  response is the real metadata from `ChatResponse`.
- **No global state library.** Three hooks and `localStorage`. If the
  app grows past a few more panels, consider [Zustand].
- **No test runner.** Run `npm run lint` (`tsc --noEmit`) for type
  checks. Add Vitest when business logic appears here (it shouldn't —
  this is a thin client).

## Adding a new backend endpoint

1. Add the request/response types to `src/types.ts`.
2. Add a method to the matching object in `src/api/client.ts`.
3. Either:
   - Drop a new panel into `ToolExplorer.tsx` (copy an existing one), or
   - Wire it into `useChat` / a new hook if it belongs in the main flow.

## Dockerize (later)

The UI builds to static files:
```bash
npm run build   # → ui/dist/
```
Serve `dist/` behind nginx or caddy. In production set
`VITE_API_URL` at build time so `/api/*` rewrites to the correct host,
or drop the proxy and put the API behind the same origin.
