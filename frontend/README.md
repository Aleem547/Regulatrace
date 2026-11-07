# Regulatrace Frontend

Initialized from the official Vite React + TypeScript template with Tailwind CSS
using `@tailwindcss/vite`. Login and registration use the FastAPI JSON endpoints,
with client-side validation, password visibility controls, pending states and
safe error messages. Registration returns to login without signing in. Successful
login opens a responsive chat workspace with logout, a collapsible mobile
sidebar, saved conversation history and a question composer. Chat posts to `/chat`
with a bearer access token and `session_id: null` for the first question; follow-ups
reuse the active session ID. Questions appear immediately,
with a waiting indicator until FastAPI returns the complete JSON answer. Answers
are displayed as plain text without rewriting, regardless of the `abstained`
flag, followed by actual source titles, pages and circular numbers when supplied.
There is no streaming.

The workspace loads `/sessions` on sign-in and refreshes it after a new chat is
saved. Titles appear newest first, with an untitled fallback. Selecting a session
loads `/sessions/{id}` and retains its ID for follow-ups. Historical sources show
only stored document/page citations; unavailable titles and URLs are not invented.
Loading and error states include retry controls. The backend currently returns
at most 100 sessions and the latest 100 messages per session.

Messages and the active session ID live in React state. New Chat clears them and
the draft, without deleting server-side sessions. Pending requests are aborted
on New Chat, logout and unmount. Request-identity and auth-identity checks ignore
late conversation results. Browser cancellation does not
guarantee backend cancellation: an interrupted request may still complete and
persist. Network/server failures are not automatically replayed; the unanswered
question returns to the composer for a deliberate retry.

Only one request may be pending per conversation. HTTP 429 displays a static
error with the `Retry-After` delay when supplied (seconds or HTTP date); if
unavailable/invalid, it asks the user to wait a minute. There are no countdowns
or rate-limit timers. Send checks the stored retry deadline before issuing
another request; this guard persists across New Chat and uses a 60-second
fallback when the header is unusable. Errors and response-shape validation live
in the API helper, not the UI. HTTP 404 asks the user to start a new conversation.

From `frontend/`, using Node.js 22.16 or another Vite-compatible Node version:

```powershell
npm ci
npm run dev
```

Production build and lint:

```powershell
npm run build
npm run lint
```

The production output is written to `dist/`. Use `npm run preview` to preview it.

Before starting Vite, copy `.env.example` to `.env.local` and set
`VITE_API_BASE_URL` to the backend origin. Restart Vite after changing it. This
setting is public, not a place for secrets. FastAPI routes have no prefix:
registration posts `{email, password}` to `/auth/register` (201), and login posts
the same fields to `/auth/login`. Confirm password is never sent to the backend.

Access and refresh tokens stay only in React memory. Reloading the page or
logging out clears them; there is no persistent login. An authenticated HTTP 401 triggers
one `/auth/refresh` request with the in-memory refresh token, then at most one
replay with the returned access token. Refresh failure or another 401 clears
authentication and returns to login. Concurrent chat/history requests share one
refresh. That refresh may finish after a conversation switch, but cannot restore
a logged-out session or overwrite a newer login. Canceled requests are not replayed.
There is no proactive refresh timer or retry loop.
Logout does not revoke server tokens. Requests omit cookies, and no token is
written to browser storage, URLs or the UI. No authentication context or router
has been added.

FastAPI CORS defaults to allowing `http://localhost:5173`, GET/POST requests and
Authorization/Content-Type headers, with cookies disabled and OPTIONS preflights
handled by the middleware. Configure additional explicit origins through the
backend's `REGULATRACE_CORS_ALLOWED_ORIGINS` JSON list and restart the backend.
`http://127.0.0.1:5173` is a different origin and needs its own allowance if used.
Network/CORS failures still show a safe connection error.

Manual browser verification for conversation history:

- Sign in and check saved titles, newest-first ordering, empty/error states and
  retry controls. Select a session and compare its messages/citations with the API.
- Continue a loaded session and confirm `/chat` sends its existing ID. New Chat
  must send null, retain server history and refresh the sidebar after success.
- Switch sessions rapidly or log out during a history request. Old messages must
  not overwrite the current conversation or return after logout.

- Confirm the first question sends a null session ID, and a follow-up uses the
  returned ID. Compare the visible answer and source metadata with the response.
- Try New Chat and logout during a slow request. Late results must not populate
  the new conversation or restore authentication. Check mobile navigation too.
- Let a development access token expire: expect `/chat` 401, one refresh, and one
  retry. Block the refresh request to confirm return to login on failure.
- Check network failures and 404/429/500/503 responses using controlled development
  response overrides. Confirm the question is restored for retry, the 429 message
  shows its delay, and early Send attempts make no request, including after New Chat.

`node_modules/`, `dist/`, and local environment files are Git-ignored. Commit
`package-lock.json` with the source for reproducible installs.

## Combined deployment

The root Dockerfile builds this frontend with `VITE_API_BASE_URL=/` and serves it
from FastAPI on the same origin. A slash is intentional: the API helper removes
the trailing slash before appending `/auth`, `/chat` or `/sessions`. No provider
credentials belong in a Vite variable. See `../docs/DEPLOYMENT.md`.
