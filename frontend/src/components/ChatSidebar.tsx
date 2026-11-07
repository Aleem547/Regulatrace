import type { ChatSession } from '../api'

type ChatSidebarProps = {
  id: string
  open: boolean
  onNewChat: () => void
  sessions: ChatSession[]
  selectedId: string | null
  loading: boolean
  error: string
  onSelect: (id: string) => void
  onRefresh: () => void
}

function ChatSidebar({ id, open, onNewChat, sessions, selectedId, loading, error, onSelect, onRefresh }: ChatSidebarProps) {
  return (
    <aside
      id={id}
      aria-label="Conversations"
      className={`${open ? 'flex' : 'hidden'} flex-col border-b border-slate-200 bg-white p-5 lg:flex lg:w-72 lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-b-0 lg:p-6`}
    >
      <button
        type="button"
        onClick={onNewChat}
        className="flex w-full items-center justify-center gap-2 rounded-lg bg-teal-800 px-4 py-3 text-sm font-semibold text-white transition hover:bg-teal-900 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" className="size-4" aria-hidden="true">
          <path d="M12 5v14M5 12h14" strokeLinecap="round" />
        </svg>
        New Chat
      </button>

      <section className="mt-8 flex-1" aria-labelledby={`${id}-heading`}>
        <h2 id={`${id}-heading`} className="text-xs font-semibold tracking-widest text-slate-500 uppercase">
          Conversation history
        </h2>
        {loading && <p role="status" className="mt-4 text-sm text-slate-500">Loading conversations...</p>}
        {error && (
          <div className="mt-4 text-sm leading-6">
            <p role="alert" className="text-rose-700">{error}</p>
            <button type="button" onClick={onRefresh} disabled={loading} className="mt-2 rounded px-2 py-1 font-medium text-teal-800 hover:bg-teal-50 focus-visible:outline-2 focus-visible:outline-teal-700 disabled:opacity-50">
              Try again
            </button>
          </div>
        )}
        {!loading && !error && sessions.length === 0 && (
          <p className="mt-4 rounded-xl border border-dashed border-slate-200 px-4 py-5 text-sm leading-6 text-slate-500">
            No saved conversations yet. Send a question to start your first chat.
          </p>
        )}
        <ul className="mt-4 space-y-2">
          {sessions.map((session) => (
            <li key={session.id}>
              <button
                type="button"
                onClick={() => onSelect(session.id)}
                aria-current={session.id === selectedId ? 'true' : undefined}
                className={`w-full rounded-lg border px-3 py-3 text-left text-sm leading-5 break-words focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 ${session.id === selectedId ? 'border-teal-200 bg-teal-50 text-teal-900' : 'border-transparent text-slate-700 hover:bg-slate-50'}`}
              >
                {session.title?.trim() || 'Untitled conversation'}
              </button>
            </li>
          ))}
        </ul>
      </section>

      <p className="mt-8 border-t border-slate-100 pt-4 text-xs leading-5 text-slate-500">
        Your sign-in stays in this page only. Reloading or logging out clears it.
      </p>
    </aside>
  )
}

export default ChatSidebar
