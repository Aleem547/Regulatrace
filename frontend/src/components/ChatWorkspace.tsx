import { useCallback, useEffect, useId, useRef, useState, type SubmitEvent } from 'react'
import { ApiError, type AuthenticatedRequest, type ChatResponse, type ChatSession, type SessionDetail } from '../api'
import ChatMessages, { type ChatMessage } from './ChatMessages'
import ChatSidebar from './ChatSidebar'

type ChatWorkspaceProps = {
  onLogout: () => void
  onRequest: AuthenticatedRequest
}

function ChatWorkspace({ onLogout, onRequest }: ChatWorkspaceProps) {
  const id = useId()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [notice, setNotice] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const [sessionLoading, setSessionLoading] = useState(false)
  const [sessionError, setSessionError] = useState('')
  const composer = useRef<HTMLTextAreaElement>(null)
  const activeRequest = useRef<AbortController | null>(null)
  const historyRequest = useRef<AbortController | null>(null)
  const retryDeadline = useRef(0)
  const conversationEnd = useRef<HTMLDivElement>(null)
  const composerDisabled = pending || sessionLoading || Boolean(sessionError)

  const loadSessions = useCallback(() => {
    historyRequest.current?.abort()
    const controller = new AbortController()
    historyRequest.current = controller
    void onRequest<ChatSession[]>('/sessions', {}, controller.signal).then((result) => {
      if (controller.signal.aborted || historyRequest.current !== controller) return
      setSessions([...result].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)))
      setHistoryError('')
    }).catch((failure: unknown) => {
      if (controller.signal.aborted || historyRequest.current !== controller) return
      setHistoryError(failure instanceof ApiError ? failure.message : 'Unable to load conversation history.')
    }).finally(() => {
      if (historyRequest.current === controller) {
        historyRequest.current = null
        setHistoryLoading(false)
      }
    })
  }, [onRequest])

  function refreshHistory() {
    setHistoryLoading(true)
    setHistoryError('')
    void loadSessions()
  }

  useEffect(() => {
    void loadSessions()
    return () => {
      activeRequest.current?.abort()
      historyRequest.current?.abort()
    }
  }, [loadSessions])

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({ block: 'nearest' })
  }, [messages, pending])

  function newChat() {
    activeRequest.current?.abort()
    activeRequest.current = null
    setPending(false)
    setSessionLoading(false)
    setSessionError('')
    setSessionId(null)
    setMessages([])
    setError('')
    setQuestion('')
    setNotice('New conversation ready.')
    setSidebarOpen(false)
    window.requestAnimationFrame(() => composer.current?.focus())
  }

  function logout() {
    activeRequest.current?.abort()
    activeRequest.current = null
    historyRequest.current?.abort()
    onLogout()
  }

  async function openSession(id: string) {
    newChat()
    const controller = new AbortController()
    activeRequest.current = controller
    setSessionId(id)
    setSessionLoading(true)
    setNotice('')
    try {
      const detail = await onRequest<SessionDetail>(`/sessions/${encodeURIComponent(id)}`, {}, controller.signal)
      if (controller.signal.aborted || activeRequest.current !== controller) return
      // Stored citations contain document/page pairs, not the full live source records.
      setMessages(detail.messages.flatMap<ChatMessage>((message) => {
        if (message.role === 'user') return [{ id: message.id, role: 'user', question: message.content }]
        if (message.role !== 'assistant') return []
        const sources = message.citations.flatMap((citation) =>
          typeof citation.doc_id === 'string' && citation.doc_id.trim()
          && typeof citation.page === 'number' && Number.isInteger(citation.page) && citation.page > 0
            ? [{ doc_id: citation.doc_id, page: citation.page }] : [],
        )
        return [{ id: message.id, role: 'assistant', answer: message.content, sources }]
      }))
      setSessionId(detail.id)
      if (!detail.messages.length) setNotice('This conversation has no saved messages yet.')
    } catch (failure) {
      if (controller.signal.aborted || activeRequest.current !== controller) return
      setSessionError(failure instanceof ApiError ? failure.message : 'Unable to load this conversation. Please try again.')
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null
        setSessionLoading(false)
      }
    }
  }

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!question.trim() || activeRequest.current || sessionError) return
    const wait = Math.ceil((retryDeadline.current - Date.now()) / 1000)
    if (wait > 0) {
      setError(`Please wait another ${wait} seconds before sending again.`)
      return
    }

    const submittedQuestion = question.trim()
    const messageId = crypto.randomUUID()
    const controller = new AbortController()
    activeRequest.current = controller
    setPending(true)
    setError('')
    setNotice('')
    setQuestion('')
    setMessages((current) => [...current, { id: messageId, role: 'user', question: submittedQuestion }])

    try {
      const response = await onRequest<ChatResponse>(
        '/chat',
        { method: 'POST', body: JSON.stringify({ question: submittedQuestion, session_id: sessionId }) },
        controller.signal,
      )
      if (controller.signal.aborted || activeRequest.current !== controller) return
      setMessages((current) => [
        ...current,
        { id: `${messageId}-answer`, role: 'assistant', answer: response.answer, sources: response.sources },
      ])
      setSessionId(response.session_id)
      if (sessionId === null) refreshHistory()
    } catch (failure) {
      if (controller.signal.aborted || activeRequest.current !== controller) return
      // Restore an unanswered question to the composer for an explicit retry.
      setMessages((current) => current.filter((message) => message.id !== messageId))
      setQuestion(submittedQuestion)
      if (failure instanceof ApiError && failure.status === 429) {
        retryDeadline.current = Date.now() + (failure.retryAfterMs ?? 60_000)
      }
      setError(failure instanceof ApiError ? failure.message : 'Unable to complete your question. Please try again.')
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null
        setPending(false)
      }
    }
  }

  return (
    <div className="flex min-h-svh flex-col bg-slate-50 text-slate-900 lg:h-dvh lg:min-h-0">
      <a
        href={`#${id}-question`}
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-10 focus:rounded-lg focus:bg-white focus:px-4 focus:py-3 focus:outline-2 focus:outline-teal-700"
      >
        Skip to question input
      </a>
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <span
            className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-teal-800 text-base font-semibold text-white"
            aria-hidden="true"
          >
            PL
          </span>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Regulatrace</h1>
            <p className="hidden text-xs text-slate-500 sm:block">
              Regulatory Intelligence for RBI &amp; SEBI
            </p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          <button
            type="button"
            aria-expanded={sidebarOpen}
            aria-controls={`${id}-sidebar`}
            aria-label={sidebarOpen ? 'Hide conversations' : 'Show conversations'}
            onClick={() => setSidebarOpen((current) => !current)}
            className="rounded-lg p-3 text-slate-600 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 lg:hidden"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" className="size-5" aria-hidden="true">
              <path
                d={sidebarOpen ? 'M6 6l12 12M6 18L18 6' : 'M4 6h16M4 12h16M4 18h16'}
                strokeLinecap="round"
              />
            </svg>
          </button>
          <button
            type="button"
            onClick={logout}
            className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700"
          >
            Log out
          </button>
        </div>
      </header>

      <div className="flex flex-1 flex-col lg:min-h-0 lg:flex-row">
        <ChatSidebar
          id={`${id}-sidebar`} open={sidebarOpen} onNewChat={newChat}
          sessions={sessions} selectedId={sessionId} loading={historyLoading} error={historyError}
          onSelect={(id) => void openSession(id)} onRefresh={refreshHistory}
        />

        <main className="flex min-w-0 flex-1 flex-col lg:min-h-0" aria-label="Regulatory chat">
          <section className={`flex-1 px-5 sm:px-8 lg:min-h-0 lg:overflow-y-auto ${messages.length ? 'py-8' : 'flex items-center justify-center py-12'}`}>
            {sessionLoading || sessionError ? (
              <div className="m-auto max-w-lg text-center text-sm leading-6">
                {sessionLoading ? <p role="status">Loading conversation...</p> : (
                  <>
                    <p role="alert" className="text-rose-700">{sessionError}</p>
                    <button type="button" onClick={() => sessionId && void openSession(sessionId)} className="mt-4 rounded-lg px-4 py-2 font-medium text-teal-800 hover:bg-teal-50 focus-visible:outline-2 focus-visible:outline-teal-700">
                      Try again
                    </button>
                  </>
                )}
              </div>
            ) : messages.length > 0 ? (
              <div>
                <ChatMessages messages={messages} />
                {pending && (
                  <p role="status" className="mx-auto mt-6 max-w-3xl px-1 text-sm text-slate-600">
                    Waiting for Regulatrace to answer...
                  </p>
                )}
                <div ref={conversationEnd} />
              </div>
            ) : (
              <div className="m-auto w-full max-w-2xl text-center">
                <div className="mx-auto flex size-14 items-center justify-center rounded-2xl border border-teal-100 bg-teal-50 text-teal-800">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="size-7" aria-hidden="true">
                    <path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8l-5-5Z" strokeLinejoin="round" />
                    <path d="M14 3v5h5M9 12h6M9 16h4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <p className="mt-6 text-xs font-semibold tracking-widest text-teal-800 uppercase">
                  Your regulatory research workspace
                </p>
                <h2 id={`${id}-welcome`} className="mt-3 text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">
                  Start with a regulatory question
                </h2>
                <p className="mx-auto mt-4 max-w-lg text-sm leading-7 text-slate-600 sm:text-base">
                  Explore a circular, reporting requirement or regulatory change.
                  Ask a question to see an answer and its returned source references.
                </p>
                <div className="mt-6 flex flex-wrap justify-center gap-2 text-xs font-medium text-slate-600">
                  <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5">RBI circulars</span>
                  <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5">SEBI requirements</span>
                  <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5">Reporting timelines</span>
                </div>
              </div>
            )}
          </section>

          <div className="shrink-0 px-4 pb-5 sm:px-8 sm:pb-6">
            <form
              onSubmit={handleSubmit}
              className="mx-auto w-full max-w-3xl"
              aria-label="Question composer"
              aria-busy={pending || sessionLoading}
            >
              <div className="rounded-2xl border border-slate-300 bg-white shadow-sm transition focus-within:border-teal-700 focus-within:ring-4 focus-within:ring-teal-700/10">
                <label htmlFor={`${id}-question`} className="sr-only">Your regulatory question</label>
                <textarea
                  ref={composer}
                  id={`${id}-question`}
                  name="question"
                  rows={3}
                  value={question}
                  disabled={composerDisabled}
                  onChange={(event) => {
                    setQuestion(event.target.value)
                    setNotice('')
                    setError('')
                  }}
                  aria-describedby={`${id}-composer-help`}
                  placeholder="Ask about RBI or SEBI regulations..."
                  className="block max-h-48 min-h-24 w-full resize-y rounded-t-2xl bg-transparent px-4 py-4 text-base leading-6 placeholder:text-slate-400 focus:outline-none disabled:cursor-wait disabled:opacity-60 sm:px-5"
                />
                <div className="flex flex-wrap items-center justify-between gap-3 px-4 pb-4 sm:px-5">
                  <p id={`${id}-composer-help`} className="text-xs leading-5 text-slate-500">
                    New Chat starts a separate conversation.
                  </p>
                  <button
                    type="submit"
                    disabled={!question.trim() || composerDisabled}
                    className="ml-auto rounded-lg bg-teal-800 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-teal-900 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
                  >
                    {pending ? 'Waiting...' : 'Send'}
                  </button>
                </div>
              </div>
              <p role="alert" className="mt-3 text-sm leading-6 text-rose-700">{error}</p>
              <p role="status" aria-live="polite" aria-atomic="true" className="mt-3 min-h-5 text-center text-xs leading-5 text-slate-600">
                {notice}
              </p>
            </form>
          </div>
        </main>
      </div>
    </div>
  )
}

export default ChatWorkspace
