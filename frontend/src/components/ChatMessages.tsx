import type { ChatSource } from '../api'

export type ChatMessage =
  | { id: string; role: 'user'; question: string }
  | { id: string; role: 'assistant'; answer: string; sources: Partial<ChatSource>[] }

// Keep the backend's ID/page rules, bracket pairing and nested-bracket guards.
const citationPattern = /(?<![[【])(\[|【)\s*([A-Za-z0-9_-]+) p\.([1-9][0-9]*)\s*([\]】])(?![\]】])/g

function presentAnswer(answer: string, sources: Partial<ChatSource>[]) {
  const cited: { doc_id: string; page: number; source?: Partial<ChatSource> }[] = []
  const text = answer.replace(citationPattern, (original, opening, docId, digits, closing) => {
    if ((opening === '[' ? ']' : '】') !== closing) return original
    const page = Number(digits)
    if (!Number.isSafeInteger(page)) return original
    let index = cited.findIndex((citation) => citation.doc_id === docId && citation.page === page)
    if (index === -1) {
      index = cited.length
      cited.push({
        doc_id: docId,
        page,
        source: sources.find((source) => source.doc_id === docId && source.page === page),
      })
    }
    return `[${index + 1}]`
  })
  const other = sources.filter((source) => !cited.some((citation) =>
    citation.doc_id === source.doc_id && citation.page === source.page,
  ))
  return { text, cited, other }
}

function SourceDetails({ source }: { source: Partial<ChatSource> }) {
  return (
    <>
      <p className="font-medium text-slate-700">
        {source.title || source.source_file || source.doc_id || 'Source metadata unavailable'}
      </p>
      {source.circular_no && <p>Circular: {source.circular_no}</p>}
      {source.page != null && <p>Page {source.page}</p>}
    </>
  )
}

function ChatMessages({ messages }: { messages: ChatMessage[] }) {
  return (
    <ol className="mx-auto w-full max-w-3xl space-y-8" aria-label="Conversation messages">
      {messages.map((message) => {
        const presentation = message.role === 'assistant'
          ? presentAnswer(message.answer, message.sources) : null
        return (
          <li key={message.id}>
            <article className={message.role === 'user' ? 'ml-auto max-w-2xl rounded-2xl border border-teal-100 bg-teal-50 px-5 py-4' : 'min-w-0 rounded-2xl border border-slate-200 bg-white px-5 py-5 sm:px-6'}>
              <h2 className="mb-2 text-xs font-semibold tracking-widest text-slate-500 uppercase">
                {message.role === 'user' ? 'You' : 'Regulatrace'}
              </h2>
              <p className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-800 sm:text-base">
                {message.role === 'user' ? message.question : presentation?.text}
              </p>
              {presentation && presentation.cited.length > 0 && (
                <section className="mt-5 border-t border-slate-100 pt-4" aria-label="Cited Sources">
                  <h3 className="text-xs font-semibold tracking-wide text-slate-600">Cited Sources</h3>
                  <ol className="mt-3 space-y-3">
                    {presentation.cited.map((citation, index) => (
                      <li key={`${citation.doc_id}:${citation.page}`} className="flex gap-2 break-words text-xs leading-5 text-slate-600">
                        <span className="shrink-0 font-semibold text-teal-800">[{index + 1}]</span>
                        <div className="min-w-0">
                          <SourceDetails source={citation.source ?? citation} />
                          {!citation.source && <p className="text-rose-700">Not found in the supplied source records.</p>}
                        </div>
                      </li>
                    ))}
                  </ol>
                </section>
              )}
              {presentation && presentation.other.length > 0 && (
                <details className="mt-5 border-t border-slate-100 pt-4">
                  <summary className="cursor-pointer rounded text-xs font-medium text-slate-600 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700">
                    Other retrieved sources
                  </summary>
                  <ul className="mt-3 space-y-3">
                    {presentation.other.map((source, index) => (
                      <li key={index} className="break-words text-xs leading-5 text-slate-600">
                        <SourceDetails source={source} />
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </article>
          </li>
        )
      })}
    </ol>
  )
}

export default ChatMessages
