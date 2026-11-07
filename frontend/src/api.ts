export interface Credentials {
  email: string
  password: string
}

export interface UserResponse {
  id: string
  email: string
  role: string
  created_at: string
}

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: 'bearer'
}

export interface ChatRequest {
  question: string
  session_id: string | null
}

export interface ChatSource {
  tag: string
  doc_id: string | null
  chunk_index: number | null
  title: string | null
  source_file: string | null
  page: number | null
  circular_no: string | null
  source_url: string | null
}

export interface ChatResponse {
  answer: string
  sources: ChatSource[]
  abstained: boolean
  citation_validation: Record<string, unknown>
  session_id: string
}

export interface ChatSession {
  id: string
  created_at: string
  title: string | null
}

export interface SessionDetail extends ChatSession {
  messages: {
    id: string
    role: string
    content: string
    citations: Record<string, unknown>[]
    created_at: string
  }[]
}

export type AuthenticatedRequest = <T>(
  path: string, options: RequestInit, signal: AbortSignal,
) => Promise<T>

export class ApiError extends Error {
  status: number | undefined
  retryAfterMs: number | undefined

  constructor(message: string, status?: number, retryAfterMs?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.retryAfterMs = retryAfterMs
  }
}

function retryAfterMs(value: string | null): number | undefined {
  if (!value?.trim()) return undefined
  const seconds = Number(value)
  if (Number.isFinite(seconds) && seconds < 0) return undefined
  const delay = Number.isFinite(seconds)
    ? seconds * 1000
    : Date.parse(value) - Date.now()
  return Number.isFinite(delay) ? Math.max(0, delay) : undefined
}

function errorMessage(status: number, code: unknown, path: string, delay?: number): string {
  if (path.startsWith('/sessions')) {
    if (status === 404) return 'This conversation is unavailable. Choose another conversation or New Chat.'
    if (status >= 500) return 'Conversation history is temporarily unavailable. Please try again.'
  }
  if (path === '/chat') {
    if (status === 404) return 'This conversation is unavailable. Choose New Chat to start another.'
    if (status === 422) return 'The question could not be accepted. Please check it and try again.'
    if (status === 500) return 'The answer could not be completed. Please try again.'
  }
  if (status === 409) return 'This email is already registered. Please log in instead.'
  if (status === 401) {
    return code === 'invalid_credentials'
      ? 'Incorrect email or password. Please try again.'
      : 'Your sign-in is no longer valid. Please log in again.'
  }
  if (status === 422) return 'Please check your email address and password requirements.'
  if (status === 429) {
    return delay === undefined
      ? 'Too many requests. Please wait a minute before trying again.'
      : `Too many requests. Please try again in ${Math.ceil(delay / 1000)} seconds.`
  }
  if (status >= 500) return 'The service is temporarily unavailable. Please try again shortly.'
  return 'Your request could not be completed. Please try again.'
}

/** Check response shapes once at the network boundary, not in UI components. */
function validateResponse(path: string, data: unknown): void {
  const value = data && typeof data === 'object' ? data as Record<string, unknown> : null
  let valid = true
  if (path === '/auth/register') {
    valid = value !== null
      && typeof value.id === 'string' && typeof value.email === 'string'
      && typeof value.role === 'string' && typeof value.created_at === 'string'
  } else if (path === '/auth/login' || path === '/auth/refresh') {
    valid = value !== null
      && typeof value.access_token === 'string' && Boolean(value.access_token.trim())
      && typeof value.refresh_token === 'string' && Boolean(value.refresh_token.trim())
      && value.token_type === 'bearer'
  } else if (path === '/sessions') {
    valid = Array.isArray(data) && data.every(isSession)
  } else if (path.startsWith('/sessions/')) {
    valid = isSession(data) && data.id === decodeURIComponent(path.slice('/sessions/'.length))
      && value !== null && Array.isArray(value.messages) && value.messages.every((message) =>
        message !== null && typeof message === 'object'
        && typeof message.id === 'string' && typeof message.role === 'string'
        && typeof message.content === 'string' && typeof message.created_at === 'string'
        && Array.isArray(message.citations) && message.citations.every((citation: unknown) =>
          citation !== null && typeof citation === 'object' && !Array.isArray(citation),
        ),
      )
  } else if (path === '/chat') {
    valid = value !== null
      && typeof value.answer === 'string'
      && typeof value.session_id === 'string' && Boolean(value.session_id.trim())
      && typeof value.abstained === 'boolean'
      && value.citation_validation !== null && typeof value.citation_validation === 'object'
      && !Array.isArray(value.citation_validation)
      && Array.isArray(value.sources) && value.sources.every((source) =>
        source !== null && typeof source === 'object' && typeof source.tag === 'string'
        && [source.doc_id, source.title, source.source_file, source.circular_no, source.source_url]
          .every((field) => field === null || typeof field === 'string')
        && [source.page, source.chunk_index].every((field) => field === null || Number.isInteger(field)),
      )
  }
  if (!valid) {
    throw new ApiError('The service returned an unexpected response. Please try again.')
  }
}

function isSession(data: unknown): data is ChatSession {
  if (!data || typeof data !== 'object') return false
  const session = data as Record<string, unknown>
  return typeof session.id === 'string' && Boolean(session.id.trim())
    && typeof session.created_at === 'string' && Number.isFinite(Date.parse(session.created_at))
    && (session.title === null || typeof session.title === 'string')
}

/** JSON requests only. Server error text is never shown or logged. */
export async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const baseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  if (!baseUrl) {
    throw new ApiError('The service connection is not configured. Please contact the administrator.')
  }
  const headers = new Headers(options.headers)
  headers.set('Accept', 'application/json')
  if (options.body !== undefined) headers.set('Content-Type', 'application/json')

  let response: Response
  try {
    response = await fetch(`${baseUrl.replace(/\/+$/, '')}${path}`, {
      ...options,
      headers,
      credentials: 'omit',
      cache: 'no-store',
    })
  } catch (error) {
    if (options.signal?.aborted) throw error
    throw new ApiError('Unable to connect. Please check your connection and try again.')
  }

  let data: unknown
  try {
    data = await response.json()
  } catch (error) {
    if (options.signal?.aborted) throw error
    if (response.ok) {
      throw new ApiError('The service returned an unexpected response. Please try again.')
    }
  }
  if (!response.ok) {
    const detail = data && typeof data === 'object' && 'detail' in data ? data.detail : null
    const code = detail && typeof detail === 'object' && 'code' in detail ? detail.code : null
    const delay = retryAfterMs(response.headers.get('Retry-After'))
    throw new ApiError(
      errorMessage(response.status, code, path, delay),
      response.status,
      delay,
    )
  }
  validateResponse(path, data)
  return data as T
}
