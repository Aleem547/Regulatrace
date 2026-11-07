import { useCallback, useRef, useState } from 'react'
import {
  ApiError,
  apiRequest,
  type TokenResponse,
} from './api'
import AuthForm from './components/AuthForm'
import ChatWorkspace from './components/ChatWorkspace'

function App() {
  const [view, setView] = useState<'login' | 'register'>('login')
  const [tokens, setTokens] = useState<TokenResponse | null>(null)
  const tokenRef = useRef<TokenResponse | null>(null)
  const refreshRequest = useRef<Promise<TokenResponse> | null>(null)
  const authVersion = useRef(0)
  const [notice, setNotice] = useState('')

  function changeView(nextView: 'login' | 'register') {
    setNotice('')
    setView(nextView)
  }

  const logout = useCallback((message = 'You have been logged out of this page.') => {
    authVersion.current += 1
    tokenRef.current = null
    refreshRequest.current = null
    setTokens(null)
    setView('login')
    setNotice(message)
  }, [])

  const authenticatedRequest = useCallback(async <T,>(
    path: string, options: RequestInit, signal: AbortSignal,
  ): Promise<T> => {
    const current = tokenRef.current
    const version = authVersion.current
    if (!current || signal.aborted) throw new DOMException('Request canceled', 'AbortError')
    const send = (accessToken: string) => {
      const headers = new Headers(options.headers)
      headers.set('Authorization', `Bearer ${accessToken}`)
      return apiRequest<T>(path, {
        ...options,
        headers,
        signal,
      })
    }

    try {
      return await send(current.access_token)
    } catch (error) {
      if (signal.aborted || authVersion.current !== version || !tokenRef.current) throw error
      if (!(error instanceof ApiError) || error.status !== 401) throw error
    }

    // History and chat may overlap: share one refresh, not multiple refresh calls.
    let renewed = tokenRef.current
    if (!renewed) throw new DOMException('Request canceled', 'AbortError')
    try {
      if (renewed === current) {
        if (!refreshRequest.current) {
          const refresh = apiRequest<TokenResponse>('/auth/refresh', {
            method: 'POST',
            body: JSON.stringify({ refresh_token: current.refresh_token }),
          }).then((result) => {
            if (tokenRef.current !== current) throw new DOMException('Request canceled', 'AbortError')
            tokenRef.current = result
            setTokens(result)
            return result
          }).finally(() => {
            if (refreshRequest.current === refresh) refreshRequest.current = null
          })
          refreshRequest.current = refresh
        }
        renewed = await refreshRequest.current
      }
    } catch (error) {
      if (tokenRef.current === current) {
        logout('Your sign-in could not be renewed. Please log in again.')
      }
      throw error
    }
    if (signal.aborted || authVersion.current !== version || tokenRef.current !== renewed) {
      throw new DOMException('Request canceled', 'AbortError')
    }

    // One replay only. A second 401 returns to login, never another refresh loop.
    try {
      return await send(renewed.access_token)
    } catch (error) {
      if (
        !signal.aborted
        && tokenRef.current === renewed
        && error instanceof ApiError
        && error.status === 401
      ) {
        logout('Your sign-in is no longer valid. Please log in again.')
      }
      throw error
    }
  }, [logout])

  if (tokens) {
    return (
      <ChatWorkspace
        onLogout={() => logout()}
        onRequest={authenticatedRequest}
      />
    )
  }

  return (
    <main className="flex min-h-svh items-center bg-slate-50 px-5 py-10 text-slate-900 sm:px-8 lg:px-12">
      <div className="mx-auto grid w-full max-w-6xl gap-10 lg:grid-cols-2 lg:items-center lg:gap-20">
        <section className="text-center lg:text-left" aria-label="About Regulatrace">
          <div className="flex items-center justify-center gap-3 lg:justify-start">
            <span
              className="flex size-11 items-center justify-center rounded-xl bg-teal-800 text-lg font-semibold tracking-tight text-white"
              aria-hidden="true"
            >
              RT
            </span>
            <span className="text-2xl font-semibold tracking-tight">Regulatrace</span>
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-600">
            Regulatory Intelligence for RBI &amp; SEBI
          </p>

          <div className="mt-16 hidden lg:block">
            <p className="text-xs font-semibold tracking-widest text-teal-800 uppercase">
              A clearer view of regulation
            </p>
            <h2 className="mt-4 max-w-md text-4xl font-semibold leading-tight tracking-tight">
              Research with evidence in focus.
            </h2>
            <p className="mt-5 max-w-sm text-base leading-7 text-slate-600">
              Explore RBI and SEBI regulatory documents with answers grounded in
              source material and citations you can follow.
            </p>
            <div className="mt-10 flex gap-6 border-t border-slate-200 pt-6 text-sm text-slate-600">
              <span>Document-grounded answers</span>
              <span>Source-page citations</span>
            </div>
          </div>
        </section>

        <section className="mx-auto w-full max-w-md" aria-label="Account access">
          <AuthForm
            key={view}
            mode={view}
            successMessage={notice}
            onChangeView={changeView}
            onRegistered={() => {
              setView('login')
              setNotice('Account created successfully. Please log in with your email and password.')
            }}
            onLogin={(receivedTokens) => {
              authVersion.current += 1
              setNotice('')
              tokenRef.current = receivedTokens
              setTokens(receivedTokens)
            }}
          />
        </section>
      </div>
    </main>
  )
}

export default App
