import { useId, useState, type SubmitEvent } from 'react'
import {
  ApiError,
  apiRequest,
  type Credentials,
  type TokenResponse,
  type UserResponse,
} from '../api'

type AuthMode = 'login' | 'register'
type FieldName = 'email' | 'password' | 'confirmPassword'
type FieldErrors = Partial<Record<FieldName, string>>

const inputStyles =
  'mt-2 block w-full rounded-lg border bg-white px-3.5 py-3 text-base text-slate-900 transition placeholder:text-slate-400 focus:outline-none focus:ring-4'

function PasswordField({
  name,
  label,
  value,
  onChange,
  autoComplete,
  error,
  hint,
}: {
  name: 'password' | 'confirmPassword'
  label: string
  value: string
  onChange: (value: string) => void
  autoComplete: 'current-password' | 'new-password'
  error?: string
  hint?: string
}) {
  const id = useId()
  const [visible, setVisible] = useState(false)

  return (
    <div>
      <label htmlFor={id} className="text-sm font-medium text-slate-800">
        {label}
      </label>
      <div className="relative">
        <input
          id={id}
          name={name}
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          autoComplete={autoComplete}
          required
          aria-invalid={Boolean(error)}
          aria-describedby={
            [hint ? `${id}-hint` : '', error ? `${id}-error` : '']
              .filter(Boolean)
              .join(' ') || undefined
          }
          className={`${inputStyles} pr-20 ${error ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100' : 'border-slate-300 focus:border-teal-700 focus:ring-teal-700/10'}`}
        />
        <button
          type="button"
          onClick={() => setVisible((current) => !current)}
          aria-label={`${visible ? 'Hide' : 'Show'} ${label.toLowerCase()}`}
          aria-pressed={visible}
          aria-controls={id}
          className="absolute inset-y-1 right-1 rounded-md px-3 text-sm font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700"
        >
          {visible ? 'Hide' : 'Show'}
        </button>
      </div>
      {hint && (
        <p id={`${id}-hint`} className="mt-2 text-xs leading-5 text-slate-500">
          {hint}
        </p>
      )}
      {error && (
        <p id={`${id}-error`} className="mt-2 text-sm text-rose-700">
          {error}
        </p>
      )}
    </div>
  )
}

function AuthForm({
  mode,
  successMessage,
  onChangeView,
  onRegistered,
  onLogin,
}: {
  mode: AuthMode
  successMessage: string
  onChangeView: (mode: AuthMode) => void
  onRegistered: () => void
  onLogin: (tokens: TokenResponse) => void
}) {
  const id = useId()
  const registering = mode === 'register'
  const [values, setValues] = useState({ email: '', password: '', confirmPassword: '' })
  const [errors, setErrors] = useState<FieldErrors>({})
  const [message, setMessage] = useState('')
  const [pending, setPending] = useState(false)

  function updateField(field: FieldName, value: string) {
    setValues((current) => ({ ...current, [field]: value }))
    setErrors((current) => ({
      ...current,
      [field]: undefined,
      ...(field === 'password' ? { confirmPassword: undefined } : {}),
    }))
    setMessage('')
  }

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    if (pending) return
    const nextErrors: FieldErrors = {}
    const email = values.email.trim()
    const passwordLength = Array.from(values.password).length

    if (!email) {
      nextErrors.email = 'Enter your email address.'
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      nextErrors.email = 'Enter a valid email address.'
    }
    if (!passwordLength) {
      nextErrors.password = 'Enter your password.'
    } else if (registering && passwordLength < 12) {
      nextErrors.password = 'Use at least 12 characters.'
    } else if (passwordLength > 128) {
      nextErrors.password = 'Use no more than 128 characters.'
    }
    if (registering && !values.confirmPassword) {
      nextErrors.confirmPassword = 'Confirm your password.'
    } else if (registering && values.password !== values.confirmPassword) {
      nextErrors.confirmPassword = 'Passwords must match.'
    }

    setErrors(nextErrors)
    setMessage('')
    const firstInvalidField = Object.keys(nextErrors)[0]
    if (firstInvalidField) {
      const field = event.currentTarget.elements.namedItem(firstInvalidField)
      if (field instanceof HTMLInputElement) field.focus()
      return
    }
    setPending(true)
    const credentials: Credentials = { email, password: values.password }
    const options = { method: 'POST', body: JSON.stringify(credentials) }
    try {
      if (registering) {
        await apiRequest<UserResponse>('/auth/register', options)
        onRegistered()
      } else {
        const tokens = await apiRequest<TokenResponse>('/auth/login', options)
        onLogin(tokens)
      }
    } catch (error) {
      setMessage(
        error instanceof ApiError
          ? error.message
          : 'Your request could not be completed. Please try again.',
      )
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-9">
      <h1 id={`${id}-heading`} className="text-2xl font-semibold tracking-tight">
        {registering ? 'Create your account' : 'Welcome back'}
      </h1>
      <p className="mt-2 text-sm leading-6 text-slate-600">
        {registering
          ? 'Start your regulatory research with Regulatrace.'
          : 'Log in to continue your regulatory research.'}
      </p>

      {successMessage && (
        <p
          role="status"
          className="mt-5 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3 text-sm leading-6 text-teal-900"
        >
          {successMessage}
        </p>
      )}

      <form
        onSubmit={handleSubmit}
        noValidate
        aria-labelledby={`${id}-heading`}
        aria-busy={pending}
        className="mt-7"
      >
        <fieldset disabled={pending} className="space-y-5 disabled:opacity-70">
          <div>
            <label htmlFor={`${id}-email`} className="text-sm font-medium text-slate-800">
              Email address
            </label>
            <input
              id={`${id}-email`}
              name="email"
              type="email"
              autoComplete="email"
              autoCapitalize="none"
              spellCheck={false}
              value={values.email}
              onChange={(event) => updateField('email', event.target.value)}
              placeholder="you@example.com"
              required
              aria-invalid={Boolean(errors.email)}
              aria-describedby={errors.email ? `${id}-email-error` : undefined}
              className={`${inputStyles} ${errors.email ? 'border-rose-400 focus:border-rose-500 focus:ring-rose-100' : 'border-slate-300 focus:border-teal-700 focus:ring-teal-700/10'}`}
            />
            {errors.email && (
              <p id={`${id}-email-error`} className="mt-2 text-sm text-rose-700">
                {errors.email}
              </p>
            )}
          </div>

          <PasswordField
            name="password"
            label="Password"
            value={values.password}
            onChange={(value) => updateField('password', value)}
            autoComplete={registering ? 'new-password' : 'current-password'}
            hint={registering ? 'Use 12-128 characters.' : undefined}
            error={errors.password}
          />
          {registering && (
            <PasswordField
              name="confirmPassword"
              label="Confirm password"
              value={values.confirmPassword}
              onChange={(value) => updateField('confirmPassword', value)}
              autoComplete="new-password"
              error={errors.confirmPassword}
            />
          )}

          <button
            type="submit"
            disabled={pending}
            className="w-full rounded-lg bg-teal-800 px-4 py-3 text-sm font-semibold text-white transition hover:bg-teal-900 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700 active:bg-teal-950 disabled:cursor-wait disabled:hover:bg-teal-800"
          >
            {pending
              ? (registering ? 'Creating account...' : 'Logging in...')
              : (registering ? 'Register' : 'Log in')}
          </button>
        </fieldset>

        <p
          role="alert"
          aria-atomic="true"
          className="mt-4 text-sm leading-6 text-rose-700"
        >
          {message}
        </p>
      </form>

      <p className="mt-6 border-t border-slate-100 pt-6 text-center text-sm text-slate-600">
        {registering ? 'Already have an account? ' : 'New to Regulatrace? '}
        <button
          type="button"
          disabled={pending}
          onClick={() => onChangeView(registering ? 'login' : 'register')}
          className="rounded-sm font-semibold text-teal-800 underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700 disabled:cursor-wait disabled:opacity-50"
        >
          {registering ? 'Log in' : 'Create an account'}
        </button>
      </p>
    </div>
  )
}

export default AuthForm
