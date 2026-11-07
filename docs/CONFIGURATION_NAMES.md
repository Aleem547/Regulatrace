# Regulatrace configuration names

The Python package, command examples, project documentation and application settings
use **Regulatrace**. Import the application from `regulatrace`; environment settings
use the `REGULATRACE_` prefix.

Use `REGULATRACE_GROQ_API_KEY` for the Groq credential and
`REGULATRACE_MODEL_WORKER_SECRET` for the authenticated model worker. Store private
values in the hosting provider's Secrets settings. Constructor arguments such as
`groq_api_key` remain available for Python callers.

The branding migration preserves existing database URLs, collection names, signing
keys and model-worker credentials while renaming their configuration keys. Already
issued tokens can be accepted for a bounded transition using
`REGULATRACE_PREVIOUS_TOKEN_ISSUER` and
`REGULATRACE_PREVIOUS_TOKEN_VALID_UNTIL`; new tokens identify Regulatrace.
The transition never extends an existing token's expiry.

The source attribution is recorded separately in `UPSTREAM.md`.
