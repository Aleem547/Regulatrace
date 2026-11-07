# Full-stack validation scope

Status: the full-stack deployment is live and passed 19 hosted checks.
Its public receipt is [LIVE_DEPLOYMENT.json](LIVE_DEPLOYMENT.json).

The prior Hugging Face upload demo reached RUNNING at commit
`7fa48499c3f4ea33caaa763e5bbf52edffdc1582`. Its observed live checks used a fictional
one-page PDF, verified a thirty-day answer and citation, continued a follow-up,
refused an unsupported question and opened an empty independent session.

The new profile separates the Vercel API's registry-only dependencies from the
root project, keeps model weights on the existing Space, adds a secret-protected
bounded model API, configures PostgreSQL NullPool, avoids Neon pooler startup
options and adds a transaction-locked shared chat limiter.

The prepared corpus contains 42 current official PDFs and 206 pages. It produces
672 normalized BGE-small vectors, with a pinned embedding revision and deterministic
document/chunk identifiers. One two-page PDF uses a recorded, source-hash-bound OCR
sidecar. Prepared file checksums, dimensions, norms, counts and page metadata are
validated before service writes. Existing unrelated evidence is rejected.

Regulatrace passed 652 offline tests, with three skips and five live-service
cases excluded. Ruff, the frontend production build and frontend lint passed.
All 672 prepared chunks validate through the application package, with stable
document identifiers and collection names. Tests cover private-setting migration,
bounded token compatibility, secret preservation and project-scoped cleanup.
The configured Windows updater also completed the provider sequence and published
the successful live receipt. Hosted checks establish the observed behavior of
that deployment; they do not establish production throughput or uptime.

The default regression excludes three live PostgreSQL ingestion cases and two
live Qdrant/model cases. Three skips are the optional RAGAS adapter and two opt-in
live Groq prompt checks. No remote credentials are used in the local unit tests.
The launcher repeats real hosted checks for authentication, retrieval, generation,
PDF-page citations, conversation history, isolation, shared limits, CORS and web
assets. During the token transition it also checks a token with the previous
issuer. A successful result records the canonical package and configuration prefix.

The historical Recall@5/Hit@5 numbers remain attributed to upstream work and were
not rerun against the recovered corpus. Stored artifact filenames and original
configuration hashes retain their upstream run identifiers; normalized-label
configuration hashes are recorded separately. No new accuracy, precision, recall
or F1 result is claimed by this release.
