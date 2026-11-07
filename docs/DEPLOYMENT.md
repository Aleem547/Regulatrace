# Regulatrace deployment

Status: source recovered and local validation performed; no hosted release is claimed.
See [UPSTREAM.md](../UPSTREAM.md) for original source attribution and licensing context.

## Hosting selection - corrected 6 November 2026

Source publication is confirmed: all 134 files at GitHub commit
`8d07a338bba12cc7c15c5d0f337aab7481005495` matched the reviewed package.
No hosted release has been created.

The previous recommendation treated CPU Basic's zero hourly rate as a free
Docker-hosting entitlement. That was incomplete. Hugging Face currently requires
a paid subscription to create Docker Spaces, even on CPU Basic. Its published
networking policy also permits outbound ports 80, 443 and 8080, which does not
establish compatibility with this app's normal PostgreSQL TCP connection.
Hugging Face is therefore not a verified zero-cost deployment path for this app.
Do not purchase a subscription as part of these instructions.

Choose a compatible host after confirming account eligibility, available memory
and outbound access to the configured PostgreSQL and Qdrant ports. A 512 MB
allowance has not been established as sufficient for the BGE cross-encoder.
Reusing an existing AWS server retains ongoing AWS costs; a public demo served
from the user's computer depends on that computer remaining online. Neither
option is asserted to provide unlimited free cloud hosting.

The Dockerfile remains portable: it builds React and serves static assets after
FastAPI routes on one origin, using `VITE_API_BASE_URL=/`. It listens on `PORT`
(default 8000). Existing API routes, JWT identifiers and configuration prefixes
remain compatible. Persistent database and retrieval state must be preserved.

## Required inputs

- GitHub write access for `Aleem547/Regulatrace` and a hosting account.
- Permission to redistribute the upstream source with its required notices.
- A privately supplied PostgreSQL connection URL with TLS, and a migrated schema.
- A Qdrant HTTPS endpoint and key with compatible server-side `qdrant/bm25`
  inference. The index setup must be tested on the selected cluster; availability
  is not inferred from successful database connectivity.
- A Groq key and an available model. The recorded upstream model is
  `openai/gpt-oss-120b`; verify current account/model availability.
- The **42 manifest PDFs**, absent from the uploaded source ZIP, or a verified
  matching index/registry restore. Do not substitute unrelated documents to claim
  the historical benchmark scores. Original links are in `data/manifest.csv`.

Use the private values listed in [production.env.example](../deploy/production.env.example).
Generate a stable JWT key locally with
`python -c "import secrets; print(secrets.token_hex(32))"`. Store it in hosting Secrets,
not in Git, browser variables or chat. A PostgreSQL URL may use `postgresql://` or
`postgresql+psycopg2://`; both explicitly select the installed psycopg2 driver.
TLS query options such as `sslmode=require` are retained. SQLAlchemy defaults are
not used to guess a different PostgreSQL driver.

## Build and initialize

From the repository root, after configuring private environment values:

```powershell
docker build -t regulatrace:local .
python -m alembic upgrade head
python -m regulatrace.prepare_index --path data/raw
python -m regulatrace.deployment --live
```

Run Python commands from an environment installed with the original API/Groq extras,
or inside the built image with the corpus mounted. The image includes the manifest,
but intentionally excludes PDFs and credentials. Windows development can continue
using the original editable install instructions. `requirements-runtime.txt` pins
the container's **Python 3.12, Linux x86_64** dependency set and CPU PyTorch wheel;
it is not a universal Windows lockfile.

The index helper refuses missing manifest PDFs and HTML files masquerading as PDFs
before creating a client or loading a model. It runs dense ingestion, creates and
backfills BM25, then verifies nonzero matching counts. Unchanged inputs are
repeatable. If changed documents leave stale BM25 points, the helper fails and
does not delete existing index data automatically; inspect a snapshot and rebuild
the stale index deliberately. Cross-store ingestion remains non-transactional.

`python -m regulatrace.deployment` validates the external-service production profile
without making network calls or printing secret values. `--live` additionally
reads the schema revision, collection counts/public metadata and Groq's model
listing. It does not generate an answer or prove evidence relevance. `/health`
remains a liveness check; `/ready` remains the original dependency-only check.

## Publish the combined app

After selecting a host that meets the account, memory and network requirements,
use the root Dockerfile. Configure non-sensitive values as environment settings
and connection strings/API keys/JWT key as private secrets. Match the host's HTTP
port to `PORT`; provide HTTPS at the public entry point. The container runs as
UID 1000 and has writable model caches. Do not place provider keys in `VITE_*`.

The Docker start command validates configuration before Uvicorn. Run migrations
and index preparation separately. Model weights load lazily; first-request
availability, memory and latency require hosted testing. Preserve database and
index state across container restarts.

## Release verification and rollback

Confirm the root page and assets, `/health`, `/ready`, registration/login, saved
owner-scoped sessions, first-question citations, follow-ups, abstention and restart
persistence. A 200 from `/health` or a built frontend is not evidence that chat works.
Record the exact commit, public URL, test output, observed latency/memory and limits.
For rollback, retain the preceding source version and take database/index snapshots
before migrations or ingestion. No automatic destructive rollback command is supplied.

## Evidence and remaining access limits

Local checks are recorded in the accompanying readiness report. Docker execution,
hosted model inference, account-backed database/index initialization and the final
authenticated hosted chat test remain unverified until the necessary services,
credentials and corpus are available. A paid provider or infrastructure resource
was not created. Free-tier terms can change and are not lifetime guarantees.

Primary references checked on 5 November 2026:

- <https://huggingface.co/docs/hub/spaces-overview>
- <https://huggingface.co/docs/hub/spaces-sdks-docker>
- <https://render.com/docs/compute-plans>
