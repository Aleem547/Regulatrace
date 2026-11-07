# Regulatrace - Regulatory Intelligence for RBI & SEBI

**Maintainer:** [Abdul Aleem Ansari (@Aleem547)](https://github.com/Aleem547).

See [source credit and permissions](UPSTREAM.md). The original research
implementation and historical benchmark artifacts are attributed to their
upstream authors. Those experiments were not rerun against the recovered corpus.

<!-- REGULATRACE_DEPLOYMENT_START -->
**Live application:** [Open Regulatrace](https://regulatrace-web-8b9b4e15.vercel.app).

**Document demo:** [Hugging Face](https://huggingface.co/spaces/abdul547/Regulatrace).

**Deployment status:** Core login, chat, stored history, corpus indexes and hosted checks passed.
See [deployment verification](docs/LIVE_DEPLOYMENT.json) and [deployment instructions](docs/FREE_FULLSTACK_DEPLOYMENT.md).
<!-- REGULATRACE_DEPLOYMENT_END -->

All 42 manifest PDFs have been recovered from current official source links and prepared
as 672 chunks across 206 pages. Hashes and source URLs are recorded in the
[recovery log](data/corpus-restore.json). One scanned PDF uses a source-hash-bound OCR sidecar.
This corpus recovery does not rerun or validate the historical benchmark results below.

Regulatory research often means finding one operative clause across long RBI and SEBI circulars. Regulatrace provides a chat workspace to ask questions, inspect document/page citations and continue research in saved conversations.

The engineering focus is **whether the right evidence reaches the model**: retrieval is benchmarked, candidates are reranked, citations are checked, and traces expose failures behind a fluent answer.

- **Research:** source-grounded answers, readable citations and an evidence-based refusal gate.
- **Application:** registration/login, authenticated chat, saved sessions and follow-up questions.
- **Diagnostics:** optional Langfuse traces, stage latency, token usage and estimated cost.

**Stack:** React/TypeScript/Tailwind, FastAPI, PostgreSQL, Qdrant, Sentence Transformers, Groq and LangChain LCEL. RAGAS provides partial judge evaluation. The full free profile places the Python API on Vercel and model inference on the existing ZeroGPU Space.

## Results at a glance

| Measurement | Recorded result | Sample and context |
| --- | --- | --- |
| Development Recall@5 | **85.61% dense -> 92.93% BM25 + BGE** | Same 41 answerable development questions; final top 5 |
| Frozen hold-out Hit@5 | **80% (8/10)** | Ten answerable questions in the 12-question hold-out |
| Manual answer review | **12 full, seven partial, one failure** | 20 selected development questions; human review |

Recall measures **retrieved gold-page coverage, not answer accuracy**. The small hold-out and selected manual sample limit generalization. [Benchmarks and source artifacts](docs/benchmarks/FINAL_BENCHMARK_REPORT.md) separate retrieval, answer review and partial RAGAS scores.

<!-- Place a real screenshot of the running chat workspace here once captured. -->

## How it works

```text
PDFs -> page extraction -> section-aware chunks -> Qdrant indexes
React -> FastAPI/JWT -> optional follow-up rewrite
      -> BM25 top 20 -> BGE top 5 -> gate/context budget
      -> Groq -> citation check -> answer and sources
         PostgreSQL sessions | optional Langfuse traces
```

The API retrieves public documents only. Cached `BAAI/bge-reranker-base` selects five chunks; the highest score must reach **0.60**. Groq `openai/gpt-oss-120b` uses prompt **v1**, temperature 0, **7,680 input tokens** and a **1,024-token completion cap**, including reasoning. Both prompt versions remain available. Citations verify supplied document/page membership, not claim support; follow-ups use bounded history and fresh retrieval.

## Engineering challenges and lessons

- **Evidence selection - G003:** The RBI Payments Banks amendment's page 2 was BM25 rank 8, then reranker rank 12: it never reached generation. Local metadata enrichment moved it to rank 1, but the 41-question experiment also regressed other questions. **Not deployed:** aggregate gains did not establish a safe replacement. [Experiment details](docs/benchmarks/FINAL_BENCHMARK_REPORT.md#metadata-enrichment-experiment-not-deployed).
- **Citation formatting:** G012/G017 had correct answers but whitespace variants escaped parsing. The regex was corrected and manually verified; original benchmark labels remain partial passes.
- **Abstention contract:** Not-found text could carry `abstained=false`. Final-response normalization is implemented; offline tests cover model refusal/source preservation. A successful post-fix end-to-end API retest remains unverified.

## Run locally

Prerequisites: Python 3.10+, Vite-compatible Node.js (for example 22.16), Docker, Groq credentials and a prepared BM25-capable Qdrant index. PowerShell, from the repository root:

### 1. Backend and services

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,api,groq]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d postgres qdrant
```

In `.env`, replace the mock default with `REGULATRACE_LLM_PROVIDER=groq`, set `REGULATRACE_LLM_MODEL=openai/gpt-oss-120b` and your `REGULATRACE_GROQ_API_KEY`. Configure database connections; Compose exposes PostgreSQL on **5433**, Qdrant on **6333**. Keep secrets out of Git.

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
$env:REGULATRACE_JWT_SECRET = & .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"
.\.venv\Scripts\python.exe -m uvicorn regulatrace.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Keep a stable signing secret for deployment. Docker does **not** populate Qdrant: the ingest CLI writes dense chunks; BM25 needs separate backfill/index preparation. See [ingestion setup](docs/rag_notes.md#ingestion-and-indexing). First-request model loading adds latency.

### 2. Frontend - another terminal

```powershell
cd frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

Open [localhost:5173](http://localhost:5173). The frontend example sets `VITE_API_BASE_URL=http://localhost:8000`; backend routes have no prefix. CORS allows `http://localhost:5173`; configure other explicit origins through `REGULATRACE_CORS_ALLOWED_ORIGINS`. [Frontend details](frontend/README.md).

### 3. API and verification

[Swagger UI](http://127.0.0.1:8000/docs) lists 13 operations. Auth, chat, sessions, `/health` and `/ready` work; feedback, document/PDF and admin workflows return 501. Chat returns JSON. Readiness returns 503 when PostgreSQL/Qdrant is unavailable.

Checks: `.\.venv\Scripts\python.exe -m pytest -q` and `.\.venv\Scripts\python.exe -m ruff check .`; in `frontend/`, `npm run build` and `npm run lint`. Regulatrace passed **652 offline tests**, with three skips and five live-service cases excluded; Ruff, the frontend build and lint passed. The live deployment passed **19 hosted checks**, recorded in [the deployment receipt](docs/LIVE_DEPLOYMENT.json). See [validation scope](docs/FREE_FULLSTACK_VALIDATION.md).

<a id="stage-latency-and-estimated-cost-phase-6-step-62"></a>

Measured request p50 was **8.23 seconds** across 24 completed sequential requests (20 benchmark + four earlier); reranking dominated. Total excludes logging/network delivery. This is not a concurrency test or production SLA. Optional `.[tracing]` uses Langfuse settings in [.env.example](.env.example); chat continues if tracing fails. [Query-log reporting](scripts/query_log_report.py) summarizes timings and estimated cost, not invoices.

## Scope and limitations

- Argon2id/JWT authentication and owner-scoped sessions; tokens stay in memory. No refresh-token revocation/rotation, complex RBAC or persistent browser login.
- **10 chat requests/user/60 seconds**, with `Retry-After`. The serverless profile uses PostgreSQL transaction locks and shared counters; the default local profile uses process-local counters.
- Multi-page coverage, irrelevant evidence and incomplete/truncated answers remain failure modes. Metadata enrichment and page diversity are not production features.
- Full RAGAS evaluation is paused. Streaming and general-purpose OCR are deferred; one recovered scanned document has a recorded OCR sidecar. Delimiters, citation checks and redaction do not guarantee factual correctness.

## Technical documentation

- [Free full-stack deployment](docs/FREE_FULLSTACK_DEPLOYMENT.md)
- [Full-stack validation scope](docs/FREE_FULLSTACK_VALIDATION.md)
- [Final benchmarks and source artifacts](docs/benchmarks/FINAL_BENCHMARK_REPORT.md)
- [Original 20-question manual review](docs/benchmarks/2026-10-27-golden-set.md)
- [RAG workflow and security evidence](docs/rag_notes.md)
- [Retrieval trade-offs and failure analysis](docs/retrieval_notes.md)
- [Engineering decisions](docs/DECISIONS.md)
