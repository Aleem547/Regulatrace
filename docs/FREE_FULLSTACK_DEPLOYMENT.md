# Free full application deployment

The upload demo is live at <https://huggingface.co/spaces/abdul547/Regulatrace>.
This profile prepares the React application with login, saved conversations and a
permanent regulatory corpus. Its hosted deployment remains pending account setup
and real end-to-end verification.

| Service | Role | Selected profile |
| --- | --- | --- |
| Vercel frontend | React/Vite interface | Hobby; personal/noncommercial use |
| Vercel backend | FastAPI, authentication and orchestration | Hobby; Python 3.12; 240-second request cap |
| Neon | Users, messages, sessions, document registry and shared rate limits | Free project; dedicated `regulatrace` database |
| Qdrant Cloud | Persistent dense vectors and separate BM25 evidence index | Free cluster; cloud BM25 inference |
| Existing Hugging Face Space | BGE reranking and optional dense query embeddings | Existing Gradio/ZeroGPU hardware |
| Groq | Answers and bounded follow-up rewriting | User's free API allowance |

The original default chat pipeline remains BM25 top 20, BGE reranker top 5 and Groq,
with the original refusal threshold and citation checks. Dense vectors are also
stored for the existing dense/hybrid retrieval tools. No model weights are installed
in the Vercel API or on the Windows deployment machine.

## 1. Create the free service resources

In [Neon](https://console.neon.tech), create a **Free** project for Regulatrace and a
database named **regulatrace**. Copy that database's connection URL with TLS
(`sslmode=require`). The API supports the pooled URL; the deployment uses NullPool
and omits unsupported session startup options. Keep this database separate from
the news verifier or other projects.

In [Qdrant Cloud](https://cloud.qdrant.io), create a **Free** cluster. Copy the HTTPS
cluster endpoint and its API key. The script does not create or upgrade a cluster.

Use a Vercel **Hobby** account. The launcher verifies its current plan before it
creates uniquely named Regulatrace API/frontend projects. It never reuses the news
verifier's projects. Keep `abdul547/Regulatrace` on the existing free ZeroGPU hardware.

Have your Groq key available. Optionally create a limited Hugging Face read token
for authenticated ZeroGPU API calls. Enter secrets only in the local prompts.
Do not paste them into chat, GitHub files or public frontend variables.

Free services have usage limits and cold starts. Hugging Face currently documents
five GPU minutes/day for an authenticated free account and a shared two-minute
anonymous allowance. A backend token shares its quota across visitors. The launcher
contains no paid-plan, billing, hardware-upgrade or card-entry operation. Free plans
can change and do not guarantee always-on availability.

## 2. Run the Windows launcher

Extract `Regulatrace_Fullstack_Package.zip` into a new folder. Python 3.12+ and Node.js
22 LTS+ must be installed. Open PowerShell in the folder containing
`Deploy-Regulatrace.ps1`, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Deploy-Regulatrace.ps1
```

Enter the private settings and complete official Vercel/Hugging Face browser sign-ins
when prompted. The existing Hugging Face CLI login on this Windows account can be reused.

The package includes 42 recovered manifest PDFs, 206 pages and 672 prepared BGE-small
vectors. The embedding revision is pinned to
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The two scanned pages of
`1785493077212.pdf` use recorded Tesseract English OCR at 240 dpi. Original PDF bytes
and page numbers are preserved. These current official downloads are not byte-verified
historical benchmark inputs. Original PDFs and prepared vectors are excluded from Git.

The launcher validates the prepared data before remote writes, refuses unrelated
database tables or index evidence, migrates Neon and resumes only corpus-specific
Qdrant collections. It confirms both 672-point indexes and the 42-document registry.
It adds the worker secret to the existing Space, uploads reviewed source and waits
for the model endpoint's health check. The upload UI remains included.

It then deploys the API/frontend, gives the frontend only `VITE_API_BASE_URL` and
restricts CORS to the real frontend origin. A successful run checks signup/login,
real retrieval/generation, an RBI page-8 citation, follow-up memory, persistent
history, owner isolation, refresh, shared rate limiting, CORS and frontend assets.
Temporary test accounts and their identified owned test rows are removed afterward.

Success writes `DEPLOYMENT_RESULT.json` with actual URLs and checks. A failed or
stopped run is not a completed deployment. Re-run the same command to resume its
saved project IDs and matching index uploads. The complete account-based sequence
and PowerShell wrappers have not been executed in this review.

Private settings and resume state stay in `.fullstack-state`, excluded from source
uploads and Git. Keep this folder private and preserve its stable JWT/worker secrets.

## 3. Publish source and the verified link to GitHub

After deployment verification succeeds, run from the same folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Update-GitHub.ps1
```

The default existing Git folder is
`Downloads\Compressed\Regulatrace_Deployment_Package\Regulatrace` under your Windows
profile. Override it with `-RepositoryPath 'C:\your\Regulatrace'` if necessary.

The script checks the exact GitHub repository, requires a clean main branch,
fetches and fast-forwards to `origin/main`, validates the reviewed patch and adds
the verified live URL to README. It commits the source allowlist and pushes normally.
An upstream conflict stops publication for review. There is no force push or saved
credential extraction. Original source attribution is retained.

To add the same URL in GitHub's About sidebar, use **About → gear → Website**.

## Scope

Authentication, JSON chat, saved sessions and the public regulatory corpus form
the deployed core. Original feedback, document/PDF-download and admin/job workflows
still return HTTP 501. Private uploads stay in the separate temporary upload demo.
Citation validation checks supplied document/page membership rather than proving
every factual claim. Historical retrieval benchmark numbers are unchanged.

## References

- [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python)
- [Vercel FastAPI](https://vercel.com/docs/frameworks/backend/fastapi)
- [Vercel Hobby](https://vercel.com/docs/plans/hobby)
- [Neon plans](https://neon.com/pricing)
- [Neon connection errors](https://neon.com/docs/connect/connection-errors)
- [Qdrant plans](https://qdrant.tech/pricing/)
- [Qdrant cloud inference](https://qdrant.tech/documentation/cloud/inference/)
- [Space API and ZeroGPU quotas](https://huggingface.co/docs/hub/spaces-api-endpoints)
