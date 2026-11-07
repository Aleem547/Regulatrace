# Regulatrace deployment handoff

The source was published to `Aleem547/Regulatrace`. The selected free deployment
is the separate Gradio/ZeroGPU profile for `abdul547/Regulatrace` on Hugging Face.
Original source credit is retained in `UPSTREAM.md`.

## Use the Hugging Face package

Download and extract `Regulatrace_HuggingFace_Package.zip`. Open PowerShell in
the folder containing `Publish-HuggingFace.ps1`. Add the private Space Secret
`REGULATRACE_GROQ_API_KEY` using your own Groq Free Plan key. Then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Publish-HuggingFace.ps1
```

Authenticate privately with a Hugging Face write token if prompted. This token
is distinct from the Groq key and should never be pasted into chat. The uploader
checks the existing owner/Space and free hardware; it does not clone a repository
or enable paid resources. See `docs/FREE_SPACE_DEPLOYMENT.md`.

The free profile supports uploaded PDFs, BM25 retrieval, BGE reranking, grounded
answers, citation checks and temporary conversations. Persistent accounts/history
remain part of the separate full React/FastAPI deployment. The original 42-PDF
corpus is not included. Hosted GPU/Groq success still needs verification.

## Update the existing GitHub checkout

The Hugging Face package includes `Update-GitHub.ps1` and a reviewed source patch.
They update the existing Windows repository, commit and push to the intended
`Aleem547/Regulatrace` remote. No force push or history replacement is used.
The script checks for a clean branch and matching remote head before mutation.
Do not initialize a new repository over the already-published source.

## Full application deployment

For the persistent React/FastAPI version, follow `docs/DEPLOYMENT.md` and
`deploy/production.env.example`. It still needs PostgreSQL, a compatible Qdrant
index, private settings and the actual regulatory corpus. The original initial
publisher is retained for reference, not as the update workflow for this release.

## Validation

Read `docs/FREE_SPACE_VALIDATION.md` and the updated readiness report. The Gradio
application was checked locally with controlled inference; actual ZeroGPU and
Groq calls, Windows script execution and the final hosted smoke test remain open.
