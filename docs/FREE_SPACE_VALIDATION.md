# Free Space validation - 6 November 2026

Current local results: **575 passed, 3 skipped, 7 deselected** in one regression
run after the build repair. This does not establish hosted GPU or Groq success.

| Check | Observed result |
| --- | --- |
| Current regression selection | 575 passed, 3 skipped, 7 deselected |
| Focused Space and publisher files | 25 passed; included in the regression selection |
| SDK dependency installation | 145 registry dependencies resolved with the platform extras; installation dry-run passed |
| Real FastEmbed BM25 | Model downloaded and produced overlapping query/document terms |
| Gradio HTTP | Mounted app returned HTTP 200; config reported Regulatrace and enabled queue |
| Root app entry point | UI and GPU-decorated ranking callback constructed using a fake model; no real GPU inference |
| Lint | Ruff passed for source, tests, root app and publisher |

Tests cover real PDF extraction and local Qdrant with synthetic sparse vectors,
separate session evidence, upload bounds, source provenance, missing/foreign
citations, abstention without generation, follow-up bounds, name aliases and
secret masking. Publisher tests use a fake API and verify account/hardware/target,
file hashes, secret exclusion and optimistic remote-head protection. Build checks
cover the absence of local-project installation, the SDK's actual OAuth/MCP
metadata, redacted diagnostic logs and stale build status during a retry.

The seven excluded integration cases require PostgreSQL (three), Qdrant/model
services (two), or additional real embedding work (two). The small embedding
dimension case did run successfully. Three skips are the optional RAGAS adapter
and two opt-in live Groq prompt checks.

Reproduce the selected regression run in an appropriately installed environment:

```powershell
python -m pytest -q `
  --deselect=tests/test_llm.py::test_embedding_factory_and_demonstration `
  --deselect=tests/test_vectorstore.py::test_embed_chunks_batching_and_normalization `
  --deselect=tests/test_vectorstore.py::test_qdrant_collection_init_and_payload_indexes `
  --deselect=tests/test_vectorstore.py::test_qdrant_upsert_deduplication_and_query `
  --deselect=tests/test_ingest.py::test_document_registry_model `
  --deselect=tests/test_ingest.py::test_test_a_ingest_idempotency_run_twice `
  --deselect=tests/test_ingest.py::test_test_b_modify_one_document_and_reingest
```

Validation used Python 3.12.14, CPU Torch 2.13.0, Gradio 6.31.0, FastEmbed 0.9.1,
Sentence Transformers 6.2.0, Qdrant Client 1.20.0 and Pydantic 2.12.5. Gradio's
OAuth/MCP extras were installed for the repair regression. The Space lock selects
CUDA-enabled Torch 2.13.0. `spaces==0.51.3` is co-resolved but not emitted because
Hugging Face supplies its platform version. The CUDA install was planned, not
executed locally; the actual GPU build remains pending.

## First hosted build and repair

The user authenticated and uploaded all 45 reviewed files to abdul547/Regulatrace
at commit ca51425726559f164355a21cc0c7a92ec1b6dc7c. The server returned BUILD_ERROR
in its requirements-installation step. Inspection found two incompatible parts
of the supplied package: `-e .` required source before the SDK copied it, and
Pydantic 2.14.3 conflicted with the SDK-injected Gradio MCP extra's upper bound.
Both failures were reproduced locally. The corrected registry-only lock selects
Pydantic 2.12.5 and includes Gradio's OAuth/MCP dependencies. It resolves together
with the actual platform installation requirements.

Still unverified: the actual ZeroGPU scheduling/weight-registration path, real
BGE reranker inference, live Groq generation, hosted citation/restart behavior,
and execution of the corrected PowerShell upload on Windows. The user's earlier
upload succeeded, but no repaired build has been published from this session.
The uploader cannot run here because there is no authenticated Hugging Face token.
Source-patch and upload-bundle consistency are checked separately while packaging.

Historical upstream Recall@5/Hit@5 figures remain historical. No new accuracy,
precision, recall/F1 benchmark or hosted memory/latency measurement is claimed.
