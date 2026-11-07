# Local validation - 5 November 2026

Source: the supplied 124-file Regulatrace-main.zip; upstream attribution is in UPSTREAM.md.
No public deployment was completed. The user subsequently published the source
from Windows; on 6 November 2026 all 134 files at main commit
`8d07a338bba12cc7c15c5d0f337aab7481005495` were verified against the reviewed package.

## Passed checks

- Offline backend subset: **549 passed, 3 skipped, 8 deselected** (560 collected).
- All 17 new deployment regressions passed as part of that subset.
- Ruff: passed for the complete source tree.
- Frontend: TypeScript and Vite production build passed with VITE_API_BASE_URL=/.
- Frontend Oxlint: passed.
- A fresh Python 3.12 runtime environment installed all 99 pinned runtime dependencies.
  Imports and bundled frontend/API route checks passed in that environment.
- Explicit psycopg2 driver selection and missing-corpus rejection passed.
- Alembic generated all three migrations as PostgreSQL SQL using a synthetic URL
  containing percent-escaped password characters; no live database was modified.
- Git whitespace checks passed. The previously ignored data/raw/.gitkeep is now tracked.

## Test selection and limits

The three skipped tests are the optional RAGAS adapter (RAGAS is not installed in
this API deployment environment) and two explicit-opt-in live Groq prompt tests.
Eight existing integration cases were deliberately deselected because they need
PostgreSQL/Qdrant services and/or downloaded model weights:

- test_embedding_factory_and_demonstration
- test_get_embedding_dimension
- test_embed_chunks_batching_and_normalization
- test_qdrant_collection_init_and_payload_indexes
- test_qdrant_upsert_deduplication_and_query
- test_document_registry_model
- test_test_a_ingest_idempotency_run_twice
- test_test_b_modify_one_document_and_reingest

The initial full baseline attempt recorded 530 passed, 3 skipped, 7 failures and
3 errors. Failures exposed unavailable external services/model weights, a missing
SOCKS dependency in the execution environment, and generic PostgreSQL URLs selecting
an uninstalled driver under current SQLAlchemy. The driver selection was corrected;
SOCKS support was installed in the validation environment. Integration cases were
not converted into passing offline results.

The final offline command was:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q --tb=short -k 'not (test_embedding_factory_and_demonstration or test_get_embedding_dimension or test_embed_chunks_batching_and_normalization or test_qdrant_collection_init_and_payload_indexes or test_qdrant_upsert_deduplication_and_query or test_document_registry_model or test_test_a_ingest_idempotency_run_twice or test_test_b_modify_one_document_and_reingest)'
```

A Docker daemon and PowerShell were unavailable here, so the Docker build/run and
PowerShell publisher were not executed. The frontend was built directly and its
bundled routing was exercised with FastAPI TestClient. No browser QA, hosted chat,
model memory/latency benchmark, live migrations, corpus ingestion or index recovery
was completed. The 42 manifest PDFs are absent. New accuracy/F1 claims are not made.

Historical retrieval JSON artifacts retain the upstream development Recall@5 of
0.9292682926829269 and hold-out Hit@5 of 0.8. Reading these files confirms their
recorded values; it does not rerun the models or establish new authorship.

## Documentation correction - 6 November 2026

The earlier free Docker Space recommendation omitted Hugging Face's paid-account
creation requirement and PostgreSQL networking constraint. Hosting guidance is now
corrected. This follow-up changes documentation only; application code is unchanged
from the tested and published source. No new test run is claimed.

Primary references:
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/huggingface_hub/main/guides/manage-spaces
