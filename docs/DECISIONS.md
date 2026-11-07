# Engineering Decisions

The choices behind Regulatrace: what each component buys, what it costs and why alternatives were deferred. Original numbers and phase context preserve decision history; later entries supersede plans. Exact results live in the [benchmark report](benchmarks/FINAL_BENCHMARK_REPORT.md).

## Decision 001: Python Packaging and Repository Foundation (Phase 0, Step 0.2A)

**Accepted.** Use a `src/` package, `pyproject.toml`, a virtual environment, typed `pydantic-settings`, Ruff and pre-commit. This separates installed code from the working directory, makes dependencies reproducible and catches basic errors before review. Environment configuration keeps provider credentials outside source control; safe settings output masks secrets.

## Decision 002: LLM Interface, Embedding Model Selection & Query Instructions (Phase 0, Step 0.3)

**Accepted; provider support evolved.** Application code uses `LLMClient`/`get_llm()` rather than embedding provider SDK calls throughout the pipeline. Current implementations support mock, Gemini/Google and Groq; OpenAI is not an implemented provider. Mock clients support offline tests. The API currently requires Groq. Early BGE query-instruction plans were not adopted by the implemented dense search; see Decision 005.

## Decision 003: PDF Parser Selection (PyMuPDF) (Phase 1, Step 1.1)

**Accepted.** PyMuPDF gives page-level text and page numbers with a small parsing interface. Low-text pages are flagged rather than silently treated as complete evidence. Reading order and tables can still be imperfect; OCR is not implemented. PyMuPDF licensing needs review for the intended distribution/deployment model.

## Decision 004: Structure-Aware Chunking (Phase 1, Step 1.4)

**Accepted.** Detect section headings, split section bodies recursively and prepend the active heading to retained chunks. Preserve page/section metadata and prevent table-of-contents headings from contaminating later pages. This gives fragments local context, but heuristics can mislabel sections and page boundaries can separate related clauses. Chunk size/overlap were not tuned in the retrieval ablation.

## Decision 005: Embedding Model Selection (BAAI/bge-small-en-v1.5) (Phase 1, Step 1.5)

**Accepted for dense retrieval.** Local, normalized 384-dimensional BGE-small embeddings balance memory, inference cost and semantic retrieval. The model is English-focused. Current dense search encodes the query without the proposed `Represent this sentence for searching relevant passages: ` prefix; its impact was not isolated in an experiment. The selected BM25 first stage does not use dense query embeddings.

## Decision 006: Canonical 14-Field Metadata Schema (Phase 1, Step 1.4)

**Accepted.** Track `doc_id`, `title`, `regulator`, `circular_no`, `issue_date`, `effective_date`, `status`, `supersedes`, `section`, `page`, `access_level`, `source_url`, `content_hash` and `chunk_index`. These support provenance, filtering and citations, but depend on correct upstream values. PostgreSQL content hashes detect file changes; deterministic UUIDv5 point IDs from document/chunk IDs support repeatable upserts. Metadata alone does not make cross-index ingestion transactional.

## Decision 007: Frozen Retrieval Baseline and Failure Analysis (Phase 2, Step 2.6)

**Completed.** Freeze dense top-5 with BGE-small in `regulatrace_chunks` before comparing alternatives. Use the same 41 answerable development questions and definitions; keep 12 hold-out questions outside tuning. The configuration hash records settings, not an index snapshot. Early failures distinguished extraction gaps, missing gold pages, near-duplicate regulations and answer-bearing text absent from an otherwise matching page. Retain those cases in [retrieval failure analysis](retrieval_notes.md#failure-modes).

## Decision 008: Separate BM25 Retrieval and Development-Set Comparison (Phase 3, Step 3.1)

**Completed.** Use a separate `regulatrace_bm25` collection with `qdrant/bm25`, sparse vector `bm25` and IDF. This preserves the dense baseline and helps exact identifiers. The recorded 669 point IDs/payloads matched across the two collections. BM25 improved overall development recall but lost some semantic-ranking cases; it was not universally better. New ingestion does not automatically synchronize the separate indexes.

## Decision 009: Native Hybrid RRF and Retrieval Ablation (Phase 3, Step 3.2)

**Completed, not selected.** Preserve baselines in a separate `regulatrace_hybrid` collection. Manual RRF smoothing 60 corresponds to native `Rrf(k=61)` under the verified rank conventions. Initial hybrid evaluation prefetched five per branch; reranked hybrid later used 20 per branch. Fusion can demote gold pages and has tie-order variation. It underperformed BM25 before reranking and tied it afterward, adding complexity without a measured quality advantage.

## Decision 010: Metadata Pre-Filtering and Regulatory Status Limitations (Phase 3, Step 3.3)

**Accepted with limits.** Apply optional regulator and inclusive issue-date filters inside Qdrant, with the same filter on both hybrid branches. Store UTC `issue_date_iso` alongside original dates; the recorded backfill/index verification changed payloads, not embeddings. Regulatory status remains unverified: PostgreSQL ingestion status cannot justify active-only regulatory filtering. The later authenticated API enforces public-only access from server policy, not client claims. Filter correctness was checked; quality improvement was not ablated.

## Decision 011: Cross-Encoder Reranking and Quality-Latency Trade-off (Phase 3, Step 3.4)

**Completed and selected with BM25.** Cached `BAAI/bge-reranker-base` scores question + chunk text; keep retrieval and reranker scores separate. Rerank 20 candidates rather than all 669 chunks. BM25 matched hybrid's measured quality with fewer components and lower reranking p50. The cost is the dominant request-latency stage; improved page recall does not guarantee correct answers.

## Decision 012: Reranker-Based Abstention Threshold (Phase 3, Step 3.6)

**Experiment completed; now integrated.** Use highest score <0.60 or no candidates to refuse before generation. Development results were 4/7 correct refusals and 1/41 wrong refusals, with three unanswerable questions still accepted. A lower tested cutoff missed more refusals; a higher one wrongly refused 24/41 answerable questions. Overlapping scores are not calibrated answer confidence. Current code also normalizes generated canonical not-found text; a successful manual API retest is not established by saved evidence.

## Decision 013: Retain the Measured Retrieval Configuration for Deployment Preparation

**Accepted, 2026-11-01.** Retain BM25 top 20 -> BGE top 5, gate 0.60. Depth 20 beat 10 on recall; 30 tied recall with worse MRR/latency. Hold-out was evaluated once; chunk settings/final depth stayed fixed. Local metadata enrichment recovered G003 and improved aggregates but regressed other questions, with console evidence only and no saved ablation. It was **not deployed**; page diversity is also unimplemented. Prioritize reproducible benchmarks and deployment preparation over adopting that trade-off. Known failures remain; deployment is not claimed.

## Decision 014: Pause Full RAGAS Evaluation and Preserve Partial Evidence

**Accepted, 2026-11-01.** Keep four official RAGAS 0.4.3 metrics, prompt v1 and local BGE-small embeddings. Earlier Gemini judging produced 40/40 scores on five questions; a later Groq run on five different questions saved 32/40 because of a refusal and HTTP 429 failures. The current async Groq/Instructor JSON-schema judge uses GPT-OSS-120B, temperature 0 and bounded SDK retries, without the old fixed Gemini pause. Full 41-question evaluation is paused; do not pool judges, subsets or missing-score denominators. Ten historical faithfulness reviews include a documented disagreement, not a fabricated proven judge error. High faithfulness can coexist with a wrong answer grounded in irrelevant text; sharing a generator/judge model adds correlated-error risk.

## Where to inspect the evidence

[Retrieval notes](retrieval_notes.md) retain depth controls, filter/RRF details and historical failures. The [final benchmark](benchmarks/FINAL_BENCHMARK_REPORT.md) links raw reports and separates retrieval, manual answer review and partial RAGAS results. The [original manual benchmark](benchmarks/2026-10-27-golden-set.md) preserves the observed labels; later code changes do not retroactively improve them.
