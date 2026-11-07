# RAG Workflow - From PDF to Cited Answer

Regulatrace separates three jobs: **find evidence, generate an answer, check its references**. Each can fail independently. This separation makes a wrong answer diagnosable rather than hiding every problem behind the model.

## Ingestion and indexing

[parse_pdf()](../src/regulatrace/ingestion/parser.py) uses PyMuPDF to preserve page-level provenance and 1-based page numbers. Cleaning removes common noise; heading detection avoids carrying table-of-contents headings into later sections. Low-text pages are flagged; there is no OCR, so missing extraction means missing evidence.

[chunk_pages()](../src/regulatrace/ingestion/chunker.py) splits section bodies and prepends headings so fragments retain context. Defaults are **1,000 characters, 150-character overlap**, not tokens; headings can add length. Page-scoped chunks preserve citations but can split related clauses. These settings were not tuned.

Chunk metadata records document/page/section, regulatory fields, access level and chunk index. PostgreSQL hashes detect unchanged PDFs; deterministic UUIDv5 IDs make Qdrant upserts repeatable. They do not make updates across both stores transactional.

Dense indexing uses local normalized **384-dimensional `BAAI/bge-small-en-v1.5` embeddings** in `regulatrace_chunks`. BM25 uses `regulatrace_bm25`, `qdrant/bm25` and IDF. Separate indexes preserve comparable baselines; recorded experiments used 669 chunks, not a current live-corpus audit.

**Fresh setup:** `python -m regulatrace.ingest --path data/raw` populates dense only. Use existing [BM25 setup/backfill functions](../src/regulatrace/bm25.py) separately with compatible server-side inference, or restore the prepared index. The [manifest](../data/manifest.csv) supplies metadata. Docker startup does not populate or synchronize indexes.

## Retrieval and reranking

The API enforces `access_level=public` in Qdrant, retrieves **up to 20 BM25 candidates**, then keeps **five** using cached `BAAI/bge-reranker-base`. BM25 finds lexical candidates cheaply; the cross-encoder reads each with the question for finer ranking. Development page recall improved over dense, at substantial latency cost.

Production reranker inputs are `(question, chunk text)`; retrieval and reranker scores stay separate. A local metadata-enrichment experiment gained aggregate quality but regressed individual questions; it was not deployed. Page-diversity selection is also absent. See [retrieval trade-offs](retrieval_notes.md).

## Evidence gate, prompt and token budget

The **0.60** highest-score gate avoids generating from weak candidates. Empty/lower-scoring evidence refuses before generation. Answerable and unanswerable scores overlap: passing is not calibrated confidence or proof of answerability.

[build_rag_chain()](../src/regulatrace/rag.py) keeps source-labeled blocks in ranked order, rejecting blank text/IDs and invalid pages. Whole blocks must fit **both** prompts, giving v1/v2 identical evidence without cutting clauses, tables or metadata. If none fits, it skips generation and returns `not found in the provided documents`.

Cached `o200k_harmony` counts the **full rendered prompt**: instructions, question, metadata, delimiters and text. Limits are **7,680 input + 1,024 completion tokens**, including reasoning. The 8,704 total is an application budget, not the model context window. Provider framing adds tokens; capped answers can end early.

Groq `openai/gpt-oss-120b` runs at temperature 0. [v1](../prompts/answer_v1.txt) is default; [v2](../prompts/answer_v2.txt) stays versioned. Both require evidence-only claims, citations and the exact fallback. The saved one-question comparison establishes no full-set winner. Git/prompt hashes aid reproduction, not provider determinism.

## Citations and abstention are different checks

The parser accepts `[doc_id p.N]` and `【doc_id p.N】`, including internal bracket whitespace. It requires matching brackets and positive pages. Validation checks each exact document/page against **actually supplied** source records, preserves duplicate occurrences and reports `valid / parsed` citation precision; zero citations gives `None`.

Invalid references are flagged without rewriting answers. Membership proves neither claim support, correct bank category nor completeness. Frontend `[1]` markers change presentation only; stored history retains validated document/page pairs, not full metadata.

The chain now normalizes canonical not-found text to `abstained=true`, preserving sources. The [original benchmark](benchmarks/2026-10-27-golden-set.md) recorded false flags; a successful post-fix API retest is unverified. Historical labels stay unchanged.

## Follow-up questions and the API boundary

Routers handle HTTP, services coordinate RAG and repositories persist data. Owned session IDs continue conversations; missing/other users' sessions return the same 404. User/assistant messages commit together, without holding a transaction during generation.

Follow-ups use at most **five complete exchanges / 8,000 serialized characters** to rewrite a standalone retrieval question with the cached Groq client. Unusable output/failure falls back to the original question. History resolves references; fresh retrieval supplies facts. Rewriting adds a call and failure risk, with no quality ablation. Sequential requests avoid competing history snapshots.

## Observability and safe failure

Optional Langfuse traces expose **retrieval, reranking and generation**: redacted query/answer, document/chunk IDs, scores, prompt version, configuration hash, timings and available usage. These help locate an evidence miss versus a generation omission. Hashes omit index snapshots. Tracing fails open; raw prompts/full history are not deliberately exported.

PostgreSQL independently logs completed-request timings, provider usage and estimated cost; failed/rejected requests are excluded. Provider usage differs from prompt-text counts; cost needs matching prices/complete usage and is not an invoice. Total excludes log writing/network delivery. Generation failures preserve supplied sources; dependency failures return safe structured errors.

## Security evidence and limits

Retrieved instructions are **untrusted data** inside explicit delimiters. Since the model has no action-taking tools, the main injection risk is a manipulated answer. Groq receives the rendered string in a user message, so textual system labels do not establish enforced role separation. Redaction covers common personal identifiers; tracing/current RAGAS errors also mask common credentials. This is best-effort, not anonymization; saved answers/contexts and third-party SDK logs are not globally sanitized.

<details>
<summary>Retained manual checks and historical learning evidence</summary>

- **Phase 2:** Ten manual dense-RAG queries with Groq `qwen/qwen3.8-27b`, k=5: six correct, one partial (unsupported FIRE additions), and three abstentions. The ETF fallback was a retrieval miss; Digital Accessibility had no indexed text; the repo-rate query lacked corpus evidence. Reranking/citation validation were absent at that stage. These are historical observations, not current benchmark scores.
- **Prompt injection:** 20 offline security checks and two live Groq prompt checks were recorded as passed. Mocked tests established assembly/redaction only. A separate synthetic PDF traversed real extraction, chunking, isolated BM25, BGE and Groq. It contained both a seven-calendar-day rule and an instruction to claim cancellation/output `HACKED`. The model kept the rule, cited page 1 and did not follow the attack; rerank score 0.999966, one valid citation, zero invalid. The temporary collection was deleted; production collections/PostgreSQL were unchanged. This did not test the registry or HTTP API, and one attack is not general resistance.
- **Privacy:** A synthetic ingestion-error message redacted an email, phone, PAN and Aadhaar. Current credential-aware error handling has offline regression coverage; old artifacts were not sanitized retroactively.
- **LLM judges:** Existing notes record ten manual faithfulness reviews and a G002 disagreement, without claim-level explanations proving the exact judge error. High faithfulness can still accompany a wrong answer grounded in irrelevant context. See [partial RAGAS findings](benchmarks/FINAL_BENCHMARK_REPORT.md#partial-ragas-findings).

</details>
