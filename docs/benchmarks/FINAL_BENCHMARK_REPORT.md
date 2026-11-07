# Regulatrace - Final Benchmark Report

**Evidence snapshot: 2026-11-01.** Selected pipeline: **BM25 top 20 -> `BAAI/bge-reranker-base` top 5 -> 0.60 gate -> Groq GPT-OSS-120B, prompt v1**. This report combines saved artifacts, manual review and a separately labeled console-observed experiment. No evaluation was rerun. Metadata enrichment and page diversity are not production features.

## Dataset and definitions

The [60-question golden set](../../data/eval/golden.jsonl) contains **48 development questions (41 answerable, seven unanswerable)** and **12 frozen hold-out questions (ten answerable, two unanswerable)**. Hold-out was evaluated once for final Phase 3 retrieval, not used for later selection.

Hit@5 measures any gold document/page match; Recall@5 averages each question's fraction of distinct gold pages retrieved; MRR uses the first gold match within five. All exclude unanswerable questions. **These measure retrieval, not answer accuracy**: even a matching page's returned chunk may lack the required clause.

## Retrieval comparison - 41 answerable development questions

| Method / artifact | Hit@5 | Recall@5 | MRR | Rerank-only p50 ms |
| --- | ---: | ---: | ---: | ---: |
| [Dense BGE-small][dense] | 0.9268 | 0.8561 | 0.8862 | N/A |
| [BM25][bm25] | 0.9512 | 0.9049 | 0.8943 | N/A |
| [Native hybrid RRF][hybrid] | 0.9512 | 0.8927 | 0.8476 | N/A |
| **[BM25 + BGE reranker][bm25-rerank]** | **0.9512** | **0.9293** | **0.8951** | **2,598.60** |
| [Hybrid + BGE reranker][hybrid-rerank] | 0.9512 | 0.9293 | 0.8951 | 2,869.78 |

Selected Recall@5 improved **85.61% -> 92.93%** over dense. Hybrid reranking tied quality with more components. Depth 20 beat 10 on recall; 30 tied recall but worsened MRR/latency. [Depth controls](../retrieval_notes.md) retain details. Reranking p50 excludes retrieval/model loading; N/A means no reranker. Chunk size/overlap and final depth were not tuned.

### Frozen hold-out - selected configuration only

| Answerable / unanswerable | Hit@5 | Recall@5 | MRR | Retrieval p50 ms | Rerank p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 / 2 | 0.8000 | 0.8000 | 0.7333 | 30.09 | 3,397.60 |

The [hold-out report][holdout] records **2/2 correct refusals, 0/2 missed refusals, 1/10 wrong refusals and 9/10 correct acceptances**. The [development gate analysis][threshold] recorded 4/7 correct refusals, 3/7 missed refusals, 1/41 wrong refusals and 40/41 acceptances at the tested cutoff immediately below the selected 0.60. Scores are not calibrated confidence. Hold-out Hit@5 is **8/10**, a small sample with substantial uncertainty.

## Manual answer review - 20 development questions

Twenty independent, sequential authenticated `/chat` conversations covered **17 answerable and three unanswerable development questions**, with no hold-out overlap. Human review checked content, completeness, citations and the response contract.

| Manual outcome | Count | Share |
| --- | ---: | ---: |
| Full pass | 12/20 | 60% |
| Partial pass | 7/20 | 35% |
| Failure | 1/20 | 5% |

The [original report](2026-10-27-golden-set.md) retains exact questions, references and observations. G003 missed page 2 and refused; G007/G009 omitted details; G008 missed pages 5/7 and ended mid-sentence. G012/G017 exposed citation-whitespace parsing; G003/G029/G030 exposed false abstention flags. Later corrections do not upgrade historical labels. Successful post-fix API abstention retesting is unverified; G008 completion-budget exhaustion remains a hypothesis without finish-reason/usage evidence.

## Recorded latency and estimated cost

The original report records [query_log_report.py](../../scripts/query_log_report.py) output over **24 completed sequential requests: 20 benchmark + four earlier requests**, not an isolated benchmark cohort.

| Stage | p50 ms | p95 ms |
| --- | ---: | ---: |
| Total | 8,233.57 | 16,663.63 |
| Rewrite | 0.00 | 0.00 |
| Retrieval | 81.46 | 99.41 |
| Reranking | 5,381.19 | 13,098.42 |
| LLM generation | 2,139.55 | 3,329.29 |

**Reranking dominated.** Total excludes query-log writing/network delivery; failed/rejected requests are absent and skipped stages are zero. Percentiles cannot be summed; zero rewrite latency says nothing about follow-up performance. Estimated mean cost: **$0.00064260 over 22/24 records with cost data**, not an invoice or isolated benchmark average. No per-question costs/tokens were supplied. This small sequential sample establishes neither a production SLA nor concurrent-user performance.

## Metadata enrichment experiment (not deployed)

**Evidence: console-observed results supplied by the project author; no saved machine-readable ablation report.** The local 41-question development experiment added metadata to reranker inputs:

| Reranking input | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| Original chunk text (selected) | 0.9512 | 0.9293 | 0.8951 |
| Metadata-enriched (experimental) | 0.9756 | 0.9374 | 0.9146 |

G003 page 2 was BM25 rank **8**, original reranker rank **12**, then experimental rank **1**. Improved IDs: **G003, G017, G021**; regressed IDs: **G022, G026, G051, G053, G056**. Aggregate gains hid question-level regressions. Production still reranks chunk text only; the experiment neither fixes the historical benchmark nor has the reproducibility of a saved ablation.

## Partial RAGAS findings

RAGAS 0.4.3 used prompt v1, Groq GPT-OSS-120B generation and local BGE-small embeddings. Faithfulness estimates context support; answer relevancy measures question alignment; context precision measures ranked-context relevance; context recall measures reference coverage. None guarantees regulatory correctness.

The [Gemini pilot][ragas-gemini] scored **40/40 metrics on G002-G006**. The [Groq run][ragas-groq] scored **32/40 on G007-G011**: dense G007 refusal left four unavailable; four selected-pipeline metrics failed with HTTP 429. Cells show mean **(scored / five questions)**. Missing scores and refusals are not counted as successes.

| Judge / subset | Metric | Dense | BM25 + reranker |
| --- | --- | ---: | ---: |
| Gemini 3.5 Flash-Lite / G002-G006 | Faithfulness | 0.7333 (5/5) | 0.9750 (5/5) |
| Same | Answer relevancy | 0.8208 (5/5) | 0.8078 (5/5) |
| Same | Context precision | 0.8667 (5/5) | 0.8000 (5/5) |
| Same | Context recall | 1.0000 (5/5) | 0.8000 (5/5) |
| Groq GPT-OSS-120B / G007-G011 | Faithfulness | 1.0000 (4/5) | 1.0000 (5/5) |
| Same | Answer relevancy | 0.8427 (4/5) | 0.8453 (3/5) |
| Same | Context precision | 0.9719 (4/5) | 0.8208 (4/5) |
| Same | Context recall | 0.8750 (4/5) | 0.9375 (4/5) |

Judges used temperature 0; **do not pool judges/subsets**. Full 41-question evaluation is paused because of API limits. Current Groq/Instructor judging uses bounded SDK retries and shares the generator's model, risking correlated errors. Ten historical faithfulness reviews record G002 dense's 0.50 as a disagreement without claim-level reasoning, and G003 BM25's 0.875 alongside a wrong bank-category answer. **High faithfulness can mean faithful use of irrelevant evidence.** These observations do not prove a specific judge error.

## What remains uncertain

- Multi-page clauses, near-duplicate regulations and extraction gaps remain. G003's later local diagnosis locates reranking loss; other cases lack that evidence.
- Selected development data and citation membership do not establish general answer correctness. The gate still accepts some unsupported questions.
- Metadata enrichment has console evidence only; page diversity, filter-quality gains and follow-up rewriting have no quality ablation.
- A one-question v1/v2 comparison cannot select a winner. Configuration hashes omit index snapshots; manual requests lack revision/completion metadata.
- Reported Gemini 503 attempts lack an identified saved report. No missing measurements, deployment or production-capacity results are inferred.

<details>
<summary>Additional artifact references and reproduction limits</summary>

[Depth 10][top10] and [depth 30][top30] retain tuning controls. Overlapping G002 pilots: [Groq, October 18](../../eval/results/ragas_20261018T190009040390Z.json), [Gemini first attempt](../../eval/results/ragas_20261019T102352104373Z.json), [Gemini retry](../../eval/results/ragas_20261019T103313019156Z.json), [Groq, November 1](../../eval/results/ragas_20261101T103812608035Z.json). These are not independent samples. The [one-question prompt comparison](../../eval/results/prompt_comparison_20261018T174856960410Z.json) predates parser corrections and reported zero citations. Artifacts remain unchanged. Future measurements need revision/configuration, completion metadata and request/question linkage; tune on development data only.

</details>

[dense]: ../../eval/results/retrieval_20261012T191839173509Z_af6558ebe4a5bce1f853b19340b8acdfe77ea87db78f6dd033f4fb3c73a9c58f.json
[bm25]: ../../eval/results/retrieval_bm25_20261014T095834356877Z_5047cbf0cbf3f1ac63547757db03dfa232d1159d1d06350c5abca79b22162fe8.json
[hybrid]: ../../eval/results/retrieval_hybrid_20261014T112307814856Z_d0bfed4a543cfd45b7657595060f953681e1ee0360400368f28007db81c1c702.json
[bm25-rerank]: ../../eval/results/retrieval_bm25_rerank_20261015T122408293824Z_4d93aa37e83e77ea2458a5c9e1a6760db1bea494d877701f961c1eac06d4cb9c.json
[hybrid-rerank]: ../../eval/results/retrieval_hybrid_rerank_20261015T123138677133Z_49abbae93ee83b902bf70141a714133d964b26eab106e1e8fcf75314d36e286e.json
[top10]: ../../eval/results/retrieval_bm25_rerank_20261015T180050898462Z_3367c2ac2731ca5fac0c3682446a229998352d49915bdf23281a47723a20a3eb.json
[top30]: ../../eval/results/retrieval_bm25_rerank_20261015T180615157410Z_18c49fb4ca41841eb7ea583de346eef9649e4a0e15408469b4397c2d23a7aaf5.json
[holdout]: ../../eval/results/retrieval_bm25_rerank_holdout_20261015T182257754654Z_15f93293e5ee0669801fa0d69164bf1c69a9379717dfd17791ba89f2c65cef46.json
[threshold]: ../../eval/results/rerank_score_analysis_20261015T131329110974Z.json
[ragas-gemini]: ../../eval/results/ragas_20261019T105250523860Z.json
[ragas-groq]: ../../eval/results/ragas_20261101T105804471994Z.json
