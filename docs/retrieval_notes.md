# Retrieval Notes - Why BM25 + Reranking?

**Selected API configuration:** public-only BM25 top 20 -> cached `BAAI/bge-reranker-base` top 5 -> highest-score gate 0.60. Retrieval scores stay separate from `rerank_score`. Final depth, chunk size and overlap were not tuned. The [final benchmark](benchmarks/FINAL_BENCHMARK_REPORT.md) holds consolidated metrics and hold-out results.

## Dense, BM25 and hybrid

| First stage | How it retrieves | Useful property | Observed trade-off |
| --- | --- | --- | --- |
| Dense BGE-small | Similarity of normalized 384-dimensional vectors | Semantic matches beyond exact wording | Similar bank-category regulations can outrank the requested provision |
| BM25 | Sparse term matching with IDF | Exact identifiers and regulatory wording | Can miss semantic matches; raw scores are not comparable to dense scores |
| Hybrid RRF | Fuse dense and BM25 ranks | Combine lexical and semantic candidates | Gold pages can be demoted; native equal-score order can vary |

The frozen baseline is [dense top 5][dense] in `regulatrace_chunks`. [BM25][bm25] uses `regulatrace_bm25`, `qdrant/bm25`, vector `bm25` and IDF. [Native hybrid][hybrid] uses a separate `regulatrace_hybrid` collection, protecting both baselines. Recorded verification found 669 matching point IDs/payloads across indexes; this review did not inspect live collections. New ingestion does not automatically synchronize them.

On **41 answerable development questions**, Recall@5 rose **85.61% dense -> 90.49% BM25 -> 92.93% BM25 + BGE**. [Hybrid reranking][hybrid-rerank] tied selected quality but added components and had higher measured reranking p50 (2,869.78 vs 2,598.60 ms). The selection trades latency for gold-page coverage; it proves neither universal BM25 superiority nor answer accuracy. Full metrics stay in the benchmark report.

## Candidate depth and latency

Only candidate depth changed; all runs used the same 41 answerable development questions, BGE reranker and final k=5.

| Candidates / artifact | Hit@5 | Recall@5 | MRR | Rerank-only p50 ms |
| ---: | ---: | ---: | ---: | ---: |
| [10][top10] | 0.9512 | 0.9049 | 0.8951 | 1,290.04 |
| [20][bm25-rerank] | 0.9512 | 0.9293 | 0.8951 | 2,598.60 |
| [30][top30] | 0.9512 | 0.9293 | 0.8829 | 4,056.33 |

Depth 20 improved recall over 10 and matched 30 with better MRR/latency. Timings exclude retrieval/model loading. Limiting cross-encoder work to 20 candidates controls cost; increasing depth to 30 did not improve this sample.

## Gate, filtering and experimental scope

The [score analysis][threshold] and [distribution plot](../eval/results/rerank_score_analysis_20261015T131329110974Z.svg) cover **41 answerable and seven unanswerable development questions**. Threshold 0.60 caught 4/7 unanswerable questions while wrongly refusing 1/41 answerable questions; three unanswerable questions still passed. The score is not calibrated answer confidence. The gate is integrated with generation. The later [one-time hold-out][holdout] used 12 questions, including ten answerable; it was not used for parameter selection and must not be reused for tuning.

Optional regulator/issue-date filters are AND conditions applied inside Qdrant, before ranking; hybrid applies the same filter to both branches. Date bounds include the full UTC day. Recorded backfill verification stored `issue_date_iso` alongside original strings and verified payload indexes without regenerating embeddings. Filters were checked functionally, not as a quality-improving ablation. Regulatory `status` remains unverified, so active-by-default filtering is blocked. The API's public-only filter is an authorization boundary, not a new ranking algorithm.

**Metadata-enriched reranking: experimented with locally, not deployed.** Console results improved aggregate metrics but regressed five questions. G003 illustrates why: page 2 was in BM25's 20 candidates but lost during reranking; enrichment recovered it. [Results, ranks and evidence limits](benchmarks/FINAL_BENCHMARK_REPORT.md#metadata-enrichment-experiment-not-deployed) distinguish this unsaved ablation from production, which still scores question + chunk text. Existing section headings are not the extra metadata experiment. **Page diversity remains unimplemented.**

Follow-up query rewriting was absent from Phase 3. Step 5.5 later added bounded session-based rewriting, without a retrieval-quality ablation. [Partial RAGAS findings](benchmarks/FINAL_BENCHMARK_REPORT.md#partial-ragas-findings) assess small development subsets; they do not replace the retrieval experiment or establish full-set answer accuracy.

## Failure modes

- **Missing page or clause:** G003 lost gold page 2 during reranking in the later local diagnosis; G008's manual benchmark lacked pages 5/7. A correct document or page is not enough if the retained chunk lacks the clause.
- **Similar regulations:** Near-identical language from a different bank category or amendment can outrank the intended source.
- **Fragmentation and extraction:** Related clauses can span pages; PDFs with no extracted text cannot supply evidence. Section-heading heuristics can mislabel fragments.
- **Crowding and ties:** Multiple chunks from one page can occupy the top five; RRF can promote common distractors and has tie-order variation.
- **Generation is a separate failure stage:** G007/G009 omitted details and G008 ended mid-sentence. Coverage and generation omissions may both matter; saved observations do not isolate every cause. A high score or citation precision is not proof of completeness.

<details>
<summary>Retained historical experiments and case-level evidence</summary>

**Phase 2 dense failure analysis:** these are earlier observations, not a fresh corpus audit or claims about every later configuration.

| ID | Recorded observation and limitation |
| --- | --- |
| G003 | Complete dense miss despite indexed Payments Banks evidence; related bank categories ranked higher. |
| G035 | Digital Accessibility PDF yielded zero extracted text/indexed chunks: an upstream coverage gap. |
| G042 | Indexed sanctions-table mapping for `5080(E)` missed by dense; exact-identifier ranking was a likely cause. |
| G007 | Page 8 retrieved, indexed page 9 missing. |
| G008 | Dense returned pages 2/7/8, missed 5/6; duplicate page-8 chunks and page 14 crowded the five slots. |
| G012 | Extension page retrieved, original circular page missed; the extension itself stated both dates. |
| G017 | AIFI page 2 retrieved, contextual page 1 missing amid similar valuation amendments. |
| G028 | Novation applicability/issue-date evidence split across pages; page 2 missing. |
| G036 | Bank delisting-route sentence crossed pages; required page 2 missing. |
| G021 | Gold page at rank 2 after the same circular's introduction: a ranking weakness, not a full miss. |
| G050 | Gold page matched but the separate chunk containing the 90-day rule was absent. |
| G052 | Gold page matched but returned chunks omitted separately indexed LRS/overseas-labelling text. |

**BM25 vs dense:** comparing recall first, then reciprocal rank, gave seven BM25 wins, five dense wins and 29 ties. BM25 recovered G042 and completed gold-page coverage for G007/G028; dense favored G009 recall and G004/G034 ranking. Neither method solved the recorded G003/G035 misses.

Type-level scores below are Hit@5 / Recall@5 / MRR, on the same development split:

| Type (n) | Dense | BM25 | Hybrid RRF |
| --- | --- | --- | --- |
| Factoid (19) | 0.8947 / 0.8421 / 0.8333 | 0.8947 / 0.8684 / 0.7719 | 0.8947 / 0.8421 / 0.7237 |
| Multi-section (10) | 1 / 0.9100 / 0.9500 | 1 / 0.9100 / 1 | 1 / 0.9100 / 0.9500 |
| Table lookup (7) | 0.8571 / 0.8571 / 0.8571 | 1 / 1 / 1 | 1 / 1 / 0.9286 |
| Date/version (5) | 1 / 0.8000 / 1 | 1 / 0.9000 / 1 | 1 / 0.9000 / 1 |

**Fusion details:** manual RRF smoothing 60 matched native `Rrf(k=61)` under the verified rank conventions. Initial hybrid prefetched five per branch. Manual/native G042 point order matched, but G021 ties varied. Against BM25, hybrid had one metric win (G004), six regressions (G008/G021/G034/G036/G042/G050), and 34 ties. For G036, a BM25 rank-4 gold page fell outside the fused five. In G034, a small sparse-rank difference across separate collections remained unexplained despite matching vectors/IDF.

**Reranking controls:** the first five of 20 candidates scored 0.9512 / 0.9049 / 0.8882 for BM25 and 0.9512 / 0.9049 / 0.8537 for hybrid. These differ from separate top-five retrieval runs. After reranking, BM25/hybrid had identical ordered document/page references on 25/41 questions and differed on 16; rank-one references agreed on all 41, with no per-question Hit/Recall/RR differences. Reports omit Qdrant point IDs, so identical document/pages do not prove identical chunks.

Early ten-query manual dense inspection also showed introductions/related circulars outranking answer clauses and an unsupported repo-rate query returning irrelevant neighbors. A pre-Phase-4 repo-rate probe later passed the 0.60 gate with score 0.7991 for an ATM circular: a concrete example of why a score threshold alone cannot establish answerability.

</details>

## Evidence and interview takeaway

The [final benchmark](benchmarks/FINAL_BENCHMARK_REPORT.md) owns the consolidated quality/latency tables; [engineering decisions](DECISIONS.md) retain the rationale. The [original manual answer review](benchmarks/2026-10-27-golden-set.md) keeps the observed response defects. Explain the trade-off as **higher measured page recall at a substantial reranking cost**, with unresolved multi-page coverage and no claim that retrieval scores prove correct answers.

[dense]: ../eval/results/retrieval_20261012T191839173509Z_af6558ebe4a5bce1f853b19340b8acdfe77ea87db78f6dd033f4fb3c73a9c58f.json
[bm25]: ../eval/results/retrieval_bm25_20261014T095834356877Z_5047cbf0cbf3f1ac63547757db03dfa232d1159d1d06350c5abca79b22162fe8.json
[hybrid]: ../eval/results/retrieval_hybrid_20261014T112307814856Z_d0bfed4a543cfd45b7657595060f953681e1ee0360400368f28007db81c1c702.json
[bm25-rerank]: ../eval/results/retrieval_bm25_rerank_20261015T122408293824Z_4d93aa37e83e77ea2458a5c9e1a6760db1bea494d877701f961c1eac06d4cb9c.json
[hybrid-rerank]: ../eval/results/retrieval_hybrid_rerank_20261015T123138677133Z_49abbae93ee83b902bf70141a714133d964b26eab106e1e8fcf75314d36e286e.json
[top10]: ../eval/results/retrieval_bm25_rerank_20261015T180050898462Z_3367c2ac2731ca5fac0c3682446a229998352d49915bdf23281a47723a20a3eb.json
[top30]: ../eval/results/retrieval_bm25_rerank_20261015T180615157410Z_18c49fb4ca41841eb7ea583de346eef9649e4a0e15408469b4397c2d23a7aaf5.json
[holdout]: ../eval/results/retrieval_bm25_rerank_holdout_20261015T182257754654Z_15f93293e5ee0669801fa0d69164bf1c69a9379717dfd17791ba89f2c65cef46.json
[threshold]: ../eval/results/rerank_score_analysis_20261015T131329110974Z.json
