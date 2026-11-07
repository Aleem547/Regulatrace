# Regulatrace Golden-Set Benchmark - 2026-10-27

## 1. Objective and methodology

Assess end-to-end answer quality, citation handling, abstention consistency, and latency in Regulatrace, a citation-grounded RAG system for RBI and SEBI regulatory PDFs.

Twenty selected development questions were manually submitted sequentially through FastAPI `POST /chat`. Each request started an independent conversation, without reusing a session. The pipeline used BM25 retrieval, BGE reranking, and Groq LLM generation. Responses were manually reviewed against the golden reference answers and source pages; **quality classifications are human assessments, not automated evaluation scores**. No new inference or database queries were performed to prepare this report.

The classifications describe the responses originally observed. A full pass means the response passed the manual review; partial passes include incomplete content or citation/response-contract defects despite otherwise correct content. They are not a separately measured factual-accuracy rate. Later fixes do not retroactively change these labels.

Reproduction inputs are the ordered IDs below and the exact golden records in Section 2. A repeat run should submit those question strings as independent authenticated `/chat` requests, retain answers, supplied sources, citation-validation results and `abstained`, then compare them with the references and source PDFs. Record the deployed revision, configuration hash, prompt/model versions and completion metadata for any repeat run; those per-request snapshots were not supplied for this benchmark. See the existing [RAG pipeline](../../src/regulatrace/rag.py) and [README](../../README.md).

## 2. Golden-set composition and holdout exclusion

Source dataset: [`data/eval/golden.jsonl`](../../data/eval/golden.jsonl), containing 60 manually verified questions. The frozen split contains 12 IDs in [`holdout_ids.txt`](../../data/eval/holdout_ids.txt), leaving 48 development questions. An ID-only membership check confirmed **zero overlap** between the 20 benchmark IDs and the holdout. This benchmark did not evaluate holdout questions.

| Selected question type | Count |
| --- | ---: |
| `factoid` | 8 |
| `multi-section` | 4 |
| `table_lookup` | 3 |
| `date_version` | 2 |
| `unanswerable` | 3 |
| **Total** | **20** |

The sample comprises 17 answerable and three unanswerable questions. It is a selected development subset, not a random sample or an independent generalization test. All selected records have `verified=true`.

<details>
<summary>Exact selected golden questions, reference answers and document/page sources</summary>

Questions and reference answers below are copied verbatim from the golden file. Source IDs and page numbers are the golden expectations, not assertions that the system retrieved every listed page. `gold_sources=[]` denotes no expected supporting source in the indexed corpus.

### G002 - `factoid`

**Question:** Under RBI/2026-27/279, who may approve internal guidelines for outward forex remittances based on online or physical Form A2 submission?

**Reference answer:** The Board, or a Board Committee or Management Committee to which the Board has delegated powers.

**Gold sources:** `ad30a12a41afa632d97786d24974653ca1c7ee026da7f3e9b2dbbe88835bbbcc` p.1

### G003 - `factoid`

**Question:** Under the October 1, 2026 Payments Banks shareholding amendment, which entities qualify as a qualifying person and what promoter-group restriction applies?

**Reference answer:** A SEBI-registered mutual fund, a PFRDA-registered pension fund, or an IRDAI-registered insurance company qualifies only if it does not belong to the promoter group of the investee banking company.

**Gold sources:** `052f30d9733ff5ef33aeca2d4bc74d83a8292dc829eee8e2f1c27a045917b8a9` p.2

### G004 - `factoid`

**Question:** Under SEBI's June 15, 2026 ETF circular, what base price is used if there is no trade in the ETF on T-1 day?

**Reference answer:** The latest available closing NAV of the ETF.

**Gold sources:** `717dcc7a29ae7bc33f6e649eec0001f0b7cca5eecb925569a42a2dcb29402e86` p.2

### G005 - `factoid`

**Question:** Under Project Jagrook, what must stock brokers display on their trading apps from November 1, 2026, and where must the investor awareness messages appear?

**Reference answer:** Stock brokers must display the specified investor awareness messages and risk disclosures on alternate days on their trading apps, if any. The investor awareness messages must appear on the app landing page.

**Gold sources:** `1d0f3b013a6c8a309e5fc206a87d8be1025e2b9f8663d0700659b2e68856df6c` p.2

### G006 - `factoid`

**Question:** Which institution is designated as the nodal agency to issue guarantee cover under the Credit Enhancement Guarantee Scheme for Scheduled Castes (CEGSSC)?

**Reference answer:** IFCI Ltd. is the nodal agency that issues guarantee cover in favour of Member Lending Institutions financing SC entrepreneurs.

**Gold sources:** `e273a00d702fa73580dde9d561689a85b3429b0ea55a9f3e4fea218644087a77` p.5

### G016 - `factoid`

**Question:** Under RBI/2026-27/273, to whom may the Board delegate monitoring of progress in implementing ATM cassette swaps?

**Reference answer:** The Board may delegate monitoring of cassette-swap implementation progress to a Committee of Executives of the bank.

**Gold sources:** `c6434c3c7f07267e9ed6653d607869d0957625786eb75d20b91b23ef5e025248` p.1

### G017 - `factoid`

**Question:** Under RBI/2026-27/268 for All India Financial Institutions, how must unquoted InvIT units be valued when the InvIT fails to compute and disclose NAV as required, and does the same treatment apply to infrequently traded units?

**Reference answer:** The units must be valued at Rs 1 for the purposes of the Directions if the InvIT fails to compute and disclose NAV in the manner and frequency specified under SEBI (Infrastructure Investment Trusts) Regulations, 2014. The same treatment applies to InvIT units classified as infrequently traded under those Regulations.

**Gold sources:** `b773a9e6ca5cd90f723e78bca396d7028c1f233b5e11c6175ca71d582e242904` p.1; `b773a9e6ca5cd90f723e78bca396d7028c1f233b5e11c6175ca71d582e242904` p.2

### G020 - `factoid`

**Question:** Under SEBI's July 21, 2026 SIF certification circular, may a holder of valid NISM Series-V-D certification distribute both Mutual Fund and SIF products without a separate Series V-A certification?

**Reference answer:** Yes. A holder of valid NISM Series-V-D - Mutual Fund - Specialized Investment Fund Distributors Certification is eligible to distribute both Mutual Fund and SIF products without separately holding NISM Series V-A - Mutual Fund Distributors Certification.

**Gold sources:** `04014f08329e2558d01f345a2f2f3bbb77f8161a13c8f05f920b053467443017` p.2

### G007 - `multi-section`

**Question:** Under RBI's Note Sorting Machines Master Direction, how often must banks test their machines, what testing certificate must they retain, and what minimum test-deck size and tolerances apply?

**Reference answer:** Banks must test accuracy and consistency quarterly, recalibrate if required, and retain a certificate signed by bank officials. Each quarter, the test deck must contain at least 2,000 pieces of soiled notes, including mutilated notes, FICNs and reissuables. The acceptable tolerance for fitness checks is 5%; tolerance for FICNs is zero. Discrepancies require advising the machine supplier to recalibrate.

**Gold sources:** `564170f3526bc009728bc6bddbb096e3e878950af2c009d15e1326fb42498638` p.8; `564170f3526bc009728bc6bddbb096e3e878950af2c009d15e1326fb42498638` p.9

### G008 - `multi-section`

**Question:** For a low-value transmission claim involving dematerialised securities with no nomination, what is the QTP threshold, which relatives are eligible, and what documents must the claimant submit?

**Reference answer:** The dematerialised-securities QTP threshold is Rs 30,000 per beneficial owner. QTP concerns transmission following the death of the sole holder or all joint holders, without disputed or competing claims. QTP is permitted only for parents, spouse, children and parents-in-law of the deceased holder. The claimant must submit the Annexure-2 transmission request form-cum-undertaking on plain paper, a document establishing the relationship, the latest client master list of the claimant's demat account, a verifiable death certificate, and the original security certificate or copy of statement of account, as applicable.

**Gold sources:** `9e9c336462c8431dc5744af201af0d9d5940581f762bee2041433d408f04c411` p.2; `9e9c336462c8431dc5744af201af0d9d5940581f762bee2041433d408f04c411` p.5; `9e9c336462c8431dc5744af201af0d9d5940581f762bee2041433d408f04c411` p.6; `9e9c336462c8431dc5744af201af0d9d5940581f762bee2041433d408f04c411` p.7; `9e9c336462c8431dc5744af201af0d9d5940581f762bee2041433d408f04c411` p.8

### G009 - `multi-section`

**Question:** Under RBI's SC/ST credit facilities Master Circular, what productive-activity and additional housing loan benefits does the DRI Scheme offer SC/ST borrowers, and what minimum share of total DRI advances must go to them?

**Reference answer:** The DRI Scheme provides finance to eligible borrowers from the weaker sections of the community up to Rs 15,000 for productive and gainful activities at 4% interest per annum. Eligible SC/ST borrowers must receive at least two-fifths (40%) of total DRI advances. The normal DRI landholding ceilings of 1 acre of irrigated land and 2.5 acres of unirrigated land do not apply to SC/ST borrowers. SC/ST members satisfying the scheme's income criteria may also obtain a housing loan up to Rs 20,000 per beneficiary, over and above the individual Rs 15,000 loan.

**Gold sources:** `e273a00d702fa73580dde9d561689a85b3429b0ea55a9f3e4fea218644087a77` p.4; `e273a00d702fa73580dde9d561689a85b3429b0ea55a9f3e4fea218644087a77` p.5

### G022 - `multi-section`

**Question:** Under GARUDA, how do launch timings differ for first and subsequent Regular AIF schemes versus AI-only funds and LVFs, and how are working days defined?

**Reference answer:** Regular AIF schemes must file their PPM with SEBI through a SEBI-registered Merchant Banker. They may launch after 10 working days from filing the associated application with SEBI, unless SEBI advises otherwise. A first Regular scheme may launch only after both the 10-working-day period and grant of SEBI registration. Subsequent AI-only fund and LVF schemes may launch immediately upon filing their PPM with SEBI. First AI-only fund and LVF schemes require both PPM filing with SEBI and grant of SEBI registration before launch. Launch means circulating the PPM to investors to solicit funds. Working days exclude Saturdays, Sundays and public holidays when the concerned SEBI office is closed, as published on SEBI's website.

**Gold sources:** `00e47c21c69b7be6711c8c9c53ab32bdd5ee59d29df7dc3aeceae8c94c4c884f` p.1; `00e47c21c69b7be6711c8c9c53ab32bdd5ee59d29df7dc3aeceae8c94c4c884f` p.3; `00e47c21c69b7be6711c8c9c53ab32bdd5ee59d29df7dc3aeceae8c94c4c884f` p.5

### G010 - `table_lookup`

**Question:** In Table 2 (Soiling Levels) of RBI's Note Sorting Machines Master Direction, what maximum density difference and minimum reflectance apply to Rs 100 and Rs 500 notes?

**Reference answer:** For Rs 100 notes: maximum density difference 0.05 and minimum reflectance 90%. For Rs 500 notes: maximum density difference 0.04 and minimum reflectance 93%.

**Gold sources:** `564170f3526bc009728bc6bddbb096e3e878950af2c009d15e1326fb42498638` p.5

### G011 - `table_lookup`

**Question:** In SEBI's ITRI weightage table for MIIs, what weightages are assigned to Availability, Security and Scalability, and what is the total weightage?

**Reference answer:** Availability: 20; Security: 20; Scalability: 5. The total weightage is 100.

**Gold sources:** `ff4bac515b58a6f97510ab0bdac59e11e792ec0503dbf00ab6e41f6bb3b176ea` p.2

### G026 - `table_lookup`

**Question:** In the HoldCo/SPV NDCF computation table in SEBI's August 14, 2026 InvIT circular, is the major road-maintenance payment line added or deducted, what funding restriction applies, and which note governs the treatment?

**Reference answer:** The line is an addition (+): payments towards major maintenance expense for road projects are added back only to the extent funded by external borrowing, subject to Note 12. It is not an unconditional add-back of all maintenance expenditure.

**Gold sources:** `b7d9dcc9438eba7ace1b558abac2fd5bc009e649f0da837702527e01a9ede45e` p.2

### G012 - `date_version`

**Question:** What implementation date did SEBI originally specify for its June 15, 2026 ETF norms, and what date replaced it under the August 28, 2026 extension circular?

**Reference answer:** The original implementation date was September 1, 2026. The August 28 extension circular changed it to September 7, 2026; all other provisions of the June 15 circular remained unchanged.

**Gold sources:** `717dcc7a29ae7bc33f6e649eec0001f0b7cca5eecb925569a42a2dcb29402e86` p.5; `d59e9a89f4f552910e560c29f042c0921b7aa69677d47323abd22713c0dbfa27` p.1

### G013 - `date_version`

**Question:** By what date must banks phase out Note Sorting Machine models discontinued from production under RBI's 2026 Master Direction?

**Reference answer:** On or before June 30, 2027.

**Gold sources:** `564170f3526bc009728bc6bddbb096e3e878950af2c009d15e1326fb42498638` p.8

### G015 - `unanswerable`

**Question:** What was the closing NAV of SBI Nifty 50 ETF on September 30, 2026?

**Reference answer:** not found in the provided documents

**Gold sources:** `[]`

### G029 - `unanswerable`

**Question:** What is the examination fee in rupees for the NISM Series-V-D certification required for SIF distribution?

**Reference answer:** not found in the provided documents

**Gold sources:** `[]`

### G030 - `unanswerable`

**Question:** What annual interest rate is charged on Exim Bank's Rs 4,850 crore Government of India-supported line of credit to Maldives?

**Reference answer:** not found in the provided documents

**Gold sources:** `[]`

</details>

## 3. Per-question results

Rows follow the supplied benchmark order. Observations summarize the manual review; they are not reconstructed model outputs.

| ID | Type | Manual classification | Main observation |
| --- | --- | --- | --- |
| G002 | factoid | Full pass | Approval-authority answer passed manual review against the reference. |
| G003 | factoid | Failure | Answerable question received `not found in the provided documents`. Expected document was retrieved, but gold page 2 was absent from the returned top five; `abstained=false`. |
| G004 | factoid | Full pass | ETF base-price answer passed manual review against the reference. |
| G005 | factoid | Full pass | Project Jagrook display requirements passed manual review. |
| G006 | factoid | Full pass | Nodal-agency answer passed manual review. |
| G016 | factoid | Full pass | ATM cassette-swap delegation answer passed manual review. |
| G017 | factoid | Partial pass | Answer was correct; citation parsing failed on extra whitespace inside brackets. |
| G020 | factoid | Full pass | SIF/Mutual Fund certification answer passed manual review. |
| G007 | multi-section | Partial pass | Main details were correct; recalibration/discrepancy details were missing. |
| G008 | multi-section | Partial pass | Correct QTP threshold and eligible relatives, but answer ended mid-sentence and checklist was incomplete. Expected pages 5 and 7 were missing from final retrieval. |
| G009 | multi-section | Partial pass | Main loan amounts and allocation percentage were correct; landholding exemption was omitted. |
| G022 | multi-section | Full pass | GARUDA launch-timing answer passed manual review. |
| G010 | table_lookup | Full pass | Soiling-table values passed manual review. |
| G011 | table_lookup | Full pass | ITRI weightages passed manual review. |
| G026 | table_lookup | Full pass | Conditional NDCF add-back answer passed manual review. |
| G012 | date_version | Partial pass | Both implementation dates were correct; citations were not recognized because of internal bracket whitespace. |
| G013 | date_version | Full pass | Note Sorting Machine phase-out date passed manual review. |
| G015 | unanswerable | Full pass | Correct not-found response passed manual review. |
| G029 | unanswerable | Partial pass | Correct text abstention, but structured response reported `abstained=false`. |
| G030 | unanswerable | Partial pass | Correct text abstention, but structured response reported `abstained=false`. |

## 4. Overall quality results

| Manual classification | Questions | Share |
| --- | ---: | ---: |
| Full pass | 12/20 | 60% |
| Partial pass | 7/20 | 35% |
| Failure | 1/20 | 5% |

These are strict, manually assigned end-to-end classifications combining content, completeness, citation parsing and response-contract behavior. They are not automated retrieval metrics or RAGAS scores. Citation precision checks reference membership in supplied sources; it does **not** establish factual accuracy, claim support or completeness.

## 5. Stage latency and estimated cost

Source: the supplied output of [`scripts/query_log_report.py`](../../scripts/query_log_report.py), run after the benchmark. It summarized **24 completed query-log records: 20 benchmark requests plus four previously logged requests**. The figures below are not isolated measurements of the 20 golden-set requests.

| Stage | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 8,233.57 | 16,663.63 |
| Rewrite | 0.00 | 0.00 |
| Retrieval | 81.46 | 99.41 |
| Reranking | 5,381.19 | 13,098.42 |
| LLM generation | 2,139.55 | 3,329.29 |

**Dominant measured stage: reranking.** It had the highest stage p50 and p95 in this mixed sample. Stage percentiles describe different distributions and must not be summed to reconstruct total latency. The report uses inclusive percentile interpolation and counts skipped stages as zero; total excludes the query-log write and network delivery, as documented in the [instrumentation notes](../../README.md#stage-latency-and-estimated-cost-phase-6-step-62).

| Cost measurement | Value | Coverage |
| --- | ---: | --- |
| Estimated average LLM API cost per query | $0.00064260 | 22 of 24 records with cost data |

This average is conditional on cost availability, not an average across all 24 records or the isolated 20-question benchmark. Costs use recorded provider usage and configured input/output token prices; they are estimates, not invoices. No per-question costs, token counts or latencies were supplied.

The 24-record sample is too small to treat p95 as a reliable production SLA. Requests were sequential and do not establish concurrent-user performance.

## 6. Key findings and root-cause hypotheses

| Issue | Confirmed finding | Hypothesis or unresolved question |
| --- | --- | --- |
| G003 retrieval/answerability failure | Expected document appeared, but gold page 2 was absent from the final top-five chunks; an answerable question received not-found text. | Missing gold-page coverage may explain the refusal. The supplied observations do not locate the loss at BM25 retrieval, reranking, or evidence selection, or establish whether other chunks could answer it. |
| G003, G029, G030 abstention flags | Not-found text conflicted with `abstained=false`. | The independently assembled answer/flag paths motivated a normalization change, but the manual API retest still returned false. The remaining cause is unresolved and deferred. |
| G012, G017 citation parsing | Extra whitespace inside citation brackets prevented recognition despite correct answer content. | A regex whitespace correction has since been manually verified; it is not yet committed. Historical classifications remain partial passes. |
| G008 truncation and missing evidence | Answer stopped mid-sentence, checklist was incomplete, and expected pages 5 and 7 were absent from final retrieval. | Completion-budget exhaustion is a possible cause of truncation, not a confirmed one. Finish reason and usage data are needed. Missing pages may contribute to the incomplete checklist but do not explain every omitted item by themselves. |
| Multi-section completeness | G007 omitted recalibration/discrepancy details; G009 omitted the landholding exemption; G008 omitted checklist details. | Retrieval coverage and generation omissions may both contribute. Claim-by-claim inspection of actually supplied chunks is needed to separate them. |
| Retrieval noise and gold-page coverage | Retrieving the correct document did not ensure the correct page for G003; G008 also lacked specified gold pages. | Distracting candidates may compete with relevant evidence, but no noise rate or candidate-by-candidate causal analysis was supplied for this benchmark. |
| Reranking latency | Reranking dominated the measured stage latency in the 24-record report. | Hardware, cold initialization, candidate lengths and model execution could affect latency; this aggregate does not isolate their contributions. |

Current development status is separate from benchmark results: citation whitespace handling is manually verified but uncommitted; the attempted abstention-classification correction has not passed the manual API retest, and that issue is deferred. Neither change upgrades an originally observed result in this report.

## 7. Known limitations

- Only 20 selected development questions were reviewed. Prior use of development questions and manual selection limit generalization claims; no holdout result is reported here.
- Judgments were manual. No independent second-reviewer agreement or automated factual-correctness measurement was supplied.
- Full raw responses, point-ID candidate lists, trace exports, deployed revision and per-request model/prompt/configuration snapshots are not included in the supplied benchmark evidence. Exact execution replay and some root-cause attribution are therefore limited.
- Document/page matches do not prove that a particular retrieved chunk contains the answer or supports every generated claim.
- Performance aggregates mix benchmark and earlier requests. There is no isolated benchmark timing distribution, concurrency measurement or production SLA evidence.
- Two query-log records lacked cost data; the cause was not supplied. No per-question metrics, completion finish reasons, token counts, RAGAS results or new evaluations are inferred.

## 8. Recommended next actions

1. Verify the whitespace correction on the known citation examples and preserve the original benchmark labels; confirm the tested changes are deployed before any new measurement.
2. Resume the deferred abstention investigation with G029/G030 and G003, checking the running API's final result assembly and deployed revision. Require agreement between not-found text and the structured flag before declaring the fix verified.
3. Inspect retained candidates and supplied chunks for G003 and G008 to identify where required pages disappear. Separate evidence-coverage failures from answer-generation omissions before changing retrieval settings.
4. Inspect G008 completion metadata and full response to establish why generation stopped. Review G007/G009 against the supplied evidence for completeness; do not assume a larger token cap alone resolves these cases.
5. For the next measurement, retain request-to-question linkage and isolate the benchmark cohort before reporting latency/cost. Profile reranking first, with controlled warm/cold conditions; assess any optimization's quality impact on development data.
6. Collect larger samples and a separate controlled concurrency test before making tail-latency claims. Keep the frozen holdout out of debugging and configuration selection.
