"""Compare dense and BM25-reranked RAG on answerable development questions only."""

import argparse
import asyncio
import inspect
import json
import math
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import mean

from groq import AsyncGroq, RateLimitError

from regulatrace import rag
from regulatrace.bm25 import retrieve_bm25
from regulatrace.config import get_settings
from regulatrace.llm import get_llm
from regulatrace.privacy import redact_for_tracing
from regulatrace.reranker import _get_reranker, decide_abstention, rerank
from regulatrace.retrieval import retrieve

from . import run_retrieval
from .compare_prompts import _load_prompts

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GROQ_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = GROQ_MODEL
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


async def _retry_429(operation, *args, **kwargs):
    """Retry Groq answer-generation rate limits for at most three attempts."""
    for attempt in range(3):
        try:
            result = operation(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        except Exception as error:
            cause = error
            while not isinstance(cause, RateLimitError) and cause.__cause__ is not None:
                cause = cause.__cause__
            if not isinstance(cause, RateLimitError) or attempt == 2:
                raise
            retry_after = cause.response.headers.get("retry-after")
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                try:
                    delay = (
                        parsedate_to_datetime(retry_after).timestamp()
                        - datetime.now(timezone.utc).timestamp()
                    )
                except (TypeError, ValueError, OverflowError):
                    delay = 60.0
            if not math.isfinite(delay) or delay < 0:
                delay = 60.0
            delay += 1  # Small safety margin beyond the provider delay or fallback window.
            print(f"Groq HTTP 429; waiting {delay:g}s before attempt {attempt + 2}/3.", flush=True)
            await asyncio.sleep(delay)


def _load_metrics():
    """Build the official collections metrics once, without HTTP interception."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    import instructor
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import InstructorLLM
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    api_key = get_settings().groq_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise ValueError("REGULATRACE_GROQ_API_KEY is required for the Groq RAGAS judge")
    # Let the SDK handle transient failures, including 429s, with two retries.
    client = AsyncGroq(api_key=api_key.get_secret_value(), timeout=60.0, max_retries=2)
    judge = InstructorLLM(
        client=instructor.from_groq(client, mode=instructor.Mode.JSON_SCHEMA),
        model=JUDGE_MODEL,
        provider="groq",
        temperature=0,
        top_p=1,
        max_tokens=2048,
    )
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL, use_api=False)
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=3),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }, client


async def _generate_answer(row, ranked, prompts, llm):
    evidence = rag._budget_evidence(row["question"], ranked, tuple(prompts.values()))
    row["sources"] = evidence["sources"]
    # Recover whole selected chunks in order, including repeated document/page pairs.
    remaining = iter(ranked)
    for source in evidence["sources"]:
        chunk = next(
            chunk
            for chunk in remaining
            if (chunk.get("metadata") or {}).get("doc_id") == source["doc_id"]
            and chunk.get("page") == source["page"]
            and (chunk.get("metadata") or {}).get("chunk_index") == source["chunk_index"]
        )
        row["contexts"].append(chunk["text"])
    if not row["contexts"]:
        row.update(status="evidence_refusal", answer=rag._NOT_FOUND)
        return
    prompt = rag._render_prompt(prompts["v1"], row["question"], evidence)
    row["answer"] = await _retry_429(
        llm.generate, prompt, temperature=0, max_completion_tokens=rag._MAX_COMPLETION_TOKENS
    )
    row["status"] = "prompt_refusal" if row["answer"].strip() == rag._NOT_FOUND else "generated"


async def _score_metrics(metrics, row):
    for name, metric in metrics.items():
        progress = f"{row['id']} {row['configuration']} {name}"
        inputs = {"user_input": row["question"]}
        if name in ("faithfulness", "answer_relevancy"):
            inputs["response"] = row["answer"]
        if name != "answer_relevancy":
            inputs["retrieved_contexts"] = row["contexts"]
        if name in ("context_precision", "context_recall"):
            inputs["reference"] = row["reference"]
        try:
            print(f"Starting {progress}", flush=True)
            result = await metric.ascore(**inputs)
            value = result.value
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError("Metric returned no finite numeric score")
            row["scores"][name] = float(value)
            if getattr(result, "reason", None):
                row["judge_reasons"][name] = result.reason
        except Exception as error:
            row["errors"][name] = redact_for_tracing(f"{type(error).__name__}: {error}")
            print(f"Failed {progress}", flush=True)
        else:
            print(f"Succeeded {progress}", flush=True)


async def evaluate(limit: int | None = None) -> dict:
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 41):
        raise ValueError("limit must be between 1 and 41")
    golden = run_retrieval.load_golden()
    holdout = run_retrieval.load_holdout({record["id"] for record in golden})
    development = [record for record in golden if record["id"] not in holdout]
    answerable = [record for record in development if record["answerable"] is True]
    if (
        len(development) != 48
        or len(answerable) != 41
        or any(record["verified"] is not True for record in development)
    ):
        raise ValueError("Expected 48 verified development records, including 41 answerable")
    selected = answerable[:limit]
    settings = get_settings()
    if (
        settings.llm_provider,
        settings.llm_model,
        settings.embedding_model,
        settings.qdrant_collection,
        settings.bm25_collection,
    ) != ("groq", GROQ_MODEL, EMBEDDING_MODEL, "regulatrace_chunks", "regulatrace_bm25"):
        raise ValueError("RAGAS comparison requires the frozen retrieval and Groq settings")
    prompts, _ = _load_prompts()
    llm, reranker = get_llm(settings), _get_reranker()
    metrics, judge_client = _load_metrics()
    rows = []
    async with judge_client:
        for record in selected:
            for configuration in ("baseline", "best"):
                row = {
                    "id": record["id"],
                    "configuration": configuration,
                    "question": record["question"],
                    "reference": record["reference_answer"],
                    "gold_sources": record["gold_sources"],
                    "answer": None,
                    "status": "error",
                    "contexts": [],
                    "sources": [],
                    "scores": dict.fromkeys(METRICS),
                    "errors": {},
                    "judge_reasons": {},
                }
                try:
                    question = record["question"]
                    if configuration == "baseline":
                        ranked = retrieve(question, k=5)  # Never gate dense similarity scores.
                    else:
                        ranked = rerank(
                            question, retrieve_bm25(question, k=20), k=5, model=reranker
                        )
                        decision = decide_abstention(ranked)
                        if decision["abstain"]:
                            row.update(status="gate_refusal", answer=decision["message"])
                    row["retrieved_contexts"] = [chunk.get("text") for chunk in ranked]
                    if row["status"] != "gate_refusal":
                        await _generate_answer(row, ranked, prompts, llm)
                    if row["status"] == "generated":
                        await _score_metrics(metrics, row)
                except Exception as error:
                    row.update(
                        status="error",
                        errors={"pipeline": redact_for_tracing(f"{type(error).__name__}: {error}")},
                    )
                rows.append(row)
            print(f"Completed {len(rows) // 2}/{len(selected)} development questions", flush=True)
    summary = {}
    for configuration in ("baseline", "best"):
        group = [row for row in rows if row["configuration"] == configuration]
        summary[configuration] = {
            "questions": len(group),
            "refusals": sum(row["status"].endswith("refusal") for row in group),
            "errors": sum(bool(row["errors"]) for row in group),
        }
        for name in METRICS:
            scores = [row["scores"][name] for row in group if row["scores"][name] is not None]
            summary[configuration][name] = {
                "mean": mean(scores) if scores else None,
                "scored": len(scores),
                "unavailable": len(group) - len(scores),
            }
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "configuration": {
            "ragas_version": "0.4.3",
            "prompt": "v1",
            "generation_model": settings.llm_model,
            "judge_model": JUDGE_MODEL,
            "judge_provider": "groq",
            "temperature": 0,
            "embeddings": EMBEDDING_MODEL,
        },
        "summary": summary,
        "per_question": rows,
        "review_note": (
            "Review ten scored judgments using the saved question, answer, reference and context. "
            "Record a human verdict and evidence for any genuine judge error found. "
            "High faithfulness means context support, not a complete or correct answer: the "
            "context can be wrong or incomplete, and the judge can err. "
            "Reasons may be unavailable. "
            "Means exclude missing scores; compare coverage counts as well as averages."
        ),
    }


def save_results(report) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"ragas_{report['timestamp_utc']}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="First N answerable development questions (1-41)")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 41:
        parser.error("--limit must be between 1 and 41")
    report = asyncio.run(evaluate(args.limit))
    path = save_results(report)
    for configuration, summary in report["summary"].items():
        print(
            f"{configuration}: {summary['refusals']} refusals; {summary['errors']} rows with errors"
        )
        for name in METRICS:
            result = summary[name]
            print(
                f"  {name}: {result['mean']} ({result['scored']} scored, "
                f"{result['unavailable']} unavailable)"
            )
    print(f"Saved: {path}; review ten judgments against their saved inputs.")


if __name__ == "__main__":
    main()
