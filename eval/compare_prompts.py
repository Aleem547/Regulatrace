"""Compare both answer prompts on the frozen development split only."""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TextIO

from groq import RateLimitError
from langchain_core.prompts import ChatPromptTemplate

from regulatrace import rag
from regulatrace.bm25 import retrieve_bm25
from regulatrace.citations import parse_citations, validate_citations
from regulatrace.config import get_settings
from regulatrace.llm import get_llm
from regulatrace.privacy import redact_for_logging
from regulatrace.reranker import (
    ABSTENTION_THRESHOLD,
    MODEL_NAME,
    _get_reranker,
    decide_abstention,
    rerank,
)

from . import run_retrieval

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"
VERSIONS = ("v1", "v2")
_PACING_TOKEN_BUDGET = 7200  # 10% headroom below the observed 8,000 TPM limit.


def _pace_generation(prompt_tokens: int, recent_requests: deque[tuple[float, int]]) -> None:
    """Reserve input plus maximum completion tokens; these are not billed-token counts."""
    estimated_tokens = prompt_tokens + rag._MAX_COMPLETION_TOKENS
    if estimated_tokens > _PACING_TOKEN_BUDGET:
        raise ValueError(
            f"Request estimate of {estimated_tokens} tokens exceeds the "
            f"{_PACING_TOKEN_BUDGET}-token pacing budget; waiting cannot make it fit."
        )
    while True:
        now = time.monotonic()
        while recent_requests and now - recent_requests[0][0] >= 60:
            recent_requests.popleft()
        if sum(tokens for _, tokens in recent_requests) + estimated_tokens <= _PACING_TOKEN_BUDGET:
            recent_requests.append((now, estimated_tokens))
            return
        delay = 60 - (now - recent_requests[0][0])
        print(f"Groq pacing; waiting {delay:.1f}s for estimated token allowance.", flush=True)
        time.sleep(delay)


def _generate_with_retry(
    llm: Any, prompt: str, prompt_tokens: int, recent_requests: deque[tuple[float, int]]
) -> str:
    """Retry only temporary Groq rate limits for the same rendered prompt."""
    for attempt in range(1, 6):
        # Conservatively reserve every attempt, including those rejected with HTTP 429.
        _pace_generation(prompt_tokens, recent_requests)
        try:
            return llm.generate(
                prompt, temperature=0, max_completion_tokens=rag._MAX_COMPLETION_TOKENS
            )
        except RateLimitError as error:
            headers = error.response.headers
            diagnostics = "; ".join(
                f"{name}={headers[name]}"
                for name in (
                    "retry-after",
                    "x-ratelimit-limit-tokens",
                    "x-ratelimit-remaining-tokens",
                    "x-ratelimit-reset-tokens",
                    "x-ratelimit-remaining-requests",
                    "x-ratelimit-reset-requests",
                )
                if name in headers
            )
            print(redact_for_logging(f"Groq HTTP 429: {error.message}; {diagnostics}"), flush=True)
            if attempt == 5:
                raise
            retry_after = error.response.headers.get("retry-after")
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                try:
                    delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    delay = 15.0
            if not math.isfinite(delay) or delay <= 0:
                delay = 15.0
            print(f"Groq rate limit; waiting {delay:g}s before retry {attempt + 1}/5.")
            time.sleep(delay)


def _load_prompts() -> tuple[dict[str, ChatPromptTemplate], dict[str, str]]:
    templates = {}
    hashes = {}
    for version in VERSIONS:
        path = ROOT / "prompts" / f"answer_{version}.txt"
        content = path.read_bytes()
        system, human = content.decode("utf-8").split("END SYSTEM INSTRUCTIONS", 1)
        templates[version] = ChatPromptTemplate.from_messages(
            [("system", system.strip()), ("human", human.strip())]
        )
        hashes[version] = hashlib.sha256(content).hexdigest()
    return templates, hashes


def check_citations(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check cited document/page membership, not factual support for a claim."""
    validation = validate_citations(answer, sources)
    return [
        {**citation, "in_supplied_sources": citation in validation["valid_citations"]}
        for citation in parse_citations(answer)
    ]


def compare_prompts(limit: int | None = None, row_output: TextIO | None = None) -> dict[str, Any]:
    """Retrieve once per development question and generate a paired answer when eligible."""
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 48):
        raise ValueError("limit must be between 1 and 48")
    golden = run_retrieval.load_golden()
    holdout = run_retrieval.load_holdout({record["id"] for record in golden})
    development = [record for record in golden if record["id"] not in holdout]
    if (
        len(development) != 48
        or sum(record["answerable"] for record in development) != 41
        or any(record["verified"] is not True for record in development)
    ):
        raise ValueError(
            "Expected 48 verified development questions: 41 answerable, 7 unanswerable"
        )
    if limit is not None:
        development = development[:limit]

    settings = get_settings()
    if (
        settings.llm_provider.strip().lower() != "groq"
        or settings.llm_model != "openai/gpt-oss-120b"
        or settings.bm25_collection != "regulatrace_bm25"
    ):
        raise ValueError("Prompt comparison requires the frozen Groq model and BM25 collection")
    prompts, prompt_hashes = _load_prompts()
    llm = get_llm(settings)
    reranker_model = _get_reranker()
    encoding = rag._prompt_encoding()
    rows = []
    recent_requests: deque[tuple[float, int]] = deque()

    for record in development:
        question = record["question"]
        candidates = retrieve_bm25(question, k=20)
        ranked = rerank(question, candidates, k=5, model=reranker_model)
        decision = decide_abstention(ranked)
        row = {
            "id": record["id"],
            "type": record["type"],
            "question": question,
            "answerable": record["answerable"],
            "reference_answer": record["reference_answer"],
            "gold_sources": record["gold_sources"],
            "top_rerank_score": decision["top_rerank_score"],
            "answers": {version: None for version in VERSIONS},
            "evidence": {"context": "", "sources": []},
        }
        if decision["abstain"]:
            row.update(stage="gate_refusal", refusal_answer=decision["message"])
        else:
            evidence = rag._budget_evidence(question, ranked, tuple(prompts.values()))
            if not evidence["sources"]:
                row.update(stage="evidence_refusal", refusal_answer=rag._NOT_FOUND)
            else:
                row["stage"] = "generated"
                row["evidence"] = evidence
                for version in VERSIONS:
                    prompt = rag._render_prompt(prompts[version], question, evidence)
                    prompt_tokens = len(encoding.encode(prompt, disallowed_special=()))
                    answer = _generate_with_retry(llm, prompt, prompt_tokens, recent_requests)
                    row["answers"][version] = {
                        "text": answer,
                        "not_found": answer == rag._NOT_FOUND,
                        "prompt_tokens": prompt_tokens,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "citations": check_citations(answer, evidence["sources"]),
                    }
        rows.append(row)
        if row_output is not None:
            row_output.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_output.flush()
        print(f"Completed {len(rows)}/{len(development)}", flush=True)

    summary = {
        "questions_evaluated": len(rows),
        "gate_refusals": sum(row["stage"] == "gate_refusal" for row in rows),
        "evidence_refusals": sum(row["stage"] == "evidence_refusal" for row in rows),
    }
    for version in VERSIONS:
        answers = [row["answers"][version] for row in rows if row["answers"][version] is not None]
        citations = [citation for answer in answers for citation in answer["citations"]]
        summary[version] = {
            "answers_generated": len(answers),
            "not_found_responses": sum(answer["not_found"] for answer in answers),
            "citations_to_supplied_sources": sum(
                citation["in_supplied_sources"] for citation in citations
            ),
            "parsed_citations": len(citations),
            "prompt_tokens_total": sum(answer["prompt_tokens"] for answer in answers),
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "configuration": {
            "split": "development",
            "retriever": "bm25",
            "collection": settings.bm25_collection,
            "candidate_k": 20,
            "reranker_model": MODEL_NAME,
            "final_k": 5,
            "abstention_threshold": ABSTENTION_THRESHOLD,
            "provider": "groq",
            "generation_model": settings.llm_model,
            "temperature": 0,
            "max_completion_tokens": rag._MAX_COMPLETION_TOKENS,
            "prompt_encoding": encoding.name,
            "max_prompt_tokens": rag._MAX_PROMPT_TOKENS,
            "prompt_file_sha256": prompt_hashes,
        },
        "citation_check_note": (
            "Citation membership in supplied sources does not prove claim support."
        ),
        "summary": summary,
        "per_question": rows,
    }


def save_results(report: dict[str, Any]) -> Path:
    """Write one timestamped report without replacing an earlier comparison."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp_utc"].replace("-", "").replace(":", "").replace(".", "")
    path = RESULTS_DIR / f"prompt_comparison_{stamp}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare answer prompts on development questions")
    parser.add_argument(
        "--limit", type=int, help="Evaluate the first N development questions (1-48)"
    )
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 48:
        parser.error("--limit must be between 1 and 48")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    progress_path = RESULTS_DIR / f"prompt_comparison_{stamp}.jsonl"
    print(f"Progress file: {progress_path}", flush=True)
    try:
        with progress_path.open("x", encoding="utf-8") as row_output:
            report = compare_prompts(limit=args.limit, row_output=row_output)
    except Exception as error:
        # The SDK exception can contain original prompts; do not emit its traceback.
        print(f"Prompt comparison failed: {type(error).__name__}.", file=sys.stderr)
        raise SystemExit(1) from None
    path = save_results(report)
    summary = report["summary"]
    print(f"Development questions: {summary['questions_evaluated']}")
    print(
        f"Refused before generation: {summary['gate_refusals']} gate, "
        f"{summary['evidence_refusals']} evidence"
    )
    for version in VERSIONS:
        result = summary[version]
        print(
            f"{version}: {result['answers_generated']} answers; "
            f"{result['not_found_responses']} exact not-found; "
            f"{result['citations_to_supplied_sources']}/{result['parsed_citations']} "
            f"parsed citations refer to supplied sources; "
            f"{result['prompt_tokens_total']} total prompt tokens"
        )
    print("Citation references require manual claim-support review.")
    print(f"Saved progress: {progress_path}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
