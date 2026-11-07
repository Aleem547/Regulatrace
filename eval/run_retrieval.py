"""Evaluate retrieval on development questions or the explicitly selected frozen hold-out."""

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from regulatrace.bm25 import retrieve_bm25
from regulatrace.config import get_settings
from regulatrace.hybrid import retrieve_hybrid
from regulatrace.reranker import (
    ABSTENTION_THRESHOLD,
    MODEL_NAME,
    _get_reranker,
    decide_abstention,
    rerank,
)
from regulatrace.retrieval import retrieve

if __package__:
    from .metrics import hit_at_k, recall_at_k, reciprocal_rank
else:
    from metrics import hit_at_k, recall_at_k, reciprocal_rank

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "eval" / "golden.jsonl"
HOLDOUT_PATH = ROOT / "data" / "eval" / "holdout_ids.txt"
RESULTS_DIR = ROOT / "eval" / "results"
REQUIRED_FIELDS = {
    "id",
    "question",
    "type",
    "gold_sources",
    "reference_answer",
    "answerable",
    "verified",
}
RERANK_CANDIDATES = 20


def _source_refs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep ordered document/page identifiers for later error analysis."""
    return [
        {
            "rank": rank,
            "doc_id": (result.get("metadata") or {}).get("doc_id"),
            "page": (result.get("metadata") or {}).get("page"),
        }
        for rank, result in enumerate(results, start=1)
    ]


def load_golden() -> list[dict[str, Any]]:
    """Read and validate the golden JSONL before any retrieval calls."""
    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_PATH}")

    records = []
    seen_ids = set()
    for line_number, line in enumerate(GOLDEN_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank golden record at line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid golden JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict) or set(record) != REQUIRED_FIELDS:
            raise ValueError(f"Invalid golden fields at line {line_number}")
        if not isinstance(record["id"], str) or not record["id"].strip():
            raise ValueError(f"Invalid golden ID at line {line_number}")
        if record["id"] in seen_ids:
            raise ValueError(f"Duplicate golden ID: {record['id']}")
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"Invalid question for {record['id']}")
        if not isinstance(record["type"], str) or not isinstance(record["reference_answer"], str):
            raise ValueError(f"Invalid type or reference answer for {record['id']}")
        if type(record["answerable"]) is not bool or type(record["verified"]) is not bool:
            raise ValueError(f"Invalid answerable or verified flag for {record['id']}")
        sources = record["gold_sources"]
        if not isinstance(sources, list) or record["answerable"] != bool(sources):
            raise ValueError(f"Invalid gold_sources for {record['id']}")
        for source in sources:
            if (
                not isinstance(source, dict)
                or set(source) != {"doc_id", "page"}
                or not isinstance(source["doc_id"], str)
                or not source["doc_id"].strip()
                or type(source["page"]) is not int
                or source["page"] < 1
            ):
                raise ValueError(f"Invalid gold source for {record['id']}")
        seen_ids.add(record["id"])
        records.append(record)

    if not records:
        raise ValueError("Golden dataset is empty")
    return records


def load_holdout(golden_ids: set[str]) -> set[str]:
    """Validate the frozen twelve-question hold-out split."""
    if not HOLDOUT_PATH.is_file():
        raise FileNotFoundError(f"Hold-out IDs not found: {HOLDOUT_PATH}")
    ids = HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
    if len(ids) != 12 or any(not item or item != item.strip() for item in ids):
        raise ValueError("Hold-out file must contain exactly 12 nonblank IDs, one per line")
    if len(set(ids)) != len(ids):
        raise ValueError("Hold-out IDs must be unique")
    unknown = set(ids) - golden_ids
    if unknown:
        raise ValueError(f"Hold-out IDs missing from golden dataset: {', '.join(sorted(unknown))}")
    return set(ids)


def _average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    count = len(rows)
    return {
        "hit_at_k": sum(row["hit_at_k"] for row in rows) / count,
        "recall_at_k": sum(row["recall_at_k"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
    }


def evaluate(
    k: int,
    method: str = "dense",
    candidate_k: int = RERANK_CANDIDATES,
    split: str = "development",
) -> dict[str, Any]:
    """Run one ordered retrieval evaluation on the selected split."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if method not in ("dense", "bm25", "hybrid", "bm25_rerank", "hybrid_rerank"):
        raise ValueError("method must be dense, bm25, hybrid, bm25_rerank, or hybrid_rerank")
    if type(candidate_k) is not int or candidate_k not in (10, 20, 30):
        raise ValueError("candidate_k must be 10, 20, or 30")
    if method != "bm25_rerank" and candidate_k != RERANK_CANDIDATES:
        raise ValueError("candidate_k tuning is supported only for bm25_rerank")
    if method == "hybrid" and k > 5:
        raise ValueError("hybrid k must not exceed candidate_k=5")
    reranking = method in ("bm25_rerank", "hybrid_rerank")
    if reranking and k != 5:
        raise ValueError("reranking methods require final k=5")
    if split not in ("development", "holdout"):
        raise ValueError("split must be development or holdout")
    if split == "holdout" and (method != "bm25_rerank" or candidate_k != 20):
        raise ValueError("holdout requires frozen BM25 reranking with candidate_k=20")
    if split == "holdout" and (
        MODEL_NAME != "BAAI/bge-reranker-base" or ABSTENTION_THRESHOLD != 0.60
    ):
        raise ValueError("Frozen reranker model or abstention threshold has changed")
    golden = load_golden()
    holdout_ids = load_holdout({record["id"] for record in golden})
    development = [record for record in golden if record["id"] not in holdout_ids]
    selected = (
        [record for record in golden if record["id"] in holdout_ids]
        if split == "holdout"
        else development
    )
    eligible = (
        selected if split == "holdout" else [record for record in selected if record["answerable"]]
    )
    if not eligible:
        message = (
            "No answerable development questions to evaluate"
            if split == "development"
            else "No hold-out questions to evaluate"
        )
        raise ValueError(message)
    if any(record["verified"] is not True for record in eligible):
        raise ValueError("All evaluated questions must have verified=true")

    settings = get_settings()
    if split == "holdout" and settings.bm25_collection != "regulatrace_bm25":
        raise ValueError("Frozen BM25 collection has changed")
    rerank_depth = candidate_k if method == "bm25_rerank" else RERANK_CANDIDATES
    retrieval_kwargs = {"k": rerank_depth if reranking else k}
    if method == "dense":
        configuration = {
            "k": k,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection,
        }
        retriever = retrieve
    elif method in ("bm25", "bm25_rerank"):
        configuration = {
            "method": method,
            "k": k,
            "qdrant_collection": settings.bm25_collection,
            "bm25_model": "qdrant/bm25",
            "sparse_vector": "bm25",
            "modifier": "idf",
        }
        if reranking:
            configuration.update(
                candidate_k=candidate_k,
                final_k=k,
                reranker_model=MODEL_NAME,
            )
        retriever = retrieve_bm25
    else:
        configuration = {
            "method": method,
            "k": k,
            "candidate_k": RERANK_CANDIDATES if reranking else 5,
            "qdrant_collection": settings.hybrid_collection,
            "embedding_model": settings.embedding_model,
            "bm25_model": "qdrant/bm25",
            "dense_vector": "dense",
            "sparse_vector": "bm25",
            "modifier": "idf",
            "rrf_smoothing": 60,
            "rrf_k": 61,
        }
        if reranking:
            configuration.update(final_k=k, reranker_model=MODEL_NAME)
        retriever = retrieve_hybrid
        retrieval_kwargs["candidate_k"] = RERANK_CANDIDATES if reranking else 5
    if split == "holdout":
        configuration.update(split="holdout", abstention_threshold=ABSTENTION_THRESHOLD)
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    configuration_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    per_question = []
    control_rows = []
    retrieval_latencies = []
    rerank_latencies = []
    model = _get_reranker() if reranking else None
    for record in eligible:
        if split == "holdout":
            start = time.perf_counter()
        results = retriever(record["question"], **retrieval_kwargs)
        if split == "holdout":
            retrieval_latencies.append((time.perf_counter() - start) * 1000)
        sources = record["gold_sources"]
        row = {"id": record["id"], "type": record["type"]}
        if reranking:
            if split == "development":
                control = {
                    "id": record["id"],
                    "type": record["type"],
                    "hit_at_k": hit_at_k(results, sources, k),
                    "recall_at_k": recall_at_k(results, sources, k),
                    "reciprocal_rank": reciprocal_rank(results, sources, k),
                }
                control_rows.append(control)
                row["control"] = {
                    key: control[key] for key in ("hit_at_k", "recall_at_k", "reciprocal_rank")
                }
                row["control_sources"] = _source_refs(results[:k])
            start = time.perf_counter()
            results = rerank(record["question"], results, k=k, model=model)
            latency_ms = (time.perf_counter() - start) * 1000
            rerank_latencies.append(latency_ms)
            if split == "development":
                row["rerank_latency_ms"] = latency_ms
                row["retrieved_sources"] = _source_refs(results)
        if split == "holdout":
            decision = decide_abstention(results)
            row.update(
                answerable=record["answerable"],
                abstain=decision["abstain"],
                top_rerank_score=decision["top_rerank_score"],
            )
        if record["answerable"]:
            row.update(
                hit_at_k=hit_at_k(results, sources, k),
                recall_at_k=recall_at_k(results, sources, k),
                reciprocal_rank=reciprocal_rank(results, sources, k),
            )
        per_question.append(row)

    if split == "holdout":
        answerable_rows = [row for row in per_question if row["answerable"]]
        unanswerable_rows = [row for row in per_question if not row["answerable"]]
        correct_refusals = sum(row["abstain"] for row in unanswerable_rows)
        wrong_refusals = sum(row["abstain"] for row in answerable_rows)
        return {
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "method": "bm25_rerank_holdout",
            "configuration_hash": configuration_hash,
            "configuration": configuration,
            "counts": {
                "holdout_questions": len(per_question),
                "answerable_evaluated": len(answerable_rows),
                "unanswerable_evaluated": len(unanswerable_rows),
                "correct_refusals": correct_refusals,
                "missed_refusals": len(unanswerable_rows) - correct_refusals,
                "wrong_refusals": wrong_refusals,
                "correct_acceptances": len(answerable_rows) - wrong_refusals,
            },
            "rates": {
                "correct_refusal_rate": correct_refusals / len(unanswerable_rows)
                if unanswerable_rows
                else None,
                "wrong_refusal_rate": wrong_refusals / len(answerable_rows)
                if answerable_rows
                else None,
            },
            "metrics": _average_metrics(answerable_rows)
            if answerable_rows
            else {"hit_at_k": None, "recall_at_k": None, "mrr": None},
            "retrieval_latency_p50_ms": statistics.median(retrieval_latencies),
            "rerank_latency_p50_ms": statistics.median(rerank_latencies),
            "per_question": per_question,
        }

    count = len(per_question)
    metrics_by_type = {}
    for question_type in dict.fromkeys(row["type"] for row in per_question):
        rows = [row for row in per_question if row["type"] == question_type]
        metrics_by_type[question_type] = {"count": len(rows), **_average_metrics(rows)}
    now = datetime.now(timezone.utc)
    report = {
        "timestamp_utc": now.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "method": method,
        "configuration_hash": configuration_hash,
        "configuration": configuration,
        "counts": {
            "golden_questions": len(golden),
            "holdout_questions": len(holdout_ids),
            "development_questions": len(development),
            "answerable_evaluated": count,
            "unanswerable_excluded": len(development) - count,
        },
        "metrics": _average_metrics(per_question),
        "metrics_by_type": metrics_by_type,
        "per_question": per_question,
    }
    if reranking:
        report["control_metrics"] = _average_metrics(control_rows)
        report["rerank_latency_p50_ms"] = statistics.median(rerank_latencies)
    return report


def save_results(report: dict[str, Any]) -> Path:
    """Create the results directory only after evaluation has completed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = report["timestamp_utc"].replace("-", "").replace(":", "").replace(".", "")
    method = report.get("method", "dense")
    prefix = f"retrieval_{method}" if method != "dense" else "retrieval"
    path = RESULTS_DIR / f"{prefix}_{timestamp}_{report['configuration_hash']}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the selected split")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved chunks (default: 5)")
    parser.add_argument(
        "--method",
        choices=("dense", "bm25", "hybrid", "bm25_rerank", "hybrid_rerank"),
        default="dense",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        choices=(10, 20, 30),
        default=RERANK_CANDIDATES,
        help="BM25 rerank candidate depth (default: 20)",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be a positive integer")
    if args.method == "hybrid" and args.k > 5:
        parser.error("hybrid --k must not exceed candidate_k=5")
    if args.method in ("bm25_rerank", "hybrid_rerank") and args.k != 5:
        parser.error("reranking methods require --k 5")
    if args.method != "bm25_rerank" and args.candidate_k != RERANK_CANDIDATES:
        parser.error("--candidate-k tuning is supported only for bm25_rerank")
    if args.split == "holdout":
        if args.method != "bm25_rerank" or args.candidate_k != RERANK_CANDIDATES:
            parser.error("holdout requires --method bm25_rerank, --k 5, and --candidate-k 20")
        report = evaluate(5, method="bm25_rerank", split="holdout")
        output_path = save_results(report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "counts",
                        "rates",
                        "metrics",
                        "retrieval_latency_p50_ms",
                        "rerank_latency_p50_ms",
                    )
                },
                indent=2,
            )
        )
        print(f"Saved: {output_path}")
        return

    report = evaluate(args.k, method=args.method, candidate_k=args.candidate_k)
    output_path = save_results(report)
    config = report["configuration"]
    counts = report["counts"]
    metrics = report["metrics"]
    if args.method == "dense":
        configuration_text = (
            f"k={config['k']}, embedding_model={config['embedding_model']}, "
            f"qdrant_collection={config['qdrant_collection']}"
        )
    elif args.method == "bm25":
        configuration_text = (
            f"method=bm25, k={config['k']}, bm25_model={config['bm25_model']}, "
            f"qdrant_collection={config['qdrant_collection']}, vector={config['sparse_vector']}"
        )
    elif args.method == "hybrid":
        configuration_text = (
            f"method=hybrid, k={config['k']}, candidate_k={config['candidate_k']}, "
            f"qdrant_collection={config['qdrant_collection']}, "
            f"embedding_model={config['embedding_model']}, bm25_model={config['bm25_model']}, "
            f"rrf_k={config['rrf_k']}"
        )
    else:
        configuration_text = (
            f"method={args.method}, final_k={config['final_k']}, "
            f"candidate_k={config['candidate_k']}, "
            f"qdrant_collection={config['qdrant_collection']}, "
            f"reranker_model={config['reranker_model']}"
        )
    rows = [
        ("Configuration", configuration_text),
        ("Development questions", counts["development_questions"]),
        ("Answerable evaluated", counts["answerable_evaluated"]),
        ("Unanswerable excluded", counts["unanswerable_excluded"]),
        (f"Hit@{args.k}", f"{metrics['hit_at_k']:.4f}"),
        (f"Recall@{args.k}", f"{metrics['recall_at_k']:.4f}"),
        ("MRR", f"{metrics['mrr']:.4f}"),
    ]
    if args.method in ("bm25_rerank", "hybrid_rerank"):
        control = report["control_metrics"]
        rows.extend(
            [
                (f"Control Hit@{args.k}", f"{control['hit_at_k']:.4f}"),
                (f"Control Recall@{args.k}", f"{control['recall_at_k']:.4f}"),
                ("Control MRR", f"{control['mrr']:.4f}"),
                ("Rerank p50 (ms)", f"{report['rerank_latency_p50_ms']:.2f}"),
            ]
        )
    print(f"{'Retrieval evaluation':<24} Value")
    for label, value in rows:
        print(f"{label:<24} {value}")
    print("By question type:")
    print(f"{'Type':<16} {'Count':>5} {f'Hit@{args.k}':>8} {f'Recall@{args.k}':>10} {'MRR':>8}")
    for question_type, values in report["metrics_by_type"].items():
        print(
            f"{question_type:<16} {values['count']:>5} "
            f"{values['hit_at_k']:>8.4f} {values['recall_at_k']:>10.4f} "
            f"{values['mrr']:>8.4f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
