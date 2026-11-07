"""Compare development-question reranker scores by answerability."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from regulatrace.bm25 import retrieve_bm25
from regulatrace.config import get_settings
from regulatrace.reranker import MODEL_NAME, _get_reranker, rerank

from . import run_retrieval

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CANDIDATE_K = 20
FINAL_K = 5


def threshold_counts(rows: list[dict], threshold: float) -> dict:
    """Treat a missing score or a score below threshold as a refusal."""
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    if not answerable or not unanswerable:
        raise ValueError("Both answerability groups are required")

    def refused(row: dict) -> bool:
        score = row["top_rerank_score"]
        return score is None or score < threshold

    correct_refusals = sum(refused(row) for row in unanswerable)
    wrong_refusals = sum(refused(row) for row in answerable)
    return {
        "threshold": threshold,
        "correct_refusals": correct_refusals,
        "missed_refusals": len(unanswerable) - correct_refusals,
        "wrong_refusals": wrong_refusals,
        "correct_acceptances": len(answerable) - wrong_refusals,
        "correct_refusal_rate": correct_refusals / len(unanswerable),
        "wrong_refusal_rate": wrong_refusals / len(answerable),
    }


def analyze_scores() -> dict:
    """Score all verified development questions, including unanswerable ones."""
    golden = run_retrieval.load_golden()
    holdout = run_retrieval.load_holdout({record["id"] for record in golden})
    development = [record for record in golden if record["id"] not in holdout]
    if (
        len(development) != 48
        or sum(record["answerable"] for record in development) != 41
        or any(record["verified"] is not True for record in development)
    ):
        raise ValueError(
            "Expected 48 verified development questions: 41 answerable and 7 unanswerable"
        )

    model = None
    rows = []
    for record in development:
        candidates = retrieve_bm25(record["question"], k=CANDIDATE_K)
        if candidates and model is None:
            model = _get_reranker()
        ranked = rerank(record["question"], candidates, k=FINAL_K, model=model)
        rows.append(
            {
                "id": record["id"],
                "answerable": record["answerable"],
                "top_rerank_score": ranked[0]["rerank_score"] if ranked else None,
            }
        )

    scores = [row["top_rerank_score"] for row in rows if row["top_rerank_score"] is not None]
    unanswerable_scores = sorted(
        {
            row["top_rerank_score"]
            for row in rows
            if not row["answerable"] and row["top_rerank_score"] is not None
        }
    )
    # Three cutoffs: just above the low, middle, and high unanswerable scores.
    positions = (0, len(unanswerable_scores) // 2, len(unanswerable_scores) - 1)
    thresholds = (
        sorted({math.nextafter(unanswerable_scores[index], math.inf) for index in positions})
        if unanswerable_scores
        else []
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "configuration": {
            "method": "bm25_rerank",
            "collection": get_settings().bm25_collection,
            "candidate_k": CANDIDATE_K,
            "final_k": FINAL_K,
            "reranker_model": MODEL_NAME,
        },
        "counts": {"development": 48, "answerable": 41, "unanswerable": 7, "holdout_excluded": 12},
        "score_behavior": {
            "activation": type(getattr(model, "activation_fn", None)).__name__,
            "observed_min": min(scores) if scores else None,
            "observed_max": max(scores) if scores else None,
            "note": "Scores are not calibrated probabilities; null means no candidates.",
        },
        "per_question": rows,
        "thresholds": [threshold_counts(rows, value) for value in thresholds],
    }


def score_plot_svg(rows: list[dict]) -> str:
    """Draw a compact dot plot on a shared raw-score axis."""
    scored = [row for row in rows if row["top_rerank_score"] is not None]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="220">',
        '<rect width="800" height="220" fill="white"/>',
        '<text x="130" y="25">Top reranker scores (not probabilities)</text>',
    ]
    if scored:
        low = min(row["top_rerank_score"] for row in scored)
        high = max(row["top_rerank_score"] for row in scored)
        for answerable, label, y, color in (
            (True, "Answerable", 80, "#1769aa"),
            (False, "Unanswerable", 140, "#d35b24"),
        ):
            group = [row for row in rows if row["answerable"] is answerable]
            missing = sum(row["top_rerank_score"] is None for row in group)
            parts.append(f'<text x="10" y="{y}">{label} ({len(group)})</text>')
            for index, row in enumerate(group):
                if row["top_rerank_score"] is not None:
                    x = 130 + (row["top_rerank_score"] - low) / (high - low or 1) * 620
                    parts.append(
                        f'<circle cx="{x:.1f}" cy="{y + (index % 5 - 2) * 4}" '
                        f'r="4" fill="{color}" opacity="0.7"/>'
                    )
            if missing:
                parts.append(f'<text x="130" y="{y + 27}">{missing} no-candidate</text>')
        parts.append(f'<text x="130" y="190">{low:.4g}</text>')
        parts.append(f'<text x="750" y="190" text-anchor="end">{high:.4g}</text>')
        parts.append('<line x1="130" y1="170" x2="750" y2="170" stroke="#333"/>')
    else:
        parts.append('<text x="130" y="100">No candidates returned; no scores to plot.</text>')
    parts.append('<text x="440" y="210" text-anchor="middle">Raw rerank_score</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def save_analysis(report: dict) -> tuple[Path, Path]:
    """Save one JSON report and its plot without overwriting prior results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp_utc"].replace("-", "").replace(":", "").replace(".", "")
    stem = f"rerank_score_analysis_{stamp}"
    json_path, plot_path = RESULTS_DIR / f"{stem}.json", RESULTS_DIR / f"{stem}.svg"
    with json_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    with plot_path.open("x", encoding="utf-8") as output:
        output.write(score_plot_svg(report["per_question"]))
    return json_path, plot_path


def main() -> None:
    report = analyze_scores()
    json_path, plot_path = save_analysis(report)
    print(
        json.dumps(
            {"score_behavior": report["score_behavior"], "thresholds": report["thresholds"]},
            indent=2,
        )
    )
    print(f"Saved: {json_path}\nPlot: {plot_path}")


if __name__ == "__main__":
    main()
