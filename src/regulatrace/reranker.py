"""Cross-encoder reranking of already retrieved chunks."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from regulatrace.model_factories import cross_encoder as CrossEncoder
from regulatrace.retrieval import _validate_retrieval_inputs

MODEL_NAME = "BAAI/bge-reranker-base"
ABSTENTION_THRESHOLD = 0.60
ABSTENTION_MESSAGE = (
    "I couldn't find sufficient evidence in the indexed regulatory documents "
    "to answer this question reliably."
)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the reranker once, on the first nonempty request without an injected model."""
    return CrossEncoder(MODEL_NAME)


def rerank(
    question: str,
    candidates: list[dict[str, Any]],
    k: int = 5,
    model: CrossEncoder | None = None,
) -> list[dict[str, Any]]:
    """Sort candidates by rerank_score; retain score as the original retrieval score."""
    _validate_retrieval_inputs(question, k)
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"candidate {index} must be a dictionary")
        if not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
            raise ValueError(f"candidate {index} must have nonempty text")
    if not candidates:
        return []

    reranker = model if model is not None else _get_reranker()
    pairs = [(question, candidate["text"]) for candidate in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    if len(scores) != len(candidates):
        raise ValueError("reranker returned a different number of scores than candidates")

    ranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        rerank_score = float(score)
        if not math.isfinite(rerank_score):
            raise ValueError("reranker returned a non-finite score")
        ranked.append({**candidate, "rerank_score": rerank_score})
    return sorted(ranked, key=lambda candidate: candidate["rerank_score"], reverse=True)[:k]


def decide_abstention(reranked_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate retrieved evidence by its highest cross-encoder score, without generating an answer."""
    if not isinstance(reranked_results, list):
        raise TypeError("reranked_results must be a list")
    scores = []
    for index, result in enumerate(reranked_results):
        if not isinstance(result, dict):
            raise TypeError(f"reranked result {index} must be a dictionary")
        score = result.get("rerank_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise ValueError(f"reranked result {index} must have a finite rerank_score")
        scores.append(float(score))

    top_score = max(scores, default=None)
    abstain = top_score is None or top_score < ABSTENTION_THRESHOLD
    return {
        "abstain": abstain,
        "top_rerank_score": top_score,
        "message": (
            ABSTENTION_MESSAGE
            if abstain
            else "Retrieved evidence meets the score threshold; answer correctness is not assured."
        ),
    }
