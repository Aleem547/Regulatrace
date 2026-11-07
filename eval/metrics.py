"""Page-level retrieval metrics for answerable golden questions."""

from typing import Any


def _source_pair(source: Any) -> tuple[str, int] | None:
    if not isinstance(source, dict):
        return None
    doc_id = source.get("doc_id")
    page = source.get("page")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return None
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        return None
    return doc_id, page


def _gold_pairs(gold_sources: list[dict[str, Any]], k: int) -> set[tuple[str, int]]:
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be a positive integer")
    if k < 1:
        raise ValueError("k must be a positive integer")
    if not gold_sources:
        raise ValueError("gold_sources must be nonempty for retrieval metrics")

    pairs = set()
    for source in gold_sources:
        pair = _source_pair(source)
        if pair is None:
            raise ValueError("gold_sources must contain a doc_id and positive integer page")
        pairs.add(pair)
    return pairs


def _retrieved_pair(result: Any) -> tuple[str, int] | None:
    if not isinstance(result, dict):
        return None
    return _source_pair(result.get("metadata"))


def hit_at_k(
    retrieved_results: list[dict[str, Any]], gold_sources: list[dict[str, Any]], k: int
) -> float:
    """Return 1.0 when any gold document page occurs within the first k chunks."""
    gold_pairs = _gold_pairs(gold_sources, k)
    return float(any(_retrieved_pair(result) in gold_pairs for result in retrieved_results[:k]))


def recall_at_k(
    retrieved_results: list[dict[str, Any]], gold_sources: list[dict[str, Any]], k: int
) -> float:
    """Return the fraction of distinct gold document pages found within k chunks."""
    gold_pairs = _gold_pairs(gold_sources, k)
    found = set()
    for result in retrieved_results[:k]:
        pair = _retrieved_pair(result)
        if pair in gold_pairs:
            found.add(pair)
    return len(found) / len(gold_pairs)


def reciprocal_rank(
    retrieved_results: list[dict[str, Any]], gold_sources: list[dict[str, Any]], k: int
) -> float:
    """Return the reciprocal of the first matching chunk's original rank within k."""
    gold_pairs = _gold_pairs(gold_sources, k)
    for rank, result in enumerate(retrieved_results[:k], start=1):
        if _retrieved_pair(result) in gold_pairs:
            return 1.0 / rank
    return 0.0
