"""Pure reciprocal rank fusion for ordered chunk IDs."""


def reciprocal_rank_fusion(
    dense_ids: list[str], sparse_ids: list[str], smoothing: int = 60
) -> list[tuple[str, float]]:
    """Fuse rankings; equal scores keep first-seen order (dense, then sparse)."""
    if isinstance(smoothing, bool) or not isinstance(smoothing, int) or smoothing <= 0:
        raise ValueError("smoothing must be a positive integer")

    scores: dict[str, float] = {}
    for ranking in (dense_ids, sparse_ids):
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (smoothing + rank)

    return sorted(scores.items(), key=lambda item: -item[1])
