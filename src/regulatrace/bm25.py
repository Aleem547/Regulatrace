"""Collection setup for the separate BM25 sparse index."""

from typing import Any

from qdrant_client import QdrantClient, models

from regulatrace.config import get_settings
from regulatrace.retrieval import _map_search_results, _validate_retrieval_inputs
from regulatrace.vectorstore import build_retrieval_filter, get_qdrant_client


def init_bm25_collection(
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> None:
    """Create the sparse-only BM25 collection or verify its existing schema."""
    settings = get_settings()
    name = collection_name or settings.bm25_collection
    if name == settings.qdrant_collection:
        raise ValueError("BM25 collection must differ from the dense collection")

    q_client = client if client is not None else get_qdrant_client()
    if not q_client.collection_exists(name):
        q_client.create_collection(
            collection_name=name,
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        return

    params = q_client.get_collection(name).config.params
    sparse = params.sparse_vectors or {}
    if (
        params.vectors not in (None, {})
        or set(sparse) != {"bm25"}
        or sparse["bm25"].modifier != models.Modifier.IDF
    ):
        raise ValueError(
            f"Collection {name!r} has an incompatible schema; expected sparse-only 'bm25' with IDF"
        )


def backfill_bm25(client: QdrantClient | None = None, batch_size: int = 16) -> int:
    """Copy dense chunk IDs and payloads into the separate BM25 collection."""
    settings = get_settings()
    source_name = settings.qdrant_collection
    target_name = settings.bm25_collection
    if source_name == target_name:
        raise ValueError("BM25 collection must differ from the dense collection")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if client is None:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        client = QdrantClient(url=settings.qdrant_url, api_key=api_key, cloud_inference=True)
    if not client.cloud_inference:
        raise ValueError("BM25 backfill requires server-side inference")
    if not client.collection_exists(target_name):
        raise ValueError("BM25 collection must already exist")
    init_bm25_collection(client=client, collection_name=target_name)

    copied = 0
    offset = None
    while True:
        source_points, next_offset = client.scroll(
            collection_name=source_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points = []
        for source in source_points:
            text = (source.payload or {}).get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Dense point {source.id} has missing or empty text")
            points.append(
                models.PointStruct(
                    id=source.id,
                    vector={"bm25": models.Document(text=text, model="qdrant/bm25")},
                    payload=source.payload,
                )
            )
        if points:
            client.upsert(collection_name=target_name, points=points, wait=True)
            copied += len(points)
        if next_offset is None:
            return copied
        offset = next_offset


def retrieve_bm25(
    question: str,
    k: int = 5,
    regulator: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    public_only: bool = False,
) -> list[dict[str, Any]]:
    """Search the separate BM25 sparse collection and return dense-compatible results."""
    _validate_retrieval_inputs(question, k)
    query_filter = build_retrieval_filter(regulator, start_date, end_date)
    if public_only:
        # This restriction is set by the backend, never by client-supplied roles.
        query_filter = models.Filter(
            must=[
                *([query_filter] if query_filter is not None else []),
                models.FieldCondition(key="access_level", match=models.MatchValue(value="public")),
            ]
        )
    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key, cloud_inference=True)
    try:
        points = client.query_points(
            collection_name=settings.bm25_collection,
            query=models.Document(text=question, model="qdrant/bm25"),
            using="bm25",
            limit=k,
            with_payload=True,
            **({"query_filter": query_filter} if query_filter is not None else {}),
        ).points
        return _map_search_results(points)
    finally:
        client.close()
