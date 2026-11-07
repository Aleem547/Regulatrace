"""Safe setup for the separate dense-and-BM25 Qdrant collection."""

from typing import Any

from qdrant_client import QdrantClient, models

from regulatrace.config import get_settings
from regulatrace.llm import get_embeddings
from regulatrace.retrieval import _map_search_results, _validate_retrieval_inputs
from regulatrace.vectorstore import build_retrieval_filter, get_qdrant_client


def init_hybrid_collection(
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> None:
    """Create the hybrid collection, or verify its existing vector schema."""
    settings = get_settings()
    name = collection_name if collection_name is not None else settings.hybrid_collection
    if name in (settings.qdrant_collection, settings.bm25_collection):
        raise ValueError("Hybrid collection must differ from dense and BM25 collections")

    q_client = client if client is not None else get_qdrant_client()
    if not q_client.collection_exists(name):
        q_client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        return

    params = q_client.get_collection(name).config.params
    vectors = params.vectors
    sparse = params.sparse_vectors or {}
    if (
        not isinstance(vectors, dict)
        or set(vectors) != {"dense"}
        or vectors["dense"].size != 384
        or vectors["dense"].distance != models.Distance.COSINE
        or set(sparse) != {"bm25"}
        or sparse["bm25"].modifier != models.Modifier.IDF
    ):
        raise ValueError(
            f"Collection {name!r} has an incompatible schema; "
            "expected dense 384/COSINE and sparse 'bm25' with IDF"
        )


def backfill_hybrid(client: QdrantClient | None = None, batch_size: int = 16) -> int:
    """Copy stored dense vectors and payloads; infer BM25 into the hybrid collection."""
    settings = get_settings()
    source_name = settings.qdrant_collection
    target_name = settings.hybrid_collection
    if len({source_name, settings.bm25_collection, target_name}) != 3:
        raise ValueError("Dense, BM25, and hybrid collections must have different names")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if client is None:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        client = QdrantClient(url=settings.qdrant_url, api_key=api_key, cloud_inference=True)
    if not client.cloud_inference:
        raise ValueError("Hybrid backfill requires server-side inference")
    if not client.collection_exists(target_name):
        raise ValueError("Hybrid collection must already exist")
    init_hybrid_collection(client=client, collection_name=target_name)

    copied = 0
    offset = None
    while True:
        source_points, next_offset = client.scroll(
            collection_name=source_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=["dense"],
        )
        points = []
        for source in source_points:
            text = (source.payload or {}).get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Dense point {source.id} has missing or empty text")
            vectors = source.vector
            dense_vector = vectors.get("dense") if isinstance(vectors, dict) else None
            if not isinstance(dense_vector, list) or len(dense_vector) != 384:
                raise ValueError(
                    f"Dense point {source.id} has missing or invalid 384-dimensional vector"
                )
            points.append(
                models.PointStruct(
                    id=source.id,
                    vector={
                        "dense": dense_vector,
                        "bm25": models.Document(text=text, model="qdrant/bm25"),
                    },
                    payload=source.payload,
                )
            )
        if points:
            client.upsert(collection_name=target_name, points=points, wait=True)
            copied += len(points)
        if next_offset is None:
            return copied
        offset = next_offset


def retrieve_hybrid(
    question: str,
    k: int = 5,
    candidate_k: int = 5,
    regulator: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return Qdrant's native RRF fusion of dense and BM25 candidates."""
    _validate_retrieval_inputs(question, k)
    if not isinstance(candidate_k, int) or isinstance(candidate_k, bool):
        raise TypeError("candidate_k must be an integer, excluding booleans")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    if candidate_k < k:
        raise ValueError("candidate_k must be at least k")

    query_filter = build_retrieval_filter(regulator, start_date, end_date)
    prefetch_filter = {"filter": query_filter} if query_filter is not None else {}
    settings = get_settings()
    dense_query = get_embeddings().encode(question, normalize_embeddings=True).tolist()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key, cloud_inference=True)
    points = client.query_points(
        collection_name=settings.hybrid_collection,
        prefetch=[
            models.Prefetch(query=dense_query, using="dense", limit=candidate_k, **prefetch_filter),
            models.Prefetch(
                query=models.Document(text=question, model="qdrant/bm25"),
                using="bm25",
                limit=candidate_k,
                **prefetch_filter,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=61)),
        limit=k,
        with_payload=True,
    ).points
    return _map_search_results(points)
