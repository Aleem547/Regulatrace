"""Qdrant vector store management, batch embedding, and point upsertion.

Provides collection initialization with dense and sparse named vectors,
deterministic UUIDv5 point IDs for idempotent ingestion, and batch embedding.
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient, models

from regulatrace.config import get_settings
from regulatrace.llm import get_embeddings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from regulatrace.ingestion.chunker import Chunk


def get_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Instantiate a Qdrant client driven by configuration or parameters."""
    settings = get_settings()
    target_url = url or settings.qdrant_url
    target_key = (
        api_key
        if api_key is not None
        else (settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None)
    )
    return QdrantClient(url=target_url, api_key=target_key)


def get_embedding_dimension(embedder: SentenceTransformer) -> int:
    """Retrieve vector dimension dynamically from the embedding model."""
    dim = embedder.get_embedding_dimension()
    if dim is None:
        raise ValueError("Unable to determine embedding dimension from model.")
    return int(dim)


def generate_point_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic UUIDv5 point ID from document ID and chunk index."""
    key = f"{doc_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def build_regulator_filter(regulator: str | None) -> models.Filter | None:
    """Restrict retrieval to one canonical regulator when requested."""
    if regulator is None:
        return None
    if regulator not in ("RBI", "SEBI"):
        raise ValueError("regulator must be 'RBI' or 'SEBI'")
    return models.Filter(
        must=[models.FieldCondition(key="regulator", match=models.MatchValue(value=regulator))]
    )


def build_retrieval_filter(
    regulator: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> models.Filter | None:
    """Build an AND pre-filter for regulator and inclusive issue calendar dates."""
    regulator_filter = build_regulator_filter(regulator)
    conditions = list(regulator_filter.must) if regulator_filter is not None else []

    def parse_date(value: str | None, name: str) -> date | None:
        if value is None:
            return None
        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError(f"{name} must be a valid YYYY-MM-DD date")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid YYYY-MM-DD date") from exc

    start = parse_date(start_date, "start_date")
    end = parse_date(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be after end_date")
    if start is not None or end is not None:
        bounds = {}
        if start is not None:
            bounds["gte"] = datetime.combine(start, time.min, tzinfo=timezone.utc)
        if end is not None:
            if end == date.max:
                bounds["lte"] = datetime.max.replace(tzinfo=timezone.utc)
            else:
                bounds["lt"] = datetime.combine(
                    end + timedelta(days=1), time.min, tzinfo=timezone.utc
                )
        conditions.append(
            models.FieldCondition(key="issue_date_iso", range=models.DatetimeRange(**bounds))
        )
    return models.Filter(must=conditions) if conditions else None


def ensure_filter_indexes(client: QdrantClient | None = None) -> None:
    """Add only missing regulator/date indexes to the three existing collections."""
    settings = get_settings()
    names = (settings.qdrant_collection, settings.bm25_collection, settings.hybrid_collection)
    if len(set(names)) != 3:
        raise ValueError("Filter collections must have different names")
    q_client = client if client is not None else get_qdrant_client()
    required = {
        "regulator": models.PayloadSchemaType.KEYWORD,
        "issue_date_iso": models.PayloadSchemaType.DATETIME,
    }
    missing = []
    for name in names:
        if not q_client.collection_exists(name):
            raise ValueError(f"Collection {name!r} does not exist")
        schema = q_client.get_collection(name).payload_schema or {}
        for field, field_type in required.items():
            existing = schema.get(field)
            if existing is None:
                missing.append((name, field, field_type))
            elif existing.data_type != field_type:
                raise ValueError(f"Collection {name!r} has incompatible {field!r} index")
    for name, field, field_type in missing:
        result = q_client.create_payload_index(
            collection_name=name, field_name=field, field_schema=field_type, wait=True
        )
        if result.status != models.UpdateStatus.COMPLETED:
            raise RuntimeError(f"Index creation did not complete for {name}.{field}")


def init_collection(
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    embedder: SentenceTransformer | None = None,
) -> None:
    """Initialize Qdrant collection with named dense and sparse vectors and payload indexes."""
    settings = get_settings()
    q_client = client or get_qdrant_client()
    c_name = collection_name or settings.qdrant_collection
    model = embedder or get_embeddings()

    if not q_client.collection_exists(c_name):
        dim = get_embedding_dimension(model)
        q_client.create_collection(
            collection_name=c_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

        payload_fields = ["regulator", "issue_date", "status", "access_level"]
        for field in payload_fields:
            q_client.create_payload_index(
                collection_name=c_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )


def embed_chunks(
    chunks: list[Chunk],
    embedder: SentenceTransformer | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """Generate normalized dense embeddings for chunks in batches."""
    if not chunks:
        return []

    settings = get_settings()
    model = embedder or get_embeddings()
    b_size = batch_size or settings.embedding_batch_size
    texts = [c.text for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=b_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vec.tolist() for vec in embeddings]


def upsert_chunks(
    chunks: list[Chunk],
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    embedder: SentenceTransformer | None = None,
    batch_size: int | None = None,
) -> int:
    """Embed and upsert chunks into Qdrant using deterministic point IDs and batching."""
    if not chunks:
        return 0

    settings = get_settings()
    q_client = client or get_qdrant_client()
    c_name = collection_name or settings.qdrant_collection
    model = embedder or get_embeddings()
    b_size = batch_size or settings.embedding_batch_size

    init_collection(client=q_client, collection_name=c_name, embedder=model)

    vectors = embed_chunks(chunks, embedder=model, batch_size=b_size)

    points: list[models.PointStruct] = []
    for chunk, vec in zip(chunks, vectors, strict=True):
        meta = chunk.to_metadata().model_dump()
        point_id = generate_point_id(doc_id=meta["doc_id"], chunk_index=meta["chunk_index"])
        payload: dict[str, Any] = {
            "text": chunk.text,
            "source_file": chunk.metadata.get("source_file"),
            **meta,
        }
        points.append(
            models.PointStruct(
                id=point_id,
                vector={"dense": vec},
                payload=payload,
            )
        )

    for i in range(0, len(points), b_size):
        batch = points[i : i + b_size]
        q_client.upsert(collection_name=c_name, points=batch)

    return len(points)


def search_chunks(
    query: str,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    embedder: SentenceTransformer | None = None,
    limit: int = 5,
    query_filter: models.Filter | None = None,
) -> list[models.ScoredPoint]:
    """Execute a dense vector search query against the Qdrant collection."""
    settings = get_settings()
    q_client = client or get_qdrant_client()
    c_name = collection_name or settings.qdrant_collection
    model = embedder or get_embeddings()

    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    res = q_client.query_points(
        collection_name=c_name,
        query=query_vector,
        using="dense",
        limit=limit,
        **({"query_filter": query_filter} if query_filter is not None else {}),
    )
    return res.points


def delete_document_chunks(
    source_file: str,
    doc_id: str | None = None,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> None:
    """Delete all indexed chunks for a given document from the Qdrant collection."""
    settings = get_settings()
    q_client = client or get_qdrant_client()
    c_name = collection_name or settings.qdrant_collection

    if not q_client.collection_exists(c_name):
        return

    filter_conditions = []
    if doc_id:
        filter_conditions.append(
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        )
    else:
        filter_conditions.append(
            models.FieldCondition(key="source_file", match=models.MatchValue(value=source_file))
        )

    q_client.delete(
        collection_name=c_name,
        points_selector=models.FilterSelector(filter=models.Filter(must=filter_conditions)),
    )
