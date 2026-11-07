"""Validate and publish prepared vectors without installing local embedding models."""

import gzip
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID

from qdrant_client import models
from sqlalchemy import select

from regulatrace.db.models import Document
from regulatrace.vectorstore import generate_point_id


def load_prepared(folder: Path) -> tuple[dict, list[dict]]:
    """Validate all input before any remote writes are possible."""
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "points.jsonl.gz"
    if (
        manifest.get("format") != 1
        or manifest.get("model") != "BAAI/bge-small-en-v1.5"
        or manifest.get("dimension") != 384
        or manifest.get("normalized") is not True
        or not 0 < manifest.get("point_count", 0) <= 100000
        or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("points_sha256")
    ):
        raise ValueError("Prepared corpus metadata or checksum is invalid")
    documents = manifest.get("documents", [])
    names = {doc["source_file"] for doc in documents}
    if len(names) != len(documents) or len(names) != manifest.get("document_count"):
        raise ValueError("Prepared document list is inconsistent")
    expected_counts = {doc["source_file"]: doc["chunks"] for doc in documents}
    counts = dict.fromkeys(names, 0)
    points, ids, size = [], set(), 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            size += len(line)
            if size > 200000000 or len(points) >= 100000:
                raise ValueError("Prepared corpus exceeds supported limits")
            point = json.loads(line)
            vector, payload = point["vector"], point["payload"]
            point_id = str(UUID(point["id"]))
            if (
                point_id in ids
                or point_id != generate_point_id(payload["doc_id"], payload["chunk_index"])
                or payload["source_file"] not in names
                or payload.get("access_level") != "public"
                or not isinstance(payload.get("text"), str)
                or not payload["text"].strip()
                or not isinstance(payload.get("page"), int)
                or payload["page"] < 1
                or len(vector) != 384
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in vector
                )
                or not 0.99 < sum(v * v for v in vector) < 1.01
                or hashlib.sha256(payload["text"].strip().encode()).hexdigest()
                != payload["content_hash"]
            ):
                raise ValueError("Prepared vector or citation metadata is invalid")
            ids.add(point_id)
            counts[payload["source_file"]] += 1
            points.append(point)
    if len(points) != manifest["point_count"] or counts != expected_counts:
        raise ValueError("Prepared corpus point counts do not match")
    return manifest, points


def collection_names(manifest: dict) -> tuple[str, str]:
    suffix = manifest["points_sha256"][:12]
    return f"regulatrace_{suffix}_dense", f"regulatrace_{suffix}_bm25"


def _check_existing(client, name: str, points: list[dict], *, dense: bool) -> bool:
    if not client.collection_exists(name):
        return False
    params = client.get_collection(name).config.params
    if dense:
        vectors = params.vectors
        if not isinstance(vectors, dict) or set(vectors) != {"dense"}:
            raise ValueError("Existing dense collection has an incompatible schema")
        if vectors["dense"].size != 384 or vectors["dense"].distance != models.Distance.COSINE:
            raise ValueError("Existing collection has a different embedding schema")
    else:
        sparse = params.sparse_vectors or {}
        if params.vectors not in (None, {}) or set(sparse) != {"bm25"}:
            raise ValueError("Existing BM25 collection has an incompatible schema")
        if sparse["bm25"].modifier != models.Modifier.IDF:
            raise ValueError("Existing BM25 collection is missing IDF")
    expected = {p["id"]: p["payload"] for p in points}
    offset = None
    while True:
        batch, offset = client.scroll(
            name, offset=offset, limit=128, with_vectors=False, with_payload=True
        )
        for point in batch:
            if expected.get(str(point.id)) != point.payload:
                raise ValueError("Existing collection contains unrelated or changed evidence")
        if offset is None:
            return True


def publish_prepared(folder: Path, client, session_factory) -> dict:
    """Resume matching data and verify both indexes and the document registry."""
    manifest, points = load_prepared(folder)
    dense, sparse = collection_names(manifest)
    if not client.cloud_inference:
        raise ValueError("BM25 requires Qdrant Cloud server-side inference")
    dense_exists = _check_existing(client, dense, points, dense=True)
    sparse_exists = _check_existing(client, sparse, points, dense=False)
    expected_docs = {doc["source_file"]: doc for doc in manifest["documents"]}
    with session_factory() as db, db.begin():
        for document in db.scalars(select(Document)).all():
            expected = expected_docs.get(document.source_file)
            if expected is None or document.content_hash != expected["sha256"]:
                raise ValueError("Database contains a different corpus; use a dedicated database")
    if not dense_exists:
        client.create_collection(
            dense,
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
    if not sparse_exists:
        client.create_collection(
            sparse,
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    for name in (dense, sparse):
        for key, schema in {
            "regulator": models.PayloadSchemaType.KEYWORD,
            "access_level": models.PayloadSchemaType.KEYWORD,
            "source_file": models.PayloadSchemaType.KEYWORD,
            "issue_date_iso": models.PayloadSchemaType.DATETIME,
        }.items():
            client.create_payload_index(name, key, schema, wait=True)
    for start in range(0, len(points), 32):
        batch = points[start : start + 32]
        client.upsert(
            dense,
            points=[
                models.PointStruct(id=p["id"], vector={"dense": p["vector"]}, payload=p["payload"])
                for p in batch
            ],
            wait=True,
        )
        client.upsert(
            sparse,
            points=[
                models.PointStruct(
                    id=p["id"],
                    vector={
                        "bm25": models.Document(text=p["payload"]["text"], model="qdrant/bm25")
                    },
                    payload=p["payload"],
                )
                for p in batch
            ],
            wait=True,
        )
        print(f"Indexed {min(start + 32, len(points))}/{len(points)} chunks", flush=True)
    dense_count, sparse_count = (client.count(n, exact=True).count for n in (dense, sparse))
    if dense_count != sparse_count or dense_count != len(points):
        raise ValueError("Published index counts do not match")
    public_filter = models.Filter(
        must=[models.FieldCondition(key="access_level", match=models.MatchValue(value="public"))]
    )
    if client.count(sparse, exact=True, count_filter=public_filter).count != len(points):
        raise ValueError("Public evidence count does not match")
    with session_factory() as db, db.begin():
        for item in manifest["documents"]:
            document = db.scalar(
                select(Document).where(Document.source_file == item["source_file"])
            )
            if document is None:
                document = Document(
                    source_file=item["source_file"],
                    content_hash=item["sha256"],
                    status="active",
                    version=1,
                )
                db.add(document)
            metadata = item["metadata"]
            document.chunk_count = item["chunks"]
            document.ingested_at = datetime.now(timezone.utc)
            document.title = metadata["title"]
            document.regulator = metadata["regulator"]
            document.circular_no = metadata["circular_no"]
            document.issue_date = date.fromisoformat(metadata["issue_date_iso"][:10])
            document.access_level = "public"
        db.flush()
        registry = len(db.scalars(select(Document)).all())
        if registry != manifest["document_count"]:
            raise ValueError("Document registry count does not match")
    return {
        "documents": registry,
        "dense_points": dense_count,
        "bm25_points": sparse_count,
        "dense_collection": dense,
        "bm25_collection": sparse,
    }
