"""Bounded PDF uploads and a session-local BM25 index for the free Space."""

import hashlib
import threading
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf
from qdrant_client import QdrantClient, models

from regulatrace.ingestion.chunker import chunk_pages
from regulatrace.ingestion.parser import ParsedPage
from regulatrace.retrieval import _map_search_results, _validate_retrieval_inputs

MAX_FILES = 5
MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 300
MAX_CHUNKS = 2000
MAX_TEXT = 2_000_000
_EMBED_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _sparse_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


def sparse_vectors(texts: list[str], *, query: bool = False) -> list[dict[str, Any]]:
    # Only the CPU model is shared; no uploaded text is stored in module globals.
    with _EMBED_LOCK:
        model = _sparse_model()
        embeddings = model.query_embed(texts) if query else model.embed(texts)
        return [{"indices": e.indices.tolist(), "values": e.values.tolist()} for e in embeddings]


def prepare_corpus(files: list[str | Path] | None) -> dict[str, Any]:
    """Build a complete bounded upload batch, with no production database writes."""
    if not files or len(files) > MAX_FILES:
        raise ValueError("Choose between 1 and 5 text-based PDFs.")
    chunks, documents, seen = [], [], set()
    page_count = text_count = 0
    for item in files:
        path = Path(item)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError("Every upload must be a PDF file.")
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Each PDF must be at most 10 MiB.")
        with path.open("rb") as handle:
            if not handle.read(8).startswith(b"%PDF-"):
                raise ValueError("An upload is not a valid PDF.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass:
                    raise ValueError("Password-protected PDFs are not supported.")
                page_count += pdf.page_count
                if not pdf.page_count or page_count > MAX_PAGES:
                    raise ValueError("The combined upload must have 1 to 300 pages.")
                pages = []
                for number, page in enumerate(pdf, start=1):
                    text = page.get_text()
                    text_count += len(text)
                    if text_count > MAX_TEXT:
                        raise ValueError("The PDFs contain too much text for this free demo.")
                    pages.append(
                        ParsedPage(
                            page_number=number, text=text, is_low_text_page=len(text.strip()) < 50
                        )
                    )
        except ValueError:
            raise
        except Exception:
            raise ValueError("An uploaded PDF could not be read.") from None
        name = "".join(c for c in path.name if c.isprintable())[:180]
        # A matching filename does not prove that this is an official manifest PDF.
        metadata = {
            "doc_id": digest[:16],
            "title": name,
            "regulator": "USER_UPLOAD",
            "source_url": None,
            "access_level": "session",
        }
        parsed = chunk_pages(pages, source_file=name, doc_metadata=metadata)
        if not parsed:
            raise ValueError("A PDF has no usable text. Scanned PDFs need OCR before uploading.")
        for chunk in parsed:
            chunks.append(
                {
                    "text": chunk.text,
                    "source_file": name,
                    "page": chunk.metadata["page"],
                    "metadata": chunk.metadata,
                }
            )
            if len(chunks) > MAX_CHUNKS:
                raise ValueError("The PDFs exceed the free demo's 2,000-chunk limit.")
        documents.append({"name": name, "pages": len(pages), "doc_id": digest[:16]})
    return {
        "chunks": chunks,
        "documents": documents,
        "pages": page_count,
        "vectors": sparse_vectors([c["text"] for c in chunks]),
    }


def retrieve_session(
    question: str, corpus: dict[str, Any], k: int = 20, *, public_only: bool = False
) -> list[dict[str, Any]]:
    """Search this session's uploads using local FastEmbed BM25 and Qdrant IDF."""
    _validate_retrieval_inputs(question, k)
    if public_only:
        raise ValueError("Session uploads are not a public production corpus.")
    chunks, vectors = corpus.get("chunks", []), corpus.get("vectors", [])
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("Index PDFs before asking a question.")
    query = sparse_vectors([question], query=True)[0]
    if not query["indices"]:
        return []
    # The client never enters Gradio state or crosses the ZeroGPU process boundary.
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "session",
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        client.upsert(
            "session",
            points=[
                models.PointStruct(
                    id=i,
                    vector={"bm25": models.SparseVector(**vector)},
                    payload={
                        **chunk["metadata"],
                        "text": chunk["text"],
                        "source_file": chunk["source_file"],
                        "page": chunk["page"],
                    },
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
            ],
            wait=True,
        )
        points = client.query_points(
            "session", query=models.SparseVector(**query), using="bm25", limit=k, with_payload=True
        ).points
        return _map_search_results([p for p in points if p.score > 0])
