"""Idempotent document ingestion pipeline and CLI for Regulatrace.

Scans regulatory documents, compares content SHA-256 hashes against PostgreSQL registry,
upserts new/updated chunks into Qdrant with deterministic IDs, and tracks ingestion versions.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from regulatrace.db.models import Document
from regulatrace.db.session import get_db
from regulatrace.ingestion.chunker import chunk_pages
from regulatrace.ingestion.metadata import compute_file_hash
from regulatrace.ingestion.parser import parse_pdf
from regulatrace.privacy import redact_for_logging
from regulatrace.vectorstore import (
    delete_document_chunks,
    get_qdrant_client,
    init_collection,
    upsert_chunks,
)


@dataclass
class IngestStats:
    """Statistics tracked during document ingestion pipeline execution."""

    documents_scanned: int = 0
    new_documents: int = 0
    updated_documents: int = 0
    skipped_documents: int = 0
    chunks_indexed: int = 0


def ingest_documents(
    path: str | Path = "data/raw",
    session: Session | None = None,
    qdrant_client: QdrantClient | None = None,
    collection_name: str | None = None,
    batch_size: int | None = None,
) -> IngestStats:
    """Run idempotent document ingestion against the file directory."""
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Ingestion directory not found: {raw_path}")

    # Gather PDF files deterministically
    pdf_files = sorted(
        {
            p.resolve(): p for p in list(raw_path.glob("*.pdf")) + list(raw_path.glob("*.PDF"))
        }.values()
    )

    stats = IngestStats(documents_scanned=len(pdf_files))
    q_client = qdrant_client or get_qdrant_client()
    init_collection(client=q_client, collection_name=collection_name)

    db_iter = [session] if session is not None else get_db()
    for db in db_iter:
        for pdf_path in pdf_files:
            filename = pdf_path.name
            content_hash = compute_file_hash(pdf_path)

            stmt = select(Document).where(Document.source_file == filename)
            doc = db.scalars(stmt).first()

            # Skip unchanged documents when SHA-256 hash matches existing registry record
            if doc is not None and doc.content_hash == content_hash:
                stats.skipped_documents += 1
                continue

            # Purge prior Qdrant chunks if document was modified before re-indexing
            if doc is not None:
                delete_document_chunks(
                    source_file=filename,
                    client=q_client,
                    collection_name=collection_name,
                )

            pages = parse_pdf(str(pdf_path))
            chunks = chunk_pages(pages, source_file=filename)
            chunk_count = len(chunks)

            if chunks:
                upsert_chunks(
                    chunks=chunks,
                    client=q_client,
                    collection_name=collection_name,
                    batch_size=batch_size,
                )

            now = datetime.now(timezone.utc)
            if doc is None:
                new_doc = Document(
                    source_file=filename,
                    content_hash=content_hash,
                    status="active",
                    version=1,
                    ingested_at=now,
                    chunk_count=chunk_count,
                )
                db.add(new_doc)
                stats.new_documents += 1
            else:
                doc.content_hash = content_hash
                doc.version += 1
                doc.chunk_count = chunk_count
                doc.ingested_at = now
                stats.updated_documents += 1

            db.commit()
            stats.chunks_indexed += chunk_count

    return stats


def print_summary(stats: IngestStats) -> None:
    """Format and print terminal summary of ingestion execution."""
    print(f"Documents scanned: {stats.documents_scanned}")
    print(f"New documents: {stats.new_documents}")
    print(f"Updated documents: {stats.updated_documents}")
    print(f"Skipped documents: {stats.skipped_documents}")
    print(f"Chunks indexed: {stats.chunks_indexed}")


def main() -> None:
    """CLI entry point for python -m regulatrace.ingest."""
    parser = argparse.ArgumentParser(description="Regulatrace document ingestion CLI")
    parser.add_argument(
        "--path",
        type=str,
        default="data/raw",
        help="Path to directory containing PDF documents (default: data/raw)",
    )
    args = parser.parse_args()

    try:
        stats = ingest_documents(path=args.path)
        print_summary(stats)
    except Exception as e:
        print(redact_for_logging(f"Ingestion error: {e}"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
