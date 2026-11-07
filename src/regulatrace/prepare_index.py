"""Prepare dense and BM25 indexes from an explicitly supplied regulatory corpus."""

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from regulatrace.bm25 import backfill_bm25, init_bm25_collection
from regulatrace.config import get_settings
from regulatrace.ingest import ingest_documents, print_summary
from regulatrace.ingestion.metadata import load_manifest


def validate_corpus(path: Path, manifest: Path) -> int:
    """Reject an incomplete corpus before initializing models or remote services."""
    records = load_manifest(manifest)
    if not records:
        raise ValueError("A nonempty corpus manifest is required.")
    present = {item.name for item in path.iterdir() if item.is_file()} if path.is_dir() else set()
    missing = set(records) - present
    if missing:
        raise ValueError(f"Missing {len(missing)} of {len(records)} manifest PDFs in the corpus.")
    unknown = {name for name in present if name.lower().endswith(".pdf")} - set(records)
    if unknown:
        raise ValueError("Corpus has PDFs absent from the manifest; review their metadata first.")
    for name in records:
        with (path / name).open("rb") as document:
            if b"%PDF-" not in document.read(1024):
                raise ValueError("A manifest file is not a PDF; do not index HTML/error pages.")
    return len(records)


def prepare_index(path: Path, manifest: Path) -> int:
    validate_corpus(path, manifest)
    # The original chunker reads this exact manifest path; reject a different file
    # rather than silently applying missing or unrelated metadata.
    if manifest.resolve() != Path("data/manifest.csv").resolve():
        raise ValueError("Run from the repository root using data/manifest.csv.")
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        cloud_inference=True,
        timeout=60,
    )
    try:
        init_bm25_collection(client=client)
        stats = ingest_documents(path=path, qdrant_client=client)
        copied = backfill_bm25(client=client)
        dense = client.count(settings.qdrant_collection, exact=True).count
        sparse = client.count(settings.bm25_collection, exact=True).count
        if not (dense > 0 and dense == sparse == copied):
            raise ValueError(
                "Index counts do not match. Inspect stale BM25 points before releasing; "
                "no existing points have been deleted automatically."
            )
        print_summary(stats)
        print(f"BM25 points copied: {copied}")
        return copied
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Regulatrace dense and BM25 indexes")
    parser.add_argument("--path", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    try:
        prepare_index(args.path, Path("data/manifest.csv"))
    except Exception:
        # SDK errors can contain URLs or credentials. Return a safe operator message.
        print("Index preparation failed. Check corpus completeness, migrations and Qdrant access.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
