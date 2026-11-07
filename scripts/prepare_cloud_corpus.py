"""Prepare checksummed BGE-small vectors once, including recorded OCR for scanned PDFs."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from regulatrace.config import get_settings
from regulatrace.ingestion.chunker import chunk_pages
from regulatrace.ingestion.metadata import get_document_metadata, load_manifest
from regulatrace.ingestion.parser import ParsedPage, parse_pdf
from regulatrace.model_worker import EMBEDDING_REVISION
from regulatrace.prepare_index import validate_corpus
from regulatrace.vectorstore import generate_point_id

MODEL = "BAAI/bge-small-en-v1.5"


def prepare(root: Path, output: Path) -> dict:
    from sentence_transformers import SentenceTransformer

    manifest = root / "data/manifest.csv"
    raw = root / "data/raw"
    validate_corpus(raw, manifest)
    records = load_manifest(manifest)
    chunks, documents = [], []
    for name in sorted(records):
        pdf = raw / name
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        pages = parse_pdf(pdf)
        method = "pdf_text"
        if not any(page.text.strip() for page in pages):
            sidecar = root / "data/ocr" / (Path(name).stem + ".json")
            ocr = json.loads(sidecar.read_text())
            if ocr["source_file"] != name or ocr["source_sha256"] != digest:
                raise ValueError("OCR does not match the source PDF")
            if [p["page_number"] for p in ocr["pages"]] != list(range(1, len(pages) + 1)):
                raise ValueError("OCR page numbers do not match the source PDF")
            pages = [ParsedPage(**p, is_low_text_page=False) for p in ocr["pages"]]
            method = ocr["method"]
        metadata = get_document_metadata(name, manifest_records=records)
        parts = chunk_pages(pages, name, doc_metadata=metadata)
        if not parts:
            raise ValueError(f"No usable chunks for {name}")
        for part in parts:
            if method != "pdf_text":
                part.metadata["text_extraction"] = method
        chunks.extend(parts)
        documents.append(
            {
                "source_file": name,
                "sha256": digest,
                "pages": len(pages),
                "chunks": len(parts),
                "text_extraction": method,
                "metadata": metadata,
            }
        )
        print(f"Parsed {len(documents)}/{len(records)} documents, {len(chunks)} chunks", flush=True)
    model = SentenceTransformer(MODEL, revision=EMBEDDING_REVISION, device="cpu")
    if model.get_embedding_dimension() != 384:
        raise ValueError("Unexpected embedding dimension")
    revision = EMBEDDING_REVISION
    output.mkdir(parents=True, exist_ok=True)
    target = output / "points.jsonl.gz"
    ids = set()
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            vectors = model.encode(
                [c.text for c in batch],
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            for chunk, vector in zip(batch, vectors, strict=True):
                metadata = chunk.to_metadata().model_dump()
                point_id = generate_point_id(metadata["doc_id"], metadata["chunk_index"])
                if point_id in ids:
                    raise ValueError("Duplicate chunk identifier")
                ids.add(point_id)
                payload = {
                    "text": chunk.text,
                    "source_file": chunk.metadata["source_file"],
                    **metadata,
                }
                if "text_extraction" in chunk.metadata:
                    payload["text_extraction"] = chunk.metadata["text_extraction"]
                record = {"id": point_id, "vector": vector.tolist(), "payload": payload}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"Embedded {min(start + 32, len(chunks))}/{len(chunks)} chunks", flush=True)
    settings = get_settings()
    report = {
        "format": 1,
        "model": MODEL,
        "model_revision": revision,
        "dimension": 384,
        "normalized": True,
        "point_count": len(ids),
        "document_count": len(documents),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "points_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(ids)} vectors for {len(documents)} manifest PDFs", flush=True)
    return report


def ocr_scanned(root: Path) -> None:
    """Operator-only explicit OCR; original PDF bytes are never rewritten."""
    import pymupdf

    folder = root / "data/ocr"
    folder.mkdir(parents=True, exist_ok=True)
    for name in load_manifest(root / "data/manifest.csv"):
        path = root / "data/raw" / name
        with pymupdf.open(path) as doc:
            if any(page.get_text().strip() for page in doc):
                continue
            pages = []
            with tempfile.TemporaryDirectory() as temporary:
                for index, page in enumerate(doc, 1):
                    image = Path(temporary) / f"page-{index}.png"
                    page.get_pixmap(dpi=240).save(image)
                    text = subprocess.check_output(
                        ["tesseract", str(image), "stdout", "-l", "eng"], text=True
                    )
                    if not text.strip():
                        raise ValueError("OCR produced no text")
                    pages.append({"page_number": index, "text": text})
            report = {
                "source_file": name,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "method": "tesseract english, 240 dpi",
                "pages": pages,
                "engine": subprocess.check_output(
                    ["tesseract", "--version"], text=True
                ).splitlines()[0],
            }
            (folder / (Path(name).stem + ".json")).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/cloud-corpus"))
    parser.add_argument("--ocr-scanned", action="store_true")
    args = parser.parse_args()
    if args.ocr_scanned:
        ocr_scanned(args.root)
    prepare(args.root, args.output)
