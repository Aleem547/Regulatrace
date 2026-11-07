"""Canonical metadata schema and helpers for Regulatrace document chunks.

Defines ChunkMetadata Pydantic model, deterministic hash generators,
and manifest mapping utilities for the ingestion pipeline.
"""

import csv
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


class ChunkMetadata(BaseModel):
    """Canonical schema for chunk-level metadata across Regulatrace."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(description="Deterministic document identifier")
    title: str = Field(description="Document title")
    regulator: str = Field(description="Issuing regulatory body (e.g., RBI, SEBI)")
    circular_no: str | None = Field(
        default=None, description="Official circular or notification number"
    )
    issue_date: str | None = Field(default=None, description="Document issue date from manifest")
    issue_date_iso: str | None = Field(
        default=None, description="Document issue date as RFC 3339 UTC midnight"
    )
    effective_date: str | None = Field(
        default=None, description="Document effective date (un-inferred, default None)"
    )
    status: Literal["active", "superseded"] | None = Field(
        default=None, description="Document regulatory status"
    )
    supersedes: list[str] | str | None = Field(
        default=None, description="References to prior regulations superseded"
    )
    section: str | None = Field(default=None, description="Section heading or structural context")
    page: int = Field(ge=1, description="1-indexed page number within document")
    access_level: str = Field(default="public", description="Access permission level")
    source_url: str | None = Field(default=None, description="Official source URL from manifest")
    content_hash: str = Field(description="Deterministic SHA-256 hash of chunk text")
    chunk_index: int = Field(ge=0, description="Sequential index of chunk within document")


def compute_content_hash(text: str) -> str:
    """Generate deterministic SHA-256 content hash of chunk text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str | Path) -> str:
    """Generate deterministic SHA-256 hash of a file's binary contents."""
    path = Path(file_path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_doc_id(source_file: str, regulator: str = "", circular_no: str = "") -> str:
    """Generate a deterministic document ID using SHA-256."""
    norm_file = Path(source_file).name.strip().lower()
    norm_reg = regulator.strip().upper()
    norm_circ = (circular_no or "").strip().upper()
    key = f"{norm_reg}|{norm_circ}|{norm_file}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def normalize_issue_date(value: str | None) -> str:
    """Convert a manifest date to an RFC 3339 UTC-midnight timestamp."""
    if not isinstance(value, str) or not value:
        raise ValueError("issue_date is required in '%b %d, %Y' format")
    match = re.fullmatch(r"([A-Z][a-z]{2}) (0[1-9]|[12][0-9]|3[01]), ([0-9]{4})", value)
    if match is None or match.group(1) not in _MONTHS:
        raise ValueError("issue_date must be a valid date in '%b %d, %Y' format")
    try:
        parsed = date(int(match.group(3)), _MONTHS[match.group(1)], int(match.group(2)))
    except ValueError as exc:
        raise ValueError("issue_date must be a valid date in '%b %d, %Y' format") from exc
    return f"{parsed.isoformat()}T00:00:00Z"


def load_manifest(
    manifest_path: str | Path = "data/manifest.csv",
) -> dict[str, dict[str, Any]]:
    """Load corpus manifest CSV and return records indexed by filename."""
    path = Path(manifest_path)
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("file", "").strip()
            if not filename:
                continue
            issue_date = (row.get("date") or "").strip()
            try:
                issue_date_iso = normalize_issue_date(issue_date)
            except ValueError as exc:
                raise ValueError(f"Invalid issue date for {filename}: {exc}") from exc
            records[filename] = {
                "source_file": filename,
                "regulator": row.get("regulator", "").strip(),
                "title": row.get("title", "").strip(),
                "circular_no": row.get("circular_number", "").strip() or None,
                "issue_date": issue_date,
                "issue_date_iso": issue_date_iso,
                "source_url": row.get("source_url", "").strip() or None,
                "effective_date": None,
                "status": None,
                "supersedes": None,
                "access_level": "public",
            }
    return records


def get_document_metadata(
    source_file: str,
    manifest_path: str | Path = "data/manifest.csv",
    manifest_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retrieve document metadata for a given file from manifest, or return defaults."""
    records = manifest_records if manifest_records is not None else load_manifest(manifest_path)
    basename = Path(source_file).name

    if source_file in records:
        record = records[source_file].copy()
    elif basename in records:
        record = records[basename].copy()
    else:
        record = {
            "source_file": source_file,
            "regulator": "UNKNOWN",
            "title": basename,
            "circular_no": None,
            "issue_date": None,
            "issue_date_iso": None,
            "source_url": None,
            "effective_date": None,
            "status": None,
            "supersedes": None,
            "access_level": "public",
        }

    record["doc_id"] = generate_doc_id(
        source_file=record["source_file"],
        regulator=record.get("regulator", ""),
        circular_no=record.get("circular_no") or "",
    )
    return record


def build_chunk_metadata(
    chunk_text: str,
    page_number: int,
    chunk_index: int,
    section_title: str | None = None,
    doc_metadata: dict[str, Any] | None = None,
    source_file: str = "unknown",
) -> ChunkMetadata:
    """Build a validated ChunkMetadata instance for a chunk."""
    doc_info = doc_metadata or get_document_metadata(source_file)
    issue_date = doc_info.get("issue_date")
    return ChunkMetadata(
        doc_id=doc_info.get("doc_id")
        or generate_doc_id(
            source_file=doc_info.get("source_file", source_file),
            regulator=doc_info.get("regulator", ""),
            circular_no=doc_info.get("circular_no") or "",
        ),
        title=doc_info.get("title", source_file),
        regulator=doc_info.get("regulator", "UNKNOWN"),
        circular_no=doc_info.get("circular_no"),
        issue_date=issue_date,
        issue_date_iso=normalize_issue_date(issue_date) if issue_date is not None else None,
        effective_date=doc_info.get("effective_date"),
        status=doc_info.get("status"),
        supersedes=doc_info.get("supersedes"),
        section=section_title,
        page=page_number,
        access_level=doc_info.get("access_level", "public"),
        source_url=doc_info.get("source_url"),
        content_hash=compute_content_hash(chunk_text),
        chunk_index=chunk_index,
    )
