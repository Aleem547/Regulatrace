"""Structure-aware recursive text chunking for regulatory documents.

Splits document pages using LangChain's RecursiveCharacterTextSplitter,
integrates section headings from the heading detector, attaches canonical metadata,
and filters low-value or noisy chunks via validators.
"""

from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from regulatrace.config import get_settings
from regulatrace.ingestion.cleaner import clean_text
from regulatrace.ingestion.heading_detector import (
    detect_section_heading,
    split_text_into_sections,
)
from regulatrace.ingestion.metadata import (
    ChunkMetadata,
    build_chunk_metadata,
    get_document_metadata,
)
from regulatrace.ingestion.parser import ParsedPage
from regulatrace.ingestion.validators import is_toc_page, is_valid_chunk


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]

    def to_metadata(self) -> ChunkMetadata:
        """Convert chunk metadata dictionary to validated ChunkMetadata instance."""
        return ChunkMetadata.model_validate(self.metadata)


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Instantiate a RecursiveCharacterTextSplitter driven by configuration."""
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def _chunk_section(
    section_title: str | None,
    body: str,
    source_file: str,
    page_number: int,
    start_chunk_id: int,
    splitter: RecursiveCharacterTextSplitter,
    doc_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Chunk a single section body and attach prepended title and metadata."""
    raw_chunks = splitter.split_text(body)
    chunks: list[Chunk] = []
    chunk_id = start_chunk_id

    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if not is_valid_chunk(raw_chunk):
            continue

        if section_title and not raw_chunk.startswith(section_title):
            chunk_text_content = f"{section_title}\n\n{raw_chunk}"
        else:
            chunk_text_content = raw_chunk

        if not is_valid_chunk(chunk_text_content):
            continue

        chunk_meta = build_chunk_metadata(
            chunk_text=chunk_text_content,
            page_number=page_number,
            chunk_index=chunk_id,
            section_title=section_title,
            doc_metadata=doc_metadata,
            source_file=source_file,
        )

        meta_dict = chunk_meta.model_dump()
        # Preserve backwards compatibility for legacy keys
        meta_dict["source_file"] = source_file
        meta_dict["page_number"] = page_number
        meta_dict["chunk_id"] = chunk_id
        meta_dict["section_title"] = section_title

        chunks.append(
            Chunk(
                text=chunk_text_content,
                metadata=meta_dict,
            )
        )
        chunk_id += 1

    return chunks


def chunk_text(
    text: str,
    source_file: str = "unknown",
    page_number: int = 1,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    section_title: str | None = None,
    start_chunk_id: int = 1,
    doc_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Perform structure-aware chunking on a single text string."""
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    cleaned = clean_text(text)
    if not cleaned.strip():
        return []

    sections, _ = split_text_into_sections(cleaned, current_section=section_title)
    if not sections and cleaned:
        sections = [(section_title, cleaned)]

    doc_info = doc_metadata or get_document_metadata(source_file)
    chunks: list[Chunk] = []
    current_id = start_chunk_id
    for sec_title, body in sections:
        sec_chunks = _chunk_section(
            section_title=sec_title,
            body=body,
            source_file=source_file,
            page_number=page_number,
            start_chunk_id=current_id,
            splitter=splitter,
            doc_metadata=doc_info,
        )
        chunks.extend(sec_chunks)
        current_id += len(sec_chunks)

    return chunks


def chunk_pages(
    pages: list[ParsedPage],
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    doc_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Perform structure-aware chunking across document pages with metadata tracking."""
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: list[Chunk] = []
    current_section: str | None = None
    next_chunk_id = 1
    doc_info = doc_metadata or get_document_metadata(source_file)

    for page in pages:
        cleaned_text = clean_text(page.text)
        if not cleaned_text.strip():
            continue

        if is_toc_page(cleaned_text):
            # Allow chunk generation for TOC if needed, but do not update current_section.
            # Headings extracted from TOC pages must not affect future pages.
            sections = [(None, cleaned_text)]
        else:
            sections, current_section = split_text_into_sections(
                cleaned_text, current_section=current_section
            )

        for sec_title, body in sections:
            sec_chunks = _chunk_section(
                section_title=sec_title,
                body=body,
                source_file=source_file,
                page_number=page.page_number,
                start_chunk_id=next_chunk_id,
                splitter=splitter,
                doc_metadata=doc_info,
            )
            all_chunks.extend(sec_chunks)
            next_chunk_id += len(sec_chunks)

    return all_chunks


__all__ = [
    "Chunk",
    "chunk_pages",
    "chunk_text",
    "detect_section_heading",
    "get_text_splitter",
    "is_toc_page",
    "is_valid_chunk",
    "split_text_into_sections",
]
