"""Public ingestion helpers, loaded lazily for the lightweight API."""

from importlib import import_module

_EXPORTS = {
    "Chunk": ("regulatrace.ingestion.chunker", "Chunk"),
    "chunk_pages": ("regulatrace.ingestion.chunker", "chunk_pages"),
    "chunk_text": ("regulatrace.ingestion.chunker", "chunk_text"),
    "get_text_splitter": ("regulatrace.ingestion.chunker", "get_text_splitter"),
    "clean_text": ("regulatrace.ingestion.cleaner", "clean_text"),
    "fix_hyphenated_line_breaks": ("regulatrace.ingestion.cleaner", "fix_hyphenated_line_breaks"),
    "normalize_unicode": ("regulatrace.ingestion.cleaner", "normalize_unicode"),
    "normalize_whitespace": ("regulatrace.ingestion.cleaner", "normalize_whitespace"),
    "canonicalize_heading": ("regulatrace.ingestion.heading_detector", "canonicalize_heading"),
    "detect_section_heading": ("regulatrace.ingestion.heading_detector", "detect_section_heading"),
    "is_subtitle_line": ("regulatrace.ingestion.heading_detector", "is_subtitle_line"),
    "split_text_into_sections": (
        "regulatrace.ingestion.heading_detector",
        "split_text_into_sections",
    ),
    "ChunkMetadata": ("regulatrace.ingestion.metadata", "ChunkMetadata"),
    "build_chunk_metadata": ("regulatrace.ingestion.metadata", "build_chunk_metadata"),
    "compute_content_hash": ("regulatrace.ingestion.metadata", "compute_content_hash"),
    "generate_doc_id": ("regulatrace.ingestion.metadata", "generate_doc_id"),
    "get_document_metadata": ("regulatrace.ingestion.metadata", "get_document_metadata"),
    "load_manifest": ("regulatrace.ingestion.metadata", "load_manifest"),
    "DEFAULT_LOW_TEXT_THRESHOLD": ("regulatrace.ingestion.parser", "DEFAULT_LOW_TEXT_THRESHOLD"),
    "EmptyPDFError": ("regulatrace.ingestion.parser", "EmptyPDFError"),
    "InvalidPDFError": ("regulatrace.ingestion.parser", "InvalidPDFError"),
    "ParsedPage": ("regulatrace.ingestion.parser", "ParsedPage"),
    "PDFParserError": ("regulatrace.ingestion.parser", "PDFParserError"),
    "parse_pdf": ("regulatrace.ingestion.parser", "parse_pdf"),
    "is_toc_page": ("regulatrace.ingestion.validators", "is_toc_page"),
    "is_valid_chunk": ("regulatrace.ingestion.validators", "is_valid_chunk"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
