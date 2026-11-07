"""Extract document/page citations without verifying their evidence support."""

import re
from typing import Any

# Match one ASCII or Unicode bracket pair; the conditional requires matching closers.
_CITATION_PATTERN = re.compile(
    r"(?<![\[【])(?:(?P<ascii>\[)|【)"
    r"\s*(?P<doc_id>[A-Za-z0-9_-]+) p\.(?P<page>[1-9][0-9]*)\s*"
    r"(?(ascii)\]|】)(?![\]】])"
)


def parse_citations(answer: str) -> list[dict[str, str | int]]:
    """Extract [doc_id p.N] or 【doc_id p.N】 in order, retaining full IDs and duplicates.

    Malformed citations and [Source N] tags are ignored. IDs may contain ASCII
    letters, digits, underscores, or hyphens; no length limit is imposed.
    Pages use positive decimal integers without leading zeros. This checks
    syntax only, not document existence or whether evidence supports a claim.
    """
    return [
        {"doc_id": match.group("doc_id"), "page": int(match.group("page"))}
        for match in _CITATION_PATTERN.finditer(answer)
    ]


def validate_citations(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Check supplied document/page membership, not factual support for claims.

    Count every parsed occurrence, including duplicates. With no parsed citations,
    precision is None rather than treating an uncited answer as perfectly cited.
    """
    supplied = {(source["doc_id"], source["page"]) for source in sources}
    valid = []
    invalid = []
    for citation in parse_citations(answer):
        if (citation["doc_id"], citation["page"]) in supplied:
            valid.append(citation)
        else:
            invalid.append(citation)
    count = len(valid) + len(invalid)
    return {
        "valid_citations": valid,
        "invalid_citations": invalid,
        "citation_count": count,
        "citation_precision": len(valid) / count if count else None,
    }
