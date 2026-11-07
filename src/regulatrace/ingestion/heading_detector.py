"""Heading detection, normalization, and section splitting for regulatory documents."""

import re

from regulatrace.ingestion.patterns import (
    BASE_DIVISION_PATTERN,
    DIVISION_KEYWORD_REGEX,
    MAX_HEADING_LENGTH,
    MAX_HEADING_WORDS,
    SECTION_HEADING_PATTERNS,
    TITLE_LIKE_PATTERN,
    TRAILING_HEADING_SEPARATOR_PATTERN,
)


def canonicalize_heading(heading: str) -> str:
    """Normalize division dash separators (e.g. 'CHAPTER – I' -> 'CHAPTER I')."""
    return re.sub(
        rf"^({DIVISION_KEYWORD_REGEX})\s*[-–—]\s*([IVXLCDM0-9A-Za-z]+)",
        r"\1 \2",
        heading,
        flags=re.IGNORECASE,
    )


def strip_trailing_heading_separators(text: str) -> str:
    """Remove trailing heading separators (e.g. ' -', ' :', ' :-', ' –', ' —')."""
    return TRAILING_HEADING_SEPARATOR_PATTERN.sub("", text).strip()


def is_subtitle_line(text: str) -> bool:
    """Check if a line qualifies as a short title or all-caps subtitle."""
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_HEADING_LENGTH:
        return False
    if stripped.endswith((".", ";", ",")):
        return False
    words = stripped.split()
    if len(words) > MAX_HEADING_WORDS:
        return False
    if BASE_DIVISION_PATTERN.match(stripped):
        return False
    return bool(TITLE_LIKE_PATTERN.match(stripped))


def detect_section_heading(line: str) -> str | None:
    """Detect if a text line represents a regulatory section heading using structural signals."""
    stripped = line.strip()
    if not stripped:
        return None

    # Support 2-line heading blocks: e.g. "CHAPTER – I\nPRELIMINARY"
    if "\n" in stripped:
        raw_lines = [part.strip() for part in stripped.split("\n") if part.strip()]
        if len(raw_lines) == 2:
            h1 = detect_section_heading(raw_lines[0])
            if h1 and BASE_DIVISION_PATTERN.match(raw_lines[0]) and is_subtitle_line(raw_lines[1]):
                return f"{h1} - {raw_lines[1]}"
        return None

    # Strip trailing punctuation separators (- : :- – —) when checking heading validity
    candidate = strip_trailing_heading_separators(stripped)
    if not candidate or len(candidate) > MAX_HEADING_LENGTH:
        return None

    # Headings do not end with sentence-ending punctuation or commas
    if candidate.endswith((".", ";", ",")):
        return None

    words = candidate.split()
    if len(words) > MAX_HEADING_WORDS:
        return None

    # Ensure candidate does not end with dangling prepositions or introductory connectives
    if words[-1].lower().rstrip(",;:") in {
        "of",
        "to",
        "by",
        "for",
        "in",
        "and",
        "or",
        "as",
        "under",
        "namely",
    }:
        return None

    for pattern in SECTION_HEADING_PATTERNS:
        match = pattern.match(candidate)
        if match:
            heading = match.group(1).strip()
            return canonicalize_heading(heading)

    return None


def split_text_into_sections(
    text: str, current_section: str | None = None
) -> tuple[list[tuple[str | None, str]], str | None]:
    """Split text into (section_title, section_body) pairs using heading detection."""
    lines = text.split("\n")
    segments: list[tuple[str | None, list[str]]] = []
    active_section = current_section
    current_lines: list[str] = []

    i = 0
    num_lines = len(lines)
    while i < num_lines:
        line = lines[i]
        stripped = line.strip()

        # Check if line is a base division heading that might have a subtitle on the next line
        heading = detect_section_heading(stripped)
        if heading and BASE_DIVISION_PATTERN.match(stripped):
            next_idx = i + 1
            while next_idx < num_lines and not lines[next_idx].strip():
                next_idx += 1
            if next_idx < num_lines and is_subtitle_line(lines[next_idx]):
                subtitle = lines[next_idx].strip()
                heading = f"{heading} - {subtitle}"
                i = next_idx

        if heading:
            if current_lines:
                segments.append((active_section, current_lines))
                current_lines = []
            active_section = heading
        else:
            current_lines.append(line)

        i += 1

    if current_lines:
        segments.append((active_section, current_lines))

    result: list[tuple[str | None, str]] = []
    for section_title, s_lines in segments:
        body = "\n".join(s_lines).strip()
        if body:
            result.append((section_title, body))

    # If the text was only a heading with no body lines
    if not result and active_section:
        result.append((active_section, active_section))

    return result, active_section
