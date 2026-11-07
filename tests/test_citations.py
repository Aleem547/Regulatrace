"""Focused citation syntax tests without retrieval, database, or LLM calls."""

import pytest

from regulatrace.citations import parse_citations, validate_citations
from regulatrace.ingestion.metadata import generate_doc_id


@pytest.mark.parametrize(("opening", "closing"), [("[", "]"), ("【", "】")])
def test_full_regulatrace_document_id_is_preserved(opening, closing):
    doc_id = generate_doc_id("policy.pdf", "RBI", "RBI/test")
    assert parse_citations(f"Supported claim {opening}{doc_id} p.12{closing}.") == [
        {"doc_id": doc_id, "page": 12}
    ]


def test_multiple_citations_preserve_order_and_duplicates():
    first = generate_doc_id("first.pdf")
    second = generate_doc_id("second.pdf")
    answer = f"First [{first} p.12], second 【{second} p.1】[{first} p.12], later 【{first} p.3】."
    assert parse_citations(answer) == [
        {"doc_id": first, "page": 12},
        {"doc_id": second, "page": 1},
        {"doc_id": first, "page": 12},
        {"doc_id": first, "page": 3},
    ]


@pytest.mark.parametrize(("opening", "closing"), [("[", "]"), ("【", "】")])
@pytest.mark.parametrize(
    "answer",
    [
        "doc_id p.12",
        "[doc_id p.12",
        "doc_id p.12]",
        "[ p.12]",
        "[doc_id p.]",
        "[doc_id p.0]",
        "[doc_id p.-1]",
        "[doc_id p.1.5]",
        "[doc_id p.01]",
        "[doc_id:p.12]",
        "[doc_id, p.12]",
        "[doc_id p:12]",
        "[doc_id  p.12]",
        "[[doc_id p.12]]",
    ],
)
def test_malformed_citations_are_ignored(answer, opening, closing):
    answer = answer.replace("[", opening).replace("]", closing)
    assert parse_citations(answer) == []


@pytest.mark.parametrize(
    "answer", ["", "not found in the provided documents", "[Source 1]", "【Source 1】"]
)
def test_outputs_without_document_citations(answer):
    assert parse_citations(answer) == []


@pytest.mark.parametrize(
    "answer",
    [
        "[doc_id p.1】",
        "【doc_id p.1]",
        "【[doc_id p.1]】",
        "[【doc_id p.1】]",
        "「doc_id p.1」",
        "［doc_id p.1］",
    ],
)
def test_mismatched_nested_and_unsupported_brackets_are_ignored(answer):
    assert parse_citations(answer) == []


def test_valid_citations_survive_mixed_malformed_output():
    answer = (
        "Bad 【broken p.0】. Good [Policy_2026-1 p.2]. [Source 1] Bad [broken p:]. 【other p.7】"
    )
    assert parse_citations(answer) == [
        {"doc_id": "Policy_2026-1", "page": 2},
        {"doc_id": "other", "page": 7},
    ]


@pytest.mark.parametrize(
    ("answer", "valid", "invalid", "precision"),
    [
        ("Rule [doc p.2].", [{"doc_id": "doc", "page": 2}], [], 1.0),
        ("Rule [invented p.2].", [], [{"doc_id": "invented", "page": 2}], 0.0),
        ("Rule [doc p.3].", [], [{"doc_id": "doc", "page": 3}], 0.0),
        (
            "Rule [doc p.2]; another [other p.2].",
            [{"doc_id": "doc", "page": 2}],
            [{"doc_id": "other", "page": 2}],
            0.5,
        ),
        ("Rule 【doc p.2】.", [{"doc_id": "doc", "page": 2}], [], 1.0),
        (
            "Rule [doc p.2]【doc p.2】; another [other p.2].",
            [{"doc_id": "doc", "page": 2}, {"doc_id": "doc", "page": 2}],
            [{"doc_id": "other", "page": 2}],
            2 / 3,
        ),
        ("Uncited answer [Source 1]; malformed [doc p.0].", [], [], None),
    ],
    ids=["valid", "invented-document", "wrong-page", "mixed", "unicode", "duplicates", "none"],
)
def test_validate_citations_against_supplied_pages(answer, valid, invalid, precision):
    sources = [{"doc_id": "doc", "page": 2}]

    assert validate_citations(answer, sources) == {
        "valid_citations": valid,
        "invalid_citations": invalid,
        "citation_count": len(valid) + len(invalid),
        "citation_precision": precision,
    }
    assert sources == [{"doc_id": "doc", "page": 2}]


def test_citations_are_invalid_without_supplied_sources():
    assert validate_citations("Rule [doc p.2].", []) == {
        "valid_citations": [],
        "invalid_citations": [{"doc_id": "doc", "page": 2}],
        "citation_count": 1,
        "citation_precision": 0.0,
    }
