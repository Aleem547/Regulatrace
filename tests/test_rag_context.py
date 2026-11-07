"""Offline evidence, prompt, budget, and citation tests."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import tiktoken
from langchain_core.prompts import ChatPromptTemplate

from regulatrace import rag
from regulatrace.llm import GroqLLM
from regulatrace.rag import build_rag_chain, format_evidence


def test_context_and_source_record_mapping():
    chunks = [
        {
            "text": "Actual regulatory evidence.",
            "score": 0.72,
            "source_file": "circular.pdf",
            "page": 2,
            "metadata": {
                "doc_id": "document-id",
                "chunk_index": 3,
                "title": "Document title",
                "circular_no": "RBI/test",
                "source_url": "https://example.com/circular",
                "regulator": "RBI",
            },
        }
    ]
    evidence = format_evidence(chunks)
    assert evidence == {
        "context": (
            "[Source 1]\n"
            "Document: Document title\n"
            "Circular: RBI/test\n"
            "File: circular.pdf\n"
            "Page: 2\n"
            "URL: https://example.com/circular\n"
            "Text:\nActual regulatory evidence.\n"
            "[End Source 1]"
        ),
        "sources": [
            {
                "tag": "Source 1",
                "doc_id": "document-id",
                "chunk_index": 3,
                "title": "Document title",
                "source_file": "circular.pdf",
                "page": 2,
                "circular_no": "RBI/test",
                "source_url": "https://example.com/circular",
            }
        ],
    }


def test_retrieval_order_preserved_without_score_sorting():
    chunks = [
        {"text": "First evidence", "score": 0.1, "source_file": "first.pdf", "page": 4},
        {"text": "Second evidence", "score": 0.9, "source_file": "second.pdf", "page": 1},
    ]
    evidence = format_evidence(chunks)
    assert [source["source_file"] for source in evidence["sources"]] == [
        "first.pdf",
        "second.pdf",
    ]
    assert [source["page"] for source in evidence["sources"]] == [4, 1]
    assert evidence["context"].index("First evidence") < evidence["context"].index(
        "Second evidence"
    )
    assert "[End Source 1]\n\n[Source 2]" in evidence["context"]


@pytest.mark.parametrize("metadata", [{}, None, {"title": None, "source_url": None}])
def test_missing_fields_remain_none(metadata):
    evidence = format_evidence([{"text": "Body", "metadata": metadata}])
    source = evidence["sources"][0]
    assert source["tag"] == "Source 1"
    assert all(value is None for field, value in source.items() if field != "tag")
    for label in ("Document", "Circular", "File", "Page", "URL"):
        assert f"{label}: unavailable" in evidence["context"]


def test_missing_metadata_dictionary():
    evidence = format_evidence([{"text": "Body", "source_file": "known.pdf", "page": 2}])
    source = evidence["sources"][0]
    assert source["source_file"] == "known.pdf"
    assert source["page"] == 2
    assert source["title"] is None  # Do not substitute the filename as a title.
    assert source["doc_id"] is None


def test_different_chunks_from_same_document_and_page_remain_distinct():
    chunks = [
        {
            "text": text,
            "source_file": "same.pdf",
            "page": 2,
            "metadata": {"doc_id": "same-document", "chunk_index": index},
        }
        for index, text in [(3, "First chunk"), (4, "Second chunk")]
    ]
    evidence = format_evidence(chunks)
    assert [source["tag"] for source in evidence["sources"]] == ["Source 1", "Source 2"]
    assert [source["chunk_index"] for source in evidence["sources"]] == [3, 4]
    assert evidence["context"].count("File: same.pdf") == 2
    assert "First chunk" in evidence["context"]
    assert "Second chunk" in evidence["context"]


def test_empty_evidence():
    assert format_evidence([]) == {"context": "", "sources": []}


@pytest.mark.parametrize("text", [None, "", " \r\n\t ", 123, False, b"bytes", ["text"]])
def test_unusable_text_has_no_source_record(text):
    assert format_evidence([{}, {"text": text}]) == {"context": "", "sources": []}


def test_usable_chunks_receive_consecutive_tags_after_skipping():
    evidence = format_evidence(
        [{"text": None}, {"text": "First"}, {"text": " "}, {"text": "Second"}]
    )
    assert [source["tag"] for source in evidence["sources"]] == ["Source 1", "Source 2"]
    assert evidence["context"].index("First") < evidence["context"].index("Second")
    assert "[Source 3]" not in evidence["context"]


def test_text_preserved_exactly_and_input_not_mutated():
    text = "  Section 45-IA\r\n\tPenalty: ₹5,00,000.\n\n{literal braces}\n  "
    chunks = [
        {
            "text": text,
            "source_file": "legal.pdf",
            "page": 1,
            "metadata": {"title": "Legal document", "supersedes": ["old-circular"]},
        },
        {"text": " \n ", "metadata": None},
    ]
    original = deepcopy(chunks)
    evidence = format_evidence(chunks)
    assert f"Text:\n{text}\n[End Source 1]" in evidence["context"]
    assert chunks == original
    evidence["sources"][0]["title"] = "Changed returned record"
    assert chunks == original


def test_chain_preserves_literal_braces_and_document_text(monkeypatch, fake_llm):
    text = '  Regulatory text: {deadline}\r\n{"category": "NBFC"}\n  '
    chunks = [{"text": text, "page": 1, "metadata": {"title": "Document {title}", "doc_id": "doc"}}]
    original = deepcopy(chunks)
    monkeypatch.setattr(rag, "retrieve_bm25", Mock(return_value=chunks))

    result = build_rag_chain(llm=fake_llm).invoke("What does {deadline} mean?")

    prompt = fake_llm.generate.call_args.args[0]
    assert f"Text:\n{text}\n[End Source 1]" in prompt
    assert "Document: Document {title}" in prompt
    assert "QUESTION\nWhat does {deadline} mean?" in prompt
    assert result["sources"] == format_evidence(chunks)["sources"]
    assert chunks == original


def test_chain_loads_prompt_from_module_relative_path(monkeypatch, tmp_path, fake_llm):
    project = tmp_path / "project"
    prompt_dir = project / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "answer_v1.txt").write_text(
        "SYSTEM INSTRUCTIONS\nInstructions from the versioned file.\n"
        "END SYSTEM INSTRUCTIONS\nQuestion: {question}\n"
        "Context: {context}\nSource records: {sources}\n",
        encoding="utf-8",
    )
    (prompt_dir / "answer_v2.txt").write_text(
        "SYSTEM INSTRUCTIONS\nSecond version.\nEND SYSTEM INSTRUCTIONS\n"
        "Question: {question}\nContext: {context}\nSource records: {sources}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "__file__", str(project / "src" / "regulatrace" / "rag.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        rag,
        "retrieve_bm25",
        Mock(return_value=[{"text": "Evidence", "page": 1, "metadata": {"doc_id": "doc"}}]),
    )

    build_rag_chain(llm=fake_llm).invoke("Question from the user")

    prompt = fake_llm.generate.call_args.args[0]
    assert "System: SYSTEM INSTRUCTIONS\nInstructions from the versioned file." in prompt
    assert "Human: Question: Question from the user" in prompt
    assert "Context: [Source 1]" in prompt
    assert "Text:\nEvidence\n[End Source 1]" in prompt
    assert '"tag": "Source 1"' in prompt
    assert '"doc_id": "doc"' in prompt
    assert '"page": 1' in prompt
    assert "{question}" not in prompt and "{context}" not in prompt and "{sources}" not in prompt


def _prompt_size(question, chunks, version):
    path = Path(rag.__file__).resolve().parents[2] / "prompts" / f"answer_{version}.txt"
    system, human = path.read_text(encoding="utf-8").split("END SYSTEM INSTRUCTIONS", 1)
    template = ChatPromptTemplate.from_messages(
        [("system", system.strip()), ("human", human.strip())]
    )
    rendered = rag._render_prompt(template, question, format_evidence(chunks))
    return len(tiktoken.get_encoding("o200k_harmony").encode(rendered, disallowed_special=()))


def test_prompt_token_budget_and_cached_encoding():
    assert rag._TOTAL_PLANNING_TOKENS == 8704
    assert rag._MAX_PROMPT_TOKENS == 7680
    assert rag._MAX_COMPLETION_TOKENS == 1024
    assert rag._MAX_PROMPT_TOKENS + rag._MAX_COMPLETION_TOKENS == rag._TOTAL_PLANNING_TOKENS
    assert rag._prompt_encoding().name == "o200k_harmony"
    assert rag._prompt_encoding() is rag._prompt_encoding()


@pytest.mark.parametrize(
    ("version", "version_rule"),
    [
        ("v1", "Keep regulatory categories distinct"),
        ("v2", "Each cited passage must support"),
    ],
)
def test_explicit_prompt_version_selection(monkeypatch, fake_llm, version, version_rule):
    monkeypatch.setattr(
        rag,
        "retrieve_bm25",
        Mock(return_value=[{"text": "Evidence", "page": 1, "metadata": {"doc_id": "doc"}}]),
    )

    result = build_rag_chain(llm=fake_llm, prompt_version=version).invoke("Question")

    assert version_rule in fake_llm.generate.call_args.args[0]
    assert result["abstained"] is False
    assert result["sources"][0]["tag"] == "Source 1"


def test_unknown_prompt_version_rejected_before_llm_initialization(monkeypatch):
    factory = Mock(side_effect=AssertionError("LLM must not initialize"))
    monkeypatch.setattr(rag, "get_llm", factory)

    with pytest.raises(ValueError, match="prompt_version must be 'v1' or 'v2'"):
        build_rag_chain(prompt_version="v3")

    factory.assert_not_called()


def test_budget_keeps_complete_ranked_blocks_and_same_evidence_for_both_versions(
    monkeypatch, fake_llm
):
    chunks = [
        {
            "text": "First complete regulatory clause.",
            "source_file": "first.pdf",
            "page": 2,
            "metadata": {
                "doc_id": "a" * 64,
                "title": "First title",
                "source_url": "https://example.com/first",
            },
        },
        {
            "text": "Second complete table and footnote.",
            "source_file": "second.pdf",
            "page": 3,
            "metadata": {"doc_id": "b" * 64, "title": "Second title"},
        },
        {
            "text": "Oversized lower-ranked chunk. " * 1000,
            "page": 4,
            "metadata": {"doc_id": "oversized"},
        },
    ]
    question = "What do the first two sources say?"
    allowed_input = max(_prompt_size(question, chunks[:2], version) for version in ("v1", "v2"))
    monkeypatch.setattr(rag, "_MAX_PROMPT_TOKENS", allowed_input)
    monkeypatch.setattr(rag, "retrieve_bm25", Mock(return_value=chunks))

    results = [
        build_rag_chain(llm=fake_llm, prompt_version=version).invoke(question)
        for version in ("v1", "v2")
    ]

    assert results[0]["sources"] == results[1]["sources"]
    assert [source["source_file"] for source in results[0]["sources"]] == [
        "first.pdf",
        "second.pdf",
    ]
    assert results[0]["sources"][0]["doc_id"] == "a" * 64
    assert results[0]["sources"][0]["source_url"] == "https://example.com/first"
    assert results[0]["sources"][1]["page"] == 3
    for entry in fake_llm.generate.call_args_list:
        prompt = entry.args[0]
        assert (
            len(tiktoken.get_encoding("o200k_harmony").encode(prompt, disallowed_special=()))
            <= allowed_input
        )
        assert "First complete regulatory clause." in prompt
        assert "Second complete table and footnote." in prompt
        assert "Oversized lower-ranked chunk." not in prompt
        assert prompt.index("First complete regulatory clause.") < prompt.index(
            "Second complete table and footnote."
        )
        assert "[End Source 2]" in prompt


def test_budget_counts_question_and_metadata_not_only_chunk_text(monkeypatch, fake_llm):
    question = "Short question"
    chunk = {"text": "Evidence", "page": 1, "metadata": {"title": "Short title", "doc_id": "doc"}}
    allowed_input = max(_prompt_size(question, [chunk], version) for version in ("v1", "v2"))
    monkeypatch.setattr(rag, "_MAX_PROMPT_TOKENS", allowed_input)
    search = Mock(return_value=[chunk])
    monkeypatch.setattr(rag, "retrieve_bm25", search)

    assert build_rag_chain(llm=fake_llm).invoke(question)["abstained"] is False
    fake_llm.generate.reset_mock()
    assert build_rag_chain(llm=fake_llm).invoke(question + "?" * 200)["answer"] == rag._NOT_FOUND
    assert fake_llm.generate.call_count == 0

    search.return_value = [{**chunk, "metadata": {"title": "Long title " * 100, "doc_id": "doc"}}]
    assert build_rag_chain(llm=fake_llm).invoke(question)["answer"] == rag._NOT_FOUND
    fake_llm.generate.assert_not_called()


def test_no_evidence_fits_returns_exact_not_found_without_generation(monkeypatch, fake_llm):
    monkeypatch.setattr(
        rag,
        "retrieve_bm25",
        Mock(return_value=[{"text": "Evidence", "page": 1, "metadata": {"doc_id": "doc"}}]),
    )
    monkeypatch.setattr(rag, "_MAX_PROMPT_TOKENS", 1)

    result = build_rag_chain(llm=fake_llm, prompt_version="v2").invoke("Question")

    assert result == {
        "answer": "not found in the provided documents",
        "sources": [],
        "abstained": True,
        "citation_validation": {
            "valid_citations": [],
            "invalid_citations": [],
            "citation_count": 0,
            "citation_precision": None,
        },
    }
    fake_llm.generate.assert_not_called()


def test_special_token_looking_pdf_text_is_counted_as_data(monkeypatch, fake_llm):
    chunk = {"text": "<|im_start|> Regulatory evidence.", "page": 1, "metadata": {"doc_id": "doc"}}
    monkeypatch.setattr(rag, "retrieve_bm25", Mock(return_value=[chunk]))

    result = build_rag_chain(llm=fake_llm).invoke("Question")

    assert result["sources"] == format_evidence([chunk])["sources"]
    assert "<|im_start|> Regulatory evidence." in fake_llm.generate.call_args.args[0]


@pytest.mark.parametrize(
    "invalid",
    [
        {"page": 1},
        {"page": 1, "metadata": {"doc_id": None}},
        {"page": 1, "metadata": {"doc_id": " \t "}},
        {"page": 1, "metadata": {"doc_id": 123}},
        {"metadata": {"doc_id": "doc"}},
        {"page": 0, "metadata": {"doc_id": "doc"}},
        {"page": -1, "metadata": {"doc_id": "doc"}},
        {"page": True, "metadata": {"doc_id": "doc"}},
        {"page": 1.0, "metadata": {"doc_id": "doc"}},
        {"page": "1", "metadata": {"doc_id": "doc"}},
    ],
)
def test_uncitable_chunk_skips_generation(monkeypatch, fake_llm, invalid):
    monkeypatch.setattr(rag, "retrieve_bm25", Mock(return_value=[{"text": "Evidence", **invalid}]))

    result = build_rag_chain(llm=fake_llm).invoke("Question")

    assert result == {
        "answer": "not found in the provided documents",
        "sources": [],
        "abstained": True,
        "citation_validation": {
            "valid_citations": [],
            "invalid_citations": [],
            "citation_count": 0,
            "citation_precision": None,
        },
    }
    fake_llm.generate.assert_not_called()


def test_mixed_citable_chunks_keep_order_and_source_metadata(monkeypatch, fake_llm):
    first = {
        "text": "First rule",
        "source_file": "first.pdf",
        "page": 2,
        "metadata": {
            "doc_id": "first",
            "title": "First title",
            "source_url": "https://example.com/first",
            "chunk_index": 7,
        },
    }
    invalid = {
        "text": "Uncitable decoy",
        "source_file": "bad.pdf",
        "page": 0,
        "metadata": {"doc_id": "bad"},
    }
    second = {
        "text": "Second rule",
        "source_file": "second.pdf",
        "page": 3,
        "metadata": {
            "doc_id": "second",
            "title": "Second title",
            "source_url": "https://example.com/second",
            "chunk_index": 8,
        },
    }
    monkeypatch.setattr(rag, "retrieve_bm25", Mock(return_value=[first, invalid, second]))

    result = build_rag_chain(llm=fake_llm).invoke("Question")

    assert result["sources"] == format_evidence([first, second])["sources"]
    assert [source["tag"] for source in result["sources"]] == ["Source 1", "Source 2"]
    prompt = fake_llm.generate.call_args.args[0]
    assert prompt.index("First rule") < prompt.index("Second rule")
    assert "Uncitable decoy" not in prompt
    assert '"source_url": "https://example.com/second"' in prompt


def test_groq_generation_enforces_1024_completion_tokens(monkeypatch):
    groq = pytest.importorskip("groq")
    candidate = {"text": "Evidence", "page": 1, "metadata": {"doc_id": "doc"}}
    monkeypatch.setattr(rag, "retrieve_bm25", Mock(return_value=[candidate]))
    monkeypatch.setattr(rag, "rerank", Mock(return_value=[{**candidate, "rerank_score": 0.8}]))
    with patch.object(groq, "Groq") as factory:
        client = GroqLLM(model="configured-model", api_key="test-key")
        create = factory.return_value.chat.completions.create
        create.return_value.choices = [Mock(message=Mock(content="Answer [doc p.1]"))]

        result = build_rag_chain(llm=client).invoke("Question")

    assert result["abstained"] is False
    assert create.call_args.kwargs["max_completion_tokens"] == 1024
    assert create.call_args.kwargs["temperature"] == 0
    assert create.call_args.kwargs["model"] == "configured-model"
