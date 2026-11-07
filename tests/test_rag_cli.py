"""Offline tests for RAG command-line behavior."""

import io
from unittest.mock import Mock

import pytest

from regulatrace import rag
from regulatrace.reranker import ABSTENTION_MESSAGE


@pytest.fixture
def cli_dependencies(monkeypatch, fake_llm):
    search = Mock(return_value=[])
    factory = Mock(return_value=fake_llm)
    monkeypatch.setattr(rag, "retrieve_bm25", search)
    monkeypatch.setattr(rag, "get_llm", factory)
    return search, factory


@pytest.mark.parametrize("k_args", [[], ["--k", "5"]])
def test_cli_answer_and_source_output(monkeypatch, capsys, fake_llm, cli_dependencies, k_args):
    search, factory = cli_dependencies
    chunks = [
        {
            "text": "First evidence",
            "source_file": "circular.pdf",
            "page": 3,
            "metadata": {
                "title": "Regulatory circular",
                "doc_id": "same-document",
                "chunk_index": 0,
                "circular_no": "RBI/test",
                "source_url": "https://example.com/circular",
            },
        },
        {"text": " "},
        {
            "text": "Second evidence",
            "source_file": "circular.pdf",
            "page": 4,
            "metadata": {"doc_id": "same-document", "chunk_index": 1},
        },
    ]
    search.return_value = chunks
    question = "By when should banks submit the monthly NRD-CSR R012 return?"
    monkeypatch.setattr("sys.argv", ["rag", "--query", question, *k_args])

    rag.main()

    output = capsys.readouterr()
    search.assert_called_once_with(question, k=20)
    factory.assert_called_once_with()
    fake_llm.generate.assert_called_once()
    assert question in fake_llm.generate.call_args.args[0]
    assert "First evidence" in fake_llm.generate.call_args.args[0]
    assert "Second evidence" in fake_llm.generate.call_args.args[0]
    assert f"Answer:\n{fake_llm.generate.return_value}\n\nSources:\n" in output.out
    for expected in (
        "[Source 1]",
        "File: circular.pdf",
        "Page: 3",
        "Document: Regulatory circular",
        "Document ID: same-document",
        "Chunk index: 0",
        "Circular: RBI/test",
        "URL: https://example.com/circular",
        "[Source 2]",
        "Page: 4",
        "Chunk index: 1",
    ):
        assert expected in output.out
    assert output.out.index("[Source 1]") < output.out.index("[Source 2]")
    assert output.out.count("File: circular.pdf") == 2
    assert "[Source 3]" not in output.out
    assert output.err == ""


def test_cli_missing_metadata_is_not_used_as_evidence(monkeypatch, capsys, cli_dependencies):
    search, _ = cli_dependencies
    search.return_value = [{"text": "Evidence without source metadata"}]
    monkeypatch.setattr("sys.argv", ["rag", "--query", "Question"])

    rag.main()

    output = capsys.readouterr()
    assert output.out == "Answer:\nnot found in the provided documents\n\nSources:\nNo sources.\n"
    assert output.err == ""


@pytest.mark.parametrize("chunks", [[], [{"text": " \n "}, {"text": None}]])
def test_cli_empty_evidence_output(monkeypatch, capsys, fake_llm, cli_dependencies, chunks):
    search, _ = cli_dependencies
    search.return_value = chunks
    monkeypatch.setattr("sys.argv", ["rag", "--query", "Question"])

    rag.main()

    output = capsys.readouterr()
    search.assert_called_once_with("Question", k=20)
    fake_llm.generate.assert_not_called()
    assert output.out == f"Answer:\n{ABSTENTION_MESSAGE}\n\nSources:\nNo sources.\n"
    assert output.err == ""


def test_cli_escapes_characters_unsupported_by_windows_output(
    monkeypatch, fake_llm, cli_dependencies
):
    search, _ = cli_dependencies
    search.return_value = [{"text": "Evidence", "page": 1, "metadata": {"doc_id": "doc"}}]
    fake_llm.generate.return_value = "Use the T‑1 price."
    output_bytes = io.BytesIO()
    output = io.TextIOWrapper(output_bytes, encoding="cp1252", errors="strict")
    monkeypatch.setattr("sys.stdout", output)
    monkeypatch.setattr("sys.argv", ["rag", "--query", "Question"])

    rag.main()

    output.flush()
    assert b"Use the T\\u20111 price." in output_bytes.getvalue()


@pytest.mark.parametrize(
    "arguments, message",
    [
        ([], "required: --query"),
        (["--query", ""], "--query must not be empty or whitespace-only"),
        (["--query", " \n\t "], "--query must not be empty or whitespace-only"),
        (["--query", "Question", "--k", "0"], "--k must be 5"),
        (["--query", "Question", "--k", "-1"], "--k must be 5"),
        (["--query", "Question", "--k", "2"], "--k must be 5"),
        (["--query", "Question", "--k", "abc"], "invalid int value"),
        (["--query", "Question", "--k", "2.5"], "invalid int value"),
        (["--query", "Question", "--k"], "expected one argument"),
        (["--query", "Question", "--unknown"], "unrecognized arguments"),
    ],
)
def test_cli_invalid_arguments_do_not_build_chain(monkeypatch, capsys, arguments, message):
    builder = Mock(side_effect=AssertionError("Must validate arguments before building the chain"))
    monkeypatch.setattr(rag, "build_rag_chain", builder)
    monkeypatch.setattr("sys.argv", ["rag", *arguments])

    with pytest.raises(SystemExit) as caught:
        rag.main()

    assert caught.value.code == 2
    builder.assert_not_called()
    output = capsys.readouterr()
    assert message in output.err
    assert "usage:" in output.err
    assert output.out == ""


def test_cli_help_does_not_build_chain(monkeypatch, capsys):
    builder = Mock(side_effect=AssertionError("Help must not initialize the chain"))
    monkeypatch.setattr(rag, "build_rag_chain", builder)
    monkeypatch.setattr("sys.argv", ["rag", "--help"])

    with pytest.raises(SystemExit) as caught:
        rag.main()

    assert caught.value.code == 0
    builder.assert_not_called()
    output = capsys.readouterr()
    assert "--query" in output.out
    assert "--k" in output.out
    assert "fixed at 5" in output.out
    assert output.err == ""


@pytest.mark.parametrize("failure_stage", ["initialization", "retrieval", "generation"])
def test_cli_failures_are_clear_without_exposing_exception_details(
    monkeypatch, capsys, fake_llm, cli_dependencies, failure_stage
):
    search, factory = cli_dependencies
    # Synthetic sentinel only: no environment secrets are read for this test.
    secret = "TEST_ONLY_SECRET_SENTINEL"
    failure = RuntimeError(f"Provider request failed with api_key={secret}")
    if failure_stage == "initialization":
        factory.side_effect = failure
    elif failure_stage == "retrieval":
        search.side_effect = failure
    else:
        search.return_value = [{"text": "Evidence", "page": 1, "metadata": {"doc_id": "doc"}}]
        fake_llm.generate.side_effect = failure
    monkeypatch.setattr("sys.argv", ["rag", "--query", "Question"])

    with pytest.raises(SystemExit) as caught:
        rag.main()

    assert caught.value.code == 1
    output = capsys.readouterr()
    assert "RAG error: could not complete retrieval or inference" in output.err
    assert "Check LLM configuration, credentials, and retrieval/API connectivity" in output.err
    assert "Failure: RuntimeError." in output.err
    assert secret not in output.out + output.err
    assert "api_key=" not in output.out + output.err
    assert "Traceback" not in output.err
    assert output.out == ""
    if failure_stage != "generation":
        fake_llm.generate.assert_not_called()


def test_cli_reports_http_status_without_provider_error_details(monkeypatch, capsys):
    class ProviderError(Exception):
        status_code = 404

    def fail_chain(*, k):
        raise ProviderError("TEST_ONLY_SECRET_SENTINEL")

    monkeypatch.setattr(rag, "build_rag_chain", fail_chain)
    monkeypatch.setattr("sys.argv", ["rag", "--query", "Question"])

    with pytest.raises(SystemExit) as caught:
        rag.main()

    assert caught.value.code == 1
    output = capsys.readouterr()
    assert "Failure: ProviderError (HTTP 404)." in output.err
    assert "TEST_ONLY_SECRET_SENTINEL" not in output.err
