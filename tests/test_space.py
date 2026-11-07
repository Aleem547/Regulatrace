"""Optional Space regression checks using real PDF parsing and local Qdrant."""

import hashlib
import re
import runpy
import zlib
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pymupdf
import pytest

gr = pytest.importorskip("gradio", reason="optional Space dependencies")

from regulatrace import rag, space_corpus, space_ui  # noqa: E402
from regulatrace.config import Settings, get_settings  # noqa: E402


@pytest.fixture
def pdf(tmp_path):
    def create(name="regulation.pdf", text=None):
        path = tmp_path / name
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_textbox(
                pymupdf.Rect(30, 30, 550, 750),
                text
                or (
                    "Reporting requirements\nBanks must submit the quarterly reporting return "
                    "within thirty days. The return includes the reporting period and the "
                    "name of the reporting bank. These requirements apply to reporting returns."
                ),
            )
            document.save(path)
        return path

    return create


@pytest.fixture(autouse=True)
def sparse_stub(monkeypatch):
    def vectors(texts, *, query=False):
        result = []
        for text in texts:
            pairs = sorted(
                (zlib.crc32(word.encode()), float(n))
                for word, n in Counter(re.findall(r"\w+", text.lower())).items()
            )
            result.append({"indices": [p[0] for p in pairs], "values": [p[1] for p in pairs]})
        return result

    monkeypatch.setattr(space_corpus, "sparse_vectors", vectors)


def rank(question, candidates, k):
    return [{**c, "rerank_score": 0.9} for c in candidates[:k]]


def test_real_pdf_and_qdrant_keep_uploaded_provenance(pdf):
    path = pdf("473MDD1229693F0604B6D996B1DF75C81E466.PDF")
    corpus = space_corpus.prepare_corpus([path, path])
    assert len(corpus["documents"]) == 1
    results = space_corpus.retrieve_session("quarterly reporting return", corpus)
    assert results[0]["page"] == 1
    meta = results[0]["metadata"]
    assert meta["regulator"] == "USER_UPLOAD" and meta["source_url"] is None
    assert meta["access_level"] == "session"
    assert meta["doc_id"] == hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def test_separate_sessions_cannot_retrieve_each_others_documents(pdf):
    first = space_corpus.prepare_corpus([pdf("first.pdf")])
    second = space_corpus.prepare_corpus(
        [
            pdf(
                "second.pdf",
                (
                    "Fruit storage requirements\nApples and oranges must be kept in refrigerated "
                    "storage. The storage temperature must be recorded each hour "
                    "by the fruit distributor."
                ),
            )
        ]
    )
    before = deepcopy(first)
    assert space_corpus.retrieve_session("quarterly reporting", second) == []
    assert space_corpus.retrieve_session("apples oranges", first) == []
    assert space_corpus.retrieve_session("apples oranges", second)[0]["source_file"] == "second.pdf"
    assert first == before


@pytest.mark.parametrize("files", [None, [], ["file.pdf"] * 6])
def test_upload_count_is_bounded(files):
    with pytest.raises(ValueError, match="between 1 and 5"):
        space_corpus.prepare_corpus(files)


def test_html_and_oversized_files_are_rejected(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("<html>not a PDF</html>")
    with pytest.raises(ValueError, match="valid PDF"):
        space_corpus.prepare_corpus([path])
    with path.open("wb") as handle:
        handle.write(b"%PDF-1.7")
        handle.truncate(space_corpus.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="10 MiB"):
        space_corpus.prepare_corpus([path])


def test_page_limit_and_scans_are_rejected(pdf, tmp_path, monkeypatch):
    monkeypatch.setattr(space_corpus, "MAX_PAGES", 0)
    with pytest.raises(ValueError, match="300 pages"):
        space_corpus.prepare_corpus([pdf()])
    monkeypatch.setattr(space_corpus, "MAX_PAGES", 300)
    blank = tmp_path / "scan.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(blank)
    with pytest.raises(ValueError, match="OCR"):
        space_corpus.prepare_corpus([blank])


def test_session_uploads_cannot_be_used_as_public_production_corpus(pdf):
    with pytest.raises(ValueError, match="not a public"):
        space_corpus.retrieve_session(
            "reporting", space_corpus.prepare_corpus([pdf()]), public_only=True
        )


def test_core_callbacks_do_not_call_production_index(monkeypatch):
    item = {
        "text": "The reporting deadline is thirty days.",
        "page": 2,
        "source_file": "report.pdf",
        "metadata": {"doc_id": "document"},
    }
    production = Mock(side_effect=AssertionError("must not contact production"))
    monkeypatch.setattr(rag, "retrieve_bm25", production)
    search = Mock(return_value=[item])
    llm = Mock()
    llm.generate.return_value = "Thirty days [document p.2]."
    result = rag.build_rag_chain(llm=llm, retriever=search, rerank_candidates=rank).invoke(
        "Deadline?"
    )
    assert result["citation_validation"]["citation_precision"] == 1
    search.assert_called_once_with("Deadline?", k=20)
    production.assert_not_called()


def test_answer_is_grounded_without_mutating_history(pdf):
    corpus = space_corpus.prepare_corpus([pdf()])
    identifier = corpus["documents"][0]["doc_id"]
    llm = Mock()
    llm.generate.return_value = f"Banks report within thirty days [{identifier} p.1]."
    history = []
    updated, rows, status = space_ui.answer_question(
        "When is the quarterly reporting return due?",
        corpus,
        history,
        rerank_candidates=rank,
        llm=llm,
    )
    assert history == [] and len(updated) == 2
    assert rows == [["regulation.pdf", 1, identifier]]
    assert status.startswith("Citations match")


@pytest.mark.parametrize("answer", ["Uncited factual answer.", "Wrong source [foreign p.99]."])
def test_missing_or_foreign_citations_are_withheld(pdf, answer):
    llm = Mock()
    llm.generate.return_value = answer
    updated, _, status = space_ui.answer_question(
        "quarterly reporting return",
        space_corpus.prepare_corpus([pdf()]),
        [],
        rerank_candidates=rank,
        llm=llm,
    )
    assert updated[-1]["content"] != answer and status.startswith("Answer withheld")


def test_no_evidence_uses_no_gpu_and_no_llm(pdf):
    llm, gpu = Mock(), Mock()
    updated, rows, _ = space_ui.answer_question(
        "zebras quokkas",
        space_corpus.prepare_corpus([pdf()]),
        [],
        rerank_candidates=gpu,
        llm=llm,
    )
    gpu.assert_not_called()
    llm.generate.assert_not_called()
    assert not rows and "sufficient evidence" in updated[-1]["content"]


def test_followup_rewrite_is_bounded():
    llm = Mock()
    llm.generate.return_value = "When is the quarterly return due?"
    history = [
        {"role": role, "content": f"{i}: reporting return"}
        for i in range(12)
        for role in ("user", "assistant")
    ]
    before = deepcopy(history)
    assert space_ui.standalone("When is it due?", history, llm) == llm.generate.return_value
    prompt = llm.generate.call_args.args[0]
    assert '"0: reporting return"' not in prompt and '"11: reporting return"' in prompt
    assert history == before


def test_ui_is_constructed_without_loading_weights_and_serializes_events():
    demo = space_ui.create_demo(rank)
    assert len(demo.fns) == 5
    for event in demo.fns.values():
        assert event.concurrency_limit == 1 and event.concurrency_id == "space_work"
        assert event.api_visibility == "private"


def test_optional_worker_uses_the_existing_serial_gpu_queue():
    worker = Mock(return_value={"protocol": 1, "ready": True})
    demo = space_ui.create_demo(rank, model_worker=worker)
    public = [event for event in demo.fns.values() if event.api_visibility == "public"]
    assert len(demo.fns) == 6 and len(public) == 1
    assert public[0].api_name == "model_worker"
    assert public[0].concurrency_id == "space_work" and public[0].concurrency_limit == 1
    worker.assert_not_called()


def test_prefixed_secret_name_constructor_alias_and_masking(monkeypatch):
    for key in ("REGULATRACE_GROQ_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-fallback")
    assert Settings(_env_file=None).groq_api_key.get_secret_value() == "synthetic-fallback"
    monkeypatch.setenv("REGULATRACE_GROQ_API_KEY", "synthetic-new")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key.get_secret_value() == "synthetic-new"
    assert "synthetic-new" not in settings.safe_repr()
    for key in ("REGULATRACE_GROQ_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key)
    assert (
        Settings(_env_file=None, groq_api_key="synthetic-direct").groq_api_key.get_secret_value()
        == "synthetic-direct"
    )


def test_space_entry_point_registers_model_and_builds_ui_without_launch(monkeypatch):
    import sentence_transformers

    monkeypatch.delenv("REGULATRACE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("REGULATRACE_LLM_MODEL", raising=False)
    monkeypatch.setenv("REGULATRACE_GROQ_API_KEY", "synthetic-entry-key")
    get_settings.cache_clear()
    model = Mock()
    model.predict.side_effect = lambda pairs, **kwargs: [0.7] * len(pairs)
    factory = Mock(return_value=model)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", factory)
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "app.py"), run_name="entry_point_test"
    )
    assert factory.call_args.kwargs["max_length"] == 512
    assert get_settings().llm_provider == "groq"
    assert get_settings().groq_api_key.get_secret_value() == "synthetic-entry-key"
    assert len(namespace["demo"].fns) == 5
    candidate = {"text": "The reporting deadline is thirty days.", "metadata": {"doc_id": "d"}}
    output = namespace["rerank_on_gpu"]("Deadline?", [candidate])
    assert output[0]["rerank_score"] == 0.7
    get_settings.cache_clear()
