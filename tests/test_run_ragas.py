"""Small offline checks for the development-only RAGAS comparison."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from eval import run_ragas
from groq import RateLimitError
from pydantic import SecretStr


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    holdout = {f"H{i}" for i in range(12)}
    # No hold-out question text or labels: these records must be excluded by ID first.
    records = (
        [{"id": item} for item in holdout]
        + [
            {
                "id": f"A{i}",
                "question": f"Question {i}",
                "answerable": True,
                "verified": True,
                "reference_answer": "Reference rule",
                "gold_sources": [],
            }
            for i in range(41)
        ]
        + [{"id": f"U{i}", "answerable": False, "verified": True} for i in range(7)]
    )
    monkeypatch.setattr(run_ragas.run_retrieval, "load_golden", lambda: records)
    monkeypatch.setattr(run_ragas.run_retrieval, "load_holdout", lambda ids: holdout)
    settings = SimpleNamespace(
        llm_provider="groq",
        llm_model=run_ragas.GROQ_MODEL,
        embedding_model=run_ragas.EMBEDDING_MODEL,
        qdrant_collection="regulatrace_chunks",
        bm25_collection="regulatrace_bm25",
    )
    monkeypatch.setattr(run_ragas, "get_settings", lambda: settings)
    chunks = [
        {
            "text": f"Clause {i}",
            "score": 0.01,
            "page": 1,
            "metadata": {"doc_id": "doc", "chunk_index": i},
        }
        for i in range(2)
    ]
    dense, sparse = Mock(return_value=chunks), Mock(return_value=chunks)
    ranking = Mock(
        side_effect=lambda question, candidates, **kw: [
            {**chunk, "rerank_score": 0.8} for chunk in reversed(candidates)
        ]
    )
    monkeypatch.setattr(run_ragas, "retrieve", dense)
    monkeypatch.setattr(run_ragas, "retrieve_bm25", sparse)
    monkeypatch.setattr(run_ragas, "rerank", ranking)
    model = object()
    monkeypatch.setattr(run_ragas, "_get_reranker", Mock(return_value=model))
    llm = SimpleNamespace(generate=Mock(return_value="The rule applies [doc p.1]."))
    monkeypatch.setattr(run_ragas, "get_llm", Mock(return_value=llm))
    metrics = {
        name: SimpleNamespace(
            ascore=AsyncMock(return_value=SimpleNamespace(value=0.75, reason=None))
        )
        for name in run_ragas.METRICS
    }
    judge = AsyncMock()
    sleep = AsyncMock()
    monkeypatch.setattr(run_ragas.asyncio, "sleep", sleep)
    monkeypatch.setattr(run_ragas, "_load_metrics", Mock(return_value=(metrics, judge)))
    monkeypatch.setattr(run_ragas, "RESULTS_DIR", tmp_path / "results")
    return SimpleNamespace(
        dense=dense,
        sparse=sparse,
        ranking=ranking,
        model=model,
        llm=llm,
        metrics=metrics,
        judge=judge,
        sleep=sleep,
    )


def test_development_selection_frozen_pipelines_and_metric_inputs(evaluation):
    report = asyncio.run(run_ragas.evaluate())
    assert len(report["per_question"]) == 82
    assert {row["id"] for row in report["per_question"]} == {f"A{i}" for i in range(41)}
    assert evaluation.dense.call_args_list == [call(f"Question {i}", k=5) for i in range(41)]
    assert evaluation.sparse.call_args_list == [call(f"Question {i}", k=20) for i in range(41)]
    assert all(
        entry.kwargs == {"k": 5, "model": evaluation.model}
        for entry in evaluation.ranking.call_args_list
    )
    baseline, best = report["per_question"][:2]
    assert baseline["contexts"] == ["Clause 0", "Clause 1"]
    assert best["contexts"] == ["Clause 1", "Clause 0"]
    assert [source["chunk_index"] for source in best["sources"]] == [1, 0]
    for name, metric in evaluation.metrics.items():
        inputs = metric.ascore.call_args_list[0].kwargs
        assert inputs["user_input"] == "Question 0"
        assert ("response" in inputs) == (name in ("faithfulness", "answer_relevancy"))
        assert ("reference" in inputs) == (name in ("context_precision", "context_recall"))
        if "response" in inputs:
            assert inputs["response"] == baseline["answer"]
        if "reference" in inputs:
            assert inputs["reference"] == baseline["reference"] == "Reference rule"
        if name != "answer_relevancy":
            assert inputs["retrieved_contexts"] == baseline["contexts"]
            assert metric.ascore.call_args_list[1].kwargs["retrieved_contexts"] == best["contexts"]
        assert metric.ascore.await_count == 82
    assert evaluation.llm.generate.call_count == 82
    assert all(
        entry.kwargs == {"temperature": 0, "max_completion_tokens": 1024}
        for entry in evaluation.llm.generate.call_args_list
    )
    assert report["configuration"]["prompt"] == "v1"
    assert report["configuration"]["generation_model"] == run_ragas.GROQ_MODEL
    assert report["configuration"]["judge_model"] == "openai/gpt-oss-120b"
    assert report["configuration"]["judge_provider"] == "groq"
    evaluation.judge.__aenter__.assert_awaited_once_with()
    evaluation.judge.__aexit__.assert_awaited_once_with(None, None, None)
    assert not run_ragas.RESULTS_DIR.exists()


def test_scoring_batches_run_without_fixed_pauses_and_report_progress(evaluation, capsys):
    observed = []

    async def score(**inputs):
        observed.append((inputs["user_input"], evaluation.sleep.await_count))
        return SimpleNamespace(value=0.75, reason=None)

    evaluation.metrics["faithfulness"].ascore.side_effect = score
    asyncio.run(run_ragas.evaluate(limit=2))

    assert observed == [("Question 0", 0), ("Question 0", 0), ("Question 1", 0), ("Question 1", 0)]
    evaluation.sleep.assert_not_awaited()
    output = capsys.readouterr().out
    assert "Waiting 60 seconds" not in output
    for question_id in ("A0", "A1"):
        for configuration in ("baseline", "best"):
            for metric in run_ragas.METRICS:
                assert f"Starting {question_id} {configuration} {metric}" in output
                assert f"Succeeded {question_id} {configuration} {metric}" in output


@pytest.mark.parametrize("case", ["gate", "empty", "not_found", "api_error"])
def test_refusals_and_errors_do_not_receive_perfect_scores(evaluation, case):
    if case == "gate":
        evaluation.ranking.side_effect = lambda question, candidates, **kw: [
            {**chunk, "rerank_score": 0.4} for chunk in candidates
        ]
    elif case == "empty":
        evaluation.dense.return_value = evaluation.sparse.return_value = []
    elif case == "not_found":
        evaluation.llm.generate.return_value = run_ragas.rag._NOT_FOUND
    else:
        evaluation.llm.generate.side_effect = RuntimeError("Mock provider error")
    report = asyncio.run(run_ragas.evaluate(limit=1))
    baseline, best = report["per_question"]
    assert all(value is None for value in best["scores"].values())
    evaluation.sleep.assert_not_awaited()
    assert report["summary"]["best"]["faithfulness"] == {
        "mean": None,
        "scored": 0,
        "unavailable": 1,
    }
    if case == "gate":
        assert baseline["status"] == "generated"  # Dense score 0.01 is never thresholded.
        assert best["status"] == "gate_refusal"
        evaluation.llm.generate.assert_called_once()
    else:
        assert baseline["status"] != "generated"
        for metric in evaluation.metrics.values():
            metric.ascore.assert_not_called()
    if case == "empty":
        evaluation.llm.generate.assert_not_called()
    if case == "api_error":
        assert "Mock provider error" in best["errors"]["pipeline"]


def test_missing_scores_and_report_saving(evaluation):
    evaluation.metrics["answer_relevancy"].ascore.return_value.value = None
    evaluation.metrics["context_precision"].ascore.return_value.value = float("nan")
    evaluation.metrics["context_recall"].ascore.side_effect = RuntimeError("Mock judge failure")
    report = asyncio.run(run_ragas.evaluate(limit=1))
    for summary in report["summary"].values():
        assert summary["faithfulness"] == {"mean": 0.75, "scored": 1, "unavailable": 0}
        assert all(summary[name]["mean"] is None for name in run_ragas.METRICS[1:])
    assert set(report["per_question"][0]["errors"]) == set(run_ragas.METRICS[1:])
    path = run_ragas.save_results(report)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        run_ragas.save_results(report)


def test_limit_and_cli(evaluation, monkeypatch, capsys):
    for value in (0, 42, True):
        with pytest.raises(ValueError, match="limit"):
            asyncio.run(run_ragas.evaluate(value))
    evaluation.dense.assert_not_called()
    monkeypatch.setattr("sys.argv", ["run_ragas", "--limit", "1"])
    run_ragas.main()
    evaluation.dense.assert_called_once_with("Question 0", k=5)
    assert len(list(run_ragas.RESULTS_DIR.glob("ragas_*.json"))) == 1
    assert "Completed 1/1" in capsys.readouterr().out


def test_rate_limit_retry_delays_bounds_and_unrelated_errors(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(run_ragas.asyncio, "sleep", sleep)
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    rate_limit = RateLimitError(
        "TPM limit",
        body=None,
        response=httpx.Response(429, request=request, headers={"retry-after": "2"}),
    )
    fallback = RateLimitError("TPM limit", body=None, response=httpx.Response(429, request=request))
    wrapped = RuntimeError("Instructor wrapper")
    wrapped.__cause__ = fallback
    for factory in (Mock, AsyncMock):
        sleep.reset_mock()
        operation = factory(side_effect=[rate_limit, wrapped, "success"])
        assert asyncio.run(run_ragas._retry_429(operation)) == "success"
        assert operation.call_count == 3
        assert sleep.await_args_list == [call(3.0), call(61.0)]
    for error, attempts in ((rate_limit, 3), (ValueError("Unrelated failure"), 1)):
        sleep.reset_mock()
        operation = AsyncMock(side_effect=error)
        with pytest.raises(type(error)):
            asyncio.run(run_ragas._retry_429(operation))
        assert operation.call_count == attempts
        assert sleep.await_count == attempts - 1


def test_ragas_adapter_setup_without_requests_or_model_downloads(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    import sentence_transformers
    from groq.resources.chat.completions import AsyncCompletions
    from groq.types.chat import ChatCompletion
    from pydantic import BaseModel

    settings = SimpleNamespace(groq_api_key=SecretStr("synthetic-groq-key"))
    monkeypatch.setattr(run_ragas, "get_settings", lambda: settings)
    loader = Mock()
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", loader)
    client_factory = Mock(wraps=run_ragas.AsyncGroq)
    monkeypatch.setattr(run_ragas, "AsyncGroq", client_factory)
    generate = AsyncMock(
        return_value=ChatCompletion(
            id="mock-completion",
            object="chat.completion",
            created=0,
            model="openai/gpt-oss-120b",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"verdict": 1}'},
                }
            ],
        )
    )
    monkeypatch.setattr(AsyncCompletions, "create", generate)

    class Judgment(BaseModel):
        verdict: int

    async def check():
        metrics, client = run_ragas._load_metrics()
        close = AsyncMock(wraps=client.close)
        monkeypatch.setattr(client, "close", close)
        async with client:
            assert set(metrics) == set(run_ragas.METRICS)
            judge = metrics["faithfulness"].llm
            assert judge.is_async and judge.model == run_ragas.JUDGE_MODEL
            assert judge.provider == "groq"
            assert all(metric.llm is judge for metric in metrics.values())
            assert judge.model_args["temperature"] == 0
            assert (await judge.agenerate("Judge the supplied evidence.", Judgment)).verdict == 1
            generate.assert_awaited_once()
            assert generate.call_args.kwargs["model"] == "openai/gpt-oss-120b"
            assert generate.call_args.kwargs["temperature"] == 0
            assert generate.call_args.kwargs["max_tokens"] == 2048
            assert generate.call_args.kwargs["response_format"]["type"] == "json_schema"
            assert (
                "verdict"
                in generate.call_args.kwargs["response_format"]["json_schema"]["schema"][
                    "properties"
                ]
            )
            client_factory.assert_called_once_with(
                api_key="synthetic-groq-key", timeout=60.0, max_retries=2
            )
            loader.assert_called_once_with(run_ragas.EMBEDDING_MODEL, device=None)
        close.assert_awaited_once_with()

    asyncio.run(check())
    for missing_key in (None, SecretStr(" ")):
        settings.groq_api_key = missing_key
        with pytest.raises(ValueError, match="REGULATRACE_GROQ_API_KEY is required"):
            run_ragas._load_metrics()
    assert client_factory.call_count == loader.call_count == 1
