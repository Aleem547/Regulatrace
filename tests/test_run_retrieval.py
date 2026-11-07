"""Offline tests for development-only dense, BM25, and hybrid evaluation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from eval import run_retrieval


@pytest.fixture
def evaluation_data(tmp_path, monkeypatch):
    def record(question_id, question_type, sources):
        return {
            "id": question_id,
            "question": question_id,
            "type": question_type,
            "gold_sources": sources,
            "reference_answer": "reference",
            "answerable": bool(sources),
            "verified": True,
        }

    def source(question_id, page=1):
        return {"doc_id": question_id, "page": page}

    holdout_ids = [f"H{number:02}" for number in range(1, 13)]
    records = [record(question_id, "factoid", [source(question_id)]) for question_id in holdout_ids]
    records += [
        record("D01", "factoid", [source("D01")]),
        record("D02", "factoid", [source("D02")]),
        record("D03", "multi-section", [source("D03"), source("D03", 2)]),
        record("D04", "table_lookup", [source("D04")]),
        record("D05", "date_version", [source("D05")]),
        record("D06", "unanswerable", []),
    ]
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    holdout_path = tmp_path / "holdout_ids.txt"
    holdout_path.write_text("\n".join(holdout_ids) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_retrieval, "GOLDEN_PATH", golden_path)
    monkeypatch.setattr(run_retrieval, "HOLDOUT_PATH", holdout_path)
    monkeypatch.setattr(run_retrieval, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        run_retrieval,
        "get_settings",
        lambda: SimpleNamespace(
            embedding_model="BAAI/bge-small-en-v1.5",
            qdrant_collection="regulatrace_chunks",
            bm25_collection="regulatrace_bm25",
            hybrid_collection="regulatrace_hybrid",
        ),
    )

    def result(question_id, page=1):
        return {"metadata": source(question_id, page)}

    responses = {
        "D01": [result("D01")],
        "D02": [],
        "D03": [result("other"), result("D03")],
        "D04": [result("D04")],
        "D05": [result("other"), result("D05")],
    }
    dense = Mock(side_effect=lambda question, k: responses[question])
    bm25 = Mock(side_effect=lambda question, k: responses[question])
    hybrid = Mock(side_effect=lambda question, k, candidate_k: responses[question])
    monkeypatch.setattr(run_retrieval, "retrieve", dense)
    monkeypatch.setattr(run_retrieval, "retrieve_bm25", bm25)
    monkeypatch.setattr(run_retrieval, "retrieve_hybrid", hybrid)
    return dense, bm25, hybrid


def test_method_selection_and_frozen_holdout_exclusion(evaluation_data):
    dense, bm25, hybrid = evaluation_data

    dense_report = run_retrieval.evaluate(5)
    bm25_report = run_retrieval.evaluate(5, method="bm25")
    hybrid_report = run_retrieval.evaluate(5, method="hybrid")

    expected = [call(question_id, k=5) for question_id in ("D01", "D02", "D03", "D04", "D05")]
    assert dense.call_args_list == expected
    assert bm25.call_args_list == expected
    assert hybrid.call_args_list == [
        call(question_id, k=5, candidate_k=5) for question_id in ("D01", "D02", "D03", "D04", "D05")
    ]
    assert dense_report["method"] == "dense"
    assert dense_report["configuration"] == {
        "k": 5,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "qdrant_collection": "regulatrace_chunks",
    }
    assert dense_report["configuration_hash"] == (
        "d0429c110f2bff2d19f3c4785442decdb4ca8fb382f9d978cf6889805a9a71ff"
    )
    assert bm25_report["configuration"] == {
        "method": "bm25",
        "k": 5,
        "qdrant_collection": "regulatrace_bm25",
        "bm25_model": "qdrant/bm25",
        "sparse_vector": "bm25",
        "modifier": "idf",
    }
    assert bm25_report["configuration_hash"] != dense_report["configuration_hash"]
    assert bm25_report["configuration_hash"] == (
        "dc117fe886ba51ab30431a1d7be6b25cd5366b903e3d883056f392b68bc5b1a6"
    )
    assert hybrid_report["configuration"] == {
        "method": "hybrid",
        "k": 5,
        "candidate_k": 5,
        "qdrant_collection": "regulatrace_hybrid",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "bm25_model": "qdrant/bm25",
        "dense_vector": "dense",
        "sparse_vector": "bm25",
        "modifier": "idf",
        "rrf_smoothing": 60,
        "rrf_k": 61,
    }
    assert hybrid_report["configuration_hash"] not in {
        dense_report["configuration_hash"],
        bm25_report["configuration_hash"],
    }
    assert hybrid_report["counts"] == bm25_report["counts"]
    assert [row["id"] for row in hybrid_report["per_question"]] == [
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
    ]
    assert bm25_report["counts"] == {
        "golden_questions": 18,
        "holdout_questions": 12,
        "development_questions": 6,
        "answerable_evaluated": 5,
        "unanswerable_excluded": 1,
    }


def test_bm25_overall_and_question_type_metrics(evaluation_data):
    report = run_retrieval.evaluate(5, method="bm25")

    assert report["metrics"] == {"hit_at_k": 0.8, "recall_at_k": 0.7, "mrr": 0.6}
    assert report["metrics_by_type"] == {
        "factoid": {"count": 2, "hit_at_k": 0.5, "recall_at_k": 0.5, "mrr": 0.5},
        "multi-section": {"count": 1, "hit_at_k": 1.0, "recall_at_k": 0.5, "mrr": 0.5},
        "table_lookup": {"count": 1, "hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0},
        "date_version": {"count": 1, "hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 0.5},
    }
    assert [row["id"] for row in report["per_question"]] == ["D01", "D02", "D03", "D04", "D05"]
    assert report["per_question"][2] == {
        "id": "D03",
        "type": "multi-section",
        "hit_at_k": 1.0,
        "recall_at_k": 0.5,
        "reciprocal_rank": 0.5,
    }


def test_cli_writes_separate_bm25_json_without_touching_dense_report(
    evaluation_data, monkeypatch, capsys
):
    baseline = run_retrieval.RESULTS_DIR / "retrieval_existing_dense_baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("existing baseline\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_retrieval", "--method", "bm25", "--k", "5"])

    run_retrieval.main()

    reports = list(run_retrieval.RESULTS_DIR.glob("retrieval_bm25_*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["method"] == "bm25"
    assert saved["metrics"]["hit_at_k"] == 0.8
    assert len(saved["per_question"]) == 5
    assert baseline.read_text(encoding="utf-8") == "existing baseline\n"
    output = capsys.readouterr().out
    assert "method=bm25" in output
    assert "By question type:" in output
    assert "multi-section" in output


def test_existing_dense_report_shape_keeps_dense_filename(evaluation_data):
    report = run_retrieval.evaluate(5)
    report.pop("method")

    path = run_retrieval.save_results(report)

    assert path.name.startswith("retrieval_")
    assert not path.name.startswith("retrieval_bm25_")


def test_cli_writes_separate_hybrid_json_without_touching_baselines(
    evaluation_data, monkeypatch, capsys
):
    results_dir = run_retrieval.RESULTS_DIR
    results_dir.mkdir(parents=True)
    baseline_paths = [
        results_dir / "retrieval_existing_dense_baseline.json",
        results_dir / "retrieval_bm25_existing_baseline.json",
    ]
    for path in baseline_paths:
        path.write_text("existing baseline\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_retrieval", "--method", "hybrid"])

    run_retrieval.main()

    reports = list(results_dir.glob("retrieval_hybrid_*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["method"] == "hybrid"
    assert saved["configuration"]["candidate_k"] == 5
    assert saved["counts"]["holdout_questions"] == 12
    assert saved["counts"]["answerable_evaluated"] == 5
    assert len(saved["per_question"]) == 5
    assert saved["metrics"]["hit_at_k"] == 0.8
    assert saved["metrics_by_type"]["multi-section"]["recall_at_k"] == 0.5
    assert all(path.read_text(encoding="utf-8") == "existing baseline\n" for path in baseline_paths)
    assert "method=hybrid" in capsys.readouterr().out


def test_hybrid_depth_validation_precedes_retrieval(evaluation_data, monkeypatch, capsys):
    dense, bm25, hybrid = evaluation_data
    with pytest.raises(ValueError, match="candidate_k=5"):
        run_retrieval.evaluate(6, method="hybrid")
    dense.assert_not_called()
    bm25.assert_not_called()
    hybrid.assert_not_called()

    monkeypatch.setattr("sys.argv", ["run_retrieval", "--method", "hybrid", "--k", "6"])
    with pytest.raises(SystemExit) as caught:
        run_retrieval.main()
    assert caught.value.code == 2
    assert "candidate_k=5" in capsys.readouterr().err
    hybrid.assert_not_called()


def test_invalid_method_is_rejected_before_retrieval(evaluation_data):
    dense, bm25, hybrid = evaluation_data

    with pytest.raises(ValueError, match="method must be"):
        run_retrieval.evaluate(5, method="unsupported")

    dense.assert_not_called()
    bm25.assert_not_called()
    hybrid.assert_not_called()


@pytest.mark.parametrize("method", ["bm25_rerank", "hybrid_rerank"])
def test_rerank_evaluation_uses_20_candidates_and_original_top_five_control(
    evaluation_data, monkeypatch, method
):
    dense, bm25, hybrid = evaluation_data

    def candidates(question, **_kwargs):
        results = [
            {"text": f"Decoy {index}", "score": 0.9, "metadata": {"doc_id": "other", "page": 1}}
            for index in range(20)
        ]
        results[5] = {
            "text": "Gold page 1",
            "score": 0.2,
            "metadata": {"doc_id": question, "page": 1},
        }
        if question == "D03":
            results[6] = {
                "text": "Gold page 2",
                "score": 0.1,
                "metadata": {"doc_id": question, "page": 2},
            }
        return results

    bm25.side_effect = candidates
    hybrid.side_effect = candidates
    model = Mock()
    model.predict.side_effect = lambda pairs, **_kwargs: [
        1.0 if text.startswith("Gold") else 0.0 for _, text in pairs
    ]
    loader = Mock(return_value=model)
    monkeypatch.setattr(run_retrieval, "_get_reranker", loader)
    clock = iter([0, 0.001, 1, 1.002, 2, 2.003, 3, 3.004, 4, 4.005])
    monkeypatch.setattr(run_retrieval.time, "perf_counter", lambda: next(clock))

    report = run_retrieval.evaluate(5, method=method)

    expected_ids = ["D01", "D02", "D03", "D04", "D05"]
    assert [row["id"] for row in report["per_question"]] == expected_ids
    assert report["counts"]["holdout_questions"] == 12
    assert report["counts"]["unanswerable_excluded"] == 1
    assert report["control_metrics"]["hit_at_k"] == 0.0
    assert report["metrics"]["hit_at_k"] == 1.0
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["rerank_latency_p50_ms"] == pytest.approx(3.0)
    assert [row["rerank_latency_ms"] for row in report["per_question"]] == pytest.approx(
        [1, 2, 3, 4, 5]
    )
    assert report["per_question"][2]["control"]["recall_at_k"] == 0.0
    assert report["per_question"][2]["recall_at_k"] == 1.0
    assert report["per_question"][2]["control_sources"][0] == {
        "rank": 1,
        "doc_id": "other",
        "page": 1,
    }
    assert report["per_question"][2]["retrieved_sources"][:2] == [
        {"rank": 1, "doc_id": "D03", "page": 1},
        {"rank": 2, "doc_id": "D03", "page": 2},
    ]
    assert report["configuration"]["candidate_k"] == 20
    assert report["configuration"]["final_k"] == 5
    assert report["configuration"]["reranker_model"] == "BAAI/bge-reranker-base"
    loader.assert_called_once_with()
    assert model.predict.call_count == 5
    assert all(len(entry.args[0]) == 20 for entry in model.predict.call_args_list)
    if method == "bm25_rerank":
        assert bm25.call_args_list == [call(question, k=20) for question in expected_ids]
        assert report["configuration"]["qdrant_collection"] == "regulatrace_bm25"
        hybrid.assert_not_called()
    else:
        assert hybrid.call_args_list == [
            call(question, k=20, candidate_k=20) for question in expected_ids
        ]
        assert report["configuration"]["rrf_k"] == 61
        assert report["configuration"]["qdrant_collection"] == "regulatrace_hybrid"
        bm25.assert_not_called()
    dense.assert_not_called()


@pytest.mark.parametrize("method", ["bm25_rerank", "hybrid_rerank"])
def test_rerank_cli_saves_distinct_report_without_changing_baselines(
    evaluation_data, monkeypatch, capsys, method
):
    _, bm25, hybrid = evaluation_data
    results_dir = run_retrieval.RESULTS_DIR
    results_dir.mkdir(parents=True)
    baseline = results_dir / "retrieval_existing_baseline.json"
    baseline.write_text("unchanged\n", encoding="utf-8")
    model = Mock()
    model.predict.side_effect = lambda pairs, **_kwargs: [0.5] * len(pairs)
    monkeypatch.setattr(run_retrieval, "_get_reranker", Mock(return_value=model))

    def candidates(question, **_kwargs):
        return [{"text": "Evidence", "score": 0.2, "metadata": {"doc_id": question, "page": 1}}]

    bm25.side_effect = candidates
    hybrid.side_effect = candidates
    monkeypatch.setattr("sys.argv", ["run_retrieval", "--method", method])

    run_retrieval.main()

    reports = list(results_dir.glob(f"retrieval_{method}_*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["method"] == method
    assert saved["control_metrics"] == saved["metrics"]
    assert len(saved["per_question"]) == 5
    assert "rerank_latency_ms" in saved["per_question"][0]
    assert "rerank_latency_p50_ms" in saved
    assert baseline.read_text(encoding="utf-8") == "unchanged\n"
    output = capsys.readouterr().out
    assert f"method={method}" in output
    assert "Rerank p50 (ms)" in output


@pytest.mark.parametrize("method", ["bm25_rerank", "hybrid_rerank"])
def test_rerank_requires_final_five_before_loading_model(evaluation_data, monkeypatch, method):
    dense, bm25, hybrid = evaluation_data
    loader = Mock(side_effect=AssertionError("Invalid depth must not load model"))
    monkeypatch.setattr(run_retrieval, "_get_reranker", loader)
    with pytest.raises(ValueError, match="final k=5"):
        run_retrieval.evaluate(6, method=method)
    loader.assert_not_called()
    dense.assert_not_called()
    bm25.assert_not_called()
    hybrid.assert_not_called()


@pytest.mark.parametrize("depth", [10, 20, 30])
def test_bm25_rerank_candidate_depth_reaches_retrieval_and_report(
    evaluation_data, monkeypatch, depth
):
    _, bm25, hybrid = evaluation_data
    bm25.side_effect = lambda question, k: [
        {"text": f"Evidence {index}", "metadata": {"doc_id": question, "page": 1}}
        for index in range(k)
    ]
    model = Mock()
    model.predict.side_effect = lambda pairs, **_kwargs: [0.5] * len(pairs)
    monkeypatch.setattr(run_retrieval, "_get_reranker", Mock(return_value=model))

    report = run_retrieval.evaluate(5, method="bm25_rerank", candidate_k=depth)

    assert bm25.call_args_list == [
        call(question, k=depth) for question in ("D01", "D02", "D03", "D04", "D05")
    ]
    assert all(len(entry.args[0]) == depth for entry in model.predict.call_args_list)
    assert report["configuration"]["candidate_k"] == depth
    assert report["configuration"]["final_k"] == 5
    assert report["counts"]["answerable_evaluated"] == 5
    assert report["counts"]["holdout_questions"] == 12
    assert "rerank_latency_p50_ms" in report
    if depth == 20:
        assert report["configuration_hash"] == (
            "c0b109ae7ff2df22e98cbe4ee452f8dc653e640df51fbf09c49c784a40ff5d29"
        )
    hybrid.assert_not_called()


@pytest.mark.parametrize("depth", [None, True, 5, 15, 20.0])
def test_invalid_candidate_depth_fails_before_retrieval(evaluation_data, depth):
    dense, bm25, hybrid = evaluation_data
    with pytest.raises(ValueError, match="candidate_k must be 10, 20, or 30"):
        run_retrieval.evaluate(5, method="bm25_rerank", candidate_k=depth)
    dense.assert_not_called()
    bm25.assert_not_called()
    hybrid.assert_not_called()


def test_candidate_depth_cli_selection_and_other_method_rejection(
    evaluation_data, monkeypatch, capsys
):
    _, bm25, hybrid = evaluation_data
    bm25.side_effect = lambda question, k: [
        {"text": "Evidence", "metadata": {"doc_id": question, "page": 1}}
    ]
    model = Mock()
    model.predict.side_effect = lambda pairs, **_kwargs: [0.5] * len(pairs)
    monkeypatch.setattr(run_retrieval, "_get_reranker", Mock(return_value=model))
    monkeypatch.setattr(
        "sys.argv", ["run_retrieval", "--method", "bm25_rerank", "--candidate-k", "10"]
    )

    run_retrieval.main()

    reports = list(run_retrieval.RESULTS_DIR.glob("retrieval_bm25_rerank_*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["configuration"]["candidate_k"] == 10
    assert "candidate_k=10" in capsys.readouterr().out
    assert bm25.call_count == 5

    monkeypatch.setattr("sys.argv", ["run_retrieval", "--method", "hybrid", "--candidate-k", "10"])
    with pytest.raises(SystemExit) as caught:
        run_retrieval.main()
    assert caught.value.code == 2
    assert "only for bm25_rerank" in capsys.readouterr().err
    hybrid.assert_not_called()


def test_holdout_uses_frozen_pipeline_and_synthetic_selected_ids(evaluation_data, monkeypatch):
    dense, bm25, hybrid = evaluation_data
    path = run_retrieval.GOLDEN_PATH
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["id"] in {"H10", "H11", "H12"}:
            record.update(answerable=False, gold_sources=[], reference_answer="not found")
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    bm25.side_effect = lambda question, k: (
        []
        if question == "H12"
        else [{"text": "Evidence", "metadata": {"doc_id": question, "page": 1}}]
    )
    model = Mock()
    model.predict.side_effect = lambda pairs, **_: [0.4 if pairs[0][0] in {"H09", "H10"} else 0.8]
    loader = Mock(return_value=model)
    monkeypatch.setattr(run_retrieval, "_get_reranker", loader)
    gate = Mock(wraps=run_retrieval.decide_abstention)
    monkeypatch.setattr(run_retrieval, "decide_abstention", gate)
    clock = iter(index * 0.001 for index in range(48))
    monkeypatch.setattr(run_retrieval.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        "sys.argv", ["run_retrieval", "--split", "holdout", "--method", "bm25_rerank"]
    )

    run_retrieval.main()

    reports = list(run_retrieval.RESULTS_DIR.glob("retrieval_bm25_rerank_holdout_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert [row["id"] for row in report["per_question"]] == [
        f"H{index:02}" for index in range(1, 13)
    ]
    assert bm25.call_args_list == [call(f"H{index:02}", k=20) for index in range(1, 13)]
    assert report["configuration"]["candidate_k"] == 20
    assert report["configuration"]["final_k"] == 5
    assert report["configuration"]["reranker_model"] == "BAAI/bge-reranker-base"
    assert report["configuration"]["abstention_threshold"] == 0.60
    assert report["metrics"] == {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0}
    assert report["counts"]["correct_refusals"] == 2
    assert report["counts"]["missed_refusals"] == 1
    assert report["counts"]["wrong_refusals"] == 1
    assert report["counts"]["correct_acceptances"] == 8
    assert report["rates"] == {
        "correct_refusal_rate": pytest.approx(2 / 3),
        "wrong_refusal_rate": pytest.approx(1 / 9),
    }
    assert report["retrieval_latency_p50_ms"] == pytest.approx(1.0)
    assert report["rerank_latency_p50_ms"] == pytest.approx(1.0)
    assert report["per_question"][-1]["top_rerank_score"] is None
    assert gate.call_count == 12
    loader.assert_called_once_with()
    dense.assert_not_called()
    hybrid.assert_not_called()


def test_holdout_empty_answerability_groups_have_null_rates(evaluation_data, monkeypatch):
    _, bm25, _ = evaluation_data
    bm25.side_effect = lambda question, k: [
        {"text": "Evidence", "metadata": {"doc_id": question, "page": 1}}
    ]
    model = Mock()
    model.predict.side_effect = lambda pairs, **_: [0.8]
    monkeypatch.setattr(run_retrieval, "_get_reranker", Mock(return_value=model))
    answerable = run_retrieval.evaluate(5, method="bm25_rerank", split="holdout")
    assert answerable["rates"]["correct_refusal_rate"] is None

    path = run_retrieval.GOLDEN_PATH
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["id"].startswith("H"):
            record.update(answerable=False, gold_sources=[], reference_answer="not found")
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    bm25.side_effect = lambda question, k: []
    unanswerable = run_retrieval.evaluate(5, method="bm25_rerank", split="holdout")
    assert unanswerable["metrics"] == {"hit_at_k": None, "recall_at_k": None, "mrr": None}
    assert unanswerable["rates"] == {"correct_refusal_rate": 1.0, "wrong_refusal_rate": None}


@pytest.mark.parametrize(
    "args",
    [
        ["--split", "holdout"],
        ["--split", "holdout", "--method", "bm25_rerank", "--k", "6"],
        ["--split", "holdout", "--method", "bm25_rerank", "--candidate-k", "10"],
    ],
)
def test_holdout_rejects_overrides_before_retrieval(evaluation_data, monkeypatch, args):
    _, bm25, _ = evaluation_data
    loader = Mock(side_effect=AssertionError("Model must not load"))
    monkeypatch.setattr(run_retrieval, "_get_reranker", loader)
    monkeypatch.setattr("sys.argv", ["run_retrieval", *args])
    with pytest.raises(SystemExit) as caught:
        run_retrieval.main()
    assert caught.value.code == 2
    loader.assert_not_called()
    bm25.assert_not_called()


def test_holdout_rejects_changed_threshold(evaluation_data, monkeypatch):
    monkeypatch.setattr(run_retrieval, "ABSTENTION_THRESHOLD", 0.61)
    with pytest.raises(ValueError, match="Frozen reranker model or abstention threshold"):
        run_retrieval.evaluate(5, method="bm25_rerank", split="holdout")
