"""Shared offline fixtures for RAG tests."""

from unittest.mock import Mock

import pytest

from regulatrace import rag


@pytest.fixture
def fake_llm(monkeypatch):
    client = Mock()
    client.generate.return_value = "Supported answer [Source 1]."
    # Each test supplies a fake client unless it explicitly exercises the factory.
    monkeypatch.setattr(rag, "get_llm", Mock(side_effect=AssertionError("Unexpected factory call")))
    monkeypatch.setattr(
        rag,
        "rerank",
        Mock(
            side_effect=lambda question, candidates, k: [
                {**candidate, "rerank_score": 0.8}
                for candidate in candidates
                if isinstance(candidate.get("text"), str) and candidate["text"].strip()
            ][:k]
        ),
    )
    return client
