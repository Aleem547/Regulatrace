"""Unit tests for Regulatrace LLM and embedding factories."""

import json
import os
from unittest.mock import Mock, patch

import numpy as np
import pytest

from regulatrace.config import Settings
from regulatrace.llm import (
    GeminiLLM,
    GroqLLM,
    MockLLM,
    compute_cosine_similarity,
    demonstrate_embedding_similarity,
    get_embeddings,
    get_llm,
)


def test_mock_llm_generation():
    """Verify MockLLM returns simulated text without network calls."""
    llm = MockLLM(model="test-model")
    res = llm.generate("Hello world")
    assert "[MockLLM:test-model]" in res
    assert "Hello world" in res


def test_get_llm_mock_provider():
    """Verify get_llm instantiates MockLLM when provider is set to 'mock'."""
    settings = Settings(llm_provider="mock", llm_model="test-mock")
    llm = get_llm(settings)
    assert isinstance(llm, MockLLM)
    assert llm.model == "test-mock"


def test_get_llm_provider_switching():
    """Verify get_llm dynamically switches provider based on settings."""
    from regulatrace.config import get_settings

    get_settings.cache_clear()
    with patch.dict(
        os.environ, {"REGULATRACE_LLM_PROVIDER": "mock", "REGULATRACE_LLM_MODEL": "custom-mock"}
    ):
        llm = get_llm()
        assert isinstance(llm, MockLLM)
        assert llm.model == "custom-mock"
    get_settings.cache_clear()


@pytest.mark.parametrize("provider", ["unsupported_provider_xyz", "openai", "groqcloud"])
def test_get_llm_unsupported_provider(provider):
    """Verify get_llm raises ValueError for unsupported provider."""
    settings = Settings(_env_file=None, llm_provider=provider)
    with pytest.raises(ValueError, match="Unsupported LLM provider") as caught:
        get_llm(settings)
    assert "'groq'" in str(caught.value)


@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_gemini_default_transport_and_timeout(api_key):
    """Use SDK defaults for address selection, with a finite SDK request timeout."""
    genai = pytest.importorskip("google.genai")
    with patch.object(genai, "Client") as client_factory:
        GeminiLLM(api_key=api_key)
    kwargs = client_factory.call_args.kwargs
    assert kwargs["http_options"].timeout == 60_000
    assert kwargs["http_options"].client_args is None
    if api_key:
        assert kwargs["api_key"] == api_key
    else:
        assert "api_key" not in kwargs


def test_gemini_ipv4_transport_and_timeout():
    genai = pytest.importorskip("google.genai")
    with (
        patch.object(genai, "Client") as client_factory,
        patch("httpx.HTTPTransport") as transport_factory,
    ):
        GeminiLLM(api_key="test-key", force_ipv4=True)
    transport_factory.assert_called_once_with(local_address="0.0.0.0")
    options = client_factory.call_args.kwargs["http_options"]
    assert options.client_args == {"transport": transport_factory.return_value}
    assert options.timeout == 60_000
    assert options.retry_options is None


@pytest.mark.parametrize("provider", ["gemini", "google"])
@pytest.mark.parametrize("force_ipv4", [False, True])
def test_gemini_factory_preserves_model_and_passes_ipv4(provider, force_ipv4):
    settings = Settings(
        _env_file=None,
        llm_provider=provider,
        llm_model="configured-gemini-model",
        gemini_api_key="test-key",
        gemini_force_ipv4=force_ipv4,
    )
    with patch("regulatrace.llm.GeminiLLM") as gemini_factory:
        assert get_llm(settings) is gemini_factory.return_value
    gemini_factory.assert_called_once_with(
        model="configured-gemini-model", api_key="test-key", force_ipv4=force_ipv4
    )


@pytest.mark.parametrize("temperature", [None, 0])
def test_gemini_generate_sdk_request_timeout_without_network(temperature):
    """Verify the SDK applies the timeout to real request construction, without network I/O."""
    pytest.importorskip("google.genai")
    import httpx

    def respond(request):
        assert request.extensions["timeout"] == {
            "connect": 60.0,
            "read": 60.0,
            "write": 60.0,
            "pool": 60.0,
        }
        generation_config = json.loads(request.content).get("generationConfig", {})
        if temperature is None:
            assert "temperature" not in generation_config
        else:
            assert generation_config["temperature"] == 0
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Mock response"}]}}]},
        )

    with patch("httpx.HTTPTransport", return_value=httpx.MockTransport(respond)):
        llm = GeminiLLM(api_key="test-key", force_ipv4=True)
    try:
        kwargs = {} if temperature is None else {"temperature": temperature}
        assert llm.generate("test prompt", **kwargs) == "Mock response"
    finally:
        llm.client.close()


@pytest.mark.parametrize("as_model", [False, True])
def test_gemini_explicit_temperature_preserves_existing_config(as_model):
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    original = {"temperature": 0.8, "max_output_tokens": 64}
    config = types.GenerateContentConfig(**original) if as_model else original.copy()
    with patch.object(genai, "Client") as client_factory:
        llm = GeminiLLM(api_key="test-key")
    generate = client_factory.return_value.models.generate_content
    generate.return_value.text = "OK"

    assert llm.generate("Question", temperature=0, config=config) == "OK"

    generate.assert_called_once()
    request = generate.call_args.kwargs
    assert (
        "temperature" not in request
    )  # Gemini accepts it inside config, not as a top-level option.
    assert request["config"].temperature == 0
    assert request["config"].max_output_tokens == 64
    assert (config.model_dump(exclude_none=True) if as_model else config) == original


@pytest.fixture
def groq_sdk():
    groq = pytest.importorskip("groq")
    with patch.object(groq, "Groq") as client_factory:
        yield client_factory


@pytest.mark.parametrize("model, temperature", [("configured-model", 0.2), ("qwen/qwen3.8-27b", 0)])
def test_groq_initialization_and_generate(groq_sdk, model, temperature):
    llm = GroqLLM(model=model, api_key="test-groq-key")
    assert llm.model == model
    assert llm.client is groq_sdk.return_value
    groq_sdk.assert_called_once_with(api_key="test-groq-key", timeout=60.0, max_retries=0)
    create = llm.client.chat.completions.create
    text = "  Supported answer [Source 1].\n"
    create.return_value.choices = [Mock(message=Mock(content=text))]
    prompt = "Question and evidence containing {literal braces}"

    assert llm.generate(prompt, temperature=temperature, max_completion_tokens=64) == text

    create.assert_called_once_with(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=64,
    )


@pytest.mark.parametrize("provider", ["groq", " GROQ "])
def test_groq_factory_uses_configured_model_and_secret(provider):
    settings = Settings(
        _env_file=None,
        llm_provider=provider,
        llm_model="configured-groq-model",
        groq_api_key="test-groq-key",
    )
    with patch("regulatrace.llm.GroqLLM") as client_factory:
        assert get_llm(settings) is client_factory.return_value
    client_factory.assert_called_once_with(model="configured-groq-model", api_key="test-groq-key")


def test_effective_groq_settings_select_matching_model_without_network(tmp_path, monkeypatch):
    from regulatrace.config import get_settings

    (tmp_path / ".env").write_text(
        "REGULATRACE_LLM_PROVIDER=groq\n"
        "REGULATRACE_LLM_MODEL=llama-3.3-70b-versatile\n"
        "REGULATRACE_GROQ_API_KEY=synthetic-test-key\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for name in ("REGULATRACE_LLM_PROVIDER", "REGULATRACE_LLM_MODEL", "REGULATRACE_GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    try:
        with patch("regulatrace.llm.GroqLLM") as client_factory:
            assert get_llm() is client_factory.return_value
        client_factory.assert_called_once_with(
            model="llama-3.3-70b-versatile", api_key="synthetic-test-key"
        )
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("api_key", [None, "", " \n\t "])
def test_groq_missing_credentials_rejected_without_sdk_environment_fallback(api_key):
    # Synthetic keys only; neither .env nor local credentials are read.
    settings = Settings(
        _env_file=None, llm_provider="groq", llm_model="configured-model", groq_api_key=api_key
    )
    with (
        patch.dict(os.environ, {"GROQ_API_KEY": "unrelated-test-key"}),
        pytest.raises(ValueError, match="set REGULATRACE_GROQ_API_KEY"),
    ):
        get_llm(settings)


@pytest.mark.parametrize("model", ["", " \n "])
def test_groq_missing_model_rejected(model):
    with pytest.raises(ValueError, match="set REGULATRACE_LLM_MODEL"):
        GroqLLM(model=model, api_key="test-groq-key")


@pytest.mark.parametrize(
    "choices",
    [
        [],
        None,
        [Mock(message=None)],
        [Mock(message=Mock(content=None))],
        [Mock(message=Mock(content=""))],
        [Mock(message=Mock(content=" \n\t "))],
        [Mock(message=Mock(content=123))],
    ],
)
def test_groq_generate_unusable_response_raises(groq_sdk, choices):
    llm = GroqLLM(model="configured-model", api_key="test-groq-key")
    llm.client.chat.completions.create.return_value.choices = choices
    with pytest.raises(RuntimeError, match="Groq returned no"):
        llm.generate("Question")


def test_groq_empty_content_reports_completion_exhaustion(groq_sdk):
    llm = GroqLLM(model="openai/gpt-oss-120b", api_key="test-groq-key")
    response = llm.client.chat.completions.create.return_value
    response.choices = [Mock(message=Mock(content=None), finish_reason="length")]
    response.usage = Mock(
        completion_tokens=1024,
        completion_tokens_details=Mock(reasoning_tokens=1024),
    )

    with pytest.raises(
        RuntimeError,
        match="finish_reason=length.*completion_tokens=1024.*reasoning_tokens=1024.*max_completion_tokens=1024",
    ):
        llm.generate("Question", max_completion_tokens=1024)

    llm.client.chat.completions.create.assert_called_once()


def test_groq_generate_sdk_errors_propagate_without_retry(groq_sdk):
    llm = GroqLLM(model="configured-model", api_key="test-groq-key")
    failure = RuntimeError("Service unavailable")
    create = llm.client.chat.completions.create
    create.side_effect = failure
    with pytest.raises(RuntimeError, match="Service unavailable") as caught:
        llm.generate("Question")
    assert caught.value is failure
    create.assert_called_once()


def test_groq_generate_rejects_streaming_without_request(groq_sdk):
    llm = GroqLLM(model="configured-model", api_key="test-groq-key")
    with pytest.raises(ValueError, match="does not support streaming"):
        llm.generate("Question", stream=True)
    llm.client.chat.completions.create.assert_not_called()


def test_manual_cosine_similarity_math():
    """Verify compute_cosine_similarity adheres to manual mathematical formula."""
    # Identical vectors -> 1.0
    u = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(compute_cosine_similarity(u, u), rel=1e-5) == 1.0

    # Orthogonal vectors -> 0.0
    v = np.array([-2.0, 1.0, 0.0])  # dot product = -2 + 2 + 0 = 0
    assert pytest.approx(compute_cosine_similarity(u, v), abs=1e-6) == 0.0

    # Opposing vectors -> -1.0
    w = np.array([-1.0, -2.0, -3.0])
    assert pytest.approx(compute_cosine_similarity(u, w), rel=1e-5) == -1.0

    # Zero vector handling -> 0.0
    z = np.array([0.0, 0.0, 0.0])
    assert compute_cosine_similarity(u, z) == 0.0


def test_embedding_factory_and_demonstration():
    """Verify BGE embedding model loads and semantic similarity holds."""
    model = get_embeddings("BAAI/bge-small-en-v1.5")
    assert model is not None

    scores = demonstrate_embedding_similarity(model)
    sim_ab = scores["sim(A, B) [RBI circular vs Reserve Bank guidance]"]
    sim_ac = scores["sim(A, C) [RBI circular vs football]"]

    # Semantic check: regulatory sentences must score significantly higher than football
    assert sim_ab > sim_ac
    assert sim_ab > 0.65
    assert sim_ac < 0.50
