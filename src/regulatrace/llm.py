"""LLM and Embedding model interfaces and factories for Regulatrace.

Provides provider-agnostic LLM client factories and SentenceTransformer
embedding initialization with manual NumPy cosine similarity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from regulatrace.config import Settings, get_settings
from regulatrace.model_factories import sentence_transformer as SentenceTransformer

SAMPLE_SENTENCES = {
    "A": "RBI issued a new circular",
    "B": "The Reserve Bank released fresh guidance",
    "C": "I like playing football",
}


class LLMClient(Protocol):
    """Protocol for provider-agnostic LLM clients."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion for a prompt."""
        ...


class MockLLM:
    """Mock LLM implementation for offline local testing and development."""

    def __init__(self, model: str = "mock-model") -> None:
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a simulated response without making external network calls."""
        return f"[MockLLM:{self.model}] Response to: {prompt[:50]}"


class GeminiLLM:
    """Gemini LLM wrapper using google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        *,
        force_ipv4: bool = False,
    ) -> None:
        from google import genai
        from google.genai import types

        client_args = None
        if force_ipv4:
            import httpx

            # Scope IPv4 source binding to this client's synchronous requests.
            client_args = {"transport": httpx.HTTPTransport(local_address="0.0.0.0")}
        http_options = types.HttpOptions(timeout=60_000, client_args=client_args)

        self.model = model
        self.client = (
            genai.Client(api_key=api_key, http_options=http_options)
            if api_key
            else genai.Client(http_options=http_options)
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate completion using Gemini API."""
        if "temperature" in kwargs:
            from google.genai import types

            # Translate only an explicit request option; leave other calls unchanged.
            config = types.GenerateContentConfig.model_validate(kwargs.get("config") or {})
            kwargs["config"] = config.model_copy(update={"temperature": kwargs.pop("temperature")})
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            **kwargs,
        )
        return str(response.text)


class GroqLLM:
    """GroqCloud completion wrapper with a configured model and no automatic retries."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Groq API key is required; set REGULATRACE_GROQ_API_KEY.")
        if not model or not model.strip():
            raise ValueError("Groq model is required; set REGULATRACE_LLM_MODEL.")

        from groq import Groq

        self.model = model
        self.client = Groq(api_key=api_key, timeout=60.0, max_retries=0)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return the first completion's text, rejecting missing or empty responses."""
        if kwargs.get("stream"):
            raise ValueError("GroqLLM.generate() does not support streaming.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        # Capture provider usage in active chat measurements, independently of Langfuse.
        from regulatrace.tracing import record_usage

        record_usage(getattr(response, "usage", None))
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices.")
        choice = response.choices[0]
        message = choice.message
        text = message.content if message is not None else None
        if not isinstance(text, str) or not text.strip():
            if getattr(choice, "finish_reason", None) == "length":
                usage = getattr(response, "usage", None)
                details = getattr(usage, "completion_tokens_details", None)
                raise RuntimeError(
                    "Groq returned no usable response text (finish_reason=length, "
                    f"completion_tokens={getattr(usage, 'completion_tokens', None)}, "
                    f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)}, "
                    f"max_completion_tokens={kwargs.get('max_completion_tokens')})."
                )
            raise RuntimeError("Groq returned no usable response text.")
        return text


def get_llm(settings: Settings | None = None) -> LLMClient:
    """Instantiate an LLM client driven entirely by configuration and environment.

    Supported providers: 'mock', 'gemini' (or 'google'), 'groq'.
    """
    cfg = settings or get_settings()
    provider = cfg.llm_provider.strip().lower()

    if provider == "mock":
        return MockLLM(model=cfg.llm_model)
    elif provider in ("gemini", "google"):
        api_key = cfg.gemini_api_key.get_secret_value() if cfg.gemini_api_key else None
        return GeminiLLM(
            model=cfg.llm_model,
            api_key=api_key,
            force_ipv4=cfg.gemini_force_ipv4,
        )
    elif provider == "groq":
        api_key = cfg.groq_api_key.get_secret_value() if cfg.groq_api_key else None
        return GroqLLM(model=cfg.llm_model, api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{cfg.llm_provider}'. "
            "Supported providers: 'mock', 'gemini' (or 'google'), 'groq'."
        )


@lru_cache
def get_embeddings(model_name: str | None = None) -> SentenceTransformer:
    """Load and cache the configured embedding model using SentenceTransformer."""
    settings = get_settings()
    name = model_name or settings.embedding_model
    if settings.model_execution == "space":
        from regulatrace.remote_models import EMBEDDING_MODEL, RemoteEmbedder

        if name != EMBEDDING_MODEL:
            raise ValueError("The remote worker requires the original BGE-small model.")
        return RemoteEmbedder()
    return SentenceTransformer(name)


def compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors manually using NumPy.

    Formula: cos(u, v) = (u . v) / (||u||_2 * ||v||_2)
    DO NOT use helper utilities; implemented via explicit dot product and norms.
    """
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def demonstrate_embedding_similarity(
    model: SentenceTransformer | None = None,
) -> dict[str, float]:
    """Embed benchmark sentences and compute manual pairwise cosine similarities.

    Validates that semantically related regulatory sentences score significantly
    higher than an unrelated sports sentence.
    """
    embedder = model or get_embeddings()

    sentences = [
        SAMPLE_SENTENCES["A"],
        SAMPLE_SENTENCES["B"],
        SAMPLE_SENTENCES["C"],
    ]

    # Generate embeddings without normalizing so manual norm calculation is verified
    embeddings = embedder.encode(sentences, normalize_embeddings=False)

    emb_a = embeddings[0]
    emb_b = embeddings[1]
    emb_c = embeddings[2]

    sim_ab = compute_cosine_similarity(emb_a, emb_b)
    sim_ac = compute_cosine_similarity(emb_a, emb_c)
    sim_bc = compute_cosine_similarity(emb_b, emb_c)

    results = {
        "sim(A, B) [RBI circular vs Reserve Bank guidance]": sim_ab,
        "sim(A, C) [RBI circular vs football]": sim_ac,
        "sim(B, C) [Reserve Bank guidance vs football]": sim_bc,
    }

    if not (sim_ab > sim_ac):
        raise ValueError(
            f"Expected similarity(A, B) > similarity(A, C), got {sim_ab:.4f} <= {sim_ac:.4f}"
        )

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Regulatrace -- Embedding Cosine Similarity Demonstration")
    print("Model: BAAI/bge-small-en-v1.5 (SentenceTransformer)")
    print("=" * 60)
    for key, text in SAMPLE_SENTENCES.items():
        print(f"Sentence {key}: '{text}'")
    print("-" * 60)

    scores = demonstrate_embedding_similarity()
    for pair, score in scores.items():
        print(f"{pair}: {score:.4f}")
    print("-" * 60)
    print("Demonstration successful: semantically related sentences score significantly higher.")
