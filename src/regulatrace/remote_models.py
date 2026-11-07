"""Bounded HTTPS model calls without installing Torch in the Vercel runtime."""

import json
import math
from time import monotonic
from urllib.parse import urlsplit

import httpx

from regulatrace.config import get_settings
from regulatrace.model_worker import EMBEDDING_REVISION
from regulatrace.retrieval import _validate_retrieval_inputs

RERANK_MODEL = "BAAI/bge-reranker-base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class ModelServiceUnavailable(Exception):
    """Safe failure without provider responses, question text or credentials."""


def call_worker(payload: dict, *, transport=None) -> dict:
    settings = get_settings()
    if not settings.model_worker_key or len(settings.model_worker_key.get_secret_value()) < 32:
        raise ModelServiceUnavailable("The model worker is not configured.")
    target = settings.model_worker_url.rstrip("/")
    parsed = urlsplit(target)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".hf.space")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or parsed.port not in (None, 443)
    ):
        raise ModelServiceUnavailable("The model worker URL is invalid.")
    body = {**payload, "key": settings.model_worker_key.get_secret_value()}
    if len(json.dumps(body).encode("utf-8")) > 128000:
        raise ValueError("Model request exceeds its size limit.")
    headers = {}
    if settings.hf_token:
        headers["Authorization"] = "Bearer " + settings.hf_token.get_secret_value()
    path = target + "/gradio_api/call/model_worker"
    deadline = monotonic() + 75
    try:
        with httpx.Client(
            timeout=httpx.Timeout(70, connect=8),
            headers=headers,
            follow_redirects=False,
            transport=transport,
        ) as client:
            submitted = client.post(path, json={"data": [body]})
            submitted.raise_for_status()
            event_id = submitted.json().get("event_id")
            if not isinstance(event_id, str) or not event_id.isalnum() or len(event_id) > 128:
                raise ValueError("Invalid worker event identifier")
            with client.stream("GET", path + "/" + event_id, timeout=15) as response:
                response.raise_for_status()
                event, data, size = "", [], 0
                for line in response.iter_lines():
                    if monotonic() > deadline:
                        raise ValueError("Worker deadline exceeded")
                    size += len(line)
                    if size > 2000000:
                        raise ValueError("Worker response exceeds its limit")
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line:
                        if event == "error":
                            raise ValueError("Worker rejected the request")
                        if event == "complete":
                            values = json.loads("\n".join(data))
                            if not isinstance(values, list) or len(values) != 1:
                                raise ValueError("Invalid worker output count")
                            result = values[0]
                            if (
                                not isinstance(result, dict)
                                or type(result.get("protocol")) is not int
                                or result["protocol"] != 1
                            ):
                                raise ValueError("Invalid worker protocol")
                            return result
                        event, data = "", []
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        raise ModelServiceUnavailable(
            "The model service is busy, unavailable or has exhausted its free GPU quota."
        ) from None
    raise ModelServiceUnavailable("The model service did not complete its response.")


def remote_rerank(question: str, candidates: list[dict], k: int = 5) -> list[dict]:
    _validate_retrieval_inputs(question, k)
    if not isinstance(candidates, list) or len(candidates) > 20:
        raise ValueError("At most 20 retrieval candidates are supported.")
    if not candidates:
        return []
    if len(question) > 1000 or any(
        not isinstance(c, dict)
        or not isinstance(c.get("text"), str)
        or not c["text"].strip()
        or len(c["text"]) > 4000
        for c in candidates
    ):
        raise ValueError("The model input exceeds its text limits.")
    result = call_worker(
        {"action": "rerank", "question": question, "texts": [c["text"] for c in candidates]}
    )
    scores = result.get("scores")
    if (
        result.get("model") != RERANK_MODEL
        or not isinstance(scores, list)
        or len(scores) != len(candidates)
        or any(
            isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s)
            for s in scores
        )
    ):
        raise ModelServiceUnavailable("The model service returned invalid ranking scores.")
    ranked = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores, strict=True)]
    return sorted(ranked, key=lambda c: c["rerank_score"], reverse=True)[:k]


class RemoteEmbedder:
    """The original BGE-small embedding interface, hosted by the authenticated worker."""

    def get_embedding_dimension(self):
        return 384

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, sentences, **kwargs):
        import numpy as np

        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        vectors = []
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            result = call_worker({"action": "embed", "texts": batch})
            values = result.get("vectors")
            if (
                result.get("model") != EMBEDDING_MODEL
                or result.get("revision") != EMBEDDING_REVISION
                or not isinstance(values, list)
            ):
                raise ModelServiceUnavailable("The model service returned invalid embeddings.")
            array = np.asarray(values, dtype=float)
            if array.shape != (len(batch), 384) or not np.isfinite(array).all():
                raise ModelServiceUnavailable("The model service returned invalid embeddings.")
            vectors.extend(array.tolist())
        array = np.asarray(vectors, dtype=float).reshape((-1, 384))
        return array[0] if single else array
