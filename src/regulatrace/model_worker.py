"""A small authenticated model-only protocol, independent of uploaded documents."""

import hmac
import json
import math

RERANK_MODEL = "BAAI/bge-reranker-base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def handle_model_request(payload, *, secret, rank, embed):
    if (
        not isinstance(payload, dict)
        or not isinstance(secret, str)
        or len(secret) < 32
        or not isinstance(payload.get("key"), str)
        or not hmac.compare_digest(payload["key"], secret)
    ):
        raise ValueError("Model worker authorization failed.")
    if len(json.dumps(payload).encode("utf-8")) > 128000:
        raise ValueError("Model request exceeds its size limit.")
    action = payload.get("action")
    if action == "health":
        return {
            "protocol": 1,
            "ready": True,
            "reranker": RERANK_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        }
    texts = payload.get("texts")
    maximum = 20 if action == "rerank" else 16
    if (
        not isinstance(texts, list)
        or not 1 <= len(texts) <= maximum
        or any(not isinstance(t, str) or not t.strip() or len(t) > 4000 for t in texts)
    ):
        raise ValueError("Model text batch exceeds its limits.")
    if action == "rerank":
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("Model question exceeds its limits.")
        ranked = rank(
            question, [{"text": text, "slot": i} for i, text in enumerate(texts)], k=len(texts)
        )
        if len(ranked) != len(texts) or {r["slot"] for r in ranked} != set(range(len(texts))):
            raise ValueError("Model scores are incomplete.")
        scores = [None] * len(texts)
        for record in ranked:
            score = record["rerank_score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError("Model returned invalid scores.")
            scores[record["slot"]] = float(score)
        return {"protocol": 1, "model": RERANK_MODEL, "scores": scores}
    if action == "embed":
        return {
            "protocol": 1,
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "vectors": embed(texts),
        }
    raise ValueError("Unsupported model action.")
