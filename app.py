"""Optional Gradio/ZeroGPU entry point; Docker and FastAPI keep their own launcher."""

import os
import sys
from pathlib import Path

import spaces  # Must precede Torch for ZeroGPU's CUDA registration hooks.
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

os.environ.setdefault("REGULATRACE_LLM_PROVIDER", "groq")
os.environ.setdefault("REGULATRACE_LLM_MODEL", "openai/gpt-oss-120b")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from regulatrace.model_worker import EMBEDDING_REVISION, handle_model_request  # noqa: E402
from regulatrace.reranker import MODEL_NAME, rerank  # noqa: E402
from regulatrace.space_ui import create_demo  # noqa: E402

# Module-scope loading registers weights with ZeroGPU. Inference happens only
# in the decorated function. Outside ZeroGPU this also supports CPU development.
model = CrossEncoder(
    MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", max_length=512
)


@spaces.GPU(duration=30)
def rerank_on_gpu(question: str, candidates: list[dict], k: int = 5) -> list[dict]:
    return rerank(question, candidates, k=k, model=model)


worker_secret = os.environ.get("REGULATRACE_MODEL_WORKER_SECRET", "")
if len(worker_secret) >= 32:
    embedding_model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        revision=EMBEDDING_REVISION,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    @spaces.GPU(duration=30)
    def embed_on_gpu(texts):
        return embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def model_worker(payload):
        import gradio as gr

        try:
            return handle_model_request(
                payload, secret=worker_secret, rank=rerank_on_gpu, embed=embed_on_gpu
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from None
        except Exception:
            raise gr.Error("The model request could not be completed.") from None

    demo = create_demo(rerank_on_gpu, model_worker=model_worker)
else:
    demo = create_demo(rerank_on_gpu)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        max_file_size=10 * 1024 * 1024,
        show_error=False,
        ssr_mode=False,
        state_session_capacity=16,
        blocked_paths=[str(Path(__file__).resolve().parent / ".env")],
    )
