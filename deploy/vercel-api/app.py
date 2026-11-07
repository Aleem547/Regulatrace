"""Vercel API entrypoint; the model worker runs on the existing free Space."""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cache = root / "tiktoken-cache"
if cache.is_dir():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
os.environ["REGULATRACE_MODEL_EXECUTION"] = "space"
os.environ["REGULATRACE_DATABASE_NULL_POOL"] = "true"
os.environ["REGULATRACE_CHAT_RATE_LIMIT_STORE"] = "postgres"
os.environ.setdefault("REGULATRACE_ENVIRONMENT", "production")
os.environ.setdefault("REGULATRACE_LLM_PROVIDER", "groq")
os.environ.setdefault("REGULATRACE_LLM_MODEL", "openai/gpt-oss-120b")
source = root / "src"
if not source.is_dir():
    source = root.parents[1] / "src"
sys.path.insert(0, str(source))

from regulatrace.api.app import create_app  # noqa: E402

app = create_app()
