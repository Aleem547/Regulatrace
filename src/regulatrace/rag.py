"""Format retrieved evidence and run a grounded LCEL RAG chain."""

import argparse
import json
import sys
from collections.abc import Callable
from contextlib import suppress
from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import Any

import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from regulatrace.bm25 import retrieve_bm25
from regulatrace.citations import validate_citations
from regulatrace.llm import GroqLLM, LLMClient, get_llm
from regulatrace.privacy import redact_for_tracing
from regulatrace.reranker import ABSTENTION_MESSAGE, decide_abstention, rerank
from regulatrace.tracing import (
    record_candidates,
    record_query,
    trace_stage,
    update_observation,
)

_CANDIDATE_K = 20
_FINAL_K = 5
# Application planning budget, not the model's maximum context window.
# Counts encoded prompt text; Groq's chat framing may add tokens.
_TOTAL_PLANNING_TOKENS = 8704
_MAX_PROMPT_TOKENS = 7680
_MAX_COMPLETION_TOKENS = 1024
_NOT_FOUND = "not found in the provided documents"


@lru_cache(maxsize=1)
def _prompt_encoding() -> tiktoken.Encoding:
    """Load the generation model's o200k_harmony encoding only once."""
    return tiktoken.get_encoding("o200k_harmony")


def format_evidence(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Format usable chunks in retrieval order without changing their text or metadata.

    Missing, non-string, and whitespace-only text is skipped. Source numbers are
    consecutive among included chunks; they are local labels, not document IDs.
    Each included chunk keeps a distinct source, even for the same document/page.
    """
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        metadata = chunk.get("metadata") or {}
        tag = f"Source {len(sources) + 1}"
        source = {
            "tag": tag,
            "doc_id": metadata.get("doc_id"),
            "chunk_index": metadata.get("chunk_index"),
            "title": metadata.get("title"),
            "source_file": chunk.get("source_file"),
            "page": chunk.get("page"),
            "circular_no": metadata.get("circular_no"),
            "source_url": metadata.get("source_url"),
        }
        lines = [f"[{tag}]"]
        for label, field in (
            ("Document", "title"),
            ("Circular", "circular_no"),
            ("File", "source_file"),
            ("Page", "page"),
            ("URL", "source_url"),
        ):
            value = source[field]
            lines.append(f"{label}: {value if value is not None else 'unavailable'}")
        lines.extend(["Text:", text, f"[End {tag}]"])
        blocks.append("\n".join(lines))
        sources.append(source)

    return {"context": "\n\n".join(blocks), "sources": sources}


def _render_prompt(prompt: ChatPromptTemplate, question: str, evidence: dict[str, Any]) -> str:
    """Render exactly the string sent to the LLM, including all source metadata."""
    return prompt.invoke(
        {
            "question": question,
            "context": evidence["context"],
            "sources": json.dumps(evidence["sources"], ensure_ascii=False, indent=2),
        }
    ).to_string()


def _budget_evidence(
    question: str, chunks: list[dict[str, Any]], prompts: tuple[ChatPromptTemplate, ...]
) -> dict[str, Any]:
    """Keep whole chunks in ranked order when both prompt versions fit the same budget."""
    included: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"context": "", "sources": []}
    encoding = _prompt_encoding()
    for chunk in chunks:
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            continue
        metadata = chunk.get("metadata")
        doc_id = metadata.get("doc_id") if isinstance(metadata, dict) else None
        page = chunk.get("page")
        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        if type(page) is not int or page <= 0:
            continue
        candidate = format_evidence([*included, chunk])
        if all(
            len(encoding.encode(_render_prompt(prompt, question, candidate), disallowed_special=()))
            <= _MAX_PROMPT_TOKENS
            for prompt in prompts
        ):
            included.append(chunk)
            evidence = candidate
    return evidence


def build_rag_chain(
    k: int = 5,
    llm: LLMClient | None = None,
    prompt_version: str = "v1",
    *,
    public_only: bool = False,
    retriever: Callable[..., list[dict[str, Any]]] | None = None,
    rerank_candidates: Callable[..., list[dict[str, Any]]] | None = None,
) -> Runnable:
    """Build the frozen BM25-to-reranker chain with one retrieval per invocation.

    Sources stay in invocation-local accepted evidence, never in model output.
    Rejected evidence skips generation; errors propagate with sources on generation failure.
    The configured client is initialized at construction unless one is supplied.
    The API requires public-only Qdrant pre-filtering; local/evaluation defaults stay unchanged.
    Optional callbacks support the isolated Space profile without replacing globals.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k != _FINAL_K:
        raise ValueError("The frozen RAG pipeline requires final k=5")
    if prompt_version not in ("v1", "v2"):
        raise ValueError("prompt_version must be 'v1' or 'v2'")
    client = llm if llm is not None else get_llm()
    prompt_dir = Path(__file__).resolve().parents[2] / "prompts"
    prompts = {}
    for version in ("v1", "v2"):
        prompt_text = (prompt_dir / f"answer_{version}.txt").read_text(encoding="utf-8")
        system_text, human_text = prompt_text.split("END SYSTEM INSTRUCTIONS", maxsplit=1)
        prompts[version] = ChatPromptTemplate.from_messages(
            [("system", system_text.strip()), ("human", human_text.strip())]
        )

    def generate(evidence: dict[str, Any]) -> str:
        prompt_text = _render_prompt(prompts[prompt_version], evidence["question"], evidence)
        options = {"temperature": 0}
        if isinstance(client, GroqLLM):
            options["max_completion_tokens"] = _MAX_COMPLETION_TOKENS
        try:
            with trace_stage(
                "llm_generation",
                as_type="generation",
                model=getattr(client, "model", None),
                version=prompt_version,
                model_parameters=options,
            ) as observation:
                if observation is not None:
                    with suppress(Exception):
                        update_observation(
                            observation,
                            metadata={
                                "prompt_version": prompt_version,
                                "prompt_text_tokens": len(
                                    _prompt_encoding().encode(prompt_text, disallowed_special=())
                                ),
                                "token_count_method": "o200k_harmony; excludes chat framing",
                                "supplied_sources": [
                                    {
                                        key: source.get(key)
                                        for key in ("doc_id", "chunk_index", "page")
                                    }
                                    for source in evidence["sources"]
                                ],
                            },
                        )
                answer_text = client.generate(prompt_text, **options)
                if observation is not None:
                    with suppress(Exception):
                        update_observation(observation, output=redact_for_tracing(answer_text))
                return answer_text
        except Exception as exc:
            # Keep invocation-local supplied sources available to the HTTP error response.
            exc.regulatrace_sources = evidence["sources"]
            raise

    generate_answer = RunnableLambda(generate) | StrOutputParser()
    answer = RunnableBranch(
        (
            lambda evidence: evidence["abstained"],
            RunnableLambda(lambda evidence: evidence["refusal_answer"]),
        ),
        generate_answer,
    )

    def prepare_evidence(state: dict[str, Any]) -> dict[str, Any]:
        decision = decide_abstention(state["chunks"])
        if decision["abstain"]:
            return {
                "question": state["question"],
                "context": "",
                "sources": [],
                "abstained": True,
                "refusal_answer": ABSTENTION_MESSAGE,
            }
        evidence = _budget_evidence(state["question"], state["chunks"], tuple(prompts.values()))
        if not evidence["sources"]:
            return {
                "question": state["question"],
                "context": "",
                "sources": [],
                "abstained": True,
                "refusal_answer": _NOT_FOUND,
            }
        return {
            "question": state["question"],
            **evidence,
            "abstained": False,
        }

    def retrieve_and_rerank(question: str) -> list[dict[str, Any]]:
        record_query(question, retrieval=True)
        with trace_stage("retrieval") as observation:
            candidates = (retriever or retrieve_bm25)(
                question, k=_CANDIDATE_K, **({"public_only": True} if public_only else {})
            )
            record_candidates(observation, candidates)
        with trace_stage("reranking") as observation:
            selected = (rerank_candidates or rerank)(question, candidates, k=_FINAL_K)
            record_candidates(observation, selected)
            return selected

    return (
        RunnableParallel(
            question=RunnablePassthrough(),
            chunks=RunnableLambda(retrieve_and_rerank),
        )
        | RunnableLambda(prepare_evidence)
        | RunnableParallel(
            answer=answer,
            sources=RunnableLambda(itemgetter("sources")),
            abstained=RunnableLambda(itemgetter("abstained")),
        )
        | RunnablePassthrough.assign(
            abstained=RunnableLambda(
                lambda result: (
                    result["abstained"]
                    or result["answer"].strip().casefold() == _NOT_FOUND.casefold()
                )
            ),
            citation_validation=RunnableLambda(
                lambda result: validate_citations(result["answer"], result["sources"])
            ),
        )
    )


def main() -> None:
    """Answer a question using the configured chain and print its returned sources."""
    parser = argparse.ArgumentParser(description="Regulatrace grounded RAG")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--k", type=int, default=5, help="Final evidence depth (fixed at 5)")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must not be empty or whitespace-only")
    if args.k != _FINAL_K:
        parser.error("--k must be 5 for the frozen RAG pipeline")

    try:
        result = build_rag_chain(k=args.k).invoke(args.query)
    except Exception as exc:
        # SDK error messages can contain credentials or request details; do not echo them.
        failure = type(exc).__name__
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and 400 <= status_code <= 599:
            failure += f" (HTTP {status_code})"
        print(
            "RAG error: could not complete retrieval or inference. "
            "Check LLM configuration, credentials, and retrieval/API connectivity. "
            f"Failure: {failure}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    if not result["sources"]:
        print("No sources.")
    for source in result["sources"]:
        print(f"[{source['tag']}]")
        for label, field in (("File", "source_file"), ("Page", "page")):
            value = source[field]
            print(f"  {label}: {value if value is not None else 'unavailable'}")
        for label, field in (
            ("Document", "title"),
            ("Document ID", "doc_id"),
            ("Chunk index", "chunk_index"),
            ("Circular", "circular_no"),
            ("URL", "source_url"),
        ):
            value = source[field]
            if value is not None:
                print(f"  {label}: {value}")


if __name__ == "__main__":
    main()
