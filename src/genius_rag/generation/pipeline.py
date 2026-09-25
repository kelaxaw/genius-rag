"""Traced /ask steps: retrieve -> generate. Shared by the API and scripts/demo_ask.py.

The logic is unchanged: these functions wrap search() and answer_question() in trace steps
(see observability.py) so Langfuse shows what was retrieved and what the model answered.
"""

from dataclasses import asdict
from typing import Protocol

from genius_rag.db import pg
from genius_rag.generation.answer import Answer, answer_question
from genius_rag.llm.provider import LLM
from genius_rag.observability import AskTrace
from genius_rag.retrieval.hybrid import HybridHit, search


class Searcher(Protocol):
    """Search callable: (question, k, artist, lang) -> hits; arguments may be passed by name."""

    def __call__(
        self, question: str, k: int, artist: str | None, lang: str | None
    ) -> list[HybridHit]: ...


def db_search(question: str, k: int, artist: str | None, lang: str | None) -> list[HybridHit]:
    """Real hybrid search, one connection per request (a pool can come later)."""
    with pg.connect() as conn:
        return search(conn, question, k=k, artist=artist, lang=lang)


def retrieve(
    trace: AskTrace,
    searcher: Searcher,
    question: str,
    *,
    k: int,
    artist: str | None,
    lang: str | None,
) -> list[HybridHit]:
    step_input = {"query": question, "k": k, "artist": artist, "lang": lang, "mode": "hybrid"}
    with trace.step("retrieve-annotations", "retriever", step_input) as span:
        hits = searcher(question=question, k=k, artist=artist, lang=lang)
        span.update(output=[asdict(h) for h in hits], metadata={"hits": len(hits)})
    return hits


def generate(trace: AskTrace, question: str, hits: list[HybridHit], llm: LLM) -> Answer:
    step_input = {"question": question, "context_ids": [h.annotation_id for h in hits]}
    with trace.step("generate-answer", "span", step_input) as span:
        answer = answer_question(question=question, hits=hits, llm=llm)
        span.update(output=answer)
    return answer
