"""HTTP API: GET /ask -> a stream of Server-Sent Events.

SSE is a single HTTP response that stays open while the server appends events as they
become ready (text/event-stream).

Event order:
  sources -> retrieved annotations (ready after retrieval, ~0.1 s)
  answer  -> the Answer from generation/answer.py (after the LLM, ~2-4 s)
  done    -> end of stream

Why stages and not tokens: structured output returns ONE JSON object, which cannot be
shown piece by piece. The gain here is that sources arrive seconds before the answer.

Sync, not async: search() (psycopg + e5 on CPU) and llm.structured (ChatOpenAI.invoke)
are blocking. FastAPI runs a plain def generator in a threadpool, keeping the event loop free.

Dependencies: search and LLM are injected with Depends, so tests replace both through
app.dependency_overrides and run without a database, network or API key.
"""

from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Protocol

from fastapi import Depends, FastAPI, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from genius_rag.db import pg
from genius_rag.embeddings.encoder import _model
from genius_rag.generation.answer import answer_question
from genius_rag.llm.provider import LLM, get_llm
from genius_rag.retrieval.hybrid import HybridHit, search


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load e5 once at startup (lru_cache keeps it); otherwise the first request waits ~8 s.
    _model()
    yield


app = FastAPI(title="genius-rag", lifespan=lifespan)


class Searcher(Protocol):
    """Search callable: (question, k, artist, lang) -> hits; arguments may be passed by name."""

    def __call__(
        self, question: str, k: int, artist: str | None, lang: str | None
    ) -> list[HybridHit]: ...


# ---- dependencies --------------------------------------------------------------------------


def get_searcher() -> Searcher:
    """Real hybrid search, one connection per request (a pool can come later)."""

    def run(question: str, k: int, artist: str | None, lang: str | None) -> list[HybridHit]:
        with pg.connect() as conn:
            return search(conn, question, k=k, artist=artist, lang=lang)

    return run


# ---- endpoint -----------------------------------------------------------------------------


@app.get("/ask", response_class=EventSourceResponse)
def ask(
    q: Annotated[str, Query(min_length=1)],
    llm: Annotated[LLM, Depends(get_llm)],
    searcher: Annotated[Searcher, Depends(get_searcher)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    artist: str | None = None,
    lang: Literal["ru", "en"] | None = None,
) -> Iterable[ServerSentEvent]:
    """Question -> events sources, answer, done."""
    hits = searcher(question=q, k=k, artist=artist, lang=lang)

    sources = [{"annotation_id": h.annotation_id, "text": h.text} for h in hits]

    yield ServerSentEvent(event="sources", data=sources)

    answer = answer_question(question=q, hits=hits, llm=llm)

    yield ServerSentEvent(event="answer", data=answer)

    yield ServerSentEvent(event="done")
