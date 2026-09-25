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

Tracing: every request is one `ask` trace in Langfuse (see observability.py).
"""

from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from genius_rag.embeddings.encoder import _model
from genius_rag.generation.pipeline import Searcher, db_search, generate, retrieve
from genius_rag.llm.provider import LLM, get_llm
from genius_rag.observability import AskTrace, get_langfuse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load e5 once at startup (lru_cache keeps it); otherwise the first request waits ~8 s.
    _model()
    yield
    # Export buffered spans before the process exits.
    get_langfuse().shutdown()


app = FastAPI(title="genius-rag", lifespan=lifespan)


# ---- dependencies --------------------------------------------------------------------------


def get_searcher() -> Searcher:
    """Real hybrid search (generation/pipeline.py); tests override it."""
    return db_search


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
    trace = AskTrace(q, k=k, artist=artist, lang=lang, channel="api")
    try:
        hits = retrieve(trace, searcher, q, k=k, artist=artist, lang=lang)

        sources = [{"annotation_id": h.annotation_id, "text": h.text} for h in hits]

        yield ServerSentEvent(event="sources", data=sources)

        answer = generate(trace, q, hits, llm)

        yield ServerSentEvent(event="answer", data=answer)
    except BaseException as e:
        # BaseException also catches GeneratorExit when the client closes the stream.
        trace.fail(e)
        raise
    trace.end(answer)

    yield ServerSentEvent(event="done")
