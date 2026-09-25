"""Langfuse tracing: one trace per question — retrieval -> generation -> answer.

Trace tree of `ask` (names are an API: filters, dashboards and evaluators refer to them):
  ask                          chain       input: question, output: Answer, metadata: k/artist/lang
  ├─ retrieve-annotations      retriever   retrieved hits with dense/fts/rrf scores
  └─ generate-answer           span        question + context ids -> Answer after the checks
     └─ generate-structured-output -> ChatOpenAI generation: model, tokens, cost

Why explicit trace_context instead of nested `with` blocks: /ask is a generator, and
FastAPI runs each of its steps in a threadpool with its OWN copy of contextvars. A span
made current before a `yield` is no longer current after it, so the generate step used to
land in a separate trace and OpenTelemetry logged "Failed to detach context". The root is
therefore created detached from the context, and every step attaches to it explicitly by
trace_id / parent_span_id.

Keys come from settings. Without keys, or with LANGFUSE_TRACING_ENABLED=false (tests),
tracing is disabled and the app behaves exactly the same.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

from genius_rag.config import settings

# Where the question came from: a trace tag to split dashboards by api vs cli.
Channel = Literal["api", "cli"]
StepType = Literal["retriever", "span"]


def tracing_enabled() -> bool:
    return bool(
        settings.langfuse_tracing_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    """Singleton client. Spans are exported in background batches; scripts flush() on exit."""
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=(
            settings.langfuse_secret_key.get_secret_value()
            if settings.langfuse_secret_key
            else None
        ),
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_tracing_environment,
        tracing_enabled=tracing_enabled(),
    )


def langchain_callbacks() -> list[BaseCallbackHandler]:
    """LangChain callback: records the model call as a generation under the current span."""
    return [CallbackHandler()] if tracing_enabled() else []


class AskTrace:
    """Trace of one question. Steps are opened with step(); the root is closed by end()."""

    def __init__(
        self,
        question: str,
        *,
        k: int,
        artist: str | None,
        lang: str | None,
        channel: Channel,
    ) -> None:
        self._langfuse = get_langfuse()
        self._tags: list[str] = [channel]
        with self._attributes():
            self._root = self._langfuse.start_observation(
                name="ask",
                as_type="chain",
                input={"question": question},
                metadata={"k": k, "artist": artist, "lang": lang, "channel": channel},
            )
        self._ctx: TraceContext = {
            "trace_id": self._root.trace_id,
            "parent_span_id": self._root.id,
        }

    def _attributes(self) -> Any:
        return propagate_attributes(trace_name="ask", tags=self._tags)

    @contextmanager
    def step(self, name: str, as_type: StepType, input: Any) -> Iterator[Any]:
        """Child step of the root, current inside the block (LangChain generations nest here)."""
        with (
            self._attributes(),
            self._langfuse.start_as_current_observation(
                name=name, as_type=as_type, input=input, trace_context=self._ctx
            ) as span,
        ):
            yield span

    def end(self, output: Any) -> None:
        self._root.update(output=output)
        self._root.end()

    def fail(self, error: BaseException) -> None:
        self._root.update(level="ERROR", status_message=repr(error))
        self._root.end()
