"""LLM provider: one interface, interchangeable backends.

generation/answer.py does not know who answers (OpenRouter, GigaChat or a stub): it calls
`llm.structured(...)` and gets a pydantic object back. Switching models means changing
LLM_MODEL in .env; generation code stays untouched.

Structured output: the model replies with JSON matching a schema (a pydantic class), and
the client validates it and builds the object.

FakeLLM serves tests and offline demos: no network, no key, no cost.
"""

from typing import Protocol, TypeVar

from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from genius_rag.config import settings

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    """Anything that turns a system + user message into an instance of `schema`.

    Structural typing: implementations do not inherit from LLM, they only need a method
    with this signature (checked by mypy).
    """

    def structured(self, system: str, user: str, schema: type[T]) -> T: ...


# ---- fake ---------------------------------------------------------------------------------


class FakeLLM:
    """Returns a preset reply and records every (system, user) pair it received.

    Tests use `calls` to check what reached the prompt, or that the LLM was not called.
    """

    def __init__(self, reply: BaseModel) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((system, user))
        if not isinstance(self.reply, schema):
            raise TypeError(
                f"FakeLLM reply is {type(self.reply).__name__}, caller asked {schema.__name__}"
            )
        return self.reply


# ---- OpenRouter ---------------------------------------------------------------------------


class OpenRouterLLM:
    """A real model via OpenRouter. OpenRouter speaks the OpenAI API protocol, so the
    client is langchain-openai's ChatOpenAI pointed at a different base URL.
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = settings.openrouter_api_key

        if api_key is None:
            raise RuntimeError("OPENROUTER_API_KEY is None")

        if api_key.get_secret_value() == "":
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        self.model = model or settings.llm_model

        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=self.model,
            api_key=api_key,
            temperature=0,
        )

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        output = self.llm.with_structured_output(schema, method="json_schema").invoke(
            [SystemMessage(system), HumanMessage(user)]
        )

        if not isinstance(output, schema):
            raise TypeError(
                f"LangChain type is {type(output).__name__}, caller asked {schema.__name__}"
            )

        return output


# ---- factory ------------------------------------------------------------------------------


def get_llm() -> LLM:
    """The real provider from settings. FakeLLM is built directly: it needs a preset reply."""
    return OpenRouterLLM()
