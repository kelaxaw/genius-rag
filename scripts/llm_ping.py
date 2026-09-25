"""Provider check: one question -> a structured reply.

Usage: make llm q="who is Oxxxymiron?"          # OpenRouter, needs OPENROUTER_API_KEY
       make llm q="who is Oxxxymiron?" fake=1   # FakeLLM, no network
Prints the object type, detected language and the answer; a reply that does not match
the schema fails validation.
"""

import argparse
import sys

from pydantic import BaseModel, Field

from genius_rag.llm.provider import LLM, FakeLLM, get_llm
from genius_rag.observability import get_langfuse

SYSTEM = "Answer in one or two sentences. Reply in the language of the question."


class Ping(BaseModel):
    language: str = Field(description="ISO 639-1 code of the question language, e.g. ru or en")
    answer: str = Field(description="Short answer to the question")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args(argv)

    llm: LLM = (
        FakeLLM(reply=Ping(language="en", answer="This is FakeLLM: no network call was made."))
        if args.fake
        else get_llm()
    )
    result = llm.structured(SYSTEM, args.question, Ping)

    print(f"[{type(llm).__name__}] {type(result).__name__}")
    print(f"language: {result.language}")
    print(f"answer:   {result.answer}")
    get_langfuse().flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
