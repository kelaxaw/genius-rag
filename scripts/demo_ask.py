"""Demo: question -> hybrid retrieval -> grounded answer with citations, or a refusal.

Two modes:
  make ask q="what is Love Yourself about?" [k=5] [artist="Justin Bieber"] [lang=en]
      in-process: retrieval + LLM called directly.
  make api  (in another terminal), then
  make ask q="what is Love Yourself about?" api=1
      client for GET /ask: reads the SSE stream and prints events as they arrive,
      prefixed with seconds since the request started (sources arrive before answer).

Prints the answer and the cited annotations ("annotation_id  text preview"), or
"REFUSED" and the refusal text.
"""

import argparse
import json
import sys
import time
from typing import Any

import httpx

from genius_rag.generation.answer import Answer
from genius_rag.generation.pipeline import db_search, generate, retrieve
from genius_rag.llm.provider import get_llm
from genius_rag.observability import AskTrace, get_langfuse


def print_answer(answer: Answer, texts: dict[int, str]) -> None:
    print(("REFUSED: " if answer.refused else "A: ") + answer.answer)
    for aid in answer.annotation_ids:
        print(f"   [{aid}] {texts[aid].replace(chr(10), ' ')[:100]}")


def run_local(args: argparse.Namespace) -> None:
    trace = AskTrace(args.question, k=args.k, artist=args.artist, lang=args.lang, channel="cli")
    try:
        hits = retrieve(
            trace, db_search, args.question, k=args.k, artist=args.artist, lang=args.lang
        )
        result = generate(trace, args.question, hits, get_llm())
    except BaseException as e:
        trace.fail(e)
        raise
    trace.end(result)
    print_answer(result, {h.annotation_id: h.text for h in hits})
    print(f"context: {[h.annotation_id for h in hits]}")


def run_api(args: argparse.Namespace) -> None:
    params: dict[str, Any] = {"q": args.question, "k": args.k}
    if args.artist:
        params["artist"] = args.artist
    if args.lang:
        params["lang"] = args.lang

    texts: dict[int, str] = {}
    event = "message"
    start = time.perf_counter()
    # stream: read the body as it arrives instead of waiting for the full response.
    with httpx.stream("GET", f"{args.url}/ask", params=params, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                raw = line.removeprefix("data:").strip()
                data: Any = json.loads(raw) if raw else None
                t = f"[{time.perf_counter() - start:5.2f}s]"
                if event == "sources":
                    texts = {h["annotation_id"]: h["text"] for h in data}
                    print(f"{t} sources: {list(texts)}")
                elif event == "answer":
                    print(f"{t} answer:")
                    print_answer(Answer.model_validate(data), texts)
                else:
                    print(f"{t} {event}: {raw}")
            elif line == "":
                event = "message"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--artist", default=None)
    parser.add_argument("--lang", choices=["ru", "en"], default=None)
    parser.add_argument(
        "--api", action="store_true", help="ask through a running `make api` server"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)

    print(f"Q: {args.question}")
    if args.api:
        run_api(args)
    else:
        run_local(args)
        # Short-lived script: flush, or buffered spans are lost on exit.
        get_langfuse().flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
