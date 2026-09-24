"""Search demo: question -> top-k chunks, hybrid (RRF), dense (kNN) or full-text.

Usage: uv run python scripts/search.py "what does the line about ... mean"
       [--mode hybrid|dense|fts] [-k 5] [--artist "Drake"] [--lang ru|en]
Prints "rrf  dense  fts  chunk_id/annotation_id  text preview" per hit; "-" means the
branch did not return that chunk. Run the same question in every mode to see what
hybrid recovers that a single retriever misses.
"""

import argparse
import sys

from genius_rag.db import pg
from genius_rag.retrieval.hybrid import search


def fmt(score: float | None) -> str:
    return f"{score:6.3f}" if score is not None else "     -"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["hybrid", "dense", "fts"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--artist", default=None)
    parser.add_argument("--lang", choices=["ru", "en"], default=None)
    args = parser.parse_args(argv)

    with pg.connect() as conn:
        hits = search(
            conn, args.query, k=args.k, mode=args.mode, artist=args.artist, lang=args.lang
        )

    filters = f" artist={args.artist!r} lang={args.lang!r}" if args.artist or args.lang else ""
    print(f"[{args.mode}]{filters} {args.query!r} -> {len(hits)} hits")
    print("   rrf  dense    fts  chunk/annot  text")
    for h in hits:
        preview = h.text.replace("\n", " ")[:90]
        print(
            f"{h.rrf:6.4f} {fmt(h.dense_score)} {fmt(h.fts_score)}  "
            f"{h.chunk_id:5d}/{h.annotation_id:<5d} {preview}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
