"""Search demo: question -> top-k chunks, dense (kNN) or full-text.

Usage: uv run python scripts/search.py "what does the line about ... mean" [--mode dense|fts] [-k 5]
Prints "score  chunk_id/annotation_id  text preview" per hit. Run the same question in both
modes to see where each retriever fails; that gap is what hybrid retrieval closes.
"""

import argparse
import sys

from genius_rag.db import pg
from genius_rag.embeddings.encoder import embed_query


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["dense", "fts"], default="dense")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    with pg.connect() as conn:
        if args.mode == "dense":
            hits = pg.knn_search(conn, embed_query(args.query), k=args.k)
        else:
            hits = pg.fts_search(conn, args.query, k=args.k)

    print(f"[{args.mode}] {args.query!r} -> {len(hits)} hits")
    for h in hits:
        preview = h.text.replace("\n", " ")[:110]
        print(f"{h.score:6.3f}  {h.chunk_id:5d}/{h.annotation_id:<5d} {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
