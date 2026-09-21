"""Annotations from the DB -> the chunks table.

Usage: uv run python scripts/build_chunks.py [--max-tokens 256]
Prints "N annotations -> M chunks", the chunks-per-annotation distribution and token stats.
Re-running yields the same numbers (DELETE + INSERT per annotation).
"""

import argparse
import sys
from collections import Counter

from genius_rag.chunking.docs import MAX_TOKENS, build_chunks
from genius_rag.db import pg


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args(argv)

    per_annotation: Counter[int] = Counter()
    token_counts: list[int] = []
    langs: Counter[str] = Counter()

    with pg.connect() as conn:
        pg.init_schema(conn)
        rows = pg.fetch_annotation_ctx(conn)
        for ctx in rows:
            chunks = build_chunks(ctx, max_tokens=args.max_tokens)
            pg.update_chunks(conn, ctx.annotation_id, chunks)
            per_annotation[len(chunks)] += 1
            token_counts.extend(c.tokens for c in chunks)
            langs.update(c.lang for c in chunks)

    total = sum(per_annotation.values())
    print(f"{total} annotations -> {len(token_counts)} chunks (max_tokens={args.max_tokens})")
    print("chunks per annotation:", dict(sorted(per_annotation.items())))
    print("lang:", dict(langs))
    token_counts.sort()
    n = len(token_counts)
    print(
        f"tokens per chunk: min={token_counts[0]} p50={token_counts[n // 2]} "
        f"p90={token_counts[int(n * 0.9)]} max={token_counts[-1]}"
    )
    over = sum(t > args.max_tokens for t in token_counts)
    if over:
        print(f"WARNING: {over} chunks exceed max_tokens (single sentence longer than budget)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
