"""Embed every chunk that has no vector yet.

Usage: uv run python scripts/build_embeddings.py [--batch-size 32]
Re-running is a no-op ("0 chunks to embed"). The first run also downloads the model (~470 MB).
"""

import argparse
import sys
import time

from genius_rag.db import pg
from genius_rag.embeddings.encoder import embed_passages


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    with pg.connect() as conn:
        pg.init_schema(conn)
        rows = pg.fetch_chunks_without_embedding(conn)
        print(f"{len(rows)} chunks to embed")
        if not rows:
            return 0

        ids = [chunk_id for chunk_id, _ in rows]
        texts = [text for _, text in rows]

        t0 = time.perf_counter()
        vectors = embed_passages(texts, batch_size=args.batch_size)
        elapsed = time.perf_counter() - t0
        print(f"encoded {vectors.shape} in {elapsed:.1f}s ({len(texts) / elapsed:.0f} chunks/s)")

        updated = pg.update_embeddings(conn, list(zip(ids, vectors, strict=True)))
        print(f"updated {updated} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
