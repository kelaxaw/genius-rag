"""Load data/raw/*.jsonl into Postgres.

Usage: uv run python scripts/load_db.py            # every file in data/raw
       uv run python scripts/load_db.py data/raw/kai_angel.jsonl
Prints one line per file, then count(*) for every table. Re-running yields the same
counts: the load is idempotent.
"""

import sys
from pathlib import Path

from psycopg import sql

from genius_rag.db import pg

RAW = Path("data/raw")
TABLES = ("artists", "songs", "song_artists", "annotations")


def print_counts(conn: pg.Conn) -> None:
    for table in TABLES:
        query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
        row = conn.execute(query).fetchone()
        print(f"  {table:<13} {row[0] if row else '?'}")


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(RAW.glob("*.jsonl"))
    if not paths:
        print(f"no .jsonl in {RAW}", file=sys.stderr)
        return 1

    with pg.connect() as conn:
        pg.init_schema(conn)
        for path in paths:
            songs, annotations = pg.load_jsonl(conn, path)
            print(f"{path.name}: {songs} songs, {annotations} annotations")
        print("counts:")
        print_counts(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
