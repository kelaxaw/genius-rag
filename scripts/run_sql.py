"""Run a .sql file and print the result as a table.

Usage: uv run python scripts/run_sql.py sql/analytics/01_songs_per_artist.sql
"""

import sys
from pathlib import Path

from genius_rag.db import pg


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_sql.py <file.sql>", file=sys.stderr)
        return 1

    query = Path(argv[0]).read_text(encoding="utf-8")
    with pg.connect() as conn:
        cur = conn.execute(query)
        if cur.description is None:
            print(f"ok, {cur.rowcount} rows affected")
            return 0
        headers = [d.name for d in cur.description]
        rows = [[str(v) for v in row] for row in cur.fetchall()]

    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)
    ]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)))
    print(f"({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
