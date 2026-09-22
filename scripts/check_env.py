"""Environment check: config, Postgres and the pgvector extension are reachable.

Run: `uv run python scripts/check_env.py` (or `make check`).
Expected: three OK lines and exit code 0; any failure is a traceback and exit code 1.
"""

import sys

import psycopg

from genius_rag.config import settings


def check_config() -> None:
    accessToken = settings.genius_access_token.get_secret_value()
    hf_token = settings.hf_token.get_secret_value()
    llmModel = settings.llm_model

    if not accessToken:
        raise RuntimeError("GENIUS_ACCESS_TOKEN is empty. Fill in .env")

    if not hf_token:
        raise RuntimeError("HF_TOKEN is empty. Fill in .env")

    else:
        print(f"config OK, model={llmModel}")


def check_postgres() -> None:
    with psycopg.connect(settings.postgres_dsn) as conn:
        row = conn.execute("SELECT version()").fetchone()
        if row is None:
            raise RuntimeError("postgres doesn't return version()")

        print(f"postgres OK: {row[0][:30]}")


def check_pgvector() -> None:
    # The extension lives in the database, not the image: enable idempotently, read version.
    with psycopg.connect(settings.postgres_dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = %s", ("vector",)
        ).fetchone()
        if row is None:
            raise RuntimeError("pgvector extension not installed in image")

        print(f"pgvector OK: v{row[0]}")


def main() -> int:
    for check in (check_config, check_postgres, check_pgvector):
        check()

    return 0


if __name__ == "__main__":
    sys.exit(main())
