"""Проверка окружения: конфиг, Qdrant и Postgres достижимы (make check)."""

import sys

import psycopg
from qdrant_client.qdrant_client import QdrantClient

from genius_rag.config import settings


def check_config() -> None:
    if not settings.genius_access_token.get_secret_value():
        raise RuntimeError("GENIUS_ACCESS_TOKEN is empty, fill in .env")
    print(f"config OK: genius token set, model={settings.llm_model}")


def check_qdrant() -> None:
    client = QdrantClient(url=settings.qdrant_url)
    collections = client.get_collections().collections
    print(f"qdrant OK: {len(collections)} collections")


def check_postgres() -> None:
    with psycopg.connect(settings.postgres_dsn) as conn:
        row = conn.execute("SELECT version()").fetchone()
        if row is None:
            raise RuntimeError("postgres did not return version()")
        print(f"postgres OK: {row[0][:30]}")


def main() -> int:
    for check in (check_config, check_qdrant, check_postgres):
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
