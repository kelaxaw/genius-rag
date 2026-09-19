"""Postgres metadata store: schema bootstrap and JSONL loading.

All statements use psycopg3 server-side parameters (`%s`); connections are used as
context managers so a whole file loads in one transaction.
"""

import json
from pathlib import Path
from typing import Any

import psycopg

from genius_rag.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

Conn = psycopg.Connection[Any]


def connect() -> Conn:
    """Open a connection from the configured DSN; use as a context manager."""
    return psycopg.connect(settings.postgres_dsn)


def init_schema(conn: Conn) -> None:
    """Apply schema.sql. Idempotent (IF NOT EXISTS throughout)."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One line per song, as written by scripts/ingest.py."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def upsert_artist(conn: Conn, name: str) -> int:
    """Insert an artist by name and return its id.

    DO UPDATE with the same value keeps RETURNING populated on conflict.
    """
    response = conn.execute(
        """INSERT INTO artists (name) VALUES (%s)
      ON CONFLICT(name) DO UPDATE SET name = EXCLUDED.name RETURNING id""",
        (name,),
    ).fetchone()

    if response is None:
        raise RuntimeError("upsert_artist returned no row")

    return response[0]


def upsert_song(conn: Conn, song_id: int, title: str, album: str | None) -> None:
    """Insert or overwrite a song by Genius id; the JSONL is the source of truth."""
    conn.execute(
        """INSERT INTO songs (id, title, album) VALUES (%s, %s, %s)
      ON CONFLICT(id) DO UPDATE SET title = EXCLUDED.title,
      album = EXCLUDED.album""",
        (song_id, title, album),
    )


def link_song_artist(conn: Conn, song_id: int, artist_id: int, position: int) -> None:
    """Song <-> artist edge. Position is refreshed from the JSONL order."""
    conn.execute(
        """INSERT INTO song_artists (song_id, artist_id, position) VALUES(%s, %s, %s)
        ON CONFLICT(song_id, artist_id) DO UPDATE SET position = EXCLUDED.position
        """,
        (song_id, artist_id, position),
    )


def insert_annotations(conn: Conn, song_id: int, annotations: list[dict[str, str]]) -> int:
    """Insert a song's annotations and return how many rows were actually inserted.

    Empty text is skipped (schema CHECK); duplicates collapse via UNIQUE + DO NOTHING.
    """
    values = [(song_id, a["fragment"], a["text"]) for a in annotations if a["text"]]

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO annotations (song_id, fragment, text) VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            values,
        )
        return cur.rowcount


def load_song(conn: Conn, record: dict[str, Any]) -> int:
    """Load one JSONL record. FK order: song -> artists -> edges -> annotations.

    Known limitation: an artist removed from the list is not removed from song_artists.
    """
    song_id = record["song_id"]

    upsert_song(conn=conn, song_id=song_id, title=record["title"], album=record["album"])

    for idx, artist in enumerate(record["artists"]):
        artist_id = upsert_artist(conn=conn, name=artist)
        link_song_artist(conn=conn, song_id=song_id, artist_id=artist_id, position=idx)

    return insert_annotations(conn=conn, song_id=song_id, annotations=record["annotations"])


def load_jsonl(conn: Conn, path: Path) -> tuple[int, int]:
    """Load a file. Returns (songs, annotations inserted)."""
    jsonl = read_jsonl(path=path)

    annotations_count = 0

    for record in jsonl:
        annotations_count += load_song(conn=conn, record=record)

    return (len(jsonl), annotations_count)
