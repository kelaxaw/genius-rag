"""Postgres metadata store: schema bootstrap and JSONL loading.

All statements use psycopg3 server-side parameters (`%s`); connections are used as
context managers so a whole file loads in one transaction.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from genius_rag.chunking.docs import AnnotationCtx, Chunk, detect_lang
from genius_rag.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

Conn = psycopg.Connection[Any]


def connect() -> Conn:
    """Open a connection from the configured DSN; use as a context manager."""
    conn = psycopg.connect(settings.postgres_dsn)
    # register_vector adapts numpy arrays to vector(384). It needs the type to exist,
    # so the extension is created here, before schema.sql runs.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    # HNSW applies WHERE only after the index scan: it returns ~ef_search (40) nearest rows
    # and the filter trims them, so a filtered top-k can come back short. hybrid_search
    # emits bare filter predicates, which lets the planner see a narrow filter (one artist,
    # ~84 rows) and switch to an exact scan. A wide filter (lang=en, 67% of rows) on a
    # larger table still goes through the index; iterative scan keeps reading it until
    # enough rows pass. relaxed_order is approximate, which is fine: RRF re-ranks anyway.
    conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
    return conn


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


def fetch_annotation_ctx(conn: Conn) -> list[AnnotationCtx]:
    """Every annotation with the context needed for its prefix; artists joined by position."""
    rows = conn.execute("""SELECT ann.id, ss.id, STRING_AGG(ar.name, ', ' ORDER BY sa.position),
      ss.title, ss.album, ann.fragment, ann.text
      FROM songs ss
      JOIN song_artists sa ON ss.id = sa.song_id
      JOIN artists ar ON ar.id = sa.artist_id
      JOIN annotations ann ON ss.id = ann.song_id
      GROUP BY ann.id, ss.id, ss.title, ss.album, ann.fragment, ann.text
      ORDER BY ann.id""").fetchall()

    return [AnnotationCtx(*row) for row in rows]


def update_chunks(conn: Conn, annotation_id: int, chunks: list[Chunk]) -> int:
    """Replace an annotation's chunks (DELETE + INSERT); return the number inserted."""
    conn.execute("""DELETE FROM chunks WHERE annotation_id = %s""", (annotation_id,))

    values = [(c.annotation_id, c.position, c.lang, c.text, c.tokens) for c in chunks]

    with conn.cursor() as curs:
        curs.executemany(
            """INSERT INTO chunks (annotation_id, position, lang, text, tokens)
          VALUES (%s, %s, %s, %s, %s)""",
            values,
        )

        return curs.rowcount


# ---- embeddings + search ------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """One search result. Higher score = more relevant; dense and full-text scales differ."""

    chunk_id: int
    annotation_id: int
    text: str
    score: float


def fetch_chunks_without_embedding(conn: Conn) -> list[tuple[int, str]]:
    """(id, text) of chunks with no embedding yet, so re-runs only encode new rows."""
    return conn.execute(
        """SELECT id, text FROM chunks WHERE embedding IS NULL ORDER BY id"""
    ).fetchall()


def update_embeddings(conn: Conn, rows: list[tuple[int, np.ndarray]]) -> int:
    """Store vectors; rows = [(chunk_id, vector), ...]. Return the number of updated rows."""
    values = [(vector, chunk_id) for chunk_id, vector in rows]

    with conn.cursor() as curr:
        curr.executemany("""UPDATE chunks SET embedding = %s WHERE id = %s""", values)

        return curr.rowcount


def knn_search(conn: Conn, query_vec: np.ndarray, k: int = 10) -> list[Hit]:
    """Top-k chunks by cosine similarity to the query vector (embedded rows only)."""
    rows = conn.execute(
        """SELECT
        id,
        annotation_id, text,
        1 - (embedding <=> %s) AS score
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s""",
        (query_vec, query_vec, k),
    ).fetchall()

    return [Hit(*r) for r in rows]


def fts_search(conn: Conn, query: str, k: int = 10) -> list[Hit]:
    """Top-k chunks by full-text match; dictionary follows the query language."""
    lang = "russian" if detect_lang(query) == "ru" else "english"

    rows = conn.execute(
        """SELECT
      id,
      annotation_id,
      text,
      ts_rank_cd(tsv, websearch_to_tsquery(%s::regconfig, %s)) AS score
      FROM chunks
      WHERE tsv @@ websearch_to_tsquery(%s::regconfig, %s)
      ORDER BY score DESC
      LIMIT %s
      """,
        (lang, query, lang, query, k),
    ).fetchall()

    return [Hit(*r) for r in rows]
