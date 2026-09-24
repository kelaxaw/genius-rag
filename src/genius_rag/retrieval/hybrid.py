"""Hybrid retrieval: dense + full-text fused by rank (RRF) in a single SQL query.

Dense and full-text scores live on different scales (cosine 0.8-0.9 vs ts_rank
0.001-0.1), so they cannot be added. Reciprocal Rank Fusion sums 1/(K + rank) instead:
the scale disappears and only the order remains. A chunk ranked high by both branches
gets the largest sum; a chunk seen by one branch gets about half. The result combines
exact term matches (fts) with semantic matches (dense).

Filters (artist, lang) are applied inside each branch before its LIMIT; filtering after
the merge would leave the top-N filled with other artists' chunks and come back empty
(the ANN post-filter trap, see the pgvector README, "Filtering").
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from genius_rag.chunking.docs import detect_lang
from genius_rag.db.pg import Conn, Hit, fts_search, knn_search
from genius_rag.embeddings.encoder import embed_query

Mode = Literal["hybrid", "dense", "fts"]

# K from Cormack et al. 2009: flattens the top ranks (1/61 vs 1/62 are nearly equal).
RRF_K = 60
# Candidates per branch before fusion. Larger than k, so a chunk from the fts top-30 can
# surface after fusion even if it was not in the dense top-5.
CANDIDATES = 30
# Over-fetch factor: collapse_by_annotation drops overlap chunks of the same annotation.
MULT_K = 5


@dataclass(frozen=True)
class HybridHit:
    chunk_id: int
    annotation_id: int
    text: str
    dense_score: float | None
    fts_score: float | None
    rrf: float


def hybrid_search(
    conn: Conn,
    query_vec: np.ndarray,
    query: str,
    k: int = 5,
    *,
    artist: str | None = None,
    lang: str | None = None,
) -> list[HybridHit]:
    """Top-k chunks by RRF(dense, fts) in one query, with filters inside both branches.

    dense_score / fts_score are None when that branch did not return the chunk.
    Ordered by rrf DESC.
    """
    tsconfig = "russian" if detect_lang(query) == "ru" else "english"

    # Filters are appended as bare predicates only when set. The optional-filter idiom
    # `(%s IS NULL OR cond)` hides selectivity from the planner, which then picks HNSW and
    # filters after the scan: 0 of 26 Daft Punk chunks came back that way.
    lang_filter = "cc.lang = %(lang)s"
    artist_filter = """cc.annotation_id IN (SELECT ann.id
    FROM annotations ann
    JOIN song_artists sa ON ann.song_id = sa.song_id
    JOIN artists ar ON ar.id = sa.artist_id
    WHERE ar.name = %(artist)s)"""

    filters = []
    if artist:
        filters.append("AND " + artist_filter)
    if lang:
        filters.append("AND " + lang_filter)
    filter_sql = " ".join(filters)

    rows = conn.execute(
        f"""
      WITH dense AS (
        SELECT
        cc.id,
        cc.annotation_id,
        cc.text,
        1 - (cc.embedding <=> %(query_vec)s) as score,
        dense_rank() OVER (ORDER BY cc.embedding <=> %(query_vec)s) as rank
        FROM chunks cc
        WHERE cc.embedding IS NOT NULL
        {filter_sql}
        ORDER BY cc.embedding <=> %(query_vec)s
        LIMIT %(candidates)s
      ),
      fts AS (
      SELECT
      cc.id,
      cc.annotation_id,
      cc.text,
      ts_rank_cd(tsv, websearch_to_tsquery(%(tsconfig)s::regconfig, %(query)s)) as score,
      dense_rank()
      OVER (ORDER BY ts_rank_cd(tsv, websearch_to_tsquery(%(tsconfig)s::regconfig, %(query)s)) DESC)
      as rank
      FROM chunks cc
      WHERE tsv @@ websearch_to_tsquery(%(tsconfig)s::regconfig, %(query)s)
      {filter_sql}
      ORDER BY score DESC
      LIMIT %(candidates)s
      )
      SELECT
      COALESCE(dense.id, fts.id) as chunk_id,
      COALESCE(dense.annotation_id, fts.annotation_id) as annotation_id,
      COALESCE(dense.text, fts.text) as text,
      dense.score AS dense_score,
      fts.score AS fts_score,
      COALESCE(1.0 / (%(rrf_k)s + dense.rank), 0) + COALESCE(1.0 / (%(rrf_k)s + fts.rank), 0) as rrf
      FROM dense FULL JOIN fts ON dense.id = fts.id
      ORDER BY rrf DESC
      LIMIT %(k)s
      """,
        {
            "artist": artist,
            "lang": lang,
            "query_vec": query_vec,
            "candidates": CANDIDATES,
            "tsconfig": tsconfig,
            "query": query,
            "rrf_k": RRF_K,
            "k": k,
        },
    ).fetchall()

    return [HybridHit(*r) for r in rows]


def to_hybrid_hit_rrf(rows: list[Hit], mode: Mode, k: int) -> list[HybridHit]:
    """Wrap single-branch results as HybridHit: own score set, the other branch None.

    rrf is computed from list position (1/(RRF_K + rank)) so every mode shares one scale
    and dense / fts / hybrid can be compared directly.
    """
    hits = []

    for idx, r in enumerate(rows, start=1):
        rrf = 1.0 / (RRF_K + idx)
        hits.append(
            HybridHit(
                chunk_id=r.chunk_id,
                annotation_id=r.annotation_id,
                text=r.text,
                dense_score=r.score if mode == "dense" else None,
                fts_score=r.score if mode == "fts" else None,
                rrf=rrf,
            )
        )

    return collapse_by_annotation(hits, k)


def search(
    conn: Conn,
    query: str,
    k: int = 5,
    *,
    mode: Mode = "hybrid",
    artist: str | None = None,
    lang: str | None = None,
) -> list[HybridHit]:
    """Single retrieval entry point: one hit per annotation, top-k.

    mode="hybrid" -> hybrid_search; "dense" / "fts" -> knn_search / fts_search wrapped as
    HybridHit (the other score None). Branches are over-fetched by MULT_K * k because
    collapse_by_annotation drops overlap chunks of the same annotation.
    Filters are supported in hybrid mode only.
    """
    if mode == "fts":
        if artist or lang:
            raise ValueError(f"artist and lang don't have support for {mode}")

        rows = fts_search(conn, query, k=MULT_K * k)

        return to_hybrid_hit_rrf(rows, mode, k)

    vec = embed_query(query)

    if mode == "hybrid":
        hits = hybrid_search(
            conn, query_vec=vec, query=query, k=MULT_K * k, artist=artist, lang=lang
        )
        return collapse_by_annotation(hits, k)

    rows = knn_search(conn, query_vec=vec, k=MULT_K * k)

    if artist or lang:
        raise ValueError(f"artist and lang don't supported for {mode}")

    return to_hybrid_hit_rrf(rows, mode, k)


def collapse_by_annotation(hits: list[HybridHit], k: int) -> list[HybridHit]:
    """Keep the best chunk per annotation_id, first k. Input order is rrf order.

    Overlapping chunks of one annotation otherwise take several top-k slots
    (e.g. chunks 5967/5969/5970 -> annotation 1264). Answers cite annotations, so one hit each.
    """
    seen = set()
    result = []
    for h in hits:
        if h.annotation_id not in seen:
            seen.add(h.annotation_id)
            result.append(h)

    return result[:k]
