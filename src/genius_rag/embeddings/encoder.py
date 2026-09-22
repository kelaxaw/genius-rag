"""Text encoder: string -> 384-dim float vector.

`intfloat/multilingual-e5-small` (ru+en, CPU-friendly). E5 was trained with the
`query: ` / `passage: ` prefixes, so they are required; they are added at encode time
only and never stored. Vectors are L2-normalized, so pgvector's `<=>` is cosine distance.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from genius_rag.config import settings

MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # ~470 MB, downloaded once into ~/.cache/huggingface; offline afterwards.
    return SentenceTransformer(MODEL_NAME, device="cpu", token=settings.hf_token.get_secret_value())


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Document vectors, shape (len(texts), DIM); row order matches input order."""
    with_prefix = [PASSAGE_PREFIX + t for t in texts]

    return _model().encode(inputs=with_prefix, batch_size=batch_size, normalize_embeddings=True)


def embed_query(text: str) -> np.ndarray:
    """Query vector, shape (DIM,)."""
    with_prefix = QUERY_PREFIX + text

    return _model().encode(inputs=with_prefix, normalize_embeddings=True)
