"""Token counter matching the embedding model.

Chunk boundaries must be measured in the model's tokens, not characters. The tokenizer is
the one used by `intfloat/multilingual-e5-small` (512-token input limit).
"""

from functools import lru_cache

from tokenizers import Tokenizer

TOKENIZER_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    # Downloaded once into ~/.cache/huggingface, offline afterwards.
    return Tokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    """Number of model tokens in a string (including the <s> </s> special tokens)."""
    return len(_tokenizer().encode(text).ids)
