"""Annotation -> retrieval documents (chunks).

One annotation is one parent. It yields 1..N child chunks, each carrying a context prefix
(artists / track / line fragment) plus a slice of the annotation text. Retrieval matches
children; answers cite the parent (annotations.id).

Chunking parameters are function arguments with defaults, not constants: the eval stage
runs A/B comparisons over them.
"""

import re
from dataclasses import dataclass

from genius_rag.chunking.tokens import count_tokens

MAX_TOKENS = 256  # child default; e5 accepts 512, headroom left for the prefix
OVERLAP_SENTENCES = 1  # trailing sentences repeated at the start of the next chunk


@dataclass(frozen=True)
class AnnotationCtx:
    """One DB row with everything the prefix needs. Built in db/pg.py."""

    annotation_id: int
    song_id: int
    artists: str  # "9mice, Kai Angel", already joined by SQL in credit order
    title: str
    album: str | None
    fragment: str
    text: str


@dataclass(frozen=True)
class Chunk:
    annotation_id: int
    position: int  # index within the annotation, from 0
    lang: str  # "ru" | "en"
    text: str  # prefix + slice; this is what gets embedded and indexed
    tokens: int


def detect_lang(s: str) -> str:
    """'ru' if Cyrillic makes up more than half of the letters, else 'en'."""
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return "en"

    ratio = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in letters) / len(letters)

    return "ru" if ratio > 0.5 else "en"


def make_prefix(ctx: AnnotationCtx) -> str:
    """Context header of a chunk: track (album) - artists : line fragment."""
    return f"{ctx.title}{f' ({ctx.album})' if ctx.album else ''} - {ctx.artists} : {ctx.fragment}"


def split_sentences(text: str) -> list[str]:
    """Text -> sentences. Paragraphs (\\n\\n) are hard boundaries; inside, split on .!? + space."""
    paragraphs = text.split("\n\n")

    res = []

    for paragraph in paragraphs:
        res.extend(re.split(r"(?<=[.!?])\s+", paragraph))

    return [p.strip() for p in res if p.strip()]


def pack_sentences(
    sentences: list[str],
    budget: int,
    overlap: int = OVERLAP_SENTENCES,
) -> list[str]:
    """Greedily join sentences into pieces of at most `budget` tokens.

    A new piece starts with the last `overlap` sentences of the closed one (context across the
    boundary) when they fit together with the next sentence. A sentence longer than the budget
    becomes a piece of its own.
    """
    packs = []
    current: list[str] = []

    for sentence in sentences:
        if current and count_tokens(" ".join(current + [sentence])) > budget:
            packs.append(" ".join(current))
            tail = current[-overlap:] if overlap else []
            current = tail if count_tokens(" ".join(tail + [sentence])) < budget else []
        current.append(sentence)

    if len(current):
        packs.append(" ".join(current))

    return packs


def build_chunks(ctx: AnnotationCtx, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """Annotation -> list of Chunk; the text budget is max_tokens minus the prefix."""
    prefix = make_prefix(ctx)

    lang = detect_lang(ctx.text)

    budget = max_tokens - count_tokens(prefix)

    pieces = pack_sentences(split_sentences(ctx.text), budget)

    return [
        Chunk(
            annotation_id=ctx.annotation_id,
            position=idx,
            lang=lang,
            text=prefix + "\n" + text,
            tokens=count_tokens(prefix + "\n" + text),
        )
        for idx, text in enumerate(pieces)
    ]
