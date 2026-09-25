"""Grounded answer tests on FakeLLM: no database, no network. Run: make test."""

from genius_rag.generation.answer import (
    REFUSAL,
    SYSTEM,
    Answer,
    answer_question,
    format_context,
)
from genius_rag.llm.provider import FakeLLM
from genius_rag.retrieval.hybrid import HybridHit

HITS = [
    HybridHit(
        chunk_id=4642,
        annotation_id=209,
        text="Love Yourself (Purpose) - Justin Bieber : you should go and love yourself",
        dense_score=0.82,
        fts_score=0.3,
        rrf=0.031,
    ),
    HybridHit(
        chunk_id=5714,
        annotation_id=1033,
        text="Die For You (Starboy) - The Weeknd : I can't afford love",
        dense_score=0.82,
        fts_score=None,
        rrf=0.016,
    ),
]
Q_EN = "What is Love Yourself about?"
Q_RU = "Что значит строчка про Love Yourself?"


def fake(refused: bool = False, answer: str = "A kiss-off song.", ids: list[int] | None = None):
    return FakeLLM(reply=Answer(refused=refused, answer=answer, annotation_ids=ids or []))


def test_system_prompt_written() -> None:
    assert SYSTEM.strip(), "SYSTEM is empty"


def test_context_has_ids_texts_and_question() -> None:
    ctx = format_context(Q_EN, HITS)
    assert "209" in ctx and "1033" in ctx
    assert HITS[0].text in ctx and HITS[1].text in ctx
    assert Q_EN in ctx


def test_no_hits_refuses_without_calling_llm() -> None:
    llm = fake(ids=[209])
    result = answer_question(Q_RU, [], llm)
    assert result.refused
    assert result.answer == REFUSAL["ru"]
    assert result.annotation_ids == []
    assert llm.calls == [], "LLM must not be called when nothing was retrieved"


def test_llm_gets_system_and_context() -> None:
    llm = fake(ids=[209])
    answer_question(Q_EN, HITS, llm)
    assert llm.calls == [(SYSTEM, format_context(Q_EN, HITS))]


def test_grounded_answer_kept() -> None:
    result = answer_question(Q_EN, HITS, fake(ids=[209]))
    assert not result.refused
    assert result.answer == "A kiss-off song."
    assert result.annotation_ids == [209]


def test_unknown_ids_dropped() -> None:
    result = answer_question(Q_EN, HITS, fake(ids=[209, 999]))
    assert not result.refused
    assert result.annotation_ids == [209]


def test_no_valid_citation_means_refusal() -> None:
    result = answer_question(Q_EN, HITS, fake(ids=[999]))
    assert result.refused
    assert result.answer == REFUSAL["en"]
    assert result.annotation_ids == []


def test_model_refusal_respected() -> None:
    result = answer_question(Q_RU, HITS, fake(refused=True, answer="whatever", ids=[209]))
    assert result.refused
    assert result.answer == REFUSAL["ru"]
    assert result.annotation_ids == []
