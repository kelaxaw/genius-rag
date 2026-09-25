"""/ask tests on TestClient with search and LLM overridden: no database, no network."""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from genius_rag.api.main import app, get_searcher
from genius_rag.generation.answer import REFUSAL, Answer
from genius_rag.llm.provider import FakeLLM, get_llm
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


class FakeSearcher:
    """Returns preset hits and records the arguments of every call."""

    def __init__(self, hits: list[HybridHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, str | None, str | None]] = []

    def __call__(
        self, question: str, k: int, artist: str | None, lang: str | None
    ) -> list[HybridHit]:
        self.calls.append((question, k, artist, lang))
        return self.hits


def parse_sse(body: str) -> list[tuple[str, Any]]:
    """text/event-stream -> [(event, data)]; events are separated by a blank line."""
    events = []
    for block in body.strip().split("\n\n"):
        event, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                raw = line.removeprefix("data:").strip()
                data = json.loads(raw) if raw else None
        if not block.startswith(":"):
            events.append((event, data))
    return events


@pytest.fixture
def setup() -> Iterator[tuple[TestClient, FakeSearcher, FakeLLM]]:
    searcher = FakeSearcher(HITS)
    llm = FakeLLM(reply=Answer(refused=False, answer="A kiss-off song.", annotation_ids=[209]))
    app.dependency_overrides[get_searcher] = lambda: searcher
    app.dependency_overrides[get_llm] = lambda: llm
    yield TestClient(app), searcher, llm
    app.dependency_overrides.clear()


def test_events_in_order(setup: tuple[TestClient, FakeSearcher, FakeLLM]) -> None:
    client, _, _ = setup
    resp = client.get("/ask", params={"q": "What is Love Yourself about?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in parse_sse(resp.text)]
    assert names == ["sources", "answer", "done"]


def test_sources_are_hits(setup: tuple[TestClient, FakeSearcher, FakeLLM]) -> None:
    client, _, _ = setup
    events = dict(parse_sse(client.get("/ask", params={"q": "love"}).text))
    assert [h["annotation_id"] for h in events["sources"]] == [209, 1033]
    assert events["sources"][0]["text"] == HITS[0].text


def test_answer_is_grounded(setup: tuple[TestClient, FakeSearcher, FakeLLM]) -> None:
    client, _, llm = setup
    events = dict(parse_sse(client.get("/ask", params={"q": "love"}).text))
    assert events["answer"] == {
        "refused": False,
        "answer": "A kiss-off song.",
        "annotation_ids": [209],
    }
    assert len(llm.calls) == 1


def test_params_reach_searcher(setup: tuple[TestClient, FakeSearcher, FakeLLM]) -> None:
    client, searcher, _ = setup
    client.get("/ask", params={"q": "love", "k": 3, "artist": "Justin Bieber", "lang": "en"})
    assert searcher.calls == [("love", 3, "Justin Bieber", "en")]


def test_empty_search_refuses_without_llm(
    setup: tuple[TestClient, FakeSearcher, FakeLLM],
) -> None:
    client, searcher, llm = setup
    searcher.hits = []
    events = dict(parse_sse(client.get("/ask", params={"q": "о чём этот трек?"}).text))
    assert events["sources"] == []
    assert events["answer"]["refused"] is True
    assert events["answer"]["answer"] == REFUSAL["ru"]
    assert llm.calls == []


def test_bad_params_rejected(setup: tuple[TestClient, FakeSearcher, FakeLLM]) -> None:
    client, _, _ = setup
    assert client.get("/ask").status_code == 422
    assert client.get("/ask", params={"q": "x", "lang": "de"}).status_code == 422
    assert client.get("/ask", params={"q": "x", "k": 0}).status_code == 422
