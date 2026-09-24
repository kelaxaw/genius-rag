"""Genius API HTTP client with auth, retries and pagination."""

from __future__ import annotations

import time
from typing import Any

import httpx

from genius_rag.config import settings

BASE_URL = "https://api.genius.com"


class GeniusClient:
    def __init__(self, token: str | None = None, timeout: float = 15.0) -> None:
        secret = token or settings.genius_access_token.get_secret_value()
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {secret}"},
            timeout=timeout,
        )

    def _get(
        self, path: str, params: dict[str, Any] | None = None, max_retries: int = 4
    ) -> dict[str, Any]:
        """GET with exponential backoff on 429/5xx. Returns the "response" field of the body."""
        resp: httpx.Response | None = None
        for attempt in range(max_retries):
            resp = self._client.get(path, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp.json()["response"]
        assert resp is not None
        resp.raise_for_status()
        return resp.json()["response"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GeniusClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def search(self, query: str) -> list[dict[str, Any]]:
        res = self._get("/search", {"q": query})
        return [hit["result"] for hit in res["hits"]]

    def song(self, song_id: int) -> dict[str, Any]:
        res = self._get(f"/songs/{song_id}", {"text_format": "plain"})
        return res["song"]

    def referents(self, song_id: int, per_page: int = 50) -> list[dict[str, Any]]:
        referents: list[dict[str, Any]] = []
        page = 1
        while True:
            res = self._get(
                "/referents",
                {"song_id": song_id, "page": page, "per_page": per_page, "text_format": "plain"},
            )
            batch = res["referents"]
            if not batch:
                break
            referents.extend(batch)
            page += 1
        return referents
