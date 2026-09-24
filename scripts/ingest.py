"""Ingest: fetch artists' songs and annotations from the Genius API into data/raw/<slug>.jsonl.

uv run python scripts/ingest.py "Oxxxymiron" "Kai Angel" ...
"""

import json
import sys
from pathlib import Path
from typing import Any

from genius_rag.ingest.genius_client import GeniusClient

RAW = Path("data/raw")


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def all_artists(song: dict[str, Any]) -> list[str]:
    names = [a["name"] for a in song.get("primary_artists", [])]
    names += [a["name"] for a in song.get("featured_artists", [])]
    return list(dict.fromkeys(names))


def ingest_artist(client: GeniusClient, artist: str, max_songs: int = 5) -> int:
    results = client.search(artist)
    mine = [r for r in results if artist.lower() in r["artist_names"].lower()][:max_songs]
    path = RAW / f"{slugify(artist)}.jsonl"
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in mine:
            song = client.song(r["id"])
            refs = client.referents(r["id"])
            annotations = [
                {"fragment": ref["fragment"], "text": ref["annotations"][0]["body"]["plain"]}
                for ref in refs
                if ref["annotations"] and ref["annotations"][0]["body"]["plain"]
            ]
            record = {
                "song_id": song["id"],
                "title": song["title"],
                "artists": all_artists(song),
                "album": (song.get("album") or {}).get("name"),
                "annotations": annotations,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    artists = sys.argv[1:] or ["Oxxxymiron"]
    RAW.mkdir(parents=True, exist_ok=True)
    with GeniusClient() as client:
        for artist in artists:
            n = ingest_artist(client, artist)
            print(f"ingested {n} songs for {artist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
