.PHONY: install up down lint typecheck check load psql chunks embed search

install:
	uv sync

up:
	docker compose up -d --wait

down:
	docker compose down

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy src scripts

check: lint typecheck
	uv run python scripts/check_env.py

load:
	uv run python scripts/load_db.py

psql:
	docker compose exec postgres psql -U genius -d genius

chunks:
	uv run python scripts/build_chunks.py

embed:
	uv run python scripts/build_embeddings.py

# make search q="question" mode=fts k=5
mode ?= dense
k ?= 5
search:
	uv run python scripts/search.py "$(q)" --mode $(mode) -k $(k)
