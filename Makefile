.PHONY: install up down lint typecheck check load psql chunks embed search llm test ask api

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

# make search q="question" mode=hybrid k=5 artist="Drake" lang=ru
mode ?= hybrid
k ?= 5
search:
	uv run python scripts/search.py "$(q)" --mode $(mode) -k $(k) \
		$(if $(artist),--artist "$(artist)") $(if $(lang),--lang $(lang))

# make llm q="who is Oxxxymiron?" [fake=1]
llm:
	uv run python scripts/llm_ping.py "$(q)" $(if $(fake),--fake)

test:
	uv run pytest -q

# make ask q="what is Love Yourself about?" k=5 artist="Justin Bieber" lang=en [api=1]
ask:
	uv run python scripts/demo_ask.py "$(q)" -k $(k) \
		$(if $(artist),--artist "$(artist)") $(if $(lang),--lang $(lang)) $(if $(api),--api)

# SSE server for make ask api=1; --reload restarts on code changes
api:
	uv run uvicorn genius_rag.api.main:app --reload --port 8000
