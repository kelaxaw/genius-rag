.PHONY: install up down lint typecheck check

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
