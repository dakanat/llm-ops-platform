.PHONY: up down logs lint format typecheck test test-integration test-all test-no-llm coverage migrate seed eval

# 開発環境
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f app

# コード品質
lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-all:
	uv run pytest tests/ -v

test-no-llm:
	uv run pytest -m "not llm"

coverage:
	uv run pytest --cov=src --cov-report=html

# DB
migrate:
	uv run alembic upgrade head

seed:
	uv run python scripts/seed_data.py

# 評価
eval:
	uv run python scripts/run_eval.py
