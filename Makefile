.PHONY: help install dev test lint format typecheck migrate seed docker-up docker-down

help:
	@echo "LEO AI Backend — Developer Commands"
	@echo "  install    : Install dependencies with pip"
	@echo "  dev        : Run FastAPI development server with reload"
	@echo "  test       : Run pytest test suite"
	@echo "  lint       : Run ruff check"
	@echo "  format     : Format code with ruff"
	@echo "  typecheck  : Run mypy type checker"
	@echo "  migrate    : Run alembic migrations"
	@echo "  seed       : Seed verified knowledge base"
	@echo "  docker-up  : Start containers with Docker Compose"
	@echo "  docker-down: Stop containers"

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy app

migrate:
	alembic upgrade head

seed:
	python scripts/seed_knowledge.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down
