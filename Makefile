.PHONY: setup dev test seed lint format

setup:
	uv sync --all-extras

dev:
	uv run uvicorn src.api.app:app --reload --port 8000

test:
	uv run pytest -v

seed:
	uv run python -m scripts.seed_food_db

lint:
	uv run ruff check .

format:
	uv run ruff format .

chat:
	uv run python -m src.cli.chat
