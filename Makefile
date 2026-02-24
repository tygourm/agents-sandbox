init:
	uv sync --frozen

tests: init
	uv run pytest

build: init
	uv build --wheel

write: init
	uv run ruff format --no-cache

check: init
	uv run ruff check --fix --no-cache && uv run ty check --error-on-warning

clean:
	rm -rf .venv/
	find . -type d -name dist -prune -exec rm -rf {} +
	find . -type d -name htmlcov -prune -exec rm -rf {} +
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
