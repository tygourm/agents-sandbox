init:
	uv sync --frozen

start: init
	uv run agents-sandbox

tests: init
	uv run pytest -s -vv

build: init
	uv build --wheel

write: init
	uv run ruff format --no-cache

check: init
	uv run ruff check --fix --no-cache
	uv run ty check --error-on-warning

clean:
	rm -rf .venv/ dist htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
