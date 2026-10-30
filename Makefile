.PHONY: setup lint format test cov data eval clean

setup:        ## create venv + install all extras (tests need them)
	uv sync --all-extras

lint:         ## ruff lint
	uv run ruff check .

format:       ## ruff format
	uv run ruff format .

test:         ## run tests
	uv run pytest

cov:          ## tests + coverage
	uv run pytest --cov --cov-report=term-missing

data:         ## clean a manually downloaded RecipeNLG CSV (research only, ADR 0003)
	uv run python scripts/download_data.py

eval:         ## run Phase 1 evaluation harness
	uv run python -m pantrychef.eval

clean:        ## remove caches
	rm -rf .pytest_cache .ruff_cache .coverage **/__pycache__
