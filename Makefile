.PHONY: setup lint format test cov data train eval serve clean

setup:        ## create venv + install deps (incl dev)
	uv sync --extra dev

lint:         ## ruff lint
	uv run ruff check .

format:       ## ruff format
	uv run ruff format .

test:         ## run tests
	uv run pytest

cov:          ## tests + coverage
	uv run pytest --cov --cov-report=term-missing

data:         ## download/prepare datasets (Phase 1)
	uv run python scripts/download_data.py

train:        ## train models (per-phase; see configs/)
	@echo "Not implemented until Phase 2."

eval:         ## run Phase 1 evaluation harness
	uv run python -m pantrychef.eval

serve:        ## run FastAPI app (Phase 6)
	@echo "Not implemented until Phase 6."

clean:        ## remove caches
	rm -rf .pytest_cache .ruff_cache .coverage **/__pycache__
