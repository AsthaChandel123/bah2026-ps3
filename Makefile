# aqi_india — developer convenience targets.
#
# Usage: make <target>. The light dependency set is enough for setup/test/demo;
# heavy extras (deep, gee, serve, geo, transport) are installed on demand.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PIP    := $(BIN)/pip
PY     := $(BIN)/python

.DEFAULT_GOAL := help

.PHONY: help setup setup-dev test test-core lint typecheck format demo clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create the virtual environment.
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV) ## Install the package with the light dependency set (editable).
	$(PIP) install -e .

setup-dev: $(VENV) ## Install with dev tooling (pytest, ruff, mypy, hypothesis).
	$(PIP) install -e ".[dev]"

test: ## Run the full test suite.
	$(BIN)/pytest -q

test-core: ## Run only the pure core tests (NAQI, H3, metrics).
	$(BIN)/pytest -q tests/test_naqi.py tests/test_h3.py tests/test_metrics.py

lint: ## Lint with ruff.
	$(BIN)/ruff check src tests

format: ## Auto-format with ruff.
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck: ## Static type-check with mypy.
	$(BIN)/mypy src

demo: ## Run the light synthetic end-to-end demo.
	$(BIN)/aqi demo run

clean: ## Remove caches and build artifacts (keeps data skeleton and venv).
	rm -rf build dist *.egg-info src/*.egg-info \
		.pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
