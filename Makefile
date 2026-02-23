# cohorte library – common targets (uv)
# Use from cohorte directory: make lint, make test, make build, etc.
#
# Publish (uv publish):
#   make publish-dev   → TestPyPI (UV_PUBLISH_TOKEN=<testpypi-token>)
#   make publish       → PyPI prod (UV_PUBLISH_TOKEN=<pypi-token>)

UV ?= uv
VENV ?= .venv

.PHONY: help venv lock install-dev lint test coverage build clean publish publish-dev

help:
	@echo "Targets: venv, lock, install-dev, lint, test, coverage, build, clean, publish, publish-dev"

venv:
	$(UV) venv $(VENV)

lock:
	$(UV) lock

install-dev: venv
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest tests

coverage:
	$(UV) run pytest tests --cov=cohorte --cov-report=term-missing

build:
	$(UV) run python -m build

clean:
	rm -rf build/ dist/ *.egg-info .eggs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Publish to TestPyPI (dev). Requires: UV_PUBLISH_TOKEN=<testpypi-token>
publish-dev: build
	$(UV) publish --index testpypi

# Publish to PyPI (production). Requires: UV_PUBLISH_TOKEN=<pypi-token>
publish: build
	UV_PUBLISH_TOKEN=$(shell cat .token) $(UV) publish
