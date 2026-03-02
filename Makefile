# portazgo library – common targets (uv)
# Use from portazgo directory: make lint, make test, make build, etc.
#
# Publish (uv publish):
#   make publish-dev   → TestPyPI (UV_PUBLISH_TOKEN=<testpypi-token>)
#   make publish       → PyPI prod; prompts for patch/minor/major bump, then publishes (UV_PUBLISH_TOKEN=<pypi-token>)

UV ?= uv
VENV ?= .venv

.PHONY: help venv lock install-dev lint test coverage build clean bump-version publish publish-dev

help:
	@echo "Targets: venv, lock, install-dev, lint, test, coverage, build, clean, bump-version, publish, publish-dev"

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
	$(UV) run pytest tests --cov=portazgo --cov-report=term-missing

build: install-dev
	$(UV) run python -m build

clean:
	rm -rf build/ dist/ *.egg-info .eggs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Checks if there are any uncommitted changes and if so, prints a warning and exits.
check-changes:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: there are uncommitted changes. Please commit or stash them first."; \
		exit 1; \
	fi

# Interactively bump version in pyproject.toml. Prompts for patch, minor, or major.
bump-version:
	@VERSION=$$(grep '^version = ' pyproject.toml | awk -F'"' '{print $$2}'); \
	echo "Current version: $$VERSION"; \
	echo "Bump type: [p]atch (0.1.2 -> 0.1.3), [m]inor (0.1.2 -> 0.2.0), [M]ajor (0.1.2 -> 1.0.0)"; \
	printf "Choice [p]: "; read BUMP; BUMP=$${BUMP:-p}; \
	MAJOR=$$(echo $$VERSION | cut -d. -f1); \
	MINOR=$$(echo $$VERSION | cut -d. -f2); \
	PATCH=$$(echo $$VERSION | cut -d. -f3); \
	case $$BUMP in \
		[M]|[Mm]ajor) MAJOR=$$((MAJOR + 1)); MINOR=0; PATCH=0 ;; \
		[m]|[m]inor) MINOR=$$((MINOR + 1)); PATCH=0 ;; \
		*) PATCH=$$((PATCH + 1)) ;; \
	esac; \
	NEW_VERSION="$$MAJOR.$$MINOR.$$PATCH"; \
	echo "Bumping to $$NEW_VERSION"; \
	sed -i.bak "s/^version = \".*\"/version = \"$$NEW_VERSION\"/" pyproject.toml && rm pyproject.toml.bak

# Publish to TestPyPI (dev). Requires: UV_PUBLISH_TOKEN=<testpypi-token>
publish-dev: build
	$(UV) publish --index testpypi

# Publish to PyPI (production). Requires: UV_PUBLISH_TOKEN=<pypi-token>
# Prompts for version bump (patch/minor/major), then builds, publishes, commits, tags, and pushes.
publish: check-changes bump-version clean build
	@VERSION=$$(grep '^version = ' pyproject.toml | awk -F'"' '{print $$2}'); \
	if [ -f .token ]; then \
		UV_PUBLISH_TOKEN=$$(cat .token) $(UV) publish --verbose; \
		echo "Published version $$VERSION to PyPI."; \
	else \
		echo "Error: .token file not found. Please create it and add your PyPI token."; \
		exit 1; \
	fi; \
	git add pyproject.toml && \
	git commit -m "Bump version to $$VERSION" && \
	git tag $$VERSION && \
	git push origin HEAD && \
	git push origin $$VERSION && \
	echo "Published version $$VERSION to PyPI and tagged the release."