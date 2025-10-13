
.PHONY: help setup lint test docs

help:
	@echo "make setup       # install deps (pip)"
	@echo "make lint        # run basic lint (flake8 if present)"
	@echo "make test        # run tests (if any)"
	@echo "make docs        # build docs (if any)"

setup:
	python -m pip install -r requirements.txt || true

lint:
	@echo "No linter configured yet."

test:
	@echo "No tests yet."

docs:
	@echo "No docs pipeline yet."
