.PHONY: test lint check
PY = .venv/Scripts/python.exe

test:
	$(PY) -m pytest -m "not slow and not integration"

lint:
	$(PY) -m ruff check src tests eval

check: lint test
