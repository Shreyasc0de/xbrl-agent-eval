.PHONY: install doctor lint typecheck test check data quarters filings questions clean

# The first supported interpreter found. Override with: make install PYTHON=python3.11
PYTHON ?= $(shell for p in python3.12 python3.11 python3.10; do command -v $$p >/dev/null 2>&1 && echo $$p && break; done)

install:
	@if [ -z "$(PYTHON)" ]; then \
		echo "No supported Python found. This project needs 3.10, 3.11 or 3.12."; \
		echo "The pinned dependencies have no wheels for newer versions, so pip"; \
		echo "would try to build them from source and fail."; \
		echo ""; \
		echo "Install one (brew install python@3.12) or point at an existing one:"; \
		echo "  make install PYTHON=/path/to/python3.12"; \
		exit 1; \
	fi
	@echo "using $(PYTHON) ($$($(PYTHON) --version))"
	rm -rf .venv
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo ""
	@.venv/bin/python -c "import ingest, questions, agents, eval, core; print('packages importable: ok')"

# Says what is set up and what is not, without needing the package to import.
doctor:
	@echo "python on PATH:   $$(command -v python3 || echo none) $$(python3 --version 2>&1)"
	@echo "chosen for venv:  $(PYTHON)"
	@if [ -d .venv ]; then \
		echo "venv:             .venv, $$(basename $$(ls -d .venv/lib/python* 2>/dev/null | head -1))"; \
		if .venv/bin/python -c "import ingest" >/dev/null 2>&1; then \
			echo "packages:         importable"; \
		else \
			echo "packages:         NOT INSTALLED, run make install"; \
		fi; \
	else \
		echo "venv:             missing, run make install"; \
	fi
	@if [ -f .env ]; then echo "env file:         .env present"; else echo "env file:         missing, cp .env.example .env"; fi
	@if [ -f data/xbrl.db ]; then echo "ground truth:     data/xbrl.db"; else echo "ground truth:     missing, run make quarters"; fi
	@if [ -f data/filings/manifest.jsonl ]; then \
		echo "filings:          $$(wc -l < data/filings/manifest.jsonl | tr -d ' ') cached"; \
	else echo "filings:          missing, run make filings"; fi
	@if [ -f data/questions/questions.jsonl ]; then \
		echo "questions:        $$(wc -l < data/questions/questions.jsonl | tr -d ' ')"; \
	else echo "questions:        missing, run make questions"; fi
	@if command -v ollama >/dev/null 2>&1; then \
		echo "ollama:           $$(ollama list 2>/dev/null | tail -n +2 | wc -l | tr -d ' ') model(s)"; \
	else echo "ollama:           not on PATH"; fi

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

# Runs on the host machine only. SEC is not reachable from the dev sandbox.
# Requires SEC_CONTACT_EMAIL in .env: SEC rejects requests without a contact.
data: quarters filings questions

quarters:
ifndef QUARTERS
	$(error QUARTERS is not set, for example: make data QUARTERS=2024q1,2024q2)
endif
	python -m ingest.dera_loader --quarters $(QUARTERS)

filings:
	python -m ingest.filings

questions:
	python -m questions.generate

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
