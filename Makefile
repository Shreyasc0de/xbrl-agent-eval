.PHONY: install doctor lint typecheck test check data quarters filings questions passages results clean

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
	@.venv/bin/python -c "import ingest, questions, annotate, agents, eval, core; print('packages importable: ok')"

# Runs on the host machine only. SEC is not reachable from the dev sandbox.
# Requires SEC_CONTACT_EMAIL in .env: SEC rejects requests without a contact.
#
# DB deliberately defaults outside the repository. A multi-gigabyte SQLite
# write lasting several minutes is the workload iCloud, Dropbox and similar
# services interact with worst, and a repository on the Desktop sits inside
# one. A file rewritten underneath SQLite mid-write kills the process with a
# bus error that says nothing about its cause.
# Matches core.paths.default_db_path, including the XBRL_DB override, so the
# Makefile and the command line entry points cannot disagree about where the
# database is. They already did once.
DB ?= $(if $(XBRL_DB),$(XBRL_DB),$(HOME)/xbrl-data/xbrl.db)

# Everything else the pipeline writes lives beside the database, which is what
# core.paths does. make doctor looked for it under data/ in the checkout
# instead, so it reported filings and questions missing while both existed.
# That is the same divergence this DB variable was added to fix, one level
# down, and it stayed invisible because doctor only ever prints.
DATA_DIR := $(dir $(DB))

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
	@if [ -f "$(DB)" ]; then \
		echo "ground truth:     $(DB) ($$(du -h "$(DB)" | cut -f1))"; \
	else echo "ground truth:     missing at $(DB), run make quarters"; fi
	@if [ -f "$(DB)" ] && command -v sqlite3 >/dev/null 2>&1; then \
		q=$$(sqlite3 -cmd ".timeout 2000" "$(DB)" "SELECT IFNULL(GROUP_CONCAT(q, ','), 'none') FROM (SELECT DISTINCT quarter AS q FROM load_log ORDER BY quarter)" 2>/dev/null); \
		if [ -n "$$q" ]; then \
			echo "quarters loaded:  $$q"; \
		else \
			echo "quarters loaded:  cannot read, database is busy (a load is probably running)"; \
		fi; \
	fi
	@if [ -f "$(DATA_DIR)filings/manifest.jsonl" ]; then \
		echo "filings:          $$(wc -l < "$(DATA_DIR)filings/manifest.jsonl" | tr -d ' ') cached"; \
	else echo "filings:          missing at $(DATA_DIR)filings, run make filings"; fi
	@if [ -f "$(DATA_DIR)questions/questions.jsonl" ]; then \
		echo "questions:        $$(wc -l < "$(DATA_DIR)questions/questions.jsonl" | tr -d ' ')"; \
	else echo "questions:        missing at $(DATA_DIR)questions, run make questions"; fi
	@if [ -f "$(DATA_DIR)questions/passages.jsonl" ]; then \
		echo "passages:         $$(wc -l < "$(DATA_DIR)questions/passages.jsonl" | tr -d ' ') annotated"; \
	else echo "passages:         missing at $(DATA_DIR)questions, run make passages"; fi
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

data: quarters filings questions passages

quarters:
ifndef QUARTERS
	$(error QUARTERS is not set, for example: make data QUARTERS=2024q1,2024q2)
endif
	python -m ingest.dera_loader --quarters $(QUARTERS) --db $(DB)

filings:
	python -m ingest.filings --db $(DB)

questions:
	python -m questions.generate --db $(DB)

passages:
	python -m annotate.run

results:
	python -m report.verify_results --write

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
