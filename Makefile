.PHONY: install lint typecheck test check data quarters filings questions clean

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

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
