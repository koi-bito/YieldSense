.PHONY: install ingest stats train report test lint format run docker-build docker-up clean

PYTHON := python
PIP := pip
PYTEST := pytest
UVICORN := uvicorn

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	pre-commit install

ingest:
	$(PYTHON) -m src.ingestion

stats:
	$(PYTHON) -m src.analysis

train:
	$(PYTHON) -m src.models

pipeline: ingest stats train

report:
	$(PYTHON) src/generate_report.py

run:
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTEST) --cov=src --cov-report=term-missing tests/

lint:
	flake8 src tests --count --show-source --statistics

format:
	black src tests
	isort src tests

docker-build:
	docker build -t yieldsense:latest .

docker-up:
	docker compose up -d

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov/
