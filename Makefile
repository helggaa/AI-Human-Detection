.PHONY: help install install-dev test lint format run train evaluate export clean

help:
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run test suite with pytest"
	@echo "  make lint         Run Ruff linter"
	@echo "  make format       Run Black formatter"
	@echo "  make run          Launch Streamlit web application"
	@echo "  make train        Execute model training pipeline"
	@echo "  make evaluate     Execute model evaluation pipeline"
	@echo "  make export       Export trained model to TFLite format"
	@echo "  make clean        Remove build, cache, and temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest -v

lint:
	ruff check .

format:
	black .

run:
	streamlit run app.py

train:
	python -m src.trainer

evaluate:
	python -m src.evaluator

export:
	python -m src.exporter

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
