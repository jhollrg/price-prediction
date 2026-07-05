PYTHON := python
VENV   := .venv

ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV)/Scripts/python
    VENV_PIP    := $(VENV)/Scripts/pip
    VENV_PYTEST := $(VENV)/Scripts/pytest
    VENV_BLACK  := $(VENV)/Scripts/black
    VENV_ISORT  := $(VENV)/Scripts/isort
else
    VENV_PYTHON := $(VENV)/bin/python
    VENV_PIP    := $(VENV)/bin/pip
    VENV_PYTEST := $(VENV)/bin/pytest
    VENV_BLACK  := $(VENV)/bin/black
    VENV_ISORT  := $(VENV)/bin/isort
endif

# Put src/ on PYTHONPATH so `python -m src.<module>` can resolve sibling imports
# (e.g. `from config import ...` inside src/train.py).
export PYTHONPATH := src

.PHONY: setup data db train test clean lint

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	mkdir -p data/raw reports/figures models notebooks

data:
	$(VENV_PYTHON) scripts/download_data.py

db:
	$(VENV_PYTHON) -m src.data_loader

train:
	$(VENV_PYTHON) -m src.train

test:
	$(VENV_PYTEST) tests/ -v --tb=short

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf models/

lint:
	$(VENV_BLACK) src/ tests/
	$(VENV_ISORT) src/ tests/
