PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

NOTEBOOKS := notebooks/01_failures.ipynb \
             notebooks/02_structured_composition.ipynb \
             notebooks/03_audit.ipynb \
             notebooks/04_ablations.ipynb

FIGURES_DIR := results/figures

.PHONY: setup test lint typecheck repro notebooks clean help

help:
	@echo "Targets:"
	@echo "  setup       Install pinned deps from requirements-dev.txt"
	@echo "  test        Run pytest -v"
	@echo "  lint        Run ruff check on src/ tests/"
	@echo "  typecheck   Run mypy --strict on src/"
	@echo "  notebooks   Execute all notebooks headless in-place"
	@echo "  repro       Run tests + notebooks + dump figures to $(FIGURES_DIR)"
	@echo "  clean       Remove caches, results/, ipynb_checkpoints"

setup:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

typecheck:
	$(PYTHON) -m mypy --strict src/

notebooks: $(FIGURES_DIR)
	@for nb in $(NOTEBOOKS); do \
		echo "[nbconvert] $$nb"; \
		$(PYTHON) -m jupyter nbconvert --to notebook --execute "$$nb" \
			--inplace --ExecutePreprocessor.timeout=600; \
	done

$(FIGURES_DIR):
	mkdir -p $(FIGURES_DIR)

repro: test notebooks
	@echo
	@echo "make repro complete. Figures in $(FIGURES_DIR)/"
	@ls -1 $(FIGURES_DIR) 2>/dev/null || echo "(no figures yet)"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf results
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
