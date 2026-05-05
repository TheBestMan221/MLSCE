# ─────────────────────────────────────────────────────────
# Makefile  –  F1 ML Prediction System
# Windows users: use the equivalent commands listed in README
# ─────────────────────────────────────────────────────────

PYTHON     = python
PIP        = pip
STREAMLIT  = streamlit
PYTEST     = pytest
DVC        = dvc
DOCKER     = docker
COMPOSE    = docker-compose
APP        = app.py
IMAGE_NAME = f1-ml-app

.PHONY: help install train evaluate run test dvc-init dvc-repro \
        docker-build docker-run docker-stop clean lint

help:
	@echo ""
	@echo "  F1 ML PROJECT — MAKE COMMANDS"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install       Install Python dependencies"
	@echo "  make train         Train all ML models"
	@echo "  make evaluate      Run evaluation & print report"
	@echo "  make run           Launch Streamlit app"
	@echo "  make test          Run all pytest tests"
	@echo "  make dvc-init      Initialize DVC in this repo"
	@echo "  make dvc-repro     Run full DVC pipeline"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-stop   Stop Docker container"
	@echo "  make clean         Remove __pycache__ files"
	@echo "  ─────────────────────────────────────────"
	@echo ""

# ── Python setup ────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── ML training ─────────────────────────────────────────
train:
	$(PYTHON) src/train_models.py

# ── Evaluation report ────────────────────────────────────
evaluate:
	$(PYTHON) src/evaluate_models.py

# ── Run Streamlit app ────────────────────────────────────
run:
	$(STREAMLIT) run $(APP)

# ── Tests ────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short

# ── DVC ─────────────────────────────────────────────────
dvc-init:
	$(DVC) init
	@echo "DVC initialised. Add a remote with: dvc remote add -d myremote <path>"

dvc-repro:
	$(DVC) repro

dvc-status:
	$(DVC) status

# ── Docker ──────────────────────────────────────────────
docker-build:
	$(DOCKER) build -t $(IMAGE_NAME) .

docker-run:
	$(COMPOSE) up -d
	@echo "App running at http://localhost:8501"

docker-stop:
	$(COMPOSE) down

# ── Cleanup ─────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache cleaned."
