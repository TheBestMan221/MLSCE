# 🏎️ Formula 1 Race Prediction & Strategy Recommendation System
### MLOps Edition — DVC · CI/CD · Docker · Experiment Tracking · Model Registry

---

## 📋 Project Overview

A fully functional ML-powered web application that:
- Predicts **Top 10 finish probability** using 5 classifiers
- Predicts **final finishing position** using 3 regressors
- Estimates **pole position probability** per qualifying slot
- Runs **Monte Carlo race simulations** (up to 1000 runs)
- Recommends **pit stop count, tyre strategy & pit windows**
- Provides **Driver** and **Team** analytics dashboards
- Compares **9 ML models** with full evaluation metrics

**MLOps additions:**
- DVC reproducible pipeline (5 stages)
- CI/CD via GitHub Actions
- Docker + docker-compose deployment
- Local model registry (JSON-based)
- Local experiment tracking (JSON logs)
- pytest test suite (3 test files, 20+ tests)

---

## Dataset

**Source:** [Formula 1 World Championship (1950-2020) — Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

### Download & Setup

1. Visit the link above (free Kaggle account required)
2. Click **Download**
3. Extract the ZIP
4. Copy these 8 files into `data/raw/`:

```
data/raw/
├── results.csv
├── races.csv
├── drivers.csv
├── constructors.csv
├── qualifying.csv
├── pit_stops.csv
├── lap_times.csv
└── circuits.csv
```

Only data from year >= 2000 is used. Do NOT rename the files.

---

## Folder Structure

```
f1-ml-project/
├── app.py                          # Streamlit web UI (8 pages)
├── params.yaml                     # DVC-tracked parameters
├── dvc.yaml                        # DVC pipeline definition
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── Makefile
├── requirements.txt
├── README.md
├── src/                            # Core ML modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── predictor.py
│   ├── simulator.py
│   └── strategy.py
├── ml_pipeline/                    # DVC stage scripts
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── mlops/
│   ├── model_registry/
│   │   ├── register_model.py
│   │   └── registry.json
│   └── experiments/
│       └── run_*.json
├── tests/
│   ├── test_data_loading.py
│   ├── test_model_files.py
│   └── test_prediction.py
├── docs/
│   └── architecture.md
├── data/
│   ├── raw/                        # Place Kaggle CSVs here
│   └── processed/                  # Auto-generated
└── models/                         # Auto-generated
```

---

## Installation

```bash
cd f1-ml-project
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Option A — Pure ML (No DVC)

```bash
# Place CSVs in data/raw/
python src/train_models.py
python src/evaluate_models.py
streamlit run app.py
```

---

## Option B — DVC Pipeline

```bash
dvc init
dvc repro
streamlit run app.py
```

### Change a parameter and retrain

```bash
# Edit params.yaml (e.g. n_estimators: 200)
dvc repro
# DVC only re-runs affected stages
```

---

## Option C — Docker

```bash
# Train models locally first, then:
docker-compose up --build
# Open http://localhost:8501
docker-compose down
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Tests auto-skip if models are not trained yet.
Simulator and strategy tests always run without any data.

---

## ML Models

**Classification (Top 10 — binary):**
Logistic Regression, Decision Tree, Random Forest (best), SVM, Naive Bayes

**Regression (Finishing Position 1-20):**
Linear Regression, Ridge Regression (best), Lasso Regression

**Clustering:** K-Means (k=5)

---

## Model Registry

```bash
# Register manually
python -c "
from mlops.model_registry.register_model import register
register('Random Forest', metrics={'accuracy':0.87,'f1':0.85},
         artifact_path='models/Random_Forest.pkl', stage='production')
"

# List all models
python -c "
from mlops.model_registry.register_model import list_models
import json; print(json.dumps(list_models(), indent=2))
"
```

Stages: development -> staging -> production

---

## Experiment Tracking

Every training run saves to `mlops/experiments/run_YYYYMMDD_HHMMSS.json`
Each file contains: full params snapshot + all model metrics + timestamp.

---

## CI/CD (GitHub Actions)

Triggers on every push and pull request. Checks:
- All source files present
- All modules compile cleanly
- params.yaml valid
- Simulator/strategy unit tests pass
- Model registry works
- dvc.yaml has all 5 stages
- Dockerfile and docker-compose.yml exist

Free — runs on GitHub's free Ubuntu runners.

---

## Windows Commands (no Make)

```cmd
pip install -r requirements.txt
python src/train_models.py
python src/evaluate_models.py
streamlit run app.py
pytest tests/ -v
dvc init
dvc repro
docker build -t f1-ml-app .
docker-compose up --build
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| FileNotFoundError for CSV | Place all 8 CSVs in data/raw/ |
| Models not found in app | Run python src/train_models.py first |
| dvc repro fails stage 1 | CSVs missing from data/raw/ |
| Docker container exits | Run training first so models/ exists |
| Port 8501 busy | streamlit run app.py --server.port 8502 |

---

## Future Scope

- MLflow visual experiment tracking (Phase 2)
- Remote DVC storage on S3/GDrive (Phase 2)
- LSTM sequence-based lap prediction (Phase 3)
- Live Ergast API integration (Phase 3)
- Kubernetes deployment (Phase 4)
- Model drift monitoring (Phase 4)

---

## Tech Stack

Python 3.11 · Scikit-learn 1.5 · Streamlit 1.35 · Plotly 5
DVC 3.51 · pytest 8 · GitHub Actions · Docker · docker-compose
Dataset: Kaggle — F1 World Championship 1950-2020
