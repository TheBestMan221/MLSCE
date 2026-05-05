# 🏗️ Architecture Documentation

**Project:** Formula 1 Race Prediction & Strategy Recommendation System  
**Version:** MLOps Edition

---

## 1. Overall System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        F1 ML SYSTEM                              │
│                                                                  │
│  ┌────────────┐   ┌─────────────────┐   ┌──────────────────┐   │
│  │  Raw Data  │──▶│   ML Pipeline   │──▶│  Trained Models  │   │
│  │  (Kaggle)  │   │  (DVC Stages)   │   │  (.pkl files)    │   │
│  └────────────┘   └─────────────────┘   └────────┬─────────┘   │
│                                                    │              │
│                          ┌─────────────────────────┘              │
│                          ▼                                         │
│                   ┌────────────┐   ┌──────────────────────────┐  │
│                   │ Streamlit  │   │  MLOps Layer             │  │
│                   │   Web UI   │   │  - Experiment Tracking   │  │
│                   │  (app.py)  │   │  - Model Registry        │  │
│                   └────────────┘   │  - CI/CD (GitHub)        │  │
│                                    │  - Docker Container       │  │
│                                    └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. ML Workflow (Pure Training)

```
data/raw/*.csv
      │
      ▼
src/data_loader.py
  └─ load_raw()        → reads 8 CSVs
  └─ build_master()    → merges all tables, filters year ≥ 2000
      │
      ▼
src/preprocessing.py
  └─ clean()           → drop bad rows, impute missing values
  └─ encode_categoricals() → LabelEncoder for driver/team/circuit
  └─ build_features()  → rolling averages, win rate, targets
  └─ get_xy()          → returns X, y_cls, y_reg
  └─ scale()           → StandardScaler (fit on train only)
      │
      ▼
src/train_models.py
  └─ 5 Classifiers     → predict Top10 (binary)
  └─ 3 Regressors      → predict finishing position
  └─ K-Means           → cluster drivers/circuits
  └─ Save .pkl files   → models/
  └─ Save meta.json    → UI dropdowns
      │
      ▼
models/
  ├── Random_Forest.pkl
  ├── metrics.json
  ├── feature_importance.json
  └── meta.json
```

---

## 3. DVC Pipeline Workflow

```
dvc repro
    │
    ├── Stage 1: data_ingestion
    │     cmd: python ml_pipeline/data_ingestion.py
    │     deps: data/raw/*.csv, params.yaml (year_cutoff)
    │     outs: data/processed/ingested_master.csv
    │
    ├── Stage 2: preprocessing
    │     cmd: python ml_pipeline/preprocessing.py
    │     deps: ingested_master.csv
    │     outs: data/processed/master.csv
    │
    ├── Stage 3: feature_engineering
    │     cmd: python ml_pipeline/feature_engineering.py
    │     deps: master.csv
    │     outs: data/processed/featured_master.csv
    │
    ├── Stage 4: train
    │     cmd: python ml_pipeline/train.py
    │     deps: featured_master.csv, params.yaml (all model params)
    │     outs: models/*.pkl, models/meta.json
    │
    └── Stage 5: evaluate
          cmd: python ml_pipeline/evaluate.py
          deps: models/*.pkl, featured_master.csv
          outs: models/metrics.json (cached: false → tracked as metric)
          → auto-registers best model in mlops/model_registry/registry.json
```

**Key DVC benefit:** DVC only re-runs stages whose dependencies have changed.
Changing `params.yaml` (e.g. `n_estimators: 200`) and running `dvc repro`
only re-trains, not re-ingests or re-preprocesses.

---

## 4. Model Registry

```
mlops/model_registry/
  ├── register_model.py    ← Python API
  └── registry.json        ← Flat JSON database

Lifecycle:
  development → staging → production

Functions:
  register(model_name, version, metrics, artifact_path, stage)
  promote(model_name, version, new_stage)
  list_models(stage=None)
  get_latest(model_name)

registry.json entry example:
{
  "model_name": "Random Forest",
  "version": "v1",
  "registered_at": "2024-08-01 14:32:00",
  "metrics": { "accuracy": 0.87, "f1": 0.85 },
  "artifact_path": "models/Random_Forest.pkl",
  "stage": "staging"
}
```

---

## 5. Experiment Tracking

```
mlops/experiments/
  └── run_20240801_143200.json   ← one file per training run

Content of each run file:
{
  "timestamp": "20240801_143200",
  "params": { ... full params.yaml snapshot ... },
  "metrics": { ... all model metrics ... }
}

To compare runs: read and compare the JSON files.
Future: integrate MLflow for visual experiment tracking.
```

---

## 6. CI/CD Workflow (GitHub Actions)

```
Developer pushes code to GitHub
         │
         ▼
.github/workflows/ci.yml triggers
         │
    ┌────┴──────────────────────────────────────┐
    │  Job: test                                 │
    │  1. Checkout code                          │
    │  2. Setup Python 3.11                      │
    │  3. Cache pip packages                     │
    │  4. pip install -r requirements.txt        │
    │  5. Check all source files present         │
    │  6. Syntax check all .py modules           │
    │  7. Validate params.yaml structure         │
    │  8. Run pytest (no-CSV tests only)         │
    │  9. Validate model registry module         │
    │  10. Validate dvc.yaml                     │
    └────────────────────────────────────────────┘
         │
    ┌────┴──────────────────────────────────────┐
    │  Job: docker-lint                          │
    │  1. Check Dockerfile exists                │
    │  2. Check docker-compose.yml exists        │
    └────────────────────────────────────────────┘
         │
         ▼
    ✅ Green = safe to merge
    ❌ Red   = fix before merge
```

---

## 7. Docker Deployment Flow

```
Host Machine
  ├── data/processed/   ← pre-processed data (volume mounted)
  ├── models/           ← trained .pkl files (volume mounted)
  └── docker-compose.yml

docker-compose up --build
         │
         ▼
┌─────────────────────────────┐
│  Container: f1_streamlit    │
│  Base image: python:3.11    │
│  Port: 8501                 │
│  CMD: streamlit run app.py  │
│                             │
│  Volume mounts:             │
│  ./models → /app/models     │
│  ./data   → /app/data       │
└─────────────────────────────┘
         │
         ▼
Browser: http://localhost:8501
```

---

## 8. Feature Engineering Details

| Feature | Type | Source | Importance |
|---|---|---|---|
| `grid` | Numeric | results.csv | Starting position on track |
| `qual_position` | Numeric | qualifying.csv | Qualifying result |
| `year` | Numeric | races.csv | Season context |
| `driverRef_enc` | Encoded | drivers.csv | Driver identity |
| `constructorRef_enc` | Encoded | constructors.csv | Team identity |
| `circuitRef_enc` | Encoded | circuits.csv | Track identity |
| `driver_avg_finish` | Rolling-5 | computed | Recent driver form |
| `team_avg_finish` | Rolling-5 | computed | Recent team form |
| `driver_win_rate` | Rolling-10 | computed | Driver win tendency |
| `pit_stop_count` | Numeric | pit_stops.csv | Race strategy indicator |
| `avg_lap_ms` | Numeric | lap_times.csv | Pace indicator |

---

## 9. For Faculty Viva — Key Points

1. **Why DVC?** — Reproducible ML pipelines. Every stage has tracked inputs/outputs.
   Changing one parameter automatically re-runs only affected stages.

2. **Why Docker?** — Environment consistency. App runs identically on any machine
   with Docker installed, regardless of OS or Python version.

3. **Why CI/CD?** — Automated quality gate. Every code push triggers tests.
   Prevents breaking changes from reaching main branch.

4. **Model Registry vs Experiment Tracking** — Registry tracks *production-ready* versions.
   Experiments track *every run* including failed ones. Different purposes.

5. **Why 9 ML models?** — Comparison. No single model is best for all datasets.
   Random Forest wins on F1 score; Ridge wins on regression R². We justify choices with metrics.

6. **Monte Carlo Simulation** — Probabilistic race outcome using 500 independent
   simulated races. Winner is the driver with lowest average simulated finish.
