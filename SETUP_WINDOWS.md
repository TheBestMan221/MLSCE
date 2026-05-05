# Windows Setup Guide — F1 ML Project
# Fix for: SSL errors, missing joblib, DVC duplicate output

---

## THE 3 ERRORS YOU HAD — AND THEIR FIXES

### Error 1: SSL Certificate Verify Failed
```
SSLError: certificate verify failed: self-signed certificate in certificate chain
```
**Cause:** Your college/corporate network has a proxy that intercepts SSL.
**Fix:** Use `--trusted-host` flags OR place `pip.ini` in the right location.

### Error 2: No module named 'joblib'
```
ModuleNotFoundError: No module named 'joblib'
```
**Cause:** pip install failed silently due to SSL error above.
**Fix:** Install with SSL bypass (see steps below).

### Error 3: DVC duplicate output
```
ERROR: output 'models\metrics.json' is already specified in stage: 'evaluate'
```
**Cause:** `metrics.json` was listed in both `train` and `evaluate` stages.
**Fix:** Already fixed in the new `dvc.yaml` — `metrics.json` only in `evaluate`.

---

## STEP-BY-STEP SETUP (Windows PowerShell)

### Step 1 — Open PowerShell as normal user in your project folder

```powershell
cd P:\okc\f1-ml-project
```

### Step 2 — Create fresh virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get "execution policy" error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install with SSL bypass (IMPORTANT — use this instead of normal pip install)

**Option A — Use the provided script (easiest):**
```powershell
.\install.ps1
```

**Option B — Manual one-liner:**
```powershell
pip install scikit-learn numpy pandas joblib streamlit plotly dvc pyyaml pytest matplotlib seaborn --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
```

**Option C — Set pip.ini permanently (best for long term):**

Create the file `C:\Users\<YourName>\pip\pip.ini` with content:
```ini
[global]
trusted-host =
    pypi.org
    files.pythonhosted.org
    pypi.python.org
```

Then normal pip install works:
```powershell
pip install -r requirements.txt
```

### Step 4 — Verify all packages installed

```powershell
python -c "import sklearn, numpy, pandas, joblib, streamlit, plotly, yaml; print('ALL OK')"
```

Expected output: `ALL OK`

---

## RUNNING THE PROJECT

### Option A — Quick ML Run (No DVC)

```powershell
# Place 8 CSV files in data\raw\ first

python src/train_models.py
streamlit run app.py
```

### Option B — DVC Pipeline

```powershell
# Step 1: Delete old DVC state if corrupted
Remove-Item -Recurse -Force .dvc\cache -ErrorAction SilentlyContinue
Remove-Item -Force .dvc\tmp\* -ErrorAction SilentlyContinue

# Step 2: Initialize DVC fresh
dvc init -f
git add .dvc .gitignore
git commit -m "init dvc"

# Step 3: Run pipeline
dvc repro

# Step 4: Launch app
streamlit run app.py
```

### If dvc repro fails with "stage already has output":

```powershell
# Clean DVC state and retry
dvc remove evaluate
dvc repro
```

OR simply delete the .dvc folder and reinitialize:
```powershell
Remove-Item -Recurse -Force .dvc
git rm -r --cached .dvc
dvc init -f
dvc repro
```

---

## RUNNING TESTS

```powershell
pytest tests/ -v --tb=short
```

Tests that need trained models will auto-skip if models don't exist yet.
Simulator and strategy tests always pass with no data needed.

---

## RUNNING DOCKER (Optional)

```powershell
# Train models first, then:
docker build -t f1-ml-app .
docker-compose up --build
# Open http://localhost:8501
```

---

## GITHUB SETUP

```powershell
# Initialize git if not done
git init
git add .
git commit -m "Initial commit - F1 ML Project"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/f1-ml-project.git
git branch -M main
git push -u origin main

# CI/CD runs automatically on push
# Check: https://github.com/YOUR_USERNAME/f1-ml-project/actions
```

---

## QUICK REFERENCE COMMANDS

| Task | Command |
|------|---------|
| Install (SSL safe) | `.\install.ps1` |
| Train models | `python src/train_models.py` |
| Run app | `streamlit run app.py` |
| Run DVC pipeline | `dvc repro` |
| Run tests | `pytest tests/ -v` |
| Docker build | `docker build -t f1-ml-app .` |
| Docker run | `docker-compose up --build` |
| View DVC status | `dvc status` |
| View experiments | `dir mlops\experiments\` |
| View registry | `type mlops\model_registry\registry.json` |

---

## EXPECTED OUTPUT AFTER TRAINING

```
============================================================
  F1 ML PROJECT - MODEL TRAINING
============================================================
[1/6] Loading raw CSVs ...
      Master shape: (26000, 35)
[2/6] Cleaning ...
      After clean: (24500, 35)
[3/6] Encoding categoricals ...
[4/6] Engineering features ...
[5/6] Splitting train/test ...
      Train: (19600, 11)  Test: (4900, 11)
[6/6] Training models ...
  >> Logistic Regression  Accuracy=0.84  F1=0.81
  >> Decision Tree        Accuracy=0.82  F1=0.80
  >> Random Forest        Accuracy=0.87  F1=0.85
  >> SVM                  Accuracy=0.85  F1=0.83
  >> Naive Bayes          Accuracy=0.76  F1=0.73
  >> Linear Regression    MAE=2.8  R2=0.71
  >> Ridge Regression     MAE=2.7  R2=0.72
  >> Lasso Regression     MAE=2.9  R2=0.70
  >> K-Means              Inertia=...
  Training complete. All models saved to /models/
```
