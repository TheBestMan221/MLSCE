"""
ml_pipeline/evaluate.py
DVC Stage: evaluate
Reloads trained models, evaluates on test split, overwrites metrics.json.
Also triggers model registry registration.
"""

import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import get_xy, scale, MODELS_DIR


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 55)
    print("  STAGE: evaluate")
    print("=" * 55)

    p = load_params()
    random_state = p["base"]["random_state"]
    test_size = p["base"]["test_size"]

    df = pd.read_csv("data/processed/featured_master.csv", low_memory=False)
    X, y_cls, y_reg, _ = get_xy(df)

    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=random_state
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_cls_te = y_cls[idx_test]
    y_reg_te = y_reg[idx_test]

    sc = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_test_s = sc.transform(X_test)
    X_train_s = sc.transform(X_train)

    metrics: dict = {}

    classifiers = ["Logistic_Regression", "Decision_Tree",
                   "Random_Forest", "SVM", "Naive_Bayes"]
    for name in classifiers:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"  SKIP {name} (not found)")
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_cls_te, y_pred).tolist()
        metrics[name.replace("_", " ")] = {
            "type": "classification",
            "accuracy": round(accuracy_score(y_cls_te, y_pred), 4),
            "precision": round(precision_score(y_cls_te, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_cls_te, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_cls_te, y_pred, zero_division=0), 4),
            "confusion_matrix": cm,
        }
        m = metrics[name.replace("_", " ")]
        print(f"  {name:<25} Acc={m['accuracy']}  F1={m['f1']}")

    regressors = ["Linear_Regression", "Ridge_Regression", "Lasso_Regression"]
    for name in regressors:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test_s)
        metrics[name.replace("_", " ")] = {
            "type": "regression",
            "mae": round(mean_absolute_error(y_reg_te, y_pred), 4),
            "mse": round(mean_squared_error(y_reg_te, y_pred), 4),
            "r2": round(r2_score(y_reg_te, y_pred), 4),
        }
        m = metrics[name.replace("_", " ")]
        print(f"  {name:<25} MAE={m['mae']}  R2={m['r2']}")

    km_path = os.path.join(MODELS_DIR, "KMeans.pkl")
    if os.path.exists(km_path):
        km = joblib.load(km_path)
        metrics["K-Means"] = {
            "type": "clustering",
            "k": int(km.n_clusters),
            "inertia": round(float(km.inertia_), 2),
        }

    # Feature importance
    rf_path = os.path.join(MODELS_DIR, "Random_Forest.pkl")
    if os.path.exists(rf_path):
        from src.preprocessing import FEATURE_COLS
        rf = joblib.load(rf_path)
        fi = dict(zip(FEATURE_COLS, [round(float(v), 6) for v in rf.feature_importances_]))
        with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
            json.dump(fi, f, indent=2)

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {metrics_path}")

    # Auto-register best classifier into model registry
    _register_best_model(metrics)

    print("  ✅ evaluate complete.")


def _register_best_model(metrics: dict):
    """Register the best-performing classifier automatically."""
    try:
        sys.path.insert(0, ".")
        from mlops.model_registry.register_model import register
        cls_models = {k: v for k, v in metrics.items()
                      if v.get("type") == "classification"}
        if cls_models:
            best = max(cls_models, key=lambda k: cls_models[k]["f1"])
            register(
                model_name=best,
                version="auto",
                metrics=cls_models[best],
                artifact_path=os.path.join(MODELS_DIR,
                                            best.replace(" ", "_") + ".pkl"),
                stage="staging",
            )
    except Exception as e:
        print(f"  [registry] skipped: {e}")


if __name__ == "__main__":
    main()
