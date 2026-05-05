"""
train_models.py
Trains all ML models and saves them to /models/.
Run this FIRST before launching the app.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_raw, build_master
from preprocessing import (
    clean, encode_categoricals, build_features,
    get_xy, scale, FEATURE_COLS, MODELS_DIR
)

# ── Classification ──────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ── Regression ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# ── Clustering ──────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans

# ── Evaluation ──────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("  F1 ML PROJECT — MODEL TRAINING")
    print("=" * 60)

    # 1. Load & merge
    print("\n[1/6] Loading raw CSVs …")
    dfs = load_raw()
    master = build_master(dfs, year_cutoff=2000)
    print(f"      Master shape: {master.shape}")

    # 2. Clean
    print("[2/6] Cleaning …")
    master = clean(master)
    print(f"      After clean: {master.shape}")

    # 3. Encode
    print("[3/6] Encoding categoricals …")
    master, encoders = encode_categoricals(master, fit=True)

    # 4. Feature engineering
    print("[4/6] Engineering features …")
    master = build_features(master)

    # Save processed
    master.to_csv(os.path.join(PROCESSED_DIR, "master.csv"), index=False)
    print("      Processed data saved.")

    # 5. Prepare X / y
    print("[5/6] Splitting train/test …")
    X, y_cls, y_reg, df_clean = get_xy(master)

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_cls, y_reg, test_size=0.2, random_state=42)
        if False  # placeholder
        else _split(X, y_cls, y_reg)
    )

    X_train_s, X_test_s, _ = scale(X_train, X_test, fit=True)
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    # 6. Train & evaluate
    print("\n[6/6] Training models …\n")
    metrics: dict = {}

    # ── CLASSIFICATION ───────────────────────────────────────────────────
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    for name, model in classifiers.items():
        print(f"  ▶ {name}")
        model.fit(X_train_s, y_cls_train)
        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_cls_test, y_pred).tolist()
        metrics[name] = {
            "type": "classification",
            "accuracy": round(accuracy_score(y_cls_test, y_pred), 4),
            "precision": round(precision_score(y_cls_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_cls_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_cls_test, y_pred, zero_division=0), 4),
            "confusion_matrix": cm,
        }
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))
        print(f"     Accuracy={metrics[name]['accuracy']}  F1={metrics[name]['f1']}")

    # ── REGRESSION ───────────────────────────────────────────────────────
    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=2000),
    }

    for name, model in regressors.items():
        print(f"  ▶ {name}")
        model.fit(X_train_s, y_reg_train)
        y_pred = model.predict(X_test_s)
        metrics[name] = {
            "type": "regression",
            "mae": round(mean_absolute_error(y_reg_test, y_pred), 4),
            "mse": round(mean_squared_error(y_reg_test, y_pred), 4),
            "r2": round(r2_score(y_reg_test, y_pred), 4),
        }
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))
        print(f"     MAE={metrics[name]['mae']}  R²={metrics[name]['r2']}")

    # ── CLUSTERING ───────────────────────────────────────────────────────
    print("  ▶ K-Means Clustering (k=5)")
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    km.fit(X_train_s)
    joblib.dump(km, os.path.join(MODELS_DIR, "KMeans.pkl"))
    metrics["K-Means"] = {"type": "clustering", "k": 5, "inertia": round(km.inertia_, 2)}
    print(f"     Inertia={km.inertia_:.2f}")

    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importances from RF
    rf = joblib.load(os.path.join(MODELS_DIR, "Random_Forest.pkl"))
    fi = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(fi, f, indent=2)

    # Save metadata for UI dropdowns
    meta = {
        "drivers": sorted(df_clean["driver_name"].dropna().unique().tolist()),
        "constructors": sorted(df_clean["team_name"].dropna().unique().tolist()),
        "circuits": sorted(df_clean["circuit_name"].dropna().unique().tolist()),
        "driverRef_map": df_clean.groupby("driver_name")["driverRef"].first().to_dict(),
        "constructorRef_map": df_clean.groupby("team_name")["constructorRef"].first().to_dict(),
        "circuitRef_map": df_clean.groupby("circuit_name")["circuitRef"].first().to_dict(),
        "years": sorted(df_clean["year"].unique().tolist()),
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Build strategy table
    _save_strategy_table(df_clean)

    print("\n" + "=" * 60)
    print("  ✅  Training complete. All models saved to /models/")
    print("=" * 60)


def _split(X, y_cls, y_reg):
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42
    )
    return (
        X[idx_train], X[idx_test],
        y_cls[idx_train], y_cls[idx_test],
        y_reg[idx_train], y_reg[idx_test],
    )


def _save_strategy_table(df: pd.DataFrame):
    """Aggregate pit-stop and lap patterns per circuit for strategy module."""
    strat = (
        df.groupby("circuitRef")
        .agg(
            avg_pit_count=("pit_stop_count", "mean"),
            avg_lap_ms=("avg_lap_ms", "mean"),
            circuit_name=("circuit_name", "first"),
            country=("country", "first"),
        )
        .reset_index()
    )
    strat["avg_pit_count"] = strat["avg_pit_count"].round(2)
    strat["avg_lap_ms"] = strat["avg_lap_ms"].round(0)
    strat.to_csv(
        os.path.join(MODELS_DIR, "..", "data", "processed", "strategy_table.csv"),
        index=False,
    )
    print("  Strategy table saved.")


if __name__ == "__main__":
    main()
