"""
ml_pipeline/train.py
DVC Stage: train
Trains all ML models from params.yaml, saves .pkl artifacts to models/.
NOTE: metrics.json is NOT saved here — that is done by ml_pipeline/evaluate.py
      to avoid DVC duplicate-output errors.
"""

import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import get_xy, scale, FEATURE_COLS, MODELS_DIR

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def _split(X, y_cls, y_reg, test_size, random_state):
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=random_state
    )
    return (
        X[idx_train], X[idx_test],
        y_cls[idx_train], y_cls[idx_test],
        y_reg[idx_train], y_reg[idx_test],
    )


def _save_meta(df_clean: pd.DataFrame):
    meta = {
        "drivers": sorted(df_clean["driver_name"].dropna().unique().tolist()),
        "constructors": sorted(df_clean["team_name"].dropna().unique().tolist()),
        "circuits": sorted(df_clean["circuit_name"].dropna().unique().tolist()),
        "driverRef_map": df_clean.groupby("driver_name")["driverRef"].first().to_dict(),
        "constructorRef_map": df_clean.groupby("team_name")["constructorRef"].first().to_dict(),
        "circuitRef_map": df_clean.groupby("circuit_name")["circuitRef"].first().to_dict(),
        "years": sorted([int(y) for y in df_clean["year"].unique().tolist()]),
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _save_strategy_table(df: pd.DataFrame):
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
    os.makedirs("data/processed", exist_ok=True)
    strat.to_csv("data/processed/strategy_table.csv", index=False)


def main():
    print("=" * 55)
    print("  STAGE: train")
    print("=" * 55)

    p = load_params()
    os.makedirs(MODELS_DIR, exist_ok=True)

    random_state = p["base"]["random_state"]
    test_size    = p["base"]["test_size"]

    print("  Loading featured data ...")
    df = pd.read_csv("data/processed/featured_master.csv", low_memory=False)

    X, y_cls, y_reg, df_clean = get_xy(df)
    X_train, X_test, y_cls_tr, y_cls_te, y_reg_tr, y_reg_te = _split(
        X, y_cls, y_reg, test_size, random_state
    )
    X_train_s, X_test_s, _ = scale(X_train, X_test, fit=True)
    print(f"  Train={X_train.shape}  Test={X_test.shape}")

    # ── Classification ──────────────────────────────────────
    cp = p["classification"]
    classifiers = {
        "Logistic_Regression": LogisticRegression(
            max_iter=cp["logistic_regression"]["max_iter"],
            random_state=random_state),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=cp["decision_tree"]["max_depth"],
            random_state=random_state),
        "Random_Forest": RandomForestClassifier(
            n_estimators=cp["random_forest"]["n_estimators"],
            n_jobs=cp["random_forest"]["n_jobs"],
            random_state=random_state),
        "SVM": SVC(probability=cp["svm"]["probability"],
                   random_state=random_state),
        "Naive_Bayes": GaussianNB(),
    }

    for name, model in classifiers.items():
        print(f"  >> {name}")
        model.fit(X_train_s, y_cls_tr)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

    # ── Regression ──────────────────────────────────────────
    rp = p["regression"]
    regressors = {
        "Linear_Regression": LinearRegression(),
        "Ridge_Regression":  Ridge(alpha=rp["ridge"]["alpha"]),
        "Lasso_Regression":  Lasso(alpha=rp["lasso"]["alpha"],
                                    max_iter=rp["lasso"]["max_iter"]),
    }

    for name, model in regressors.items():
        print(f"  >> {name}")
        model.fit(X_train_s, y_reg_tr)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

    # ── Clustering ──────────────────────────────────────────
    cp2 = p["clustering"]["kmeans"]
    print("  >> KMeans")
    km = KMeans(n_clusters=cp2["n_clusters"], random_state=random_state,
                n_init=cp2["n_init"])
    km.fit(X_train_s)
    joblib.dump(km, os.path.join(MODELS_DIR, "KMeans.pkl"))

    # ── Save meta & strategy (needed by app UI) ─────────────
    _save_meta(df_clean)
    _save_strategy_table(df_clean)

    # ── Log experiment (params only; metrics added by evaluate) ─
    _log_experiment(p)

    print("  [OK] train complete.")


def _log_experiment(params: dict):
    os.makedirs("mlops/experiments", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = {
        "timestamp": timestamp,
        "stage": "train",
        "params": params,
        "metrics": "pending",
    }
    path = os.path.join("mlops", "experiments", f"run_{timestamp}_train.json")
    with open(path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"  Experiment log -> {path}")


if __name__ == "__main__":
    main()
