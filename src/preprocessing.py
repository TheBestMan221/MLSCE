"""
preprocessing.py
Cleans and prepares the master DataFrame for modelling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unusable rows, impute missing values.
    Returns a cleaned copy.
    """
    out = df.copy()

    # drivers who didn't start (grid == 0) are noise for position prediction
    out = out[out["grid"] > 0].copy()

    # impute qualifying with grid if missing
    out["qual_position"] = out["qual_position"].fillna(out["grid"])

    # pit stops: 0 if no data
    out["pit_stop_count"] = out["pit_stop_count"].fillna(0)
    out["pit_total_ms"] = out["pit_total_ms"].fillna(0)

    # avg lap: fill with race median
    race_lap_median = out.groupby("raceId")["avg_lap_ms"].transform("median")
    out["avg_lap_ms"] = out["avg_lap_ms"].fillna(race_lap_median)
    out["avg_lap_ms"] = out["avg_lap_ms"].fillna(out["avg_lap_ms"].median())

    # fastestLapSpeed
    out["fastestLapSpeed"] = out["fastestLapSpeed"].fillna(out["fastestLapSpeed"].median())

    # drop rows where positionOrder is still NaN (DNQ etc.)
    out = out.dropna(subset=["positionOrder", "grid"])

    return out.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame, fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    If fit=True, create and save encoders. Else load them.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    cat_cols = ["driverRef", "constructorRef", "circuitRef"]
    encoders: dict[str, LabelEncoder] = {}
    out = df.copy()

    for col in cat_cols:
        enc_path = os.path.join(MODELS_DIR, f"le_{col}.pkl")
        if fit:
            le = LabelEncoder()
            out[col + "_enc"] = le.fit_transform(out[col].astype(str))
            joblib.dump(le, enc_path)
            encoders[col] = le
        else:
            le = joblib.load(enc_path)
            # handle unseen labels gracefully
            out[col + "_enc"] = out[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )
            encoders[col] = le

    return out, encoders


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features (rolling averages, etc.)."""
    out = df.sort_values(["driverId", "year", "raceId"]).copy()

    # driver rolling average finish (last 5 races)
    out["driver_avg_finish"] = (
        out.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    out["driver_avg_finish"] = out["driver_avg_finish"].fillna(out["positionOrder"].mean())

    # team rolling average finish
    out["team_avg_finish"] = (
        out.groupby("constructorId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    out["team_avg_finish"] = out["team_avg_finish"].fillna(out["positionOrder"].mean())

    # driver win rate (rolling 10)
    out["driver_win_rate"] = (
        out.groupby("driverId")["positionOrder"]
        .transform(lambda x: (x.shift(1) == 1).rolling(10, min_periods=1).mean())
    )

    # target variables
    out["top10"] = (out["positionOrder"] <= 10).astype(int)
    out["top3"] = (out["positionOrder"] <= 3).astype(int)

    return out


FEATURE_COLS = [
    "grid", "qual_position", "year",
    "driverRef_enc", "constructorRef_enc", "circuitRef_enc",
    "driver_avg_finish", "team_avg_finish", "driver_win_rate",
    "pit_stop_count", "avg_lap_ms",
]

CLASSIFICATION_TARGET = "top10"
REGRESSION_TARGET = "positionOrder"


def get_xy(df: pd.DataFrame):
    """Return feature matrix X and both targets."""
    df_clean = df.dropna(subset=FEATURE_COLS + [CLASSIFICATION_TARGET, REGRESSION_TARGET])
    X = df_clean[FEATURE_COLS].values
    y_cls = df_clean[CLASSIFICATION_TARGET].values
    y_reg = df_clean[REGRESSION_TARGET].values
    return X, y_cls, y_reg, df_clean


def scale(X_train, X_test, fit: bool = True):
    """StandardScaler with optional save/load."""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if fit:
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)
        joblib.dump(sc, scaler_path)
    else:
        sc = joblib.load(scaler_path)
        X_train_s = sc.transform(X_train)
        X_test_s = sc.transform(X_test)
    return X_train_s, X_test_s, sc
