"""
feature_engineering.py
Standalone feature engineering utilities.
All logic is also embedded in preprocessing.py; this file exposes them
individually for inspection or notebook use.
"""

import pandas as pd
import numpy as np


def add_rolling_driver_avg(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add driver rolling average finish position (last `window` races)."""
    out = df.sort_values(["driverId", "year", "raceId"]).copy()
    out["driver_avg_finish"] = (
        out.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    out["driver_avg_finish"] = out["driver_avg_finish"].fillna(out["positionOrder"].mean())
    return out


def add_rolling_team_avg(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add constructor rolling average finish position."""
    out = df.sort_values(["constructorId", "year", "raceId"]).copy()
    out["team_avg_finish"] = (
        out.groupby("constructorId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    out["team_avg_finish"] = out["team_avg_finish"].fillna(out["positionOrder"].mean())
    return out


def add_driver_win_rate(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling win rate over last `window` races per driver."""
    out = df.sort_values(["driverId", "year", "raceId"]).copy()
    out["driver_win_rate"] = (
        out.groupby("driverId")["positionOrder"]
        .transform(lambda x: (x.shift(1) == 1).rolling(window, min_periods=1).mean())
    )
    return out


def add_top10_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["top10"] = (df["positionOrder"] <= 10).astype(int)
    return df


def add_top3_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["top3"] = (df["positionOrder"] <= 3).astype(int)
    return df


def add_grid_vs_finish_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Positive = gained places; negative = lost places."""
    df = df.copy()
    df["grid_delta"] = df["grid"] - df["positionOrder"]
    return df


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    df = add_rolling_driver_avg(df)
    df = add_rolling_team_avg(df)
    df = add_driver_win_rate(df)
    df = add_top10_target(df)
    df = add_top3_target(df)
    df = add_grid_vs_finish_delta(df)
    return df
