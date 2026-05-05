"""
ml_pipeline/feature_engineering.py
DVC Stage: feature_engineering
Adds rolling averages, win rate, and target columns.
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import build_features
from src.feature_engineering import run_all


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 55)
    print("  STAGE: feature_engineering")
    print("=" * 55)

    import pandas as pd

    in_path = "data/processed/master.csv"
    out_path = "data/processed/featured_master.csv"

    print(f"  Reading {in_path} …")
    df = pd.read_csv(in_path, low_memory=False)
    print(f"  Input shape: {df.shape}")

    # Apply rolling features and target columns
    df = build_features(df)
    print(f"  After feature engineering: {df.shape}")

    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    print(f"  ✅ feature_engineering complete.")


if __name__ == "__main__":
    main()
