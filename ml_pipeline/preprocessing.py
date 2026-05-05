"""
ml_pipeline/preprocessing.py
DVC Stage: preprocessing
Cleans and encodes the ingested master CSV.
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import clean, encode_categoricals


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 55)
    print("  STAGE: preprocessing")
    print("=" * 55)

    import pandas as pd

    params = load_params()
    in_path = "data/processed/ingested_master.csv"
    out_path = "data/processed/master.csv"

    print(f"  Reading {in_path} …")
    df = pd.read_csv(in_path, low_memory=False)
    print(f"  Input shape: {df.shape}")

    df = clean(df)
    print(f"  After clean: {df.shape}")

    df, _ = encode_categoricals(df, fit=True)
    print(f"  Encoded categoricals. Saving → {out_path}")

    df.to_csv(out_path, index=False)
    print(f"  ✅ preprocessing complete.")


if __name__ == "__main__":
    main()
