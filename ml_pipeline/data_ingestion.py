"""
ml_pipeline/data_ingestion.py
DVC Stage: data_ingestion
Loads raw CSVs, merges them, filters by year, saves ingested master CSV.
"""

import os
import sys
import yaml
import hashlib
import json

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_raw, build_master


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 55)
    print("  STAGE: data_ingestion")
    print("=" * 55)

    params = load_params()
    year_cutoff = params["base"]["year_cutoff"]

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Loading raw CSVs (year >= {year_cutoff}) …")
    dfs = load_raw()
    master = build_master(dfs, year_cutoff=year_cutoff)

    out_path = os.path.join(out_dir, "ingested_master.csv")
    master.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  shape={master.shape}")

    # Save dataset hash for experiment tracking
    meta = {
        "rows": int(master.shape[0]),
        "cols": int(master.shape[1]),
        "year_cutoff": year_cutoff,
        "file_hashes": {},
    }
    raw_dir = params["data"]["raw_dir"]
    for fname in ["results.csv", "races.csv", "drivers.csv", "constructors.csv",
                  "qualifying.csv", "pit_stops.csv", "lap_times.csv", "circuits.csv"]:
        fpath = os.path.join(raw_dir, fname)
        if os.path.exists(fpath):
            meta["file_hashes"][fname] = file_md5(fpath)

    with open(os.path.join(out_dir, "ingestion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ data_ingestion complete. Rows={meta['rows']}")


if __name__ == "__main__":
    main()
