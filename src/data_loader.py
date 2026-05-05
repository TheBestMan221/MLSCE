"""
data_loader.py
Loads and merges raw F1 CSV files into a unified DataFrame.
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _path(filename: str) -> str:
    return os.path.join(RAW_DIR, filename)


def load_raw() -> dict[str, pd.DataFrame]:
    """Load every required CSV and return as a dict keyed by table name."""
    files = {
        "results": "results.csv",
        "races": "races.csv",
        "drivers": "drivers.csv",
        "constructors": "constructors.csv",
        "qualifying": "qualifying.csv",
        "pit_stops": "pit_stops.csv",
        "lap_times": "lap_times.csv",
        "circuits": "circuits.csv",
    }
    dfs: dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        full = _path(fname)
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"Missing file: {full}\n"
                "Please download the Kaggle dataset and place all CSVs in data/raw/"
            )
        dfs[key] = pd.read_csv(full, low_memory=False)
    return dfs


def build_master(dfs: dict[str, pd.DataFrame], year_cutoff: int = 2000) -> pd.DataFrame:
    """
    Merge all tables into one analysis-ready DataFrame,
    filtered to year >= year_cutoff.
    """
    races = dfs["races"][["raceId", "year", "circuitId", "name"]].rename(
        columns={"name": "race_name"}
    )
    races = races[races["year"] >= year_cutoff]

    results = dfs["results"][[
        "raceId", "driverId", "constructorId", "grid", "position",
        "positionOrder", "points", "laps", "milliseconds", "fastestLapSpeed",
        "statusId",
    ]].copy()

    drivers = dfs["drivers"][["driverId", "driverRef", "forename", "surname", "nationality"]].copy()
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]

    constructors = dfs["constructors"][["constructorId", "constructorRef", "name"]].rename(
        columns={"name": "team_name"}
    )

    circuits = dfs["circuits"][["circuitId", "circuitRef", "name", "country"]].rename(
        columns={"name": "circuit_name"}
    )

    qual = dfs["qualifying"][["raceId", "driverId", "position"]].rename(
        columns={"position": "qual_position"}
    )
    # keep best qualifying entry per driver per race
    qual = qual.sort_values("qual_position").groupby(["raceId", "driverId"]).first().reset_index()

    # pit stop aggregates per race/driver
    pit = dfs["pit_stops"].copy()
    pit_agg = (
        pit.groupby(["raceId", "driverId"])
        .agg(pit_stop_count=("stop", "max"), pit_total_ms=("milliseconds", "sum"))
        .reset_index()
    )

    # avg lap time per race/driver
    lap = dfs["lap_times"].copy()
    lap_agg = (
        lap.groupby(["raceId", "driverId"])
        .agg(avg_lap_ms=("milliseconds", "mean"))
        .reset_index()
    )

    # --- merge chain ---
    df = results.merge(races, on="raceId", how="inner")
    df = df.merge(drivers, on="driverId", how="left")
    df = df.merge(constructors, on="constructorId", how="left")
    df = df.merge(circuits, on="circuitId", how="left")
    df = df.merge(qual, on=["raceId", "driverId"], how="left")
    df = df.merge(pit_agg, on=["raceId", "driverId"], how="left")
    df = df.merge(lap_agg, on=["raceId", "driverId"], how="left")

    # convert position to numeric, \N → NaN
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["qual_position"] = pd.to_numeric(df["qual_position"], errors="coerce")
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")
    df["fastestLapSpeed"] = pd.to_numeric(df["fastestLapSpeed"], errors="coerce")

    return df
