"""
tests/test_data_loading.py
Verifies that processed data exists and has expected structure.
"""

import os
import pytest
import pandas as pd

PROCESSED_PATH = os.path.join("data", "processed", "master.csv")
FEATURED_PATH = os.path.join("data", "processed", "featured_master.csv")
STRATEGY_PATH = os.path.join("data", "processed", "strategy_table.csv")


def test_processed_master_exists():
    assert os.path.exists(PROCESSED_PATH), (
        f"Processed master CSV not found at {PROCESSED_PATH}. "
        "Run: python src/train_models.py  OR  dvc repro"
    )


def test_processed_master_not_empty():
    if not os.path.exists(PROCESSED_PATH):
        pytest.skip("master.csv not found — skipping content test")
    df = pd.read_csv(PROCESSED_PATH)
    assert len(df) > 100, "master.csv has fewer than 100 rows — something is wrong"


def test_processed_master_has_expected_columns():
    if not os.path.exists(PROCESSED_PATH):
        pytest.skip("master.csv not found")
    df = pd.read_csv(PROCESSED_PATH)
    required = ["raceId", "driverId", "constructorId", "grid",
                 "positionOrder", "year", "driverRef", "constructorRef", "circuitRef"]
    for col in required:
        assert col in df.columns, f"Missing expected column: {col}"


def test_featured_master_exists():
    if not os.path.exists(FEATURED_PATH):
        pytest.skip("featured_master.csv not found — skipping (run dvc repro)")
    df = pd.read_csv(FEATURED_PATH)
    assert "driver_avg_finish" in df.columns
    assert "top10" in df.columns


def test_strategy_table_exists():
    if not os.path.exists(STRATEGY_PATH):
        pytest.skip("strategy_table.csv not found")
    df = pd.read_csv(STRATEGY_PATH)
    assert "circuitRef" in df.columns
    assert "avg_pit_count" in df.columns
    assert len(df) > 5
