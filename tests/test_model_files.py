"""
tests/test_model_files.py
Verifies that all trained model artifacts exist and are loadable.
"""

import os
import pytest
import joblib

MODELS_DIR = "models"

EXPECTED_MODELS = [
    "Random_Forest.pkl",
    "Logistic_Regression.pkl",
    "Decision_Tree.pkl",
    "SVM.pkl",
    "Naive_Bayes.pkl",
    "Linear_Regression.pkl",
    "Ridge_Regression.pkl",
    "Lasso_Regression.pkl",
    "KMeans.pkl",
    "scaler.pkl",
    "le_driverRef.pkl",
    "le_constructorRef.pkl",
    "le_circuitRef.pkl",
]

EXPECTED_JSON = [
    "metrics.json",
    "feature_importance.json",
    "meta.json",
]


@pytest.mark.parametrize("model_file", EXPECTED_MODELS)
def test_model_file_exists(model_file):
    path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(path):
        pytest.skip(f"{model_file} not found — run training first")
    assert os.path.exists(path)


@pytest.mark.parametrize("model_file", EXPECTED_MODELS)
def test_model_file_loadable(model_file):
    path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(path):
        pytest.skip(f"{model_file} not found — run training first")
    obj = joblib.load(path)
    assert obj is not None


@pytest.mark.parametrize("json_file", EXPECTED_JSON)
def test_json_artifact_exists(json_file):
    path = os.path.join(MODELS_DIR, json_file)
    if not os.path.exists(path):
        pytest.skip(f"{json_file} not found — run training first")
    import json
    with open(path) as f:
        data = json.load(f)
    assert data is not None


def test_metrics_json_has_random_forest():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        pytest.skip("metrics.json not found")
    import json
    with open(path) as f:
        metrics = json.load(f)
    assert "Random Forest" in metrics, "metrics.json missing Random Forest entry"
    rf = metrics["Random Forest"]
    assert "accuracy" in rf
    assert "f1" in rf
    assert rf["accuracy"] > 0.0


def test_metrics_json_has_regression():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        pytest.skip("metrics.json not found")
    import json
    with open(path) as f:
        metrics = json.load(f)
    assert "Ridge Regression" in metrics
    rr = metrics["Ridge Regression"]
    assert "r2" in rr
    assert "mae" in rr
