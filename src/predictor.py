"""
predictor.py
Loads trained models and returns predictions for a single driver/race entry.
"""

import os
import json
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _load(name: str):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))


def _scaler():
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))


def _le(col: str):
    return joblib.load(os.path.join(MODELS_DIR, f"le_{col}.pkl"))


def _encode(le, value: str) -> int:
    try:
        return int(le.transform([value])[0])
    except Exception:
        return 0  # unknown → 0


def build_input_vector(
    grid: int,
    qual_position: int,
    year: int,
    driver_ref: str,
    constructor_ref: str,
    circuit_ref: str,
    driver_avg_finish: float = 10.0,
    team_avg_finish: float = 10.0,
    driver_win_rate: float = 0.1,
    pit_stop_count: float = 2.0,
    avg_lap_ms: float = 90000.0,
) -> np.ndarray:
    le_driver = _le("driverRef")
    le_constructor = _le("constructorRef")
    le_circuit = _le("circuitRef")

    x = np.array([[
        grid,
        qual_position,
        year,
        _encode(le_driver, driver_ref),
        _encode(le_constructor, constructor_ref),
        _encode(le_circuit, circuit_ref),
        driver_avg_finish,
        team_avg_finish,
        driver_win_rate,
        pit_stop_count,
        avg_lap_ms,
    ]], dtype=float)
    return x


def predict_top10(x_raw: np.ndarray, model_name: str = "Random_Forest") -> dict:
    sc = _scaler()
    x = sc.transform(x_raw)
    model = _load(model_name)
    prob = model.predict_proba(x)[0][1]
    pred = int(prob >= 0.5)
    return {"probability": round(float(prob), 4), "prediction": pred}


def predict_position(x_raw: np.ndarray, model_name: str = "Ridge_Regression") -> dict:
    sc = _scaler()
    x = sc.transform(x_raw)
    model = _load(model_name)
    pos = float(model.predict(x)[0])
    pos = max(1.0, min(20.0, pos))
    return {"predicted_position": round(pos, 2)}


def predict_all_models(x_raw: np.ndarray) -> dict:
    """Run every classifier and regressor, return aggregated results."""
    sc = _scaler()
    x = sc.transform(x_raw)

    classifiers = [
        "Logistic_Regression", "Decision_Tree",
        "Random_Forest", "SVM", "Naive_Bayes",
    ]
    regressors = ["Linear_Regression", "Ridge_Regression", "Lasso_Regression"]

    cls_results = {}
    for name in classifiers:
        try:
            m = _load(name)
            prob = float(m.predict_proba(x)[0][1])
            cls_results[name] = {"top10_prob": round(prob, 4)}
        except Exception as e:
            cls_results[name] = {"error": str(e)}

    reg_results = {}
    for name in regressors:
        try:
            m = _load(name)
            pos = float(m.predict(x)[0])
            pos = max(1.0, min(20.0, pos))
            reg_results[name] = {"predicted_position": round(pos, 2)}
        except Exception as e:
            reg_results[name] = {"error": str(e)}

    return {"classification": cls_results, "regression": reg_results}


def load_meta() -> dict:
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        return json.load(f)


def load_metrics() -> dict:
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        return json.load(f)


def load_feature_importance() -> dict:
    with open(os.path.join(MODELS_DIR, "feature_importance.json")) as f:
        return json.load(f)
