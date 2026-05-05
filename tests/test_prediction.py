"""
tests/test_prediction.py
Verifies that predictor, simulator, and strategy modules return valid output.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_TRAINED = os.path.exists(os.path.join("models", "Random_Forest.pkl"))


# ── Predictor tests ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_input():
    if not MODELS_TRAINED:
        pytest.skip("Models not trained — skipping predictor tests")
    from src.predictor import build_input_vector
    return build_input_vector(
        grid=3,
        qual_position=3,
        year=2019,
        driver_ref="hamilton",
        constructor_ref="mercedes",
        circuit_ref="bahrain",
        driver_avg_finish=2.5,
        team_avg_finish=2.0,
        driver_win_rate=0.4,
        pit_stop_count=2.0,
        avg_lap_ms=90000.0,
    )


def test_input_vector_shape(sample_input):
    assert sample_input.shape == (1, 11), f"Expected (1,11), got {sample_input.shape}"


def test_predict_top10_returns_probability(sample_input):
    from src.predictor import predict_top10
    result = predict_top10(sample_input, "Random_Forest")
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in [0, 1]


def test_predict_position_returns_valid_range(sample_input):
    from src.predictor import predict_position
    result = predict_position(sample_input, "Ridge_Regression")
    assert "predicted_position" in result
    pos = result["predicted_position"]
    assert 1.0 <= pos <= 20.0, f"Position {pos} out of valid range 1–20"


def test_predict_all_models_structure(sample_input):
    from src.predictor import predict_all_models
    result = predict_all_models(sample_input)
    assert "classification" in result
    assert "regression" in result
    assert len(result["classification"]) == 5
    assert len(result["regression"]) == 3


# ── Simulator tests ──────────────────────────────────────────────────────────

def test_simulator_basic():
    from src.simulator import run_simulation
    drivers = ["Hamilton", "Verstappen", "Leclerc", "Norris"]
    probs = [0.85, 0.80, 0.70, 0.65]
    result = run_simulation(drivers, probs, n_sims=100, seed=1)
    assert "results" in result
    assert len(result["results"]) == 4
    assert "winner" in result


def test_simulator_probabilities_valid():
    from src.simulator import run_simulation
    drivers = ["A", "B", "C"]
    probs = [0.9, 0.6, 0.3]
    result = run_simulation(drivers, probs, n_sims=200)
    for r in result["results"]:
        assert 0.0 <= r["win_prob"] <= 1.0
        assert 0.0 <= r["podium_prob"] <= 1.0
        assert 0.0 <= r["top10_prob"] <= 1.0


def test_simulator_positions_unique():
    from src.simulator import run_simulation
    drivers = [f"Driver_{i}" for i in range(10)]
    probs = [0.9 - i * 0.05 for i in range(10)]
    result = run_simulation(drivers, probs, n_sims=100)
    positions = [r["sim_position"] for r in result["results"]]
    assert sorted(positions) == list(range(1, 11))


# ── Strategy tests ───────────────────────────────────────────────────────────

def test_strategy_returns_dict():
    from src.strategy import recommend
    result = recommend("bahrain", grid_position=5)
    assert isinstance(result, dict)


def test_strategy_valid_pit_count():
    from src.strategy import recommend
    for grid in [1, 10, 20]:
        result = recommend("monza", grid_position=grid)
        assert 1 <= result["recommended_pit_stops"] <= 3


def test_strategy_tyre_string_not_empty():
    from src.strategy import recommend
    result = recommend("monaco", grid_position=3)
    assert len(result["tyre_strategy"]) > 0
    assert "→" in result["tyre_strategy"]


def test_strategy_pit_windows_list():
    from src.strategy import recommend
    result = recommend("silverstone", grid_position=5)
    assert isinstance(result["pit_windows"], list)
    assert len(result["pit_windows"]) >= 1
