"""
strategy.py
Rule-based + data-driven pit stop and tyre strategy recommender.
"""

import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TYRE_STRATEGIES = {
    1: ["Soft → Hard"],
    2: ["Soft → Medium → Hard", "Soft → Hard → Medium", "Medium → Hard"],
    3: ["Soft → Soft → Medium → Hard", "Soft → Medium → Medium → Hard"],
}

CIRCUIT_NOTES = {
    "monaco": "Low overtaking venue – undercut strategy is critical. Pit as early as lap 20.",
    "monza": "High-speed track with long straights. Hard tyres preferred. 1-stop likely.",
    "spa": "Variable weather likely. Be ready to switch to Intermediate/Wets.",
    "silverstone": "High-energy corners cause tyre degradation. 2-stop recommended.",
    "bahrain": "High tyre deg due to sand and heat. 2-stop with Medium–Hard.",
    "default": "Standard strategy applies. Monitor tyre temperatures and safety car windows.",
}


def _load_strategy_table() -> pd.DataFrame | None:
    path = os.path.join(PROCESSED_DIR, "strategy_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def recommend(
    circuit_ref: str,
    grid_position: int,
    avg_lap_ms: float = 90000.0,
) -> dict:
    """
    Returns a strategy recommendation dict.
    """
    strat_table = _load_strategy_table()

    # look up historical pit count for this circuit
    hist_pit_count = 2.0
    if strat_table is not None:
        row = strat_table[strat_table["circuitRef"] == circuit_ref]
        if not row.empty:
            hist_pit_count = float(row["avg_pit_count"].values[0])

    recommended_stops = max(1, min(3, round(hist_pit_count)))

    # adjust for grid: starting from back → consider more aggressive strategy
    if grid_position > 15:
        recommended_stops = min(3, recommended_stops + 1)
    elif grid_position <= 3:
        recommended_stops = max(1, recommended_stops - 1)

    # tyre suggestion
    tyre_options = TYRE_STRATEGIES.get(recommended_stops, TYRE_STRATEGIES[2])
    tyre_strategy = tyre_options[0]

    # pit window estimation (very rough: lap 20–40 zone)
    race_laps = 57  # average F1 race
    if recommended_stops == 1:
        pit_windows = ["Lap 28–35"]
    elif recommended_stops == 2:
        pit_windows = ["Lap 15–20", "Lap 38–45"]
    else:
        pit_windows = ["Lap 10–15", "Lap 28–33", "Lap 45–50"]

    note_key = circuit_ref if circuit_ref in CIRCUIT_NOTES else "default"
    circuit_note = CIRCUIT_NOTES[note_key]

    # lap time bucket
    lap_sec = avg_lap_ms / 1000.0
    if lap_sec < 70:
        pace_label = "High-speed circuit (< 70 s/lap)"
    elif lap_sec < 90:
        pace_label = "Medium-speed circuit (70–90 s/lap)"
    else:
        pace_label = "Slow/technical circuit (> 90 s/lap)"

    return {
        "circuit_ref": circuit_ref,
        "recommended_pit_stops": recommended_stops,
        "tyre_strategy": tyre_strategy,
        "pit_windows": pit_windows,
        "avg_historical_stops": round(hist_pit_count, 2),
        "pace_category": pace_label,
        "circuit_note": circuit_note,
        "grid_position": grid_position,
    }
