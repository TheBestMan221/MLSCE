"""
simulator.py
Monte Carlo race simulation using predicted top-10 probabilities.
"""

import numpy as np
from typing import Any


def run_simulation(
    drivers: list[str],
    top10_probs: list[float],
    n_sims: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Parameters
    ----------
    drivers      : list of driver names (up to 20)
    top10_probs  : corresponding top-10 finish probabilities
    n_sims       : number of Monte Carlo runs
    seed         : RNG seed for reproducibility

    Returns
    -------
    dict with per-driver:
        win_prob, podium_prob, top10_prob, avg_finish, sim_position
    """
    rng = np.random.default_rng(seed)
    n = len(drivers)
    probs = np.clip(np.array(top10_probs, dtype=float), 0.01, 0.99)

    win_counts = np.zeros(n, dtype=int)
    podium_counts = np.zeros(n, dtype=int)
    top10_counts = np.zeros(n, dtype=int)
    finish_sum = np.zeros(n, dtype=float)

    for _ in range(n_sims):
        # simulate finishing score: lower = better
        # base score from prob (inverted) + noise
        noise = rng.exponential(scale=1.0 / probs)
        order = np.argsort(noise)   # best first

        for finish_pos, driver_idx in enumerate(order):
            pos = finish_pos + 1
            finish_sum[driver_idx] += pos
            if pos == 1:
                win_counts[driver_idx] += 1
            if pos <= 3:
                podium_counts[driver_idx] += 1
            if pos <= 10:
                top10_counts[driver_idx] += 1

    avg_finish = finish_sum / n_sims

    results = []
    for i, drv in enumerate(drivers):
        results.append({
            "driver": drv,
            "win_prob": round(win_counts[i] / n_sims, 4),
            "podium_prob": round(podium_counts[i] / n_sims, 4),
            "top10_prob": round(top10_counts[i] / n_sims, 4),
            "avg_finish": round(float(avg_finish[i]), 2),
        })

    # sort by avg_finish ascending
    results.sort(key=lambda r: r["avg_finish"])

    # assign simulated finishing order
    for rank, r in enumerate(results, 1):
        r["sim_position"] = rank

    return {
        "n_sims": n_sims,
        "results": results,
        "winner": results[0]["driver"],
    }
