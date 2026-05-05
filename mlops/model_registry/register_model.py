"""
mlops/model_registry/register_model.py
Local model registry — stores model metadata in registry.json.
Stages: development → staging → production
"""

import os
import json
from datetime import datetime

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.json")


def _load() -> list:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return []


def _save(registry: list):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _next_version(registry: list, model_name: str) -> str:
    versions = [
        int(r["version"].replace("v", ""))
        for r in registry
        if r["model_name"] == model_name and r["version"].startswith("v")
    ]
    return f"v{max(versions) + 1}" if versions else "v1"


def register(
    model_name: str,
    version: str = "auto",
    metrics: dict = None,
    artifact_path: str = "",
    stage: str = "development",
) -> dict:
    """
    Register a model run.

    Parameters
    ----------
    model_name    : e.g. "Random Forest"
    version       : "auto" → auto-increment, or explicit e.g. "v3"
    metrics       : dict of metric values
    artifact_path : path to .pkl file
    stage         : "development" | "staging" | "production"

    Returns the registry entry dict.
    """
    registry = _load()

    if version == "auto":
        version = _next_version(registry, model_name)

    entry = {
        "model_name": model_name,
        "version": version,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics or {},
        "artifact_path": artifact_path,
        "stage": stage,
    }

    registry.append(entry)
    _save(registry)
    print(f"  [Registry] Registered: {model_name} {version} → {stage}")
    return entry


def promote(model_name: str, version: str, new_stage: str):
    """Promote a specific model version to a new stage."""
    registry = _load()
    found = False
    for r in registry:
        if r["model_name"] == model_name and r["version"] == version:
            r["stage"] = new_stage
            r["promoted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break
    if not found:
        print(f"  [Registry] Not found: {model_name} {version}")
    else:
        _save(registry)
        print(f"  [Registry] Promoted {model_name} {version} → {new_stage}")


def list_models(stage: str = None) -> list:
    """Return all registry entries, optionally filtered by stage."""
    registry = _load()
    if stage:
        return [r for r in registry if r["stage"] == stage]
    return registry


def get_latest(model_name: str) -> dict | None:
    """Return the most recently registered entry for a model."""
    registry = _load()
    matches = [r for r in registry if r["model_name"] == model_name]
    return matches[-1] if matches else None


if __name__ == "__main__":
    # Demo
    entry = register(
        model_name="Random Forest",
        metrics={"accuracy": 0.87, "f1": 0.85},
        artifact_path="models/Random_Forest.pkl",
        stage="staging",
    )
    print(json.dumps(entry, indent=2))

    all_models = list_models()
    print(f"\nTotal registered: {len(all_models)}")
