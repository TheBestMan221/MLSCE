"""
evaluate_models.py
Utility to re-evaluate models and print a report.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

from predictor import load_metrics, load_feature_importance


def print_report():
    metrics = load_metrics()
    fi = load_feature_importance()

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION REPORT")
    print("=" * 60)

    print("\n── Classification Models ──────────────────────────────────")
    for name, m in metrics.items():
        if m.get("type") != "classification":
            continue
        print(f"\n  {name}")
        print(f"    Accuracy  : {m['accuracy']}")
        print(f"    Precision : {m['precision']}")
        print(f"    Recall    : {m['recall']}")
        print(f"    F1 Score  : {m['f1']}")

    print("\n── Regression Models ──────────────────────────────────────")
    for name, m in metrics.items():
        if m.get("type") != "regression":
            continue
        print(f"\n  {name}")
        print(f"    MAE : {m['mae']}")
        print(f"    MSE : {m['mse']}")
        print(f"    R²  : {m['r2']}")

    print("\n── Clustering ─────────────────────────────────────────────")
    km = metrics.get("K-Means", {})
    print(f"  K-Means  k={km.get('k')}  Inertia={km.get('inertia')}")

    print("\n── Feature Importances (Random Forest) ────────────────────")
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_fi:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_report()
