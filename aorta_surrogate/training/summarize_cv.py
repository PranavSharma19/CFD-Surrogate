"""Aggregate patient-separated development folds without opening the locked test set."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPERIMENTS = {
    "gat": ("geodesic_fold{fold}", "gat_fold{fold}_result.json"),
    "pointnet": ("pointnet_fold{fold}", "pointnet_fold{fold}_result.json"),
    "equivariant": ("equivariant_tangent_fold{fold}", "equivariant_fold{fold}_result.json"),
    "equivariant_multiscale": (
        "equivariant_multiscale_fold{fold}", "equivariant_multiscale_fold{fold}_result.json"
    ),
    "equivariant_long": ("equivariant_long_fold{fold}", "equivariant_fold{fold}_result.json"),
}


def summarize_model(canonical_root: Path, model_name: str) -> dict[str, object]:
    directory_pattern, file_pattern = EXPERIMENTS[model_name]
    folds: list[dict[str, object]] = []
    patients: dict[str, dict[str, float]] = {}
    absolute_sum = target_sum = vector_sum = nodes = 0.0
    angular_values: list[float] = []
    for fold in range(3):
        path = canonical_root / "experiments" / directory_pattern.format(fold=fold) / file_pattern.format(fold=fold)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["locked_test_cases_not_accessed"] != [
            "0033_H_ABAO_AAA", "0039_H_ABAO_AAA", "0042_H_ABAO_AAA"
        ]:
            raise ValueError(f"unexpected locked-test declaration in {path}")
        history_path = path.with_name(path.name.replace("_result.json", "_validation_history.json"))
        validation_history = json.loads(history_path.read_text(encoding="utf-8"))
        best_validation = min(
            validation_history, key=lambda row: row["metrics"]["wss_magnitude_mae_pa"]
        )
        metrics = best_validation["metrics"]
        fold_nodes = float(metrics["sampled_node_count"])
        fold_absolute = float(metrics["wss_magnitude_mae_pa"]) * fold_nodes
        absolute_sum += fold_absolute
        target_sum += fold_absolute / float(metrics["wss_magnitude_relative_error"])
        vector_sum += float(metrics["wss_vector_mae_pa"]) * fold_nodes
        nodes += fold_nodes
        angular_values.append(float(metrics["mean_angular_error_degrees"]))
        patients.update(metrics["per_patient"])
        folds.append(
            {
                "fold_index": fold,
                "best_validation_step": best_validation["step"],
                "wss_magnitude_relative_error": metrics["wss_magnitude_relative_error"],
                "wss_magnitude_mae_pa": metrics["wss_magnitude_mae_pa"],
                "wss_vector_mae_pa": metrics["wss_vector_mae_pa"],
                "mean_angular_error_degrees": metrics["mean_angular_error_degrees"],
                "peak_vram_gb": result["peak_vram_gb"],
                "seconds_per_training_step": result["seconds_per_training_step"],
            }
        )
    patient_relative = [float(row["wss_magnitude_relative_error"]) for row in patients.values()]
    return {
        "model": model_name,
        "folds": folds,
        "pooled_sampled_node_count": int(nodes),
        "pooled_wss_magnitude_relative_error": absolute_sum / target_sum,
        "pooled_wss_magnitude_mae_pa": absolute_sum / nodes,
        "pooled_wss_vector_mae_pa": vector_sum / nodes,
        "mean_fold_angular_error_degrees": statistics.fmean(angular_values),
        "median_patient_wss_magnitude_relative_error": statistics.median(patient_relative),
        "patient_relative_error_range": [min(patient_relative), max(patient_relative)],
        "per_patient": dict(sorted(patients.items())),
    }


def summarize(canonical_root: Path) -> dict[str, object]:
    models = {name: summarize_model(canonical_root, name) for name in EXPERIMENTS}
    comparison = {
        "schema_version": "1.0.0",
        "status": "development_cross_validation_only",
        "locked_test_opened": False,
        "models": models,
        "interpretation": {
            "magnitude": "Lower is better. Neither baseline approaches the 15% research gate.",
            "direction": "The tangent-conditioned equivariant model has the lowest mean angular error.",
            "decision": "Advance the simpler tangent-equivariant model to longer development training; multiscale edges do not improve all folds.",
        },
    }
    output = canonical_root / "experiments" / "cv_baseline_comparison.json"
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.canonical_root), indent=2))


if __name__ == "__main__":
    main()
