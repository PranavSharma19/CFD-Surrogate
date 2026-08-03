"""Evaluate frozen development checkpoints under primary and severe WSS masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from aorta_surrogate.training.fold_trainer import (
    FoldTrainingConfig,
    _build_model,
    _evaluate,
)


DEFAULT_EXPERIMENTS = tuple(f"equivariant_long_fold{fold}" for fold in range(3))
POLICIES = ("primary", "severe_sensitivity")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_overall(rows: list[dict[str, object]]) -> dict[str, object]:
    nodes = sum(int(row["sampled_node_count"]) for row in rows)
    if nodes <= 0:
        raise ValueError("cannot aggregate an empty evaluation")
    vector_abs = sum(
        float(row["wss_vector_mae_pa"]) * int(row["sampled_node_count"])
        for row in rows
    )
    vector_square = sum(
        float(row["wss_vector_rmse_pa"]) ** 2 * int(row["sampled_node_count"])
        for row in rows
    )
    magnitude_abs = sum(
        float(row["wss_magnitude_mae_pa"]) * int(row["sampled_node_count"])
        for row in rows
    )
    target_magnitude = sum(
        float(row["zero_baseline_vector_mae_pa"]) * int(row["sampled_node_count"])
        for row in rows
    )
    angular_nodes = sum(int(row["angular_metric_node_count"]) for row in rows)
    angular_sum = sum(
        float(row["mean_angular_error_degrees"])
        * int(row["angular_metric_node_count"])
        for row in rows
    )
    return {
        "sampled_node_count": nodes,
        "wss_vector_mae_pa": vector_abs / nodes,
        "wss_vector_rmse_pa": (vector_square / nodes) ** 0.5,
        "wss_magnitude_mae_pa": magnitude_abs / nodes,
        "wss_magnitude_relative_error": magnitude_abs / max(target_magnitude, 1.0e-12),
        "mean_angular_error_degrees": angular_sum / max(angular_nodes, 1),
        "angular_metric_node_count": angular_nodes,
        "zero_baseline_vector_mae_pa": target_magnitude / nodes,
    }


def _aggregate_named_buckets(
    rows: list[dict[str, dict[str, object]]]
) -> dict[str, dict[str, object]]:
    accumulators: dict[str, dict[str, float]] = {}
    for named_rows in rows:
        for name, row in named_rows.items():
            nodes = int(row["sampled_node_count"])
            magnitude_abs = float(row["wss_magnitude_mae_pa"]) * nodes
            relative = float(row["wss_magnitude_relative_error"])
            target_magnitude = magnitude_abs / relative if relative > 0.0 else 0.0
            bucket = accumulators.setdefault(
                name,
                {"nodes": 0.0, "vector_abs": 0.0, "magnitude_abs": 0.0, "target": 0.0},
            )
            bucket["nodes"] += nodes
            bucket["vector_abs"] += float(row["wss_vector_mae_pa"]) * nodes
            bucket["magnitude_abs"] += magnitude_abs
            bucket["target"] += target_magnitude

    result: dict[str, dict[str, object]] = {}
    for name, bucket in sorted(accumulators.items()):
        nodes = int(bucket["nodes"])
        result[name] = {
            "sampled_node_count": nodes,
            "wss_vector_mae_pa": bucket["vector_abs"] / nodes,
            "wss_magnitude_mae_pa": bucket["magnitude_abs"] / nodes,
            "wss_magnitude_relative_error": bucket["magnitude_abs"]
            / max(bucket["target"], 1.0e-12),
        }
    return result


def _region_balanced(regions: dict[str, dict[str, object]]) -> dict[str, object]:
    relative = [float(row["wss_magnitude_relative_error"]) for row in regions.values()]
    magnitude_mae = [float(row["wss_magnitude_mae_pa"]) for row in regions.values()]
    vector_mae = [float(row["wss_vector_mae_pa"]) for row in regions.values()]
    minimum_supported_nodes = 1_000
    supported = {
        name: row
        for name, row in regions.items()
        if int(row["sampled_node_count"]) >= minimum_supported_nodes
    }
    supported_relative = [
        float(row["wss_magnitude_relative_error"]) for row in supported.values()
    ]
    return {
        "region_count": len(regions),
        "macro_wss_vector_mae_pa": statistics.fmean(vector_mae),
        "macro_wss_magnitude_mae_pa": statistics.fmean(magnitude_mae),
        "macro_wss_magnitude_relative_error": statistics.fmean(relative),
        "median_region_wss_magnitude_relative_error": statistics.median(relative),
        "region_relative_error_range": [min(relative), max(relative)],
        "support_aware": {
            "minimum_sampled_nodes": minimum_supported_nodes,
            "included_regions": sorted(supported),
            "excluded_regions": sorted(set(regions) - set(supported)),
            "region_count": len(supported),
            "macro_wss_magnitude_relative_error": statistics.fmean(
                supported_relative
            ),
            "median_region_wss_magnitude_relative_error": statistics.median(
                supported_relative
            ),
        },
    }


def _delta(severe: float, primary: float) -> dict[str, float]:
    return {
        "absolute": severe - primary,
        "relative_fraction": (severe - primary) / max(abs(primary), 1.0e-12),
    }


def run_mask_ablation(
    canonical_root: Path,
    output_dir: Path,
    experiment_names: tuple[str, ...] = DEFAULT_EXPERIMENTS,
) -> dict[str, object]:
    split_path = canonical_root / "patient_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development = set(split["development"])
    locked = set(split["locked_test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    fold_results: list[dict[str, object]] = []

    for experiment_name in experiment_names:
        experiment_dir = canonical_root / "experiments" / experiment_name
        result_paths = list(experiment_dir.glob("*_result.json"))
        if len(result_paths) != 1:
            raise ValueError(f"expected exactly one result JSON in {experiment_dir}")
        training_result = json.loads(result_paths[0].read_text(encoding="utf-8"))
        validation_cases = list(training_result["validation_cases"])
        if not set(validation_cases) <= development or set(validation_cases) & locked:
            raise ValueError(f"invalid development validation cases in {result_paths[0]}")
        if set(training_result["locked_test_cases_not_accessed"]) != locked:
            raise ValueError(f"unexpected locked-test declaration in {result_paths[0]}")

        checkpoint_path = experiment_dir / training_result["best_checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if validation_cases != list(checkpoint["validation_cases"]):
            raise ValueError(f"checkpoint validation cases disagree in {checkpoint_path}")
        config = FoldTrainingConfig(**checkpoint["config"])
        model = _build_model(config.model_name)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device).eval()
        stats_path = experiment_dir / f"fold{config.fold_index}_normalization.json"

        policy_results = {
            policy: _evaluate(
                model,
                canonical_root,
                validation_cases,
                stats_path,
                config,
                device,
                target_mask_policy=policy,
            )
            for policy in POLICIES
        }
        registered = checkpoint.get("validation_metrics", {})
        primary_reproduction = {
            "registered_best_wss_magnitude_mae_pa": registered.get(
                "wss_magnitude_mae_pa"
            ),
            "reevaluated_wss_magnitude_mae_pa": policy_results["primary"][
                "wss_magnitude_mae_pa"
            ],
        }
        if registered:
            difference = abs(
                float(registered["wss_magnitude_mae_pa"])
                - float(policy_results["primary"]["wss_magnitude_mae_pa"])
            )
            primary_reproduction["absolute_difference_pa"] = difference
            # CUDA scatter reductions can vary by a few 1e-5 Pa across process
            # launches.  This tolerance is still five orders of magnitude below
            # the observed validation MAE and catches a wrong checkpoint/patch.
            primary_reproduction["tolerance_pa"] = 1.0e-4
            primary_reproduction["within_tolerance"] = difference <= 1.0e-4
            if difference > 1.0e-4:
                raise ValueError(
                    f"primary reevaluation did not reproduce {checkpoint_path}: {difference} Pa"
                )

        fold_results.append(
            {
                "fold_index": config.fold_index,
                "experiment": experiment_name,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "normalization_sha256": _sha256(stats_path),
                "validation_cases": validation_cases,
                "primary_reproduction": primary_reproduction,
                "policies": policy_results,
            }
        )

    if set().union(*(set(row["validation_cases"]) for row in fold_results)) != development:
        raise ValueError("the selected folds do not cover every development patient")
    for left_index, left in enumerate(fold_results):
        for right in fold_results[left_index + 1 :]:
            if set(left["validation_cases"]) & set(right["validation_cases"]):
                raise ValueError("a development patient appears in multiple validation folds")

    aggregate: dict[str, object] = {}
    for policy in POLICIES:
        overall = _aggregate_overall(
            [row["policies"][policy] for row in fold_results]
        )
        per_patient = _aggregate_named_buckets(
            [row["policies"][policy]["per_patient"] for row in fold_results]
        )
        per_region = _aggregate_named_buckets(
            [row["policies"][policy]["per_region"] for row in fold_results]
        )
        aggregate[policy] = {
            "overall": overall,
            "region_balanced": _region_balanced(per_region),
            "per_patient": per_patient,
            "per_region": per_region,
        }

    primary = aggregate["primary"]
    severe = aggregate["severe_sensitivity"]
    excluded_nodes = (
        int(primary["overall"]["sampled_node_count"])
        - int(severe["overall"]["sampled_node_count"])
    )
    comparison = {
        "excluded_sampled_node_count": excluded_nodes,
        "excluded_sampled_node_fraction": excluded_nodes
        / int(primary["overall"]["sampled_node_count"]),
        "wss_vector_mae_pa": _delta(
            severe["overall"]["wss_vector_mae_pa"],
            primary["overall"]["wss_vector_mae_pa"],
        ),
        "wss_magnitude_mae_pa": _delta(
            severe["overall"]["wss_magnitude_mae_pa"],
            primary["overall"]["wss_magnitude_mae_pa"],
        ),
        "wss_magnitude_relative_error": _delta(
            severe["overall"]["wss_magnitude_relative_error"],
            primary["overall"]["wss_magnitude_relative_error"],
        ),
        "mean_angular_error_degrees": _delta(
            severe["overall"]["mean_angular_error_degrees"],
            primary["overall"]["mean_angular_error_degrees"],
        ),
        "macro_region_wss_magnitude_relative_error": _delta(
            severe["region_balanced"]["macro_wss_magnitude_relative_error"],
            primary["region_balanced"]["macro_wss_magnitude_relative_error"],
        ),
        "support_aware_macro_region_wss_magnitude_relative_error": _delta(
            severe["region_balanced"]["support_aware"][
                "macro_wss_magnitude_relative_error"
            ],
            primary["region_balanced"]["support_aware"][
                "macro_wss_magnitude_relative_error"
            ],
        ),
        "per_region_relative_error_change": {
            region: _delta(
                severe["per_region"][region]["wss_magnitude_relative_error"],
                primary["per_region"][region]["wss_magnitude_relative_error"],
            )
            for region in primary["per_region"]
        },
    }
    report = {
        "schema_version": "1.0.0",
        "status": "development_cross_validation_mask_ablation_complete",
        "research_only": True,
        "design": "frozen checkpoints, normalization, patches, phases, and predictions; evaluation mask only",
        "patient_split_sha256": _sha256(split_path),
        "locked_test_opened": False,
        "development_patient_count": len(development),
        "fold_count": len(fold_results),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started,
        "folds": fold_results,
        "aggregate": aggregate,
        "comparison": comparison,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "equivariant_mask_ablation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    args = parser.parse_args()
    result = run_mask_ablation(
        args.canonical_root, args.output_dir, tuple(args.experiments)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
