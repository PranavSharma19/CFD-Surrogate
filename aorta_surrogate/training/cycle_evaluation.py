"""Evaluate complete predicted cycles on fixed connected validation patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aorta_surrogate.hemodynamics import HemodynamicMetrics, compute_hemodynamic_metrics
from aorta_surrogate.training.fold_trainer import _build_model
from aorta_surrogate.training.pyg_adapter import make_training_patch


STABILITY_FLOOR_PA = 0.1


def _metric_errors(predicted, target) -> dict[str, object]:
    tawss_absolute = np.abs(predicted.tawss - target.tawss)
    osi_absolute = np.abs(predicted.osi - target.osi)
    osi_stable = target.tawss >= STABILITY_FLOOR_PA
    rrt_mask = predicted.rrt_valid & target.rrt_valid & np.isfinite(predicted.rrt) & np.isfinite(target.rrt)
    rrt_absolute = np.abs(predicted.rrt[rrt_mask] - target.rrt[rrt_mask])
    return {
        "node_count": int(len(target.tawss)),
        "tawss_mae_pa": float(tawss_absolute.mean()),
        "tawss_relative_error": float(tawss_absolute.sum() / max(target.tawss.sum(), 1.0e-12)),
        "osi_mae": float(osi_absolute[osi_stable].mean()) if osi_stable.any() else None,
        "osi_mae_all_nodes": float(osi_absolute.mean()),
        "osi_stable_node_count": int(osi_stable.sum()),
        "rrt_valid_node_count": int(rrt_mask.sum()),
        "rrt_mae_pa_inv": float(rrt_absolute.mean()) if rrt_absolute.size else None,
        "rrt_relative_error": (
            float(rrt_absolute.sum() / max(np.abs(target.rrt[rrt_mask]).sum(), 1.0e-12))
            if rrt_absolute.size else None
        ),
    }


def evaluate_experiment(canonical_root: Path, experiment_dir: Path) -> dict[str, object]:
    result_paths = list(experiment_dir.glob("*_result.json"))
    if len(result_paths) != 1:
        raise ValueError(f"expected exactly one result JSON in {experiment_dir}")
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    if not result.get("locked_test_cases_not_accessed"):
        raise ValueError("experiment does not declare its locked-test exclusion")
    checkpoint_path = experiment_dir / result["best_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = _build_model(config["model_name"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    stats_path = experiment_dir / f"fold{config['fold_index']}_normalization.json"

    case_results: dict[str, object] = {}
    pooled_predicted: list[HemodynamicMetrics] = []
    pooled_target: list[HemodynamicMetrics] = []
    for case_offset, case_id in enumerate(result["validation_cases"]):
        patch_seed = int(config["seed"] + 50_000 + case_offset * 137)
        predicted_phases: list[np.ndarray] = []
        target_phases: list[np.ndarray] = []
        node_ids: np.ndarray | None = None
        target_valid: np.ndarray | None = None
        with torch.no_grad():
            for phase_index in range(21):
                patch = make_training_patch(
                    canonical_root,
                    case_id,
                    phase_index,
                    max_nodes=int(config["max_nodes"]),
                    seed=patch_seed,
                    normalization_path=stats_path,
                    patch_method=config["patch_method"],
                ).to(device)
                current_ids = patch.global_node_ids.detach().cpu().numpy()
                if node_ids is None:
                    node_ids = current_ids
                elif not np.array_equal(node_ids, current_ids):
                    raise ValueError("cycle phases do not share identical patch nodes")
                current_valid = patch.target_valid_mask.detach().cpu().numpy().astype(bool)
                if target_valid is None:
                    target_valid = current_valid
                elif not np.array_equal(target_valid, current_valid):
                    raise ValueError("cycle phases do not share identical quality masks")
                predicted_phases.append(
                    (model(patch) * patch.target_scale_pa).detach().float().cpu().numpy()
                )
                target_phases.append(
                    (patch.y * patch.target_scale_pa).detach().float().cpu().numpy()
                )
        predicted_wss = np.stack(predicted_phases)
        target_wss = np.stack(target_phases)
        if target_valid is None or not target_valid.any():
            raise ValueError("cycle patch contains no valid target nodes")
        predicted_wss = predicted_wss[:, target_valid]
        target_wss = target_wss[:, target_valid]
        import zarr

        target_group = zarr.open_group(str(canonical_root / case_id / "targets.zarr"), mode="r")
        times = np.asarray(target_group["time_seconds"])
        predicted_metrics = compute_hemodynamic_metrics(
            predicted_wss, times, tawss_floor_pa=STABILITY_FLOOR_PA
        )
        target_metrics = compute_hemodynamic_metrics(
            target_wss, times, tawss_floor_pa=STABILITY_FLOOR_PA
        )
        case_results[case_id] = _metric_errors(predicted_metrics, target_metrics)
        pooled_predicted.append(predicted_metrics)
        pooled_target.append(target_metrics)

    # Node correspondence is unnecessary for pooled scalar metric errors.
    def concatenate_metrics(rows: list[HemodynamicMetrics]) -> HemodynamicMetrics:
        return HemodynamicMetrics(
            tawss=np.concatenate([row.tawss for row in rows]),
            osi=np.concatenate([row.osi for row in rows]),
            rrt=np.concatenate([row.rrt for row in rows]),
            rrt_valid=np.concatenate([row.rrt_valid for row in rows]),
        )

    pooled_predicted_metrics = concatenate_metrics(pooled_predicted)
    pooled_target_metrics = concatenate_metrics(pooled_target)
    report = {
        "schema_version": "1.0.0",
        "status": "development_validation_complete_cycles",
        "model_name": config["model_name"],
        "fold_index": config["fold_index"],
        "phase_count": 21,
        "tawss_stability_floor_pa": STABILITY_FLOOR_PA,
        "patch_method": config["patch_method"],
        "locked_test_opened": False,
        "validation_cases": result["validation_cases"],
        "pooled": _metric_errors(pooled_predicted_metrics, pooled_target_metrics),
        "per_patient": case_results,
    }
    (experiment_dir / "cycle_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_experiment(args.canonical_root, args.experiment_dir), indent=2))


if __name__ == "__main__":
    main()
