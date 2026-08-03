"""Compute leakage-safe normalization statistics from development patients only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class RunningMoments:
    def __init__(self, width: int):
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.sum_squares = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1, self.sum.size)
        self.count += flat.shape[0]
        self.sum += flat.sum(axis=0)
        self.sum_squares += np.square(flat).sum(axis=0)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot normalize an empty dataset")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_squares / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std < 1.0e-12] = 1.0
        return mean, std


def _split_hash(case_ids: list[str]) -> str:
    payload = "\n".join(sorted(case_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_case_stats(
    canonical_root: Path,
    case_ids: list[str],
    *,
    source_split: str,
    excluded_case_ids: list[str] | None = None,
) -> dict[str, object]:
    import zarr

    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    development = list(case_ids)
    locked_test = set(split["locked_test"])
    if locked_test.intersection(development):
        raise ValueError("patient leakage detected between development and locked test")
    unknown = set(development) - set(split["development"])
    if unknown:
        raise ValueError(f"normalization cases are outside development split: {sorted(unknown)}")

    node_moments: RunningMoments | None = None
    target_moments = RunningMoments(3)
    conditioning_rows: list[np.ndarray] = []
    magnitude_samples: list[np.ndarray] = []
    masked_target_nodes = 0
    boundary_distance_moments = RunningMoments(2)

    from aorta_surrogate.data.canonical_dataset import phase_conditioning

    feature_names: list[str] | None = None
    for case_id in development:
        case_dir = canonical_root / case_id
        features = zarr.open_group(str(case_dir / "features.zarr"), mode="r")
        targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
        boundary = json.loads((case_dir / "boundary_conditions.json").read_text(encoding="utf-8"))
        case_feature_names = list(features.attrs["feature_names"])
        if feature_names is None:
            feature_names = case_feature_names
            node_moments = RunningMoments(len(feature_names))
        elif feature_names != case_feature_names:
            raise ValueError(f"feature contract differs for {case_id}")

        node_moments.update(np.asarray(features["node_features"]))
        if "inlet_geodesic_distance_mm" in features and "outlet_geodesic_distance_mm" in features:
            boundary_distance_moments.update(
                np.column_stack(
                    [
                        np.asarray(features["inlet_geodesic_distance_mm"]),
                        np.asarray(features["outlet_geodesic_distance_mm"]),
                    ]
                )
            )
        wss = np.asarray(targets["wss_pa"])
        quality_mask_path = case_dir / "quality_masks.zarr"
        if quality_mask_path.exists():
            quality_masks = zarr.open_group(str(quality_mask_path), mode="r")
            target_valid = np.asarray(quality_masks["target_valid"], dtype=bool)
            if target_valid.shape != (wss.shape[1],):
                raise ValueError(f"target-valid mask shape differs for {case_id}")
        else:
            target_valid = np.ones(wss.shape[1], dtype=bool)
        masked_target_nodes += int((~target_valid).sum())
        valid_wss = wss[:, target_valid]
        target_moments.update(valid_wss)
        magnitude = np.linalg.norm(valid_wss, axis=-1).reshape(-1)
        stride = max(1, magnitude.size // 50_000)
        magnitude_samples.append(magnitude[::stride][:50_000])
        for time_value in np.asarray(targets["time_seconds"]):
            conditioning_rows.append(phase_conditioning(float(time_value), boundary))

    node_mean, node_std = node_moments.result()
    target_mean, target_std = target_moments.result()
    target_scalar_scale = float(np.sqrt(np.mean(np.square(target_std) + np.square(target_mean))))
    position_scalar_scale = float(np.sqrt(np.mean(np.square(node_std[:3]) + np.square(node_mean[:3]))))
    conditioning = np.stack(conditioning_rows)
    condition_mean = conditioning.mean(axis=0, dtype=np.float64)
    condition_std = conditioning.std(axis=0, dtype=np.float64)
    condition_std[condition_std < 1.0e-12] = 1.0
    boundary_distance_mean, boundary_distance_std = boundary_distance_moments.result()

    # Branch identifiers are categorical and the availability bit is a mask;
    # neither is standardized as a continuous physical feature.
    for index in (8, 9):
        node_mean[index] = 0.0
        node_std[index] = 1.0

    sampled_magnitudes = np.concatenate(magnitude_samples)
    return {
        "schema_version": "1.0.0",
        "source_split": source_split,
        "development_case_ids": sorted(development),
        "locked_test_case_ids_excluded": sorted(locked_test),
        "other_case_ids_excluded": sorted(excluded_case_ids or []),
        "development_split_sha256": _split_hash(development),
        "node_feature_names": feature_names,
        "node_mean": node_mean.tolist(),
        "node_std": node_std.tolist(),
        "categorical_feature_indices": [8],
        "mask_feature_indices": [9],
        "conditioning_names": [
            "sin_phase", "cos_phase", "flow_at_phase_m3_s", "peak_abs_flow_m3_s",
            "heart_period_seconds", "blood_density_kg_m3", "dynamic_viscosity_pa_s",
        ],
        "conditioning_mean": condition_mean.tolist(),
        "conditioning_std": condition_std.tolist(),
        "target_mean_pa": target_mean.tolist(),
        "target_std_pa": target_std.tolist(),
        "equivariant_target_mean_pa": [0.0, 0.0, 0.0],
        "equivariant_target_scale_pa": target_scalar_scale,
        "equivariant_position_scale_mm": position_scalar_scale,
        "boundary_distance_names": ["inlet_geodesic_distance_mm", "outlet_geodesic_distance_mm"],
        "boundary_distance_mean": boundary_distance_mean.tolist(),
        "boundary_distance_std": boundary_distance_std.tolist(),
        "sampled_wss_magnitude_percentiles_pa": {
            str(q): float(np.percentile(sampled_magnitudes, q)) for q in (50, 90, 95, 99, 99.9)
        },
        "quality_mask_policy": "exclude quality_masks.zarr target_valid == false when present",
        "masked_static_target_node_count": masked_target_nodes,
    }


def compute_development_stats(canonical_root: Path) -> dict[str, object]:
    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    return compute_case_stats(
        canonical_root,
        list(split["development"]),
        source_split="development",
        excluded_case_ids=list(split["locked_test"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    stats = compute_development_stats(args.canonical_root)
    output = args.canonical_root / "normalization_stats.json"
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
