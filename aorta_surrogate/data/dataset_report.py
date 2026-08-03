"""Build feasibility-cohort QC, outlier, and patient-split reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assign_cv_folds(case_rows: list[dict], fold_count: int = 3) -> list[list[str]]:
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    loads = [0 for _ in range(fold_count)]
    for row in sorted(case_rows, key=lambda item: (-item["wall_nodes"], item["case_id"])):
        target = min(range(fold_count), key=lambda index: (loads[index], index))
        folds[target].append(row["case_id"])
        loads[target] += row["wall_nodes"]
    return [sorted(fold) for fold in folds]


def create_patient_split(case_rows: list[dict]) -> dict[str, object]:
    """Lock tests using input complexity only, never target performance."""

    if len(case_rows) != 15:
        raise ValueError(f"expected 15 canonical cases, found {len(case_rows)}")
    smallest = min(case_rows, key=lambda row: (row["wall_nodes"], row["case_id"]))
    largest = max(case_rows, key=lambda row: (row["wall_nodes"], row["case_id"]))
    remaining = [row for row in case_rows if row["case_id"] not in {smallest["case_id"], largest["case_id"]}]
    longest_period = max(remaining, key=lambda row: (row["heart_period_seconds"], row["case_id"]))
    locked_test = sorted([smallest["case_id"], largest["case_id"], longest_period["case_id"]])
    development_rows = [row for row in case_rows if row["case_id"] not in locked_test]
    development = sorted(row["case_id"] for row in development_rows)
    return {
        "schema_version": "1.0.0",
        "split_policy": "input-only extremes: smallest mesh, largest mesh, longest heart period",
        "locked_test": locked_test,
        "development": development,
        "development_cv_folds": _assign_cv_folds(development_rows),
    }


def build_report(canonical_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    import zarr

    rows: list[dict] = []
    thresholds = (20.0, 50.0, 100.0)
    for case_dir in sorted(path for path in canonical_root.glob("*_H_ABAO_AAA") if path.is_dir()):
        quality = _load_json(case_dir / "quality_report.json")
        boundary = _load_json(case_dir / "boundary_conditions.json")
        group = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
        wss = np.asarray(group["wss_pa"])
        magnitude = np.linalg.norm(wss, axis=-1)
        rows.append(
            {
                "case_id": case_dir.name,
                "qc_status": quality["status"],
                "wall_nodes": quality["canonical_wall_points"],
                "available_phases": quality["available_wss_phases"],
                "selected_phases": quality["selected_phases"],
                "heart_period_seconds": boundary["heart_period_seconds"],
                "rcr_outlets": len(boundary["outlet_parameters"]["outlets"]),
                "centerline_paths": quality["centerlines"]["path_count"],
                "wss_max_pa": float(magnitude.max()),
                "wss_p99_pa": float(np.percentile(magnitude, 99)),
                "node_phase_fraction_above_threshold": {
                    str(threshold): float(np.mean(magnitude > threshold)) for threshold in thresholds
                },
            }
        )

    split = create_patient_split(rows)
    report = {
        "schema_version": "1.0.0",
        "case_count": len(rows),
        "all_qc_pass": all(row["qc_status"] == "pass" for row in rows),
        "total_wall_nodes": sum(row["wall_nodes"] for row in rows),
        "node_range": [min(row["wall_nodes"] for row in rows), max(row["wall_nodes"] for row in rows)],
        "heart_period_range_seconds": [
            min(row["heart_period_seconds"] for row in rows),
            max(row["heart_period_seconds"] for row in rows),
        ],
        "cases": rows,
    }
    return report, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    report, split = build_report(args.canonical_root)
    (args.canonical_root / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.canonical_root / "patient_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps({"summary": {key: value for key, value in report.items() if key != "cases"}, "split": split}, indent=2))


if __name__ == "__main__":
    main()

