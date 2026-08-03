"""Apply frozen source-volume mesh QC rules to development cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from aorta_surrogate.data.wss_followup_audit import (
    EXTREME_ASPECT_RATIO,
    EXTREME_SCALED_JACOBIAN,
    SEVERE_ASPECT_RATIO,
    SEVERE_SCALED_JACOBIAN,
    _extract_unique_suffix,
    _percentiles,
    _surface_to_volume_mapping,
    point_incident_quality,
)


WSS_THRESHOLDS_PA = (20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0)


def mask_enrichment(mask: np.ndarray, population: np.ndarray) -> float:
    """Return mask prevalence in a population divided by overall prevalence."""
    values = np.asarray(mask, dtype=bool)
    active = np.asarray(population, dtype=bool)
    if not active.any() or not values.any():
        return 0.0
    return float(values[active].mean() / values.mean())


def audit_volume_case(case_dir: Path, project_archive: Path) -> dict[str, object]:
    """Audit one canonical surface against its original tetrahedral volume mesh."""
    import pyvista as pv
    import zarr

    diagnostics_dir = case_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    surface = pv.read(case_dir / "surface.vtp").triangulate()
    targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
    wss = np.asarray(targets["wss_pa"], dtype=np.float32)
    magnitude = np.linalg.norm(wss, axis=-1)
    max_wss = magnitude.max(axis=0)
    peak_phase, peak_node = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)

    with TemporaryDirectory(prefix=f"volume_qc_{case_dir.name}_") as temporary:
        with ZipFile(project_archive) as archive:
            volume_path = _extract_unique_suffix(
                archive, "mesh-complete.mesh.vtu", Path(temporary)
            )
        volume_mesh = pv.read(volume_path)

    if set(np.asarray(volume_mesh.celltypes).tolist()) != {10}:
        raise ValueError(f"{case_dir.name} source volume is not all tetrahedra")
    quality = volume_mesh.cell_quality(["scaled_jacobian", "aspect_ratio", "volume"])
    connectivity = np.asarray(volume_mesh.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(connectivity[:, 0] == 4):
        raise ValueError(f"{case_dir.name} contains unexpected cell connectivity")
    tetrahedra = connectivity[:, 1:]
    scaled_jacobian = np.asarray(quality.cell_data["scaled_jacobian"], dtype=np.float64)
    aspect_ratio = np.asarray(quality.cell_data["aspect_ratio"], dtype=np.float64)
    cell_volume = np.asarray(quality.cell_data["volume"], dtype=np.float64)
    point_min_jacobian, point_max_aspect, point_min_volume = point_incident_quality(
        volume_mesh.n_points,
        tetrahedra,
        scaled_jacobian,
        aspect_ratio,
        cell_volume,
    )
    mapping, mapping_distance = _surface_to_volume_mapping(surface.points, volume_mesh)
    if float(mapping_distance.max()) > 1.0e-3:
        raise ValueError(f"{case_dir.name} surface-to-volume mapping exceeded 0.001 cm")
    surface_min_jacobian = point_min_jacobian[mapping]
    surface_max_aspect = point_max_aspect[mapping]
    surface_min_volume = point_min_volume[mapping]
    severe = (surface_min_jacobian < SEVERE_SCALED_JACOBIAN) | (
        surface_max_aspect > SEVERE_ASPECT_RATIO
    )
    extreme = (surface_min_jacobian < EXTREME_SCALED_JACOBIAN) | (
        surface_max_aspect > EXTREME_ASPECT_RATIO
    )

    threshold_report: dict[str, object] = {}
    for threshold in WSS_THRESHOLDS_PA:
        active = max_wss > threshold
        threshold_report[str(int(threshold))] = {
            "node_count": int(active.sum()),
            "severe_node_count": int((active & severe).sum()),
            "extreme_node_count": int((active & extreme).sum()),
            "severe_enrichment": mask_enrichment(severe, active),
            "extreme_enrichment": mask_enrichment(extreme, active),
        }

    top_count = min(20, surface.n_points)
    top_nodes = np.argpartition(max_wss, -top_count)[-top_count:]
    top_nodes = top_nodes[np.argsort(max_wss[top_nodes])[::-1]]
    top_rows = [
        {
            "surface_node_id": int(node),
            "maximum_wss_pa": float(max_wss[node]),
            "phase_index": int(np.argmax(magnitude[:, node])),
            "minimum_incident_scaled_jacobian": float(surface_min_jacobian[node]),
            "maximum_incident_aspect_ratio": float(surface_max_aspect[node]),
            "minimum_incident_cell_volume_cm3": float(surface_min_volume[node]),
            "severe": bool(severe[node]),
            "extreme": bool(extreme[node]),
        }
        for node in top_nodes
    ]

    overlay = surface.copy(deep=True)
    overlay.point_data["wss_max_pa"] = max_wss.astype(np.float32)
    overlay.point_data["volume_min_incident_scaled_jacobian"] = surface_min_jacobian.astype(np.float32)
    overlay.point_data["volume_max_incident_aspect_ratio"] = surface_max_aspect.astype(np.float32)
    overlay.point_data["volume_min_incident_cell_volume_cm3"] = surface_min_volume.astype(np.float32)
    overlay.point_data["candidate_severe_volume_qc"] = severe.astype(np.uint8)
    overlay.point_data["candidate_extreme_volume_qc"] = extreme.astype(np.uint8)
    overlay.point_data["global_peak_node"] = (np.arange(surface.n_points) == peak_node).astype(np.uint8)
    overlay_path = diagnostics_dir / "volume_mesh_qc.vtp"
    overlay.save(overlay_path)

    report = {
        "schema_version": "1.0.0",
        "case_id": case_dir.name,
        "scope": "development candidate masks only; canonical targets unchanged",
        "frozen_thresholds": {
            "severe": {
                "minimum_incident_scaled_jacobian_below": SEVERE_SCALED_JACOBIAN,
                "maximum_incident_aspect_ratio_above": SEVERE_ASPECT_RATIO,
            },
            "extreme": {
                "minimum_incident_scaled_jacobian_below": EXTREME_SCALED_JACOBIAN,
                "maximum_incident_aspect_ratio_above": EXTREME_ASPECT_RATIO,
            },
        },
        "source_volume": {
            "point_count": int(volume_mesh.n_points),
            "tetrahedron_count": int(volume_mesh.n_cells),
            "scaled_jacobian_percentiles": _percentiles(scaled_jacobian),
            "aspect_ratio_percentiles": _percentiles(aspect_ratio),
            "cell_volume_cm3_percentiles": _percentiles(cell_volume),
            "minimum_scaled_jacobian": float(scaled_jacobian.min()),
            "maximum_aspect_ratio": float(aspect_ratio.max()),
            "minimum_cell_volume_cm3": float(cell_volume.min()),
        },
        "surface_mapping": {
            "node_count": int(surface.n_points),
            "maximum_distance_cm": float(mapping_distance.max()),
            "p99_distance_cm": float(np.percentile(mapping_distance, 99)),
        },
        "candidate_masks": {
            "severe_node_count": int(severe.sum()),
            "severe_node_fraction": float(severe.mean()),
            "extreme_node_count": int(extreme.sum()),
            "extreme_node_fraction": float(extreme.mean()),
        },
        "wss": {
            "maximum_pa": float(max_wss.max()),
            "global_peak_phase": int(peak_phase),
            "global_peak_node": int(peak_node),
            "global_peak_severe": bool(severe[peak_node]),
            "global_peak_extreme": bool(extreme[peak_node]),
            "maximum_pa_outside_severe_mask": float(max_wss[~severe].max()),
            "maximum_pa_outside_extreme_mask": float(max_wss[~extreme].max()),
            "thresholds": threshold_report,
            "top_20_nodes": top_rows,
        },
        "artifacts": {"surface_overlay": str(overlay_path.relative_to(case_dir))},
    }
    report_path = diagnostics_dir / "volume_mesh_qc.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_development_cohort(canonical_root: Path, project_root: Path) -> dict[str, object]:
    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    case_ids = list(split["development"])
    locked = set(split["locked_test"])
    if set(case_ids) & locked:
        raise ValueError("development and locked-test cases overlap")

    reports: list[dict[str, object]] = []
    for case_id in case_ids:
        report = audit_volume_case(
            canonical_root / case_id, project_root / f"{case_id}.zip"
        )
        reports.append(report)
        masks = report["candidate_masks"]
        print(
            f"PASS {case_id}: {masks['severe_node_count']} severe, "
            f"{masks['extreme_node_count']} extreme surface nodes",
            flush=True,
        )

    total_nodes = sum(row["surface_mapping"]["node_count"] for row in reports)
    total_severe = sum(row["candidate_masks"]["severe_node_count"] for row in reports)
    total_extreme = sum(row["candidate_masks"]["extreme_node_count"] for row in reports)
    threshold_totals: dict[str, object] = {}
    for threshold in WSS_THRESHOLDS_PA:
        key = str(int(threshold))
        population = sum(row["wss"]["thresholds"][key]["node_count"] for row in reports)
        severe = sum(row["wss"]["thresholds"][key]["severe_node_count"] for row in reports)
        extreme = sum(row["wss"]["thresholds"][key]["extreme_node_count"] for row in reports)
        threshold_totals[key] = {
            "node_count": int(population),
            "severe_node_count": int(severe),
            "extreme_node_count": int(extreme),
            "severe_fraction": float(severe / max(population, 1)),
            "extreme_fraction": float(extreme / max(population, 1)),
        }

    summary = {
        "schema_version": "1.0.0",
        "scope": "all development cases; locked test cases not accessed",
        "locked_test_opened": False,
        "frozen_before_cohort_run": True,
        "case_count": len(reports),
        "case_ids": case_ids,
        "total_surface_nodes": int(total_nodes),
        "candidate_masks": {
            "severe_node_count": int(total_severe),
            "severe_node_fraction": float(total_severe / total_nodes),
            "extreme_node_count": int(total_extreme),
            "extreme_node_fraction": float(total_extreme / total_nodes),
        },
        "wss_threshold_totals": threshold_totals,
        "global_peak_mask_status": {
            "severe_case_ids": [row["case_id"] for row in reports if row["wss"]["global_peak_severe"]],
            "extreme_case_ids": [row["case_id"] for row in reports if row["wss"]["global_peak_extreme"]],
        },
        "cases": [
            {
                "case_id": row["case_id"],
                "surface_nodes": row["surface_mapping"]["node_count"],
                "tetrahedra": row["source_volume"]["tetrahedron_count"],
                "maximum_wss_pa": row["wss"]["maximum_pa"],
                "severe_node_fraction": row["candidate_masks"]["severe_node_fraction"],
                "extreme_node_fraction": row["candidate_masks"]["extreme_node_fraction"],
                "global_peak_severe": row["wss"]["global_peak_severe"],
                "global_peak_extreme": row["wss"]["global_peak_extreme"],
                "maximum_wss_outside_severe_mask_pa": row["wss"]["maximum_pa_outside_severe_mask"],
                "maximum_wss_outside_extreme_mask_pa": row["wss"]["maximum_pa_outside_extreme_mask"],
            }
            for row in reports
        ],
    }
    output_path = canonical_root / "volume_mesh_qc_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def publish_quality_masks(canonical_root: Path) -> dict[str, object]:
    """Publish the frozen extreme rule as a loss mask; keep severe as sensitivity only."""
    import pyvista as pv
    import zarr

    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    case_ids = list(split["development"])
    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        case_dir = canonical_root / case_id
        overlay = pv.read(case_dir / "diagnostics" / "volume_mesh_qc.vtp")
        severe = np.asarray(overlay.point_data["candidate_severe_volume_qc"], dtype=bool)
        extreme = np.asarray(overlay.point_data["candidate_extreme_volume_qc"], dtype=bool)
        target_valid = ~extreme
        output = zarr.open_group(str(case_dir / "quality_masks.zarr"), mode="w")
        chunk = (min(overlay.n_points, 32768),)
        output.create_array("target_valid", data=target_valid, chunks=chunk)
        output.create_array("volume_mesh_extreme_invalid", data=extreme, chunks=chunk)
        output.create_array("volume_mesh_severe_sensitivity", data=severe, chunks=chunk)
        output.attrs.update(
            {
                "schema_version": "1.0.0",
                "target_valid_policy": "exclude extreme source-volume mesh nodes from losses and metrics",
                "extreme_minimum_incident_scaled_jacobian_below": EXTREME_SCALED_JACOBIAN,
                "extreme_maximum_incident_aspect_ratio_above": EXTREME_ASPECT_RATIO,
                "severe_mask_policy": "sensitivity analysis only; not excluded from primary training",
                "canonical_targets_modified": False,
            }
        )
        rows.append(
            {
                "case_id": case_id,
                "node_count": int(overlay.n_points),
                "invalid_node_count": int(extreme.sum()),
                "valid_node_count": int(target_valid.sum()),
                "severe_sensitivity_node_count": int(severe.sum()),
            }
        )
    report = {
        "schema_version": "1.0.0",
        "scope": "development only; locked test cases not accessed",
        "locked_test_opened": False,
        "decision": {
            "primary_loss_mask": "volume_mesh_extreme_invalid",
            "primary_invalid_node_count": int(sum(row["invalid_node_count"] for row in rows)),
            "primary_total_node_count": int(sum(row["node_count"] for row in rows)),
            "severe_rule": "sensitivity_only",
            "canonical_targets_modified": False,
        },
        "cases": rows,
    }
    (canonical_root / "quality_mask_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--publish-existing-extreme-mask",
        action="store_true",
        help="publish quality_masks.zarr from existing volume_mesh_qc.vtp overlays",
    )
    args = parser.parse_args()
    if args.publish_existing_extreme_mask:
        result = publish_quality_masks(args.canonical_root)
    else:
        result = audit_development_cohort(args.canonical_root, args.project_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
