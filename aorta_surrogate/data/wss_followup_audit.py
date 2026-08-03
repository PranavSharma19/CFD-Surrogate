"""Focused follow-up audits for Stanford cases 0032 and 0041."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from aorta_surrogate.data.features import REGION_NAMES
from aorta_surrogate.hemodynamics import HemodynamicMetrics, compute_hemodynamic_metrics


SEVERE_SCALED_JACOBIAN = 0.01
SEVERE_ASPECT_RATIO = 100.0
EXTREME_SCALED_JACOBIAN = 0.001
EXTREME_ASPECT_RATIO = 1000.0
TAWSS_FLOOR_PA = 0.1


def point_incident_quality(
    node_count: int,
    tetrahedra: np.ndarray,
    scaled_jacobian: np.ndarray,
    aspect_ratio: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the worst incident tetrahedral quality values to mesh points."""
    minimum_jacobian = np.full(node_count, np.inf, dtype=np.float64)
    maximum_aspect = np.zeros(node_count, dtype=np.float64)
    minimum_volume = np.full(node_count, np.inf, dtype=np.float64)
    for column in range(4):
        point_ids = tetrahedra[:, column]
        np.minimum.at(minimum_jacobian, point_ids, scaled_jacobian)
        np.maximum.at(maximum_aspect, point_ids, aspect_ratio)
        np.minimum.at(minimum_volume, point_ids, volume)
    return minimum_jacobian, maximum_aspect, minimum_volume


def _percentiles(values: np.ndarray, quantiles=(0, 1, 50, 95, 99, 100)) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    return {str(q): float(np.percentile(finite, q)) for q in quantiles}


def _extract_unique_suffix(archive: ZipFile, suffix: str, destination: Path) -> Path:
    names = [name for name in archive.namelist() if name.lower().endswith(suffix.lower())]
    if len(names) != 1:
        raise ValueError(f"expected one '*{suffix}' member, found {len(names)}")
    return Path(archive.extract(names[0], destination))


def _surface_to_volume_mapping(surface_points_mm: np.ndarray, volume) -> tuple[np.ndarray, np.ndarray]:
    import vtk

    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(volume)
    locator.BuildLocator()
    source_points_cm = np.asarray(surface_points_mm, dtype=np.float64) / 10.0
    mapping = np.fromiter(
        (locator.FindClosestPoint(point) for point in source_points_cm),
        dtype=np.int64,
        count=len(source_points_cm),
    )
    distance = np.linalg.norm(np.asarray(volume.points)[mapping] - source_points_cm, axis=1)
    return mapping, distance


def audit_0032_peak(canonical_root: Path, project_archive: Path) -> dict[str, object]:
    """Relate case 0032's global WSS peak to source volume-mesh quality."""
    import pyvista as pv
    import zarr

    case_dir = canonical_root / "0032_H_ABAO_AAA"
    diagnostics_dir = case_dir / "diagnostics"
    surface = pv.read(case_dir / "surface.vtp").triangulate()
    targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
    features = zarr.open_group(str(case_dir / "features.zarr"), mode="r")
    wss = np.asarray(targets["wss_pa"], dtype=np.float32)
    magnitude = np.linalg.norm(wss, axis=-1)
    peak_phase, peak_node = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)

    with TemporaryDirectory(prefix="wss_0032_volume_audit_") as temporary:
        with ZipFile(project_archive) as archive:
            volume_path = _extract_unique_suffix(
                archive, "mesh-complete.mesh.vtu", Path(temporary)
            )
        volume_mesh = pv.read(volume_path)

    if set(np.asarray(volume_mesh.celltypes).tolist()) != {10}:
        raise ValueError("case 0032 volume audit requires an all-tetrahedral VTK mesh")
    quality_mesh = volume_mesh.cell_quality(
        ["scaled_jacobian", "aspect_ratio", "volume"]
    )
    tetrahedra = np.asarray(volume_mesh.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(tetrahedra[:, 0] == 4):
        raise ValueError("unexpected non-tetrahedral connectivity")
    tetrahedra = tetrahedra[:, 1:]
    scaled_jacobian = np.asarray(quality_mesh.cell_data["scaled_jacobian"], dtype=np.float64)
    aspect_ratio = np.asarray(quality_mesh.cell_data["aspect_ratio"], dtype=np.float64)
    cell_volume = np.asarray(quality_mesh.cell_data["volume"], dtype=np.float64)
    point_min_jacobian, point_max_aspect, point_min_volume = point_incident_quality(
        volume_mesh.n_points,
        tetrahedra,
        scaled_jacobian,
        aspect_ratio,
        cell_volume,
    )
    mapping, mapping_distance_cm = _surface_to_volume_mapping(surface.points, volume_mesh)
    if float(mapping_distance_cm.max()) > 1.0e-3:
        raise ValueError("surface-to-volume mapping exceeded 0.001 cm")

    surface_min_jacobian = point_min_jacobian[mapping]
    surface_max_aspect = point_max_aspect[mapping]
    surface_min_volume = point_min_volume[mapping]
    severe = (surface_min_jacobian < SEVERE_SCALED_JACOBIAN) | (
        surface_max_aspect > SEVERE_ASPECT_RATIO
    )
    extreme = (surface_min_jacobian < EXTREME_SCALED_JACOBIAN) | (
        surface_max_aspect > EXTREME_ASPECT_RATIO
    )
    max_wss = magnitude.max(axis=0)
    high = max_wss > 100.0

    peak_volume_point = int(mapping[peak_node])
    peak_cell_ids = np.asarray(volume_mesh.point_cell_ids(peak_volume_point), dtype=np.int64)
    peak_cell_rows = [
        {
            "cell_id": int(cell_id),
            "scaled_jacobian": float(scaled_jacobian[cell_id]),
            "aspect_ratio": float(aspect_ratio[cell_id]),
            "volume_cm3": float(cell_volume[cell_id]),
            "volume_point_ids": tetrahedra[cell_id].astype(int).tolist(),
        }
        for cell_id in peak_cell_ids
    ]
    worst_peak_cell_id = int(
        peak_cell_ids[np.argmax(aspect_ratio[peak_cell_ids])]
    )

    overlay = surface.copy(deep=True)
    overlay.point_data["wss_max_pa"] = max_wss.astype(np.float32)
    overlay.point_data[f"wss_phase_{peak_phase}_pa"] = magnitude[peak_phase].astype(np.float32)
    overlay.point_data["volume_min_incident_scaled_jacobian"] = surface_min_jacobian.astype(np.float32)
    overlay.point_data["volume_max_incident_aspect_ratio"] = surface_max_aspect.astype(np.float32)
    overlay.point_data["volume_min_incident_cell_volume_cm3"] = surface_min_volume.astype(np.float32)
    overlay.point_data["candidate_severe_volume_qc"] = severe.astype(np.uint8)
    overlay.point_data["candidate_extreme_volume_qc"] = extreme.astype(np.uint8)
    overlay.point_data["global_peak_node"] = (np.arange(surface.n_points) == peak_node).astype(np.uint8)
    overlay_path = diagnostics_dir / "0032_peak_volume_mesh_audit.vtp"
    overlay.save(overlay_path)

    peak_cells = quality_mesh.extract_cells(peak_cell_ids)
    peak_cells_path = diagnostics_dir / "0032_peak_incident_volume_cells.vtu"
    peak_cells.save(peak_cells_path)

    image_path = diagnostics_dir / "0032_peak_volume_mesh_audit.png"
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 800))
    plotter.subplot(0, 0)
    plotter.add_mesh(overlay, color="lightgray", opacity=0.25)
    plotter.add_points(surface.points[high], color="red", point_size=4, render_points_as_spheres=True)
    plotter.add_points(surface.points[extreme], color="blue", point_size=8, render_points_as_spheres=True)
    plotter.add_points(surface.points[[peak_node]], color="yellow", point_size=18, render_points_as_spheres=True)
    plotter.add_text("Red: WSS >100 Pa; blue: extreme volume QC; yellow: peak", font_size=10)
    plotter.view_isometric()
    plotter.camera.zoom(1.2)
    plotter.subplot(0, 1)
    local_distance = np.linalg.norm(surface.points - surface.points[peak_node], axis=1)
    local = overlay.extract_points(local_distance <= 8.0, adjacent_cells=True)
    plotter.add_mesh(
        local,
        scalars=f"wss_phase_{peak_phase}_pa",
        cmap="turbo",
        show_edges=True,
        scalar_bar_args={"title": f"WSS phase {peak_phase} (Pa)"},
    )
    plotter.add_points(surface.points[[peak_node]], color="yellow", point_size=18, render_points_as_spheres=True)
    plotter.add_text("8 mm peak neighbourhood", font_size=11)
    plotter.view_isometric()
    plotter.screenshot(image_path)
    plotter.close()

    semantic_region = int(np.asarray(features["semantic_region_id"])[peak_node])
    report = {
        "schema_version": "1.0.0",
        "case_id": case_dir.name,
        "scope": "source volume-mesh follow-up; canonical targets unchanged",
        "peak": {
            "phase_index": int(peak_phase),
            "surface_node_id": int(peak_node),
            "wss_pa": float(magnitude[peak_phase, peak_node]),
            "coordinate_mm": np.asarray(surface.points[peak_node]).astype(float).tolist(),
            "semantic_region": REGION_NAMES.get(semantic_region, str(semantic_region)),
            "mapped_volume_point_id": peak_volume_point,
            "mapped_point_distance_cm": float(mapping_distance_cm[peak_node]),
            "minimum_incident_scaled_jacobian": float(surface_min_jacobian[peak_node]),
            "maximum_incident_aspect_ratio": float(surface_max_aspect[peak_node]),
            "minimum_incident_cell_volume_cm3": float(surface_min_volume[peak_node]),
            "incident_cells": peak_cell_rows,
            "worst_incident_cell_id": worst_peak_cell_id,
        },
        "volume_mesh": {
            "point_count": int(volume_mesh.n_points),
            "tetrahedron_count": int(volume_mesh.n_cells),
            "scaled_jacobian_percentiles": _percentiles(scaled_jacobian),
            "aspect_ratio_percentiles": _percentiles(aspect_ratio),
            "cell_volume_cm3_percentiles": _percentiles(cell_volume),
            "global_worst_aspect_ratio": float(aspect_ratio.max()),
            "global_minimum_scaled_jacobian": float(scaled_jacobian.min()),
            "global_minimum_cell_volume_cm3": float(cell_volume.min()),
        },
        "surface_mapping": {
            "maximum_distance_cm": float(mapping_distance_cm.max()),
            "p99_distance_cm": float(np.percentile(mapping_distance_cm, 99)),
        },
        "registered_qc_sensitivity_masks": {
            "severe_definition": (
                f"incident scaled Jacobian < {SEVERE_SCALED_JACOBIAN} or "
                f"incident aspect ratio > {SEVERE_ASPECT_RATIO}"
            ),
            "severe_surface_nodes": int(severe.sum()),
            "severe_surface_fraction": float(severe.mean()),
            "high_wss_nodes_in_severe_mask": int((high & severe).sum()),
            "high_wss_nodes_total": int(high.sum()),
            "extreme_definition": (
                f"incident scaled Jacobian < {EXTREME_SCALED_JACOBIAN} or "
                f"incident aspect ratio > {EXTREME_ASPECT_RATIO}"
            ),
            "extreme_surface_nodes": int(extreme.sum()),
            "high_wss_nodes_in_extreme_mask": int((high & extreme).sum()),
            "nodes_above_200_pa": int((max_wss > 200.0).sum()),
            "nodes_above_200_pa_in_severe_mask": int(((max_wss > 200.0) & severe).sum()),
            "nodes_above_300_pa": int((max_wss > 300.0).sum()),
            "nodes_above_300_pa_in_severe_mask": int(((max_wss > 300.0) & severe).sum()),
        },
        "decision": {
            "global_peak_reliable_for_training": False,
            "reason": (
                "the peak touches the globally worst aspect-ratio tetrahedron and a "
                "near-zero-volume, near-zero-Jacobian cell"
            ),
            "canonical_target_modified": False,
            "next_rule": (
                "evaluate the same volume-mesh QC thresholds across every development case "
                "before adopting a cohort-wide loss mask"
            ),
        },
        "artifacts": {
            "surface_overlay": str(overlay_path.relative_to(case_dir)),
            "peak_volume_cells": str(peak_cells_path.relative_to(case_dir)),
            "image": str(image_path.relative_to(case_dir)),
        },
    }
    report_path = diagnostics_dir / "0032_peak_volume_mesh_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _metric_difference(reference: HemodynamicMetrics, candidate: HemodynamicMetrics) -> dict[str, object]:
    tawss_absolute = np.abs(candidate.tawss - reference.tawss)
    stable = reference.tawss >= TAWSS_FLOOR_PA
    osi_absolute = np.abs(candidate.osi - reference.osi)
    rrt_valid = reference.rrt_valid & candidate.rrt_valid
    rrt_absolute = np.abs(candidate.rrt[rrt_valid] - reference.rrt[rrt_valid])
    return {
        "tawss_mae_pa": float(tawss_absolute.mean()),
        "tawss_relative_error": float(tawss_absolute.sum() / max(reference.tawss.sum(), 1.0e-12)),
        "tawss_absolute_difference_pa_percentiles": _percentiles(
            tawss_absolute, (50, 90, 95, 99, 99.9, 100)
        ),
        "osi_mae": float(osi_absolute[stable].mean()),
        "osi_absolute_difference_percentiles": _percentiles(
            osi_absolute[stable], (50, 90, 95, 99, 99.9, 100)
        ),
        "rrt_mae_pa_inv": float(rrt_absolute.mean()),
        "rrt_relative_error": float(
            rrt_absolute.sum() / max(np.abs(reference.rrt[rrt_valid]).sum(), 1.0e-12)
        ),
        "rrt_valid_node_count": int(rrt_valid.sum()),
    }


def audit_cycle_sensitivity(
    canonical_root: Path, case_id: str
) -> dict[str, object]:
    """Measure a development case's cycle-metric sensitivity to endpoint closure."""
    import zarr

    case_dir = canonical_root / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"canonical case not found: {case_dir}")
    targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
    features = zarr.open_group(str(case_dir / "features.zarr"), mode="r")
    wss = np.asarray(targets["wss_pa"], dtype=np.float64)
    times = np.asarray(targets["time_seconds"], dtype=np.float64)
    reference = compute_hemodynamic_metrics(wss, times, tawss_floor_pa=TAWSS_FLOOR_PA)

    first_closure = wss.copy()
    first_closure[-1] = first_closure[0]
    last_closure = wss.copy()
    last_closure[0] = last_closure[-1]
    midpoint_closure = wss.copy()
    midpoint = 0.5 * (midpoint_closure[0] + midpoint_closure[-1])
    midpoint_closure[0] = midpoint
    midpoint_closure[-1] = midpoint
    variants = {
        "first_endpoint_wins": first_closure,
        "last_endpoint_wins": last_closure,
        "midpoint_periodic_closure": midpoint_closure,
    }
    variant_metrics = {
        name: compute_hemodynamic_metrics(values, times, tawss_floor_pa=TAWSS_FLOOR_PA)
        for name, values in variants.items()
    }
    differences = {
        name: _metric_difference(reference, metrics)
        for name, metrics in variant_metrics.items()
    }

    region_ids = np.asarray(features["semantic_region_id"], dtype=np.int8)
    region_sensitivity: dict[str, object] = {}
    for region_id in sorted(np.unique(region_ids)):
        active = region_ids == region_id
        region_reference = HemodynamicMetrics(
            tawss=reference.tawss[active],
            osi=reference.osi[active],
            rrt=reference.rrt[active],
            rrt_valid=reference.rrt_valid[active],
        )
        region_sensitivity[REGION_NAMES.get(int(region_id), str(region_id))] = {
            name: _metric_difference(
                region_reference,
                HemodynamicMetrics(
                    tawss=metrics.tawss[active],
                    osi=metrics.osi[active],
                    rrt=metrics.rrt[active],
                    rrt_valid=metrics.rrt_valid[active],
                ),
            )
            for name, metrics in variant_metrics.items()
        }

    worst_tawss = max(row["tawss_relative_error"] for row in differences.values())
    worst_osi = max(row["osi_mae"] for row in differences.values())
    worst_rrt = max(row["rrt_relative_error"] for row in differences.values())
    accepted = worst_tawss <= 0.01 and worst_osi <= 0.01 and worst_rrt <= 0.05
    report = {
        "schema_version": "1.0.0",
        "case_id": case_dir.name,
        "scope": "cycle-endpoint sensitivity; canonical targets unchanged",
        "registered_variants": {
            "raw": "original first and last fields",
            "first_endpoint_wins": "replace final field with first field",
            "last_endpoint_wins": "replace first field with final field",
            "midpoint_periodic_closure": "replace both endpoint fields by their vector midpoint",
        },
        "differences_from_raw": differences,
        "per_region_differences_from_raw": region_sensitivity,
        "acceptance_thresholds": {
            "maximum_tawss_relative_error": 0.01,
            "maximum_osi_mae": 0.01,
            "maximum_rrt_relative_error": 0.05,
        },
        "worst_observed": {
            "tawss_relative_error": float(worst_tawss),
            "osi_mae": float(worst_osi),
            "rrt_relative_error": float(worst_rrt),
        },
        "decision": {
            "cycle_metrics_accepted_for_development": bool(accepted),
            "canonical_target_modified": False,
            "reason": (
                "TAWSS, OSI, and RRT remain within registered endpoint-sensitivity thresholds"
                if accepted
                else "one or more cycle metrics exceed endpoint-sensitivity thresholds"
            ),
        },
    }
    report_path = (
        case_dir
        / "diagnostics"
        / f"{case_id.split('_', 1)[0]}_cycle_endpoint_sensitivity.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_0041_cycle_sensitivity(canonical_root: Path) -> dict[str, object]:
    """Backward-compatible entry point for the registered case-0041 audit."""
    return audit_cycle_sensitivity(canonical_root, "0041_H_ABAO_AAA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    peak = audit_0032_peak(
        args.canonical_root, args.project_root / "0032_H_ABAO_AAA.zip"
    )
    cycle = audit_0041_cycle_sensitivity(args.canonical_root)
    summary = {
        "schema_version": "1.0.0",
        "locked_test_opened": False,
        "case_0032_decision": peak["decision"],
        "case_0041_decision": cycle["decision"],
    }
    output = args.canonical_root / "wss_followup_audit_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
