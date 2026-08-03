"""Audit high-range Stanford WSS labels without modifying training targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from aorta_surrogate.data.features import REGION_NAMES
from aorta_surrogate.data.outlier_diagnostics import (
    _nearest_distance,
    _triangles,
    active_components,
    boundary_vertices,
)


DEFAULT_CASES = (
    "0032_H_ABAO_AAA",
    "0040_H_ABAO_AAA",
    "0041_H_ABAO_AAA",
    "0036_H_ABAO_AAA",
    "0031_H_ABAO_AAA",
)
HIGH_WSS_PA = 100.0


def vector_tangentiality(wss: np.ndarray, normals: np.ndarray, floor_pa: float = 0.1) -> np.ndarray:
    """Return |WSS dot normal| / |WSS|, with low-magnitude entries marked NaN."""
    values = np.asarray(wss, dtype=np.float64)
    unit_normals = np.asarray(normals, dtype=np.float64)
    unit_normals /= np.maximum(np.linalg.norm(unit_normals, axis=-1, keepdims=True), 1.0e-12)
    magnitude = np.linalg.norm(values, axis=-1)
    normal_component = np.abs(np.einsum("pni,ni->pn", values, unit_normals))
    ratio = np.full(magnitude.shape, np.nan, dtype=np.float64)
    valid = magnitude >= floor_pa
    ratio[valid] = normal_component[valid] / magnitude[valid]
    return ratio


def triangle_quality(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle area and longest/shortest edge ratio."""
    vertices = np.asarray(points, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    edge_lengths = np.stack(
        [
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
        ],
        axis=1,
    )
    area = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1
    )
    edge_ratio = edge_lengths.max(axis=1) / np.maximum(edge_lengths.min(axis=1), 1.0e-12)
    return area, edge_ratio


def point_mesh_quality(
    node_count: int, triangles: np.ndarray, area: np.ndarray, edge_ratio: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map worst incident triangle area/aspect proxies to surface vertices."""
    minimum_area = np.full(node_count, np.inf, dtype=np.float64)
    maximum_edge_ratio = np.zeros(node_count, dtype=np.float64)
    for column in range(3):
        np.minimum.at(minimum_area, triangles[:, column], area)
        np.maximum.at(maximum_edge_ratio, triangles[:, column], edge_ratio)
    return minimum_area, maximum_edge_ratio


def endpoint_relative_l2(wss: np.ndarray) -> float:
    """Measure mismatch between the duplicate start/end samples of a complete cycle."""
    values = np.asarray(wss, dtype=np.float64)
    numerator = np.linalg.norm(values[-1] - values[0])
    denominator = 0.5 * (np.linalg.norm(values[-1]) + np.linalg.norm(values[0]))
    return float(numerator / max(denominator, 1.0e-12))


def endpoint_pointwise_report(wss: np.ndarray, floor_pa: float = 0.1) -> dict[str, object]:
    """Summarize nodewise mismatch between the first and last cycle samples."""
    values = np.asarray(wss, dtype=np.float64)
    absolute = np.linalg.norm(values[-1] - values[0], axis=-1)
    scale = 0.5 * (
        np.linalg.norm(values[-1], axis=-1) + np.linalg.norm(values[0], axis=-1)
    )
    relative = absolute / np.maximum(scale, floor_pa)
    return {
        "absolute_difference_pa_percentiles": _percentiles(
            absolute, (50, 90, 95, 99, 99.9, 100)
        ),
        "relative_difference_percentiles": _percentiles(
            relative, (50, 90, 95, 99, 99.9, 100)
        ),
    }


def peak_temporal_ratio(magnitude: np.ndarray) -> dict[str, object]:
    """Describe the global peak relative to the same node at adjacent phases."""
    values = np.asarray(magnitude, dtype=np.float64)
    phase, node = np.unravel_index(int(np.argmax(values)), values.shape)
    unique_phase_count = values.shape[0] - 1 if values.shape[0] > 2 else values.shape[0]
    canonical_phase = phase % unique_phase_count
    previous_phase = (canonical_phase - 1) % unique_phase_count
    next_phase = (canonical_phase + 1) % unique_phase_count
    peak = float(values[phase, node])
    adjacent = [float(values[previous_phase, node]), float(values[next_phase, node])]
    return {
        "phase_index": int(phase),
        "node_index": int(node),
        "peak_pa": peak,
        "adjacent_phase_values_pa": adjacent,
        "peak_to_adjacent_mean_ratio": float(peak / max(float(np.mean(adjacent)), 1.0e-12)),
        "node_cycle_values_pa": values[:, node].tolist(),
    }


def spatial_peak_coherence(magnitude: np.ndarray, edge_index: np.ndarray) -> dict[str, object]:
    """Compare the global peak with its surface neighbours and active component."""
    values = np.asarray(magnitude, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    phase, node = np.unravel_index(int(np.argmax(values)), values.shape)
    neighbors = np.unique(
        np.concatenate([edges[1, edges[0] == node], edges[0, edges[1] == node]])
    )
    neighbor_values = values[phase, neighbors]
    peak = float(values[phase, node])
    threshold_100 = values[phase] > HIGH_WSS_PA
    threshold_half_peak = values[phase] > 0.5 * peak
    components_100 = active_components(edges, threshold_100)
    components_half_peak = active_components(edges, threshold_half_peak)
    return {
        "phase_index": int(phase),
        "node_index": int(node),
        "neighbor_count": int(len(neighbors)),
        "neighbor_wss_pa": neighbor_values.tolist(),
        "maximum_neighbor_wss_pa": float(neighbor_values.max()),
        "median_neighbor_wss_pa": float(np.median(neighbor_values)),
        "peak_to_maximum_neighbor_ratio": float(peak / max(float(neighbor_values.max()), 1.0e-12)),
        "peak_to_median_neighbor_ratio": float(peak / max(float(np.median(neighbor_values)), 1.0e-12)),
        "phase_nodes_above_100_pa": int(threshold_100.sum()),
        "largest_phase_component_above_100_pa": int(components_100[0]) if components_100 else 0,
        "phase_nodes_above_half_peak": int(threshold_half_peak.sum()),
        "largest_phase_component_above_half_peak": (
            int(components_half_peak[0]) if components_half_peak else 0
        ),
    }


def _percentiles(values: np.ndarray, quantiles=(0, 50, 90, 95, 99, 99.9, 100)) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    return {str(q): float(np.percentile(finite, q)) for q in quantiles}


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_verification(
    case_dir: Path,
    project_archive: Path,
    result_archive: Path,
    phase_indices: list[int],
) -> dict[str, object]:
    """Independently re-read selected source arrays and compare with canonical WSS."""
    import zarr
    from vtk.util.numpy_support import vtk_to_numpy

    from aorta_surrogate.data.convert_stanford import (
        _read_polydata,
        _unique_member,
        _wall_to_result_mapping,
    )

    quality = json.loads((case_dir / "quality_report.json").read_text(encoding="utf-8"))
    timestep_ids = quality["selected_timestep_ids"]
    arrays = [f"vWSS_{timestep_ids[index]}" for index in phase_indices]
    targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
    canonical = np.asarray(targets["wss_pa"])[phase_indices]

    with TemporaryDirectory(prefix=f"wss_source_audit_{case_dir.name}_") as temporary:
        temporary_dir = Path(temporary)
        with ZipFile(project_archive) as archive:
            wall_path = Path(archive.extract(_unique_member(archive, "walls_combined.vtp"), temporary_dir))
        with ZipFile(result_archive) as archive:
            result_path = Path(archive.extract(_unique_member(archive, ".vtp"), temporary_dir))
        wall = _read_polydata(wall_path)
        result = _read_polydata(result_path, arrays)
        mapping, distances = _wall_to_result_mapping(wall, result, tolerance_cm=1.0e-3)
        source_cgs = np.stack(
            [vtk_to_numpy(result.GetPointData().GetArray(name))[mapping] for name in arrays]
        )

    # Check both an independent float64 conversion and the converter's documented
    # float32 operation order. The former can differ by a few ULPs.
    source_pa = np.asarray(source_cgs, dtype=np.float64) * 0.1
    converter_order_pa = source_cgs.astype(np.float32) * np.float32(0.1)
    difference = source_pa - np.asarray(canonical, dtype=np.float64)
    return {
        "project_archive": project_archive.name,
        "result_archive": result_archive.name,
        "project_sha256": _sha256(project_archive),
        "result_sha256": _sha256(result_archive),
        "phase_indices_checked": phase_indices,
        "source_array_names": arrays,
        "mapping_unique": bool(np.unique(mapping).size == mapping.size),
        "mapping_max_distance_cm": float(distances.max()),
        "mapping_p99_distance_cm": float(np.percentile(distances, 99)),
        "maximum_absolute_wss_difference_pa": float(np.abs(difference).max()),
        "mean_absolute_wss_difference_pa": float(np.abs(difference).mean()),
        "within_numerical_tolerance": bool(
            np.allclose(source_pa, canonical, rtol=1.0e-6, atol=1.0e-5)
        ),
        "exact_converter_float32_reproduction": bool(
            np.array_equal(converter_order_pa, canonical.astype(np.float32))
        ),
    }


def _region_reports(region_ids: np.ndarray, magnitude: np.ndarray) -> dict[str, object]:
    max_wss = magnitude.max(axis=0)
    result: dict[str, object] = {}
    for region_id in sorted(np.unique(region_ids)):
        active = region_ids == region_id
        result[REGION_NAMES.get(int(region_id), f"region_{region_id}")] = {
            "node_count": int(active.sum()),
            "maximum_wss_pa": float(max_wss[active].max()),
            "nodes_ever_above_100_pa": int((max_wss[active] > HIGH_WSS_PA).sum()),
            "wss_magnitude_percentiles_pa": _percentiles(magnitude[:, active], (50, 95, 99, 99.9, 100)),
        }
    return result


def _write_overlay_and_image(
    case_dir: Path,
    mesh,
    magnitude: np.ndarray,
    tangentiality: np.ndarray,
    boundary_distance: np.ndarray,
    point_min_area: np.ndarray,
    point_max_edge_ratio: np.ndarray,
) -> tuple[str, str]:
    import pyvista as pv

    diagnostics_dir = case_dir / "diagnostics"
    output = mesh.copy(deep=True)
    max_wss = magnitude.max(axis=0)
    peak_phase = int(np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)[0])
    output.point_data["audit_wss_max_pa"] = max_wss.astype(np.float32)
    output.point_data["audit_log10_1p_wss_max"] = np.log10(1.0 + max_wss).astype(np.float32)
    output.point_data["audit_wss_at_global_peak_phase_pa"] = magnitude[peak_phase].astype(np.float32)
    output.point_data["audit_fraction_phases_gt_100_pa"] = (magnitude > HIGH_WSS_PA).mean(axis=0).astype(np.float32)
    output.point_data["audit_tangentiality_ratio_max"] = np.max(
        np.nan_to_num(tangentiality, nan=0.0), axis=0
    ).astype(np.float32)
    output.point_data["audit_distance_to_open_boundary_mm"] = boundary_distance.astype(np.float32)
    output.point_data["audit_min_incident_triangle_area_mm2"] = point_min_area.astype(np.float32)
    output.point_data["audit_max_incident_edge_ratio"] = point_max_edge_ratio.astype(np.float32)
    overlay_path = diagnostics_dir / "wss_label_audit.vtp"
    output.save(overlay_path)

    image_path = diagnostics_dir / "wss_label_audit.png"
    plotter = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1800, 1000))
    views = (
        ("audit_log10_1p_wss_max", "log10(1 + maximum cycle WSS)", None),
        ("high_nodes", "Nodes ever above 100 Pa; yellow = global peak", None),
        (
            "audit_wss_at_global_peak_phase_pa",
            f"WSS at global peak phase {peak_phase} (Pa)",
            (0.0, float(np.percentile(magnitude[peak_phase], 99.5))),
        ),
        ("audit_tangentiality_ratio_max", "Maximum normal-component fraction", (0.0, 0.1)),
    )
    peak_node = int(np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)[1])
    high_nodes = max_wss > HIGH_WSS_PA
    for index, (scalars, title, limits) in enumerate(views):
        plotter.subplot(index // 2, index % 2)
        if scalars == "high_nodes":
            plotter.add_mesh(output, color="lightgray", opacity=0.30, show_edges=False)
            # A case with no values above the review threshold is a valid audit
            # result.  PyVista rejects an empty point cloud, so only add the red
            # overlay when such nodes actually exist.
            if high_nodes.any():
                plotter.add_points(
                    np.asarray(output.points)[high_nodes],
                    color="red",
                    point_size=4,
                    render_points_as_spheres=True,
                )
            plotter.add_points(
                np.asarray(output.points)[[peak_node]], color="yellow", point_size=16,
                render_points_as_spheres=True,
            )
        else:
            plotter.add_mesh(
                output, scalars=scalars, cmap="turbo", clim=limits, show_edges=False,
                scalar_bar_args={"title": title},
            )
        plotter.add_text(title, font_size=11)
        plotter.view_isometric()
        plotter.camera.zoom(1.2)
    plotter.link_views()
    plotter.screenshot(image_path)
    plotter.close()
    return str(overlay_path.relative_to(case_dir)), str(image_path.relative_to(case_dir))


def audit_case(
    case_dir: Path,
    *,
    project_archive: Path | None = None,
    result_archive: Path | None = None,
    verify_source: bool = False,
    render: bool = True,
) -> dict[str, object]:
    import pyvista as pv
    import zarr

    mesh = pv.read(case_dir / "surface.vtp").triangulate()
    targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
    features = zarr.open_group(str(case_dir / "features.zarr"), mode="r")
    wss = np.asarray(targets["wss_pa"], dtype=np.float32)
    times = np.asarray(targets["time_seconds"], dtype=np.float64)
    boundary_conditions = json.loads(
        (case_dir / "boundary_conditions.json").read_text(encoding="utf-8")
    )
    magnitude = np.linalg.norm(wss, axis=-1)
    node_features = np.asarray(features["node_features"], dtype=np.float32)
    normals = node_features[:, 3:6]
    tangentiality = vector_tangentiality(wss, normals)
    triangles = _triangles(mesh)
    boundary = boundary_vertices(triangles, mesh.n_points)
    boundary_distance = _nearest_distance(np.asarray(mesh.points), np.asarray(mesh.points)[boundary])
    area, edge_ratio = triangle_quality(np.asarray(mesh.points), triangles)
    point_min_area, point_max_edge_ratio = point_mesh_quality(
        mesh.n_points, triangles, area, edge_ratio
    )
    max_wss = magnitude.max(axis=0)
    high = max_wss > HIGH_WSS_PA
    worst_aspect_threshold = float(np.percentile(point_max_edge_ratio, 99))
    peak_report = peak_temporal_ratio(magnitude)
    spatial_peak = spatial_peak_coherence(
        magnitude, np.asarray(features["edge_index"], dtype=np.int64)
    )
    phase_indices = sorted({0, int(peak_report["phase_index"]), wss.shape[0] - 1})
    region_ids = np.asarray(features["semantic_region_id"], dtype=np.int8)
    endpoint_by_region = {
        REGION_NAMES.get(int(region_id), f"region_{region_id}"): endpoint_relative_l2(
            wss[:, region_ids == region_id]
        )
        for region_id in sorted(np.unique(region_ids))
    }

    source_report = None
    if verify_source:
        if project_archive is None or result_archive is None:
            raise ValueError("source verification requires project and result archives")
        source_report = _source_verification(case_dir, project_archive, result_archive, phase_indices)

    high_boundary_distances = boundary_distance[high]
    high_worst_aspect_fraction = float(
        (point_max_edge_ratio[high] >= worst_aspect_threshold).mean()
    ) if high.any() else 0.0
    overall_worst_aspect_fraction = float((point_max_edge_ratio >= worst_aspect_threshold).mean())
    report: dict[str, object] = {
        "schema_version": "1.0.0",
        "case_id": case_dir.name,
        "scope": "development-only WSS label audit; targets were not modified",
        "phase_count": int(wss.shape[0]),
        "node_count": int(wss.shape[1]),
        "finite_wss": bool(np.isfinite(wss).all()),
        "wss_magnitude_pa": {
            "percentiles": _percentiles(magnitude),
            "nodes_ever_above_100_pa": int(high.sum()),
            "node_fraction_ever_above_100_pa": float(high.mean()),
        },
        "source_conversion_and_mapping": source_report,
        "tangentiality": {
            "definition": "absolute normal WSS component divided by WSS magnitude for WSS >= 0.1 Pa",
            "ratio_percentiles": _percentiles(tangentiality),
            "fraction_ratio_above_0_05": float(np.nanmean(tangentiality > 0.05)),
            "fraction_ratio_above_0_10": float(np.nanmean(tangentiality > 0.10)),
        },
        "temporal_coherence": {
            "cycle_endpoint_relative_vector_l2": endpoint_relative_l2(wss),
            "cycle_endpoint_pointwise": endpoint_pointwise_report(wss),
            "cycle_endpoint_relative_vector_l2_by_region": endpoint_by_region,
            "target_cycle_duration_seconds": float(times[-1] - times[0]),
            "boundary_condition_period_seconds": float(
                boundary_conditions["heart_period_seconds"]
            ),
            "period_alignment_absolute_difference_seconds": float(
                abs((times[-1] - times[0]) - boundary_conditions["heart_period_seconds"])
            ),
            "boundary_condition_flow_endpoint_difference_m3_s": float(
                abs(
                    boundary_conditions["inlet_flow_m3_s"][-1]
                    - boundary_conditions["inlet_flow_m3_s"][0]
                )
            ),
            "global_peak": peak_report,
            "phase_maximum_wss_pa": magnitude.max(axis=1).astype(float).tolist(),
            "phase_p99_9_wss_pa": np.percentile(magnitude, 99.9, axis=1).astype(float).tolist(),
            "nodes_above_100_pa_by_phase": (magnitude > HIGH_WSS_PA).sum(axis=1).astype(int).tolist(),
        },
        "spatial_peak_coherence": spatial_peak,
        "boundary_relation": {
            "surface_boundary_node_count": int(boundary.sum()),
            "high_wss_boundary_node_count": int((boundary & high).sum()),
            "high_wss_boundary_node_fraction": float((boundary & high).sum() / max(high.sum(), 1)),
            "high_wss_distance_to_open_boundary_mm_percentiles": _percentiles(
                high_boundary_distances, (0, 25, 50, 75, 100)
            ) if high.any() else {},
            "high_wss_fraction_within_2mm_of_boundary": float(
                (high_boundary_distances <= 2.0).mean()
            ) if high.any() else 0.0,
        },
        "mesh_relation": {
            "triangle_area_mm2_percentiles": _percentiles(area, (0, 1, 50, 99, 100)),
            "triangle_edge_ratio_percentiles": _percentiles(edge_ratio, (0, 50, 95, 99, 100)),
            "high_wss_median_min_incident_area_mm2": float(np.median(point_min_area[high])) if high.any() else None,
            "all_node_median_min_incident_area_mm2": float(np.median(point_min_area)),
            "high_wss_fraction_in_worst_1pct_edge_ratio": high_worst_aspect_fraction,
            "all_node_fraction_in_worst_1pct_edge_ratio": overall_worst_aspect_fraction,
            "worst_aspect_enrichment": float(
                high_worst_aspect_fraction / max(overall_worst_aspect_fraction, 1.0e-12)
            ),
        },
        "semantic_regions": _region_reports(region_ids, magnitude),
        "automated_flags": [],
        "provisional_decision": "visual_review_required",
    }

    flags: list[str] = report["automated_flags"]  # type: ignore[assignment]
    if not report["finite_wss"]:
        flags.append("non_finite_wss")
    if source_report is not None and not source_report["within_numerical_tolerance"]:
        flags.append("source_to_canonical_mismatch")
    if report["tangentiality"]["ratio_percentiles"]["99"] > 0.05:  # type: ignore[index]
        flags.append("wss_has_material_normal_component")
    if report["temporal_coherence"]["cycle_endpoint_relative_vector_l2"] > 0.05:  # type: ignore[index]
        flags.append("cycle_endpoint_mismatch")
    if peak_report["peak_to_adjacent_mean_ratio"] > 2.0:
        flags.append("global_peak_is_temporal_spike")
    if spatial_peak["peak_to_maximum_neighbor_ratio"] > 2.0:
        flags.append("global_peak_is_spatially_discontinuous")
    if report["boundary_relation"]["high_wss_fraction_within_2mm_of_boundary"] > 0.50:  # type: ignore[index]
        flags.append("high_wss_concentrated_near_open_boundaries")
    if report["mesh_relation"]["worst_aspect_enrichment"] > 3.0:  # type: ignore[index]
        flags.append("high_wss_enriched_on_worst_aspect_elements")

    if render:
        overlay, image = _write_overlay_and_image(
            case_dir,
            mesh,
            magnitude,
            tangentiality,
            boundary_distance,
            point_min_area,
            point_max_edge_ratio,
        )
        report["overlay"] = overlay
        report["image"] = image

    report_path = case_dir / "diagnostics" / "wss_label_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_dataset(
    canonical_root: Path,
    case_ids: list[str],
    *,
    source_root: Path | None = None,
    verify_source: bool = False,
    render: bool = True,
) -> dict[str, object]:
    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    development = set(split["development"])
    prohibited = [case_id for case_id in case_ids if case_id not in development]
    if prohibited:
        raise ValueError(f"label audit is restricted to development cases: {prohibited}")

    reports = []
    for case_id in case_ids:
        project_archive = None
        result_archive = None
        if source_root is not None:
            project_archive = source_root / "projects" / f"{case_id}.zip"
            result_archive = source_root / "surface_results" / f"{case_id}_3D_RIGID_VTP.zip"
        reports.append(
            audit_case(
                canonical_root / case_id,
                project_archive=project_archive,
                result_archive=result_archive,
                verify_source=verify_source,
                render=render,
            )
        )
    # Build the cohort summary from every completed development-case audit, not
    # just the cases requested in this invocation.  This makes incremental
    # auditing safe: a later batch cannot erase earlier evidence from the
    # summary, and locked patients are never traversed.
    completed_reports: list[dict[str, object]] = []
    for development_case_id in split["development"]:
        report_path = (
            canonical_root
            / development_case_id
            / "diagnostics"
            / "wss_label_audit.json"
        )
        if not report_path.exists():
            continue
        completed_report = json.loads(report_path.read_text(encoding="utf-8"))
        if completed_report.get("case_id") != development_case_id:
            raise ValueError(
                f"label-audit report identity mismatch at {report_path}: "
                f"expected {development_case_id!r}, got {completed_report.get('case_id')!r}"
            )
        completed_reports.append(completed_report)

    completed_case_ids = [str(row["case_id"]) for row in completed_reports]
    remaining_case_ids = [
        case_id for case_id in split["development"] if case_id not in completed_case_ids
    ]
    source_verified_case_ids = [
        str(row["case_id"])
        for row in completed_reports
        if row.get("source_conversion_and_mapping") is not None
        and row["source_conversion_and_mapping"].get("within_numerical_tolerance", False)
    ]

    summary = {
        "schema_version": "1.1.0",
        "scope": "development cases only; locked test cases not accessed",
        "requested_case_ids": case_ids,
        "requested_verify_source": verify_source,
        "case_ids": completed_case_ids,
        "case_count": len(completed_case_ids),
        "remaining_unaudited_case_ids": remaining_case_ids,
        "source_verified_case_ids": source_verified_case_ids,
        "source_verified_case_count": len(source_verified_case_ids),
        "cases": [
            {
                "case_id": row["case_id"],
                "maximum_wss_pa": row["wss_magnitude_pa"]["percentiles"]["100"],
                "nodes_ever_above_100_pa": row["wss_magnitude_pa"]["nodes_ever_above_100_pa"],
                "tangentiality_p99": row["tangentiality"]["ratio_percentiles"]["99"],
                "cycle_endpoint_relative_vector_l2": row["temporal_coherence"]["cycle_endpoint_relative_vector_l2"],
                "peak_to_adjacent_mean_ratio": row["temporal_coherence"]["global_peak"]["peak_to_adjacent_mean_ratio"],
                "peak_to_maximum_neighbor_ratio": row["spatial_peak_coherence"]["peak_to_maximum_neighbor_ratio"],
                "high_wss_boundary_fraction": row["boundary_relation"]["high_wss_boundary_node_fraction"],
                "worst_aspect_enrichment": row["mesh_relation"]["worst_aspect_enrichment"],
                "source_verified": (
                    row.get("source_conversion_and_mapping") is not None
                    and row["source_conversion_and_mapping"].get(
                        "within_numerical_tolerance", False
                    )
                ),
                "automated_flags": row["automated_flags"],
                "provisional_decision": row["provisional_decision"],
            }
            for row in completed_reports
        ],
    }
    (canonical_root / "wss_label_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.verify_source and args.source_root is None:
        parser.error("--verify-source requires --source-root")
    result = audit_dataset(
        args.canonical_root,
        args.cases,
        source_root=args.source_root,
        verify_source=args.verify_source,
        render=not args.no_render,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
