"""Export spatial WSS diagnostics without modifying or clipping CFD labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


THRESHOLDS_PA = (20.0, 50.0, 100.0)


def _triangles(mesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        raise ValueError("surface has no polygon faces")
    rows = faces.reshape(-1, 4)
    if not np.all(rows[:, 0] == 3):
        raise ValueError("diagnostics currently require a triangular surface")
    return rows[:, 1:]


def boundary_vertices(triangles: np.ndarray, node_count: int) -> np.ndarray:
    """Return a mask for vertices incident to a one-sided surface edge."""
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary_edges = edges[counts[inverse] == 1]
    mask = np.zeros(node_count, dtype=bool)
    if boundary_edges.size:
        mask[np.unique(boundary_edges)] = True
    return mask


def active_components(edge_index: np.ndarray, active: np.ndarray) -> list[int]:
    """Sizes of connected active-node components, largest first."""
    active_ids = np.flatnonzero(active)
    if active_ids.size == 0:
        return []
    parent = np.arange(active.size, dtype=np.int64)
    size = np.ones(active.size, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in edge_index.T:
        left = int(left)
        right = int(right)
        if not (active[left] and active[right]):
            continue
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
    return sorted(Counter(find(int(node)) for node in active_ids).values(), reverse=True)


def _nearest_distance(points: np.ndarray, reference: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float32)
    if len(reference) == 0:
        return np.full(len(points), np.nan, dtype=np.float32)
    result = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        squared = np.square(chunk[:, None, :] - reference[None, :, :]).sum(axis=2)
        result[start : start + len(chunk)] = np.sqrt(squared.min(axis=1))
    return result


def diagnose_case(case_dir: Path) -> dict[str, object]:
    import pyvista as pv
    import zarr

    mesh = pv.read(case_dir / "surface.vtp")
    targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
    features = zarr.open_group(str(case_dir / "features.zarr"), mode="r")
    wss = np.asarray(targets["wss_pa"], dtype=np.float32)
    magnitude = np.linalg.norm(wss, axis=-1)
    if magnitude.shape[1] != mesh.n_points:
        raise ValueError(f"surface/target node mismatch for {case_dir.name}")

    triangles = _triangles(mesh)
    boundary = boundary_vertices(triangles, mesh.n_points)
    node_features = np.asarray(features["node_features"], dtype=np.float32)
    branch_id = np.rint(node_features[:, 8]).astype(np.int32)
    max_wss = magnitude.max(axis=0)
    per_node_p99 = np.percentile(magnitude, 99, axis=0).astype(np.float32)
    phase_of_peak = magnitude.argmax(axis=0).astype(np.int16)

    output = mesh.copy(deep=True)
    output.point_data["wss_max_pa"] = max_wss
    output.point_data["wss_p99_over_cycle_pa"] = per_node_p99
    output.point_data["phase_index_of_peak_wss"] = phase_of_peak
    output.point_data["tawss_pa"] = np.asarray(targets["tawss_pa"], dtype=np.float32)
    output.point_data["osi"] = np.asarray(targets["osi"], dtype=np.float32)
    output.point_data["rrt_pa_inv"] = np.asarray(targets["rrt_pa_inv"], dtype=np.float32)
    output.point_data["is_surface_boundary"] = boundary.astype(np.uint8)
    output.point_data["nearest_centerline_branch_id"] = branch_id

    threshold_reports: dict[str, object] = {}
    for threshold in THRESHOLDS_PA:
        fraction = (magnitude > threshold).mean(axis=0).astype(np.float32)
        active = max_wss > threshold
        name = f"fraction_phases_gt_{int(threshold)}pa"
        output.point_data[name] = fraction
        components = active_components(np.asarray(features["edge_index"]), active)
        distances = _nearest_distance(mesh.points[active], mesh.points[boundary])
        branch_counts = Counter(str(int(value)) for value in branch_id[active])
        threshold_reports[str(int(threshold))] = {
            "active_node_count": int(active.sum()),
            "active_node_fraction": float(active.mean()),
            "boundary_active_node_count": int((active & boundary).sum()),
            "connected_component_count": len(components),
            "largest_component_nodes": int(components[0]) if components else 0,
            "largest_component_fraction_of_active": float(components[0] / active.sum()) if components else 0.0,
            "distance_to_open_boundary_mm_percentiles": {
                str(q): float(np.nanpercentile(distances, q)) if distances.size else None
                for q in (0, 25, 50, 75, 100)
            },
            "nearest_centerline_branch_counts": dict(sorted(branch_counts.items())),
        }

    diagnostics_dir = case_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    overlay_path = diagnostics_dir / "wss_outliers.vtp"
    output.save(overlay_path)
    report = {
        "schema_version": "1.0.0",
        "case_id": case_dir.name,
        "interpretation": "Descriptive CFD-label audit only; no values were clipped or excluded.",
        "phase_count": int(magnitude.shape[0]),
        "node_count": int(magnitude.shape[1]),
        "surface_boundary_node_count": int(boundary.sum()),
        "global_wss_magnitude_pa": {
            "max": float(magnitude.max()),
            "percentiles": {str(q): float(np.percentile(magnitude, q)) for q in (50, 90, 95, 99, 99.9)},
        },
        "thresholds_pa": threshold_reports,
        "overlay": str(overlay_path.relative_to(case_dir)),
    }
    (diagnostics_dir / "wss_outlier_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def diagnose_dataset(canonical_root: Path) -> dict[str, object]:
    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    case_ids = sorted(set(split["development"]) | set(split["locked_test"]))
    reports = [diagnose_case(canonical_root / case_id) for case_id in case_ids]
    summary = {
        "schema_version": "1.0.0",
        "case_count": len(reports),
        "cases": [
            {
                "case_id": report["case_id"],
                "max_wss_pa": report["global_wss_magnitude_pa"]["max"],
                "nodes_ever_gt_100pa": report["thresholds_pa"]["100"]["active_node_count"],
                "largest_gt_100pa_component": report["thresholds_pa"]["100"]["largest_component_nodes"],
            }
            for report in reports
        ],
    }
    (canonical_root / "wss_outlier_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose_dataset(args.canonical_root), indent=2))


if __name__ == "__main__":
    main()
