"""Build reusable surface-graph features for canonical aortic cases."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np


FEATURE_NAMES = (
    "x_centered_mm",
    "y_centered_mm",
    "z_centered_mm",
    "normal_x",
    "normal_y",
    "normal_z",
    "mean_curvature_per_mm",
    "centerline_distance_mm",
    "nearest_branch_id",
    "centerline_available",
)

REGION_NAMES = {
    0: "aorta",
    1: "renal",
    2: "mesenteric",
    3: "celiac_hepatic_splenic",
    4: "iliac",
    5: "explicit_aneurysm_path",
    6: "other",
}


def surface_edges(mesh) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if faces.size == 0:
        raise ValueError("surface has no polygonal faces")
    cells = faces.reshape(-1, 4)
    if not np.all(cells[:, 0] == 3):
        raise ValueError("canonical surface must be triangular")
    triangles = cells[:, 1:]
    undirected = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=0,
    )
    undirected.sort(axis=1)
    undirected = np.unique(undirected, axis=0)
    directed = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    return directed.T.astype(np.int64, copy=False)


def edge_index_to_csr(edge_index: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic surface adjacency for connected geodesic sampling."""
    order = np.lexsort((edge_index[1], edge_index[0]))
    source = edge_index[0, order]
    indices = edge_index[1, order].astype(np.int64, copy=False)
    counts = np.bincount(source, minlength=node_count)
    indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return indptr, indices


def boundary_components(triangles: np.ndarray) -> list[np.ndarray]:
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    adjacency: dict[int, list[int]] = {}
    for left, right in boundary_edges:
        adjacency.setdefault(int(left), []).append(int(right))
        adjacency.setdefault(int(right), []).append(int(left))
    remaining = set(adjacency)
    components: list[np.ndarray] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return sorted(components, key=len, reverse=True)


def _centerline_endpoint_assignments(centerlines, boundary_centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    endpoints: list[np.ndarray] = []
    for cell_index in range(centerlines.n_cells):
        point_ids = np.asarray(centerlines.get_cell(cell_index).point_ids, dtype=np.int64)
        endpoints.extend([centerlines.points[point_ids[0]], centerlines.points[point_ids[-1]]])
    endpoint_array = np.asarray(endpoints)
    distances = np.linalg.norm(endpoint_array[:, None, :] - boundary_centroids[None, :, :], axis=2)
    assignment = distances.argmin(axis=1)
    return assignment, distances[np.arange(len(endpoint_array)), assignment]


def _cap_centroids(project_archive: Path) -> tuple[list[str], np.ndarray]:
    import pyvista as pv

    with ZipFile(project_archive) as archive, TemporaryDirectory(prefix="aorta_caps_") as temporary:
        members = [
            name for name in archive.namelist()
            if "/mesh-surfaces/" in name.lower()
            and name.lower().endswith(".vtp")
            and "wall" not in PurePosixPath(name).stem.lower()
        ]
        if not members:
            raise ValueError(f"no SimVascular cap surfaces found in {project_archive}")
        names: list[str] = []
        centroids: list[np.ndarray] = []
        for member in sorted(members):
            extracted = Path(archive.extract(member, temporary))
            cap = pv.read(extracted)
            names.append(PurePosixPath(member).stem)
            centroids.append(np.asarray(cap.points).mean(axis=0) * 10.0)  # cm -> mm
    return names, np.asarray(centroids)


def _one_to_one_centroid_assignment(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(left) != len(right):
        raise ValueError("boundary-loop and cap counts differ")
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    assignment = np.full(len(left), -1, dtype=np.int64)
    used_right: set[int] = set()
    for flat_index in np.argsort(distance, axis=None):
        left_index, right_index = np.unravel_index(flat_index, distance.shape)
        if assignment[left_index] >= 0 or int(right_index) in used_right:
            continue
        assignment[left_index] = right_index
        used_right.add(int(right_index))
        if len(used_right) == len(right):
            break
    if np.any(assignment < 0):
        raise ValueError("could not assign all topological boundaries to cap surfaces")
    return assignment, distance[np.arange(len(left)), assignment]


def weighted_geodesic_distance(
    points: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    distance = np.full(len(points), np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for source in np.unique(sources):
        source = int(source)
        distance[source] = 0.0
        heapq.heappush(queue, (0.0, source))
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        for neighbor in indices[indptr[node] : indptr[node + 1]]:
            neighbor = int(neighbor)
            candidate = current_distance + float(np.linalg.norm(points[node] - points[neighbor]))
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    if not np.isfinite(distance).all():
        raise ValueError("surface graph is disconnected from the requested boundary")
    return distance.astype(np.float32)


def coarse_geodesic_edges(
    points: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    *,
    hops: int = 4,
    neighbors_per_node: int = 4,
) -> np.ndarray:
    """Sparse long-range edges chosen from an exact geodesic-hop frontier."""
    sources: list[int] = []
    targets: list[int] = []
    for node in range(len(points)):
        visited = {node}
        frontier = {node}
        for _ in range(hops):
            next_frontier: set[int] = set()
            for current in frontier:
                next_frontier.update(int(value) for value in indices[indptr[current] : indptr[current + 1]])
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        candidates = np.asarray(sorted(frontier), dtype=np.int64)
        if len(candidates) > neighbors_per_node:
            squared = np.square(points[candidates] - points[node]).sum(axis=1)
            chosen = np.argpartition(squared, -neighbors_per_node)[-neighbors_per_node:]
            candidates = np.sort(candidates[chosen])
        sources.extend([node] * len(candidates))
        targets.extend(candidates.tolist())
    return np.asarray([sources, targets], dtype=np.int64)


def classify_branch_name(name: str) -> int:
    normalized = name.lower()
    if "aneurysm" in normalized or "aysm" in normalized:
        return 5
    if "renal" in normalized:
        return 1
    if "sma" in normalized or "ima" in normalized or "mesenteric" in normalized:
        return 2
    if any(token in normalized for token in ("celiac", "hepatic", "splenic")):
        return 3
    if "iliac" in normalized or "illiac" in normalized:
        return 4
    if "aorta" in normalized:
        return 0
    return 6


def _semantic_regions(centerlines, branch_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    cell_branch_ids = np.asarray(centerlines.cell_data["branch_id"], dtype=np.int32)
    cell_branch_names = [str(value) for value in centerlines.cell_data["branch_name"]]
    names_by_id = dict(zip(cell_branch_ids.tolist(), cell_branch_names, strict=True))
    class_by_id = {branch_id: classify_branch_name(name) for branch_id, name in names_by_id.items()}
    semantic = np.asarray([class_by_id.get(int(value), 6) for value in branch_ids], dtype=np.int8)
    aneurysm_source_available = any(value == 5 for value in class_by_id.values())
    explicit_aneurysm = (semantic == 5).astype(np.uint8)
    availability = np.full(len(semantic), int(aneurysm_source_available), dtype=np.uint8)
    return semantic, explicit_aneurysm, availability, names_by_id


def _nearest_centerline_features(mesh, centerlines) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(centerlines)
    locator.BuildLocator()
    points = np.asarray(mesh.points)
    centerline_points = np.asarray(centerlines.points)
    indices = np.fromiter(
        (locator.FindClosestPoint(point) for point in points),
        dtype=np.int64,
        count=len(points),
    )
    distances = np.linalg.norm(points - centerline_points[indices], axis=1)
    branch_array = centerlines.GetPointData().GetArray("branch_id")
    if branch_array is None:
        raise ValueError("centerlines are missing point-data branch_id")
    branch_ids = vtk_to_numpy(branch_array)[indices]
    tangent_array = centerlines.GetPointData().GetArray("tangent")
    if tangent_array is None:
        raise ValueError("centerlines are missing point-data tangent")
    tangents = vtk_to_numpy(tangent_array)[indices].astype(np.float32)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-8)
    return distances.astype(np.float32), branch_ids.astype(np.float32), tangents


def build_case_features(case_dir: Path, *, force: bool = False) -> dict[str, object]:
    import pyvista as pv
    import zarr

    output = case_dir / "features.zarr"
    report_path = case_dir / "feature_report.json"
    if output.exists() and report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))

    mesh = pv.read(case_dir / "surface.vtp").triangulate()
    centerlines = pv.read(case_dir / "centerlines.vtp")
    with_normals = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
    )
    normals = np.asarray(with_normals.point_data["Normals"], dtype=np.float32)
    curvature = np.asarray(mesh.curvature(curv_type="mean"), dtype=np.float64)
    curvature_finite = np.isfinite(curvature)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    centerline_distance, branch_ids, centerline_tangent = _nearest_centerline_features(mesh, centerlines)
    semantic_region, explicit_aneurysm, aneurysm_available, branch_names = _semantic_regions(
        centerlines, branch_ids
    )
    surface_points = np.asarray(mesh.points, dtype=np.float32)
    surface_centroid = surface_points.mean(axis=0, dtype=np.float64)
    centered_points = surface_points - surface_centroid.astype(np.float32)

    node_features = np.column_stack(
        [
            centered_points,
            normals,
            curvature.astype(np.float32),
            centerline_distance,
            branch_ids,
            np.ones(mesh.n_points, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    edges = surface_edges(mesh)
    adjacency_indptr, adjacency_indices = edge_index_to_csr(edges, mesh.n_points)
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    open_boundaries = boundary_components(triangles)
    boundary_centroids = np.stack([surface_points[component].mean(axis=0) for component in open_boundaries])
    endpoint_assignment, endpoint_distance = _centerline_endpoint_assignments(centerlines, boundary_centroids)
    assignment_counts = np.bincount(endpoint_assignment, minlength=len(open_boundaries))
    boundary_conditions = json.loads((case_dir / "boundary_conditions.json").read_text(encoding="utf-8"))
    expected_boundaries = len(boundary_conditions["outlet_parameters"]["outlets"]) + 1
    if len(open_boundaries) != expected_boundaries:
        raise ValueError(
            f"found {len(open_boundaries)} open boundaries but boundary conditions imply {expected_boundaries}"
        )
    project_archive = case_dir.parent.parent / "projects" / f"{case_dir.name}.zip"
    cap_names, cap_centroids = _cap_centroids(project_archive)
    cap_assignment, cap_distance = _one_to_one_centroid_assignment(boundary_centroids, cap_centroids)
    inlet_cap_indices = [
        index for index, name in enumerate(cap_names) if "inflow" in name.lower() or "inlet" in name.lower()
    ]
    if len(inlet_cap_indices) != 1:
        raise ValueError(f"expected one named inflow/inlet cap, found {inlet_cap_indices}")
    inlet_matches = np.flatnonzero(cap_assignment == inlet_cap_indices[0])
    if len(inlet_matches) != 1:
        raise ValueError("named inlet cap did not map uniquely to a topological boundary")
    inlet_boundary = int(inlet_matches[0])
    inlet_distance = weighted_geodesic_distance(
        surface_points, adjacency_indptr, adjacency_indices, open_boundaries[inlet_boundary]
    )
    outlet_sources = np.concatenate(
        [component for index, component in enumerate(open_boundaries) if index != inlet_boundary]
    )
    outlet_distance = weighted_geodesic_distance(
        surface_points, adjacency_indptr, adjacency_indices, outlet_sources
    )
    coarse_edges = coarse_geodesic_edges(surface_points, adjacency_indptr, adjacency_indices)

    group = zarr.open_group(str(output), mode="w")
    group.create_array("node_features", data=node_features, chunks=(min(mesh.n_points, 32768), len(FEATURE_NAMES)))
    group.create_array("edge_index", data=edges, chunks=(2, min(edges.shape[1], 262144)))
    group.create_array("adjacency_indptr", data=adjacency_indptr, chunks=(min(len(adjacency_indptr), 262144),))
    group.create_array("adjacency_indices", data=adjacency_indices, chunks=(min(len(adjacency_indices), 262144),))
    group.create_array("coarse_edge_index", data=coarse_edges, chunks=(2, min(coarse_edges.shape[1], 262144)))
    group.create_array("inlet_geodesic_distance_mm", data=inlet_distance, chunks=(min(mesh.n_points, 32768),))
    group.create_array("outlet_geodesic_distance_mm", data=outlet_distance, chunks=(min(mesh.n_points, 32768),))
    group.create_array("semantic_region_id", data=semantic_region, chunks=(min(mesh.n_points, 32768),))
    group.create_array("centerline_tangent", data=centerline_tangent, chunks=(min(mesh.n_points, 32768), 3))
    group.create_array("explicit_aneurysm_region", data=explicit_aneurysm, chunks=(min(mesh.n_points, 32768),))
    group.create_array("explicit_aneurysm_available", data=aneurysm_available, chunks=(min(mesh.n_points, 32768),))
    group.attrs["feature_names"] = list(FEATURE_NAMES)
    group.attrs["semantic_region_names"] = {str(key): value for key, value in REGION_NAMES.items()}
    group.attrs["branch_names_by_id"] = {str(key): value for key, value in branch_names.items()}
    group.attrs["schema_version"] = "1.1.0"

    report = {
        "case_id": case_dir.name,
        "node_count": int(mesh.n_points),
        "directed_edge_count": int(edges.shape[1]),
        "feature_count": int(node_features.shape[1]),
        "feature_names": list(FEATURE_NAMES),
        "curvature_finite_fraction": float(curvature_finite.mean()),
        "centerline_distance_percentiles_mm": {
            str(q): float(np.percentile(centerline_distance, q)) for q in (1, 50, 95, 99, 100)
        },
        "branch_count": int(np.unique(branch_ids).size),
        "semantic_region_counts": {
            REGION_NAMES[int(region_id)]: int((semantic_region == region_id).sum())
            for region_id in np.unique(semantic_region)
        },
        "explicit_aneurysm_label": {
            "available": bool(aneurysm_available[0]),
            "node_count": int(explicit_aneurysm.sum()),
            "provenance": "nearest annotated Stanford centerline path whose name explicitly identifies aneurysm",
        },
        "patch_adjacency": "CSR surface connectivity",
        "centerline_tangent_provenance": "nearest annotated Stanford centerline path tangent",
        "multiscale_edges": {
            "coarse_directed_edge_count": int(coarse_edges.shape[1]),
            "hop_frontier": 4,
            "maximum_neighbors_per_node": 4,
        },
        "boundary_distance_provenance": {
            "method": "weighted surface-graph geodesic distance to topological open-boundary loops",
            "open_boundary_count": len(open_boundaries),
            "expected_from_boundary_conditions": expected_boundaries,
            "inlet_component_index": inlet_boundary,
            "inlet_cap_name": cap_names[inlet_cap_indices[0]],
            "boundary_cap_names": [cap_names[index] for index in cap_assignment],
            "boundary_to_cap_centroid_distance_mm_max": float(cap_distance.max()),
            "centerline_endpoint_assignment_counts": assignment_counts.tolist(),
            "centerline_endpoint_distance_diagnostic_mm_max": float(endpoint_distance.max()),
            "inlet_distance_mm_max": float(inlet_distance.max()),
            "nearest_outlet_distance_mm_max": float(outlet_distance.max()),
        },
        "surface_centroid_mm": surface_centroid.tolist(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    failures: dict[str, str] = {}
    reports: list[dict[str, object]] = []
    for case_dir in sorted(args.canonical_root.glob("*_H_ABAO_AAA")):
        try:
            report = build_case_features(case_dir, force=args.force)
            reports.append(report)
            print(f"PASS {case_dir.name}: {report['node_count']} nodes, {report['directed_edge_count']} edges", flush=True)
        except Exception as exc:
            failures[case_dir.name] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {case_dir.name}: {failures[case_dir.name]}", flush=True)
    summary = {"completed": len(reports), "failures": failures, "cases": reports}
    (args.canonical_root / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
