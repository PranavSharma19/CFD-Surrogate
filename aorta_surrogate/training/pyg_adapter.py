"""Convert lazy canonical samples into bounded PyG training patches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aorta_surrogate.data.canonical_dataset import CanonicalAortaDataset


def connected_geodesic_nodes(
    indptr: np.ndarray,
    indices: np.ndarray,
    seed_node: int,
    max_nodes: int,
) -> np.ndarray:
    """Breadth-first surface patch, guaranteeing connectivity to the seed."""
    node_count = len(indptr) - 1
    if not 0 <= seed_node < node_count:
        raise IndexError("seed node is outside the surface graph")
    target = min(max_nodes, node_count)
    selected = np.zeros(node_count, dtype=bool)
    queue = np.empty(target, dtype=np.int64)
    queue[0] = seed_node
    selected[seed_node] = True
    head, tail = 0, 1
    while head < tail and tail < target:
        node = int(queue[head])
        head += 1
        for neighbor in indices[indptr[node] : indptr[node + 1]]:
            neighbor = int(neighbor)
            if selected[neighbor]:
                continue
            selected[neighbor] = True
            queue[tail] = neighbor
            tail += 1
            if tail == target:
                break
    if tail != target:
        raise ValueError(f"surface component contains only {tail} of {target} requested nodes")
    return np.sort(queue[:tail])


def load_target_valid_mask(
    case_dir: Path, node_count: int, policy: str = "primary"
) -> np.ndarray:
    """Load the registered primary or severe-sensitivity target-valid mask."""
    if policy not in {"primary", "severe_sensitivity"}:
        raise ValueError("target mask policy must be 'primary' or 'severe_sensitivity'")
    mask_path = case_dir / "quality_masks.zarr"
    if not mask_path.exists():
        if policy == "severe_sensitivity":
            raise FileNotFoundError(
                f"severe-sensitivity evaluation requires quality masks: {mask_path}"
            )
        return np.ones(node_count, dtype=bool)
    import zarr

    group = zarr.open_group(str(mask_path), mode="r")
    mask = np.asarray(group["target_valid"], dtype=bool)
    if mask.shape != (node_count,):
        raise ValueError(f"target-valid mask shape {mask.shape} does not match {node_count} nodes")
    if policy == "severe_sensitivity":
        if "volume_mesh_severe_sensitivity" not in group:
            raise KeyError(
                f"severe-sensitivity mask is missing from {mask_path}"
            )
        severe = np.asarray(group["volume_mesh_severe_sensitivity"], dtype=bool)
        if severe.shape != (node_count,):
            raise ValueError(
                f"severe-sensitivity mask shape {severe.shape} does not match {node_count} nodes"
            )
        mask &= ~severe
    return mask


def make_training_patch(
    canonical_root: Path,
    case_id: str,
    phase_index: int,
    *,
    max_nodes: int = 8192,
    seed: int = 17,
    normalization_path: Path | None = None,
    patch_method: str = "geodesic",
    target_mask_policy: str = "primary",
    seed_node: int | None = None,
):
    import torch
    from torch_geometric.data import Data

    stats_path = normalization_path or canonical_root / "normalization_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    if case_id not in split["development"]:
        raise ValueError("training patches may only be drawn from development patients")
    sample = CanonicalAortaDataset(canonical_root, split["development"]).sample(case_id, phase_index)
    features = sample.node_features.astype(np.float32, copy=True)
    node_count = len(features)
    target_valid = load_target_valid_mask(
        canonical_root / case_id, node_count, policy=target_mask_policy
    )

    if max_nodes < node_count:
        if seed_node is None:
            rng = np.random.default_rng(seed)
            selected_seed_node = int(rng.integers(node_count))
        else:
            selected_seed_node = int(seed_node)
            if not 0 <= selected_seed_node < node_count:
                raise IndexError("seed_node is outside the case surface")
        if patch_method == "geodesic":
            import zarr

            feature_group = zarr.open_group(str(canonical_root / case_id / "features.zarr"), mode="r")
            selected = connected_geodesic_nodes(
                np.asarray(feature_group["adjacency_indptr"]),
                np.asarray(feature_group["adjacency_indices"]),
                selected_seed_node,
                max_nodes,
            )
        elif patch_method == "euclidean":
            squared_distance = np.square(
                features[:, :3] - features[selected_seed_node, :3]
            ).sum(axis=1)
            selected = np.sort(np.argpartition(squared_distance, max_nodes)[:max_nodes])
        else:
            raise ValueError("patch_method must be 'geodesic' or 'euclidean'")
    else:
        selected = np.arange(node_count)

    remap = np.full(node_count, -1, dtype=np.int64)
    remap[selected] = np.arange(len(selected), dtype=np.int64)
    edge_index = sample.edge_index
    edge_mask = (remap[edge_index[0]] >= 0) & (remap[edge_index[1]] >= 0)
    local_edges = remap[edge_index[:, edge_mask]]
    if local_edges.shape[1] == 0:
        raise ValueError("selected patch contains no surface edges")

    node_mean = np.asarray(stats["node_mean"], dtype=np.float32)
    node_std = np.asarray(stats["node_std"], dtype=np.float32)
    normalized_features = (features[selected] - node_mean) / node_std
    source = normalized_features[local_edges[0], :3]
    destination = normalized_features[local_edges[1], :3]
    relative = destination - source
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    edge_attr = np.concatenate([distance, relative], axis=1).astype(np.float32)

    condition_mean = np.asarray(stats["conditioning_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["conditioning_std"], dtype=np.float32)
    conditioning = (sample.conditioning - condition_mean) / condition_std
    target_scale = float(stats["equivariant_target_scale_pa"])

    import zarr

    feature_group = zarr.open_group(str(canonical_root / case_id / "features.zarr"), mode="r")
    coarse_edges = np.asarray(feature_group["coarse_edge_index"])
    coarse_mask = (remap[coarse_edges[0]] >= 0) & (remap[coarse_edges[1]] >= 0)
    local_coarse_edges = remap[coarse_edges[:, coarse_mask]]
    semantic_region = np.asarray(feature_group["semantic_region_id"])[selected].astype(np.int64)
    region_one_hot = np.eye(7, dtype=np.float32)[semantic_region]
    scalar_continuous = normalized_features[:, 6:8]
    scalar_features = np.column_stack(
        [
            scalar_continuous,
            region_one_hot,
            features[selected, 9],
            np.asarray(feature_group["explicit_aneurysm_region"])[selected],
            np.asarray(feature_group["explicit_aneurysm_available"])[selected],
        ]
    ).astype(np.float32)
    position_scale = float(
        stats.get(
            "equivariant_position_scale_mm",
            np.sqrt(np.mean(np.square(node_std[:3]) + np.square(node_mean[:3]))),
        )
    )
    normals = features[selected, 3:6]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
    boundary_distances = np.column_stack(
        [
            np.asarray(feature_group["inlet_geodesic_distance_mm"])[selected],
            np.asarray(feature_group["outlet_geodesic_distance_mm"])[selected],
        ]
    ).astype(np.float32)
    boundary_mean = np.asarray(stats["boundary_distance_mean"], dtype=np.float32)
    boundary_std = np.asarray(stats["boundary_distance_std"], dtype=np.float32)
    scalar_features_multiscale = np.column_stack(
        [scalar_features, (boundary_distances - boundary_mean) / boundary_std]
    ).astype(np.float32)

    return Data(
        x=torch.from_numpy(normalized_features),
        scalar_x=torch.from_numpy(scalar_features),
        scalar_x_multiscale=torch.from_numpy(scalar_features_multiscale),
        pos=torch.from_numpy(features[selected, :3] / position_scale),
        normal=torch.from_numpy(normals.astype(np.float32)),
        tangent=torch.from_numpy(np.asarray(feature_group["centerline_tangent"])[selected].astype(np.float32)),
        edge_index=torch.from_numpy(local_edges),
        coarse_edge_index=torch.from_numpy(local_coarse_edges),
        edge_attr=torch.from_numpy(edge_attr),
        conditioning=torch.from_numpy(conditioning),
        y=torch.from_numpy(sample.target_wss_pa[selected] / target_scale),
        target_valid_mask=torch.from_numpy(target_valid[selected]),
        target_mask_policy=target_mask_policy,
        global_node_ids=torch.from_numpy(selected),
        semantic_region_id=torch.from_numpy(semantic_region),
        explicit_aneurysm_region=torch.from_numpy(np.asarray(feature_group["explicit_aneurysm_region"])[selected]),
        explicit_aneurysm_available=torch.from_numpy(np.asarray(feature_group["explicit_aneurysm_available"])[selected]),
        case_id=case_id,
        phase_index=phase_index,
        target_scale_pa=target_scale,
        patch_method=patch_method,
        seed_node=(selected_seed_node if max_nodes < node_count else None),
    )
