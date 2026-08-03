import numpy as np

from aorta_surrogate.data.outlier_diagnostics import active_components, boundary_vertices
from aorta_surrogate.data.features import (
    boundary_components,
    coarse_geodesic_edges,
    edge_index_to_csr,
    weighted_geodesic_distance,
)
from aorta_surrogate.training.pyg_adapter import connected_geodesic_nodes


def test_boundary_vertices_finds_open_square_boundary():
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]])
    np.testing.assert_array_equal(boundary_vertices(triangles, 4), np.ones(4, dtype=bool))


def test_active_components_respects_mask():
    edge_index = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]])
    active = np.asarray([True, True, False, True, True])
    assert active_components(edge_index, active) == [2, 2]


def test_geodesic_patch_is_seed_connected_and_bounded():
    edge_index = np.asarray(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=np.int64
    )
    indptr, indices = edge_index_to_csr(edge_index, 5)
    np.testing.assert_array_equal(connected_geodesic_nodes(indptr, indices, 2, 3), [1, 2, 3])


def test_weighted_distance_and_coarse_frontier_on_chain():
    points = np.column_stack([np.arange(6), np.zeros(6), np.zeros(6)]).astype(float)
    edge_index = np.asarray(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=np.int64
    )
    indptr, indices = edge_index_to_csr(edge_index, 6)
    np.testing.assert_allclose(weighted_geodesic_distance(points, indptr, indices, np.asarray([0])), np.arange(6))
    coarse = coarse_geodesic_edges(points, indptr, indices, hops=2, neighbors_per_node=2)
    assert (coarse[:, coarse[0] == 0] == np.asarray([[0], [2]])).all()


def test_boundary_components_separates_loops():
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]])
    components = boundary_components(triangles)
    assert {tuple(component) for component in components} == {(0, 1, 2), (3, 4, 5)}
