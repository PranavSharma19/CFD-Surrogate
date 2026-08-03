import numpy as np

from aorta_surrogate.data.wss_label_audit import (
    endpoint_relative_l2,
    endpoint_pointwise_report,
    peak_temporal_ratio,
    point_mesh_quality,
    spatial_peak_coherence,
    triangle_quality,
    vector_tangentiality,
)


def test_vector_tangentiality_separates_tangent_and_normal_components():
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    wss = np.asarray([[[2.0, 0.0, 0.0], [0.0, 3.0, 4.0]]])
    ratio = vector_tangentiality(wss, normals)
    np.testing.assert_allclose(ratio, [[0.0, 0.6]])


def test_endpoint_relative_l2_is_zero_for_periodic_cycle():
    wss = np.asarray([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])
    assert endpoint_relative_l2(wss) == 0.0
    report = endpoint_pointwise_report(wss)
    assert report["absolute_difference_pa_percentiles"]["100"] == 0.0


def test_peak_temporal_ratio_uses_cyclic_neighbors_without_duplicate_endpoint():
    magnitude = np.asarray([[1.0], [2.0], [10.0], [4.0], [1.0]])
    report = peak_temporal_ratio(magnitude)
    assert report["phase_index"] == 2
    assert report["adjacent_phase_values_pa"] == [2.0, 4.0]
    assert report["peak_to_adjacent_mean_ratio"] == 10.0 / 3.0


def test_spatial_peak_coherence_compares_peak_with_surface_neighbors():
    magnitude = np.asarray([[1.0, 2.0, 1.0], [3.0, 10.0, 4.0], [1.0, 2.0, 1.0]])
    edges = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]])
    report = spatial_peak_coherence(magnitude, edges)
    assert report["node_index"] == 1
    assert report["maximum_neighbor_wss_pa"] == 4.0
    assert report["peak_to_maximum_neighbor_ratio"] == 2.5


def test_triangle_quality_and_point_mapping_detect_skinny_triangle():
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, 0.01, 0.0]])
    triangles = np.asarray([[0, 1, 2], [0, 1, 3]])
    area, edge_ratio = triangle_quality(points, triangles)
    assert area[0] == 0.5
    assert edge_ratio[1] > edge_ratio[0]
    minimum_area, maximum_ratio = point_mesh_quality(4, triangles, area, edge_ratio)
    assert minimum_area[2] == 0.5
    assert maximum_ratio[3] == edge_ratio[1]
