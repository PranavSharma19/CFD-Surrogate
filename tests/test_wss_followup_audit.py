import numpy as np

from aorta_surrogate.data.wss_followup_audit import point_incident_quality


def test_point_incident_quality_maps_worst_tetrahedra_to_points():
    tetrahedra = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]])
    jacobian = np.asarray([0.5, 0.01])
    aspect = np.asarray([2.0, 100.0])
    volume = np.asarray([1.0, 0.1])
    point_jacobian, point_aspect, point_volume = point_incident_quality(
        5, tetrahedra, jacobian, aspect, volume
    )
    assert point_jacobian[0] == 0.5
    assert point_jacobian[1] == 0.01
    assert point_aspect[1] == 100.0
    assert point_volume[4] == 0.1
