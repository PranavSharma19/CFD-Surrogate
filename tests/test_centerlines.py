import numpy as np

from aorta_surrogate.data.centerlines import parse_path_xml


def test_parse_path_xml():
    payload = b'''<path><path_element><path_points>
    <path_point><pos x="1" y="2" z="3"/><tangent x="0" y="0" z="1"/></path_point>
    <path_point><pos x="2" y="2" z="3"/><tangent x="1" y="0" z="0"/></path_point>
    </path_points></path_element></path>'''
    path = parse_path_xml("aorta", payload)

    assert path.name == "aorta"
    np.testing.assert_allclose(path.points_cm, [[1, 2, 3], [2, 2, 3]])
    np.testing.assert_allclose(path.tangents, [[0, 0, 1], [1, 0, 0]])
