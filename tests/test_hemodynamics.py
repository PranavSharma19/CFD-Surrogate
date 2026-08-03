import numpy as np
import pytest

from aorta_surrogate.hemodynamics import compute_hemodynamic_metrics


def test_constant_wss_has_zero_osi_and_expected_rrt():
    wss = np.zeros((5, 2, 3), dtype=np.float64)
    wss[..., 0] = 2.0

    metrics = compute_hemodynamic_metrics(wss)

    np.testing.assert_allclose(metrics.tawss, [2.0, 2.0])
    np.testing.assert_allclose(metrics.osi, [0.0, 0.0])
    np.testing.assert_allclose(metrics.rrt, [0.5, 0.5])
    assert metrics.rrt_valid.all()


def test_balanced_reversal_has_maximum_osi_and_invalid_rrt():
    wss = np.zeros((3, 1, 3), dtype=np.float64)
    wss[:, 0, 0] = [1.0, -1.0, 1.0]

    metrics = compute_hemodynamic_metrics(wss, times=[0.0, 0.5, 1.0])

    np.testing.assert_allclose(metrics.tawss, [1.0])
    np.testing.assert_allclose(metrics.osi, [0.5])
    assert not metrics.rrt_valid[0]
    assert np.isnan(metrics.rrt[0])


def test_rejects_non_monotonic_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_hemodynamic_metrics(np.zeros((3, 1, 3)), times=[0.0, 0.5, 0.4])

