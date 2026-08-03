import numpy as np

from aorta_surrogate.data.normalization import RunningMoments


def test_running_moments_matches_numpy():
    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    moments = RunningMoments(3)
    moments.update(values[:4])
    moments.update(values[4:])
    mean, std = moments.result()

    np.testing.assert_allclose(mean, values.mean(axis=0))
    np.testing.assert_allclose(std, values.std(axis=0))
