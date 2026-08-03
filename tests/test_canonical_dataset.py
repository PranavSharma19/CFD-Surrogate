import numpy as np

from aorta_surrogate.data.canonical_dataset import phase_conditioning


def test_phase_conditioning_preserves_flow_and_period():
    boundary = {
        "heart_period_seconds": 1.0,
        "time_seconds": [0.0, 0.5, 1.0],
        "inlet_flow_m3_s": [-1.0, -2.0, -1.0],
        "blood_density_kg_m3": 1060.0,
        "dynamic_viscosity_pa_s": 0.004,
    }
    conditioning = phase_conditioning(0.25, boundary)

    np.testing.assert_allclose(conditioning[:2], [1.0, 0.0], atol=1e-6)
    assert conditioning[2] == -1.5
    assert conditioning[3] == 2.0
    assert conditioning[4] == 1.0
