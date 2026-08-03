from pathlib import Path
from zipfile import ZipFile

import pytest

from aorta_surrogate.data.stanford_project import (
    extract_boundary_conditions,
    parse_flow_waveform,
)


def test_extracts_and_converts_simvascular_boundary_conditions(tmp_path: Path):
    archive_path = tmp_path / "0031_H_ABAO_AAA.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("case/flow-files/inflow_3d.flow", "0.0 -20\n1.0 -20\n")
        archive.writestr(
            "case/Simulations/run/solver.inp",
            "Density: 1.06\nViscosity: 0.04\nNumber of RCR Surfaces: 1\nList of RCR Surfaces: 3\n",
        )
        archive.writestr(
            "case/Simulations/run/rcrt.dat",
            "2\n2\n1000\n0.0001\n15000\n0.0 0.0\n1.0 0.0\n",
        )

    conditions = extract_boundary_conditions(archive_path)

    assert conditions.blood_density_kg_m3 == pytest.approx(1060.0)
    assert conditions.dynamic_viscosity_pa_s == pytest.approx(0.004)
    assert conditions.inlet_flow_m3_s == pytest.approx([-2.0e-5, -2.0e-5])
    assert conditions.heart_period_seconds == pytest.approx(1.0)
    assert conditions.outlet_parameters["outlets"][0]["solver_surface_id"] == 3


def test_rejects_non_monotonic_waveform():
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_flow_waveform("0.0 1.0\n0.0 2.0\n")
