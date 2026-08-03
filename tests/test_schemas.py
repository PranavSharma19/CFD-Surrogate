import pytest

from aorta_surrogate.schemas import BoundaryConditions, CaseManifest


def test_boundary_condition_length_mismatch_fails():
    bc = BoundaryConditions(
        time_seconds=[0.0, 1.0],
        inlet_flow_m3_s=[1.0e-5],
        heart_period_seconds=1.0,
        blood_density_kg_m3=1060.0,
        dynamic_viscosity_pa_s=0.004,
        outlet_model="RCR",
    )
    with pytest.raises(ValueError, match="equal length"):
        bc.validate()


def test_manifest_rejects_empty_patient_id():
    manifest = CaseManifest(
        dataset="stanford_vmr",
        patient_id="",
        anatomy_id="case",
        simulation_id="rigid",
        intervention_state="preoperative",
        coordinate_unit="mm",
        target_unit="Pa",
        source_archive="case.zip",
    )
    with pytest.raises(ValueError, match="patient_id"):
        manifest.validate()

