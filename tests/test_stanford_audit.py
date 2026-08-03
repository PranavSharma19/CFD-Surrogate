from pathlib import Path
from zipfile import ZipFile

from aorta_surrogate.data.stanford import audit_case


VTP = b'''<?xml version="1.0"?>
<VTKFile type="PolyData"><PolyData><Piece NumberOfPoints="4" NumberOfPolys="2">
<PointData><DataArray Name="wall_shear" NumberOfComponents="3"></DataArray></PointData>
</Piece></PolyData></VTKFile>'''


def test_audits_project_and_surface_result(tmp_path: Path):
    case_id = "0031_H_ABAO_AAA"
    project = tmp_path / f"{case_id}.zip"
    result = tmp_path / f"{case_id}_3D_RIGID_VTP.zip"

    with ZipFile(project, "w") as archive:
        archive.writestr(f"{case_id}/flow-files/inflow_3d.flow", "0 1")
        archive.writestr(f"{case_id}/Simulations/run/mesh-complete/walls_combined.vtp", VTP)
        archive.writestr(f"{case_id}/Simulations/run/solver.inp", "")
        archive.writestr(f"{case_id}/Simulations/run/rcrt.dat", "")
    with ZipFile(result, "w") as archive:
        archive.writestr("result_000.vtp", VTP)
        archive.writestr("result_001.vtp", VTP)

    audit = audit_case(case_id, project, result, verify_crc=True)

    assert audit.project_valid
    assert audit.result_valid
    assert audit.result_complete
    assert audit.has_wall_mesh
    assert audit.has_inflow_waveform
    assert audit.has_solver_settings
    assert audit.has_outlet_conditions
    assert audit.result_vtp_files == 2
    assert audit.result_headers[0].point_arrays == ("wall_shear",)
    assert audit.result_headers[0].point_array_count == 1
    assert audit.result_headers[0].wss_array_count == 0
