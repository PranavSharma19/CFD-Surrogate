"""Convert a Stanford VMR AAA case into the canonical surface data contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from aorta_surrogate.data.stanford_project import (
    extract_boundary_conditions,
    parse_solver_settings,
)
from aorta_surrogate.data.centerlines import write_centerlines
from aorta_surrogate.hemodynamics import compute_hemodynamic_metrics
from aorta_surrogate.schemas import CaseManifest


WSS_RE = re.compile(r"^vWSS_(\d+)$")


def select_evenly_spaced_timesteps(available: list[int], count: int = 21) -> list[int]:
    """Select registered phases including both endpoints of a complete cycle."""

    ordered = sorted(set(available))
    if count < 2:
        raise ValueError("at least two selected phases are required")
    if len(ordered) < count:
        raise ValueError(f"requested {count} phases from only {len(ordered)} available")
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    selected = [ordered[index] for index in indices]
    if len(set(selected)) != count:
        raise ValueError("phase selection produced duplicate timesteps")
    return selected


def _unique_member(archive: ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        raise ValueError(f"expected one '*{suffix}' member, found {len(matches)}")
    return matches[0]


def _read_polydata(path: Path, enabled_point_arrays: list[str] | None = None):
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("VTK is required; install the project's 'io' dependency group") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    point_selection = reader.GetPointDataArraySelection()
    cell_selection = reader.GetCellDataArraySelection()
    point_selection.DisableAllArrays()
    cell_selection.DisableAllArrays()
    for name in enabled_point_arrays or []:
        point_selection.EnableArray(name)
    reader.Update()
    output = reader.GetOutput()
    if output.GetNumberOfPoints() == 0:
        raise ValueError(f"VTK returned an empty surface for {path}")
    return output


def _available_point_arrays(path: Path) -> list[str]:
    try:
        import vtk
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("VTK is required; install the project's 'io' dependency group") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    selection = reader.GetPointDataArraySelection()
    return [selection.GetArrayName(index) for index in range(selection.GetNumberOfArrays())]


def _wall_to_result_mapping(wall, result, tolerance_cm: float) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(result)
    locator.BuildLocator()
    wall_points = vtk_to_numpy(wall.GetPoints().GetData())
    result_points = vtk_to_numpy(result.GetPoints().GetData())
    indices = np.fromiter(
        (locator.FindClosestPoint(point) for point in wall_points),
        dtype=np.int64,
        count=len(wall_points),
    )
    distances = np.linalg.norm(result_points[indices] - wall_points, axis=1)
    if np.unique(indices).size != indices.size:
        raise ValueError("wall-to-result mapping is not one-to-one")
    if float(distances.max()) > tolerance_cm:
        raise ValueError(
            f"wall-to-result mapping exceeded {tolerance_cm:g} cm: {float(distances.max()):g} cm"
        )
    return indices, distances


def convert_case(
    project_archive: Path,
    result_archive: Path,
    output_dir: Path,
    *,
    phase_count: int = 21,
    mapping_tolerance_cm: float = 1.0e-3,
) -> dict[str, object]:
    """Convert geometry, 21 WSS phases, metrics, BCs, and QC for one patient."""

    import pyvista as pv
    import zarr
    from vtk.util.numpy_support import vtk_to_numpy

    case_id = project_archive.stem
    if not result_archive.name.startswith(case_id):
        raise ValueError("project and result archives do not refer to the same case")
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f"{case_id}_", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        with ZipFile(project_archive) as project_zip:
            wall_member = _unique_member(project_zip, "walls_combined.vtp")
            solver_member = _unique_member(project_zip, "solver.inp")
            wall_path = Path(project_zip.extract(wall_member, temporary_dir))
            solver = parse_solver_settings(project_zip.read(solver_member).decode("utf-8"))
        with ZipFile(result_archive) as result_zip:
            result_member = _unique_member(result_zip, ".vtp")
            result_path = Path(result_zip.extract(result_member, temporary_dir))

        array_names = _available_point_arrays(result_path)
        available_timesteps = [int(match.group(1)) for name in array_names if (match := WSS_RE.fullmatch(name))]
        selected_timesteps = select_evenly_spaced_timesteps(available_timesteps, phase_count)
        selected_arrays = [f"vWSS_{timestep}" for timestep in selected_timesteps]

        wall = _read_polydata(wall_path)
        result = _read_polydata(result_path, selected_arrays)
        mapping, distances = _wall_to_result_mapping(wall, result, mapping_tolerance_cm)

        wss_cgs = np.stack(
            [vtk_to_numpy(result.GetPointData().GetArray(name))[mapping] for name in selected_arrays],
            axis=0,
        ).astype(np.float32, copy=False)
        # SimVascular traction/WSS is dyn/cm^2; 1 dyn/cm^2 = 0.1 Pa.
        wss_pa = wss_cgs * np.float32(0.1)

        if "Time Step Size" not in solver:
            raise ValueError("solver.inp does not contain Time Step Size")
        solver_dt = float(solver["Time Step Size"])
        time_seconds = (np.asarray(selected_timesteps, dtype=np.float64) - selected_timesteps[0]) * solver_dt
        metrics = compute_hemodynamic_metrics(wss_pa, time_seconds)

        wall_mesh = pv.wrap(wall).copy(deep=True)
        wall_mesh.points = np.asarray(wall_mesh.points) * 10.0  # cm -> mm
        wall_mesh.save(output_dir / "surface.vtp")

    target_group = zarr.open_group(str(output_dir / "targets.zarr"), mode="w")
    target_group.create_array(
        "wss_pa",
        data=wss_pa,
        chunks=(1, min(wss_pa.shape[1], 32768), 3),
    )
    target_group.create_array("time_seconds", data=time_seconds)
    target_group.create_array("tawss_pa", data=metrics.tawss.astype(np.float32))
    target_group.create_array("osi", data=metrics.osi.astype(np.float32))
    target_group.create_array("rrt_pa_inv", data=metrics.rrt.astype(np.float32))
    target_group.create_array("rrt_valid", data=metrics.rrt_valid)
    target_group.attrs.update(
        {
            "schema_version": "1.0.0",
            "source_wss_unit": "dyn/cm^2",
            "target_wss_unit": "Pa",
            "phase_endpoint_policy": "both cycle endpoints retained",
        }
    )

    boundary_conditions = extract_boundary_conditions(project_archive)
    boundary_conditions.to_json(output_dir / "boundary_conditions.json")
    centerline_report = write_centerlines(project_archive, output_dir / "centerlines.vtp")

    quality = {
        "status": "pass",
        "warnings": [],
        "source_result_points": int(result.GetNumberOfPoints()),
        "canonical_wall_points": int(wss_pa.shape[1]),
        "selected_phases": phase_count,
        "available_wss_phases": len(available_timesteps),
        "selected_timestep_ids": selected_timesteps,
        "max_mapping_distance_cm": float(distances.max()),
        "p99_mapping_distance_cm": float(np.percentile(distances, 99)),
        "mapping_is_one_to_one": True,
        "finite_wss": bool(np.isfinite(wss_pa).all()),
        "wss_magnitude_min_pa": float(np.linalg.norm(wss_pa, axis=-1).min()),
        "wss_magnitude_max_pa": float(np.linalg.norm(wss_pa, axis=-1).max()),
        "wss_magnitude_percentiles_pa": {
            str(percentile): float(np.percentile(np.linalg.norm(wss_pa, axis=-1), percentile))
            for percentile in (1, 50, 95, 99, 99.9)
        },
        "tawss_percentiles_pa": {
            str(percentile): float(np.percentile(metrics.tawss, percentile))
            for percentile in (1, 50, 95, 99, 99.9)
        },
        "coordinate_conversion": "SimVascular cm to canonical mm",
        "wss_conversion": "SimVascular dyn/cm^2 to Pa",
        "centerlines": centerline_report,
    }
    (output_dir / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    manifest = CaseManifest(
        dataset="stanford_vmr",
        patient_id=case_id,
        anatomy_id=case_id,
        simulation_id="3D_RIGID_VTP",
        intervention_state="preoperative",
        coordinate_unit="mm",
        target_unit="Pa",
        source_archive=result_archive.name,
        solver_provenance={
            "method": "rigid_wall",
            "source_units": "SimVascular CGS",
            "project_archive": project_archive.name,
        },
        qc_status="pass",
    )
    manifest.to_json(output_dir / "case_manifest.json")
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", type=int, default=21)
    args = parser.parse_args()
    quality = convert_case(args.project, args.result, args.output, phase_count=args.phases)
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
