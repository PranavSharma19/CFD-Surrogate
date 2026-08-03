"""Extract canonical boundary conditions from Stanford SimVascular projects."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZipFile

from aorta_surrogate.schemas import BoundaryConditions


KEY_VALUE_RE = re.compile(r"^([^:#]+):\s*(.+?)\s*$")


def _unique_member(archive: ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        raise ValueError(f"expected one '*{suffix}' member, found {len(matches)}")
    return matches[0]


def _read_text(archive: ZipFile, member: str) -> str:
    return archive.read(member).decode("utf-8", errors="strict")


def parse_solver_settings(text: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw_line in text.splitlines():
        match = KEY_VALUE_RE.match(raw_line.strip())
        if match:
            settings[match.group(1).strip()] = match.group(2).strip()
    return settings


def parse_flow_waveform(text: str) -> tuple[list[float], list[float]]:
    times: list[float] = []
    flow_cgs: list[float] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 2:
            raise ValueError(f"invalid inflow row {line_number}: expected two columns")
        times.append(float(columns[0]))
        flow_cgs.append(float(columns[1]))
    if len(times) < 2:
        raise ValueError("inflow waveform contains fewer than two samples")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("inflow waveform times are not strictly increasing")
    return times, flow_cgs


def parse_rcr(text: str, expected_surfaces: int) -> list[dict[str, object]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("rcrt.dat is empty")

    index = 1  # The first line is the file format/version used by SimVascular.
    outlets: list[dict[str, object]] = []
    while index < len(lines):
        pressure_samples = int(lines[index])
        index += 1
        if pressure_samples < 1 or index + 3 + pressure_samples > len(lines):
            raise ValueError("malformed RCR outlet block")
        proximal_resistance = float(lines[index])
        capacitance = float(lines[index + 1])
        distal_resistance = float(lines[index + 2])
        index += 3
        distal_pressure = []
        for _ in range(pressure_samples):
            time_value, pressure_value = (float(value) for value in lines[index].split())
            distal_pressure.append([time_value, pressure_value])
            index += 1
        outlets.append(
            {
                "proximal_resistance_cgs": proximal_resistance,
                "capacitance_cgs": capacitance,
                "distal_resistance_cgs": distal_resistance,
                "distal_pressure_cgs": distal_pressure,
            }
        )

    if len(outlets) != expected_surfaces:
        raise ValueError(f"expected {expected_surfaces} RCR blocks, found {len(outlets)}")
    return outlets


def extract_boundary_conditions(project_archive: Path) -> BoundaryConditions:
    """Convert a Stanford SimVascular project's CGS inputs to the SI contract."""

    with ZipFile(project_archive) as archive:
        flow_member = _unique_member(archive, "flow-files/inflow_3d.flow")
        solver_member = _unique_member(archive, "solver.inp")
        rcr_member = _unique_member(archive, "rcrt.dat")
        times, flow_cgs = parse_flow_waveform(_read_text(archive, flow_member))
        solver = parse_solver_settings(_read_text(archive, solver_member))

        required = ["Density", "Viscosity", "Number of RCR Surfaces", "List of RCR Surfaces"]
        missing = [name for name in required if name not in solver]
        if missing:
            raise ValueError(f"solver settings missing: {', '.join(missing)}")

        rcr_count = int(solver["Number of RCR Surfaces"])
        surface_ids = [int(value) for value in solver["List of RCR Surfaces"].split()]
        if len(surface_ids) != rcr_count:
            raise ValueError("RCR surface ID count does not match Number of RCR Surfaces")
        outlets = parse_rcr(_read_text(archive, rcr_member), rcr_count)
        for surface_id, outlet in zip(surface_ids, outlets):
            outlet["solver_surface_id"] = surface_id

    # SimVascular cardiovascular projects use CGS solver units: density in
    # g/cm^3, viscosity in g/(cm*s), and volumetric flow in cm^3/s.
    density_kg_m3 = float(solver["Density"]) * 1000.0
    viscosity_pa_s = float(solver["Viscosity"]) * 0.1
    flow_m3_s = [value * 1.0e-6 for value in flow_cgs]
    period_seconds = times[-1] - times[0]

    boundary_conditions = BoundaryConditions(
        time_seconds=times,
        inlet_flow_m3_s=flow_m3_s,
        heart_period_seconds=period_seconds,
        blood_density_kg_m3=density_kg_m3,
        dynamic_viscosity_pa_s=viscosity_pa_s,
        outlet_model="RCR",
        outlet_parameters={
            "outlets": outlets,
            "source_units": "SimVascular CGS",
            "flow_sign_preserved": True,
            "flow_member": flow_member,
            "solver_member": solver_member,
            "rcr_member": rcr_member,
        },
        provenance="stanford_vmr_simvascular_cgs_converted_to_si",
    )
    boundary_conditions.validate()
    return boundary_conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract_boundary_conditions(args.project_archive).to_json(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

