"""Audit Stanford VMR AAA project and time-resolved surface archives."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import BadZipFile, ZipFile


CASE_RE = re.compile(r"^(00(?:3[1-9]|4[0-5])_H_ABAO_AAA)$")
DATA_ARRAY_RE = re.compile(rb"<DataArray\b([^>]*)>", re.IGNORECASE)
ATTRIBUTE_RE = re.compile(rb"([A-Za-z][A-Za-z0-9_]*)=[\"']([^\"']*)[\"']")
PIECE_RE = re.compile(rb"<Piece\b([^>]*)>", re.IGNORECASE)


@dataclass(frozen=True)
class VtpHeader:
    member: str
    number_of_points: int | None
    number_of_polys: int | None
    point_arrays: tuple[str, ...]
    point_array_count: int
    wss_array_count: int
    velocity_array_count: int
    timestep_min: int | None
    timestep_max: int | None
    timestep_stride: int | None


@dataclass
class ArchiveAudit:
    case_id: str
    project_archive: str | None = None
    result_archive: str | None = None
    project_valid: bool = False
    result_valid: bool = False
    result_complete: bool = False
    project_members: int = 0
    result_members: int = 0
    result_vtp_files: int = 0
    result_headers: list[VtpHeader] = field(default_factory=list)
    has_wall_mesh: bool = False
    has_inflow_waveform: bool = False
    has_solver_settings: bool = False
    has_outlet_conditions: bool = False
    errors: list[str] = field(default_factory=list)


def _attributes(raw: bytes) -> dict[str, str]:
    return {
        key.decode("ascii", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in ATTRIBUTE_RE.findall(raw)
    }


def inspect_vtp_member(archive: ZipFile, member: str, header_bytes: int = 2 * 1024 * 1024) -> VtpHeader:
    """Read VTP XML metadata without loading appended mesh arrays."""

    with archive.open(member) as stream:
        prefix = stream.read(header_bytes)

    piece_match = PIECE_RE.search(prefix)
    piece = _attributes(piece_match.group(1)) if piece_match else {}
    arrays: list[str] = []
    for match in DATA_ARRAY_RE.finditer(prefix):
        attrs = _attributes(match.group(1))
        name = attrs.get("Name")
        if name and name not in arrays:
            arrays.append(name)

    def optional_int(value: str | None) -> int | None:
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    temporal_pattern = re.compile(r"^(vWSS|velocity)_(\d+)$")
    temporal: dict[str, list[int]] = {"vWSS": [], "velocity": []}
    static_arrays: list[str] = []
    for name in arrays:
        match = temporal_pattern.fullmatch(name)
        if match:
            temporal[match.group(1)].append(int(match.group(2)))
        else:
            static_arrays.append(name)
    timestep_ids = sorted(set(temporal["vWSS"]) | set(temporal["velocity"]))
    strides = [b - a for a, b in zip(timestep_ids, timestep_ids[1:])]
    timestep_stride = strides[0] if strides and len(set(strides)) == 1 else None

    return VtpHeader(
        member=member,
        number_of_points=optional_int(piece.get("NumberOfPoints")),
        number_of_polys=optional_int(piece.get("NumberOfPolys")),
        point_arrays=tuple(static_arrays),
        point_array_count=len(arrays),
        wss_array_count=len(temporal["vWSS"]),
        velocity_array_count=len(temporal["velocity"]),
        timestep_min=timestep_ids[0] if timestep_ids else None,
        timestep_max=timestep_ids[-1] if timestep_ids else None,
        timestep_stride=timestep_stride,
    )


def _validate_zip(path: Path, verify_crc: bool) -> tuple[bool, str | None]:
    try:
        with ZipFile(path) as archive:
            if verify_crc:
                bad_member = archive.testzip()
                if bad_member:
                    return False, f"CRC failure in {bad_member}"
        return True, None
    except (BadZipFile, OSError) as exc:
        return False, str(exc)


def _case_id_from_archive(path: Path) -> str | None:
    match = re.search(r"(00(?:3[1-9]|4[0-5])_H_ABAO_AAA)", path.name)
    return match.group(1) if match else None


def _project_signals(names: Iterable[str]) -> dict[str, bool]:
    lowered = [name.lower() for name in names]
    return {
        "has_wall_mesh": any(name.endswith("walls_combined.vtp") for name in lowered),
        "has_inflow_waveform": any("inflow" in name and name.endswith(".flow") for name in lowered),
        "has_solver_settings": any(name.endswith("solver.inp") or name.endswith(".sjb") for name in lowered),
        "has_outlet_conditions": any(name.endswith("rcrt.dat") for name in lowered),
    }


def audit_case(
    case_id: str,
    project_path: Path | None,
    result_path: Path | None,
    *,
    verify_crc: bool = False,
    header_sample_count: int = 3,
) -> ArchiveAudit:
    if not CASE_RE.fullmatch(case_id):
        raise ValueError(f"unsupported Stanford AAA case identifier: {case_id}")

    audit = ArchiveAudit(case_id=case_id)
    if project_path and project_path.exists():
        audit.project_archive = str(project_path)
        audit.project_valid, error = _validate_zip(project_path, verify_crc)
        if error:
            audit.errors.append(f"project: {error}")
        if audit.project_valid:
            with ZipFile(project_path) as archive:
                names = archive.namelist()
                audit.project_members = len(names)
                for key, value in _project_signals(names).items():
                    setattr(audit, key, value)
    else:
        audit.errors.append("project archive is missing")

    if result_path and result_path.exists() and result_path.suffix == ".zip":
        audit.result_archive = str(result_path)
        audit.result_valid, error = _validate_zip(result_path, verify_crc)
        if error:
            audit.errors.append(f"result: {error}")
        if audit.result_valid:
            with ZipFile(result_path) as archive:
                names = archive.namelist()
                vtp_names = sorted(name for name in names if PurePosixPath(name).suffix.lower() == ".vtp")
                audit.result_members = len(names)
                audit.result_vtp_files = len(vtp_names)
                sample_indices = sorted({0, len(vtp_names) // 2, len(vtp_names) - 1}) if vtp_names else []
                for index in sample_indices[:header_sample_count]:
                    audit.result_headers.append(inspect_vtp_member(archive, vtp_names[index]))
                audit.result_complete = bool(vtp_names)
    elif result_path and result_path.suffix == ".part":
        audit.result_archive = str(result_path)
        audit.errors.append("result archive download is incomplete")
    else:
        audit.errors.append("result archive is missing")

    return audit


def audit_dataset(root: Path, *, verify_crc: bool = False) -> list[ArchiveAudit]:
    projects_dir = root / "projects"
    results_dir = root / "surface_results"
    audits: list[ArchiveAudit] = []
    for number in range(31, 46):
        case_id = f"{number:04d}_H_ABAO_AAA"
        project_path = projects_dir / f"{case_id}.zip"
        result_path = results_dir / f"{case_id}_3D_RIGID_VTP.zip"
        if not result_path.exists():
            partial = Path(f"{result_path}.part")
            result_path = partial if partial.exists() else result_path
        audits.append(
            audit_case(case_id, project_path, result_path, verify_crc=verify_crc)
        )
    return audits


def summarize(audits: list[ArchiveAudit]) -> dict[str, int]:
    return {
        "cases": len(audits),
        "valid_projects": sum(a.project_valid for a in audits),
        "valid_results": sum(a.result_valid for a in audits),
        "complete_results": sum(a.result_complete for a in audits),
        "projects_with_wall_mesh": sum(a.has_wall_mesh for a in audits),
        "projects_with_inflow": sum(a.has_inflow_waveform for a in audits),
        "projects_with_solver_settings": sum(a.has_solver_settings for a in audits),
        "projects_with_outlet_conditions": sum(a.has_outlet_conditions for a in audits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Stanford VMR download root")
    parser.add_argument("--output", type=Path, help="Optional JSON audit report")
    parser.add_argument("--verify-crc", action="store_true", help="Read every compressed member and verify CRCs")
    args = parser.parse_args()

    audits = audit_dataset(args.root, verify_crc=args.verify_crc)
    report = {
        "summary": summarize(audits),
        "cases": [asdict(audit) for audit in audits],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
