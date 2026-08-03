"""Convert annotated SimVascular path files into a combined centerline VTP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class AnnotatedPath:
    name: str
    points_cm: np.ndarray
    tangents: np.ndarray


def parse_path_xml(name: str, payload: bytes) -> AnnotatedPath:
    # Legacy SimVascular .pth files contain sibling <format/> and <path/>
    # elements rather than a single XML document root.
    text = payload.decode("utf-8", errors="strict")
    text = text.lstrip("\ufeff")
    text = text.replace('<?xml version="1.0" encoding="UTF-8" ?>', "", 1)
    root = ElementTree.fromstring(f"<simvascular_path>{text}</simvascular_path>")
    path_points = root.findall(".//path_point")
    points: list[list[float]] = []
    tangents: list[list[float]] = []
    for path_point in path_points:
        position = path_point.find("pos")
        tangent = path_point.find("tangent")
        if position is None or tangent is None:
            continue
        points.append([float(position.attrib[axis]) for axis in "xyz"])
        tangents.append([float(tangent.attrib[axis]) for axis in "xyz"])
    if len(points) < 2:
        raise ValueError(f"path {name!r} contains fewer than two path points")
    return AnnotatedPath(
        name=name,
        points_cm=np.asarray(points, dtype=np.float64),
        tangents=np.asarray(tangents, dtype=np.float64),
    )


def read_annotated_paths(project_archive: Path) -> list[AnnotatedPath]:
    paths: list[AnnotatedPath] = []
    with ZipFile(project_archive) as archive:
        members = sorted(
            name for name in archive.namelist()
            if "/Paths/" in name and PurePosixPath(name).suffix.lower() == ".pth"
        )
        for member in members:
            paths.append(parse_path_xml(PurePosixPath(member).stem, archive.read(member)))
    if not paths:
        raise ValueError(f"no annotated path files found in {project_archive}")
    return paths


def write_centerlines(project_archive: Path, output_path: Path) -> dict[str, object]:
    import pyvista as pv

    paths = read_annotated_paths(project_archive)
    points: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    branch_ids: list[np.ndarray] = []
    lines: list[np.ndarray] = []
    branch_names: list[str] = []
    offset = 0
    for branch_id, path in enumerate(paths):
        count = len(path.points_cm)
        points.append(path.points_cm * 10.0)  # cm -> mm
        tangents.append(path.tangents)
        branch_ids.append(np.full(count, branch_id, dtype=np.int32))
        lines.append(np.concatenate(([count], np.arange(offset, offset + count, dtype=np.int64))))
        branch_names.append(path.name)
        offset += count

    polydata = pv.PolyData(
        np.concatenate(points, axis=0),
        lines=np.concatenate(lines),
    )
    polydata.point_data["branch_id"] = np.concatenate(branch_ids)
    polydata.point_data["tangent"] = np.concatenate(tangents, axis=0)
    polydata.cell_data["branch_id"] = np.arange(len(paths), dtype=np.int32)
    polydata.cell_data["branch_name"] = np.asarray(branch_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    polydata.save(output_path)
    return {
        "path_count": len(paths),
        "centerline_points": int(polydata.n_points),
        "path_names": branch_names,
        "coordinate_conversion": "SimVascular cm to canonical mm",
    }
