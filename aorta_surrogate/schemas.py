"""Versioned, JSON-serializable contracts for the aortic dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class BoundaryConditions:
    time_seconds: list[float]
    inlet_flow_m3_s: list[float]
    heart_period_seconds: float
    blood_density_kg_m3: float
    dynamic_viscosity_pa_s: float
    outlet_model: str
    outlet_parameters: dict[str, Any] = field(default_factory=dict)
    provenance: str = "source_dataset"

    def validate(self) -> None:
        if len(self.time_seconds) < 2:
            raise ValueError("at least two boundary-condition time samples are required")
        if len(self.time_seconds) != len(self.inlet_flow_m3_s):
            raise ValueError("inlet flow and time arrays must have equal length")
        if any(b <= a for a, b in zip(self.time_seconds, self.time_seconds[1:])):
            raise ValueError("boundary-condition times must be strictly increasing")
        if self.heart_period_seconds <= 0.0:
            raise ValueError("heart period must be positive")
        if self.blood_density_kg_m3 <= 0.0 or self.dynamic_viscosity_pa_s <= 0.0:
            raise ValueError("blood density and viscosity must be positive")

    def to_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class CaseManifest:
    dataset: str
    patient_id: str
    anatomy_id: str
    simulation_id: str
    intervention_state: Literal["preoperative", "post_evar", "candidate_evar"]
    coordinate_unit: str
    target_unit: str
    source_archive: str
    source_sha256: str | None = None
    license: str | None = None
    solver_provenance: dict[str, Any] = field(default_factory=dict)
    qc_status: Literal["pending", "pass", "fail"] = "pending"
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        required = {
            "dataset": self.dataset,
            "patient_id": self.patient_id,
            "anatomy_id": self.anatomy_id,
            "simulation_id": self.simulation_id,
            "coordinate_unit": self.coordinate_unit,
            "target_unit": self.target_unit,
            "source_archive": self.source_archive,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing required manifest fields: {', '.join(missing)}")
        if self.source_sha256 is not None:
            if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256.lower()):
                raise ValueError("source_sha256 must be a 64-character hexadecimal digest")

    def to_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
