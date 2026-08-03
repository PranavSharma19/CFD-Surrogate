"""Lazy, framework-neutral interface for canonical aortic phase samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CONDITIONING_NAMES = (
    "sin_phase",
    "cos_phase",
    "flow_at_phase_m3_s",
    "peak_abs_flow_m3_s",
    "heart_period_seconds",
    "blood_density_kg_m3",
    "dynamic_viscosity_pa_s",
)


@dataclass(frozen=True)
class CanonicalPhaseSample:
    case_id: str
    phase_index: int
    node_features: np.ndarray
    edge_index: np.ndarray
    conditioning: np.ndarray
    target_wss_pa: np.ndarray


def phase_conditioning(
    phase_time_seconds: float,
    boundary_conditions: dict,
) -> np.ndarray:
    period = float(boundary_conditions["heart_period_seconds"])
    phase = phase_time_seconds / period
    waveform_times = np.asarray(boundary_conditions["time_seconds"], dtype=np.float64)
    waveform_flow = np.asarray(boundary_conditions["inlet_flow_m3_s"], dtype=np.float64)
    flow_at_phase = float(np.interp(phase_time_seconds, waveform_times, waveform_flow))
    return np.asarray(
        [
            np.sin(2.0 * np.pi * phase),
            np.cos(2.0 * np.pi * phase),
            flow_at_phase,
            float(np.max(np.abs(waveform_flow))),
            period,
            float(boundary_conditions["blood_density_kg_m3"]),
            float(boundary_conditions["dynamic_viscosity_pa_s"]),
        ],
        dtype=np.float32,
    )


class CanonicalAortaDataset:
    def __init__(self, canonical_root: Path, case_ids: list[str]):
        self.canonical_root = canonical_root
        self.case_ids = list(case_ids)

    @classmethod
    def from_split(cls, canonical_root: Path, split_name: str) -> "CanonicalAortaDataset":
        split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
        if split_name not in {"development", "locked_test"}:
            raise ValueError("split_name must be 'development' or 'locked_test'")
        return cls(canonical_root, split[split_name])

    def __len__(self) -> int:
        return len(self.case_ids) * 21

    def sample(self, case_id: str, phase_index: int) -> CanonicalPhaseSample:
        import zarr

        if case_id not in self.case_ids:
            raise KeyError(f"case {case_id} is not in this dataset split")
        if not 0 <= phase_index < 21:
            raise IndexError("phase_index must be between 0 and 20")
        case_dir = self.canonical_root / case_id
        targets = zarr.open_group(str(case_dir / "targets.zarr"), mode="r")
        features = zarr.open_group(str(case_dir / "features.zarr"), mode="r")
        boundary = json.loads((case_dir / "boundary_conditions.json").read_text(encoding="utf-8"))
        phase_time = float(targets["time_seconds"][phase_index])
        return CanonicalPhaseSample(
            case_id=case_id,
            phase_index=phase_index,
            node_features=np.asarray(features["node_features"]),
            edge_index=np.asarray(features["edge_index"]),
            conditioning=phase_conditioning(phase_time, boundary),
            target_wss_pa=np.asarray(targets["wss_pa"][phase_index]),
        )

