"""Compare interrupted/resumed and uninterrupted WATcloud smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_resume_equivalence(
    resumed_dir: Path,
    uninterrupted_dir: Path,
    *,
    parameter_tolerance: float = 1.0e-3,
    tawss_tolerance: float = 1.0e-4,
) -> dict[str, object]:
    resumed = torch.load(
        resumed_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    uninterrupted = torch.load(
        uninterrupted_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    resumed_history = _jsonl(resumed_dir / "training_history.jsonl")
    uninterrupted_history = _jsonl(uninterrupted_dir / "training_history.jsonl")
    resumed_steps = [int(row["step"]) for row in resumed_history]
    uninterrupted_steps = [int(row["step"]) for row in uninterrupted_history]
    sampled_regions_equal = [row["sampled_region_ids"] for row in resumed_history] == [
        row["sampled_region_ids"] for row in uninterrupted_history
    ]
    maximum_difference = 0.0
    maximum_name = None
    for name, resumed_tensor in resumed["model_state"].items():
        if name not in uninterrupted["model_state"]:
            raise ValueError(f"uninterrupted checkpoint is missing {name}")
        difference = float(
            (resumed_tensor - uninterrupted["model_state"][name]).abs().max()
        )
        if difference > maximum_difference:
            maximum_difference = difference
            maximum_name = name
    resumed_result = json.loads((resumed_dir / "result.json").read_text(encoding="utf-8"))
    uninterrupted_result = json.loads(
        (uninterrupted_dir / "result.json").read_text(encoding="utf-8")
    )
    tawss_difference = abs(
        float(resumed_result["best_validation_complete_cycle_tawss_relative_error"])
        - float(uninterrupted_result["best_validation_complete_cycle_tawss_relative_error"])
    )
    passed = (
        resumed_steps == uninterrupted_steps
        and sampled_regions_equal
        and maximum_difference <= parameter_tolerance
        and tawss_difference <= tawss_tolerance
    )
    report = {
        "schema_version": "1.0.0",
        "status": "pass" if passed else "fail",
        "comparison": "intentional interruption/resume versus uninterrupted twin",
        "step_sequence_equal": resumed_steps == uninterrupted_steps,
        "sampled_region_sequence_equal": sampled_regions_equal,
        "resumed_steps": resumed_steps,
        "maximum_model_parameter_absolute_difference": maximum_difference,
        "maximum_difference_parameter": maximum_name,
        "parameter_tolerance": parameter_tolerance,
        "complete_cycle_tawss_relative_error_absolute_difference": tawss_difference,
        "tawss_tolerance": tawss_tolerance,
        "interpretation": "Nonzero tolerance permits CUDA atomic reduction ordering; state, data sequence, and scientific result must remain equivalent.",
    }
    output = resumed_dir / "resume_equivalence_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        raise ValueError(f"resume equivalence failed; see {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumed-dir", type=Path, required=True)
    parser.add_argument("--uninterrupted-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_resume_equivalence(args.resumed_dir, args.uninterrupted_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
