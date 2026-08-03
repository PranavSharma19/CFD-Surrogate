"""Select the largest registered WATcloud patch that passes the VRAM sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_patch(contract_path: Path, sweep_root: Path) -> dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    candidates = list(contract["patch_protocol"]["patch_node_candidates"])
    maximum_vram = 22.0
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        directory = sweep_root / f"patch_{candidate}"
        row: dict[str, object] = {
            "patch_nodes": candidate,
            "status": "fail",
            "reason": "missing completed artifacts",
        }
        result_path = directory / "result.json"
        runtime_path = directory / "runtime_and_vram.json"
        history_path = directory / "training_history.jsonl"
        if result_path.exists() and runtime_path.exists() and history_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            steps = [
                json.loads(line)["step"]
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            peak = runtime["peak_allocated_vram_gib"]
            passed = (
                result.get("status") in {"completed", "early_stopped"}
                and steps == [1, 2]
                and peak is not None
                and float(peak) <= maximum_vram
                and (directory / "best_checkpoint.pt").exists()
                and (directory / "latest_checkpoint.pt").exists()
            )
            row = {
                "patch_nodes": candidate,
                "status": "pass" if passed else "fail",
                "reason": (
                    "completed interrupted/resumed smoke within VRAM limit"
                    if passed
                    else "completed artifacts failed one or more registered gates"
                ),
                "steps": steps,
                "peak_allocated_vram_gib": peak,
                "complete_cycle_tawss_relative_error": result.get(
                    "best_validation_complete_cycle_tawss_relative_error"
                ),
            }
        failure_path = directory / "failure.json"
        if failure_path.exists() and row["status"] != "pass":
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            row["reason"] = failure.get("reason", row["reason"])
        rows.append(row)
    passing = [int(row["patch_nodes"]) for row in rows if row["status"] == "pass"]
    if not passing:
        raise ValueError("no registered patch candidate passed the WATcloud sweep")
    selected = max(passing)
    report = {
        "schema_version": "1.0.0",
        "experiment_id": contract["experiment_id"],
        "status": "patch_selected",
        "selection_rule": contract["patch_protocol"]["patch_selection_rule"],
        "maximum_vram_gib": maximum_vram,
        "selected_patch_nodes": selected,
        "candidates": rows,
    }
    output = sweep_root / "patch_selection.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select_patch(args.contract, args.sweep_root), indent=2))


if __name__ == "__main__":
    main()
