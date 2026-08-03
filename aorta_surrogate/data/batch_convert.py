"""Resumable batch conversion for the 15 Stanford AAA feasibility cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aorta_surrogate.data.convert_stanford import convert_case


REQUIRED_ARTIFACTS = (
    "case_manifest.json",
    "surface.vtp",
    "centerlines.vtp",
    "boundary_conditions.json",
    "targets.zarr/zarr.json",
    "quality_report.json",
)


def case_complete(case_dir: Path) -> bool:
    return all((case_dir / relative).exists() for relative in REQUIRED_ARTIFACTS)


def batch_convert(root: Path, output_root: Path, *, force: bool = False) -> dict[str, object]:
    completed: list[str] = []
    skipped: list[str] = []
    failures: dict[str, str] = {}
    for number in range(31, 46):
        case_id = f"{number:04d}_H_ABAO_AAA"
        project = root / "projects" / f"{case_id}.zip"
        result = root / "surface_results" / f"{case_id}_3D_RIGID_VTP.zip"
        case_dir = output_root / case_id
        if case_complete(case_dir) and not force:
            skipped.append(case_id)
            print(f"SKIP {case_id}: canonical artifacts already exist", flush=True)
            continue
        try:
            print(f"CONVERT {case_id}", flush=True)
            convert_case(project, result, case_dir)
            completed.append(case_id)
            print(f"PASS {case_id}", flush=True)
        except Exception as exc:  # keep independent patients resumable
            failures[case_id] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {case_id}: {failures[case_id]}", flush=True)

    summary = {
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "total_complete": sum(case_complete(output_root / f"{number:04d}_H_ABAO_AAA") for number in range(31, 46)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(batch_convert(args.root, args.output, force=args.force), indent=2))


if __name__ == "__main__":
    main()

