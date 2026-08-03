"""Freeze a development-only experiment contract and its runtime data inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT_RUNTIME_FILES = (
    "patient_split.json",
    "quality_mask_manifest.json",
    "volume_mesh_qc_summary.json",
    "wss_label_audit_adjudication.json",
)
CASE_RUNTIME_ARTIFACTS = (
    "case_manifest.json",
    "boundary_conditions.json",
    "features.zarr",
    "targets.zarr",
    "quality_masks.zarr",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['sha256']}  {row['size_bytes']}  {row['relative_path']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _validate_contract(
    contract: dict[str, object], split: dict[str, object]
) -> tuple[list[str], list[str]]:
    data = contract["data"]
    development = list(data["development_case_ids"])
    locked = list(data["locked_case_ids_not_staged"])
    if development != list(split["development"]):
        raise ValueError("frozen development cases differ from patient_split.json")
    if locked != list(split["locked_test"]):
        raise ValueError("frozen locked cases differ from patient_split.json")
    if set(development) & set(locked):
        raise ValueError("development and locked cases overlap")
    folds = [list(fold) for fold in data["development_cv_folds"]]
    if folds != list(split["development_cv_folds"]):
        raise ValueError("frozen CV folds differ from patient_split.json")
    flattened = [case_id for fold in folds for case_id in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(development):
        raise ValueError("CV folds must partition the development cases exactly once")
    if data["runtime_case_artifacts"] != list(CASE_RUNTIME_ARTIFACTS):
        raise ValueError("contract runtime artifacts differ from the freeze implementation")
    if data["source_archives_required"] is not False:
        raise ValueError("WATcloud V1 must not require original source archives")
    return development, locked


def _collect_runtime_files(
    canonical_root: Path, development: list[str], locked: list[str]
) -> list[Path]:
    files: list[Path] = []
    for relative in ROOT_RUNTIME_FILES:
        path = canonical_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required root runtime file is missing: {path}")
        files.append(path)
    for case_id in development:
        case_dir = canonical_root / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"development case is missing: {case_dir}")
        for relative in CASE_RUNTIME_ARTIFACTS:
            path = case_dir / relative
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                artifact_files = sorted(row for row in path.rglob("*") if row.is_file())
                if not artifact_files:
                    raise FileNotFoundError(f"runtime artifact is empty: {path}")
                files.extend(artifact_files)
            else:
                raise FileNotFoundError(f"required runtime artifact is missing: {path}")

    locked_prefixes = tuple(f"{case_id}/" for case_id in locked)
    for path in files:
        relative = path.relative_to(canonical_root).as_posix()
        if relative.startswith(locked_prefixes):
            raise ValueError(f"locked patient entered runtime manifest: {relative}")
    return sorted(set(files), key=lambda path: path.relative_to(canonical_root).as_posix())


def freeze_experiment(
    contract_path: Path, canonical_root: Path, output_dir: Path
) -> dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    split_path = canonical_root / "patient_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development, locked = _validate_contract(contract, split)

    adjudication_path = canonical_root / "wss_label_audit_adjudication.json"
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if adjudication["locked_test_opened"] is not False:
        raise ValueError("label adjudication does not certify locked-test closure")
    if adjudication["cohort_decision"]["accepted_case_count"] != len(development):
        raise ValueError("not every development case is accepted in label adjudication")

    runtime_files = _collect_runtime_files(canonical_root, development, locked)
    rows = []
    total_bytes = 0
    for path in runtime_files:
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "relative_path": path.relative_to(canonical_root).as_posix(),
                "size_bytes": size,
                "sha256": _sha256(path),
            }
        )

    report = {
        "schema_version": "1.0.0",
        "experiment_id": contract["experiment_id"],
        "status": "experiment_contract_and_development_inputs_frozen",
        "research_only": True,
        "contract_path": contract_path.as_posix(),
        "contract_sha256": _sha256(contract_path),
        "patient_split_sha256": _sha256(split_path),
        "label_adjudication_sha256": _sha256(adjudication_path),
        "quality_mask_manifest_sha256": _sha256(
            canonical_root / "quality_mask_manifest.json"
        ),
        "development_case_ids": development,
        "locked_case_ids_excluded": locked,
        "locked_case_file_count": 0,
        "source_archives_included": False,
        "runtime_file_count": len(rows),
        "runtime_bytes": total_bytes,
        "runtime_gib": total_bytes / 1024**3,
        "runtime_tree_sha256": _tree_sha256(rows),
        "runtime_files": rows,
        "next_state": "implementation must satisfy the frozen contract; launch creates a separate resolved code/environment manifest",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "freeze_manifest.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_experiment(args.contract, args.canonical_root, args.output_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "experiment_id",
                    "status",
                    "contract_sha256",
                    "development_case_ids",
                    "locked_case_ids_excluded",
                    "runtime_file_count",
                    "runtime_gib",
                    "runtime_tree_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
