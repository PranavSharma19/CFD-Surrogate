"""Verify a staged WATcloud canonical tree against its frozen runtime manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aorta_surrogate.training.freeze_experiment import _tree_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_staged_runtime(
    canonical_root: Path,
    freeze_manifest_path: Path,
    *,
    allow_locked_source_directories: bool = False,
) -> dict[str, object]:
    manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    development = list(manifest["development_case_ids"])
    locked = list(manifest["locked_case_ids_excluded"])
    case_directories = sorted(
        path.name
        for path in canonical_root.iterdir()
        if path.is_dir() and (path / "case_manifest.json").is_file()
    )
    expected_directories = (
        sorted(development + locked)
        if allow_locked_source_directories
        else development
    )
    if case_directories != expected_directories:
        raise ValueError(
            "case directories differ from the permitted frozen set: "
            f"expected {expected_directories}, got {case_directories}"
        )
    present_locked = [case_id for case_id in locked if (canonical_root / case_id).exists()]
    if present_locked and not allow_locked_source_directories:
        raise ValueError(f"locked case directories are staged: {present_locked}")

    verified_rows: list[dict[str, object]] = []
    total_bytes = 0
    for expected in manifest["runtime_files"]:
        relative = str(expected["relative_path"])
        path = canonical_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"staged runtime file is missing: {relative}")
        size = path.stat().st_size
        if size != int(expected["size_bytes"]):
            raise ValueError(f"staged runtime size mismatch: {relative}")
        digest = _sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"staged runtime hash mismatch: {relative}")
        total_bytes += size
        verified_rows.append(
            {"relative_path": relative, "size_bytes": size, "sha256": digest}
        )
    tree_hash = _tree_sha256(verified_rows)
    if tree_hash != manifest["runtime_tree_sha256"]:
        raise ValueError("staged runtime tree hash does not reproduce")
    if len(verified_rows) != int(manifest["runtime_file_count"]):
        raise ValueError("staged runtime file count does not reproduce")
    return {
        "schema_version": "1.0.0",
        "status": "pass",
        "development_case_count": len(development),
        "verification_mode": (
            "local_source_before_selective_packaging"
            if allow_locked_source_directories
            else "exclusive_cloud_stage"
        ),
        "locked_case_directories_present": present_locked,
        "verified_file_count": len(verified_rows),
        "verified_bytes": total_bytes,
        "runtime_tree_sha256": tree_hash,
        "contract_sha256": manifest["contract_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_staged_runtime(args.canonical_root, args.freeze_manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
