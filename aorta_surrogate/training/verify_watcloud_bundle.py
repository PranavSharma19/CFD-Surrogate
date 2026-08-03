"""Verify the upload data archive without extracting patient data to disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from aorta_surrogate.training.freeze_experiment import _tree_sha256


FREEZE_MEMBER = "canonical/experiments/watcloud_preop_v1/freeze_manifest.json"


def _member_sha256(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"archive member is not a regular file: {member.name}")
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify_data_bundle(bundle_path: Path) -> dict[str, object]:
    with tarfile.open(bundle_path, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if len(members) != len(archive.getmembers()):
            raise ValueError("archive contains duplicate member names")
        if FREEZE_MEMBER not in members:
            raise ValueError("archive is missing its freeze manifest")
        freeze_handle = archive.extractfile(members[FREEZE_MEMBER])
        if freeze_handle is None:
            raise ValueError("freeze manifest is not a regular file")
        freeze = json.load(freeze_handle)
        expected_members = {
            f"canonical/{row['relative_path']}" for row in freeze["runtime_files"]
        } | {FREEZE_MEMBER}
        actual_members = set(members)
        if actual_members != expected_members:
            missing = sorted(expected_members - actual_members)
            extra = sorted(actual_members - expected_members)
            raise ValueError(
                f"archive membership differs from frozen runtime; missing={missing}, extra={extra}"
            )
        locked_prefixes = tuple(
            f"canonical/{case_id}/" for case_id in freeze["locked_case_ids_excluded"]
        )
        present_locked = sorted(
            name for name in actual_members if name.startswith(locked_prefixes)
        )
        if present_locked:
            raise ValueError(f"archive contains locked patient files: {present_locked}")

        verified_rows: list[dict[str, object]] = []
        for row in freeze["runtime_files"]:
            member = members[f"canonical/{row['relative_path']}"]
            if member.size != int(row["size_bytes"]):
                raise ValueError(f"archive member size mismatch: {member.name}")
            digest = _member_sha256(archive, member)
            if digest != row["sha256"]:
                raise ValueError(f"archive member hash mismatch: {member.name}")
            verified_rows.append(
                {
                    "relative_path": row["relative_path"],
                    "size_bytes": member.size,
                    "sha256": digest,
                }
            )
        tree_hash = _tree_sha256(verified_rows)
        if tree_hash != freeze["runtime_tree_sha256"]:
            raise ValueError("archive runtime tree hash does not reproduce")
        return {
            "schema_version": "1.0.0",
            "status": "pass",
            "bundle": str(bundle_path),
            "development_case_count": len(freeze["development_case_ids"]),
            "locked_patient_files_present": [],
            "verified_file_count": len(verified_rows),
            "verified_bytes": sum(int(row["size_bytes"]) for row in verified_rows),
            "runtime_tree_sha256": tree_hash,
            "contract_sha256": freeze["contract_sha256"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-bundle", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_data_bundle(args.data_bundle), indent=2))


if __name__ == "__main__":
    main()
