"""Derive a hardware-specific freeze manifest over an unchanged runtime tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from aorta_surrogate.training.freeze_experiment import _validate_contract
from aorta_surrogate.training.verify_staged_runtime import verify_staged_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_freeze_manifest(
    canonical_root: Path,
    base_manifest_path: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, object]:
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    split = json.loads((canonical_root / "patient_split.json").read_text(encoding="utf-8"))
    development, locked = _validate_contract(contract, split)
    verify_staged_runtime(canonical_root, base_manifest_path)
    if development != list(base["development_case_ids"]):
        raise ValueError("derived contract changes the frozen development cohort")
    if locked != list(base["locked_case_ids_excluded"]):
        raise ValueError("derived contract changes the locked cohort")
    if int(base["locked_case_file_count"]) != 0:
        raise ValueError("base manifest contains locked-patient files")

    derived = deepcopy(base)
    derived.update(
        {
            "experiment_id": contract["experiment_id"],
            "status": "hardware_contract_and_existing_development_inputs_frozen",
            "contract_path": contract_path.as_posix(),
            "contract_sha256": _sha256(contract_path),
            "derived_from": {
                "experiment_id": base["experiment_id"],
                "contract_sha256": base["contract_sha256"],
                "runtime_tree_sha256": base["runtime_tree_sha256"],
                "runtime_files_changed": False,
            },
            "next_state": (
                "AWS implementation must satisfy this hardware contract; the runtime "
                "tree remains byte-identical to the source freeze"
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(derived, indent=2), encoding="utf-8")
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = derive_freeze_manifest(
        args.canonical_root, args.base_manifest, args.contract, args.output
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "experiment_id",
                    "status",
                    "contract_sha256",
                    "runtime_tree_sha256",
                    "runtime_file_count",
                    "derived_from",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
