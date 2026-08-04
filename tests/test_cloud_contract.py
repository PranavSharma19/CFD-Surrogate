from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aorta_surrogate.training.cloud_contract import (
    registered_gpu_names,
    registered_vram_limit_gib,
    validate_registered_gpu,
)
from aorta_surrogate.training.derive_freeze_manifest import derive_freeze_manifest
from aorta_surrogate.training.freeze_experiment import _tree_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_gpu_validation_is_contract_driven() -> None:
    aws = {
        "experiment_id": "aws-test",
        "hardware": {
            "target_gpu": "NVIDIA L40S",
            "accepted_gpu_name_substrings": ["L40S"],
            "maximum_allocated_vram_gib": 22.0,
        },
    }
    assert registered_gpu_names(aws) == ("L40S",)
    validate_registered_gpu(aws, "NVIDIA L40S")
    assert registered_vram_limit_gib(aws) == 22.0
    with pytest.raises(RuntimeError, match="requires a registered GPU"):
        validate_registered_gpu(aws, "NVIDIA A10G")


def test_legacy_watcloud_contract_defaults_to_its_target_and_22_gib() -> None:
    contract = json.loads(
        (REPOSITORY_ROOT / "configs/watcloud_preop_v1_frozen.json").read_text()
    )
    assert registered_gpu_names(contract) == ("NVIDIA GeForce RTX 4090",)
    assert registered_vram_limit_gib(contract) == 22.0


def test_aws_contract_changes_execution_only() -> None:
    watcloud = json.loads(
        (REPOSITORY_ROOT / "configs/watcloud_preop_v1_frozen.json").read_text()
    )
    aws = json.loads((REPOSITORY_ROOT / "configs/aws_preop_v1_frozen.json").read_text())
    for key in ("model", "loss", "optimization", "evaluation", "patch_protocol"):
        assert aws[key] == watcloud[key]
    for key in (
        "development_case_ids",
        "locked_case_ids_not_staged",
        "development_cv_folds",
        "phases_per_cycle",
        "runtime_case_artifacts",
        "normalization",
        "primary_target_mask",
        "sensitivity_target_mask",
        "canonical_targets_modified",
        "source_archives_required",
    ):
        assert aws["data"][key] == watcloud["data"][key]
    assert aws["experiment_id"] != watcloud["experiment_id"]
    assert aws["hardware"]["target_gpu"] == "NVIDIA L40S"


def test_derive_manifest_preserves_runtime_tree(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    case_dir = canonical / "case_a"
    case_dir.mkdir(parents=True)
    runtime_file = case_dir / "case_manifest.json"
    runtime_file.write_text("{}", encoding="utf-8")
    split = {
        "development": ["case_a"],
        "locked_test": ["case_locked"],
        "development_cv_folds": [["case_a"]],
    }
    (canonical / "patient_split.json").write_text(json.dumps(split), encoding="utf-8")
    row = {
        "relative_path": "case_a/case_manifest.json",
        "size_bytes": runtime_file.stat().st_size,
        "sha256": _sha256(runtime_file),
    }
    base = {
        "schema_version": "1.0.0",
        "experiment_id": "base",
        "status": "frozen",
        "contract_sha256": "base-contract",
        "development_case_ids": ["case_a"],
        "locked_case_ids_excluded": ["case_locked"],
        "locked_case_file_count": 0,
        "runtime_files": [row],
        "runtime_file_count": 1,
        "runtime_bytes": runtime_file.stat().st_size,
        "runtime_tree_sha256": _tree_sha256([row]),
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    contract = {
        "experiment_id": "derived",
        "data": {
            "development_case_ids": ["case_a"],
            "locked_case_ids_not_staged": ["case_locked"],
            "development_cv_folds": [["case_a"]],
            "runtime_case_artifacts": [
                "case_manifest.json",
                "boundary_conditions.json",
                "features.zarr",
                "targets.zarr",
                "quality_masks.zarr",
            ],
            "source_archives_required": False,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "derived.json"
    result = derive_freeze_manifest(canonical, base_path, contract_path, output)
    assert result["runtime_tree_sha256"] == base["runtime_tree_sha256"]
    assert result["runtime_files"] == base["runtime_files"]
    assert result["contract_sha256"] == _sha256(contract_path)
    assert result["derived_from"]["runtime_files_changed"] is False
