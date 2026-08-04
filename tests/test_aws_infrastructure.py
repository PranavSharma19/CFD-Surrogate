from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AWS_ROOT = REPOSITORY_ROOT / "infra/aws"


def test_aws_container_is_digest_pinned_and_contract_labeled() -> None:
    dockerfile = (AWS_ROOT / "Dockerfile").read_text()
    assert "FROM pytorch/pytorch@sha256:" in dockerfile
    assert "ARG CONTRACT_SHA256" in dockerfile
    assert "configs/aws_preop_v1_frozen.json" in dockerfile
    assert "43000bd90a93afad3fefc41b5cef8cd9b3042a341ea59485138ee44fc6b8f17a" in dockerfile


def test_aws_staging_requires_registered_bundle_gpu_and_private_s3() -> None:
    common = (AWS_ROOT / "common.sh").read_text()
    prepare = (AWS_ROOT / "prepare_instance.sh").read_text()
    assert "7431ad11ab26706f90ddcd6f40be3b7b417cfd86c93e885fe3b34e3a1f538dcf" in common
    assert "AWS_PROJECT_ROOT" in common and "/mnt/*" in common
    assert "AWS_S3_URI" in common and "s3://" in common
    assert "L40S" in common
    assert "verify_watcloud_bundle" in prepare
    assert "derive_freeze_manifest" in prepare
    assert "verify_staged_runtime" in prepare


def test_aws_sweep_and_training_preserve_resume_and_sync() -> None:
    sweep = (AWS_ROOT / "memory_sweep.sh").read_text()
    train = (AWS_ROOT / "train_fold.sh").read_text()
    for patch_nodes in (8192, 12288, 16384, 20480, 24576):
        assert str(patch_nodes) in sweep
    assert "--stop-after-step 1" in sweep
    assert "--resume" in sweep
    assert "select_watcloud_patch" in sweep
    assert "sync_results" in sweep
    assert "periodic_sync" in train
    assert "latest_checkpoint.pt" in train
    assert "result.json" in train


def test_aws_contract_keeps_spot_disabled_for_first_fold() -> None:
    contract = json.loads(
        (REPOSITORY_ROOT / "configs/aws_preop_v1_frozen.json").read_text()
    )
    hardware = contract["hardware"]
    assert hardware["target_instance_type"] == "g6e.2xlarge"
    assert hardware["purchase_model_for_first_fold"] == "on_demand"
    assert hardware["maximum_allocated_vram_gib"] == 22.0
    assert hardware["spot_policy"].startswith("disabled")
