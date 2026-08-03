from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aorta_surrogate.training.freeze_experiment import _tree_sha256
from aorta_surrogate.training.package_watcloud import _deterministic_tar_gz
from aorta_surrogate.training.select_watcloud_patch import select_patch
from aorta_surrogate.training.verify_staged_runtime import verify_staged_runtime
from aorta_surrogate.training.verify_watcloud_bundle import verify_data_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deterministic_archive_reproduces_byte_for_byte(tmp_path: Path) -> None:
    first_source = tmp_path / "b.txt"
    second_source = tmp_path / "a.txt"
    first_source.write_text("beta\n", encoding="utf-8")
    second_source.write_text("alpha\n", encoding="utf-8")
    rows = [(first_source, "payload/b.txt"), (second_source, "payload/a.txt")]
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _deterministic_tar_gz(first, rows)
    _deterministic_tar_gz(second, list(reversed(rows)))
    assert _sha256(first) == _sha256(second)


def test_bundle_verifier_rejects_unregistered_archive_member(tmp_path: Path) -> None:
    source = tmp_path / "unexpected.txt"
    source.write_text("unexpected", encoding="utf-8")
    bundle = tmp_path / "bad.tar.gz"
    _deterministic_tar_gz(bundle, [(source, "canonical/unexpected.txt")])
    with pytest.raises(ValueError, match="freeze manifest"):
        verify_data_bundle(bundle)


def test_staged_runtime_verifier_rejects_a_locked_case(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    development = canonical / "case_a"
    development.mkdir(parents=True)
    runtime_file = development / "case_manifest.json"
    runtime_file.write_text("{}", encoding="utf-8")
    row = {
        "relative_path": "case_a/case_manifest.json",
        "size_bytes": runtime_file.stat().st_size,
        "sha256": _sha256(runtime_file),
    }
    manifest = {
        "development_case_ids": ["case_a"],
        "locked_case_ids_excluded": ["case_locked"],
        "runtime_files": [row],
        "runtime_file_count": 1,
        "runtime_tree_sha256": _tree_sha256([row]),
        "contract_sha256": "contract",
    }
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_staged_runtime(canonical, manifest_path)["status"] == "pass"
    locked = canonical / "case_locked"
    locked.mkdir()
    (locked / "case_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="permitted frozen set"):
        verify_staged_runtime(canonical, manifest_path)


def _write_candidate(root: Path, patch_nodes: int, peak_vram_gib: float) -> None:
    candidate = root / f"patch_{patch_nodes}"
    candidate.mkdir(parents=True)
    (candidate / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_validation_complete_cycle_tawss_relative_error": 0.2,
            }
        ),
        encoding="utf-8",
    )
    (candidate / "runtime_and_vram.json").write_text(
        json.dumps({"peak_allocated_vram_gib": peak_vram_gib}), encoding="utf-8"
    )
    (candidate / "training_history.jsonl").write_text(
        '{"step": 1}\n{"step": 2}\n', encoding="utf-8"
    )
    (candidate / "best_checkpoint.pt").write_bytes(b"best")
    (candidate / "latest_checkpoint.pt").write_bytes(b"latest")


def test_patch_selector_chooses_largest_candidate_under_registered_limit(
    tmp_path: Path,
) -> None:
    contract = {
        "experiment_id": "test",
        "patch_protocol": {
            "patch_node_candidates": [8192, 12288, 16384],
            "patch_selection_rule": "largest passing",
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    sweep = tmp_path / "sweep"
    _write_candidate(sweep, 8192, 8.0)
    _write_candidate(sweep, 12288, 18.0)
    _write_candidate(sweep, 16384, 22.5)
    report = select_patch(contract_path, sweep)
    assert report["selected_patch_nodes"] == 12288
    assert json.loads((sweep / "patch_selection.json").read_text())["status"] == "patch_selected"


def test_cloud_files_pin_contract_gpu_and_container() -> None:
    dockerfile = (REPOSITORY_ROOT / "infra/watcloud/Dockerfile").read_text()
    sweep = (REPOSITORY_ROOT / "infra/watcloud/slurm/memory_sweep.sbatch").read_text()
    folds = (REPOSITORY_ROOT / "infra/watcloud/slurm/train_folds.sbatch").read_text()
    requirements = (REPOSITORY_ROOT / "infra/watcloud/requirements.lock").read_text()
    assert "FROM pytorch/pytorch@sha256:" in dockerfile
    assert "2c04fec795409ec6d8f768aac4bd68f276e40acf76e157d44d31bb4e8ee75cf4" in dockerfile
    assert "#SBATCH --nodelist=trpro-slurm2" in sweep
    assert "#SBATCH --gres=gpu:rtx_4090:1,tmpdisk:30720" in sweep
    assert "--stop-after-step 1" in sweep
    assert "--resume" in sweep
    assert "#SBATCH --partition=compute_dense" in folds
    assert "#SBATCH --array=0-2%1" in folds
    assert "patch_selection.json" in folds
    assert "google-crc32c==" in requirements
    assert "typing-extensions==" in requirements
    assert "xxhash==" in requirements
