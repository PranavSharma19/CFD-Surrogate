"""Bounded patient-cycle training for a leakage-safe development fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from aorta_surrogate.data.normalization import compute_case_stats
from aorta_surrogate.data.features import REGION_NAMES
from aorta_surrogate.models.gat_baseline import AorticGATBaseline
from aorta_surrogate.models.pointnet_baseline import AorticPointNetBaseline
from aorta_surrogate.models.equivariant_surface import EquivariantSurfaceGNN
from aorta_surrogate.training.pyg_adapter import make_training_patch


@dataclass(frozen=True)
class FoldTrainingConfig:
    model_name: str = "gat"
    fold_index: int = 0
    steps: int = 240
    max_nodes: int = 4096
    patch_method: str = "geodesic"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    seed: int = 20260802
    vector_weight: float = 1.0
    magnitude_weight: float = 0.2
    direction_weight: float = 0.05
    temporal_weight: float = 0.05
    direction_floor_pa: float = 0.5
    validation_interval: int = 200
    validation_phases: tuple[int, ...] = (0, 5, 10, 15, 20)


def _build_model(
    model_name: str,
    *,
    hidden_dim: int | None = None,
    layers: int | None = None,
    gradient_checkpointing: bool = False,
):
    if model_name == "gat":
        return AorticGATBaseline()
    if model_name == "pointnet":
        return AorticPointNetBaseline()
    if model_name == "equivariant":
        kwargs = {"gradient_checkpointing": gradient_checkpointing}
        if hidden_dim is not None:
            kwargs["hidden_dim"] = hidden_dim
        if layers is not None:
            kwargs["layers"] = layers
        return EquivariantSurfaceGNN(**kwargs)
    if model_name == "equivariant_multiscale":
        kwargs = {
            "scalar_dim": 14,
            "use_multiscale": True,
            "gradient_checkpointing": gradient_checkpointing,
        }
        if hidden_dim is not None:
            kwargs["hidden_dim"] = hidden_dim
        if layers is not None:
            kwargs["layers"] = layers
        return EquivariantSurfaceGNN(**kwargs)
    raise ValueError("unsupported model_name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale_pa: float,
    config,
    valid_mask: torch.Tensor | None = None,
):
    if valid_mask is not None:
        if not bool(valid_mask.any()):
            raise ValueError("training patch contains no valid target nodes")
        prediction = prediction[valid_mask]
        target = target[valid_mask]
    vector = functional.smooth_l1_loss(prediction, target, beta=0.25)
    predicted_magnitude = torch.linalg.vector_norm(prediction, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target, dim=-1)
    magnitude = functional.smooth_l1_loss(predicted_magnitude, target_magnitude, beta=0.25)
    stable = target_magnitude > config.direction_floor_pa / scale_pa
    if bool(stable.any()):
        direction = (1.0 - functional.cosine_similarity(prediction[stable], target[stable], dim=-1)).mean()
    else:
        direction = prediction.new_zeros(())
    return vector, magnitude, direction


def _evaluate(
    model,
    canonical_root,
    case_ids,
    stats_path,
    config,
    device,
    *,
    target_mask_policy: str = "primary",
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    model.eval()
    vector_abs_sum = vector_square_sum = magnitude_abs_sum = target_magnitude_sum = 0.0
    angular_sum = target_square_sum = 0.0
    angular_count = node_count = 0
    regional: dict[int, dict[str, float]] = {}
    per_case: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for case_offset, case_id in enumerate(case_ids):
            for phase_index in config.validation_phases:
                patch = make_training_patch(
                    canonical_root,
                    case_id,
                    phase_index,
                    max_nodes=config.max_nodes,
                    seed=config.seed + 10_000 + 101 * case_offset + phase_index,
                    normalization_path=stats_path,
                    patch_method=config.patch_method,
                    target_mask_policy=target_mask_policy,
                ).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype or torch.float32,
                    enabled=autocast_dtype is not None,
                ):
                    prediction = model(patch) * patch.target_scale_pa
                prediction = prediction.float()
                target = patch.y * patch.target_scale_pa
                valid = patch.target_valid_mask
                prediction = prediction[valid]
                target = target[valid]
                difference = prediction - target
                vector_error = torch.linalg.vector_norm(difference, dim=-1)
                predicted_magnitude = torch.linalg.vector_norm(prediction, dim=-1)
                target_magnitude = torch.linalg.vector_norm(target, dim=-1)
                magnitude_error = torch.abs(predicted_magnitude - target_magnitude)
                case_bucket = per_case.setdefault(
                    case_id, {"nodes": 0.0, "vector_abs": 0.0, "magnitude_abs": 0.0, "target_magnitude": 0.0}
                )
                case_bucket["nodes"] += int(valid.sum())
                case_bucket["vector_abs"] += float(vector_error.sum())
                case_bucket["magnitude_abs"] += float(magnitude_error.sum())
                case_bucket["target_magnitude"] += float(target_magnitude.sum())
                valid_regions = patch.semantic_region_id[valid]
                for region_id_tensor in torch.unique(valid_regions):
                    region_id = int(region_id_tensor)
                    region_mask = valid_regions == region_id
                    bucket = regional.setdefault(
                        region_id,
                        {"nodes": 0.0, "vector_abs": 0.0, "magnitude_abs": 0.0, "target_magnitude": 0.0},
                    )
                    bucket["nodes"] += int(region_mask.sum())
                    bucket["vector_abs"] += float(vector_error[region_mask].sum())
                    bucket["magnitude_abs"] += float(magnitude_error[region_mask].sum())
                    bucket["target_magnitude"] += float(target_magnitude[region_mask].sum())
                vector_abs_sum += float(vector_error.sum())
                vector_square_sum += float(torch.square(vector_error).sum())
                magnitude_abs_sum += float(magnitude_error.sum())
                target_magnitude_sum += float(target_magnitude.sum())
                target_square_sum += float(torch.square(target_magnitude).sum())
                stable = target_magnitude > config.direction_floor_pa
                if bool(stable.any()):
                    cosine = functional.cosine_similarity(prediction[stable], target[stable], dim=-1).clamp(-1, 1)
                    angular_sum += float(torch.rad2deg(torch.acos(cosine)).sum())
                    angular_count += int(stable.sum())
                node_count += int(valid.sum())
    def finalize(bucket: dict[str, float]) -> dict[str, float | int]:
        count = max(bucket["nodes"], 1.0)
        return {
            "sampled_node_count": int(bucket["nodes"]),
            "wss_vector_mae_pa": bucket["vector_abs"] / count,
            "wss_magnitude_mae_pa": bucket["magnitude_abs"] / count,
            "wss_magnitude_relative_error": bucket["magnitude_abs"] / max(bucket["target_magnitude"], 1.0e-12),
        }

    return {
        "target_mask_policy": target_mask_policy,
        "sampled_node_count": node_count,
        "wss_vector_mae_pa": vector_abs_sum / node_count,
        "wss_vector_rmse_pa": (vector_square_sum / node_count) ** 0.5,
        "wss_magnitude_mae_pa": magnitude_abs_sum / node_count,
        "wss_magnitude_relative_error": magnitude_abs_sum / max(target_magnitude_sum, 1.0e-12),
        "mean_angular_error_degrees": angular_sum / max(angular_count, 1),
        "angular_metric_node_count": angular_count,
        "zero_baseline_vector_mae_pa": target_magnitude_sum / node_count,
        "zero_baseline_vector_rmse_pa": (target_square_sum / node_count) ** 0.5,
        "zero_baseline_magnitude_relative_error": 1.0,
        "per_patient": {case_id: finalize(bucket) for case_id, bucket in sorted(per_case.items())},
        "per_region": {
            REGION_NAMES.get(region_id, f"unknown_{region_id}"): finalize(bucket)
            for region_id, bucket in sorted(regional.items())
        },
    }


def train_fold(
    canonical_root: Path,
    output_dir: Path,
    config: FoldTrainingConfig = FoldTrainingConfig(),
    resume_checkpoint: Path | None = None,
) -> dict[str, object]:
    split_path = canonical_root / "patient_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    folds = split["development_cv_folds"]
    if not 0 <= config.fold_index < len(folds):
        raise ValueError(f"fold index must be in [0, {len(folds) - 1}]")
    validation_cases = list(folds[config.fold_index])
    training_cases = sorted(set(split["development"]) - set(validation_cases))
    locked_test = set(split["locked_test"])
    if locked_test & (set(training_cases) | set(validation_cases)):
        raise ValueError("locked-test leakage detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / f"fold{config.fold_index}_normalization.json"
    stats = compute_case_stats(
        canonical_root,
        training_cases,
        source_split=f"development_cv_fold_{config.fold_index}_train",
        excluded_case_ids=validation_cases + list(locked_test),
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config.model_name).to(device)
    resumed_from: str | None = None
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model_state"])
        resumed_from = str(resume_checkpoint.resolve())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.steps, 1), eta_min=config.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int | str]] = []
    validation_history: list[dict[str, object]] = []
    validation_elapsed = 0.0
    best_validation_mae = float("inf")
    best_step = 0
    artifact_prefix = f"{config.model_name}_fold{config.fold_index}"
    best_checkpoint_path = output_dir / f"{artifact_prefix}_best.pt"
    started = time.perf_counter()

    for step in range(config.steps):
        case_id = training_cases[int(rng.integers(len(training_cases)))]
        phase_index = int(rng.integers(20))
        patch_seed = int(rng.integers(2**31 - 1))
        current = make_training_patch(
            canonical_root,
            case_id,
            phase_index,
            max_nodes=config.max_nodes,
            seed=patch_seed,
            normalization_path=stats_path,
            patch_method=config.patch_method,
        ).to(device)
        following = make_training_patch(
            canonical_root,
            case_id,
            phase_index + 1,
            max_nodes=config.max_nodes,
            seed=patch_seed,
            normalization_path=stats_path,
            patch_method=config.patch_method,
        ).to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(current)
            following_prediction = model(following)
            vector, magnitude, direction = _loss_terms(
                prediction,
                current.y,
                current.target_scale_pa,
                config,
                current.target_valid_mask,
            )
            temporal_valid = current.target_valid_mask & following.target_valid_mask
            if not bool(temporal_valid.any()):
                raise ValueError("paired training patches contain no valid temporal targets")
            temporal = functional.smooth_l1_loss(
                (following_prediction - prediction)[temporal_valid],
                (following.y - current.y)[temporal_valid],
                beta=0.25,
            )
            loss = (
                config.vector_weight * vector
                + config.magnitude_weight * magnitude
                + config.direction_weight * direction
                + config.temporal_weight * temporal
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        history.append(
            {
                "step": step + 1,
                "case_id": case_id,
                "phase_index": phase_index,
                "loss": float(loss.detach().cpu()),
                "vector_loss": float(vector.detach().cpu()),
                "magnitude_loss": float(magnitude.detach().cpu()),
                "direction_loss": float(direction.detach().cpu()),
                "temporal_loss": float(temporal.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if (step + 1) % config.validation_interval == 0 or step + 1 == config.steps:
            validation_started = time.perf_counter()
            metrics = _evaluate(
                model, canonical_root, validation_cases, stats_path, config, device
            )
            validation_elapsed += time.perf_counter() - validation_started
            validation_history.append({"step": step + 1, "metrics": metrics})
            if metrics["wss_magnitude_mae_pa"] < best_validation_mae:
                best_validation_mae = metrics["wss_magnitude_mae_pa"]
                best_step = step + 1
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(config),
                        "training_cases": training_cases,
                        "validation_cases": validation_cases,
                        "normalization_stats": stats,
                        "validation_metrics": metrics,
                        "step": best_step,
                    },
                    best_checkpoint_path,
                )
            # Keep useful progress if a local session or remote worker is interrupted.
            (output_dir / f"{artifact_prefix}_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            (output_dir / f"{artifact_prefix}_validation_history.json").write_text(
                json.dumps(validation_history, indent=2), encoding="utf-8"
            )

    training_elapsed = time.perf_counter() - started
    validation_metrics = validation_history[-1]["metrics"]
    checkpoint_path = output_dir / f"{artifact_prefix}_last.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "training_cases": training_cases,
            "validation_cases": validation_cases,
            "normalization_stats": stats,
        },
        checkpoint_path,
    )
    result = {
        "status": "completed_development_run",
        "research_only": True,
        "fold_index": config.fold_index,
        "training_cases": training_cases,
        "validation_cases": validation_cases,
        "locked_test_cases_not_accessed": sorted(locked_test),
        "resumed_from": resumed_from,
        "patient_split_sha256": _sha256(split_path),
        "normalization_sha256": _sha256(stats_path),
        "config": asdict(config),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_elapsed_seconds": training_elapsed,
        "validation_elapsed_seconds": validation_elapsed,
        "seconds_per_training_step": training_elapsed / config.steps,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None,
        "initial_loss_mean_10": float(np.mean([row["loss"] for row in history[:10]])),
        "final_loss_mean_10": float(np.mean([row["loss"] for row in history[-10:]])),
        "validation_metrics": validation_metrics,
        "best_validation_step": best_step,
        "best_validation_wss_magnitude_mae_pa": best_validation_mae,
        "best_checkpoint": best_checkpoint_path.name,
        "last_checkpoint": checkpoint_path.name,
    }
    (output_dir / f"{artifact_prefix}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / f"{artifact_prefix}_validation_history.json").write_text(
        json.dumps(validation_history, indent=2), encoding="utf-8"
    )
    (output_dir / f"{artifact_prefix}_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--model", choices=("gat", "pointnet", "equivariant", "equivariant_multiscale"), default="gat"
    )
    args = parser.parse_args()
    config = FoldTrainingConfig(
        fold_index=args.fold_index,
        model_name=args.model,
        steps=args.steps,
        max_nodes=args.max_nodes,
        validation_interval=args.validation_interval,
    )
    print(json.dumps(train_fold(
        args.canonical_root, args.output_dir, config, resume_checkpoint=args.resume
    ), indent=2))


if __name__ == "__main__":
    main()
