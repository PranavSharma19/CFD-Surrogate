"""Contract-driven cloud trainer for frozen preoperative AAA experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional

from aorta_surrogate.data.features import REGION_NAMES
from aorta_surrogate.data.normalization import compute_case_stats
from aorta_surrogate.hemodynamics import HemodynamicMetrics, compute_hemodynamic_metrics
from aorta_surrogate.training.cycle_evaluation import STABILITY_FLOOR_PA, _metric_errors
from aorta_surrogate.training.cloud_contract import (
    registered_vram_limit_gib,
    validate_registered_gpu,
)
from aorta_surrogate.training.fold_trainer import _build_model, _evaluate
from aorta_surrogate.training.freeze_experiment import _validate_contract
from aorta_surrogate.training.pyg_adapter import make_training_patch


SUPPORTED_REGION_IDS = (0, 1, 2, 3, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_region_seed(
    semantic_region_ids: np.ndarray,
    rng: np.random.Generator,
    supported_region_ids: tuple[int, ...] = SUPPORTED_REGION_IDS,
) -> tuple[int, int]:
    """Choose a supported region uniformly, then a node uniformly within it."""
    available = [
        region_id
        for region_id in supported_region_ids
        if bool(np.any(semantic_region_ids == region_id))
    ]
    if not available:
        raise ValueError("case contains none of the registered supported regions")
    region_id = available[int(rng.integers(len(available)))]
    candidates = np.flatnonzero(semantic_region_ids == region_id)
    node_id = int(candidates[int(rng.integers(len(candidates)))])
    return region_id, node_id


def region_balanced_reduce(
    per_node_values: torch.Tensor,
    region_ids: torch.Tensor,
    supported_region_ids: tuple[int, ...] = SUPPORTED_REGION_IDS,
) -> torch.Tensor:
    """Blend the global mean with an equal-weight supported-region macro mean."""
    if per_node_values.ndim != 1 or region_ids.shape != per_node_values.shape:
        raise ValueError("regional reduction expects aligned one-dimensional tensors")
    if per_node_values.numel() == 0:
        raise ValueError("cannot reduce an empty loss tensor")
    regional = [
        per_node_values[region_ids == region_id].mean()
        for region_id in supported_region_ids
        if bool((region_ids == region_id).any())
    ]
    if not regional:
        return per_node_values.mean()
    return 0.5 * per_node_values.mean() + 0.5 * torch.stack(regional).mean()


def balanced_loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale_pa: float,
    semantic_region_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    direction_floor_pa: float,
) -> dict[str, torch.Tensor]:
    if not bool(valid_mask.any()):
        raise ValueError("training patch contains no valid target nodes")
    prediction = prediction[valid_mask]
    target = target[valid_mask]
    regions = semantic_region_ids[valid_mask]
    vector_per_node = functional.smooth_l1_loss(
        prediction, target, beta=0.25, reduction="none"
    ).mean(dim=-1)
    predicted_magnitude = torch.linalg.vector_norm(prediction, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target, dim=-1)
    magnitude_per_node = functional.smooth_l1_loss(
        predicted_magnitude, target_magnitude, beta=0.25, reduction="none"
    )
    stable = target_magnitude > direction_floor_pa / scale_pa
    if bool(stable.any()):
        direction_per_node = 1.0 - functional.cosine_similarity(
            prediction[stable], target[stable], dim=-1
        )
        direction = region_balanced_reduce(direction_per_node, regions[stable])
    else:
        direction = prediction.new_zeros(())
    return {
        "vector": region_balanced_reduce(vector_per_node, regions),
        "magnitude": region_balanced_reduce(magnitude_per_node, regions),
        "direction": direction,
    }


def balanced_temporal_loss(
    prediction_delta: torch.Tensor,
    target_delta: torch.Tensor,
    semantic_region_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(valid_mask.any()):
        raise ValueError("paired patches contain no valid temporal targets")
    per_node = functional.smooth_l1_loss(
        prediction_delta[valid_mask],
        target_delta[valid_mask],
        beta=0.25,
        reduction="none",
    ).mean(dim=-1)
    return region_balanced_reduce(per_node, semantic_region_ids[valid_mask])


def _concatenate_metrics(rows: list[HemodynamicMetrics]) -> HemodynamicMetrics:
    return HemodynamicMetrics(
        tawss=np.concatenate([row.tawss for row in rows]),
        osi=np.concatenate([row.osi for row in rows]),
        rrt=np.concatenate([row.rrt for row in rows]),
        rrt_valid=np.concatenate([row.rrt_valid for row in rows]),
    )


def evaluate_complete_cycles_live(
    model,
    canonical_root: Path,
    case_ids: list[str],
    normalization_path: Path,
    *,
    patch_nodes: int,
    seed: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    target_mask_policy: str,
) -> dict[str, object]:
    """Evaluate fixed complete-cycle patches without writing or loading checkpoints."""
    import zarr

    model.eval()
    per_patient: dict[str, object] = {}
    predicted_rows: list[HemodynamicMetrics] = []
    target_rows: list[HemodynamicMetrics] = []
    regional_predicted: dict[int, list[HemodynamicMetrics]] = {}
    regional_target: dict[int, list[HemodynamicMetrics]] = {}
    with torch.no_grad():
        for case_offset, case_id in enumerate(case_ids):
            patch_seed = seed + 50_000 + case_offset * 137
            predicted_phases: list[np.ndarray] = []
            target_phases: list[np.ndarray] = []
            node_ids = valid_mask = regions = None
            for phase_index in range(21):
                patch = make_training_patch(
                    canonical_root,
                    case_id,
                    phase_index,
                    max_nodes=patch_nodes,
                    seed=patch_seed,
                    normalization_path=normalization_path,
                    patch_method="geodesic",
                    target_mask_policy=target_mask_policy,
                ).to(device)
                current_ids = patch.global_node_ids.detach().cpu().numpy()
                current_valid = patch.target_valid_mask.detach().cpu().numpy().astype(bool)
                current_regions = patch.semantic_region_id.detach().cpu().numpy()
                if node_ids is None:
                    node_ids, valid_mask, regions = current_ids, current_valid, current_regions
                elif not (
                    np.array_equal(node_ids, current_ids)
                    and np.array_equal(valid_mask, current_valid)
                    and np.array_equal(regions, current_regions)
                ):
                    raise ValueError("complete-cycle patches changed across phases")
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=device.type == "cuda",
                ):
                    predicted = model(patch) * patch.target_scale_pa
                predicted_phases.append(predicted.float().cpu().numpy())
                target_phases.append(
                    (patch.y * patch.target_scale_pa).float().cpu().numpy()
                )

            if valid_mask is None or regions is None or not valid_mask.any():
                raise ValueError("complete-cycle patch contains no valid target nodes")
            predicted_wss = np.stack(predicted_phases)[:, valid_mask]
            target_wss = np.stack(target_phases)[:, valid_mask]
            valid_regions = regions[valid_mask]
            targets = zarr.open_group(
                str(canonical_root / case_id / "targets.zarr"), mode="r"
            )
            times = np.asarray(targets["time_seconds"])
            predicted_metrics = compute_hemodynamic_metrics(
                predicted_wss, times, tawss_floor_pa=STABILITY_FLOOR_PA
            )
            target_metrics = compute_hemodynamic_metrics(
                target_wss, times, tawss_floor_pa=STABILITY_FLOOR_PA
            )
            per_patient[case_id] = _metric_errors(predicted_metrics, target_metrics)
            predicted_rows.append(predicted_metrics)
            target_rows.append(target_metrics)
            for region_id in np.unique(valid_regions):
                active = valid_regions == region_id
                regional_predicted.setdefault(int(region_id), []).append(
                    HemodynamicMetrics(
                        tawss=predicted_metrics.tawss[active],
                        osi=predicted_metrics.osi[active],
                        rrt=predicted_metrics.rrt[active],
                        rrt_valid=predicted_metrics.rrt_valid[active],
                    )
                )
                regional_target.setdefault(int(region_id), []).append(
                    HemodynamicMetrics(
                        tawss=target_metrics.tawss[active],
                        osi=target_metrics.osi[active],
                        rrt=target_metrics.rrt[active],
                        rrt_valid=target_metrics.rrt_valid[active],
                    )
                )

    pooled_predicted = _concatenate_metrics(predicted_rows)
    pooled_target = _concatenate_metrics(target_rows)
    return {
        "schema_version": "1.0.0",
        "target_mask_policy": target_mask_policy,
        "phase_count": 21,
        "validation_cases": case_ids,
        "pooled": _metric_errors(pooled_predicted, pooled_target),
        "per_patient": per_patient,
        "per_region": {
            REGION_NAMES.get(region_id, f"unknown_{region_id}"): _metric_errors(
                _concatenate_metrics(regional_predicted[region_id]),
                _concatenate_metrics(regional_target[region_id]),
            )
            for region_id in sorted(regional_predicted)
        },
    }


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rng_state(rng: np.random.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, object], rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    rng.bit_generator.state = state["numpy_generator"]
    # torch.load(map_location=<cuda>) also relocates saved RNG tensors.  The
    # default CPU generator explicitly requires a CPU ByteTensor, while CUDA's
    # setter accepts the corresponding CPU state tensors for each device.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all([row.cpu() for row in state["torch_cuda"]])


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    best_cycle_tawss: float,
    stale_cycle_validations: int,
    rng: np.random.Generator,
    resolved_manifest: dict[str, object],
    cycle_metrics: dict[str, object] | None,
) -> None:
    torch.save(
        {
            "schema_version": "1.0.0",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": step,
            "best_cycle_tawss_relative_error": best_cycle_tawss,
            "stale_cycle_validations": stale_cycle_validations,
            "rng_state": _rng_state(rng),
            "resolved_manifest": resolved_manifest,
            "cycle_metrics": cycle_metrics,
        },
        path,
    )


def train_watcloud_fold(
    contract_path: Path,
    canonical_root: Path,
    freeze_manifest_path: Path,
    output_dir: Path,
    *,
    fold_index: int,
    patch_nodes: int,
    precision: str = "bf16_mixed",
    precision_fallback_reason: str | None = None,
    resume_checkpoint: Path | None = None,
    smoke_steps: int | None = None,
    allow_cpu_smoke: bool = False,
    stop_after_step: int | None = None,
) -> dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    split_path = canonical_root / "patient_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development, locked = _validate_contract(contract, split)
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    contract_sha = _sha256(contract_path)
    if freeze_manifest["contract_sha256"] != contract_sha:
        raise ValueError("freeze manifest does not match the frozen contract")
    if freeze_manifest["locked_case_file_count"] != 0:
        raise ValueError("freeze manifest contains locked-patient files")
    if not 0 <= fold_index < len(split["development_cv_folds"]):
        raise ValueError("invalid fold index")
    candidates = list(contract["patch_protocol"]["patch_node_candidates"])
    production_mode = smoke_steps is None
    if production_mode and patch_nodes not in candidates:
        raise ValueError("production patch size must come from the registered VRAM sweep")
    if precision not in {"bf16_mixed", "fp16_mixed"}:
        raise ValueError("precision must be bf16_mixed or fp16_mixed")
    if precision == "fp16_mixed" and not precision_fallback_reason:
        raise ValueError("FP16 fallback requires a documented reason")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not (smoke_steps is not None and allow_cpu_smoke):
        raise RuntimeError("cloud production training requires CUDA")
    if production_mode:
        validate_registered_gpu(contract, torch.cuda.get_device_name(0))
    autocast_dtype = torch.bfloat16 if precision == "bf16_mixed" else torch.float16
    validation_cases = list(split["development_cv_folds"][fold_index])
    training_cases = sorted(set(development) - set(validation_cases))
    if set(training_cases) & set(locked) or set(validation_cases) & set(locked):
        raise ValueError("locked-test leakage detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is None and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory without --resume: {output_dir}"
        )
    normalization_path = output_dir / "fold_normalization.json"
    if resume_checkpoint is None:
        stats = compute_case_stats(
            canonical_root,
            training_cases,
            source_split=f"{contract['experiment_id']}_fold_{fold_index}_train",
            excluded_case_ids=validation_cases + locked,
        )
        normalization_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    elif not normalization_path.exists():
        raise FileNotFoundError("resume output is missing fold_normalization.json")

    optimization = contract["optimization"]
    seed = int(optimization["fold_seeds"][str(fold_index)])
    # Seed before parameter construction.  Seeding only before sampling would
    # reproduce patches while silently giving each process different weights.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_contract = contract["model"]
    model = _build_model(
        "equivariant",
        hidden_dim=int(model_contract["hidden_dim"]),
        layers=int(model_contract["message_layers"]),
        gradient_checkpointing=bool(contract["optimization"]["gradient_checkpointing"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    maximum_steps = int(smoke_steps or optimization["maximum_steps_per_fold"])
    if stop_after_step is not None and not 0 < stop_after_step <= maximum_steps:
        raise ValueError("stop_after_step must be within the configured run")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=maximum_steps,
        eta_min=float(optimization["learning_rate"]) * 0.05,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16_mixed"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(seed)

    resolved_manifest = {
        "schema_version": "1.0.0",
        "experiment_id": contract["experiment_id"],
        "run_mode": "production" if production_mode else "nonproduction_smoke",
        "contract_sha256": contract_sha,
        "runtime_tree_sha256": freeze_manifest["runtime_tree_sha256"],
        "fold_index": fold_index,
        "training_cases": training_cases,
        "validation_cases": validation_cases,
        "locked_cases_not_accessed": locked,
        "patch_nodes": patch_nodes,
        "precision": precision,
        "precision_fallback_reason": precision_fallback_reason,
        "model": model_contract,
        "maximum_steps": maximum_steps,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (output_dir / "resolved_experiment_manifest.json").write_text(
        json.dumps(resolved_manifest, indent=2), encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": resolved_manifest["gpu_name"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output_dir / "environment_manifest.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    semantic_regions: dict[str, np.ndarray] = {}
    import zarr

    for case_id in training_cases:
        features = zarr.open_group(
            str(canonical_root / case_id / "features.zarr"), mode="r"
        )
        semantic_regions[case_id] = np.asarray(features["semantic_region_id"])

    start_step = 0
    best_cycle_tawss = float("inf")
    stale_cycle_validations = 0
    latest_cycle_metrics = None
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        previous = resumed["resolved_manifest"]
        for key in ("contract_sha256", "runtime_tree_sha256", "fold_index", "patch_nodes", "precision"):
            if previous[key] != resolved_manifest[key]:
                raise ValueError(f"resume checkpoint disagrees on {key}")
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        scheduler.load_state_dict(resumed["scheduler_state"])
        scaler.load_state_dict(resumed["scaler_state"])
        start_step = int(resumed["step"])
        best_cycle_tawss = float(resumed["best_cycle_tawss_relative_error"])
        stale_cycle_validations = int(resumed["stale_cycle_validations"])
        latest_cycle_metrics = resumed["cycle_metrics"]
        _restore_rng_state(resumed["rng_state"], rng)

    training_history_path = output_dir / "training_history.jsonl"
    validation_history_path = output_dir / "validation_history.jsonl"
    monitoring_config = SimpleNamespace(
        validation_phases=tuple(optimization["monitoring_validation_phases"]),
        max_nodes=patch_nodes,
        seed=seed,
        patch_method="geodesic",
        direction_floor_pa=float(contract["loss"]["direction"]["target_magnitude_floor_pa"]),
    )
    accumulation = int(optimization["gradient_accumulation_steps"])
    loss_weights = {
        "vector": float(contract["loss"]["robust_vector"]["weight"]),
        "magnitude": float(contract["loss"]["magnitude"]["weight"]),
        "direction": float(contract["loss"]["direction"]["weight"]),
        "temporal": float(contract["loss"]["temporal_delta"]["weight"]),
    }
    monitoring_interval = int(optimization["monitoring_validation_interval_steps"])
    cycle_interval = int(optimization["cycle_validation_interval_steps"])
    started = time.perf_counter()
    stopped_early = False
    last_completed_step = start_step

    for step_index in range(start_step, maximum_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = {name: 0.0 for name in (*loss_weights, "total")}
        sampled_regions: list[int] = []
        for _ in range(accumulation):
            case_id = training_cases[int(rng.integers(len(training_cases)))]
            phase_index = int(rng.integers(20))
            region_id, seed_node = choose_region_seed(semantic_regions[case_id], rng)
            sampled_regions.append(region_id)
            current = make_training_patch(
                canonical_root,
                case_id,
                phase_index,
                max_nodes=patch_nodes,
                seed=int(rng.integers(2**31 - 1)),
                seed_node=seed_node,
                normalization_path=normalization_path,
                patch_method="geodesic",
                target_mask_policy="primary",
            ).to(device)
            following = make_training_patch(
                canonical_root,
                case_id,
                phase_index + 1,
                max_nodes=patch_nodes,
                seed=0,
                seed_node=seed_node,
                normalization_path=normalization_path,
                patch_method="geodesic",
                target_mask_policy="primary",
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = model(current)
                following_prediction = model(following)
                terms = balanced_loss_terms(
                    prediction,
                    current.y,
                    current.target_scale_pa,
                    current.semantic_region_id,
                    current.target_valid_mask,
                    float(contract["loss"]["direction"]["target_magnitude_floor_pa"]),
                )
                temporal_valid = current.target_valid_mask & following.target_valid_mask
                temporal = balanced_temporal_loss(
                    following_prediction - prediction,
                    following.y - current.y,
                    current.semantic_region_id,
                    temporal_valid,
                )
                total = (
                    loss_weights["vector"] * terms["vector"]
                    + loss_weights["magnitude"] * terms["magnitude"]
                    + loss_weights["direction"] * terms["direction"]
                    + loss_weights["temporal"] * temporal
                )
            scaler.scale(total / accumulation).backward()
            for name in ("vector", "magnitude", "direction"):
                accumulated[name] += float(terms[name].detach().float().cpu()) / accumulation
            accumulated["temporal"] += float(temporal.detach().float().cpu()) / accumulation
            accumulated["total"] += float(total.detach().float().cpu()) / accumulation

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        completed_step = step_index + 1
        last_completed_step = completed_step
        _append_jsonl(
            training_history_path,
            {
                "step": completed_step,
                "losses": accumulated,
                "sampled_region_ids": sampled_regions,
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
        )

        forced_stop = stop_after_step == completed_step
        monitoring_due = completed_step % monitoring_interval == 0 or completed_step == maximum_steps
        cycle_due = completed_step % cycle_interval == 0 or completed_step == maximum_steps
        if monitoring_due:
            monitoring = _evaluate(
                model,
                canonical_root,
                validation_cases,
                normalization_path,
                monitoring_config,
                device,
                target_mask_policy="primary",
                autocast_dtype=autocast_dtype if device.type == "cuda" else None,
            )
            _append_jsonl(
                validation_history_path,
                {"step": completed_step, "kind": "five_phase_monitoring", "metrics": monitoring},
            )

        if cycle_due:
            latest_cycle_metrics = evaluate_complete_cycles_live(
                model,
                canonical_root,
                validation_cases,
                normalization_path,
                patch_nodes=patch_nodes,
                seed=seed,
                device=device,
                autocast_dtype=autocast_dtype,
                target_mask_policy="primary",
            )
            cycle_tawss = float(latest_cycle_metrics["pooled"]["tawss_relative_error"])
            _append_jsonl(
                validation_history_path,
                {"step": completed_step, "kind": "complete_cycle", "metrics": latest_cycle_metrics},
            )
            if cycle_tawss < best_cycle_tawss:
                best_cycle_tawss = cycle_tawss
                stale_cycle_validations = 0
                _save_checkpoint(
                    output_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=completed_step,
                    best_cycle_tawss=best_cycle_tawss,
                    stale_cycle_validations=stale_cycle_validations,
                    rng=rng,
                    resolved_manifest=resolved_manifest,
                    cycle_metrics=latest_cycle_metrics,
                )
            else:
                stale_cycle_validations += 1

        if monitoring_due or cycle_due or forced_stop:
            _save_checkpoint(
                output_dir / "latest_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=completed_step,
                best_cycle_tawss=best_cycle_tawss,
                stale_cycle_validations=stale_cycle_validations,
                rng=rng,
                resolved_manifest=resolved_manifest,
                cycle_metrics=latest_cycle_metrics,
            )

        if forced_stop:
            interrupted = {
                "status": "intentional_interruption_checkpointed",
                "resolved_manifest": resolved_manifest,
                "completed_step": completed_step,
                "resume_checkpoint": str(output_dir / "latest_checkpoint.pt"),
            }
            (output_dir / "result.json").write_text(
                json.dumps(interrupted, indent=2), encoding="utf-8"
            )
            return interrupted

        if (
            production_mode
            and completed_step >= int(optimization["minimum_steps_before_early_stopping"])
            and stale_cycle_validations
            >= int(optimization["early_stopping_patience_cycle_validations"])
        ):
            stopped_early = True
            break

    best_path = output_dir / "best_checkpoint.pt"
    if not best_path.exists():
        raise RuntimeError("training completed without a cycle-selected checkpoint")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    primary_metrics = evaluate_complete_cycles_live(
        model,
        canonical_root,
        validation_cases,
        normalization_path,
        patch_nodes=patch_nodes,
        seed=seed,
        device=device,
        autocast_dtype=autocast_dtype,
        target_mask_policy="primary",
    )
    severe_metrics = evaluate_complete_cycles_live(
        model,
        canonical_root,
        validation_cases,
        normalization_path,
        patch_nodes=patch_nodes,
        seed=seed,
        device=device,
        autocast_dtype=autocast_dtype,
        target_mask_policy="severe_sensitivity",
    )
    (output_dir / "primary_metrics.json").write_text(
        json.dumps(primary_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "severe_mask_sensitivity_metrics.json").write_text(
        json.dumps(severe_metrics, indent=2), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    peak_allocated_vram = (
        torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
    )
    vram_limit_gib = registered_vram_limit_gib(contract)
    within_registered_vram_limit = (
        peak_allocated_vram <= vram_limit_gib
        if peak_allocated_vram is not None
        else None
    )
    runtime = {
        "elapsed_seconds": elapsed,
        "last_completed_optimizer_step": last_completed_step,
        "selected_checkpoint_step": int(selected["step"]),
        "stopped_early": stopped_early,
        "peak_allocated_vram_gib": peak_allocated_vram,
        "peak_reserved_vram_gib": (
            torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else None
        ),
        "registered_vram_limit_gib": vram_limit_gib,
        "within_registered_vram_limit": within_registered_vram_limit,
        "within_frozen_22_gib_vram_limit": (
            within_registered_vram_limit if vram_limit_gib == 22.0 else None
        ),
    }
    (output_dir / "runtime_and_vram.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    result = {
        "status": "completed" if not stopped_early else "early_stopped",
        "resolved_manifest": resolved_manifest,
        "best_step": int(selected["step"]),
        "best_validation_complete_cycle_tawss_relative_error": float(
            primary_metrics["pooled"]["tawss_relative_error"]
        ),
        "severe_sensitivity_complete_cycle_tawss_relative_error": float(
            severe_metrics["pooled"]["tawss_relative_error"]
        ),
        "runtime": runtime,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if production_mode and not runtime["within_registered_vram_limit"]:
        raise RuntimeError(
            f"production run exceeded the registered {vram_limit_gib:g} GiB VRAM limit"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--patch-nodes", type=int, required=True)
    parser.add_argument("--precision", choices=("bf16_mixed", "fp16_mixed"), default="bf16_mixed")
    parser.add_argument("--precision-fallback-reason")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    result = train_watcloud_fold(
        args.contract,
        args.canonical_root,
        args.freeze_manifest,
        args.output_dir,
        fold_index=args.fold_index,
        patch_nodes=args.patch_nodes,
        precision=args.precision,
        precision_fallback_reason=args.precision_fallback_reason,
        resume_checkpoint=args.resume,
        smoke_steps=args.smoke_steps,
        allow_cpu_smoke=args.allow_cpu_smoke,
        stop_after_step=args.stop_after_step,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
