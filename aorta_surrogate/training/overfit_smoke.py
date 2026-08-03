"""Reduced one-case overfit smoke test for the local 4 GB GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from aorta_surrogate.models.gat_baseline import AorticGATBaseline
from aorta_surrogate.training.pyg_adapter import make_training_patch


def run_smoke(
    canonical_root: Path,
    output: Path,
    *,
    case_id: str = "0031_H_ABAO_AAA",
    phase_index: int = 5,
    max_nodes: int = 8192,
    steps: int = 30,
) -> dict[str, object]:
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_training_patch(
        canonical_root,
        case_id,
        phase_index,
        max_nodes=max_nodes,
    ).to(device)
    model = AorticGATBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-5)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data)
        valid = data.target_valid_mask
        vector_loss = functional.smooth_l1_loss(prediction[valid], data.y[valid], beta=0.25)
        magnitude_loss = functional.smooth_l1_loss(
            torch.linalg.vector_norm(prediction[valid], dim=-1),
            torch.linalg.vector_norm(data.y[valid], dim=-1),
            beta=0.25,
        )
        loss = vector_loss + 0.1 * magnitude_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    elapsed = time.perf_counter() - started
    result = {
        "case_id": case_id,
        "phase_index": phase_index,
        "nodes": int(data.num_nodes),
        "valid_target_nodes": int(data.target_valid_mask.sum()),
        "invalid_target_nodes": int((~data.target_valid_mask).sum()),
        "edges": int(data.edge_index.shape[1]),
        "steps": steps,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction_fraction": 1.0 - losses[-1] / losses[0],
        "elapsed_seconds": elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
        "passed": losses[-1] < losses[0] * 0.5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "result": result}, output.with_suffix(".pt"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", default="0031_H_ABAO_AAA")
    parser.add_argument("--phase-index", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run_smoke(
        args.canonical_root,
        args.output,
        case_id=args.case_id,
        phase_index=args.phase_index,
        max_nodes=args.max_nodes,
        steps=args.steps,
    ), indent=2))


if __name__ == "__main__":
    main()
