"""Shared validation helpers for frozen cloud-execution contracts."""

from __future__ import annotations


def registered_gpu_names(contract: dict[str, object]) -> tuple[str, ...]:
    hardware = contract["hardware"]
    configured = hardware.get("accepted_gpu_name_substrings")
    names = tuple(str(name).strip() for name in (configured or [hardware["target_gpu"]]))
    if not names or any(not name for name in names):
        raise ValueError("hardware contract must register at least one GPU name")
    return names


def validate_registered_gpu(contract: dict[str, object], actual_name: str) -> None:
    expected = registered_gpu_names(contract)
    normalized_actual = actual_name.casefold()
    if not any(name.casefold() in normalized_actual for name in expected):
        raise RuntimeError(
            f"experiment {contract['experiment_id']} requires a registered GPU "
            f"matching {list(expected)}, got {actual_name!r}"
        )


def registered_vram_limit_gib(contract: dict[str, object]) -> float:
    hardware = contract["hardware"]
    limit = float(hardware.get("maximum_allocated_vram_gib", 22.0))
    if not 0.0 < limit:
        raise ValueError("maximum_allocated_vram_gib must be positive")
    return limit
