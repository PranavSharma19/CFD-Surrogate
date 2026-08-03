"""Deterministic surface-hemodynamic metrics derived from transient WSS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class HemodynamicMetrics:
    """Per-node metrics for one complete cardiac cycle."""

    tawss: FloatArray
    osi: FloatArray
    rrt: FloatArray
    rrt_valid: npt.NDArray[np.bool_]


def _validate_cycle(
    wss: npt.ArrayLike,
    times: npt.ArrayLike | None,
) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(wss, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("wss must have shape [phase, node, xyz]")
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("wss must contain at least two phases and one node")
    if not np.isfinite(values).all():
        raise ValueError("wss contains NaN or infinite values")

    if times is None:
        sample_times = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    else:
        sample_times = np.asarray(times, dtype=np.float64)
        if sample_times.shape != (values.shape[0],):
            raise ValueError("times must contain one value per WSS phase")
        if not np.isfinite(sample_times).all():
            raise ValueError("times contains NaN or infinite values")
        if np.any(np.diff(sample_times) <= 0.0):
            raise ValueError("times must be strictly increasing")

    if sample_times[-1] <= sample_times[0]:
        raise ValueError("the cardiac-cycle duration must be positive")
    return values, sample_times


def compute_hemodynamic_metrics(
    wss: npt.ArrayLike,
    times: npt.ArrayLike | None = None,
    *,
    tawss_floor_pa: float = 1.0e-4,
    rrt_denominator_floor: float = 1.0e-8,
) -> HemodynamicMetrics:
    """Compute TAWSS, OSI, and RRT from one complete WSS cycle.

    Parameters
    ----------
    wss:
        WSS vectors in Pa with shape ``[phase, node, xyz]``.
    times:
        Strictly increasing time samples. Uniform normalized time is used when
        omitted. Callers must include or reconstruct a complete cycle.
    tawss_floor_pa:
        Nodes below this TAWSS are marked invalid for RRT reporting.
    rrt_denominator_floor:
        Numerical floor for the RRT denominator.
    """

    if tawss_floor_pa < 0.0 or rrt_denominator_floor <= 0.0:
        raise ValueError("metric stability floors must be non-negative")

    values, sample_times = _validate_cycle(wss, times)
    duration = float(sample_times[-1] - sample_times[0])
    magnitudes = np.linalg.norm(values, axis=-1)

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    magnitude_integral = trapezoid(magnitudes, sample_times, axis=0)
    vector_integral = trapezoid(values, sample_times, axis=0)
    tawss = magnitude_integral / duration

    vector_integral_magnitude = np.linalg.norm(vector_integral, axis=-1)
    stable_magnitude = magnitude_integral > (tawss_floor_pa * duration)
    ratio = np.divide(
        vector_integral_magnitude,
        magnitude_integral,
        out=np.zeros_like(magnitude_integral),
        where=stable_magnitude,
    )
    osi = np.clip(0.5 * (1.0 - ratio), 0.0, 0.5)

    denominator = (1.0 - 2.0 * osi) * tawss
    rrt_valid = stable_magnitude & (denominator > rrt_denominator_floor)
    rrt = np.full_like(tawss, np.nan)
    np.divide(1.0, denominator, out=rrt, where=rrt_valid)

    return HemodynamicMetrics(
        tawss=tawss,
        osi=osi,
        rrt=rrt,
        rrt_valid=rrt_valid,
    )
