"""Shared trajectory resampling helpers for planner rollouts."""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Fine step aligned with nuPlan Diffusion-Planner (80 poses / 8 s).
DEFAULT_KINEMATICS_UPSAMPLE_DT = 0.1


def upsample_trajectory_waypoints(
    ego_np: np.ndarray,
    dt: float,
    target_dt: float,
) -> Tuple[np.ndarray, float]:
    """Linearly upsample ``[x, y, hcos, hsin]`` waypoints for kinematics.

    Consecutive coarse waypoints are assumed uniformly spaced by ``dt``.
    Returns a denser trajectory whose step evenly divides ``dt`` (e.g. 0.5 s
    coarse → 0.1 s fine). Heading components are interpolated and
    re-normalized to the unit circle.

    Invalid coarse steps (non-finite x/y/hcos/hsin) are returned unchanged.
    """
    dt = max(1e-6, float(dt))
    target_dt = max(1e-6, float(target_dt))
    ego_np = np.asarray(ego_np, dtype=np.float64)
    if ego_np.ndim != 2 or ego_np.shape[0] < 2 or target_dt >= dt:
        return ego_np, dt

    ratio = int(round(dt / target_dt))
    if ratio <= 1:
        return ego_np, dt
    fine_dt = dt / ratio

    valid = np.isfinite(ego_np[:, :4]).all(axis=1)
    if not np.all(valid):
        return ego_np, dt

    t_coarse = np.arange(int(ego_np.shape[0]), dtype=np.float64) * dt
    t_fine = np.linspace(t_coarse[0], t_coarse[-1], (t_coarse.size - 1) * ratio + 1)
    if t_fine.size < 2:
        return ego_np, dt

    out = np.empty((t_fine.size, ego_np.shape[1]), dtype=np.float64)
    for ch in range(min(4, ego_np.shape[1])):
        out[:, ch] = np.interp(t_fine, t_coarse, ego_np[:, ch])
    hnorm = np.linalg.norm(out[:, 2:4], axis=1, keepdims=True)
    out[:, 2:4] /= np.maximum(hnorm, 1e-6)
    if ego_np.shape[1] > 4:
        for ch in range(4, ego_np.shape[1]):
            out[:, ch] = np.interp(t_fine, t_coarse, ego_np[:, ch])
    return out, fine_dt
