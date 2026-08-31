"""Shared kinematic feasibility checks for adversarial trajectory generation."""

from __future__ import annotations

import math

import numpy as np
import torch

# Tighter defaults for post-hoc feasibility checks / phy_feas_kwargs_from_type_dict.
# Diffusion guidance uses looser limits in diffuser/functions.py.
DEFAULT_PHY_A_MAX = 5.5
DEFAULT_PHY_A_LAT_MAX = 5.5
DEFAULT_PHY_INFEAS_WEIGHT = 3.0
DEFAULT_PHY_MIN_DS_FOR_CURVATURE = 1.0
# Global path efficiency: path_length / net_displacement above this is treated as looping.
DEFAULT_PHY_MAX_PATH_NET_RATIO = 3.0
# High cumulative heading change with small net displacement (loopy turn without progress).
DEFAULT_PHY_YAW_DISP_MIN_YAW = math.radians(200.0)
DEFAULT_PHY_YAW_DISP_MAX_NET = 10.0
# 1-indexed step 6 -> 0-based index 5; min distance to origin from this step onward.
DEFAULT_PHY_YAW_DISP_NET_START_STEP = 5
# Fallback scale when kinematics are non-finite; ~1e5 matches typical phy_infeas_loss magnitude.
NONFINITE_PHY_INFEAS_FALLBACK_SCALE = 2e4


def add_phy_feasibility_args(parser):
    parser.add_argument(
        '--phy_a_max', type=float, default=DEFAULT_PHY_A_MAX,
        help='Max longitudinal acceleration magnitude (m/s^2) for physical feasibility.',
    )
    parser.add_argument(
        '--phy_a_lat_max', type=float, default=DEFAULT_PHY_A_LAT_MAX,
        help='Max lateral acceleration magnitude (m/s^2) for physical feasibility.',
    )
    parser.add_argument(
        '--phy_infeas_weight', type=float, default=DEFAULT_PHY_INFEAS_WEIGHT,
        help='Weight for smooth kinematic penalty during diffusion guidance.',
    )
    parser.add_argument(
        '--phy_min_ds_for_curvature', type=float, default=DEFAULT_PHY_MIN_DS_FOR_CURVATURE,
        help='Min step displacement (m) before applying lateral-acceleration checks.',
    )
    return parser


def apply_phy_feas_cfg_to_type_dict(type_dict, cfg):
    type_dict['phy_a_max'] = getattr(cfg, 'phy_a_max', DEFAULT_PHY_A_MAX)
    type_dict['phy_a_lat_max'] = getattr(cfg, 'phy_a_lat_max', DEFAULT_PHY_A_LAT_MAX)
    type_dict['phy_infeas_weight'] = getattr(cfg, 'phy_infeas_weight', DEFAULT_PHY_INFEAS_WEIGHT)
    type_dict['phy_min_ds_for_curvature'] = getattr(
        cfg, 'phy_min_ds_for_curvature', DEFAULT_PHY_MIN_DS_FOR_CURVATURE)
    return type_dict


def phy_feas_kwargs_from_type_dict(type_dict):
    return {
        'a_max': type_dict.get('phy_a_max', DEFAULT_PHY_A_MAX),
        'a_lat_max': type_dict.get('phy_a_lat_max', DEFAULT_PHY_A_LAT_MAX),
        'weight': type_dict.get('phy_infeas_weight', DEFAULT_PHY_INFEAS_WEIGHT),
        'min_ds_for_curvature': type_dict.get(
            'phy_min_ds_for_curvature', DEFAULT_PHY_MIN_DS_FOR_CURVATURE),
        'max_path_net_ratio': type_dict.get('phy_max_path_net_ratio', DEFAULT_PHY_MAX_PATH_NET_RATIO),
        'yaw_disp_min_yaw': type_dict.get('phy_yaw_disp_min_yaw', DEFAULT_PHY_YAW_DISP_MIN_YAW),
        'yaw_disp_max_net': type_dict.get('phy_yaw_disp_max_net', DEFAULT_PHY_YAW_DISP_MAX_NET),
        'yaw_disp_net_start_step': type_dict.get(
            'phy_yaw_disp_net_start_step', DEFAULT_PHY_YAW_DISP_NET_START_STEP),
    }


def _wrap_pi_torch(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _yaw_disp_min_net_torch(pos, start_step=DEFAULT_PHY_YAW_DISP_NET_START_STEP):
    """Min distance to initial position from 1-indexed step 6 (0-based index start_step) onward."""
    xy = pos[..., :2]
    origin = xy[:, :1]
    if xy.shape[1] <= start_step:
        return torch.linalg.norm(xy[:, -1] - xy[:, 0], dim=-1)
    dists = torch.linalg.norm(xy[:, start_step:] - origin, dim=-1)
    return dists.min(dim=-1).values


def _cum_heading_change_torch(pos):
    """Sum of |Δheading| over the trajectory."""
    if pos.shape[-1] >= 4:
        h = pos[..., 2:4]
        heading = torch.atan2(h[..., 1], h[..., 0])
        dtheta = _wrap_pi_torch(heading[:, 1:] - heading[:, :-1])
        return dtheta.abs().sum(dim=-1)

    xy = pos[..., :2]
    step_dxy = xy[:, 1:] - xy[:, :-1]
    vel_heading = torch.atan2(step_dxy[..., 1], step_dxy[..., 0] + 1e-8)
    if vel_heading.shape[1] < 2:
        return vel_heading.new_zeros(vel_heading.shape[0])
    dtheta = _wrap_pi_torch(vel_heading[:, 1:] - vel_heading[:, :-1])
    return dtheta.abs().sum(dim=-1)


def _path_net_ratio_torch(pos):
    """path_length / net_displacement for each agent."""
    xy = pos[..., :2]
    step_ds = torch.linalg.norm(xy[:, 1:] - xy[:, :-1], dim=-1)
    path_length = step_ds.sum(dim=-1)
    net_disp = torch.linalg.norm(xy[:, -1] - xy[:, 0], dim=-1)
    return path_length / (net_disp + 0.5)


def _compute_path_efficiency_penalty_torch(pos, max_path_net_ratio=DEFAULT_PHY_MAX_PATH_NET_RATIO):
    """Penalty for loopy trajectories with high path_length / net_displacement."""
    if pos.shape[1] < 3:
        return None
    path_net_ratio = _path_net_ratio_torch(pos)
    return [torch.relu(path_net_ratio - max_path_net_ratio).pow(2)]


def _compute_yaw_disp_penalty_torch(
    pos,
    yaw_disp_min_yaw=DEFAULT_PHY_YAW_DISP_MIN_YAW,
    yaw_disp_max_net=DEFAULT_PHY_YAW_DISP_MAX_NET,
    yaw_disp_net_start_step=DEFAULT_PHY_YAW_DISP_NET_START_STEP,
):
    """Penalty when cumulative |Δheading| is high but min net progress stays small."""
    if pos.shape[1] < 3:
        return None
    cum_yaw = _cum_heading_change_torch(pos)
    min_net = _yaw_disp_min_net_torch(pos, start_step=yaw_disp_net_start_step)
    yaw_excess = torch.relu(cum_yaw - yaw_disp_min_yaw)
    disp_shortfall = torch.relu(yaw_disp_max_net - min_net)
    return [(yaw_excess * disp_shortfall).pow(2)]


def _has_excessive_yaw_small_disp_torch(
    pos,
    yaw_disp_min_yaw=DEFAULT_PHY_YAW_DISP_MIN_YAW,
    yaw_disp_max_net=DEFAULT_PHY_YAW_DISP_MAX_NET,
    yaw_disp_net_start_step=DEFAULT_PHY_YAW_DISP_NET_START_STEP,
):
    """Return per-agent bool: True if yaw-vs-displacement profile is acceptable."""
    if pos.shape[1] < 3:
        return torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    cum_yaw = _cum_heading_change_torch(pos)
    min_net = _yaw_disp_min_net_torch(pos, start_step=yaw_disp_net_start_step)
    return ~((cum_yaw >= yaw_disp_min_yaw) & (min_net < yaw_disp_max_net))


def _has_loopy_path_torch(pos, max_path_net_ratio=DEFAULT_PHY_MAX_PATH_NET_RATIO):
    """Return per-agent bool: True if path efficiency is acceptable."""
    if pos.shape[1] < 3:
        return torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    return _path_net_ratio_torch(pos) <= max_path_net_ratio


def _compute_kinematics_torch(pos, dt, min_ds_for_curvature=DEFAULT_PHY_MIN_DS_FOR_CURVATURE):
    pos_xy = pos[..., :2]
    if pos_xy.shape[1] < 3:
        return None, None, None

    vel = (pos_xy[:, 1:] - pos_xy[:, :-1]) / dt
    if vel.shape[1] < 2:
        return None, None, None

    acc = (vel[:, 1:] - vel[:, :-1]) / dt
    acc_norm = torch.linalg.norm(acc, dim=-1)

    v1 = vel[:, :-1]
    v2 = vel[:, 1:]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(dim=-1) + 1e-6
    angle = torch.atan2(cross, dot)
    speed = torch.linalg.norm(v1, dim=-1) + 1e-6
    curvature = angle / (dt * speed)
    a_lat = (speed ** 2) * curvature

    seg_ds = torch.linalg.norm(pos_xy[:, 1:] - pos_xy[:, :-1], dim=-1)
    curv_mask = seg_ds[:, :-1] >= min_ds_for_curvature
    return acc_norm, a_lat, curv_mask


def check_physical_feasibility(
    pos,
    dt,
    a_max=DEFAULT_PHY_A_MAX,
    a_lat_max=DEFAULT_PHY_A_LAT_MAX,
    min_ds_for_curvature=DEFAULT_PHY_MIN_DS_FOR_CURVATURE,
    max_path_net_ratio=DEFAULT_PHY_MAX_PATH_NET_RATIO,
    yaw_disp_min_yaw=DEFAULT_PHY_YAW_DISP_MIN_YAW,
    yaw_disp_max_net=DEFAULT_PHY_YAW_DISP_MAX_NET,
    yaw_disp_net_start_step=DEFAULT_PHY_YAW_DISP_NET_START_STEP,
):
    acc_norm, a_lat, curv_mask = _compute_kinematics_torch(
        pos, dt, min_ds_for_curvature=min_ds_for_curvature)
    if acc_norm is None:
        return True

    acc_ok = acc_norm.max(dim=-1).values <= a_max

    a_lat_abs = a_lat.abs()
    masked_a_lat = torch.where(curv_mask, a_lat_abs, torch.zeros_like(a_lat_abs))
    has_curv = curv_mask.any(dim=-1)
    a_lat_max_per_agent = masked_a_lat.max(dim=-1).values
    a_lat_ok = (~has_curv) | (a_lat_max_per_agent <= a_lat_max)

    path_ok = _has_loopy_path_torch(pos, max_path_net_ratio=max_path_net_ratio)
    yaw_disp_ok = _has_excessive_yaw_small_disp_torch(
        pos,
        yaw_disp_min_yaw=yaw_disp_min_yaw,
        yaw_disp_max_net=yaw_disp_max_net,
        yaw_disp_net_start_step=yaw_disp_net_start_step,
    )
    return (acc_ok & a_lat_ok & path_ok & yaw_disp_ok).all().item()


def phy_infeas_penalty(
    trajs,
    atk_mask,
    dt,
    a_max=DEFAULT_PHY_A_MAX,
    a_lat_max=DEFAULT_PHY_A_LAT_MAX,
    weight=DEFAULT_PHY_INFEAS_WEIGHT,
    min_ds_for_curvature=DEFAULT_PHY_MIN_DS_FOR_CURVATURE,
    max_path_net_ratio=DEFAULT_PHY_MAX_PATH_NET_RATIO,
    yaw_disp_min_yaw=DEFAULT_PHY_YAW_DISP_MIN_YAW,
    yaw_disp_max_net=DEFAULT_PHY_YAW_DISP_MAX_NET,
    yaw_disp_net_start_step=DEFAULT_PHY_YAW_DISP_NET_START_STEP,
):
    # Guidance should never return NaN; convert non-finite kinematics into a large penalty
    # instead of letting it poison the full guidance objective.
    if not isinstance(dt, (float, int)):
        try:
            dt = float(dt)
        except Exception:
            dt = None
    if dt is None or not math.isfinite(dt) or dt <= 0.0:
        return trajs.new_tensor(0.0)

    pos = trajs[atk_mask]
    if pos.shape[0] == 0:
        return trajs.new_tensor(0.0)
    if not torch.isfinite(pos).all():
        # If the trajectory itself already contains NaNs/Infs, return a large penalty
        # to push the optimizer away from this region, while keeping the scalar finite.
        return trajs.new_tensor(float(weight) * NONFINITE_PHY_INFEAS_FALLBACK_SCALE)

    acc_norm, a_lat, curv_mask = _compute_kinematics_torch(
        pos, dt, min_ds_for_curvature=min_ds_for_curvature)
    if acc_norm is None:
        return trajs.new_tensor(0.0)

    acc_excess = torch.relu(acc_norm - a_max)
    a_lat_excess = torch.relu(a_lat.abs() - a_lat_max)
    a_lat_excess = torch.where(curv_mask, a_lat_excess, torch.zeros_like(a_lat_excess))
    penalty = acc_excess.pow(2).mean() + a_lat_excess.pow(2).mean()

    path_penalties = _compute_path_efficiency_penalty_torch(
        pos, max_path_net_ratio=max_path_net_ratio)
    if path_penalties is not None:
        for term in path_penalties:
            penalty = penalty + term.mean()

    yaw_disp_penalties = _compute_yaw_disp_penalty_torch(
        pos,
        yaw_disp_min_yaw=yaw_disp_min_yaw,
        yaw_disp_max_net=yaw_disp_max_net,
        yaw_disp_net_start_step=yaw_disp_net_start_step,
    )
    if yaw_disp_penalties is not None:
        for term in yaw_disp_penalties:
            penalty = penalty + term.mean()

    out = weight * penalty
    # Final safety: guarantee a finite scalar even if upstream math produced NaNs/Infs.
    if not torch.isfinite(out):
        fallback = float(weight) * NONFINITE_PHY_INFEAS_FALLBACK_SCALE
        out = torch.nan_to_num(out, nan=fallback, posinf=fallback, neginf=fallback)
    return out


def _compute_kinematics_numpy(pos, dt, min_ds_for_curvature=DEFAULT_PHY_MIN_DS_FOR_CURVATURE):
    if pos.ndim == 3:
        pos = pos[0]
    pos = np.asarray(pos, dtype=np.float64)[..., :2]
    if pos.shape[0] < 3:
        return None, None, None

    vel = (pos[1:] - pos[:-1]) / dt
    if vel.shape[0] < 2:
        return None, None, None

    acc = (vel[1:] - vel[:-1]) / dt
    acc_norm = np.linalg.norm(acc, axis=-1)

    v1, v2 = vel[:-1], vel[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(axis=-1) + 1e-6
    angle = np.arctan2(cross, dot)
    speed = np.linalg.norm(v1, axis=-1) + 1e-6
    curvature = angle / (dt * speed)
    a_lat = (speed ** 2) * np.abs(curvature)

    seg_ds = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    curv_mask = seg_ds[:-1] >= min_ds_for_curvature
    return acc_norm, a_lat, curv_mask


def check_physical_feasibility_xy(
    pos,
    dt,
    a_max=DEFAULT_PHY_A_MAX,
    a_lat_max=DEFAULT_PHY_A_LAT_MAX,
    min_ds_for_curvature=DEFAULT_PHY_MIN_DS_FOR_CURVATURE,
):
    acc_norm, a_lat, curv_mask = _compute_kinematics_numpy(
        pos, dt, min_ds_for_curvature=min_ds_for_curvature)
    if acc_norm is None:
        return True, np.array([]), np.array([])

    acc_ok = acc_norm.max() <= a_max
    if curv_mask.any():
        a_lat_ok = a_lat[curv_mask].max() <= a_lat_max
    else:
        a_lat_ok = True
    return bool(acc_ok and a_lat_ok), acc_norm, a_lat
