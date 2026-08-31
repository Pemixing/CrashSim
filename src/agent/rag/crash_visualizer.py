"""Visualization and clustering utilities for crash profiles."""

import json
import math
import re
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .crash_data_processor import (
    NHTSA_CODE_NAMES,
    CrashSample,
    DT_CTRL,
    REF_TRAJ_STEPS,
    WHEELBASE,
    compute_kinematics,
)


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def ets_to_timeseries(ets: Dict[str, List[Dict]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an event_trajectory_sequence dict into two 2-D arrays:
      ego_ts:    shape (T_ego, 2)   columns [x, y]
      target_ts: shape (T_target, 2)
    """
    def _parse(seq: List[Dict]) -> np.ndarray:
        if not seq:
            return np.zeros((1, 2), dtype=float)
        return np.array(
            [[s.get("x", 0.0), s.get("y", 0.0)] for s in seq],
            dtype=float,
        )
    return _parse(ets.get("ego_trajectory_sequence", [])), _parse(ets.get("target_trajectory_sequence", []))


# 6 s pre-collision window at DT_CTRL (12 steps, collision step inclusive).
PRE_COLLISION_STEPS = 12
PRE_COLLISION_S = PRE_COLLISION_STEPS * DT_CTRL


def _traj_list_to_xy(seq: List[Dict]) -> np.ndarray:
    if not seq:
        return np.zeros((0, 2), dtype=float)
    return np.array(
        [[p.get("x", 0.0), p.get("y", 0.0)] for p in seq],
        dtype=float,
    )


def _impact_step_index_inclusive(seq: List[Dict]) -> int:
    """
    1-based index of the collision timestep in ``seq``.

    ``event_trajectory_sequence`` is already truncated to pre-impact steps when
    built; the last stored point is the endpoint closest to collision and is
    treated as the inclusive collision step (12-step window ends here).
    """
    return len(seq)


def _slice_traj_through_impact(
    seq: List[Dict],
    n_steps: int = PRE_COLLISION_STEPS,
) -> np.ndarray:
    """Last ``n_steps`` trajectory points up to and including the collision step."""
    if not seq:
        return np.zeros((1, 2), dtype=float)
    impact_idx = _impact_step_index_inclusive(seq)
    start_1 = max(1, impact_idx - n_steps + 1)
    return _traj_list_to_xy(seq[start_1 - 1 : impact_idx])


def _slice_traj_dict_through_impact(
    seq: List[Dict],
    n_steps: int = PRE_COLLISION_STEPS,
) -> List[Dict]:
    """Last ``n_steps`` trajectory dicts up to and including the collision step."""
    if not seq:
        return []
    impact_idx = _impact_step_index_inclusive(seq)
    start_1 = max(1, impact_idx - n_steps + 1)
    return list(seq[start_1 - 1 : impact_idx])


def ets_to_timeseries_pre_collision(
    ets: Dict[str, List[Dict]],
    n_steps: int = PRE_COLLISION_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ego/target (T, 2) from event_trajectory_sequence, 12 steps through collision."""
    ego_seq = ets.get("ego_trajectory_sequence") or []
    tgt_seq = ets.get("target_trajectory_sequence") or []
    return (
        _slice_traj_through_impact(ego_seq, n_steps),
        _slice_traj_through_impact(tgt_seq, n_steps),
    )


def trajectory_to_start_origin(arr: np.ndarray) -> np.ndarray:
    """Shift trajectory so the first (x, y) point is at the origin."""
    if arr.size == 0:
        return arr
    out = arr.astype(float, copy=True)
    out -= out[0]
    return out


def pad_timeseries_3d(series_list: List[np.ndarray]) -> np.ndarray:
    """
    Pad a list of 2-D arrays (T_i, F) to equal length and stack into
    a 3-D array (N, T_max, F) expected by tslearn.
    """
    max_t = max(s.shape[0] for s in series_list)
    n_feat = series_list[0].shape[1]
    out = np.zeros((len(series_list), max_t, n_feat), dtype=float)
    for i, s in enumerate(series_list):
        out[i, :s.shape[0], :] = s
    return out


def trajectory_to_kinematics(
    traj_seq: List[Dict],
    dt_ctrl: float = DT_CTRL,
    wheelbase: float = WHEELBASE,
) -> List[Dict]:
    """
    Compute acceleration and steering_angle from a trajectory sequence (x, y).
    Returns a control sequence list: [{t, acceleration, steering_angle}, ...].
    """
    if len(traj_seq) < 2:
        return [{"t": s["t"], "acceleration": 0.0, "steering_angle": 0.0} for s in traj_seq]

    x = np.array([s["x"] for s in traj_seq], dtype=float)
    y = np.array([s["y"] for s in traj_seq], dtype=float)

    dx = np.gradient(x, dt_ctrl)
    dy = np.gradient(y, dt_ctrl)
    v = np.sqrt(dx**2 + dy**2)
    psi = np.arctan2(dy, dx)

    acc, steer_deg = compute_kinematics(v, psi, dt=dt_ctrl, wheelbase=wheelbase)

    controls = []
    for i, s in enumerate(traj_seq):
        controls.append({
            "t": s["t"],
            "acceleration": round(float(acc[i]), 3),
            "steering_angle": round(float(steer_deg[i]), 3),
        })
    return controls


def reconstruct_trajectory_from_controls(
    control_seq: List[Dict],
    initial_speed: float = 8.0,
    initial_x: float = 0.0,
    initial_y: float = 0.0,
    initial_yaw: float = 0.0,
    dt_ctrl: float = DT_CTRL,
    wheelbase: float = WHEELBASE,
) -> List[Dict]:
    """
    Reconstruct an (x, y) trajectory from acceleration + steering controls.
    Initial position and heading are taken from the first frame of
    event_trajectory_sequence so the reconstruction is world-frame aligned.
    Returns [{t, x, y}, ...].
    """
    if not control_seq:
        return []

    if len(control_seq) >= 2:
        diffs = []
        for i in range(1, len(control_seq)):
            t_prev = float(control_seq[i - 1].get("t", 0.0))
            t_cur = float(control_seq[i].get("t", 0.0))
            if t_cur > t_prev:
                diffs.append(t_cur - t_prev)
        if diffs:
            dt_ctrl = float(np.median(np.array(diffs, dtype=float)))

    x = initial_x
    y = initial_y
    yaw = initial_yaw
    speed = max(0.0, float(initial_speed))
    traj: List[Dict] = []

    for i, step in enumerate(control_seq):
        t = float(step.get("t", (i + 1) * dt_ctrl))
        acc = float(step.get("acceleration", 0.0))
        steer_deg = float(step.get("steering_angle", 0.0))
        steer_rad = math.radians(steer_deg)

        # Integrate position using current (pre-update) speed and yaw, then update state.
        # This matches the forward-Euler convention used in compute_kinematics_from_trajectory.
        x += speed * math.cos(yaw) * dt_ctrl
        y += speed * math.sin(yaw) * dt_ctrl
        yaw += (speed / max(1e-6, wheelbase)) * math.tan(steer_rad) * dt_ctrl
        speed = max(0.0, speed + acc * dt_ctrl)
        traj.append({"t": round(t, 3), "x": round(x, 4), "y": round(y, 4)})

    return traj


def reconstruct_trajectory_from_deltas(
    control_seq: List[Dict],
    initial_x: float = 0.0,
    initial_y: float = 0.0,
    initial_yaw: float = 0.0,
    dt_ctrl: float = DT_CTRL,
) -> List[Dict]:
    """
    Reconstruct an (x, y) trajectory from per-step delta_s + delta_heading.
    Returns [{t, x, y}, ...].
    """
    if not control_seq:
        return []

    if len(control_seq) >= 2:
        diffs = []
        for i in range(1, len(control_seq)):
            t_prev = float(control_seq[i - 1].get("t", 0.0))
            t_cur = float(control_seq[i].get("t", 0.0))
            if t_cur > t_prev:
                diffs.append(t_cur - t_prev)
        if diffs:
            dt_ctrl = float(np.median(np.array(diffs, dtype=float)))

    x = float(initial_x)
    y = float(initial_y)
    yaw = float(initial_yaw)
    traj: List[Dict] = []

    for i, step in enumerate(control_seq):
        t = float(step.get("t", (i + 1) * dt_ctrl))
        ds = float(step.get("delta_s", 0.0))
        d_yaw = float(step.get("delta_heading", 0.0))

        # Use mid-heading when integrating arc-length increments for smoother reconstruction.
        yaw_mid = yaw + d_yaw * 0.5
        x += ds * math.cos(yaw_mid)
        y += ds * math.sin(yaw_mid)
        yaw += d_yaw
        traj.append({"t": round(t, 3), "x": round(x, 4), "y": round(y, 4)})

    return traj


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_descriptions(samples: List[CrashSample]) -> str:
    """
    Aggregate descriptions from all variants into a single primary description.
    Parses structured tags, merges unique values, and aggregates kinematic ranges.
    """
    if len(samples) == 1:
        return samples[0].description

    scenario_set: List[str] = []
    conflict_set: List[str] = []
    behavior_set: List[str] = []
    ego_acc_mins, ego_acc_maxs, ego_steer_maxs = [], [], []
    tgt_acc_mins, tgt_acc_maxs, tgt_steer_maxs = [], [], []

    for s in samples:
        for line in s.description.split("\n"):
            line = line.strip()
            if line.startswith("[scenario]"):
                val = line.split("]", 1)[1].strip()
                if val not in scenario_set:
                    scenario_set.append(val)
            elif line.startswith("[conflict]"):
                val = line.split("]", 1)[1].strip()
                if val not in conflict_set:
                    conflict_set.append(val)
            elif line.startswith("[behavior]"):
                val = line.split("]", 1)[1].strip()
                if val not in behavior_set:
                    behavior_set.append(val)
            elif line.startswith("[kinematics_ego]"):
                m_acc = re.search(r"acc_range=\[([-\d.]+),\s*([-\d.]+)\]", line)
                m_steer = re.search(r"max_steer=([\d.]+)", line)
                if m_acc:
                    ego_acc_mins.append(float(m_acc.group(1)))
                    ego_acc_maxs.append(float(m_acc.group(2)))
                if m_steer:
                    ego_steer_maxs.append(float(m_steer.group(1)))
            elif line.startswith("[kinematics_target]"):
                m_acc = re.search(r"acc_range=\[([-\d.]+),\s*([-\d.]+)\]", line)
                m_steer = re.search(r"max_steer=([\d.]+)", line)
                if m_acc:
                    tgt_acc_mins.append(float(m_acc.group(1)))
                    tgt_acc_maxs.append(float(m_acc.group(2)))
                if m_steer:
                    tgt_steer_maxs.append(float(m_steer.group(1)))

    parts = []
    if scenario_set:
        parts.append(f"[scenario] {scenario_set[0]}")
    if conflict_set:
        parts.append(f"[conflict] {'; '.join(conflict_set)}")
    if ego_acc_mins:
        parts.append(
            f"[kinematics_ego] acc_range=[{np.mean(ego_acc_mins):.2f}, {np.mean(ego_acc_maxs):.2f}]m/s², "
            f"max_steer={np.mean(ego_steer_maxs):.1f}deg (avg over {len(samples)} samples)"
        )
    if tgt_acc_mins:
        parts.append(
            f"[kinematics_target] acc_range=[{np.mean(tgt_acc_mins):.2f}, {np.mean(tgt_acc_maxs):.2f}]m/s², "
            f"max_steer={np.mean(tgt_steer_maxs):.1f}deg (avg over {len(samples)} samples)"
        )
    for b in behavior_set:
        parts.append(f"[behavior] {b}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def xy_array_to_traj_seq(arr: np.ndarray, dt: float = DT_CTRL) -> List[Dict]:
    """Convert (T, 2) x/y array to [{t, x, y}, ...] trajectory dicts."""
    return [
        {"t": round((i + 1) * dt, 1), "x": round(float(x), 4), "y": round(float(y), 4)}
        for i, (x, y) in enumerate(arr)
    ]


def _stack_trajectories_for_indices(
    ego_all: List[np.ndarray],
    tgt_all: List[np.ndarray],
    indices: np.ndarray,
    start_origin: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack ego/target trajectories for sample indices into (N, T, 2) arrays."""
    max_t_ego = max(ego_all[int(i)].shape[0] for i in indices)
    max_t_tgt = max(tgt_all[int(i)].shape[0] for i in indices)
    ego_stack = np.zeros((len(indices), max_t_ego, 2), dtype=float)
    tgt_stack = np.zeros((len(indices), max_t_tgt, 2), dtype=float)
    for li, gi in enumerate(indices):
        gi = int(gi)
        e, t = ego_all[gi], tgt_all[gi]
        if start_origin:
            e = trajectory_to_start_origin(e)
            t = trajectory_to_start_origin(t)
        ego_stack[li, :e.shape[0]] = e
        tgt_stack[li, :t.shape[0]] = t
    return ego_stack, tgt_stack


def _concat_ego_target_padded(ego_ts: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
    """Pad ego/target to equal length and concatenate features for DTW clustering."""
    max_t = max(ego_ts.shape[0], tgt_ts.shape[0])
    ego_pad = np.zeros((max_t, 2), dtype=float)
    tgt_pad = np.zeros((max_t, 2), dtype=float)
    ego_pad[:ego_ts.shape[0]] = ego_ts
    tgt_pad[:tgt_ts.shape[0]] = tgt_ts
    return np.concatenate([ego_pad, tgt_pad], axis=1)


def _build_concat_timeseries(
    samples: List[CrashSample],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Return (ego_all, tgt_all, concat_list) — one row per sample.

    Uses event_trajectory_sequence; keeps the last PRE_COLLISION_STEPS (6 s)
    points through the collision timestep (inclusive).
    """
    ego_all: List[np.ndarray] = []
    tgt_all: List[np.ndarray] = []
    concat_list: List[np.ndarray] = []
    for s in samples:
        ego_ts, tgt_ts = ets_to_timeseries_pre_collision(s.event_trajectory_sequence)
        ego_all.append(ego_ts)
        tgt_all.append(tgt_ts)
        concat_list.append(_concat_ego_target_padded(ego_ts, tgt_ts))
    return ego_all, tgt_all, concat_list


def _pick_medoid_index(
    cluster_indices: np.ndarray,
    ego_all: List[np.ndarray],
    tgt_all: List[np.ndarray],
    ts_data: Optional[np.ndarray] = None,
) -> int:
    """Pick medoid sample index; prefer concat-space distance when ts_data is given."""
    if len(cluster_indices) == 1:
        return int(cluster_indices[0])
    if ts_data is not None:
        cluster_vecs = ts_data[cluster_indices]
        centroid = cluster_vecs.mean(axis=0)
        dists = [float(np.linalg.norm(cluster_vecs[i] - centroid)) for i in range(len(cluster_indices))]
        return int(cluster_indices[int(np.argmin(dists))])
    ego_stack, tgt_stack = _stack_trajectories_for_indices(ego_all, tgt_all, cluster_indices)
    ego_centroid = ego_stack.mean(axis=0)
    tgt_centroid = tgt_stack.mean(axis=0)
    dists = [
        float(np.linalg.norm(ego_stack[li] - ego_centroid) + np.linalg.norm(tgt_stack[li] - tgt_centroid))
        for li in range(len(cluster_indices))
    ]
    return int(cluster_indices[int(np.argmin(dists))])


def _largest_cluster_label(labels: np.ndarray) -> int:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return int(unique_labels[int(np.argmax(counts))])


def build_cluster_trajectory_summary(
    samples: List[CrashSample],
    labels: np.ndarray,
    ego_all: List[np.ndarray],
    tgt_all: List[np.ndarray],
    ts_data: Optional[np.ndarray] = None,
) -> Dict:
    """
    Export only the largest cluster: medoid event's raw ego/target trajectories
    (PRE_COLLISION_STEPS through collision, 6 s at DT_CTRL).
    """
    top_label = _largest_cluster_label(labels)
    cl_idx = np.where(labels == top_label)[0]
    medoid_idx = _pick_medoid_index(cl_idx, ego_all, tgt_all, ts_data)
    medoid_ets = samples[medoid_idx].event_trajectory_sequence
    unique_labels = np.unique(labels)
    return {
        "cluster_id": top_label,
        "n_clusters": int(len(unique_labels)),
        "n_samples_in_cluster": int(len(cl_idx)),
        "medoid_event_id": samples[medoid_idx].event_id,
        "ego_trajectory_sequence": _slice_traj_dict_through_impact(
            medoid_ets.get("ego_trajectory_sequence") or [],
        ),
        "target_trajectory_sequence": _slice_traj_dict_through_impact(
            medoid_ets.get("target_trajectory_sequence") or [],
        ),
    }


def cluster_event_trajectory_sequence(
    samples: List[CrashSample],
    max_k: int = 5,
) -> Tuple[Dict, np.ndarray, object]:
    """
    Use TimeSeriesKMeans (DTW metric) to cluster event_trajectory_sequences.

    Returns (cluster_summary, labels, fitted_model).
    cluster_summary is the largest cluster only (medoid event's 12-step ego/target trajectories).
    """
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import dtw as tslearn_dtw
    from sklearn.metrics import silhouette_score

    n = len(samples)
    ego_all, tgt_all, concat_list = _build_concat_timeseries(samples)

    if n == 0:
        return {}, np.array([]), None

    if n == 1:
        labels = np.array([0])
        summary = build_cluster_trajectory_summary(samples, labels, ego_all, tgt_all, None)
        return summary, labels, None

    ts_data = pad_timeseries_3d(concat_list)

    if n == 2:
        labels = np.array([0, 0])
        summary = build_cluster_trajectory_summary(samples, labels, ego_all, tgt_all, ts_data)
        return summary, labels, None

    best_score, best_labels, best_model = -1.0, None, None
    for k in range(2, min(max_k, n - 1) + 1):
        km = TimeSeriesKMeans(
            n_clusters=k, metric="dtw", max_iter=30,
            random_state=42, n_init=3, verbose=0,
        )
        labels = km.fit_predict(ts_data)
        if len(set(labels)) < 2:
            continue
        dist_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = tslearn_dtw(ts_data[i], ts_data[j])
                dist_mat[i, j] = d
                dist_mat[j, i] = d
        score = silhouette_score(dist_mat, labels, metric="precomputed")
        if score > best_score:
            best_score, best_labels, best_model = score, labels, km

    if best_labels is None:
        km = TimeSeriesKMeans(
            n_clusters=2, metric="dtw", max_iter=30,
            random_state=42, n_init=3, verbose=0,
        )
        best_labels = km.fit_predict(ts_data)
        best_model = km

    summary = build_cluster_trajectory_summary(samples, best_labels, ego_all, tgt_all, ts_data)
    return summary, best_labels, best_model


def _per_second_point_indices(seq: List[Dict], dt_fallback: float = DT_CTRL) -> List[int]:
    """Return indices of points closest to each integer second in ``seq``."""
    if not seq:
        return []
    times = [
        float(p.get("t", (i + 1) * dt_fallback))
        for i, p in enumerate(seq)
    ]
    t_min = int(math.floor(min(times)))
    t_max = int(math.ceil(max(times)))
    indices: List[int] = []
    for sec in range(t_min, t_max + 1):
        best_i = min(range(len(times)), key=lambda i: abs(times[i] - sec))
        if abs(times[best_i] - sec) <= dt_fallback * 0.51:
            if not indices or best_i != indices[-1]:
                indices.append(best_i)
    return indices


def _plot_per_second_traj_markers(
    ax,
    seq: List[Dict],
    color: str,
    *,
    ms: float = 11.0,
    mew: float = 2.2,
    alpha: float = 0.95,
    zorder: int = 6,
    label: Optional[str] = None,
    dt_fallback: float = DT_CTRL,
) -> None:
    """Bold scatter at ~1 Hz; point spacing along the path reflects speed."""
    idx = _per_second_point_indices(seq, dt_fallback=dt_fallback)
    if not idx:
        return
    xs = [float(seq[i].get("x", 0.0)) for i in idx]
    ys = [float(seq[i].get("y", 0.0)) for i in idx]
    ax.scatter(
        xs, ys,
        c=color,
        s=ms**2,
        linewidths=mew,
        edgecolors="white",
        alpha=alpha,
        zorder=zorder,
        label=label,
    )


def _plot_per_second_xy_markers(
    ax,
    xy: np.ndarray,
    color: str,
    *,
    dt: float = DT_CTRL,
    ms: float = 10.0,
    mew: float = 2.0,
    alpha: float = 0.9,
    zorder: int = 6,
    label: Optional[str] = None,
) -> None:
    """Bold scatter on (T, 2) arrays at ~1 Hz (every ``1/dt`` steps)."""
    if xy.size == 0:
        return
    step = max(1, int(round(1.0 / dt)))
    idx = list(range(0, xy.shape[0], step))
    ax.scatter(
        xy[idx, 0], xy[idx, 1],
        c=color,
        s=ms**2,
        linewidths=mew,
        edgecolors="white",
        alpha=alpha,
        zorder=zorder,
        label=label,
    )


def _impact_time_index_from_variant(variant: Dict) -> Optional[int]:
    """
    Resolve ``impact_time_index`` for a crash_profiles variant.

    Checks ``event_trajectory_sequence``, variant root, then the first
    ``sample_control_sequence`` entry (``start_index + anchor_start + impact_time_index``,
    where ``impact_time_index`` is local to the trailing 12-step horizon).
    """
    ets = variant.get("event_trajectory_sequence") or {}
    if isinstance(ets, dict) and ets.get("impact_time_index") is not None:
        return int(ets["impact_time_index"])
    if variant.get("impact_time_index") is not None:
        return int(variant["impact_time_index"])
    for sample in variant.get("sample_control_sequence") or []:
        imp = sample.get("impact_time_index")
        if imp is None:
            continue
        ego_traj = sample.get("ego_trajectory_sequence") or []
        anchor_start = max(0, len(ego_traj) - REF_TRAJ_STEPS)
        return int(sample.get("start_index", 0)) + anchor_start + int(imp)
    return None


def _plot_impact_time_index_markers(
    ax,
    ego_seq: List[Dict],
    tgt_seq: List[Dict],
    impact_idx: int,
) -> None:
    """Mark ego/target positions at ``impact_time_index`` on the plotted trajectories."""
    if not ego_seq or not tgt_seq:
        return
    n = min(len(ego_seq), len(tgt_seq))
    if n <= 0:
        return
    plot_idx = max(0, min(int(impact_idx), n - 1))
    beyond = int(impact_idx) >= n

    ex = float(ego_seq[plot_idx].get("x", 0.0))
    ey = float(ego_seq[plot_idx].get("y", 0.0))
    tx = float(tgt_seq[plot_idx].get("x", 0.0))
    ty = float(tgt_seq[plot_idx].get("y", 0.0))
    t_imp = tgt_seq[plot_idx].get("t", plot_idx * DT_CTRL)
    dist = math.hypot(ex - tx, ey - ty)

    suffix = " (pre-impact endpoint)" if beyond else ""
    ax.plot(
        ex, ey,
        marker="*",
        color="navy",
        ms=16,
        mec="white",
        mew=0.8,
        zorder=8,
        label=f"Ego @ impact_time_index={impact_idx}{suffix}",
    )
    ax.plot(
        tx, ty,
        marker="*",
        color="darkred",
        ms=16,
        mec="white",
        mew=0.8,
        zorder=8,
        label=f"Target @ impact_time_index={impact_idx}{suffix}",
    )
    ax.plot(
        [ex, tx], [ey, ty],
        color="gold",
        ls="-",
        lw=2.0,
        alpha=0.9,
        zorder=7,
        label=f"impact_time_index span (d={dist:.2f}m, t={t_imp}s)",
    )
    ax.plot(
        tx, ty,
        marker="D",
        color="gold",
        ms=10,
        mec="k",
        mew=1.0,
        zorder=9,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def viz_cluster_trajectories(
    samples: List[CrashSample],
    labels: np.ndarray,
    nhtsa_code: str,
    save_dir: str = "out/nhtsa_rag/cluster_viz",
) -> Optional[str]:
    """
    Visualise ego and target trajectory sequences (x, y) coloured by cluster.
    Returns path to the saved figure, or None on failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(samples) == 0 or labels is None:
        return None

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    label_to_color = {int(lab): i for i, lab in enumerate(unique_labels)}
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

    largest_label = _largest_cluster_label(labels)

    fig, (ax_ego, ax_tgt) = plt.subplots(1, 2, figsize=(14, 6))

    ego_all, tgt_all, concat_list = _build_concat_timeseries(samples)
    if len(ego_all) == 0 or len(labels) != len(samples):
        return None

    ts_data = pad_timeseries_3d(concat_list) if len(samples) > 1 else None

    cluster_indices = np.where(labels == largest_label)[0]
    medoid_idx = _pick_medoid_index(cluster_indices, ego_all, tgt_all, ts_data)

    for i, _s in enumerate(samples):
        ego_ts = trajectory_to_start_origin(ego_all[i])
        tgt_ts = trajectory_to_start_origin(tgt_all[i])
        c = cmap(label_to_color[int(labels[i])])
        alpha = 0.35
        lw = 0.8
        ax_ego.plot(ego_ts[:, 0], ego_ts[:, 1], color=c, alpha=alpha, lw=lw)
        ax_tgt.plot(tgt_ts[:, 0], tgt_ts[:, 1], color=c, alpha=alpha, lw=lw)
        _plot_per_second_xy_markers(ax_ego, ego_ts, c, alpha=alpha * 0.85, ms=7, mew=1.2)
        _plot_per_second_xy_markers(ax_tgt, tgt_ts, c, alpha=alpha * 0.85, ms=7, mew=1.2)

    ego_med = trajectory_to_start_origin(ego_all[medoid_idx])
    tgt_med = trajectory_to_start_origin(tgt_all[medoid_idx])
    med_color = cmap(label_to_color[int(labels[medoid_idx])])
    ax_ego.plot(ego_med[:, 0], ego_med[:, 1], color=med_color, lw=2.5, ls="--", label="medoid (top cluster)")
    ax_tgt.plot(tgt_med[:, 0], tgt_med[:, 1], color=med_color, lw=2.5, ls="--", label="medoid (top cluster)")
    _plot_per_second_xy_markers(
        ax_ego, ego_med, med_color, ms=11, mew=2.2, label="1 Hz markers (medoid)",
    )
    _plot_per_second_xy_markers(ax_tgt, tgt_med, med_color, ms=11, mew=2.2)

    for ci, cl in enumerate(unique_labels):
        cl = int(cl)
        cl_idx = np.where(labels == cl)[0]
        ego_stack, tgt_stack = _stack_trajectories_for_indices(
            ego_all, tgt_all, cl_idx, start_origin=True,
        )
        e_mean = ego_stack.mean(axis=0)
        t_mean = tgt_stack.mean(axis=0)
        cc = cmap(ci)
        ax_ego.plot(e_mean[:, 0], e_mean[:, 1], color=cc, lw=2.0, ls="-", alpha=0.9,
                    label=f"cluster {cl} mean (n={len(cl_idx)})")
        ax_tgt.plot(t_mean[:, 0], t_mean[:, 1], color=cc, lw=2.0, ls="-", alpha=0.9,
                    label=f"cluster {cl} mean (n={len(cl_idx)})")

    nhtsa_name = NHTSA_CODE_NAMES.get(nhtsa_code, nhtsa_code)
    fig.suptitle(
        f"P{nhtsa_code.zfill(2)}: {nhtsa_name}  --  {len(samples)} samples, {n_clusters} clusters "
        f"(start-origin frame)",
        fontsize=13,
    )
    ax_ego.set_xlabel("ΔX from ego start (m)")
    ax_ego.set_ylabel("ΔY from ego start (m)")
    ax_ego.set_title("Ego (start-origin)")
    ax_ego.set_aspect("equal", adjustable="datalim")
    ax_ego.legend(fontsize=8, loc="upper right")
    ax_ego.grid(True, alpha=0.3)

    ax_tgt.set_xlabel("ΔX from target start (m)")
    ax_tgt.set_ylabel("ΔY from target start (m)")
    ax_tgt.set_title("Target (start-origin)")
    ax_tgt.set_aspect("equal", adjustable="datalim")
    ax_tgt.legend(fontsize=8, loc="upper right")
    ax_tgt.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fig_path = save_path / f"cluster_P{nhtsa_code.zfill(2)}.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Saved cluster viz: {fig_path}")
    return str(fig_path)


def viz_single_trajectory(
    json_path: str,
    save_dir: str = "out/nhtsa_rag/traj_viz",
    plot_reconstruction_traj: bool = False,
    use_control: bool = False,
) -> None:
    """
    Visualise the event_trajectory_sequence of all variants for each NHTSA class.
    Reads from crash_profiles.json.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for code, profile in data.get("profiles", {}).items():
        variants = profile.get("variants", [])
        if not variants:
            continue

        nhtsa_name = profile.get("nhtsa_type", code)

        for vi, variant in enumerate(variants):
            ets = variant.get("event_trajectory_sequence", {})
            ego_seq = ets.get("ego_trajectory_sequence", [])
            tgt_seq = ets.get("target_trajectory_sequence", [])
            ego_recon_seq: List[Dict] = []
            tgt_recon_seq: List[Dict] = []

            if plot_reconstruction_traj:
                ecs = variant.get("event_control_sequence", {})
                ego_ctrl = ecs.get("ego_control_sequence", [])
                tgt_ctrl = ecs.get("target_control_sequence", [])
                recon_init_state = variant.get("reconstruction_initial_state", {}) or {}
                recon_init_ego = recon_init_state.get("ego", {}) if isinstance(recon_init_state, dict) else {}
                recon_init_tgt = recon_init_state.get("target", {}) if isinstance(recon_init_state, dict) else {}

                # Infer explicit t=0 initial states directly from endpoint trajectory sequences.
                # NOTE: In this codebase, trajectory sequences store *endpoint* positions at
                # t = (k+1) * DT_CTRL (see crash_data_processor.py). Therefore, to obtain a
                # t=0 initial state, we back-extrapolate one step from the first 1-2 points.
                ego_init_speed_fallback = float(variant.get("ego_v_mean", 8.0) or 8.0)
                tgt_init_speed_fallback = float(variant.get("target_v_mean", 8.0) or 8.0)

                def _infer_init_from_endpoint_traj(
                    seq: List[Dict],
                    fallback_speed: float,
                    fallback_xyh: Optional[Dict] = None,
                ) -> Tuple[float, float, float, float]:
                    """
                    Given endpoint trajectory points at t=(k+1)*DT_CTRL, infer a t=0 state.
                    Returns (x0, y0, yaw0, speed0).
                    """
                    if fallback_xyh is None:
                        fallback_xyh = {}

                    def _fallback() -> Tuple[float, float, float, float]:
                        x0 = float(fallback_xyh.get("x", 0.0) or 0.0)
                        y0 = float(fallback_xyh.get("y", 0.0) or 0.0)
                        yaw0 = float(fallback_xyh.get("heading", 0.0) or 0.0)
                        spd0 = float(fallback_xyh.get("speed", fallback_speed) or fallback_speed)
                        if not math.isfinite(spd0) or spd0 <= 0.0:
                            spd0 = float(fallback_speed)
                        return x0, y0, yaw0, spd0

                    if not seq:
                        return _fallback()

                    # Determine dt using timestamps if present; otherwise default to DT_CTRL.
                    dt = float(DT_CTRL)
                    if len(seq) >= 2 and seq[0].get("t", None) is not None and seq[1].get("t", None) is not None:
                        try:
                            dt_candidate = float(seq[1]["t"]) - float(seq[0]["t"])
                            if math.isfinite(dt_candidate) and dt_candidate > 1e-9:
                                dt = float(dt_candidate)
                        except Exception:
                            pass
                    dt = max(dt, 1e-6)

                    x1 = float(seq[0].get("x", 0.0) or 0.0)
                    y1 = float(seq[0].get("y", 0.0) or 0.0)
                    if len(seq) >= 2:
                        x2 = float(seq[1].get("x", x1) or x1)
                        y2 = float(seq[1].get("y", y1) or y1)
                        # Back-extrapolate one step: p0 ≈ 2*p1 - p2
                        x0 = 2.0 * x1 - x2
                        y0 = 2.0 * y1 - y2
                    else:
                        # With only one endpoint, assume t=0 at the same position.
                        x0, y0 = x1, y1

                    dx = x1 - x0
                    dy = y1 - y0
                    yaw0 = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else float(fallback_xyh.get("heading", 0.0) or 0.0)
                    spd0 = math.hypot(dx, dy) / dt
                    if not math.isfinite(spd0) or spd0 <= 0.0:
                        spd0 = float(fallback_speed)

                    return float(x0), float(y0), float(yaw0), float(spd0)

                # Prefer trajectory-derived init; fall back to stored recon_init_* when trajectories are missing.
                ex0, ey0, eyaw0, ego_init_speed = _infer_init_from_endpoint_traj(
                    ego_seq, ego_init_speed_fallback, fallback_xyh=recon_init_ego if isinstance(recon_init_ego, dict) else {}
                )
                tx0, ty0, tyaw0, tgt_init_speed = _infer_init_from_endpoint_traj(
                    tgt_seq, tgt_init_speed_fallback, fallback_xyh=recon_init_tgt if isinstance(recon_init_tgt, dict) else {}
                )

                if use_control:
                    ego_recon_seq = reconstruct_trajectory_from_controls(
                        ego_ctrl, initial_speed=ego_init_speed,
                        initial_x=ex0, initial_y=ey0, initial_yaw=eyaw0,
                    )
                    tgt_recon_seq = reconstruct_trajectory_from_controls(
                        tgt_ctrl, initial_speed=tgt_init_speed,
                        initial_x=tx0, initial_y=ty0, initial_yaw=tyaw0,
                    )
                else:
                    ego_recon_seq = reconstruct_trajectory_from_deltas(
                        ego_ctrl,
                        initial_x=ex0,
                        initial_y=ey0,
                        initial_yaw=eyaw0,
                    )
                    tgt_recon_seq = reconstruct_trajectory_from_deltas(
                        tgt_ctrl,
                        initial_x=tx0,
                        initial_y=ty0,
                        initial_yaw=tyaw0,
                    )

            if not ego_seq and not tgt_seq and not ego_recon_seq and not tgt_recon_seq:
                continue

            fig, ax = plt.subplots(figsize=(8, 8))

            if ego_seq:
                ex = [p["x"] for p in ego_seq]
                ey = [p["y"] for p in ego_seq]
                ax.plot(ex, ey, "b-", lw=1.5, label="Ego")
                ax.plot(ex[0], ey[0], "bs", ms=8, label="Ego start")
                ax.plot(ex[-1], ey[-1], "b^", ms=8, label="Ego end")
                _plot_per_second_traj_markers(
                    ax, ego_seq, "blue", label="Ego (1 Hz)",
                )

            if tgt_seq:
                tx = [p["x"] for p in tgt_seq]
                ty = [p["y"] for p in tgt_seq]
                ax.plot(tx, ty, "r-", lw=1.5, label="Target")
                ax.plot(tx[0], ty[0], "rs", ms=8, label="Target start")
                ax.plot(tx[-1], ty[-1], "r^", ms=8, label="Target end")
                _plot_per_second_traj_markers(
                    ax, tgt_seq, "red", label="Target (1 Hz)",
                )

            if ego_recon_seq:
                rx = [p["x"] for p in ego_recon_seq]
                ry = [p["y"] for p in ego_recon_seq]
                ax.plot(rx, ry, color="cyan", ls="--", lw=1.3, label="Ego recon")
                _plot_per_second_traj_markers(ax, ego_recon_seq, "cyan", ms=9, mew=1.8)

            if tgt_recon_seq:
                rx = [p["x"] for p in tgt_recon_seq]
                ry = [p["y"] for p in tgt_recon_seq]
                ax.plot(rx, ry, color="magenta", ls="--", lw=1.3, label="Target recon")
                _plot_per_second_traj_markers(ax, tgt_recon_seq, "magenta", ms=9, mew=1.8)

            impact_state = variant.get("impact_timestamp_state", {}) or {}
            impact_ego = impact_state.get("ego", {}) if isinstance(impact_state, dict) else {}
            impact_tgt = impact_state.get("target", {}) if isinstance(impact_state, dict) else {}
            impact_time = variant.get("impact_time", None)

            ex_imp = impact_ego.get("x") if isinstance(impact_ego, dict) else None
            ey_imp = impact_ego.get("y") if isinstance(impact_ego, dict) else None
            tx_imp = impact_tgt.get("x") if isinstance(impact_tgt, dict) else None
            ty_imp = impact_tgt.get("y") if isinstance(impact_tgt, dict) else None

            if ex_imp is not None and ey_imp is not None:
                ax.plot(
                    ex_imp, ey_imp,
                    marker="*", color="navy", ms=14, mec="white", mew=0.8,
                    label=f"Ego impact state (t={impact_time}s)" if impact_time is not None else "Ego impact state",
                )

            if tx_imp is not None and ty_imp is not None:
                ax.plot(
                    tx_imp, ty_imp,
                    marker="*", color="darkred", ms=14, mec="white", mew=0.8,
                    label=f"Target impact state (t={impact_time}s)" if impact_time is not None else "Target impact state",
                )

            if (
                ex_imp is not None and ey_imp is not None
                and tx_imp is not None and ty_imp is not None
            ):
                ax.plot(
                    [ex_imp, tx_imp],
                    [ey_imp, ty_imp],
                    color="k",
                    ls=":",
                    lw=1.1,
                    alpha=0.8,
                    label="Impact ego-target span",
                )

            impact_idx = _impact_time_index_from_variant(variant)
            if impact_idx is not None and ego_seq and tgt_seq:
                _plot_impact_time_index_markers(ax, ego_seq, tgt_seq, impact_idx)

            eid = variant.get("source_event_id", "?")
            imp_title = (
                f", impact_time_index={impact_idx}"
                if impact_idx is not None
                else ""
            )
            ax.set_title(
                f"P{code.zfill(2)}: {nhtsa_name}\n(event {eid}, variant {vi}{imp_title})",
                fontsize=11,
            )
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig_path = save_path / f"traj_P{code.zfill(2)}_v{vi:03d}.png"
            fig.savefig(str(fig_path), dpi=150)
            plt.close(fig)
            print(f"  Saved trajectory viz: {fig_path}")


def viz_ref_trajectories(
    json_path: Optional[str] = None,
    profiles: Optional[Dict] = None,
    save_dir: str = "out/nhtsa_rag/ref_traj_viz",
    overlay_variants: bool = False,
) -> List[str]:
    """
    Visualise per-class representative ``ref_trajectory`` from crash_profiles.json
    or an in-memory profiles dict.

    Returns paths to saved figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if profiles is None:
        if not json_path:
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = data.get("profiles", {})

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for code, profile in profiles.items():
        ref = profile.get("ref_trajectory")
        if not ref:
            continue

        ego_seq = ref.get("ego_trajectory_sequence") or []
        tgt_seq = ref.get("target_trajectory_sequence") or []
        if not ego_seq and not tgt_seq:
            continue

        nhtsa_name = profile.get("nhtsa_type", NHTSA_CODE_NAMES.get(code, code))
        impact_idx_raw = ref.get("impact_time_index")
        impact_idx = int(impact_idx_raw if impact_idx_raw is not None else len(ego_seq) - 1)

        fig, ax = plt.subplots(figsize=(8, 8))

        if overlay_variants:
            for variant in profile.get("variants", []):
                ets = variant.get("event_trajectory_sequence", {}) or {}
                v_ego = ets.get("ego_trajectory_sequence") or []
                v_tgt = ets.get("target_trajectory_sequence") or []
                if v_ego:
                    ve = _traj_list_to_xy(v_ego)
                    ax.plot(ve[:, 0], ve[:, 1], color="steelblue", alpha=0.12, lw=0.7)
                if v_tgt:
                    vt = _traj_list_to_xy(v_tgt)
                    ax.plot(vt[:, 0], vt[:, 1], color="salmon", alpha=0.12, lw=0.7)

        if ego_seq:
            ex = [float(p.get("x", 0.0)) for p in ego_seq]
            ey = [float(p.get("y", 0.0)) for p in ego_seq]
            ax.plot(ex, ey, "b-", lw=2.0, label="Ego (ref)")
            ax.plot(ex[0], ey[0], "bs", ms=8)
            ax.plot(ex[-1], ey[-1], "b^", ms=8)
            _plot_per_second_traj_markers(ax, ego_seq, "blue", label="Ego (1 Hz)")

        if tgt_seq:
            tx = [float(p.get("x", 0.0)) for p in tgt_seq]
            ty = [float(p.get("y", 0.0)) for p in tgt_seq]
            ax.plot(tx, ty, "r-", lw=2.0, label="Target (ref)")
            ax.plot(tx[0], ty[0], "rs", ms=8)
            ax.plot(tx[-1], ty[-1], "r^", ms=8)
            _plot_per_second_traj_markers(ax, tgt_seq, "red", label="Target (1 Hz)")

        if ego_seq and tgt_seq:
            _plot_impact_time_index_markers(ax, ego_seq, tgt_seq, impact_idx)

        eid = ref.get("source_event_id", "?")
        tid = ref.get("target_id", "?")
        score = ref.get("selection_score")
        score_txt = f", score={score}" if score is not None else ""
        n_samples = profile.get("n_real_samples", "?")
        ax.set_title(
            f"P{code.zfill(2)}: {nhtsa_name}\n"
            f"ref_trajectory (event {eid}, target {tid}{score_txt})\n"
            f"{n_samples} class samples, {len(ego_seq)} steps, impact_time_index={impact_idx}",
            fontsize=11,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = save_path / f"ref_traj_P{code.zfill(2)}.png"
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        saved.append(str(fig_path))
        print(f"  Saved ref_trajectory viz: {fig_path}")

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    
    root_dir = Path(__file__).parent.parent.parent.parent

    parser = argparse.ArgumentParser(description="Visualize crash profiles.")
    parser.add_argument(
        "--json_path", type=str,
        default=str(root_dir.resolve() / "out/nhtsa_rag/crash_profiles.json"),
        help="Path to crash_profiles.json",
    )
    parser.add_argument(
        "--save_dir", type=str, default=str(root_dir.resolve() / "out/nhtsa_rag/traj_viz"),
        help="Directory to save figures (default: sibling traj_viz/ of json_path)",
    )
    parser.add_argument(
        "--plot_reconstruction_traj",
        action="store_true",
        help="If set, reconstruct trajectories from event_control_sequence and plot them.",
    )
    parser.add_argument(
            "--use_control",
            dest="use_control",
            action="store_true",
            help="Use acceleration + steering_angle.",
        )
    parser.add_argument(
        "--ref_trajectory",
        action="store_true",
        help="Visualise per-class ref_trajectory instead of all variant trajectories.",
    )
    parser.add_argument(
        "--overlay_variants",
        action="store_true",
        help="With --ref_trajectory, draw faded variant trajectories in the background.",
    )
    args = parser.parse_args()

    if args.ref_trajectory:
        ref_save_dir = str(Path(args.save_dir).parent / "ref_traj_viz") if args.save_dir else str(
            Path(args.json_path).parent / "ref_traj_viz"
        )
        viz_ref_trajectories(
            json_path=args.json_path,
            save_dir=ref_save_dir,
            overlay_variants=args.overlay_variants,
        )
        return

    save_dir = args.save_dir or str(Path(args.json_path).parent / "traj_viz")
    viz_single_trajectory(
        args.json_path,
        save_dir=save_dir,
        plot_reconstruction_traj=args.plot_reconstruction_traj,
        use_control=args.use_control,
    )


if __name__ == "__main__":
    main()
