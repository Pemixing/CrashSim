"""
Process T7UUC1 real crash trajectory data into NHTSA 21-class profiles.

Reads event_data.h5 + event_meta.csv or event_data.csv, extracts impact-centered windows,
computes kinematics, classifies into 21 NHTSA categories, and outputs to JSON.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT_RAW = 0.1          # 10 Hz raw data
DT_CTRL = 0.5         # control-sequence time step
SAMPLE_WINDOW_S = 8.0  # fixed-length sample window for RAG / policy (16 steps at DT_CTRL)
SAMPLE_WINDOW_STEPS = int(round(SAMPLE_WINDOW_S / DT_CTRL))
# Within each sliding window, snapshot ego/target heading & speed at this offset from window start.
SAMPLE_STATE_T_S = 2.0
# Mean speed in sample_control_sequence items: average raw speed over this many seconds from window start.
SAMPLE_V_MEAN_FIRST_S = 2.0
REF_TRAJ_STEPS = 12
REF_TRAJ_WINDOW_S = REF_TRAJ_STEPS * DT_CTRL
WHEELBASE = 2.8        # metres (bicycle model)
# Impact-centered slice: use up to [impact-10s, impact+7s], require coverage of [impact-3s, impact+7s]
PRE_IMPACT_MAX_S = 8.0
PRE_IMPACT_MIN_S = 3.0
POST_IMPACT_S = 5.0
WINDOW_TIME_TOL_S = 0.01
MIN_RAW_SAMPLES_IN_WINDOW = 16
TRAJ_SMOOTH_WINDOW = 5          # smoothing window (raw samples, odd size preferred)
TRAJ_MAX_STEP_DT_S = 0.5        # drop segments with very large timestamp gaps
TRAJ_MAX_STEP_SPEED_MPS = 50.0  # drop segments with implausible jumps
TRAJ_MAX_LATERAL_ACCEL_MPS2 = 12.0   # drop segments with excessive lateral acceleration
TRAJ_MAX_CURVATURE_INV_M = 1.0  # drop segments with implausibly sharp local turns
# Stationary spin: need longer window + moderate yaw threshold to catch slow pirouettes
# (short 0.5s / 90° missed slow turns; tight circles still need curvature with smaller min_ds).
TRAJ_SPIN_WIN_S = 1.0                # time window for in-place spin / drift detection (s)
TRAJ_SPIN_MAX_DISP_M = 1.0           # net displacement threshold inside spin window (m)
TRAJ_SPIN_MAX_YAW_RAD = math.pi / 2  # ~60° cumulative |Δheading| inside spin window
TRAJ_MIN_DS_FOR_CURVATURE_M = 1.5    # min local arc length before applying curvature / a_lat checks

MAX_LONGITUDINAL_M = 12.0
MAX_LATERAL_M = 5.0
# Opposite-direction detection (relaxed to retain more NHTSA classes 3 and 8).
OPPOSITE_COS_ALIGN_MAX = math.cos(math.radians(145.0))  # velocity angle >= 145°
OPPOSITE_REL_HEADING_MIN_RAD = math.radians(145.0)  # ~2.53 rad
OPPOSITE_MIN_SPEED_MPS = 0.15
# Junction refinement (relaxed turn / lateral gates for classes 3 and 8).
JUNCTION_TURN_STRONG_RAD = 0.30
JUNCTION_LAT_MIN_M = 1.2
JUNCTION_LEFT_TURN_MIN_RAD = 0.25
JUNCTION_STRAIGHT_CROSS_MAX_TURN_RAD = math.radians(20.0)
# Event features: fraction of samples with a <= this threshold count as "hard decel"
# (aligned with ``build_auto_description`` hard-braking heuristics).
EVENT_HARD_DECEL_ACC_MS2 = -2.0
# Minimum OBB-OBB distance (m) between ego and target boxes that counts as
# a collision when locating the impact moment from raw trajectories.
VEHICLE_COLLISION_DISTANCE_THRESH_M = 0.1
# Ref-trajectory target pre-impact action: mild brake / small evasive (not emergency-only).
REF_BRAKE_ACC_MS2 = -1.0
REF_EMERGENCY_DECEL_ACC_MS2 = REF_BRAKE_ACC_MS2  # legacy alias
REF_EVASIVE_STEER_DEG = 1.5
REF_EVASIVE_LATERAL_SHIFT_M = 0.25
REF_MAX_ABS_ACC_MS2 = 6.0
REF_MAX_ABS_STEER_DEG = 45.0
REF_MAX_ABS_JERK_MS3 = 8.0
REF_MAX_STEER_RATE_DEGPS = 120.0
REF_MAX_STEP_SPEED_MPS = 50.0

STAGE_INTENTION_FORMATION = "intention_formation"
STAGE_RISK_AWARENESS = "risk_awareness"
STAGE_EMERGENCY_RESPONSE = "emergency_response"
STAGE_COLLISION = "collision"
STAGE_NAME_BY_ID = {
    1: STAGE_INTENTION_FORMATION,
    2: STAGE_RISK_AWARENESS,
    3: STAGE_EMERGENCY_RESPONSE,
    4: STAGE_COLLISION,
}
# Stages 2–3 only: stage 1 always starts at index 0; collision uses impact_time_index.
STAGE_TIME_INDEX_NAMES = (
    STAGE_RISK_AWARENESS,
    STAGE_EMERGENCY_RESPONSE,
)
# Kinematic gates for classify_interaction_stages (relaxed vs. early strict TTC-only rules).
STAGE2_TTC_MAX_S = 5.0
STAGE2_CLOSING_MIN_MPS = 1.25
STAGE2_DIST_FRAC_OF_D0 = 0.75
STAGE3_TTC_MAX_S = 4.0
STAGE3_CLOSING_MIN_MPS = 2.5
STAGE3_DIST_FRAC_OF_D0 = 0.50
STAGE3_DECEL_MIN_MS2 = -1.5
STAGE3_STRONG_DECEL_MS2 = -2.5

NHTSA_CODE_NAMES = {
    "1": "Lead Vehicle Stopped",                                            # 27.85%
    "2": "Lead Vehicle Decelerating",                                       # 12.22%
    "3": "Left Turn Across Path From Opposite Directions at Junction",      # 11.71%
    "4": "Vehicle(s) Changing Lanes - Same Direction",                      # 9.65%
    "5": "Straight Crossing Paths at Junction",                             # 7.53%
    "6": "Vehicle(s) Turning - Same Direction",                             # 6.33%
    "7": "Lead Vehicle Moving at Lower Constant Speed",                     # 5.99%
    "8": "Vehicle(s) - Opposite Direction",                                 # 3.97%
    "9": "Backing Up Into Another Vehicle",                                 # 3.73%
    "10": "Vehicle(s) Drifting - Same Direction",                           # 2.80%
    "11": "Following Vehicle Making a Maneuver",                            # 2.44%
    "12": "Evasive Action",                                                 # 1.98%
    "13": "Vehicle(s) Parking - Same Direction",                            # 1.37%
    "14": "Vehicle Turning Right at Junction",                              # 1.00%
    "15": "Lead Vehicle Accelerating",                                      # 0.39%
    "16": "Other",                                                          # 1.04%
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EventMeta:
    event_id: int
    first: str
    conflict: str
    ego_width: float
    ego_length: float
    target_width: float
    target_length: float
    start_timestamp: float
    end_timestamp: float
    impact_timestamp: float
    severity: float
    duration_enough: bool


@dataclass
class EventTimeSeries:
    """Per-target time series extracted from the HDF5 store."""
    event_id: int
    target_id: int
    time: np.ndarray       # shape (N,)
    x_ego: np.ndarray
    y_ego: np.ndarray
    v_ego: np.ndarray
    psi_ego: np.ndarray
    acc_ego: np.ndarray
    x_sur: np.ndarray
    y_sur: np.ndarray
    v_sur: np.ndarray
    psi_sur: np.ndarray


@dataclass
class ImpactKinematics:
    """Ego-frame geometry and pre-impact turn cues at the impact instant."""
    lon: float
    lat: float
    rel_h: float
    cos_align: float
    ego_heading_change: float
    sur_heading_change: float
    ego_abs_heading_change: float
    sur_abs_heading_change: float
    ego_mean_signed_yaw: float
    sur_mean_signed_yaw: float


@dataclass
class CrashSample:
    """One classified crash sample ready for export."""
    event_id: int
    target_id: int
    ego_width: float
    ego_length: float
    target_width: float
    target_length: float
    severity: float
    nhtsa_code: str
    nhtsa_type: str
    description: str
    event_control_sequence: Dict[str, List[Dict]]
    event_trajectory_sequence: Dict[str, List[Dict]]
    sample_control_sequence: List[Dict[str, object]]
    reconstruction_initial_state: Dict[str, Dict[str, Optional[float]]]
    impact_time: Optional[float]
    impact_timestamp_state: Dict[str, Dict[str, Optional[float]]]
    impact_timestamp_mdc: Optional[float]
    event_feature: Dict[str, Optional[float]]
    # True only when impact_time came from min-box-distance<=threshold collision in raw data.
    is_crash: bool


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

def load_event_meta(csv_path: str) -> Dict[int, EventMeta]:
    df = pd.read_csv(csv_path)
    meta_map: Dict[int, EventMeta] = {}
    for _, row in df.iterrows():
        eid = int(row["event_id"])
        meta_map[eid] = EventMeta(
            event_id=eid,
            first=str(row.get("first", "")).strip(),
            conflict=str(row.get("conflict", "")).strip(),
            ego_width=float(row.get("ego_width", 1.8)),
            ego_length=float(row.get("ego_length", 4.5)),
            target_width=float(row.get("target_width", 1.8) or 1.8),
            target_length=float(row.get("target_length", 4.5) or 4.5),
            start_timestamp=float(row.get("start_timestamp", 0)),
            end_timestamp=float(row.get("end_timestamp", 0)),
            impact_timestamp=float(row.get("impact_timestamp", 0)),
            severity=float(row.get("severity_first", 0) or 0),
            duration_enough=bool(row.get("duration_enough", False)),
        )
    return meta_map


def load_event_data(h5_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return (float_data, int_data, float_columns, int_columns)."""
    with h5py.File(h5_path, "r") as f:
        grp = f["data"]
        float_cols = [x.decode() for x in grp["block0_items"][:]]
        int_cols = [x.decode() for x in grp["block1_items"][:]]
        float_data = grp["block0_values"][:]
        int_data = grp["block1_values"][:]
    return float_data, int_data, float_cols, int_cols


def extract_event_timeseries(
    event_id: int,
    float_data: np.ndarray,
    int_data: np.ndarray,
    float_cols: List[str],
    int_cols: List[str],
) -> List[EventTimeSeries]:
    """Extract per-target time series for one event."""
    eid_col = int_cols.index("event_id")
    tid_col = int_cols.index("target_id")
    mask = int_data[:, eid_col] == event_id
    if not mask.any():
        return []

    ev_float = float_data[mask]
    ev_int = int_data[mask]
    col = {name: i for i, name in enumerate(float_cols)}

    results = []
    for tid in np.unique(ev_int[:, tid_col]):
        tmask = ev_int[:, tid_col] == tid
        tf = ev_float[tmask]
        sort_idx = np.argsort(tf[:, col["time"]])
        tf = tf[sort_idx]
        results.append(EventTimeSeries(
            event_id=event_id,
            target_id=int(tid),
            time=tf[:, col["time"]],
            x_ego=tf[:, col["x_ego"]],
            y_ego=tf[:, col["y_ego"]],
            v_ego=tf[:, col["v_ego"]],
            psi_ego=tf[:, col["psi_ego"]],
            acc_ego=tf[:, col["acc_ego"]],
            x_sur=tf[:, col["x_sur"]],
            y_sur=tf[:, col["y_sur"]],
            v_sur=tf[:, col["v_sur"]],
            psi_sur=tf[:, col["psi_sur"]],
        ))
    return results


# ---------------------------------------------------------------------------
# Time window extraction
# ---------------------------------------------------------------------------

def _slice_event_timeseries(ts: EventTimeSeries, selector) -> EventTimeSeries:
    """Return a shallow-sliced copy of an EventTimeSeries."""
    return EventTimeSeries(
        event_id=ts.event_id,
        target_id=ts.target_id,
        time=ts.time[selector],
        x_ego=ts.x_ego[selector],
        y_ego=ts.y_ego[selector],
        v_ego=ts.v_ego[selector],
        psi_ego=ts.psi_ego[selector],
        acc_ego=ts.acc_ego[selector],
        x_sur=ts.x_sur[selector],
        y_sur=ts.y_sur[selector],
        v_sur=ts.v_sur[selector],
        psi_sur=ts.psi_sur[selector],
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3 or window <= 1:
        return arr.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > len(arr):
        w = len(arr) if (len(arr) % 2 == 1) else (len(arr) - 1)
    if w <= 1:
        return arr.copy()
    pad = w // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(arr_pad, kernel, mode="valid")


def _smooth_xy_trajectories(
    ts: EventTimeSeries,
    smooth_window: int = TRAJ_SMOOTH_WINDOW,
) -> EventTimeSeries:
    """Apply light smoothing on ego/target XY trajectories only."""
    return EventTimeSeries(
        event_id=ts.event_id,
        target_id=ts.target_id,
        time=ts.time.copy(),
        x_ego=_moving_average(ts.x_ego, smooth_window),
        y_ego=_moving_average(ts.y_ego, smooth_window),
        v_ego=ts.v_ego.copy(),
        psi_ego=ts.psi_ego.copy(),
        acc_ego=ts.acc_ego.copy(),
        x_sur=_moving_average(ts.x_sur, smooth_window),
        y_sur=_moving_average(ts.y_sur, smooth_window),
        v_sur=ts.v_sur.copy(),
        psi_sur=ts.psi_sur.copy(),
    )


def _truncate_low_quality_segment(
    ts: EventTimeSeries,
    impact_time: float,
    time_tol_s: float = WINDOW_TIME_TOL_S,
    max_step_dt_s: float = TRAJ_MAX_STEP_DT_S,
    max_step_speed_mps: float = TRAJ_MAX_STEP_SPEED_MPS,
    max_lateral_accel_mps2: float = TRAJ_MAX_LATERAL_ACCEL_MPS2,
    max_curvature_inv_m: float = TRAJ_MAX_CURVATURE_INV_M,
    spin_win_s: float = TRAJ_SPIN_WIN_S,
    spin_max_disp_m: float = TRAJ_SPIN_MAX_DISP_M,
    spin_max_yaw_rad: float = TRAJ_SPIN_MAX_YAW_RAD,
    min_ds_for_curvature_m: float = TRAJ_MIN_DS_FOR_CURVATURE_M,
) -> Optional[EventTimeSeries]:
    """
    Truncate noisy/outlier parts and keep one continuous segment around impact.

    Quality checks are based on:
      1) timestamp gaps / invalid timestamps
      2) non-finite positions
      3) implausible per-step position jumps
      4) excessive local curvature / lateral acceleration
      5) in-place spin / drift windows: tiny net displacement but large
         cumulative heading change over ~TRAJ_SPIN_WIN_S (see module constants)

    Steps marked as bad are treated as *break points* between samples. The time
    series is split into multiple continuous segments separated by bad steps,
    and we keep exactly one segment that still covers the impact time (within
    `time_tol_s`). No stitching is performed.
    """
    n = len(ts.time)
    if n < 2:
        return None

    t = np.asarray(ts.time, dtype=float)
    x_ego = np.asarray(ts.x_ego, dtype=float)
    y_ego = np.asarray(ts.y_ego, dtype=float)
    x_sur = np.asarray(ts.x_sur, dtype=float)
    y_sur = np.asarray(ts.y_sur, dtype=float)

    finite_sample = (
        np.isfinite(t)
        & np.isfinite(x_ego) & np.isfinite(y_ego)
        & np.isfinite(x_sur) & np.isfinite(y_sur)
    )

    dt = np.diff(t)  # shape: (n - 1,)

    # bad_step[i] corresponds to step i -> i+1
    bad_step = (~np.isfinite(dt)) | (dt <= 1e-4) | (dt > max_step_dt_s)
    bad_step |= ~(finite_sample[:-1] & finite_sample[1:])

    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _bad_jump(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return boolean array of shape (n-1,), aligned with step indices.
        Mark a step bad if its implied speed is too large.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        dist = np.hypot(np.diff(x), np.diff(y))
        step_speed = np.full(n - 1, np.nan, dtype=float)

        valid_dt = np.isfinite(dt) & (dt > 1e-6)
        step_speed[valid_dt] = dist[valid_dt] / dt[valid_dt]

        return (~np.isfinite(step_speed)) | (step_speed > max_step_speed_mps)

    def _bad_curvature_or_lateral_accel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return a boolean array of shape (n-1,), aligned with step indices.

        For each middle vertex (formed by step j and step j+1), evaluate:
          curvature ≈ |Δheading| / ds
          lateral acceleration ≈ v^2 * curvature

        Vertices with enough local motion (ds > min_ds_for_curvature_m) use
        the curvature / lateral-acceleration checks.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        bad = np.zeros(n - 1, dtype=bool)
        if n < 3:
            return bad

        dx = np.diff(x)                         # (n-1,)
        dy = np.diff(y)                         # (n-1,)
        seg_len = np.hypot(dx, dy)              # (n-1,)
        heading = np.arctan2(dy, dx)            # (n-1,)
        dtheta = _wrap_to_pi(np.diff(heading))  # (n-2,)

        ds = 0.5 * (seg_len[:-1] + seg_len[1:])   # (n-2,)
        dt_mid = 0.5 * (dt[:-1] + dt[1:])         # (n-2,)

        # Guard against degenerate segments (zero-length) which produce
        # meaningless heading values and therefore bogus dtheta.
        seg_len_valid = (seg_len[:-1] > 1e-6) & (seg_len[1:] > 1e-6)

        # Can be geometrically computed
        computable = (
            np.isfinite(dtheta)
            & np.isfinite(ds)
            & np.isfinite(dt_mid)
            & (dt_mid > 1e-6)
            & seg_len_valid
        )

        # Enough local displacement for direct curvature estimate
        eval_mask = computable & (ds > min_ds_for_curvature_m)

        bad_vertex = np.zeros(n - 2, dtype=bool)

        if np.any(eval_mask):
            curvature = np.abs(dtheta[eval_mask]) / ds[eval_mask]
            speed_mid = ds[eval_mask] / dt_mid[eval_mask]
            lat_acc = (speed_mid ** 2) * curvature

            bad_eval = (
                (~np.isfinite(curvature))
                | (~np.isfinite(lat_acc))
                | (curvature > max_curvature_inv_m)
                | (lat_acc > max_lateral_accel_mps2)
            )

            idx = np.flatnonzero(eval_mask)
            bad_vertex[idx] |= bad_eval

        # Truly invalid geometry is marked bad
        invalid_geom = ~computable
        bad_vertex |= invalid_geom

        # A bad middle vertex affects both adjacent steps
        if np.any(bad_vertex):
            bad[:-1] |= bad_vertex
            bad[1:] |= bad_vertex

        return bad

    def _bad_stationary_spin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return a boolean array of shape (n-1,), aligned with step indices.

        Detects "in-place spin" / jittery drift segments: inside a sliding
        time window of length ~spin_win_s, the net displacement is tiny
        (< spin_max_disp_m) but the cumulative absolute heading change is
        large (> spin_max_yaw_rad). Every step inside such a window is
        flagged as bad.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        bad = np.zeros(n - 1, dtype=bool)
        if n < 3 or spin_win_s <= 0.0:
            return bad

        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.hypot(dx, dy)
        heading = np.arctan2(dy, dx)
        # Invalidate heading when a step is degenerate (zero length) so that
        # a stationary pair does not contribute a spurious 180 deg flip.
        heading[seg_len <= 1e-6] = np.nan
        dtheta_abs = np.abs(_wrap_to_pi(np.diff(heading)))  # (n-2,)
        dtheta_abs = np.where(np.isfinite(dtheta_abs), dtheta_abs, 0.0)

        # cum_dtheta[k] = sum of dtheta_abs[0..k-1]; length n-1, cum_dtheta[0] = 0.
        cum_dtheta = np.concatenate(([0.0], np.cumsum(dtheta_abs)))

        j = 1
        for i in range(n - 1):
            if j <= i + 1:
                j = i + 1
            # Grow j so that t[j] - t[i] >= spin_win_s (smallest such j).
            while j < n and (t[j] - t[i]) < spin_win_s:
                j += 1
            if j >= n:
                break
            if not (np.isfinite(t[i]) and np.isfinite(t[j])):
                continue
            disp = math.hypot(x[j] - x[i], y[j] - y[i])
            if not math.isfinite(disp):
                continue
            # Cumulative |dtheta| over heading transitions in steps [i..j-1].
            # Those transitions are indexed by dtheta_abs[i..j-2] which equals
            # cum_dtheta[j - 1] - cum_dtheta[i].
            if j - 1 > i:
                yaw_sum = float(cum_dtheta[j - 1] - cum_dtheta[i])
            else:
                yaw_sum = 0.0
            if disp < spin_max_disp_m and yaw_sum > spin_max_yaw_rad:
                bad[i:j] = True

        return bad

    def _bad_small_loop_or_spin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return a boolean array of shape (n-1,), aligned with step indices.

        Broader "unreasonable looping" detector beyond strict in-place spin:
        within a sliding time window, if the motion stays in a small spatial
        envelope but accumulates substantial travel distance and large heading
        change, the segment likely reflects jitter / local circling rather than
        a physically plausible trajectory for crash reconstruction.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        bad = np.zeros(n - 1, dtype=bool)
        if n < 4:
            return bad

        # Derive thresholds from existing "spin" settings to avoid adding new
        # module-level knobs while still catching small-radius circling.
        loop_win_s = float(max(2.0 * spin_win_s, 2.0))
        loop_max_bbox_diag_m = float(max(2.5 * spin_max_disp_m, 3.0))
        loop_min_path_m = float(max(3.0 * spin_max_disp_m, 4.0))
        loop_max_net_to_path = 0.35
        loop_min_yaw_rad = float(max(spin_max_yaw_rad, math.pi / 2))

        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.hypot(dx, dy)  # (n-1,)
        # Path length prefix sum over segments: seg_cum[k] = sum(seg_len[0..k-1])
        seg_cum = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(seg_len), seg_len, 0.0))))

        heading = np.arctan2(dy, dx)
        heading[seg_len <= 1e-6] = np.nan
        dtheta_abs = np.abs(_wrap_to_pi(np.diff(heading)))  # (n-2,)
        dtheta_abs = np.where(np.isfinite(dtheta_abs), dtheta_abs, 0.0)
        # cum_dtheta[k] = sum of dtheta_abs[0..k-1]; length n-1, cum_dtheta[0] = 0.
        cum_dtheta = np.concatenate(([0.0], np.cumsum(dtheta_abs)))

        j = 1
        for i in range(n - 1):
            if j <= i + 1:
                j = i + 1
            while j < n and (t[j] - t[i]) < loop_win_s:
                j += 1
            if j >= n:
                break
            if not (np.isfinite(t[i]) and np.isfinite(t[j])):
                continue

            xw = x[i : j + 1]
            yw = y[i : j + 1]
            if xw.size < 2:
                continue
            if not (np.all(np.isfinite(xw)) and np.all(np.isfinite(yw))):
                continue

            min_x = float(np.min(xw))
            max_x = float(np.max(xw))
            min_y = float(np.min(yw))
            max_y = float(np.max(yw))
            bbox_diag = math.hypot(max_x - min_x, max_y - min_y)

            net_disp = math.hypot(float(x[j] - x[i]), float(y[j] - y[i]))
            path_len = float(seg_cum[j] - seg_cum[i])
            if not (math.isfinite(bbox_diag) and math.isfinite(net_disp) and math.isfinite(path_len)):
                continue
            if path_len <= 1e-6:
                continue

            if j - 1 > i:
                yaw_sum = float(cum_dtheta[j - 1] - cum_dtheta[i])
            else:
                yaw_sum = 0.0

            # "Small area" + "lots of travel" + "doesn't go anywhere" + "turns a lot".
            if (
                bbox_diag < loop_max_bbox_diag_m
                and path_len > loop_min_path_m
                and (net_disp / path_len) < loop_max_net_to_path
                and yaw_sum > loop_min_yaw_rad
            ):
                bad[i:j] = True

        return bad

    def _bad_large_loop_or_backtrack(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return a boolean array of shape (n-1,), aligned with step indices.

        Catch larger-scale looping/backtracking that is not "small envelope".
        Typical failure mode: trajectory makes a big loop or circles around and
        returns close to itself within a few seconds (high path length and yaw,
        but small net displacement), which is very unlikely for this dataset and
        usually indicates reconstruction jitter or ID swap.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        bad = np.zeros(n - 1, dtype=bool)
        if n < 5:
            return bad

        # Keep thresholds self-consistent with existing spin knobs, but allow
        # bigger spatial envelopes than `_bad_small_loop_or_spin`.
        win_s = float(max(3.0 * spin_win_s, 2.5))
        bbox_min_diag_m = float(max(3.0 * spin_max_disp_m, 3.0))
        bbox_max_diag_m = 30.0
        min_path_m = float(max(8.0 * spin_max_disp_m, 8.0))
        max_net_to_path = 0.25
        min_yaw_rad = float(max(1.8 * spin_max_yaw_rad, math.pi))  # >= ~180°

        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.hypot(dx, dy)  # (n-1,)
        seg_len_f = np.where(np.isfinite(seg_len), seg_len, 0.0)
        seg_cum = np.concatenate(([0.0], np.cumsum(seg_len_f)))

        heading = np.arctan2(dy, dx)
        heading[seg_len <= 1e-6] = np.nan
        dtheta_abs = np.abs(_wrap_to_pi(np.diff(heading)))  # (n-2,)
        dtheta_abs = np.where(np.isfinite(dtheta_abs), dtheta_abs, 0.0)
        cum_dtheta = np.concatenate(([0.0], np.cumsum(dtheta_abs)))

        # Per-step speed for a coarse "must be moving" gate.
        step_speed = np.full(n - 1, np.nan, dtype=float)
        valid_dt = np.isfinite(dt) & (dt > 1e-6)
        step_speed[valid_dt] = seg_len[valid_dt] / dt[valid_dt]

        j = 1
        for i in range(n - 1):
            if j <= i + 1:
                j = i + 1
            while j < n and (t[j] - t[i]) < win_s:
                j += 1
            if j >= n:
                break
            if not (np.isfinite(t[i]) and np.isfinite(t[j])):
                continue

            xw = x[i : j + 1]
            yw = y[i : j + 1]
            if xw.size < 2:
                continue
            if not (np.all(np.isfinite(xw)) and np.all(np.isfinite(yw))):
                continue

            min_x = float(np.min(xw))
            max_x = float(np.max(xw))
            min_y = float(np.min(yw))
            max_y = float(np.max(yw))
            bbox_diag = math.hypot(max_x - min_x, max_y - min_y)

            net_disp = math.hypot(float(x[j] - x[i]), float(y[j] - y[i]))
            path_len = float(seg_cum[j] - seg_cum[i])
            if not (math.isfinite(bbox_diag) and math.isfinite(net_disp) and math.isfinite(path_len)):
                continue
            if path_len <= 1e-6:
                continue

            if j - 1 > i:
                yaw_sum = float(cum_dtheta[j - 1] - cum_dtheta[i])
            else:
                yaw_sum = 0.0

            # Must be actually moving in this window (avoid classifying mild noise).
            sp = step_speed[i:j]
            med_speed = float(np.nanmedian(sp)) if sp.size else float("nan")

            if (
                bbox_diag >= bbox_min_diag_m
                and bbox_diag <= bbox_max_diag_m
                and path_len >= min_path_m
                and (net_disp / path_len) <= max_net_to_path
                and yaw_sum >= min_yaw_rad
                and math.isfinite(med_speed)
                and med_speed >= 0.5
            ):
                bad[i:j] = True

        return bad

    bad_step |= _bad_jump(x_ego, y_ego)
    bad_step |= _bad_jump(x_sur, y_sur)
    bad_step |= _bad_curvature_or_lateral_accel(x_ego, y_ego)
    bad_step |= _bad_curvature_or_lateral_accel(x_sur, y_sur)
    bad_step |= _bad_stationary_spin(x_ego, y_ego)
    bad_step |= _bad_stationary_spin(x_sur, y_sur)
    bad_step |= _bad_small_loop_or_spin(x_ego, y_ego)
    bad_step |= _bad_small_loop_or_spin(x_sur, y_sur)
    bad_step |= _bad_large_loop_or_backtrack(x_ego, y_ego)
    bad_step |= _bad_large_loop_or_backtrack(x_sur, y_sur)

    if not np.any(bad_step):
        return ts

    # bad_step[i] breaks the continuity between i and i+1.
    break_idx = np.flatnonzero(bad_step)  # indices in [0, n-2]
    starts = np.concatenate(([0], break_idx + 1))
    ends = np.concatenate((break_idx, [n - 1]))

    best = None  # (length, start, end)
    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)
        if s > e:
            continue
        # Segment must cover impact time (within tolerance).
        if float(t[s]) > impact_time + time_tol_s:
            continue
        if float(t[e]) < impact_time - time_tol_s:
            continue
        seg_len = e - s + 1
        cand = (seg_len, s, e)
        if best is None or cand > best:
            best = cand

    if best is None:
        return None

    _, left, right = best
    return _slice_event_timeseries(ts, slice(left, right + 1))


def extract_impact_window(
    ts: EventTimeSeries,
    impact_time: float,
    pre_max_s: float = PRE_IMPACT_MAX_S,
    pre_min_s: float = PRE_IMPACT_MIN_S,
    post_s: float = POST_IMPACT_S,
    time_tol_s: float = WINDOW_TIME_TOL_S,
) -> Optional[EventTimeSeries]:
    """
    Slice the time series to [impact_time - pre_max_s, impact_time + post_s] at most.
    Truncate low-quality trajectory tails around impact and apply light XY smoothing.
    Require recorded samples to cover at least [impact_time - pre_min_s, impact_time]
    (within time_tol_s) after truncation; otherwise the event is skipped.
    """
    t_span_lo = impact_time - pre_max_s
    t_span_hi = impact_time + post_s
    t_req_lo = impact_time - pre_min_s
    t_req_hi = impact_time

    mask = (ts.time >= t_span_lo) & (ts.time <= t_span_hi)
    if int(mask.sum()) < MIN_RAW_SAMPLES_IN_WINDOW:
        return None

    windowed = _slice_event_timeseries(ts, mask)
    windowed = _truncate_low_quality_segment(windowed, impact_time, time_tol_s=time_tol_s)
    if windowed is None:
        return None
    if len(windowed.time) < MIN_RAW_SAMPLES_IN_WINDOW:
        return None
    if float(np.min(windowed.time)) > t_req_lo + time_tol_s:
        return None
    if float(np.max(windowed.time)) < t_req_hi - time_tol_s:
        return None

    return windowed


# ---------------------------------------------------------------------------
# Coordinate transform: global -> ego-local at t=0
# ---------------------------------------------------------------------------

def transform_to_ego_local(ts: EventTimeSeries) -> EventTimeSeries:
    """
    Transform all positions and headings into ego's local coordinate frame
    at t=0: ego origin = (0, 0), ego heading = 0.
    """
    x0 = float(ts.x_ego[0])
    y0 = float(ts.y_ego[0])
    psi0 = float(ts.psi_ego[0])

    cos_a = math.cos(-psi0)
    sin_a = math.sin(-psi0)

    def _rotate(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = x - x0
        dy = y - y0
        x_local = cos_a * dx - sin_a * dy
        y_local = sin_a * dx + cos_a * dy
        return x_local, y_local

    x_ego_l, y_ego_l = _rotate(ts.x_ego, ts.y_ego)
    x_sur_l, y_sur_l = _rotate(ts.x_sur, ts.y_sur)
    psi_ego_l = ts.psi_ego - psi0
    psi_sur_l = ts.psi_sur - psi0

    return EventTimeSeries(
        event_id=ts.event_id,
        target_id=ts.target_id,
        time=ts.time.copy(),
        x_ego=x_ego_l,
        y_ego=y_ego_l,
        v_ego=ts.v_ego.copy(),
        psi_ego=psi_ego_l,
        acc_ego=ts.acc_ego.copy(),
        x_sur=x_sur_l,
        y_sur=y_sur_l,
        v_sur=ts.v_sur.copy(),
        psi_sur=psi_sur_l,
    )


# ---------------------------------------------------------------------------
# Kinematics computation
# ---------------------------------------------------------------------------

def compute_kinematics(
    v: np.ndarray,
    psi: np.ndarray,
    dt: float = DT_RAW,
    wheelbase: float = WHEELBASE,
    acc_measured: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute acceleration and steering angle from speed + heading sequences.

    Returns (acceleration, steering_angle_deg) arrays matching input length.
    """
    if acc_measured is not None and len(acc_measured) == len(v):
        acc = np.asarray(acc_measured, dtype=float).copy()
        bad = ~np.isfinite(acc)
        if bad.any():
            acc[bad] = np.gradient(v, dt)[bad]
    else:
        acc = np.gradient(v, dt)

    psi_cont = np.unwrap(np.asarray(psi, dtype=float))
    yaw_rate = np.gradient(psi_cont, dt)

    v_safe = np.maximum(np.abs(v), 1.5)
    low_speed_mask = np.abs(v) < 1.5
    steer_rad = np.arctan(yaw_rate * wheelbase / v_safe)
    steer_rad[low_speed_mask] = 0.0
    steer_deg = np.degrees(steer_rad)

    return acc, steer_deg


def compute_kinematics_from_trajectory(
    time_raw: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wheelbase: float = WHEELBASE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute acceleration and steering angle from trajectory points.

    Returns (acceleration, steering_angle_deg) arrays matching input length.
    """
    t = np.asarray(time_raw, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float)

    # Ensure strictly increasing time for stable derivatives.
    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 1e-6):
        t = np.arange(n, dtype=float) * DT_RAW

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = np.hypot(vx, vy)
    acc = np.gradient(speed, t)

    heading = np.unwrap(np.arctan2(vy, vx))
    yaw_rate = np.gradient(heading, t)
    v_safe = np.maximum(speed, 1.5)
    low_speed_mask = speed < 1.5
    steer_rad = np.arctan(yaw_rate * wheelbase / v_safe)
    steer_rad[low_speed_mask] = 0.0
    steer_deg = np.degrees(steer_rad)

    return acc, steer_deg


def _trajectory_time_xy_heading(
    traj_seq: List[Dict],
    dt_default: float = DT_CTRL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time/x/y/heading arrays for a trajectory sequence."""
    if not traj_seq:
        z = np.zeros(0, dtype=float)
        return z, z, z, z

    n = len(traj_seq)
    x = np.asarray([float(p.get("x", 0.0)) for p in traj_seq], dtype=float)
    y = np.asarray([float(p.get("y", 0.0)) for p in traj_seq], dtype=float)
    if all("t" in p for p in traj_seq):
        t = np.asarray([float(p.get("t", i * dt_default)) for i, p in enumerate(traj_seq)], dtype=float)
    else:
        t = np.arange(n, dtype=float) * float(dt_default)

    if n >= 2:
        dt = np.diff(t)
        if np.any(~np.isfinite(dt)) or np.any(dt <= 1e-6):
            t = np.arange(n, dtype=float) * float(dt_default)

    heading = None
    if all("heading" in p for p in traj_seq):
        h = np.asarray([float(p.get("heading", 0.0)) for p in traj_seq], dtype=float)
        if h.size == n and np.all(np.isfinite(h)):
            heading = np.unwrap(h)

    if heading is None:
        if n == 1:
            heading = np.zeros(1, dtype=float)
        else:
            vx = np.gradient(x, t)
            vy = np.gradient(y, t)
            heading = np.unwrap(np.arctan2(vy, vx))
    return t, x, y, heading


def _target_longitudinal_accel_from_xy_heading(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
) -> np.ndarray:
    """Signed longitudinal acceleration from (x, y, heading)."""
    if len(t) <= 1:
        return np.zeros(len(t), dtype=float)
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    s_long = vx * np.cos(heading) + vy * np.sin(heading)
    return np.gradient(s_long, t)


def _instantaneous_risk_metrics(
    t: np.ndarray,
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    tgt_x: np.ndarray,
    tgt_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-step (distance, closing speed, TTC)."""
    if len(t) <= 1:
        z = np.zeros(len(t), dtype=float)
        return z, z, z

    evx = np.gradient(ego_x, t)
    evy = np.gradient(ego_y, t)
    tvx = np.gradient(tgt_x, t)
    tvy = np.gradient(tgt_y, t)
    rel_px = tgt_x - ego_x
    rel_py = tgt_y - ego_y
    rel_vx = tvx - evx
    rel_vy = tvy - evy

    dist = np.hypot(rel_px, rel_py)
    dot = rel_px * rel_vx + rel_py * rel_vy
    closing = np.divide(
        -dot,
        dist,
        out=np.zeros_like(dist),
        where=dist > 1e-6,
    )
    ttc = np.divide(
        dist,
        closing,
        out=np.full_like(dist, np.inf),
        where=closing > 1e-6,
    )
    ttc = np.where(dist <= 1e-3, 0.0, ttc)
    return dist, closing, ttc


def prepare_stage_distribution_pair(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    *,
    tail_steps: int = REF_TRAJ_STEPS,
    dt_default: float = DT_CTRL,
    ego_length: Optional[float] = None,
    ego_width: Optional[float] = None,
    target_length: Optional[float] = None,
    target_width: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    """Keep the trailing ``tail_steps`` (default 12 / 6 s) from aligned trajectories.

    Caller should pass crash_profiles trajectories after the usual ``[3:]`` trim.
    Collision timing is inferred later from box geometry inside
    ``classify_interaction_stages``.
    """
    if len(ego_traj) < tail_steps or len(target_traj) < tail_steps:
        return None
    n = min(len(ego_traj), len(target_traj))
    sub_start = n - tail_steps
    return {
        "ego_trajectory_sequence": ego_traj[sub_start:n],
        "target_trajectory_sequence": target_traj[sub_start:n],
        "dt": float(dt_default),
        "ego_length": ego_length,
        "ego_width": ego_width,
        "target_length": target_length,
        "target_width": target_width,
    }


def _first_consecutive_true_index(mask: np.ndarray, need: int) -> int:
    """Start index of the first run of ``need`` consecutive True values, else -1."""
    if need <= 0 or mask.size == 0:
        return -1
    count = 0
    for i in range(int(mask.size)):
        if bool(mask[i]):
            count += 1
            if count >= need:
                return i - need + 1
        else:
            count = 0
    return -1


def _active_after_onset(mask: np.ndarray, need: int) -> np.ndarray:
    """True from the first ``need``-step run of ``mask`` onward (inclusive), else all False."""
    out = np.zeros(int(mask.size), dtype=bool)
    onset = _first_consecutive_true_index(mask, need)
    if onset >= 0:
        out[onset:] = mask[onset:]
    return out


def _assign_stages_by_onset(
    pre_collision_end: int,
    onset2: int,
    onset3: int,
) -> np.ndarray:
    """Label stages 1–3 before collision from stage-2 / stage-3 onset indices.

    Assignment follows temporal onset order: stage 2 occupies
    ``[onset2, onset3)`` when ``onset2 < onset3``; stage 3 begins at ``onset3``.
    Intermediate stages may be skipped (e.g. 1→3→4 when ``onset3`` precedes
    ``onset2``). When both onsets coincide, the shared step is stage 2 and stage 3
    starts on the following step.
    """
    stage = np.ones(int(pre_collision_end), dtype=np.int64)
    for i in range(int(pre_collision_end)):
        if onset2 >= 0 and i >= onset2:
            in_stage2 = (
                onset3 < 0
                or i < onset3
                or (onset3 == onset2 and i == onset2)
            )
            if in_stage2:
                stage[i] = 2
        if onset3 >= 0 and i >= onset3:
            if not (onset3 == onset2 and i == onset2):
                stage[i] = 3
    return stage


def _stage_instant_masks(
    *,
    ttc: np.ndarray,
    closing: np.ndarray,
    dist: np.ndarray,
    d0: float,
    a_long: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-step stage-2 / stage-3 eligibility masks (evaluated independently)."""
    ttc_eff = np.where(np.isfinite(ttc), ttc, np.inf)
    risk_geom = (
        (ttc_eff <= STAGE2_TTC_MAX_S)
        | (closing >= STAGE2_CLOSING_MIN_MPS)
        | (dist <= STAGE2_DIST_FRAC_OF_D0 * d0)
    )
    emerg_geom = (
        (ttc_eff <= STAGE3_TTC_MAX_S)
        | (closing >= STAGE3_CLOSING_MIN_MPS)
        | (dist <= STAGE3_DIST_FRAC_OF_D0 * d0)
    )
    emerg_action = a_long <= STAGE3_DECEL_MIN_MS2
    emerg_strong = a_long <= STAGE3_STRONG_DECEL_MS2
    cond_stage2 = risk_geom
    cond_stage3 = (emerg_geom & emerg_action) | emerg_strong
    return cond_stage2, cond_stage3


def classify_interaction_stages(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    *,
    dt_default: float = DT_CTRL,
    ego_length: Optional[float] = None,
    ego_width: Optional[float] = None,
    target_length: Optional[float] = None,
    target_width: Optional[float] = None,
    min_consecutive_steps: int = 1,
) -> Dict[str, object]:
    """Classify one ego-target interaction into 4 monotonic crash stages.

    Stages follow order 1→2→3→4 (never decrease) but intermediate stages may be
    skipped—for example 1→3→4 without risk awareness, or 1→2→4 without emergency.

    Stage 4 onset follows ego/target box contact on the passed trajectory. Stages
    2–3 use relaxed kinematic thresholds (TTC, closing speed, distance, target
    decel). Each stage is anchored by its first eligible run; labels
    follow onset order (stage-2 window before stage-3 when stage 2 begins first).
    No external ``impact_time_index`` is used.
    """
    n = min(len(ego_traj), len(target_traj))
    if n < 3:
        return {
            "stage_ids": np.zeros(0, dtype=np.int64),
            "stage_counts": {k: 0 for k in STAGE_NAME_BY_ID.values()},
            "stage_percentages": {k: 0.0 for k in STAGE_NAME_BY_ID.values()},
            "n_steps": 0,
        }

    has_box_dims = (
        ego_length is not None
        and ego_width is not None
        and target_length is not None
        and target_width is not None
        and float(ego_length) > 0.0
        and float(ego_width) > 0.0
        and float(target_length) > 0.0
        and float(target_width) > 0.0
    )
    coll_idx = -1
    if has_box_dims:
        coll_idx = first_box_collision_step_index(
            ego_traj,
            target_traj,
            ego_length=float(ego_length),
            ego_width=float(ego_width),
            target_length=float(target_length),
            target_width=float(target_width),
        )

    if coll_idx >= 0:
        clip_n = int(max(2, min(coll_idx + 1, n)))
    else:
        clip_n = n

    t_e, x_e, y_e, h_e = _trajectory_time_xy_heading(ego_traj[:clip_n], dt_default=dt_default)
    t_t, x_t, y_t, h_t = _trajectory_time_xy_heading(target_traj[:clip_n], dt_default=dt_default)
    m = min(len(t_e), len(t_t))
    if m < 3:
        return {
            "stage_ids": np.zeros(0, dtype=np.int64),
            "stage_counts": {k: 0 for k in STAGE_NAME_BY_ID.values()},
            "stage_percentages": {k: 0.0 for k in STAGE_NAME_BY_ID.values()},
            "n_steps": 0,
        }

    t = t_t[:m]
    x_e = x_e[:m]
    y_e = y_e[:m]
    h_e = h_e[:m]
    x_t = x_t[:m]
    y_t = y_t[:m]
    h_t = h_t[:m]

    dist, closing, ttc = _instantaneous_risk_metrics(t, x_e, y_e, x_t, y_t)
    a_long = _target_longitudinal_accel_from_xy_heading(t, x_t, y_t, h_t)

    d0 = float(np.nanmedian(dist[: min(3, len(dist))])) if len(dist) else 1.0
    d0 = max(d0, 1e-3)

    cond_stage2, cond_stage3 = _stage_instant_masks(
        ttc=ttc,
        closing=closing,
        dist=dist,
        d0=d0,
        a_long=a_long,
    )

    need = max(1, int(min_consecutive_steps))
    coll_start = -1
    if has_box_dims and coll_idx >= 0:
        coll_start = int(max(0, min(coll_idx, m - 1)))

    pre_collision_end = coll_start if coll_start >= 0 else m
    pre_mask2 = cond_stage2[:pre_collision_end]
    pre_mask3 = cond_stage3[:pre_collision_end]
    onset2 = _first_consecutive_true_index(pre_mask2, need)
    onset3 = _first_consecutive_true_index(pre_mask3, need)

    stage = np.ones(m, dtype=np.int64)
    stage[:pre_collision_end] = _assign_stages_by_onset(
        pre_collision_end,
        onset2,
        onset3,
    )
    if coll_start >= 0:
        stage[coll_start:] = 4

    counts: Dict[str, int] = {}
    for sid, sname in STAGE_NAME_BY_ID.items():
        counts[sname] = int(np.sum(stage == sid))
    total = float(sum(counts.values()))
    if total <= 0:
        percentages = {k: 0.0 for k in counts}
    else:
        percentages = {k: (float(v) / total) * 100.0 for k, v in counts.items()}

    return {
        "stage_ids": stage,
        "stage_counts": counts,
        "stage_percentages": percentages,
        "n_steps": int(m),
    }


def _stage_time_indices(stage_ids: np.ndarray) -> Dict[str, Optional[int]]:
    """First local time index at which stages 2–3 begin (stage 1 omitted)."""
    indices: Dict[str, Optional[int]] = {
        name: None for name in STAGE_TIME_INDEX_NAMES
    }
    if stage_ids.size == 0:
        return indices
    for sid in (2, 3):
        hits = np.where(stage_ids >= sid)[0]
        if hits.size > 0:
            indices[STAGE_NAME_BY_ID[sid]] = int(hits[0])
    return indices


def sanitize_stage_time_indices_vs_impact(
    stage_time_indices: Dict[str, Optional[int]],
    impact_time_index: Optional[int],
) -> Dict[str, Optional[int]]:
    """Drop stage onsets that are not strictly before ``impact_time_index``.

    Collision must occur after risk awareness: when
    ``impact_time_index <= risk_awareness``, both ``risk_awareness`` and
    ``emergency_response`` are cleared. When only emergency onset violates the
    ordering, ``emergency_response`` alone is cleared.
    """
    out = dict(stage_time_indices)
    if impact_time_index is None:
        return out
    imp = int(impact_time_index)
    risk = out.get(STAGE_RISK_AWARENESS)
    if risk is not None and imp <= int(risk):
        out[STAGE_RISK_AWARENESS] = None
        out[STAGE_EMERGENCY_RESPONSE] = None
        return out
    emerg = out.get(STAGE_EMERGENCY_RESPONSE)
    if emerg is not None and imp <= int(emerg):
        out[STAGE_EMERGENCY_RESPONSE] = None
    return out


def aggregate_stage_time_distribution(
    trajectory_pairs: List[Dict[str, object]],
    *,
    dt_default: float = DT_CTRL,
    min_consecutive_steps: int = 1,
) -> Dict[str, object]:
    """Aggregate stage occupancy percentages across many trajectory pairs."""
    totals = {k: 0 for k in STAGE_NAME_BY_ID.values()}
    n_valid = 0
    for pair in trajectory_pairs:
        out = classify_interaction_stages(
            pair.get("ego_trajectory_sequence") or [],
            pair.get("target_trajectory_sequence") or [],
            dt_default=float(pair.get("dt", dt_default)),
            ego_length=pair.get("ego_length"),
            ego_width=pair.get("ego_width"),
            target_length=pair.get("target_length"),
            target_width=pair.get("target_width"),
            min_consecutive_steps=min_consecutive_steps,
        )
        if int(out.get("n_steps", 0)) <= 0:
            continue
        n_valid += 1
        cc = out["stage_counts"]
        for k in totals:
            totals[k] += int(cc.get(k, 0))

    denom = float(sum(totals.values()))
    if denom <= 0:
        percentages = {k: 0.0 for k in totals}
    else:
        percentages = {k: float(v) * 100.0 / denom for k, v in totals.items()}

    return {
        "stage_counts": totals,
        "stage_percentages": percentages,
        "n_valid_pairs": int(n_valid),
    }


# ---------------------------------------------------------------------------
# Resample to 0.5s control steps
# ---------------------------------------------------------------------------

def resample_to_control_steps(
    time_raw: np.ndarray,
    acc_raw: np.ndarray,
    steer_raw: np.ndarray,
    dt_ctrl: float = DT_CTRL,
) -> List[Dict]:
    """Resample raw 10Hz acc/steer into 0.5s bins."""
    if len(time_raw) == 0:
        return []

    t0 = float(time_raw[0])
    t_end = float(time_raw[-1])
    duration = max(t_end - t0, 1e-6)
    n_steps = max(1, int(math.ceil(duration / dt_ctrl)))
    time_scaled = time_raw

    controls = []
    for i in range(n_steps):
        t_center = t0 + (i + 0.5) * dt_ctrl
        t_lo = t0 + i * dt_ctrl
        t_hi = t0 + (i + 1) * dt_ctrl
        if i == n_steps - 1:
            mask = (time_scaled >= t_lo) & (time_scaled <= t_end)
        else:
            mask = (time_scaled >= t_lo) & (time_scaled < t_hi)
        if mask.sum() > 0:
            a = float(np.mean(acc_raw[mask]))
            s = float(np.mean(steer_raw[mask]))
        else:
            idx = int(np.argmin(np.abs(time_scaled - t_center)))
            a = float(acc_raw[idx])
            s = float(steer_raw[idx])
        controls.append({
            "t": round((i + 1) * dt_ctrl, 1),
            "acceleration": round(a, 3),
            "steering_angle": round(s, 3),
        })
    return controls


def resample_to_trajectory_steps(
    time_raw: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    dt_ctrl: float = DT_CTRL,
) -> List[Dict]:
    """Resample raw 10Hz (x, y) positions to 0.5s steps via endpoint interpolation.

    Each step i stores the interpolated position at t = t0 + (i+1)*dt_ctrl, so that
    control step i (covering [i*dt_ctrl, (i+1)*dt_ctrl]) integrates the vehicle from
    the previous trajectory point to the current one. This makes the trajectory and
    control sequences directly comparable for reconstruction verification.
    """
    if len(time_raw) == 0:
        return []

    t0 = float(time_raw[0])
    t_end = float(time_raw[-1])
    duration = max(t_end - t0, 1e-6)
    n_steps = max(1, int(math.ceil(duration / dt_ctrl)))

    trajectory = []
    for i in range(n_steps):
        t_query = float(np.clip(t0 + (i + 1) * dt_ctrl, t0, t_end))
        x_val = float(np.interp(t_query, time_raw, x_raw))
        y_val = float(np.interp(t_query, time_raw, y_raw))
        trajectory.append({
            "t": round((i + 1) * dt_ctrl, 1),
            "x": round(x_val, 4),
            "y": round(y_val, 4),
        })
    return trajectory


def _count_resampled_steps_before_impact(
    traj: List[Dict],
    window_t0: float,
    impact_time: float,
    dt_ctrl: float = DT_CTRL,
) -> int:
    """How many resampled steps have endpoint time strictly before impact_time.

    Trajectory dict ``t`` is relative to ``window_t0`` (first sample of the impact window).
    """
    cut = 0
    for i, step in enumerate(traj):
        t_rel = float(step.get("t", (i + 1) * dt_ctrl))
        if window_t0 + t_rel < impact_time:
            cut = i + 1
        else:
            break
    return cut


def _build_ref_trajectory_candidate(
    sample: CrashSample,
    start_index: int,
    n_steps: int = REF_TRAJ_STEPS,
    dt_ctrl: float = DT_CTRL,
) -> Optional[Dict[str, object]]:
    """Build a fixed-length ego/target trajectory candidate from one sample window."""
    event_traj = sample.event_trajectory_sequence or {}
    ego_seq = event_traj.get("ego_trajectory_sequence", [])
    target_seq = event_traj.get("target_trajectory_sequence", [])
    if not ego_seq or not target_seq:
        return None
    if start_index < 0:
        return None
    end = start_index + n_steps
    if end > len(ego_seq) or end > len(target_seq):
        return None

    ego_slice = ego_seq[start_index:end]
    target_slice = target_seq[start_index:end]
    if len(ego_slice) != n_steps or len(target_slice) != n_steps:
        return None

    impact_idx = int(math.ceil((float(sample.impact_time) - float(start_index * dt_ctrl)) / dt_ctrl)) - 1
    if impact_idx < 0 or impact_idx >= n_steps:
        return None

    return {
        "source_event_id": int(sample.event_id),
        "target_id": int(sample.target_id),
        "start_index": int(start_index),
        "impact_time_index": int(impact_idx),
        "impact_time": round(float(sample.impact_time), 3),
        "ego_trajectory_sequence": ego_slice,
        "target_trajectory_sequence": target_slice,
    }


def _trajectory_step_speeds(traj_seq: List[Dict], dt_ctrl: float = DT_CTRL) -> np.ndarray:
    if len(traj_seq) < 2:
        return np.zeros(0, dtype=float)
    x = np.asarray([float(p.get("x", 0.0)) for p in traj_seq], dtype=float)
    y = np.asarray([float(p.get("y", 0.0)) for p in traj_seq], dtype=float)
    ds = np.hypot(np.diff(x), np.diff(y))
    return ds / max(float(dt_ctrl), 1e-6)


def _trajectory_lateral_shift(traj_seq: List[Dict], impact_idx: int) -> float:
    if len(traj_seq) < 2:
        return 0.0
    i = max(1, min(int(impact_idx), len(traj_seq) - 1))
    x0 = float(traj_seq[0].get("x", 0.0))
    y0 = float(traj_seq[0].get("y", 0.0))
    x1 = float(traj_seq[i].get("x", x0))
    y1 = float(traj_seq[i].get("y", y0))
    dx = x1 - x0
    dy = y1 - y0
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return 0.0
    nx = -dy / norm
    ny = dx / norm
    max_shift = 0.0
    for step in traj_seq[: i + 1]:
        sx = float(step.get("x", 0.0)) - x0
        sy = float(step.get("y", 0.0)) - y0
        lat = abs(nx * sx + ny * sy)
        if lat > max_shift:
            max_shift = lat
    return float(max_shift)


def _trajectory_headings_rad(traj_seq: List[Dict]) -> np.ndarray:
    """Infer heading at each trajectory step from successive (x, y) deltas."""
    n = len(traj_seq)
    if n == 0:
        return np.zeros(0, dtype=float)
    headings = np.zeros(n, dtype=float)
    for i in range(n):
        if i + 1 < n:
            dx = float(traj_seq[i + 1].get("x", 0.0)) - float(traj_seq[i].get("x", 0.0))
            dy = float(traj_seq[i + 1].get("y", 0.0)) - float(traj_seq[i].get("y", 0.0))
        elif i > 0:
            dx = float(traj_seq[i].get("x", 0.0)) - float(traj_seq[i - 1].get("x", 0.0))
            dy = float(traj_seq[i].get("y", 0.0)) - float(traj_seq[i - 1].get("y", 0.0))
        else:
            dx, dy = 1.0, 0.0
        if math.hypot(dx, dy) < 1e-6:
            headings[i] = headings[i - 1] if i > 0 else 0.0
        else:
            headings[i] = math.atan2(dy, dx)
    return headings


def _ref_trajectory_box_distance_at_step(
    sample: CrashSample,
    ego_traj: List[Dict],
    target_traj: List[Dict],
    ego_headings: np.ndarray,
    tgt_headings: np.ndarray,
    step_idx: int,
) -> float:
    """OBB-OBB minimum distance (m) between ego and target at one ref step."""
    return trajectory_pair_box_distance_at_step(
        ego_traj,
        target_traj,
        ego_headings,
        tgt_headings,
        step_idx,
        ego_length=float(sample.ego_length),
        ego_width=float(sample.ego_width),
        target_length=float(sample.target_length),
        target_width=float(sample.target_width),
    )


def _ref_trajectory_has_collision(
    sample: CrashSample,
    ego_traj: List[Dict],
    target_traj: List[Dict],
    impact_idx: int,
    distance_thresh_m: float = VEHICLE_COLLISION_DISTANCE_THRESH_M,
) -> bool:
    """
    True when ego/target OBBs contact within ``distance_thresh_m`` at impact
    (or the adjacent step), matching raw-data collision semantics.
    """
    if not sample.is_crash:
        return False
    return trajectory_pair_has_box_contact_at_impact(
        ego_traj,
        target_traj,
        impact_idx,
        ego_length=float(sample.ego_length),
        ego_width=float(sample.ego_width),
        target_length=float(sample.target_length),
        target_width=float(sample.target_width),
        distance_thresh_m=float(distance_thresh_m),
    )


def _relative_pose_at_impact(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    impact_idx: int,
) -> Tuple[float, float, float]:
    """Longitudinal/lateral offset and relative heading (rad) at impact in ego frame."""
    i = max(0, min(int(impact_idx), len(ego_traj) - 1, len(target_traj) - 1))
    ex = float(ego_traj[i].get("x", 0.0))
    ey = float(ego_traj[i].get("y", 0.0))
    tx = float(target_traj[i].get("x", 0.0))
    ty = float(target_traj[i].get("y", 0.0))
    ego_h = _trajectory_headings_rad(ego_traj)
    tgt_h = _trajectory_headings_rad(target_traj)
    psi_e = float(ego_h[i]) if len(ego_h) > i else 0.0
    psi_t = float(tgt_h[i]) if len(tgt_h) > i else 0.0
    c, s = math.cos(psi_e), math.sin(psi_e)
    dx, dy = tx - ex, ty - ey
    lon = dx * c + dy * s
    lat = -dx * s + dy * c
    rel_heading = (psi_t - psi_e + math.pi) % (2.0 * math.pi) - math.pi
    return float(lon), float(lat), float(rel_heading)


def _target_pre_impact_lateral_rate_mps(
    target_traj: List[Dict],
    impact_idx: int,
    dt_ctrl: float = DT_CTRL,
) -> float:
    """Max |dy/dt| on target path before impact (ego-frame lateral motion)."""
    i = max(1, min(int(impact_idx), len(target_traj) - 1))
    y = np.asarray(
        [float(p.get("y", 0.0)) for p in target_traj[: i + 1]], dtype=float
    )
    if len(y) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(y))) / max(float(dt_ctrl), 1e-6))


def _nhtsa_semantic_fit_for_ref(
    nhtsa_code: str,
    ego_traj: List[Dict],
    target_traj: List[Dict],
    target_ctrl: List[Dict],
    impact_idx: int,
    win: Optional[Dict[str, object]] = None,
    dt_ctrl: float = DT_CTRL,
) -> Tuple[bool, float]:
    """
    Check whether a ref trajectory matches the NHTSA class semantics.

    Returns (passes_hard_filter, penalty). Lower penalty is a better fit.
    """
    code = str(nhtsa_code)
    lon, lat, rel_h = _relative_pose_at_impact(ego_traj, target_traj, impact_idx)
    tgt_speeds = _trajectory_step_speeds(target_traj, dt_ctrl=dt_ctrl)
    i_imp = max(0, min(int(impact_idx), len(target_ctrl) - 1))
    pre_acc = np.asarray(
        [
            float(c.get("acceleration", 0.0))
            for c in target_ctrl[: i_imp + 1]
        ],
        dtype=float,
    )
    tgt_v_end = float(tgt_speeds[i_imp - 1]) if i_imp > 0 and len(tgt_speeds) >= i_imp else (
        float(tgt_speeds[-1]) if len(tgt_speeds) else 0.0
    )
    tgt_v_mean = float(np.mean(tgt_speeds[: max(1, i_imp)])) if len(tgt_speeds) else 0.0
    max_lat_rate = _target_pre_impact_lateral_rate_mps(
        target_traj, impact_idx, dt_ctrl=dt_ctrl
    )

    tcs = (win or {}).get("target_current_state") or {}
    if tcs:
        lon = float(tcs.get("delta_x", lon))
        lat = float(tcs.get("delta_y", lat))
        rel_h = float(tcs.get("heading", rel_h))

    penalty = 0.0

    def _soft_fail(extra: float = 120.0) -> Tuple[bool, float]:
        return True, penalty + extra

    # --- Leading vehicle family: target ahead, speed profile matches subclass ---
    if code in ("1", "2", "7", "15"):
        if lon < 2.0:
            return _soft_fail(80.0 + max(0.0, 2.0 - lon) * 25.0)
        if code == "1":
            if tgt_v_end > 1.2 or tgt_v_mean > 1.5:
                penalty += 80.0 + 20.0 * tgt_v_end
            if tgt_v_end > 0.6:
                penalty += 30.0
        elif code == "2":
            if len(pre_acc) and float(np.min(pre_acc)) > -1.0:
                penalty += 60.0
            if len(pre_acc) and float(np.mean(pre_acc)) > -0.3:
                penalty += 25.0
        elif code == "15":
            if len(pre_acc) and float(np.max(pre_acc)) < 0.8:
                penalty += 60.0
        elif code == "7":
            if tgt_v_mean < 0.4 or tgt_v_mean > 12.0:
                penalty += 40.0
            if len(pre_acc) and float(np.max(np.abs(pre_acc))) > 3.0:
                penalty += 20.0
        return True, penalty

    # --- Lane change vs drift ---
    if code == "4":
        if max_lat_rate < 0.8:
            return _soft_fail(60.0 + max(0.0, 0.8 - max_lat_rate) * 40.0)
        penalty += max(0.0, 1.2 - max_lat_rate) * 20.0
        return True, penalty
    if code == "10":
        if max_lat_rate > 1.2:
            return _soft_fail(50.0 + (max_lat_rate - 1.2) * 30.0)
        if max_lat_rate < 0.15:
            penalty += 40.0
        return True, penalty

    # --- Junction / path crossing ---
    if code == "5":
        if abs(lat) < 2.0 or abs(lat) < abs(lon) * 0.35:
            return _soft_fail(
                70.0
                + max(0.0, 2.0 - abs(lat)) * 20.0
                + max(0.0, abs(lon) * 0.35 - abs(lat)) * 15.0
            )
        penalty += max(0.0, 3.0 - abs(lat)) * 10.0
        return True, penalty

    if code == "3":
        tgt_h = _trajectory_headings_rad(target_traj)
        if len(tgt_h) >= 2:
            h_change = abs(
                (tgt_h[min(i_imp, len(tgt_h) - 1)] - tgt_h[0] + math.pi)
                % (2.0 * math.pi)
                - math.pi
            )
        else:
            h_change = 0.0
        if h_change < 0.35 and abs(lat) < 1.5:
            return _soft_fail(
                65.0
                + max(0.0, 0.35 - h_change) * 40.0
                + max(0.0, 1.5 - abs(lat)) * 20.0
            )
        penalty += max(0.0, 0.5 - h_change) * 30.0
        return True, penalty

    if code in ("6", "14"):
        tgt_h = _trajectory_headings_rad(target_traj)
        h_change = 0.0
        if len(tgt_h) >= 2:
            h_change = abs(
                (tgt_h[min(i_imp, len(tgt_h) - 1)] - tgt_h[0] + math.pi)
                % (2.0 * math.pi)
                - math.pi
            )
        if h_change < 0.2:
            penalty += 50.0
        return True, penalty

    if code == "8":
        if abs(rel_h) < 2.0:
            return _soft_fail(60.0 + max(0.0, 2.0 - abs(rel_h)) * 35.0)
        penalty += max(0.0, 2.4 - abs(rel_h)) * 40.0
        return True, penalty

    if code == "9":
        if lon > -1.5:
            return _soft_fail(55.0 + max(0.0, lon + 1.5) * 25.0)
        penalty += max(0.0, -lon - 3.0) * 5.0
        return True, penalty

    if code == "11":
        if lon < -1.0:
            penalty += 30.0
        return True, penalty

    if code == "12":
        if not _target_has_emergency_decel_or_evasive_action(
            target_ctrl, target_traj, impact_idx=impact_idx
        ):
            penalty += 80.0 + _target_pre_impact_action_penalty(
                target_ctrl, target_traj, impact_idx=impact_idx
            )
        return True, penalty

    if code == "13":
        if tgt_v_mean > 4.0:
            penalty += 40.0
        if abs(rel_h) > 1.2:
            penalty += 25.0
        return True, penalty

    # 16 Other and any unlisted code: soft ranking only
    return True, penalty


def _is_ref_trajectory_kinematically_feasible(
    ego_ctrl: List[Dict],
    target_ctrl: List[Dict],
    ego_traj: List[Dict],
    target_traj: List[Dict],
    impact_idx: int,
    dt_ctrl: float = DT_CTRL,
) -> Tuple[bool, float]:
    """Check hard kinematic constraints and return smoothness penalty."""
    if len(ego_ctrl) != REF_TRAJ_STEPS or len(target_ctrl) != REF_TRAJ_STEPS:
        return False, float("inf")
    if len(ego_traj) != REF_TRAJ_STEPS or len(target_traj) != REF_TRAJ_STEPS:
        return False, float("inf")

    ego_acc = np.asarray([float(c.get("acceleration", 0.0)) for c in ego_ctrl], dtype=float)
    tgt_acc = np.asarray([float(c.get("acceleration", 0.0)) for c in target_ctrl], dtype=float)
    ego_steer = np.asarray([float(c.get("steering_angle", 0.0)) for c in ego_ctrl], dtype=float)
    tgt_steer = np.asarray([float(c.get("steering_angle", 0.0)) for c in target_ctrl], dtype=float)

    if not (
        np.all(np.isfinite(ego_acc))
        and np.all(np.isfinite(tgt_acc))
        and np.all(np.isfinite(ego_steer))
        and np.all(np.isfinite(tgt_steer))
    ):
        return False, float("inf")

    if np.max(np.abs(ego_acc)) > REF_MAX_ABS_ACC_MS2 or np.max(np.abs(tgt_acc)) > REF_MAX_ABS_ACC_MS2:
        return False, float("inf")
    if np.max(np.abs(ego_steer)) > REF_MAX_ABS_STEER_DEG or np.max(np.abs(tgt_steer)) > REF_MAX_ABS_STEER_DEG:
        return False, float("inf")

    ego_jerk = np.diff(ego_acc) / max(float(dt_ctrl), 1e-6)
    tgt_jerk = np.diff(tgt_acc) / max(float(dt_ctrl), 1e-6)
    if len(ego_jerk) and np.max(np.abs(ego_jerk)) > REF_MAX_ABS_JERK_MS3:
        return False, float("inf")
    if len(tgt_jerk) and np.max(np.abs(tgt_jerk)) > REF_MAX_ABS_JERK_MS3:
        return False, float("inf")

    ego_steer_rate = np.diff(ego_steer) / max(float(dt_ctrl), 1e-6)
    tgt_steer_rate = np.diff(tgt_steer) / max(float(dt_ctrl), 1e-6)
    if len(ego_steer_rate) and np.max(np.abs(ego_steer_rate)) > REF_MAX_STEER_RATE_DEGPS:
        return False, float("inf")
    if len(tgt_steer_rate) and np.max(np.abs(tgt_steer_rate)) > REF_MAX_STEER_RATE_DEGPS:
        return False, float("inf")

    ego_speed = _trajectory_step_speeds(ego_traj, dt_ctrl=dt_ctrl)
    tgt_speed = _trajectory_step_speeds(target_traj, dt_ctrl=dt_ctrl)
    if len(ego_speed) and np.max(ego_speed) > REF_MAX_STEP_SPEED_MPS:
        return False, float("inf")
    if len(tgt_speed) and np.max(tgt_speed) > REF_MAX_STEP_SPEED_MPS:
        return False, float("inf")

    # Smaller penalty means smoother trajectory/control profile.
    smooth_penalty = (
        float(np.mean(np.abs(ego_jerk))) if len(ego_jerk) else 0.0
    ) + (
        float(np.mean(np.abs(tgt_jerk))) if len(tgt_jerk) else 0.0
    ) + 0.2 * (
        float(np.mean(np.abs(ego_steer_rate))) if len(ego_steer_rate) else 0.0
    ) + 0.2 * (
        float(np.mean(np.abs(tgt_steer_rate))) if len(tgt_steer_rate) else 0.0
    )
    return True, smooth_penalty


def _target_pre_impact_action_penalty(
    target_ctrl: List[Dict],
    target_traj: List[Dict],
    impact_idx: int,
) -> float:
    """Ranking penalty when target lacks pre-impact brake / mild evasive action (lower is better)."""
    if impact_idx <= 0:
        return 80.0
    pre_ctrl = target_ctrl[: impact_idx + 1]
    if not pre_ctrl:
        return 80.0

    acc = np.asarray([float(c.get("acceleration", 0.0)) for c in pre_ctrl], dtype=float)
    steer = np.asarray([float(c.get("steering_angle", 0.0)) for c in pre_ctrl], dtype=float)
    min_acc = float(np.min(acc)) if len(acc) else 0.0
    max_steer = float(np.max(np.abs(steer))) if len(steer) else 0.0
    lat_shift = _trajectory_lateral_shift(target_traj, impact_idx=impact_idx)

    brake_gap = max(0.0, min_acc - REF_BRAKE_ACC_MS2)
    steer_gap = max(0.0, REF_EVASIVE_STEER_DEG - max_steer)
    lat_gap = max(0.0, REF_EVASIVE_LATERAL_SHIFT_M - lat_shift)
    return 25.0 + 30.0 * brake_gap + 8.0 * steer_gap + 20.0 * lat_gap


def _target_has_emergency_decel_or_evasive_action(
    target_ctrl: List[Dict],
    target_traj: List[Dict],
    impact_idx: int,
) -> bool:
    """Target shows mild pre-impact braking and/or small evasive steering/lateral motion."""
    if impact_idx <= 0:
        return False
    pre_ctrl = target_ctrl[: impact_idx + 1]
    if not pre_ctrl:
        return False

    acc = np.asarray([float(c.get("acceleration", 0.0)) for c in pre_ctrl], dtype=float)
    steer = np.asarray([float(c.get("steering_angle", 0.0)) for c in pre_ctrl], dtype=float)
    has_brake = bool(np.any(acc <= REF_BRAKE_ACC_MS2))
    has_evasive_steer = bool(np.any(np.abs(steer) >= REF_EVASIVE_STEER_DEG))
    lat_shift = _trajectory_lateral_shift(target_traj, impact_idx=impact_idx)
    has_evasive_lateral = lat_shift >= REF_EVASIVE_LATERAL_SHIFT_M
    return has_brake or has_evasive_steer or has_evasive_lateral


def _extract_ref_trajectory_subwindow(
    win: Dict[str, object],
    n_steps: int = REF_TRAJ_STEPS,
) -> Optional[Dict[str, object]]:
    """
    Take the last ``n_steps`` from one sample_control_sequence window.

    Sliding windows use SAMPLE_WINDOW_S (e.g. 16 steps); ref trajectory uses the
    trailing n_steps (12 / 6s). ``impact_time_index`` is already local to that
    tail slice (0..n_steps-1).
    """
    ego_ctrl = win.get("ego_control_sequence", [])
    tgt_ctrl = win.get("target_control_sequence", [])
    ego_traj = win.get("ego_trajectory_sequence", [])
    tgt_traj = win.get("target_trajectory_sequence", [])
    w_len = len(ego_ctrl)
    if w_len < n_steps:
        return None
    if len(tgt_ctrl) != w_len or len(ego_traj) != w_len or len(tgt_traj) != w_len:
        return None

    impact_idx = win.get("impact_time_index")
    if impact_idx is None:
        return None
    impact_idx = int(impact_idx)
    if impact_idx < 0 or impact_idx >= n_steps:
        return None

    sub_start = w_len - n_steps
    sub_end = w_len

    return {
        "start_index": int(win.get("start_index", 0)) + sub_start,
        "impact_time_index": impact_idx,
        "ego_control_sequence": ego_ctrl[sub_start:sub_end],
        "target_control_sequence": tgt_ctrl[sub_start:sub_end],
        "ego_trajectory_sequence": ego_traj[sub_start:sub_end],
        "target_trajectory_sequence": tgt_traj[sub_start:sub_end],
    }


def _samples_with_min_impact_mdc(samples: List[CrashSample]) -> List[CrashSample]:
    """Keep only samples whose impact_timestamp_mdc equals the class minimum."""
    scored: List[Tuple[CrashSample, float]] = []
    for sample in samples:
        mdc = sample.impact_timestamp_mdc
        if mdc is None or not np.isfinite(float(mdc)):
            continue
        scored.append((sample, float(mdc)))
    if not scored:
        return list(samples)
    min_mdc = min(m for _, m in scored)
    return [s for s, m in scored if abs(m - min_mdc) <= 1e-6]


def _gather_ref_trajectory_candidates(
    samples: List[CrashSample],
    code: str,
    n_steps: int = REF_TRAJ_STEPS,
    dt_ctrl: float = DT_CTRL,
    *,
    require_collision: bool = True,
    require_kinematic: bool = True,
    require_semantic_pass: bool = False,
    tier_penalty: float = 0.0,
    allow_non_crash: bool = False,
) -> List[Tuple[float, Dict[str, object]]]:
    """Collect ref-trajectory candidates under configurable filter strictness."""
    candidates: List[Tuple[float, Dict[str, object]]] = []

    for sample in samples:
        if not sample.is_crash and not allow_non_crash:
            continue
        if not np.isfinite(float(sample.impact_time)):
            continue

        for win in sample.sample_control_sequence or []:
            sub = _extract_ref_trajectory_subwindow(win, n_steps=n_steps)
            if sub is None:
                continue

            ego_ctrl = sub["ego_control_sequence"]
            tgt_ctrl = sub["target_control_sequence"]
            ego_traj = sub["ego_trajectory_sequence"]
            tgt_traj = sub["target_trajectory_sequence"]
            impact_idx = int(sub["impact_time_index"])
            start_index = int(sub["start_index"])

            has_collision = _ref_trajectory_has_collision(
                sample, ego_traj, tgt_traj, impact_idx
            )
            if require_collision and not has_collision:
                continue

            ok_feasible, smooth_penalty = _is_ref_trajectory_kinematically_feasible(
                ego_ctrl, tgt_ctrl, ego_traj, tgt_traj, impact_idx, dt_ctrl=dt_ctrl
            )
            if require_kinematic and not ok_feasible:
                continue
            if not ok_feasible:
                smooth_penalty = 200.0

            ok_sem, semantic_penalty = _nhtsa_semantic_fit_for_ref(
                code,
                ego_traj,
                tgt_traj,
                tgt_ctrl,
                impact_idx,
                win=win,
                dt_ctrl=dt_ctrl,
            )
            if require_semantic_pass and not ok_sem:
                continue
            if not ok_sem:
                semantic_penalty = 500.0

            ref_payload = {
                "source_event_id": int(sample.event_id),
                "target_id": int(sample.target_id),
                "start_index": start_index,
                "impact_time_index": impact_idx,
                "impact_time": round(float(sample.impact_time), 3),
                "ego_trajectory_sequence": ego_traj,
                "target_trajectory_sequence": tgt_traj,
                "ego_control_sequence": ego_ctrl,
                "target_control_sequence": tgt_ctrl,
            }
            built_candidate = _build_ref_trajectory_candidate(
                sample,
                start_index=start_index,
                n_steps=n_steps,
                dt_ctrl=dt_ctrl,
            )
            if built_candidate is not None:
                b_ego = built_candidate["ego_trajectory_sequence"]
                b_tgt = built_candidate["target_trajectory_sequence"]
                b_imp = int(built_candidate["impact_time_index"])
                if _ref_trajectory_has_collision(sample, b_ego, b_tgt, b_imp):
                    ref_payload.update({
                        "source_event_id": int(built_candidate["source_event_id"]),
                        "target_id": int(built_candidate["target_id"]),
                        "start_index": int(built_candidate["start_index"]),
                        "impact_time_index": b_imp,
                        "impact_time": float(built_candidate["impact_time"]),
                        "ego_trajectory_sequence": b_ego,
                        "target_trajectory_sequence": b_tgt,
                    })

            tgt_ctrl_use = ref_payload["target_control_sequence"]
            tgt_traj_use = ref_payload["target_trajectory_sequence"]
            imp_use = int(ref_payload["impact_time_index"])
            action_penalty = _target_pre_impact_action_penalty(
                tgt_ctrl_use, tgt_traj_use, impact_idx=imp_use
            )
            if _target_has_emergency_decel_or_evasive_action(
                tgt_ctrl_use, tgt_traj_use, impact_idx=imp_use
            ):
                action_penalty *= 0.35

            collision_penalty = 0.0 if has_collision else 120.0
            rank = (
                float(tier_penalty)
                + float(semantic_penalty)
                + float(smooth_penalty)
                + float(action_penalty)
                + float(collision_penalty)
            )
            if code == "8":
                _, _, rel_h_rank = _relative_pose_at_impact(
                    ego_traj, tgt_traj, impact_idx
                )
                rank += max(
                    0.0, OPPOSITE_REL_HEADING_MIN_RAD - abs(float(rel_h_rank))
                ) * 80.0
            if allow_non_crash:
                ref_payload["nearest_approach"] = True
                mdc = sample.impact_timestamp_mdc
                if mdc is not None and np.isfinite(float(mdc)):
                    ref_payload["impact_timestamp_mdc"] = round(float(mdc), 4)
            ref_payload["selection_score"] = round(rank, 4)
            ref_payload["semantic_penalty"] = round(float(semantic_penalty), 4)
            ref_payload["smoothness_penalty"] = round(float(smooth_penalty), 4)
            ref_payload["action_penalty"] = round(float(action_penalty), 4)
            candidates.append((rank, ref_payload))

    return candidates


def _fallback_ref_trajectory_from_event_sequences(
    samples: List[CrashSample],
    n_steps: int = REF_TRAJ_STEPS,
    dt_ctrl: float = DT_CTRL,
    *,
    allow_non_crash: bool = False,
) -> Optional[Dict[str, object]]:
    """Last-resort ref trajectory from full-event control/trajectory slices at impact."""
    best: Optional[Tuple[float, Dict[str, object]]] = None

    for sample in samples:
        if not allow_non_crash and not sample.is_crash:
            continue
        if not np.isfinite(float(sample.impact_time)):
            continue

        event_cs = sample.event_control_sequence or {}
        event_ts = sample.event_trajectory_sequence or {}
        ego_ctrl = event_cs.get("ego_control_sequence", [])
        tgt_ctrl = event_cs.get("target_control_sequence", [])
        ego_traj = event_ts.get("ego_trajectory_sequence", [])
        tgt_traj = event_ts.get("target_trajectory_sequence", [])
        seq_len = min(len(ego_ctrl), len(tgt_ctrl), len(ego_traj), len(tgt_traj))
        if seq_len < n_steps:
            continue

        impact_step = int(round(float(sample.impact_time) / max(float(dt_ctrl), 1e-6)))
        impact_step = max(0, min(impact_step, seq_len - 1))
        start_index = max(0, min(impact_step - n_steps + 1, seq_len - n_steps))
        end = start_index + n_steps
        impact_idx = impact_step - start_index
        if impact_idx < 0 or impact_idx >= n_steps:
            continue

        ego_ctrl_s = ego_ctrl[start_index:end]
        tgt_ctrl_s = tgt_ctrl[start_index:end]
        ego_traj_s = ego_traj[start_index:end]
        tgt_traj_s = tgt_traj[start_index:end]
        rank = 300.0 + _target_pre_impact_action_penalty(
            tgt_ctrl_s, tgt_traj_s, impact_idx=impact_idx
        )
        if not _ref_trajectory_has_collision(sample, ego_traj_s, tgt_traj_s, impact_idx):
            rank += 150.0

        payload = {
            "source_event_id": int(sample.event_id),
            "target_id": int(sample.target_id),
            "start_index": int(start_index),
            "impact_time_index": int(impact_idx),
            "impact_time": round(float(sample.impact_time), 3),
            "ego_trajectory_sequence": ego_traj_s,
            "target_trajectory_sequence": tgt_traj_s,
            "ego_control_sequence": ego_ctrl_s,
            "target_control_sequence": tgt_ctrl_s,
            "selection_score": round(float(rank), 4),
            "semantic_penalty": 500.0,
            "smoothness_penalty": 200.0,
            "action_penalty": round(
                float(_target_pre_impact_action_penalty(
                    tgt_ctrl_s, tgt_traj_s, impact_idx=impact_idx
                )),
                4,
            ),
            "fallback": True,
        }
        if allow_non_crash:
            payload["nearest_approach"] = True
            mdc = sample.impact_timestamp_mdc
            if mdc is not None and np.isfinite(float(mdc)):
                payload["impact_timestamp_mdc"] = round(float(mdc), 4)
        if best is None or rank < best[0]:
            best = (rank, payload)

    return best[1] if best is not None else None


def _target_ref_trajectory_is_opposite_direction(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    impact_idx: int,
) -> bool:
    """True when target travel direction is roughly opposite to ego at impact."""
    _, _, rel_h = _relative_pose_at_impact(ego_traj, target_traj, impact_idx)
    return abs(float(rel_h)) >= OPPOSITE_REL_HEADING_MIN_RAD


def _reverse_target_ref_trajectory_payload(
    ref: Dict[str, object],
) -> Dict[str, object]:
    """
    Time-reverse target ref trajectory so motion is opposite-direction (NHTSA 8).

    Keeps ego and impact index unchanged; negates target acceleration/steering.
    """
    out = dict(ref)
    tgt_traj = list(ref.get("target_trajectory_sequence", []))
    tgt_ctrl_src = list(ref.get("target_control_sequence", []))
    tgt_ctrl_rev: List[Dict] = []
    for c in reversed(tgt_ctrl_src):
        nc = dict(c)
        nc["acceleration"] = round(-float(c.get("acceleration", 0.0)), 3)
        nc["steering_angle"] = round(-float(c.get("steering_angle", 0.0)), 3)
        tgt_ctrl_rev.append(nc)
    out["target_trajectory_sequence"] = list(reversed(tgt_traj))
    out["target_control_sequence"] = tgt_ctrl_rev
    out["target_time_reversed"] = True
    return out


def _ensure_opposite_direction_ref_trajectory(
    ref: Optional[Dict[str, object]],
    nhtsa_code: str,
) -> Optional[Dict[str, object]]:
    """NHTSA 8: representative ref must show opposite-direction target motion."""
    if ref is None or str(nhtsa_code) != "8":
        return ref
    ego = ref.get("ego_trajectory_sequence", [])
    tgt = ref.get("target_trajectory_sequence", [])
    imp = int(ref.get("impact_time_index", 0))
    if not ego or not tgt:
        return ref
    if _target_ref_trajectory_is_opposite_direction(ego, tgt, imp):
        return ref
    return _reverse_target_ref_trajectory_payload(ref)


def select_representative_ref_trajectory_for_class(
    samples: List[CrashSample],
    nhtsa_code: Optional[str] = None,
    n_steps: int = REF_TRAJ_STEPS,
    dt_ctrl: float = DT_CTRL,
) -> Optional[Dict[str, object]]:
    """
    Select one representative 12-step (6s) ref trajectory for one NHTSA class.

    Prefer crash windows with OBB contact, feasible kinematics, and NHTSA-consistent
    geometry; target pre-impact mild brake / small evasive action is ranked, not hard-filtered.

    If no candidate passes the preferred tier, progressively relax filters and still
    return the best-ranked ref trajectory when any valid subwindow exists.

    When every sample has ``is_crash=False``, only samples with the minimum
    ``impact_timestamp_mdc`` are considered (nearest-approach ref).

    For NHTSA class 8 (opposite direction), the returned target ref trajectory is
    time-reversed when needed so travel direction at impact opposes ego.
    """
    if not samples:
        return None
    code = str(nhtsa_code or samples[0].nhtsa_code)

    allow_non_crash = not any(s.is_crash for s in samples)
    working_samples = (
        _samples_with_min_impact_mdc(samples) if allow_non_crash else samples
    )

    search_tiers = [
        {
            "require_collision": True,
            "require_kinematic": True,
            "require_semantic_pass": False,
            "tier_penalty": 0.0,
        },
        {
            "require_collision": True,
            "require_kinematic": False,
            "require_semantic_pass": False,
            "tier_penalty": 50.0,
        },
        {
            "require_collision": False,
            "require_kinematic": False,
            "require_semantic_pass": False,
            "tier_penalty": 150.0,
        },
    ]

    for tier in search_tiers:
        tier_candidates = _gather_ref_trajectory_candidates(
            working_samples,
            code,
            n_steps=n_steps,
            dt_ctrl=dt_ctrl,
            allow_non_crash=allow_non_crash,
            **tier,
        )
        if tier_candidates:
            tier_candidates.sort(key=lambda x: x[0])
            return _ensure_opposite_direction_ref_trajectory(
                tier_candidates[0][1], code
            )

    return _ensure_opposite_direction_ref_trajectory(
        _fallback_ref_trajectory_from_event_sequences(
            working_samples,
            n_steps=n_steps,
            dt_ctrl=dt_ctrl,
            allow_non_crash=allow_non_crash,
        ),
        code,
    )


def _obb_corners(
    cx: float,
    cy: float,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Return 4 CCW corners of an OBB centred at ``(cx, cy)`` and oriented along ``heading``.

    Convention matches the rest of the file: heading=0 points along +x, +y is left,
    so ``length`` is the longitudinal (forward) extent and ``width`` is the lateral
    (left-right) extent.
    """
    hl = max(float(length), 0.0) * 0.5
    hw = max(float(width), 0.0) * 0.5
    local = np.array(
        [
            [hl, -hw],   # front-right
            [hl, hw],    # front-left
            [-hl, hw],   # rear-left
            [-hl, -hw],  # rear-right
        ],
        dtype=float,
    )
    c, s = math.cos(float(heading)), math.sin(float(heading))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return local @ rot.T + np.array([float(cx), float(cy)], dtype=float)


def _polygon_area(poly: np.ndarray) -> float:
    """Unsigned area of a simple polygon via the shoelace formula."""
    if poly.ndim != 2 or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _convex_polygon_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman intersection of two convex CCW polygons."""
    output = [tuple(p) for p in subject]
    n_clip = clip.shape[0]
    for i in range(n_clip):
        if not output:
            return np.empty((0, 2), dtype=float)
        a = clip[i]
        b = clip[(i + 1) % n_clip]
        edge_x = b[0] - a[0]
        edge_y = b[1] - a[1]
        # Left-hand normal of edge a->b; >=0 means inside for CCW polygons.
        nx, ny = -edge_y, edge_x

        def side(px: float, py: float) -> float:
            return nx * (px - a[0]) + ny * (py - a[1])

        input_list = output
        output = []
        s_prev = input_list[-1]
        d_prev = side(s_prev[0], s_prev[1])
        for p in input_list:
            d_curr = side(p[0], p[1])
            if d_curr >= 0.0:
                if d_prev < 0.0:
                    denom = d_prev - d_curr
                    t = d_prev / denom if denom != 0.0 else 0.0
                    ix = s_prev[0] + t * (p[0] - s_prev[0])
                    iy = s_prev[1] + t * (p[1] - s_prev[1])
                    output.append((ix, iy))
                output.append(tuple(p))
            elif d_prev >= 0.0:
                denom = d_prev - d_curr
                t = d_prev / denom if denom != 0.0 else 0.0
                ix = s_prev[0] + t * (p[0] - s_prev[0])
                iy = s_prev[1] + t * (p[1] - s_prev[1])
                output.append((ix, iy))
            s_prev = p
            d_prev = d_curr

    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _obb_iou(ego_corners: np.ndarray, target_corners: np.ndarray) -> float:
    """IoU between two oriented bounding boxes given as CCW 4x2 corner arrays."""
    ego_area = _polygon_area(ego_corners)
    tgt_area = _polygon_area(target_corners)
    if ego_area <= 0.0 or tgt_area <= 0.0:
        return 0.0

    # Quick AABB rejection to avoid the polygon clip in the common no-overlap case.
    ego_min = ego_corners.min(axis=0)
    ego_max = ego_corners.max(axis=0)
    tgt_min = target_corners.min(axis=0)
    tgt_max = target_corners.max(axis=0)
    if ego_max[0] < tgt_min[0] or tgt_max[0] < ego_min[0]:
        return 0.0
    if ego_max[1] < tgt_min[1] or tgt_max[1] < ego_min[1]:
        return 0.0

    inter = _convex_polygon_intersection(ego_corners, target_corners)
    inter_area = _polygon_area(inter)
    if inter_area <= 0.0:
        return 0.0
    union = ego_area + tgt_area - inter_area
    return inter_area / union if union > 0.0 else 0.0


def _point_to_segment_dist_m(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    eps = 1e-12

    def _orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def _on_seg(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True
    if abs(o1) <= eps and _on_seg(a1, b1, a2):
        return True
    if abs(o2) <= eps and _on_seg(a1, b2, a2):
        return True
    if abs(o3) <= eps and _on_seg(b1, a1, b2):
        return True
    if abs(o4) <= eps and _on_seg(b1, a2, b2):
        return True
    return False


def _obb_min_distance_m(ego_corners: np.ndarray, target_corners: np.ndarray) -> float:
    """Minimum boundary distance between two oriented boxes."""
    ego = np.asarray(ego_corners, dtype=float)
    tgt = np.asarray(target_corners, dtype=float)
    if ego.shape != (4, 2) or tgt.shape != (4, 2):
        return float("inf")

    for i in range(4):
        a1 = ego[i]
        a2 = ego[(i + 1) % 4]
        for j in range(4):
            b1 = tgt[j]
            b2 = tgt[(j + 1) % 4]
            if _segments_intersect(a1, a2, b1, b2):
                return 0.0

    dmin = float("inf")
    for i in range(4):
        p = ego[i]
        for j in range(4):
            q1 = tgt[j]
            q2 = tgt[(j + 1) % 4]
            dmin = min(dmin, _point_to_segment_dist_m(p, q1, q2))
    for i in range(4):
        p = tgt[i]
        for j in range(4):
            q1 = ego[j]
            q2 = ego[(j + 1) % 4]
            dmin = min(dmin, _point_to_segment_dist_m(p, q1, q2))
    return float(dmin if np.isfinite(dmin) else float("inf"))


def trajectory_pair_box_distance_at_step(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    ego_headings: np.ndarray,
    tgt_headings: np.ndarray,
    step_idx: int,
    *,
    ego_length: float,
    ego_width: float,
    target_length: float,
    target_width: float,
) -> float:
    """OBB-OBB minimum distance (m) between ego and target at one trajectory step."""
    i = int(step_idx)
    if i < 0 or i >= len(ego_traj) or i >= len(target_traj):
        return float("inf")
    ego_len = float(ego_length)
    ego_wid = float(ego_width)
    tgt_len = float(target_length)
    tgt_wid = float(target_width)
    if min(ego_len, ego_wid, tgt_len, tgt_wid) <= 0.0:
        ex = float(ego_traj[i].get("x", 0.0))
        ey = float(ego_traj[i].get("y", 0.0))
        tx = float(target_traj[i].get("x", 0.0))
        ty = float(target_traj[i].get("y", 0.0))
        return math.hypot(tx - ex, ty - ey)

    ego_box = _obb_corners(
        float(ego_traj[i].get("x", 0.0)),
        float(ego_traj[i].get("y", 0.0)),
        float(ego_headings[i]),
        ego_len,
        ego_wid,
    )
    tgt_box = _obb_corners(
        float(target_traj[i].get("x", 0.0)),
        float(target_traj[i].get("y", 0.0)),
        float(tgt_headings[i]),
        tgt_len,
        tgt_wid,
    )
    return _obb_min_distance_m(ego_box, tgt_box)


def first_box_collision_step_index(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    *,
    ego_length: float,
    ego_width: float,
    target_length: float,
    target_width: float,
    distance_thresh_m: float = VEHICLE_COLLISION_DISTANCE_THRESH_M,
) -> int:
    """First 0-based step with ego/target box distance <= threshold, else -1."""
    n = min(len(ego_traj), len(target_traj))
    if n <= 0:
        return -1
    if min(float(ego_length), float(ego_width), float(target_length), float(target_width)) <= 0.0:
        return -1

    ego_h = _trajectory_headings_rad(ego_traj)
    tgt_h = _trajectory_headings_rad(target_traj)
    if len(ego_h) == 0 or len(tgt_h) == 0:
        return -1

    thresh = float(distance_thresh_m)
    for i in range(n):
        if trajectory_pair_box_distance_at_step(
            ego_traj,
            target_traj,
            ego_h,
            tgt_h,
            i,
            ego_length=float(ego_length),
            ego_width=float(ego_width),
            target_length=float(target_length),
            target_width=float(target_width),
        ) <= thresh:
            return int(i)
    return -1


def trajectory_pair_has_box_contact_at_impact(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    impact_idx: int,
    *,
    ego_length: float,
    ego_width: float,
    target_length: float,
    target_width: float,
    distance_thresh_m: float = VEHICLE_COLLISION_DISTANCE_THRESH_M,
) -> bool:
    """True when ego/target OBB distance <= threshold at impact (or adjacent step).

    Reuses the same contact semantics as ``first_box_collision_step_index``.
    """
    coll_idx = first_box_collision_step_index(
        ego_traj,
        target_traj,
        ego_length=float(ego_length),
        ego_width=float(ego_width),
        target_length=float(target_length),
        target_width=float(target_width),
        distance_thresh_m=float(distance_thresh_m),
    )
    if coll_idx < 0:
        return False

    ego_h = _trajectory_headings_rad(ego_traj)
    tgt_h = _trajectory_headings_rad(target_traj)
    if len(ego_h) == 0 or len(tgt_h) == 0:
        return False

    imp = max(0, min(int(impact_idx), len(ego_h) - 1, len(tgt_h) - 1))
    check_steps = {imp, coll_idx}
    if imp > 0:
        check_steps.add(imp - 1)
    if imp + 1 < len(ego_h):
        check_steps.add(imp + 1)

    thresh = float(distance_thresh_m)
    for k in check_steps:
        if trajectory_pair_box_distance_at_step(
            ego_traj,
            target_traj,
            ego_h,
            tgt_h,
            k,
            ego_length=float(ego_length),
            ego_width=float(ego_width),
            target_length=float(target_length),
            target_width=float(target_width),
        ) <= thresh:
            return True
    return False


def _impact_time_from_raw_closest_approach(
    ts: EventTimeSeries,
    fallback_time: float,
    meta: Optional[EventMeta] = None,
    distance_thresh_m: float = VEHICLE_COLLISION_DISTANCE_THRESH_M,
) -> Tuple[float, bool]:
    """Locate the impact moment from raw ego/target trajectories.

    When vehicle dimensions are available via ``meta``, collision detection is
    performed by building oriented bounding boxes (using each vehicle's position,
    heading, length and width) and computing OBB minimum boundary distance at
    every raw sample. The impact is reported as the **first** time this distance
    falls below ``distance_thresh_m`` (default 0.1m). If no frame reaches the
    threshold we fall back to the time of minimum OBB distance; if box distance
    cannot be computed we fall back to legacy minimum-center-distance behaviour.

    When ``meta`` is ``None`` (or dimensions are missing/invalid) the function
    degenerates to the original closest-center-of-mass approach so existing
    callers remain compatible.

    Returns ``(impact_time, is_crash)``. ``is_crash`` is True only when the
    chosen time comes from the first frame with box distance <= threshold.
    """
    try:
        t = np.asarray(ts.time, dtype=float)
        xe = np.asarray(ts.x_ego, dtype=float)
        ye = np.asarray(ts.y_ego, dtype=float)
        xs = np.asarray(ts.x_sur, dtype=float)
        ys = np.asarray(ts.y_sur, dtype=float)
        pe = np.asarray(ts.psi_ego, dtype=float)
        ps = np.asarray(ts.psi_sur, dtype=float)
    except Exception:
        return float(fallback_time), False

    if t.size == 0:
        return float(fallback_time), False

    d2 = (xs - xe) ** 2 + (ys - ye) ** 2
    base_mask = np.isfinite(t) & np.isfinite(d2)
    if not np.any(base_mask):
        return float(fallback_time), False

    has_dims = (
        meta is not None
        and math.isfinite(float(meta.ego_length))
        and math.isfinite(float(meta.ego_width))
        and math.isfinite(float(meta.target_length))
        and math.isfinite(float(meta.target_width))
        and float(meta.ego_length) > 0.0
        and float(meta.ego_width) > 0.0
        and float(meta.target_length) > 0.0
        and float(meta.target_width) > 0.0
    )

    if has_dims:
        heading_mask = base_mask & np.isfinite(pe) & np.isfinite(ps)
        if np.any(heading_mask):
            idx_all = np.where(heading_mask)[0]
            ego_len = float(meta.ego_length)
            ego_wid = float(meta.ego_width)
            tgt_len = float(meta.target_length)
            tgt_wid = float(meta.target_width)

            min_dists = np.full(idx_all.size, np.inf, dtype=float)
            for k, i in enumerate(idx_all):
                ego_box = _obb_corners(
                    float(xe[i]), float(ye[i]), float(pe[i]), ego_len, ego_wid
                )
                tgt_box = _obb_corners(
                    float(xs[i]), float(ys[i]), float(ps[i]), tgt_len, tgt_wid
                )
                min_dists[k] = _obb_min_distance_m(ego_box, tgt_box)

            collide = min_dists <= float(distance_thresh_m)
            if np.any(collide):
                # First frame where ego and target boxes are within distance threshold.
                first_k = int(np.argmax(collide))
                t_imp = float(t[idx_all[first_k]])
                if np.isfinite(t_imp):
                    return t_imp, True
            elif np.any(np.isfinite(min_dists)):
                # No frame reaches threshold: use the closest box-distance frame.
                best_k = int(np.nanargmin(min_dists))
                t_imp = float(t[idx_all[best_k]])
                if np.isfinite(t_imp):
                    return t_imp, False

    # Fallback: minimum centre-to-centre distance (legacy behaviour).
    idx = int(np.argmin(d2[base_mask]))
    t_best = float(t[base_mask][idx])
    return (
        (t_best if np.isfinite(t_best) else float(fallback_time)),
        False,
    )


def _infer_t0_position_from_endpoint_traj(traj_seq: List[Dict]) -> Tuple[float, float]:
    """Infer approximate t=0 position from a sequence of t=DT_CTRL endpoints."""
    if not traj_seq:
        return 0.0, 0.0
    if len(traj_seq) == 1:
        return float(traj_seq[0].get("x", 0.0)), float(traj_seq[0].get("y", 0.0))

    x_a = float(traj_seq[0].get("x", 0.0))
    y_a = float(traj_seq[0].get("y", 0.0))
    x_b = float(traj_seq[1].get("x", x_a))
    y_b = float(traj_seq[1].get("y", y_a))
    return 2.0 * x_a - x_b, 2.0 * y_a - y_b


def controls_from_trajectory_steps(
    traj_seq: List[Dict],
    initial_x: float,
    initial_y: float,
    dt_ctrl: float = DT_CTRL,
    wheelbase: float = WHEELBASE,
) -> List[Dict]:
    """Derive per-step controls from endpoint trajectory samples at DT_CTRL."""
    def _wrap_pi_local(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    if not traj_seq:
        return []

    n = len(traj_seq)
    t = np.array(
        [float(step.get("t", (i + 1) * dt_ctrl)) for i, step in enumerate(traj_seq)],
        dtype=float,
    )
    x = np.array([float(step.get("x", 0.0)) for step in traj_seq], dtype=float)
    y = np.array([float(step.get("y", 0.0)) for step in traj_seq], dtype=float)

    t_full = np.concatenate(([0.0], t))
    x_full = np.concatenate(([float(initial_x)], x))
    y_full = np.concatenate(([float(initial_y)], y))

    if len(t_full) >= 3:
        vx = np.gradient(x_full, t_full)
        vy = np.gradient(y_full, t_full)
    else:
        dt = max(float(t_full[-1] - t_full[0]), 1e-6)
        vx = np.array([(x_full[1] - x_full[0]) / dt, (x_full[1] - x_full[0]) / dt], dtype=float)
        vy = np.array([(y_full[1] - y_full[0]) / dt, (y_full[1] - y_full[0]) / dt], dtype=float)

    speed = np.hypot(vx, vy)
    if len(t_full) >= 3:
        acc = np.gradient(speed, t_full)
        heading = np.unwrap(np.arctan2(vy, vx))
        yaw_rate = np.gradient(heading, t_full)
    else:
        acc = np.zeros_like(speed)
        heading = np.unwrap(np.arctan2(vy, vx))
        yaw_rate = np.zeros_like(speed)

    v_safe = np.maximum(speed, 1.5)
    low_speed_mask = speed < 1.5
    steer_rad = np.arctan(yaw_rate * wheelbase / v_safe)
    steer_rad[low_speed_mask] = 0.0
    steer_deg = np.degrees(steer_rad)

    controls: List[Dict] = []
    for i in range(n):
        dx = float(x_full[i + 1] - x_full[i])
        dy = float(y_full[i + 1] - y_full[i])
        delta_s = math.hypot(dx, dy)
        delta_heading = _wrap_pi_local(float(heading[i + 1] - heading[i]))
        controls.append({
            "t": round(float(t[i]), 1),
            "acceleration": round(float(acc[i + 1]), 3),
            "steering_angle": round(float(steer_deg[i + 1]), 3),
            "delta_s": round(float(delta_s), 4),
            "delta_heading": round(float(delta_heading), 4),
        })
    return controls


def build_reconstruction_initial_state(ts: EventTimeSeries) -> Dict[str, Dict[str, Optional[float]]]:
    """Build explicit t=0 initial states for trajectory reconstruction."""
    ego_speed0 = float(ts.v_ego[0]) if len(ts.v_ego) else float("nan")
    target_speed0 = float(ts.v_sur[0]) if len(ts.v_sur) else float("nan")
    # Initial state is defined in ego frame (ego heading == 0.0), so target heading
    # must be expressed relative to ego's initial yaw.
    ego_heading0 = float(ts.psi_ego[0]) if len(ts.psi_ego) else float("nan")
    target_heading0 = float(ts.psi_sur[0]) if len(ts.psi_sur) else float("nan")
    if np.isfinite(target_heading0) and np.isfinite(ego_heading0):
        target_heading0 = target_heading0 - ego_heading0
    if np.isfinite(target_heading0):
        target_heading0 = (target_heading0 + math.pi) % (2.0 * math.pi) - math.pi

    return {
        "ego": {
            "x": _to_json_float(0.0),
            "y": _to_json_float(0.0),
            "heading": _to_json_float(0.0),
            "speed": _to_json_float(max(0.0, ego_speed0), digits=3),
        },
        "target": {
            "x": _to_json_float(float(ts.x_sur[0]) if len(ts.x_sur) else float("nan")),
            "y": _to_json_float(float(ts.y_sur[0]) if len(ts.y_sur) else float("nan")),
            "heading": _to_json_float(target_heading0),
            "speed": _to_json_float(max(0.0, target_speed0), digits=3),
        },
    }


def _mean_speed_in_interval(
    time_raw: np.ndarray,
    v: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> float:
    mask = (time_raw >= t_lo) & (time_raw < t_hi)
    if not mask.any():
        return float(np.mean(v)) if len(v) else float("nan")
    return float(np.mean(v[mask]))


def _state_at_time(
    time_raw: np.ndarray,
    v: np.ndarray,
    psi: np.ndarray,
    t_query: float,
) -> Tuple[float, float]:
    """Linearly interpolate speed and heading at t_query."""
    if len(time_raw) == 0:
        return float("nan"), float("nan")
    t0, t1 = float(time_raw[0]), float(time_raw[-1])
    tq = float(np.clip(t_query, t0, t1))
    spd = float(np.interp(tq, time_raw, v))
    psi_u = np.unwrap(np.asarray(psi, dtype=float))
    psi_q = float(np.interp(tq, time_raw, psi_u))
    psi_q = (psi_q + math.pi) % (2.0 * math.pi) - math.pi
    return spd, psi_q


def _position_at_time(
    time_raw: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_query: float,
) -> Tuple[float, float]:
    """Linearly interpolate position at t_query."""
    if len(time_raw) == 0:
        return float("nan"), float("nan")
    t0, t1 = float(time_raw[0]), float(time_raw[-1])
    tq = float(np.clip(t_query, t0, t1))
    x_q = float(np.interp(tq, time_raw, x))
    y_q = float(np.interp(tq, time_raw, y))
    return x_q, y_q


def _scalar_at_time(time_raw: np.ndarray, values: np.ndarray, t_query: float) -> float:
    """Linearly interpolate scalar value at t_query."""
    if len(time_raw) == 0:
        return float("nan")
    t0, t1 = float(time_raw[0]), float(time_raw[-1])
    tq = float(np.clip(t_query, t0, t1))
    return float(np.interp(tq, time_raw, values))


def _yaw_rate_at_time(
    time_raw: np.ndarray,
    psi_raw: np.ndarray,
    t_query: float,
) -> float:
    """Signed yaw rate dψ/dt (rad/s) at t_query from gradient of unwrapped heading."""
    if len(time_raw) < 2:
        return float("nan")
    t = np.asarray(time_raw, dtype=float)
    psi_u = np.unwrap(np.asarray(psi_raw, dtype=float))
    yr = np.gradient(psi_u, t)
    return _scalar_at_time(t, yr, t_query)


def _to_json_float(value: float, digits: int = 4) -> Optional[float]:
    """Convert finite float to rounded JSON-safe value; otherwise return None."""
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _vehicle_state_at_time(
    ts: EventTimeSeries,
    acc: np.ndarray,
    t_query: float,
    actor: str,
) -> Dict[str, Optional[float]]:
    """Interpolate actor state (x, y, speed, acc, heading) at t_query."""
    if actor == "ego":
        x, y = _position_at_time(ts.time, ts.x_ego, ts.y_ego, t_query)
        speed, heading = _state_at_time(ts.time, ts.v_ego, ts.psi_ego, t_query)
    else:
        x, y = _position_at_time(ts.time, ts.x_sur, ts.y_sur, t_query)
        speed, heading = _state_at_time(ts.time, ts.v_sur, ts.psi_sur, t_query)
    acc_q = _scalar_at_time(ts.time, acc, t_query)
    return {
        "x": _to_json_float(x),
        "y": _to_json_float(y),
        "speed": _to_json_float(speed, digits=3),
        "acc": _to_json_float(acc_q, digits=3),
        "heading": _to_json_float(heading),
    }


def _pre_impact_extremes(
    ts: EventTimeSeries,
    acc: np.ndarray,
    steer_deg: np.ndarray,
    impact_time: float,
    actor: str,
) -> Dict[str, Optional[float]]:
    """Compute pre-impact extremes for an actor (strictly before impact_time)."""
    t_arr = np.asarray(ts.time, dtype=float)
    n_t = len(t_arr)
    j_pre = int(np.searchsorted(t_arr, float(impact_time), side="left"))
    if n_t == 0:
        end_pre = 0
    elif j_pre <= 0:
        end_pre = 1
    else:
        end_pre = min(j_pre, n_t)

    if actor == "ego":
        v_pre = np.asarray(ts.v_ego, dtype=float)[:end_pre]
    else:
        v_pre = np.asarray(ts.v_sur, dtype=float)[:end_pre]
    acc_pre = np.asarray(acc, dtype=float)[:end_pre]
    steer_pre = np.asarray(steer_deg, dtype=float)[:end_pre]

    def _finite_or_nan(values: np.ndarray, fn) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            return float("nan")
        return float(fn(values[np.isfinite(values)]))

    max_speed = _finite_or_nan(v_pre, np.max)
    max_acc = _finite_or_nan(acc_pre, np.max)
    max_decel = _finite_or_nan(acc_pre, np.min)  # most negative acceleration
    max_abs_steer = _finite_or_nan(np.abs(steer_pre), np.max)

    return {
        "pre_impact_max_speed": _to_json_float(max_speed, digits=3),
        "pre_impact_max_acceleration": _to_json_float(max_acc, digits=3),
        "pre_impact_max_deceleration": _to_json_float(max_decel, digits=3),
        "pre_impact_max_abs_steering_angle": _to_json_float(max_abs_steer, digits=3),
    }


def _mean_abs_delta_heading_per_step(
    time_raw: np.ndarray,
    psi_raw: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean |Δheading| (rad) between consecutive samples within ``mask``.

    Using unwrapped heading to avoid 2π jumps. Returns NaN when fewer than 2
    valid samples are available.
    """
    t_sel = np.asarray(time_raw, dtype=float)[mask]
    psi_sel = np.asarray(psi_raw, dtype=float)[mask]
    if len(t_sel) < 2:
        return float("nan")
    psi_u = np.unwrap(psi_sel)
    dpsi = np.abs(np.diff(psi_u))
    if len(dpsi) == 0 or not np.isfinite(dpsi).any():
        return float("nan")
    return float(np.nanmean(dpsi))


def _signed_yaw_rate_series(
    time_raw: np.ndarray,
    psi_raw: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Step-wise signed dψ/dt (rad/s) for consecutive samples within ``mask``."""
    t_sel = np.asarray(time_raw, dtype=float)[mask]
    psi_sel = np.asarray(psi_raw, dtype=float)[mask]
    if len(t_sel) < 2:
        return np.array([], dtype=float)
    dt = np.diff(t_sel)
    valid = np.isfinite(dt) & (dt > 1e-6)
    dpsi = np.diff(np.unwrap(psi_sel))
    rate = np.full_like(dt, np.nan, dtype=float)
    rate[valid] = dpsi[valid] / dt[valid]
    return rate


def _yaw_rate_series(
    time_raw: np.ndarray,
    psi_raw: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Step-wise |dψ/dt| (rad/s) for consecutive samples within ``mask``.

    Returned length is ``mask.sum() - 1`` (NaN for invalid dt).
    """
    t_sel = np.asarray(time_raw, dtype=float)[mask]
    psi_sel = np.asarray(psi_raw, dtype=float)[mask]
    if len(t_sel) < 2:
        return np.array([], dtype=float)
    dt = np.diff(t_sel)
    valid = np.isfinite(dt) & (dt > 1e-6)
    dpsi = np.abs(np.diff(np.unwrap(psi_sel)))
    rate = np.full_like(dt, np.nan, dtype=float)
    rate[valid] = dpsi[valid] / dt[valid]
    return rate


def _mean_yaw_rate(
    time_raw: np.ndarray,
    psi_raw: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean |dψ/dt| (rad/s) estimated from unwrapped heading within ``mask``."""
    t_sel = np.asarray(time_raw, dtype=float)[mask]
    psi_sel = np.asarray(psi_raw, dtype=float)[mask]
    if len(t_sel) < 2:
        return float("nan")
    dt = np.diff(t_sel)
    valid = np.isfinite(dt) & (dt > 1e-6)
    if not np.any(valid):
        return float("nan")
    dpsi = np.abs(np.diff(np.unwrap(psi_sel)))
    rate = np.full_like(dt, np.nan, dtype=float)
    rate[valid] = dpsi[valid] / dt[valid]
    if not np.isfinite(rate).any():
        return float("nan")
    return float(np.nanmean(rate[np.isfinite(rate)]))


def _sum_abs_heading_change(psi_sel: np.ndarray) -> float:
    """Sum of |Δψ| between consecutive unwrapped headings (rad)."""
    if psi_sel.size < 2:
        return float("nan")
    psi_u = np.unwrap(np.asarray(psi_sel, dtype=float))
    return float(np.nansum(np.abs(np.diff(psi_u))))


def _yaw_rate_sign_change_count(signed_rates: np.ndarray) -> float:
    """Count strict sign flips on finite signed yaw-rate samples (0 ignored)."""
    x = signed_rates[np.isfinite(signed_rates)]
    if x.size < 2:
        return float("nan")
    prev: Optional[float] = None
    changes = 0
    for v in x:
        s = 0.0 if abs(v) < 1e-12 else (1.0 if v > 0 else -1.0)
        if s == 0.0:
            continue
        if prev is not None and s != prev:
            changes += 1
        prev = s
    return float(changes)


def _accel_behavior_ratios(acc: np.ndarray) -> Tuple[float, float, float]:
    """Fractions of samples with a<0, a>0, and hard decel (a<=EVENT_HARD_DECEL_ACC_MS2)."""
    a = np.asarray(acc, dtype=float)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return float("nan"), float("nan"), float("nan")
    dec = float(np.mean(fin < 0.0))
    accp = float(np.mean(fin > 0.0))
    hard = float(np.mean(fin <= float(EVENT_HARD_DECEL_ACC_MS2)))
    return dec, accp, hard


def _lin_slope_vs_time(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope dy/dt; NaN if fewer than two finite pairs."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    if int(m.sum()) < 2:
        return float("nan")
    tt = t[m]
    yy = y[m]
    tt = tt - tt[0]
    c = np.polyfit(tt, yy, 1)
    return float(c[0])


def _circular_mean_angle(rad: np.ndarray) -> float:
    """Mean direction (rad) for a sample of angles; NaN if no finite input."""
    x = np.asarray(rad, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(math.atan2(float(np.mean(np.sin(x))), float(np.mean(np.cos(x)))))


_EVENT_FEATURE_FIELDS: Tuple[str, ...] = (
    # Group A: means over event window
    "ego_mean_v", "ego_mean_acc", "ego_mean_delta_heading", "ego_mean_yaw_rate",
    "sur_mean_v", "sur_mean_acc", "sur_mean_delta_heading", "sur_mean_yaw_rate",

    # Group B: speed start/end/min/max/delta
    "ego_v_start", "ego_v_end", "ego_v_min", "ego_v_max", "ego_delta_v",
    "sur_v_start", "sur_v_end", "sur_v_min", "sur_v_max", "sur_delta_v",

    # Group C: acceleration extremes and ratios
    "ego_acc_min", "ego_acc_max", "ego_decel_ratio", "ego_accel_ratio",
    "ego_hard_decel_ratio",
    "sur_acc_min", "sur_acc_max", "sur_decel_ratio", "sur_accel_ratio",
    "sur_hard_decel_ratio",

    # Group D: heading change, yaw, lateral displacement
    "ego_total_heading_change", "sur_total_heading_change",
    "ego_abs_total_heading_change", "sur_abs_total_heading_change",
    "ego_max_abs_yaw_rate", "sur_max_abs_yaw_rate",
    "ego_yaw_rate_sign_changes", "sur_yaw_rate_sign_changes",
    "ego_lateral_disp", "sur_lateral_disp",
    "ego_lateral_disp_abs", "sur_lateral_disp_abs",

    # Group E: relative geometry over event window
    "rel_x_start", "rel_y_start",
    "rel_x_end", "rel_y_end",
    "rel_x_mean", "rel_y_mean",
    "rel_dist_start", "rel_dist_end", "rel_dist_min", "rel_dist_slope",
    "rel_bearing_start", "rel_bearing_end", "rel_bearing_mean",

    # Group F: relative dynamics over event window
    "relative_heading_start", "relative_heading_end",
    "relative_heading_mean", "relative_heading_change",
    "relative_speed_start", "relative_speed_end", "relative_speed_mean",
    "closing_speed_start", "closing_speed_end",
    "closing_speed_mean", "closing_speed_max",
    "closing_acc_mean",

    # Group G: TTC / risk evolution
    "ttc_min", "ttc_end", "ttc_slope",
    "inv_ttc_max", "time_to_min_dist",

    # Group H: vehicle dimensions (meters; from event metadata)
    "ego_width", "ego_length", "target_width", "target_length",
)


def _empty_event_feature() -> Dict[str, Optional[float]]:
    return {name: None for name in _EVENT_FEATURE_FIELDS}


def _event_vehicle_dimension_features(
    *,
    ego_width: float,
    ego_length: float,
    target_width: float,
    target_length: float,
) -> Dict[str, Optional[float]]:
    return {
        "ego_width": _to_json_float(float(ego_width), digits=3),
        "ego_length": _to_json_float(float(ego_length), digits=3),
        "target_width": _to_json_float(float(target_width), digits=3),
        "target_length": _to_json_float(float(target_length), digits=3),
    }


def _wrap_pi(angle: float) -> float:
    if not np.isfinite(angle):
        return float("nan")
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def build_event_feature(
    ts: EventTimeSeries,
    acc_ego: np.ndarray,
    acc_sur: np.ndarray,
    impact_time: float,
    *,
    ego_width: float = float("nan"),
    ego_length: float = float("nan"),
    target_width: float = float("nan"),
    target_length: float = float("nan"),
) -> Dict[str, Optional[float]]:
    """
    Summarise ego/target motion over the event window (samples with ``time <= impact_time``).

    Covers per-actor kinematic means, speed/acceleration summaries, turn and
    lateral motion, ego-frame relative geometry (``rel_*``), line-of-sight
    closing speed / TTC evolution, risk scalars (``inv_ttc_max``,
    ``time_to_min_dist``), and vehicle dimensions (Group H). See
    ``_EVENT_FEATURE_FIELDS`` for the full schema.

    All values are JSON-safe (``None`` when undefined). Positions in ``ts`` are
    expected to already be in the initial ego-local frame (see
    ``transform_to_ego_local``), which makes lateral and ``rel_*`` features
    meaningful.
    """
    dim_feats = _event_vehicle_dimension_features(
        ego_width=ego_width,
        ego_length=ego_length,
        target_width=target_width,
        target_length=target_length,
    )
    t_arr = np.asarray(ts.time, dtype=float)
    if len(t_arr) == 0:
        out = _empty_event_feature()
        out.update(dim_feats)
        return out

    mask = t_arr <= float(impact_time)
    if int(mask.sum()) < 2:
        mask = np.ones_like(t_arr, dtype=bool)

    v_ego = np.asarray(ts.v_ego, dtype=float)[mask]
    v_sur = np.asarray(ts.v_sur, dtype=float)[mask]
    acc_ego_sel = (
        np.asarray(acc_ego, dtype=float)[mask]
        if len(acc_ego) == len(t_arr)
        else np.array([], dtype=float)
    )
    acc_sur_sel = (
        np.asarray(acc_sur, dtype=float)[mask]
        if len(acc_sur) == len(t_arr)
        else np.array([], dtype=float)
    )

    def _safe_mean(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            return float("nan")
        return float(np.nanmean(values))

    def _safe_stat(values: np.ndarray, fn) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            return float("nan")
        return float(fn(values[np.isfinite(values)]))

    # -------- Group A: means --------
    ego_delta_heading = _mean_abs_delta_heading_per_step(t_arr, ts.psi_ego, mask)
    sur_delta_heading = _mean_abs_delta_heading_per_step(t_arr, ts.psi_sur, mask)
    ego_yaw_rate_mean = _mean_yaw_rate(t_arr, ts.psi_ego, mask)
    sur_yaw_rate_mean = _mean_yaw_rate(t_arr, ts.psi_sur, mask)

    # -------- Group B: speed --------
    ego_v_start = float(v_ego[0]) if v_ego.size else float("nan")
    ego_v_end = float(v_ego[-1]) if v_ego.size else float("nan")
    ego_v_min = _safe_stat(v_ego, np.min)
    ego_v_max = _safe_stat(v_ego, np.max)
    ego_delta_v = ego_v_end - ego_v_start if v_ego.size else float("nan")

    sur_v_start = float(v_sur[0]) if v_sur.size else float("nan")
    sur_v_end = float(v_sur[-1]) if v_sur.size else float("nan")
    sur_v_min = _safe_stat(v_sur, np.min)
    sur_v_max = _safe_stat(v_sur, np.max)
    sur_delta_v = sur_v_end - sur_v_start if v_sur.size else float("nan")

    # -------- Group C: acceleration --------
    ego_acc_min = _safe_stat(acc_ego_sel, np.min)
    ego_acc_max = _safe_stat(acc_ego_sel, np.max)
    ego_decel_ratio, ego_accel_ratio, ego_hard_decel_ratio = _accel_behavior_ratios(
        acc_ego_sel
    )

    sur_acc_min = _safe_stat(acc_sur_sel, np.min)
    sur_acc_max = _safe_stat(acc_sur_sel, np.max)
    sur_decel_ratio, sur_accel_ratio, sur_hard_decel_ratio = _accel_behavior_ratios(
        acc_sur_sel
    )

    # -------- Group D: heading / yaw / lateral --------
    psi_ego_sel = np.unwrap(np.asarray(ts.psi_ego, dtype=float)[mask])
    psi_sur_sel = np.unwrap(np.asarray(ts.psi_sur, dtype=float)[mask])
    if psi_ego_sel.size >= 2:
        ego_total_heading = _wrap_pi(float(psi_ego_sel[-1] - psi_ego_sel[0]))
    else:
        ego_total_heading = float("nan")
    if psi_sur_sel.size >= 2:
        sur_total_heading = _wrap_pi(float(psi_sur_sel[-1] - psi_sur_sel[0]))
    else:
        sur_total_heading = float("nan")

    psi_ego_raw_m = np.asarray(ts.psi_ego, dtype=float)[mask]
    psi_sur_raw_m = np.asarray(ts.psi_sur, dtype=float)[mask]
    ego_abs_total_heading = _sum_abs_heading_change(psi_ego_raw_m)
    sur_abs_total_heading = _sum_abs_heading_change(psi_sur_raw_m)

    ego_yaw_rates = _yaw_rate_series(t_arr, ts.psi_ego, mask)
    sur_yaw_rates = _yaw_rate_series(t_arr, ts.psi_sur, mask)
    ego_max_yr = _safe_stat(ego_yaw_rates, np.max)
    sur_max_yr = _safe_stat(sur_yaw_rates, np.max)

    ego_signed_yr = _signed_yaw_rate_series(t_arr, ts.psi_ego, mask)
    sur_signed_yr = _signed_yaw_rate_series(t_arr, ts.psi_sur, mask)
    ego_yr_sign_ch = _yaw_rate_sign_change_count(ego_signed_yr)
    sur_yr_sign_ch = _yaw_rate_sign_change_count(sur_signed_yr)

    y_ego_sel = np.asarray(ts.y_ego, dtype=float)[mask]
    y_sur_sel = np.asarray(ts.y_sur, dtype=float)[mask]
    if y_ego_sel.size >= 2:
        ego_lateral_disp = float(y_ego_sel[-1] - y_ego_sel[0])
    else:
        ego_lateral_disp = float("nan")
    if y_sur_sel.size >= 2:
        sur_lateral_disp = float(y_sur_sel[-1] - y_sur_sel[0])
    else:
        sur_lateral_disp = float("nan")
    ego_lateral_disp_abs = abs(ego_lateral_disp) if np.isfinite(ego_lateral_disp) else float("nan")
    sur_lateral_disp_abs = abs(sur_lateral_disp) if np.isfinite(sur_lateral_disp) else float("nan")

    # -------- Groups E–G: relative geometry, closing, TTC (vectorised on mask) --------
    t_pre = t_arr[mask]
    n_pre = int(t_pre.size)

    # Defaults for empty window
    nan = float("nan")
    (
        rel_x_start,
        rel_y_start,
        rel_x_end,
        rel_y_end,
        rel_x_mean,
        rel_y_mean,
        rel_dist_start,
        rel_dist_end,
        rel_dist_min,
        rel_dist_slope,
        rel_bearing_start,
        rel_bearing_end,
        rel_bearing_mean,
        relative_heading_start,
        relative_heading_end,
        relative_heading_mean,
        relative_heading_change,
        rel_speed_start,
        rel_speed_end,
        rel_speed_mean,
        closing_speed_start,
        closing_speed_end,
        closing_speed_mean,
        closing_speed_max,
        closing_acc_mean,
        ttc_min,
        ttc_end,
        ttc_slope,
        inv_ttc_max,
        time_to_min_dist,
    ) = (nan,) * 30

    if n_pre >= 1:
        x_e = np.asarray(ts.x_ego, dtype=float)[mask]
        y_e = np.asarray(ts.y_ego, dtype=float)[mask]
        x_s = np.asarray(ts.x_sur, dtype=float)[mask]
        y_s = np.asarray(ts.y_sur, dtype=float)[mask]
        rel_x = x_s - x_e
        rel_y = y_s - y_e
        rel_dist = np.hypot(rel_x, rel_y)
        rel_bearing = np.arctan2(rel_y, rel_x)

        rel_x_start = float(rel_x[0])
        rel_y_start = float(rel_y[0])
        rel_x_end = float(rel_x[-1])
        rel_y_end = float(rel_y[-1])
        rel_x_mean = _safe_mean(rel_x)
        rel_y_mean = _safe_mean(rel_y)
        rel_dist_start = float(rel_dist[0])
        rel_dist_end = float(rel_dist[-1])
        rel_dist_min = _safe_stat(rel_dist, np.min)
        if n_pre >= 2:
            rel_dist_slope = _lin_slope_vs_time(t_pre, rel_dist)
        rel_bearing_start = float(rel_bearing[0])
        rel_bearing_end = float(rel_bearing[-1])
        bear_for_mean = rel_bearing[np.isfinite(rel_bearing) & (rel_dist > 1e-6)]
        rel_bearing_mean = _circular_mean_angle(bear_for_mean) if bear_for_mean.size else nan

        psi_e = np.asarray(ts.psi_ego, dtype=float)[mask]
        psi_s = np.asarray(ts.psi_sur, dtype=float)[mask]
        v_e = np.asarray(ts.v_ego, dtype=float)[mask]
        v_s = np.asarray(ts.v_sur, dtype=float)[mask]
        dh = psi_s - psi_e
        rel_heading_wrap = np.arctan2(np.sin(dh), np.cos(dh))
        relative_heading_start = float(rel_heading_wrap[0])
        relative_heading_end = float(rel_heading_wrap[-1])
        relative_heading_mean = _circular_mean_angle(rel_heading_wrap)
        relative_heading_change = _wrap_pi(
            float(rel_heading_wrap[-1] - rel_heading_wrap[0])
        )

        vx_e = v_e * np.cos(psi_e)
        vy_e = v_e * np.sin(psi_e)
        vx_s = v_s * np.cos(psi_s)
        vy_s = v_s * np.sin(psi_s)
        vx_rel = vx_s - vx_e
        vy_rel = vy_s - vy_e
        safe_dist = np.where(rel_dist > 1e-6, rel_dist, np.nan)
        closing_speed_arr = -((rel_x * vx_rel + rel_y * vy_rel) / safe_dist)

        rel_speed_mean = _safe_mean(closing_speed_arr)
        rel_speed_start = float(closing_speed_arr[0]) if np.isfinite(closing_speed_arr[0]) else nan
        rel_speed_end = float(closing_speed_arr[-1]) if np.isfinite(closing_speed_arr[-1]) else nan
        closing_speed_mean = rel_speed_mean
        closing_speed_start = rel_speed_start
        closing_speed_end = rel_speed_end
        closing_speed_max = _safe_stat(closing_speed_arr, np.max)

        if n_pre >= 2:
            dt_seg = np.diff(t_pre)
            d_cl = np.diff(closing_speed_arr)
            ok = np.isfinite(dt_seg) & (dt_seg > 1e-9) & np.isfinite(d_cl)
            if np.any(ok):
                closing_acc_mean = float(np.mean(d_cl[ok] / dt_seg[ok]))
            else:
                closing_acc_mean = nan
        else:
            closing_acc_mean = nan

        ttc_arr = np.full_like(rel_dist, np.inf, dtype=float)
        contact = rel_dist <= 1e-6
        ttc_arr[contact] = 0.0
        closing_ok = np.isfinite(closing_speed_arr) & (closing_speed_arr > 1e-6)
        ttc_ok = (~contact) & np.isfinite(rel_dist) & closing_ok
        ttc_arr[ttc_ok] = rel_dist[ttc_ok] / closing_speed_arr[ttc_ok]

        finite_ttc = ttc_arr[np.isfinite(ttc_arr) & (ttc_arr < 1e9)]
        if finite_ttc.size:
            ttc_min = float(np.min(finite_ttc))
        ttc_end = float(ttc_arr[-1]) if np.isfinite(ttc_arr[-1]) else nan

        ttc_for_slope = ttc_arr.copy()
        ttc_for_slope[~np.isfinite(ttc_for_slope) | (ttc_for_slope >= 1e8)] = np.nan
        if n_pre >= 2 and np.isfinite(ttc_for_slope).sum() >= 2:
            ttc_slope = _lin_slope_vs_time(t_pre, ttc_for_slope)

        inv_mask = np.isfinite(ttc_arr) & (ttc_arr > 1e-3) & (ttc_arr < 1e6)
        if np.any(inv_mask):
            inv_ttc_max = float(np.max(1.0 / ttc_arr[inv_mask]))
        else:
            inv_ttc_max = nan

        if np.isfinite(rel_dist).any():
            imin = int(np.nanargmin(rel_dist))
            time_to_min_dist = float(t_pre[imin] - t_pre[0])

    return {
        "ego_mean_v": _to_json_float(_safe_mean(v_ego), digits=3),
        "ego_mean_acc": _to_json_float(_safe_mean(acc_ego_sel), digits=3),
        "ego_mean_delta_heading": _to_json_float(ego_delta_heading, digits=4),
        "ego_mean_yaw_rate": _to_json_float(ego_yaw_rate_mean, digits=4),
        "sur_mean_v": _to_json_float(_safe_mean(v_sur), digits=3),
        "sur_mean_acc": _to_json_float(_safe_mean(acc_sur_sel), digits=3),
        "sur_mean_delta_heading": _to_json_float(sur_delta_heading, digits=4),
        "sur_mean_yaw_rate": _to_json_float(sur_yaw_rate_mean, digits=4),
        "ego_v_start": _to_json_float(ego_v_start, digits=3),
        "ego_v_end": _to_json_float(ego_v_end, digits=3),
        "ego_v_min": _to_json_float(ego_v_min, digits=3),
        "ego_v_max": _to_json_float(ego_v_max, digits=3),
        "ego_delta_v": _to_json_float(ego_delta_v, digits=3),
        "sur_v_start": _to_json_float(sur_v_start, digits=3),
        "sur_v_end": _to_json_float(sur_v_end, digits=3),
        "sur_v_min": _to_json_float(sur_v_min, digits=3),
        "sur_v_max": _to_json_float(sur_v_max, digits=3),
        "sur_delta_v": _to_json_float(sur_delta_v, digits=3),
        "ego_acc_min": _to_json_float(ego_acc_min, digits=3),
        "ego_acc_max": _to_json_float(ego_acc_max, digits=3),
        "ego_decel_ratio": _to_json_float(ego_decel_ratio, digits=4),
        "ego_accel_ratio": _to_json_float(ego_accel_ratio, digits=4),
        "ego_hard_decel_ratio": _to_json_float(ego_hard_decel_ratio, digits=4),
        "sur_acc_min": _to_json_float(sur_acc_min, digits=3),
        "sur_acc_max": _to_json_float(sur_acc_max, digits=3),
        "sur_decel_ratio": _to_json_float(sur_decel_ratio, digits=4),
        "sur_accel_ratio": _to_json_float(sur_accel_ratio, digits=4),
        "sur_hard_decel_ratio": _to_json_float(sur_hard_decel_ratio, digits=4),
        "ego_total_heading_change": _to_json_float(ego_total_heading, digits=4),
        "sur_total_heading_change": _to_json_float(sur_total_heading, digits=4),
        "ego_abs_total_heading_change": _to_json_float(ego_abs_total_heading, digits=4),
        "sur_abs_total_heading_change": _to_json_float(sur_abs_total_heading, digits=4),
        "ego_max_abs_yaw_rate": _to_json_float(ego_max_yr, digits=4),
        "sur_max_abs_yaw_rate": _to_json_float(sur_max_yr, digits=4),
        "ego_yaw_rate_sign_changes": _to_json_float(ego_yr_sign_ch, digits=2),
        "sur_yaw_rate_sign_changes": _to_json_float(sur_yr_sign_ch, digits=2),
        "ego_lateral_disp": _to_json_float(ego_lateral_disp, digits=4),
        "sur_lateral_disp": _to_json_float(sur_lateral_disp, digits=4),
        "ego_lateral_disp_abs": _to_json_float(ego_lateral_disp_abs, digits=4),
        "sur_lateral_disp_abs": _to_json_float(sur_lateral_disp_abs, digits=4),
        "rel_x_start": _to_json_float(rel_x_start, digits=3),
        "rel_y_start": _to_json_float(rel_y_start, digits=3),
        "rel_x_end": _to_json_float(rel_x_end, digits=3),
        "rel_y_end": _to_json_float(rel_y_end, digits=3),
        "rel_x_mean": _to_json_float(rel_x_mean, digits=3),
        "rel_y_mean": _to_json_float(rel_y_mean, digits=3),
        "rel_dist_start": _to_json_float(rel_dist_start, digits=3),
        "rel_dist_end": _to_json_float(rel_dist_end, digits=3),
        "rel_dist_min": _to_json_float(rel_dist_min, digits=3),
        "rel_dist_slope": _to_json_float(rel_dist_slope, digits=4),
        "rel_bearing_start": _to_json_float(rel_bearing_start, digits=4),
        "rel_bearing_end": _to_json_float(rel_bearing_end, digits=4),
        "rel_bearing_mean": _to_json_float(rel_bearing_mean, digits=4),
        "relative_heading_start": _to_json_float(relative_heading_start, digits=4),
        "relative_heading_end": _to_json_float(relative_heading_end, digits=4),
        "relative_heading_mean": _to_json_float(relative_heading_mean, digits=4),
        "relative_heading_change": _to_json_float(relative_heading_change, digits=4),
        "relative_speed_start": _to_json_float(rel_speed_start, digits=3),
        "relative_speed_end": _to_json_float(rel_speed_end, digits=3),
        "relative_speed_mean": _to_json_float(rel_speed_mean, digits=3),
        "closing_speed_start": _to_json_float(closing_speed_start, digits=3),
        "closing_speed_end": _to_json_float(closing_speed_end, digits=3),
        "closing_speed_mean": _to_json_float(closing_speed_mean, digits=3),
        "closing_speed_max": _to_json_float(closing_speed_max, digits=3),
        "closing_acc_mean": _to_json_float(closing_acc_mean, digits=4),
        "ttc_min": _to_json_float(ttc_min, digits=3),
        "ttc_end": _to_json_float(ttc_end, digits=3),
        "ttc_slope": _to_json_float(ttc_slope, digits=4),
        "inv_ttc_max": _to_json_float(inv_ttc_max, digits=4),
        "time_to_min_dist": _to_json_float(time_to_min_dist, digits=3),
        **dim_feats,
    }


def _distance_between_ego_sur_at_time(ts: EventTimeSeries, t_query: float) -> float:
    """Compute ego-sur Euclidean distance at t_query."""
    x_ego, y_ego = _position_at_time(ts.time, ts.x_ego, ts.y_ego, t_query)
    x_sur, y_sur = _position_at_time(ts.time, ts.x_sur, ts.y_sur, t_query)
    return math.hypot(x_sur - x_ego, y_sur - y_ego)


def _timestamp_relative_to_start_s(ts_ms: float, start_ms: float) -> Optional[float]:
    """Return timestamp offset in seconds relative to start timestamp."""
    if not np.isfinite(ts_ms) or not np.isfinite(start_ms):
        return None
    return round((float(ts_ms) - float(start_ms)) / 1000.0, 3)


def _ttc_at_time(ts: EventTimeSeries, t_query: float) -> float:
    """
    Compute TTC at t_query based on relative position and line-of-sight closing speed.
    Returns +inf when vehicles are not closing.
    """
    x_ego, y_ego = _position_at_time(ts.time, ts.x_ego, ts.y_ego, t_query)
    x_sur, y_sur = _position_at_time(ts.time, ts.x_sur, ts.y_sur, t_query)
    rel_x = x_sur - x_ego
    rel_y = y_sur - y_ego
    dist = math.hypot(rel_x, rel_y)
    if dist <= 1e-6:
        return 0.0

    v_ego, psi_ego = _state_at_time(ts.time, ts.v_ego, ts.psi_ego, t_query)
    v_sur, psi_sur = _state_at_time(ts.time, ts.v_sur, ts.psi_sur, t_query)
    vx_ego = v_ego * math.cos(psi_ego)
    vy_ego = v_ego * math.sin(psi_ego)
    vx_sur = v_sur * math.cos(psi_sur)
    vy_sur = v_sur * math.sin(psi_sur)
    rel_vx = vx_sur - vx_ego
    rel_vy = vy_sur - vy_ego

    closing_speed = -((rel_x * rel_vx + rel_y * rel_vy) / dist)
    if closing_speed <= 1e-6:
        return float("inf")
    return dist / closing_speed


def should_process_crash_sample(
    ts: EventTimeSeries,
    impact_time: float,
    max_longitudinal_m: float = MAX_LONGITUDINAL_M,
    max_lateral_m: float = MAX_LATERAL_M,
) -> bool:
    """
    Keep sample only if, at impact_time, surrogate relative position projected onto ego
    heading stays within bounds: |longitudinal| < max_longitudinal_m and
    |lateral| < max_lateral_m. Longitudinal is along ego forward; lateral is along
    ego left (perpendicular to forward), matching vx = v*cos(psi), vy = v*sin(psi).
    """
    x_ego_imp, y_ego_imp = _position_at_time(ts.time, ts.x_ego, ts.y_ego, impact_time)
    x_sur_imp, y_sur_imp = _position_at_time(ts.time, ts.x_sur, ts.y_sur, impact_time)
    _, psi_ego_imp = _state_at_time(ts.time, ts.v_ego, ts.psi_ego, impact_time)
    rel_x = x_sur_imp - x_ego_imp
    rel_y = y_sur_imp - y_ego_imp
    if not (
        math.isfinite(rel_x)
        and math.isfinite(rel_y)
        and math.isfinite(psi_ego_imp)
    ):
        return False
    c, s = math.cos(psi_ego_imp), math.sin(psi_ego_imp)
    longitudinal_m = rel_x * c + rel_y * s
    lateral_m = -rel_x * s + rel_y * c
    return (
        abs(longitudinal_m) < max_longitudinal_m and abs(lateral_m) < max_lateral_m
    )


def build_sample_control_sequence(
    ego_controls: List[Dict],
    target_controls: List[Dict],
    ego_traj: List[Dict],
    target_traj: List[Dict],
    window_t0: float,
    impact_time: float,
    time_raw: np.ndarray,
    v_ego_raw: np.ndarray,
    psi_ego_raw: np.ndarray,
    v_sur_raw: np.ndarray,
    psi_sur_raw: np.ndarray,
    acc_ego_raw: np.ndarray,
    acc_sur_raw: np.ndarray,
    x_ego_raw: np.ndarray,
    y_ego_raw: np.ndarray,
    x_sur_raw: np.ndarray,
    y_sur_raw: np.ndarray,
    meta: EventMeta,
    ts: EventTimeSeries,
    nhtsa_code: str,
    steer_ego_raw: np.ndarray,
    steer_sur_raw: np.ndarray,
    window_s: float = SAMPLE_WINDOW_S,
    dt_ctrl: float = DT_CTRL,
    state_time_s: float = SAMPLE_STATE_T_S,
) -> List[Dict[str, object]]:
    """
    Slide a fixed window along event_control_sequence from the start.
    Each window dict has ego/target slices, start_index, impact_time_index
    (local index on the trailing 12-step / 6s horizon, 0..11),
    stage_time_indices (same 12-step horizon; risk/emergency stage onsets),
    ego_v_mean, target_v_mean, ego_current_state, target_current_state
    (including acc, yaw_rate), and description (RAG text for that window's
    pre-impact raw kinematics within [t_lo, t_window_end)).
    """
    n = max(1, int(math.ceil(window_s / dt_ctrl)))
    le, lt = len(ego_controls), len(target_controls)
    le_t, lt_t = len(ego_traj), len(target_traj)
    if le == 0 or lt == 0 or le_t == 0 or lt_t == 0:
        return []
    L = min(le, lt, le_t, lt_t)
    if L < n:
        return []

    rel_impact_s = impact_time - window_t0
    # Convert impact time (relative seconds) into an index consistent with this
    # function's discrete timeline where step k corresponds to t = (k+1)*dt_ctrl.
    # Using floor() makes rel_impact_s == window_s land at index n (out of range),
    # producing impact_time_index=None for the first window. Use ceil()-1 instead.
    i_impact_global = int(math.ceil(rel_impact_s / dt_ctrl)) - 1
    if i_impact_global < 0:
        i_impact_global = 0
    elif i_impact_global >= L:
        i_impact_global = L - 1

    windows: List[Dict[str, object]] = []
    start = 0
    while start + n <= L:
        local = i_impact_global - start
        if local < 0:
            impact_idx: Optional[int] = None
        elif local >= n:
            impact_idx = None
        else:
            impact_idx = local

        t_lo = window_t0 + start * dt_ctrl
        t_window_end = window_t0 + (start + n) * dt_ctrl
        t_hi = min(t_lo + SAMPLE_V_MEAN_FIRST_S, t_window_end)
        ego_v_mean = round(_mean_speed_in_interval(time_raw, v_ego_raw, t_lo, t_hi), 2)
        target_v_mean = round(_mean_speed_in_interval(time_raw, v_sur_raw, t_lo, t_hi), 2)
        t_state = window_t0 + start * dt_ctrl + state_time_s
        ego_spd, ego_psi = _state_at_time(time_raw, v_ego_raw, psi_ego_raw, t_state)
        tgt_spd, tgt_psi = _state_at_time(time_raw, v_sur_raw, psi_sur_raw, t_state)
        ego_acc_q = _scalar_at_time(time_raw, acc_ego_raw, t_state)
        tgt_acc_q = _scalar_at_time(time_raw, acc_sur_raw, t_state)
        ego_yaw_rate = _yaw_rate_at_time(time_raw, psi_ego_raw, t_state)
        tgt_yaw_rate_rel = (
            _yaw_rate_at_time(time_raw, psi_sur_raw, t_state) - ego_yaw_rate
        )

        # Trajectory slice transformed to ego frame at state_time_s
        # Use trajectory point at state_time so ego is exactly (0,0)
        state_k = int(state_time_s / dt_ctrl) - 1
        x_ref = ego_traj[start + state_k]["x"]
        y_ref = ego_traj[start + state_k]["y"]
        cos_a = math.cos(-ego_psi)
        sin_a = math.sin(-ego_psi)

        # Target position in ego frame at state_time_s (for current state deltas)
        tt_state = target_traj[start + state_k]
        dx_t_state, dy_t_state = tt_state["x"] - x_ref, tt_state["y"] - y_ref
        delta_x = round(cos_a * dx_t_state - sin_a * dy_t_state, 4)
        delta_y = round(sin_a * dx_t_state + cos_a * dy_t_state, 4)
        relative_speed = round(tgt_spd - ego_spd, 3)
        relative_acc = round(tgt_acc_q - ego_acc_q, 3)

        ego_slice: List[Dict[str, object]] = []
        tar_slice: List[Dict[str, object]] = []
        ego_traj_slice = []
        tar_traj_slice = []
        for k in range(n):
            t_rel = (k + 1) * dt_ctrl

            eg = {**ego_controls[start + k]}
            tg = {**target_controls[start + k]}
            eg["t"] = round(t_rel, 1)
            tg["t"] = round(t_rel, 1)

            et = ego_traj[start + k]
            tt = target_traj[start + k]
            dx_e, dy_e = et["x"] - x_ref, et["y"] - y_ref
            ego_x = round(cos_a * dx_e - sin_a * dy_e, 4)
            ego_y = round(sin_a * dx_e + cos_a * dy_e, 4)
            ego_traj_slice.append({"t": round(t_rel, 1), "x": ego_x, "y": ego_y})

            dx_t, dy_t = tt["x"] - x_ref, tt["y"] - y_ref
            tar_x = round(cos_a * dx_t - sin_a * dy_t, 4)
            tar_y = round(sin_a * dx_t + cos_a * dy_t, 4)
            tar_traj_slice.append({"t": round(t_rel, 1), "x": tar_x, "y": tar_y})

            ego_slice.append(eg)
            tar_slice.append(tg)

        # impact_time_index and stage classification use the trailing REF_TRAJ_STEPS
        # (12 / 6s), matching ref trajectory, anchor horizon, and inference futures.
        stage_start = max(0, n - REF_TRAJ_STEPS)
        if impact_idx is not None:
            anchor_impact_idx = int(impact_idx) - stage_start
            if anchor_impact_idx < 0 or anchor_impact_idx >= REF_TRAJ_STEPS:
                impact_idx = None
            else:
                impact_idx = anchor_impact_idx

        stage_out = classify_interaction_stages(
            ego_traj_slice[stage_start:],
            tar_traj_slice[stage_start:],
            dt_default=dt_ctrl,
            ego_length=meta.ego_length,
            ego_width=meta.ego_width,
            target_length=meta.target_length,
            target_width=meta.target_width,
        )
        stage_time_indices = sanitize_stage_time_indices_vs_impact(
            _stage_time_indices(stage_out["stage_ids"]),
            impact_idx,
        )

        windows.append({
            "start_index": start,
            "ego_v_mean": ego_v_mean,
            "target_v_mean": target_v_mean,
            "ego_current_state": {
                "heading": 0.0,
                "speed": round(ego_spd, 3),
                "acc": round(ego_acc_q, 3),
                "yaw_rate": round(ego_yaw_rate, 4),
            },
            "target_current_state": {
                "heading": round(((tgt_psi - ego_psi + math.pi) % (2.0 * math.pi)) - math.pi, 4),
                "speed": round(tgt_spd, 3),
                "acc": round(tgt_acc_q, 3),
                "yaw_rate": round(tgt_yaw_rate_rel, 4),
                "delta_x": delta_x,
                "delta_y": delta_y,
                "relative_speed": relative_speed,
                "relative_acc": relative_acc,
            },
            "ego_control_sequence": ego_slice,
            "target_control_sequence": tar_slice,
            "ego_trajectory_sequence": ego_traj_slice,
            "target_trajectory_sequence": tar_traj_slice,
            "impact_time_index": impact_idx,
            "stage_time_indices": stage_time_indices,
            "description": generate_description(
                meta,
                ts,
                nhtsa_code,
                acc_ego_raw,
                steer_ego_raw,
                acc_sur_raw,
                steer_sur_raw,
                impact_time,
                t_window_lo=t_lo,
                t_window_hi=t_window_end,
            ),
        })
        start += 1

    return windows


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

def _classify_leading(ts: EventTimeSeries, impact_time_s: float) -> str:
    """Sub-classify 'leading' using surrogate speed."""
    end_t = impact_time_s
    t = np.asarray(ts.time, dtype=float)
    v_sur = np.asarray(ts.v_sur, dtype=float)
    pre = t < end_t
    if not np.any(pre):
        return "16"
    v_pre = v_sur[pre]
    n = len(v_pre)
    k = max(1, n // 5)
    v_start = float(np.mean(v_pre[:k]))
    v_end = float(np.mean(v_pre[-k:]))
    v_min = float(np.min(v_pre))

    if v_min < 0.5 and v_end < 0.5:
        return "1"  # Lead Vehicle Stopped
    deltas = v_pre - v_start
    if float(np.min(deltas)) < -1.0:
        return "2"  # Lead Vehicle Decelerating
    if float(np.max(deltas)) > 1.0:
        return "15"  # Lead Vehicle Accelerating
    return "7"      # Lead Vehicle at Lower Constant Speed


def _classify_adjacent_lane(ts: EventTimeSeries, impact_time_s: float) -> str:
    """Sub-classify adjacent_lane.

    Base behavior: lane change (fast lateral) vs drift (slow).
    Extra: when kinematics strongly indicate junction/opposite-direction
    interactions, emit the corresponding NHTSA codes (3/6/8/14).
    """
    t = np.asarray(ts.time, dtype=float)
    pre = t < impact_time_s
    if int(np.count_nonzero(pre)) < 2:
        return "16"

    # Reuse junction/opposite-direction kinematics to catch cases that are
    # mislabeled as adjacent_lane in metadata.
    k = _impact_kinematics(ts, impact_time_s)
    opposite = _kinematics_opposite(k, ts, impact_time_s)

    # 3: Left Turn Across Path From Opposite Directions at Junction
    # Only emit when we have both the left-turn-across cue and opposite travel.
    if _kinematics_left_turn_across_opposite(k):
        return "3"

    # 8: Vehicle(s) - Opposite Direction
    if opposite:
        return "8"

    # 6/14: Vehicle(s) Turning - Same Direction / Vehicle Turning Right at Junction
    turn_branch = _refine_junction_turn_branch(k, default_code="4", opposite=False)
    if turn_branch in ("6", "14"):
        return turn_branch

    y_pre = np.asarray(ts.y_sur, dtype=float)[pre]
    dy_sur = np.abs(np.diff(y_pre))
    max_lat_rate = float(np.max(dy_sur)) / DT_RAW if len(dy_sur) > 0 else 0.0
    if max_lat_rate > 1.0:
        return "4"   # Changing Lanes
    return "10"       # Drifting


def _median_velocity_xy(
    x: np.ndarray,
    y: np.ndarray,
    t_pre: np.ndarray,
) -> Tuple[float, float]:
    """Median (vx, vy) from position gradients over pre-impact samples."""
    if len(t_pre) < 2:
        return 0.0, 0.0
    dx = np.gradient(np.asarray(x, dtype=float), t_pre)
    dy = np.gradient(np.asarray(y, dtype=float), t_pre)
    return float(np.nanmedian(dx)), float(np.nanmedian(dy))


def _classify_parked(ts: EventTimeSeries, impact_time_s: float) -> str:
    """Sub-classify parked: same direction (parking) vs opposite (backing)."""
    t = np.asarray(ts.time, dtype=float)
    pre = t < impact_time_s
    if not np.any(pre):
        return "13"  # Vehicle(s) Parking - Same Direction
    t_pre = t[pre]

    ex, ey = _median_velocity_xy(ts.x_ego[pre], ts.y_ego[pre], t_pre)
    sx, sy = _median_velocity_xy(ts.x_sur[pre], ts.y_sur[pre], t_pre)
    v_ego = np.hypot(ex, ey)
    v_sur = np.hypot(sx, sy)

    spd_min = 0.2

    if v_sur >= spd_min and v_ego >= spd_min:
        cos_align = (ex * sx + ey * sy) / (v_ego * v_sur + 1e-9)
        return "9" if cos_align < 0.0 else "13"
    if v_sur >= spd_min and v_ego < spd_min:
        v_s = np.asarray(ts.v_ego[pre], dtype=float)
        if np.isfinite(v_s).any() and float(np.nanmedian(v_s)) < -0.15:
            return "9"  # Backing Up Into Another Vehicle
        return "13"

    psi_s = float(np.nanmedian(np.unwrap(np.asarray(ts.psi_sur[pre], dtype=float))))
    fx, fy = math.cos(psi_s), math.sin(psi_s)
    cos_ego_to_sur_forward = ex * fx + ey * fy
    if cos_ego_to_sur_forward < -0.15:
        return "9"
    if cos_ego_to_sur_forward > 0.15:
        return "13"

    v = np.asarray(ts.v_ego[pre], dtype=float)
    if np.isfinite(v).any() and float(np.nanmedian(v)) < -0.15:
        return "9"
    x = np.asarray(ts.x_ego, dtype=float)[pre]
    y = np.asarray(ts.y_ego, dtype=float)[pre]
    psi_e = np.unwrap(np.asarray(ts.psi_ego, dtype=float)[pre])
    if len(t_pre) >= 2:
        dx = np.gradient(x, t_pre)
        dy = np.gradient(y, t_pre)
        v_long = dx * np.cos(psi_e) + dy * np.sin(psi_e)
        if float(np.nanmedian(v_long)) < -0.25:
            return "9"
    return "13"


def _is_opposite_direction_at_impact(
    ts: EventTimeSeries,
    impact_time_s: float,
    min_speed_mps: float = OPPOSITE_MIN_SPEED_MPS,
    cos_threshold: float = OPPOSITE_COS_ALIGN_MAX,
) -> bool:
    """Whether ego and target move in opposite directions at impact."""
    v_ego, psi_ego = _state_at_time(ts.time, ts.v_ego, ts.psi_ego, impact_time_s)
    v_sur, psi_sur = _state_at_time(ts.time, ts.v_sur, ts.psi_sur, impact_time_s)

    if not np.isfinite(v_ego) or not np.isfinite(v_sur):
        return False
    if abs(v_ego) < min_speed_mps or abs(v_sur) < min_speed_mps:
        return False

    vx_ego = v_ego * math.cos(psi_ego)
    vy_ego = v_ego * math.sin(psi_ego)
    vx_sur = v_sur * math.cos(psi_sur)
    vy_sur = v_sur * math.sin(psi_sur)

    norm_ego = math.hypot(vx_ego, vy_ego)
    norm_sur = math.hypot(vx_sur, vy_sur)
    if norm_ego < min_speed_mps or norm_sur < min_speed_mps:
        return False

    cos_align = (vx_ego * vx_sur + vy_ego * vy_sur) / (norm_ego * norm_sur + 1e-9)
    return cos_align <= cos_threshold


def _pre_impact_mask(time_raw: np.ndarray, impact_time_s: float) -> np.ndarray:
    t = np.asarray(time_raw, dtype=float)
    pre = t < float(impact_time_s)
    if not np.any(pre):
        pre = t <= float(impact_time_s)
    return pre


def _net_heading_change(psi_raw: np.ndarray) -> float:
    psi = np.asarray(psi_raw, dtype=float)
    if psi.size < 2:
        return 0.0
    psi_u = np.unwrap(psi)
    return _wrap_pi(float(psi_u[-1] - psi_u[0]))


def _mean_signed_yaw_rate(
    time_raw: np.ndarray,
    psi_raw: np.ndarray,
    mask: np.ndarray,
) -> float:
    rates = _signed_yaw_rate_series(time_raw, psi_raw, mask)
    fin = rates[np.isfinite(rates)]
    if fin.size == 0:
        return 0.0
    return float(np.nanmean(fin))


def _impact_kinematics(ts: EventTimeSeries, impact_time_s: float) -> ImpactKinematics:
    """Longitudinal/lateral offset, relative heading, and turn cues at impact."""
    t_imp = float(impact_time_s)
    x_e, y_e = _position_at_time(ts.time, ts.x_ego, ts.y_ego, t_imp)
    x_s, y_s = _position_at_time(ts.time, ts.x_sur, ts.y_sur, t_imp)
    _, psi_e = _state_at_time(ts.time, ts.v_ego, ts.psi_ego, t_imp)
    v_s, psi_s = _state_at_time(ts.time, ts.v_sur, ts.psi_sur, t_imp)
    c, s = math.cos(psi_e), math.sin(psi_e)
    dx, dy = x_s - x_e, y_s - y_e
    lon = dx * c + dy * s
    lat = -dx * s + dy * c
    rel_h = _wrap_pi(psi_s - psi_e)

    v_ego, _ = _state_at_time(ts.time, ts.v_ego, ts.psi_ego, t_imp)
    vx_e = v_ego * math.cos(psi_e)
    vy_e = v_ego * math.sin(psi_e)
    vx_s = v_s * math.cos(psi_s)
    vy_s = v_s * math.sin(psi_s)
    n_e = math.hypot(vx_e, vy_e)
    n_s = math.hypot(vx_s, vy_s)
    if n_e < 0.2 or n_s < 0.2:
        cos_align = float("nan")
    else:
        cos_align = (vx_e * vx_s + vy_e * vy_s) / (n_e * n_s + 1e-9)

    pre = _pre_impact_mask(ts.time, t_imp)
    ego_h = _net_heading_change(np.asarray(ts.psi_ego, dtype=float)[pre])
    sur_h = _net_heading_change(np.asarray(ts.psi_sur, dtype=float)[pre])
    ego_abs = _sum_abs_heading_change(np.asarray(ts.psi_ego, dtype=float)[pre])
    sur_abs = _sum_abs_heading_change(np.asarray(ts.psi_sur, dtype=float)[pre])
    if not np.isfinite(ego_abs):
        ego_abs = 0.0
    if not np.isfinite(sur_abs):
        sur_abs = 0.0

    return ImpactKinematics(
        lon=float(lon),
        lat=float(lat),
        rel_h=float(rel_h),
        cos_align=float(cos_align),
        ego_heading_change=float(ego_h),
        sur_heading_change=float(sur_h),
        ego_abs_heading_change=float(ego_abs),
        sur_abs_heading_change=float(sur_abs),
        ego_mean_signed_yaw=_mean_signed_yaw_rate(ts.time, ts.psi_ego, pre),
        sur_mean_signed_yaw=_mean_signed_yaw_rate(ts.time, ts.psi_sur, pre),
    )


def _pre_impact_velocity_opposite(
    ts: EventTimeSeries,
    impact_time_s: float,
    min_speed_mps: float = OPPOSITE_MIN_SPEED_MPS,
    cos_threshold: float = OPPOSITE_COS_ALIGN_MAX,
) -> bool:
    """Opposite travel from pre-impact median velocity (robust when impact speeds are low)."""
    pre = _pre_impact_mask(ts.time, impact_time_s)
    t_pre = np.asarray(ts.time, dtype=float)[pre]
    if len(t_pre) < 2:
        return False
    ex, ey = _median_velocity_xy(ts.x_ego[pre], ts.y_ego[pre], t_pre)
    sx, sy = _median_velocity_xy(ts.x_sur[pre], ts.y_sur[pre], t_pre)
    v_e = math.hypot(ex, ey)
    v_s = math.hypot(sx, sy)
    if v_e < min_speed_mps or v_s < min_speed_mps:
        return False
    cos_align = (ex * sx + ey * sy) / (v_e * v_s + 1e-9)
    return cos_align <= cos_threshold


def _kinematics_opposite(k: ImpactKinematics, ts: EventTimeSeries, impact_time_s: float) -> bool:
    """NHTSA 8 cue: vehicles traveling in opposite directions (turn not required)."""
    if _is_opposite_direction_at_impact(ts, impact_time_s):
        return True
    if _pre_impact_velocity_opposite(ts, impact_time_s):
        return True
    if abs(k.rel_h) >= OPPOSITE_REL_HEADING_MIN_RAD:
        return True
    return np.isfinite(k.cos_align) and k.cos_align <= OPPOSITE_COS_ALIGN_MAX


def _kinematics_straight_crossing(k: ImpactKinematics) -> bool:
    if k.sur_abs_heading_change >= JUNCTION_STRAIGHT_CROSS_MAX_TURN_RAD:
        return False
    return (
        abs(k.lat) >= 3.0
        and abs(k.lat) >= abs(k.lon) * 0.35
    )


def _kinematics_left_turn_across(k: ImpactKinematics) -> bool:
    """Left turn across path: lateral crossing + left turn on target vehicle."""
    if k.sur_abs_heading_change < JUNCTION_LEFT_TURN_MIN_RAD or abs(k.lat) < JUNCTION_LAT_MIN_M:
        return False
    return k.sur_heading_change > 0.2 or k.sur_mean_signed_yaw > 0.04


def _kinematics_turn_maneuver(k: ImpactKinematics, min_abs_turn: float = 0.2) -> bool:
    return k.sur_abs_heading_change >= min_abs_turn


def _kinematics_right_turn(k: ImpactKinematics) -> bool:
    return (k.sur_heading_change < -0.25) and (k.sur_mean_signed_yaw < -0.04)


def _kinematics_left_turn(k: ImpactKinematics) -> bool:
    return (k.sur_heading_change > 0.25) and (k.sur_mean_signed_yaw > 0.04)


def _kinematics_left_turn_across_opposite(k: ImpactKinematics) -> bool:
    """NHTSA 3: left turn across path at a junction.

    Historically this was gated by an "opposite-direction" cue, but that gate
    is overly restrictive for noisy heading/velocity estimates and leads to
    under-matching code 3. Keep the label driven by the left-turn-across-path
    kinematics alone.
    """
    return _kinematics_left_turn_across(k)


def _kinematics_opposite_direction(k: ImpactKinematics, opposite: bool) -> bool:
    """NHTSA 8: opposite-direction travel; turn is not required."""
    return opposite


def _refine_junction_turn_branch(
    k: ImpactKinematics,
    default_code: str,
    *,
    opposite: bool,
) -> Optional[str]:
    """NHTSA 6 / 14: same-direction turn at junction (not opposite-direction)."""
    if opposite or not _kinematics_turn_maneuver(k, 0.2):
        return None
    if _kinematics_right_turn(k):
        return "14"
    if _kinematics_left_turn(k):
        return "6"
    # Do not force a 6/14 label when the turn direction cue is weak/ambiguous.
    return default_code if default_code in ("6", "14") else None


def _refine_junction_nhtsa(
    ts: EventTimeSeries,
    impact_time_s: float,
    default_code: str,
) -> str:
    """
    Refine junction / path-crossing labels using impact geometry.

    Semantic priority (differs by metadata default):
      3 – opposite + left turn across path
      8 – opposite-direction travel (no turn required)
      5 – straight crossing paths (same-direction, low turn)
      6/14 – same-direction turn at junction
    """
    k = _impact_kinematics(ts, impact_time_s)
    opposite = _kinematics_opposite(k, ts, impact_time_s)
    lt3 = _kinematics_left_turn_across_opposite(k)
    opp8 = _kinematics_opposite_direction(k, opposite)
    straight5 = _kinematics_straight_crossing(k) and not opposite

    # Policy: if the motion indicates opposite-direction travel, classify as 8.
    # This intentionally overrides "left-turn-across" cues for code 3.
    if opposite:
        return "8"

    if default_code == "8":
        # oncoming / turning_into_opposite: trust metadata unless clear 3 or 5
        if lt3:
            return "3"
        if straight5:
            return "5"
        return "8"

    if default_code == "3":
        # Metadata turning_across_opposite: trust unless clear 5 or same-dir 6/14.
        # Do not downgrade to 8 on opp8 alone — opposite travel is expected here.
        if lt3:
            return "3"
        if straight5:
            return "5"
        branch = _refine_junction_turn_branch(k, default_code, opposite=opposite)
        return branch if branch is not None else "3"

    if default_code == "5":
        if straight5:
            return "5"
        if lt3:
            return "3"
        if opp8:
            return "8"
        branch = _refine_junction_turn_branch(k, default_code, opposite=opposite)
        return branch if branch is not None else "5"

    # turning_across_parallel (6) / turning_into_parallel (14)
    if lt3:
        return "3"
    if opp8:
        return "8"
    if straight5:
        return "5"
    branch = _refine_junction_turn_branch(k, default_code, opposite=opposite)
    return branch if branch is not None else default_code


def classify_event(
    meta: EventMeta,
    ts: EventTimeSeries,
    impact_time_s: float,
) -> str:
    """Map T7UUC1 conflict type to NHTSA precrash code (1-16).

    Layer-1 uses ``conflict`` / ``first``; junction conflicts are refined by
    ``_refine_junction_nhtsa`` (3/5/6/8/14) from impact kinematics.

    ``impact_time_s`` must match the impact instant used for ``ts`` (pipeline
    closest-approach time when ``use_impact_time`` is False).
    """
    conflict = meta.conflict.lower().strip()
    first = meta.first.lower().strip() if meta.first else ""

    if conflict == "leading" or first == "leading":
        return _classify_leading(ts, impact_time_s)
    if conflict == "following" or first == "following":
        return "11"  # Following Vehicle Making a Maneuver
    if conflict == "adjacent_lane" or first == "adjacent_lane":
        return _classify_adjacent_lane(ts, impact_time_s)
    if conflict == "merging" or first == "merging":
        return "4"  # Vehicle(s) Changing Lanes - Same Direction
    if conflict == "turning_across_opposite" or first == "turning_across_opposite":
        return _refine_junction_nhtsa(ts, impact_time_s, "3")
    if conflict == "turning_into_opposite" or first == "turning_into_opposite":
        return _refine_junction_nhtsa(ts, impact_time_s, "8")
    if conflict == "turning_across_parallel" or first == "turning_across_parallel":
        return _refine_junction_nhtsa(ts, impact_time_s, "6")
    if conflict == "turning_into_parallel" or first == "turning_into_parallel":
        return _refine_junction_nhtsa(ts, impact_time_s, "14")
    if conflict == "intersection_crossing" or first == "intersection_crossing":
        return _refine_junction_nhtsa(ts, impact_time_s, "5")
    if conflict == "oncoming" or first == "oncoming":
        return _refine_junction_nhtsa(ts, impact_time_s, "8")
    if conflict == "parked" or first == "parked":
        return _classify_parked(ts, impact_time_s)
    if conflict == "obstacle" or first == "obstacle":
        return "12"
    else:
        print(f"conflict: {conflict}")

    return "16"


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------

def generate_description(
    meta: EventMeta,
    ts: EventTimeSeries,
    nhtsa_code: str,
    acc_ego: np.ndarray,
    steer_ego_deg: np.ndarray,
    acc_sur: np.ndarray,
    steer_sur_deg: np.ndarray,
    impact_time: float,
    *,
    t_window_lo: Optional[float] = None,
    t_window_hi: Optional[float] = None,
) -> str:
    """Auto-generate a textual description from ego + target kinematics for RAG retrieval."""
    nhtsa_name = NHTSA_CODE_NAMES.get(nhtsa_code, "Unknown")

    t_arr = np.asarray(ts.time, dtype=float)
    n_t = len(t_arr)
    j_pre = int(np.searchsorted(t_arr, float(impact_time), side="left"))
    if n_t == 0:
        end_pre = 0
    elif j_pre <= 0:
        end_pre = 1
    else:
        end_pre = min(j_pre, n_t)

    pre = np.zeros(n_t, dtype=bool)
    if end_pre > 0:
        pre[:end_pre] = True

    win = np.ones(n_t, dtype=bool)
    if t_window_lo is not None:
        win &= t_arr >= float(t_window_lo)
    if t_window_hi is not None:
        win &= t_arr < float(t_window_hi)
    sel = pre & win

    v_sur_sel = np.asarray(ts.v_sur, dtype=float)[sel]
    if len(v_sur_sel) == 0:
        sur_v_end = 0.0
    else:
        k_end = max(1, len(v_sur_sel) // 5)
        sur_v_end = float(np.mean(v_sur_sel[-k_end:]))

    ae_pre = np.asarray(acc_ego, dtype=float)[sel]
    se_pre = np.asarray(steer_ego_deg, dtype=float)[sel]
    as_pre = np.asarray(acc_sur, dtype=float)[sel]
    ss_pre = np.asarray(steer_sur_deg, dtype=float)[sel]

    if ae_pre.size == 0:
        ae_min = ae_max = as_min = as_max = se_max = ss_max = 0.0
    else:
        ae_min, ae_max = float(np.min(ae_pre)), float(np.max(ae_pre))
        as_min, as_max = float(np.min(as_pre)), float(np.max(as_pre))
        se_max = float(np.max(np.abs(se_pre)))
        ss_max = float(np.max(np.abs(ss_pre)))

    parts = [
        f"[scenario] {nhtsa_name}",
        f"[conflict] {meta.first}",
        f"[kinematics_ego] acc_range=[{ae_min:.2f}, {ae_max:.2f}]m/s², "
        f"max_steer={se_max:.1f}deg",
        f"[kinematics_target] acc_range=[{as_min:.2f}, {as_max:.2f}]m/s², "
        f"max_steer={ss_max:.1f}deg",
    ]

    if ae_min < -2.0:
        parts.append("[behavior] ego hard braking")
    if as_min < -2.0:
        parts.append("[behavior] target hard braking")
    if se_max > 10.0:
        parts.append("[behavior] ego significant steering")
    if ss_max > 10.0:
        parts.append("[behavior] target significant steering")
    if sur_v_end < 0.5:
        parts.append("[behavior] target vehicle stopped")
    if ae_max > 2.0:
        parts.append("[behavior] ego aggressive acceleration")
    if as_max > 2.0:
        parts.append("[behavior] target aggressive acceleration")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_crash_data(
    data_dir: str,
    output_dir: str,
    max_per_class: Optional[int] = None,
    use_impact_time: bool = False,
) -> Dict:
    """
    Full pipeline: load -> window -> kinematics -> classify -> export.

    Outputs both crash_profiles.json (into *output_dir*) and
    PostgreSQL tables (via *db_dsn*).

    If *use_impact_time* is True, impact time comes from metadata
    (``meta.impact_timestamp``). If False, it is the absolute time of the
    resampled ego/target trajectory step with minimum planar distance.

    If *max_per_class* is an integer, cap samples per NHTSA class; if ``None``,
    no per-class limit is applied.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [data_dir]
    candidates.extend(sorted(p for p in data_dir.iterdir() if p.is_dir()))

    work_dirs: List[Path] = []
    for folder in candidates:
        if not (folder / "event_data.h5").is_file():
            continue
        csv_path = folder / "event_meta.csv"
        if not csv_path.is_file():
            csv_path = folder / "event_data.csv"
        if not csv_path.is_file():
            continue
        work_dirs.append(folder)

    if not work_dirs:
        raise FileNotFoundError(
            f"No event_data.h5 with event_meta.csv or event_data.csv in {data_dir.resolve()} "
            "or its immediate subdirectories"
        )

    class_buckets: Dict[str, List[CrashSample]] = {c: [] for c in NHTSA_CODE_NAMES}
    stats = {"total_events": 0, "processed": 0, "skipped_no_duration": 0,
             "skipped_no_data": 0, "skipped_short_window": 0,
             "skipped_conflict_animal": 0,
             "skipped_by_impact_distance": 0}

    for folder in work_dirs:
        h5_path = folder / "event_data.h5"
        csv_path = folder / "event_meta.csv"
        if not csv_path.is_file():
            csv_path = folder / "event_data.csv"

        print(f"\nProcessing {folder} ...")
        print(f"Loading event metadata from {csv_path} ...")
        meta_map = load_event_meta(str(csv_path))
        print(f"  {len(meta_map)} events in metadata")

        print(f"Loading event data from {h5_path} ...")
        float_data, int_data, float_cols, int_cols = load_event_data(str(h5_path))
        print(f"  {float_data.shape[0]} rows, {len(float_cols)} float cols, {len(int_cols)} int cols")

        for eid, meta in sorted(meta_map.items()):
            stats["total_events"] += 1
            conflict_lc = meta.conflict.lower().strip()
            if conflict_lc in ("animal"):
                stats["skipped_conflict_animal"] += 1
                continue
            if not meta.duration_enough:
                stats["skipped_no_duration"] += 1
                continue

            impact_time_meta = meta.impact_timestamp / 1000.0
            ts_list = extract_event_timeseries(eid, float_data, int_data, float_cols, int_cols)
            if not ts_list:
                stats["skipped_no_data"] += 1
                continue

            for ts in ts_list:
                # Decide which impact time to use *before* extract_impact_window and any
                # filtering that depends on impact_time.
                if use_impact_time:
                    impact_time = impact_time_meta
                    is_crash = meta.severity == 1
                else:
                    impact_time, is_crash = _impact_time_from_raw_closest_approach(
                        ts, fallback_time=impact_time_meta, meta=meta
                    )

                if not should_process_crash_sample(ts, impact_time):
                    stats["skipped_by_impact_distance"] += 1
                    continue

                windowed = extract_impact_window(ts, impact_time)
                if windowed is None:
                    stats["skipped_short_window"] += 1
                    continue

                windowed_global = windowed
                windowed = transform_to_ego_local(windowed)

                nhtsa_code = classify_event(meta, windowed, impact_time)
                if nhtsa_code not in class_buckets:
                    continue
                if max_per_class is not None and len(class_buckets[nhtsa_code]) >= max_per_class:
                    continue

                acc_ego, steer_ego = compute_kinematics_from_trajectory(
                    windowed.time, windowed.x_ego, windowed.y_ego
                )
                acc_sur, steer_sur = compute_kinematics_from_trajectory(
                    windowed.time, windowed.x_sur, windowed.y_sur
                )
                ego_traj = resample_to_trajectory_steps(windowed.time, windowed.x_ego, windowed.y_ego)
                target_traj = resample_to_trajectory_steps(windowed.time, windowed.x_sur, windowed.y_sur)
                # Controls are derived from global-frame trajectories so kinematics
                # (speed, heading, yaw rate) reflect true world motion rather than
                # ego-local coordinates.
                ego_traj_global = resample_to_trajectory_steps(
                    windowed_global.time, windowed_global.x_ego, windowed_global.y_ego
                )
                target_traj_global = resample_to_trajectory_steps(
                    windowed_global.time, windowed_global.x_sur, windowed_global.y_sur
                )
                target_x0_global, target_y0_global = _infer_t0_position_from_endpoint_traj(
                    target_traj_global
                )
                ego_controls = controls_from_trajectory_steps(
                    ego_traj_global,
                    initial_x=float(windowed_global.x_ego[0]),
                    initial_y=float(windowed_global.y_ego[0]),
                )
                target_controls = controls_from_trajectory_steps(
                    target_traj_global,
                    initial_x=target_x0_global,
                    initial_y=target_y0_global,
                )
                t0_ctrl = float(windowed.time[0])
                pre_impact_n = _count_resampled_steps_before_impact(
                    ego_traj, t0_ctrl, impact_time
                )
                pre_impact_n = min(
                    pre_impact_n,
                    len(target_traj),
                    len(ego_controls),
                    len(target_controls),
                )
                event_ts = {
                    "ego_trajectory_sequence": ego_traj[:pre_impact_n],
                    "target_trajectory_sequence": target_traj[:pre_impact_n],
                }
                event_cs = {
                    "ego_control_sequence": ego_controls[:pre_impact_n],
                    "target_control_sequence": target_controls[:pre_impact_n],
                }
                sample_cs = build_sample_control_sequence(
                    ego_controls,
                    target_controls,
                    ego_traj,
                    target_traj,
                    t0_ctrl,
                    impact_time,
                    windowed.time,
                    windowed.v_ego,
                    windowed.psi_ego,
                    windowed.v_sur,
                    windowed.psi_sur,
                    acc_ego,
                    acc_sur,
                    windowed.x_ego,
                    windowed.y_ego,
                    windowed.x_sur,
                    windowed.y_sur,
                    meta,
                    windowed,
                    nhtsa_code,
                    steer_ego,
                    steer_sur,
                )
                # If sample_control_sequence is empty, skip this sample entirely.
                if not sample_cs:
                    continue
                impact_state = {
                    "ego": {
                        **_vehicle_state_at_time(windowed, acc_ego, impact_time, actor="ego"),
                        **_pre_impact_extremes(
                            windowed, acc_ego, steer_ego, impact_time, actor="ego"
                        ),
                    },
                    "target": {
                        **_vehicle_state_at_time(windowed, acc_sur, impact_time, actor="target"),
                        **_pre_impact_extremes(
                            windowed, acc_sur, steer_sur, impact_time, actor="target"
                        ),
                    },
                }
                recon_init_state = build_reconstruction_initial_state(windowed)
                impact_min_dist = _to_json_float(
                    _distance_between_ego_sur_at_time(windowed, impact_time), digits=4
                )
                event_feature = build_event_feature(
                    windowed,
                    acc_ego,
                    acc_sur,
                    impact_time,
                    ego_width=meta.ego_width,
                    ego_length=meta.ego_length,
                    target_width=meta.target_width,
                    target_length=meta.target_length,
                )
                desc = generate_description(
                    meta,
                    windowed,
                    nhtsa_code,
                    acc_ego,
                    steer_ego,
                    acc_sur,
                    steer_sur,
                    impact_time,
                )

                sample = CrashSample(
                    event_id=eid,
                    target_id=ts.target_id,
                    ego_width=meta.ego_width,
                    ego_length=meta.ego_length,
                    target_width=meta.target_width,
                    target_length=meta.target_length,
                    severity=meta.severity,
                    nhtsa_code=nhtsa_code,
                    nhtsa_type=NHTSA_CODE_NAMES[nhtsa_code],
                    description=desc,
                    event_control_sequence=event_cs,
                    event_trajectory_sequence=event_ts,
                    sample_control_sequence=sample_cs,
                    reconstruction_initial_state=recon_init_state,
                    impact_time=impact_time,
                    impact_timestamp_state=impact_state,
                    impact_timestamp_mdc=impact_min_dist,
                    event_feature=event_feature,
                    is_crash=is_crash,
                )
                class_buckets[nhtsa_code].append(sample)
                stats["processed"] += 1

    # ---- Build output structure ----
    profiles: Dict = {}
    total_samples = 0
    for code in sorted(class_buckets.keys(), key=lambda x: int(x)):
        samples = class_buckets[code]
        if not samples:
            continue
        ref_trajectory = select_representative_ref_trajectory_for_class(
            samples,
            nhtsa_code=code,
            n_steps=REF_TRAJ_STEPS,
            dt_ctrl=DT_CTRL,
        )
        variants = []
        for s in samples:
            variants.append({
                "source_event_id": s.event_id,
                "target_id": s.target_id,
                "ego_width": s.ego_width,
                "ego_length": s.ego_length,
                "target_width": s.target_width,
                "target_length": s.target_length,
                "severity": s.severity,
                "description": s.description,
                "event_feature": s.event_feature,
                "event_control_sequence": s.event_control_sequence,
                "event_trajectory_sequence": s.event_trajectory_sequence,
                "reconstruction_initial_state": s.reconstruction_initial_state,
                "impact_time": s.impact_time,
                "impact_timestamp_state": s.impact_timestamp_state,
                "impact_timestamp_mdc": s.impact_timestamp_mdc,
                "sample_control_sequence": s.sample_control_sequence,
                "is_crash": s.is_crash,
            })
        
        profiles[code] = {
            "nhtsa_type": NHTSA_CODE_NAMES[code],
            "n_real_samples": len(samples),
            "ref_trajectory": ref_trajectory,
            "variants": variants,
        }
        total_samples += len(samples)

    output = {
        "metadata": {
            "source": str(data_dir.resolve()),
            "total_samples": total_samples,
            "classes_with_data": len(profiles),
            "max_per_class": max_per_class,
            "stats": stats,
        },
        "profiles": profiles,
    }

    # ---- Write JSON ----
    json_path = output_dir / "crash_profiles.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ---- Summary ----
    print(f"\n=== Processing complete ===")
    print(f"  Directories processed: {len(work_dirs)}")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Processed samples: {stats['processed']}")
    print(f"  Skipped (no duration): {stats['skipped_no_duration']}")
    print(f"  Skipped (no data): {stats['skipped_no_data']}")
    print(f"  Skipped (short window): {stats['skipped_short_window']}")
    print(f"  Skipped (conflict animal): {stats['skipped_conflict_animal']}")
    print(f"  Skipped (impact distance): {stats['skipped_by_impact_distance']}")
    print(f"  Classes with data: {len(profiles)}/21")
    for code in sorted(profiles.keys(), key=lambda x: int(x)):
        p = profiles[code]
        print(f"    P{code.zfill(2)}: {p['nhtsa_type']} -- {p['n_real_samples']} samples")
    print(f"  Output JSON: {json_path}")

    if profiles:
        from .crash_visualizer import viz_ref_trajectories
        ref_viz_dir = str(output_dir / "ref_traj_viz")
        viz_ref_trajectories(profiles=profiles, save_dir=ref_viz_dir)
        print(f"  Ref trajectory viz dir: {ref_viz_dir}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    root_dir = Path(__file__).parent.parent.parent.parent
    
    parser = argparse.ArgumentParser(description="Process T7UUC1 crash data into NHTSA profiles.")
    parser.add_argument(
        "--data_dir", type=str, default=f"{root_dir.resolve()}/raw_data/Crash",
        help="Directory: uses event_data.h5 + event_meta.csv or event_data.csv here or in each child folder",
    )
    parser.add_argument(
        "--output_dir", type=str, default=f"{root_dir.resolve()}/out/nhtsa_rag",
        help="Output directory for crash_profiles.json",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=None,
        help="Cap samples per NHTSA class (omit for no limit)",
    )
    args = parser.parse_args()

    process_crash_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
