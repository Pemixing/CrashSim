#!/usr/bin/env python3
"""
crashsim_post_rollout_eval.py

Deterministic post-rollout interpretable evaluator for CrashSim planner rollouts.

This module is intentionally self-contained: it depends only on ``numpy`` (and,
optionally, on ``torch`` at the call site) so it can be unit-tested with
synthetic trajectories without importing the full planner / dataset stack.

The single public entry point is :func:`evaluate_rollout`, which consumes the
unified ``ego_plan_fut``-derived trajectories produced by
``crashsim_planner.py`` *after* the planner rollout has completed, and returns a
per-scene record, per-step traces and data-quality flags.

Coordinate / state convention (matches ``crashsim_planner.py``):
    state = [x, y, hcos, hsin]  (unnormalized, global frame)
    heading = atan2(hsin, hcos)

Terminology note (important, see repository audit):
    The legacy field ``ttc_sec = coll_step * dt`` is a *collision time*, NOT a
    time-to-collision. This module exposes it as ``collision_time_sec`` and adds
    genuine kinematic TTC via ``min_ttc_sec`` / ``ttc_at_min_clearance_sec``.

Nothing in this module calls an LLM, reads a network resource, or runs a planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.eval.trajectory_utils import (
    DEFAULT_KINEMATICS_UPSAMPLE_DT,
    upsample_trajectory_waypoints,
)

# ---------------------------------------------------------------------------
# Versioning constants
# ---------------------------------------------------------------------------

SCENE_RECORD_SCHEMA_VERSION = "1.0.0"
RISK_DETECTION_VERSION = "risk-v1"
RESPONSE_DETECTION_VERSION = "response-v1"
FAILURE_TAXONOMY_VERSION = "taxonomy-v4.1"
TTC_METHOD_RANGE_RATE = "range_rate"
TTC_METHOD_CROSSING_WARNING = "range_rate_crossing_conflict_warning"

# ---------------------------------------------------------------------------
# Failure taxonomy (taxonomy-v4.1): 7 actionable failure patterns + 4 non-failure
# diagnostic buckets used elsewhere (Figure 4d). Panel g uses TRUE_FAILURE only.
# v4 keeps the same seven labels as v3 but gates every failure pattern on an
# adverse outcome (collision / near miss) — except the no-outcome over-caution
# pattern — so benign rollouts no longer inflate ``no_evident_response`` and
# ``unstable_evasive_maneuver``.
# v4.1 widens inadequate-lateral detection (oblique headings + lane-change category)
# and raises its evidence strength above delayed/maladaptive competitors.
# ---------------------------------------------------------------------------

TRUE_FAILURE_PATTERNS: Tuple[str, ...] = (
    "no_evident_response",
    "delayed_response",
    "insufficient_longitudinal_response",
    "inadequate_lateral_avoidance",
    "maladaptive_response",
    "overly_conservative_response",
    "unstable_evasive_maneuver",
)

NON_FAILURE_DIAGNOSTIC_LABELS: Tuple[str, ...] = (
    "safe_resolution",
    "near_miss_resolution",
    "no_risk_signal",
    "no_clear_diagnostic_conclusion",
)

FAILURE_TAXONOMY: Tuple[str, ...] = TRUE_FAILURE_PATTERNS + NON_FAILURE_DIAGNOSTIC_LABELS

# Backward-compatible aliases for records produced under taxonomy-v2.
LEGACY_FAILURE_LABEL_ALIASES: Dict[str, str] = {
    "unstable_evasive_response": "unstable_evasive_maneuver",
    "incorrect_interaction_handling": "maladaptive_response",
    "post_avoidance_recovery_failure": "unstable_evasive_maneuver",
    "recovery_failure": "unstable_evasive_maneuver",
}


def normalize_failure_label(label: Optional[str]) -> Optional[str]:
    """Map legacy taxonomy labels to taxonomy-v3 identifiers."""
    if not label:
        return None
    return LEGACY_FAILURE_LABEL_ALIASES.get(label, label)


def resolve_failure_label(record: Dict[str, Any]) -> Optional[str]:
    """Resolve the taxonomy-v3 failure label for a scene record.

    Splits legacy ``delayed_or_absent_response`` using response-onset evidence when
    available so older rollouts remain compatible with panel-g filtering.
    """
    raw = record.get("primary_rule_based_failure_label")
    if not raw:
        return None
    if raw == "delayed_or_absent_response":
        if record.get("response_onset_time_sec") is None:
            return "no_evident_response"
        return "delayed_response"
    return normalize_failure_label(raw)


def is_true_failure_pattern(label: Optional[str]) -> bool:
    """Return True when ``label`` is one of the seven actionable failure patterns."""
    norm = normalize_failure_label(label)
    return norm in TRUE_FAILURE_PATTERNS

# Outcome classes for the near-miss / collision consistency logic (section 8.6).
OUTCOME_COLLISION = "collision"
OUTCOME_GEOMETRY_WARNING = "geometry_consistency_warning"
OUTCOME_NEAR_MISS = "near_miss"
OUTCOME_SAFE = "safe_interaction"

# ---------------------------------------------------------------------------
# High-level crash-category taxonomy (specification section 十七) and the
# mapping from the nuScenes-Crash ``behavior_tag`` pre-crash scenario typology.
#
# The synthesized nuScenes-Crash scenes only carry a ``behavior_tag`` (a
# pre-crash scenario descriptor); they do NOT record ``crash_category_high``.
# We derive the coarse category from the tag so the cross-category comparison
# (figure panel c) is populated. Matching is keyword-based and case-insensitive
# so minor tag-wording changes degrade gracefully to ``other_conflicts``.
# ---------------------------------------------------------------------------

CRASH_CATEGORY_LONGITUDINAL = "longitudinal_car_following"
CRASH_CATEGORY_LANE_LATERAL = "lane_change_and_lateral_intrusion"
CRASH_CATEGORY_INTERSECTION = "intersection_crossing_and_turning"
CRASH_CATEGORY_OPPOSING = "opposing_direction"
CRASH_CATEGORY_OTHER = "other_conflicts"

CRASH_CATEGORY_ORDER: Tuple[str, ...] = (
    CRASH_CATEGORY_LONGITUDINAL,
    CRASH_CATEGORY_LANE_LATERAL,
    CRASH_CATEGORY_INTERSECTION,
    CRASH_CATEGORY_OPPOSING,
    CRASH_CATEGORY_OTHER,
)


def behavior_tag_to_crash_category(behavior_tag: Optional[str]) -> Optional[str]:
    """Map a nuScenes-Crash ``behavior_tag`` to a high-level crash category.

    Returns ``None`` when no tag is available (so callers can preserve any
    authoritative ``crash_category_high`` already present). Unknown tags map to
    ``other_conflicts`` rather than being dropped.
    """
    if not behavior_tag or not isinstance(behavior_tag, str):
        return None
    t = behavior_tag.strip().lower()
    if not t:
        return None

    # Junction / turning conflicts take priority over the generic "same
    # direction" wording that some junction tags also contain.
    if "junction" in t or "crossing paths" in t or "turning right" in t:
        return CRASH_CATEGORY_INTERSECTION
    # Head-on / opposite-direction (not at a junction; junction LTAP handled above).
    if "opposite direction" in t:
        return CRASH_CATEGORY_OPPOSING
    # Lane change / lateral drift within the same travel direction.
    if "changing lanes" in t or "drifting" in t:
        return CRASH_CATEGORY_LANE_LATERAL
    # Rear-end / car-following family (lead vehicle *, following vehicle).
    if "lead vehicle" in t or "following vehicle" in t:
        return CRASH_CATEGORY_LONGITUDINAL
    # Generic same-direction turning maneuvers -> treat as turning conflict.
    if "turning" in t:
        return CRASH_CATEGORY_INTERSECTION
    return CRASH_CATEGORY_OTHER


# Fine-grained nuScenes-Crash pattern codes for lane-change / lateral-intrusion family
# (paper panel: Lane Change and Lateral Intrusion Conflicts — P04, P10, P11, P12).
LATERAL_INTRUSION_PATTERN_CODES: Tuple[str, ...] = ("P04", "P10", "P11", "P12")


def _resolve_crash_category(scene: Dict[str, Any]) -> Optional[str]:
    """Return the high-level crash category for a scene record."""
    cat = scene.get("crash_category_high")
    if cat:
        return str(cat)
    return behavior_tag_to_crash_category(scene.get("behavior_tag"))


def _is_longitudinal_conflict_heading(rel_heading_rad: float, half_band_deg: float) -> bool:
    """True when relative heading is same-direction or head-on (not oblique/lateral)."""
    band = math.radians(float(half_band_deg))
    rh = abs(float(rel_heading_rad))
    return rh <= band or abs(rh - math.pi) <= band


def _is_lateral_conflict_context(
    scene: Dict[str, Any],
    rel_heading: Optional[float],
    thresholds: "PostRolloutThresholds",
) -> Tuple[bool, List[str]]:
    """Return whether the scene is a lateral / oblique conflict for inadequate-lateral labeling.

    taxonomy-v4.1 relaxes the v4 crossing-only band (90° ± 45°):
      - Any collision heading that is NOT clearly longitudinal/head-on counts as oblique/lateral.
      - ``lane_change_and_lateral_intrusion`` (P04/P10/P11/P12 family) always counts as lateral,
        even when ``relative_heading_at_collision_rad`` is missing or near-longitudinal.
    """
    fields: List[str] = []

    cat = _resolve_crash_category(scene)
    if cat == CRASH_CATEGORY_LANE_LATERAL:
        fields.append("crash_category_high")

    crash_fine = str(scene.get("crash_type_fine") or "").strip().upper()
    if crash_fine in LATERAL_INTRUSION_PATTERN_CODES:
        fields.append("crash_type_fine")

    if fields:
        return True, fields

    if rel_heading is None:
        return False, fields

    if _is_longitudinal_conflict_heading(
        rel_heading, thresholds.longitudinal_conflict_half_band_deg
    ):
        return False, fields

    fields.append("relative_heading_at_collision_rad")
    return True, fields


@dataclass(frozen=True)
class PostRolloutThresholds:
    """Transparent, fully-exposed thresholds for the deterministic layer.

    Every value is serialized into the scene record so downstream analysis and
    the paper methods section can reproduce the exact decision boundaries.
    """

    # Risk onset (section 8.3) — spread_v3 balanced defaults (threshold sensitivity)
    ttc_risk_threshold_sec: float = 2.5
    margin_risk_threshold_m: float = 0.75
    risk_min_consecutive_steps: int = 2

    # Response onset (section 8.4)
    longitudinal_decel_threshold_mps2: float = -0.3
    response_min_consecutive_steps: int = 1
    delayed_latency_threshold_sec: float = 1.5
    lateral_yaw_rate_threshold_radps: float = 0.15
    lateral_accel_threshold_mps2: float = 1.5
    lateral_disp_threshold_m: float = 0.5

    # Near miss (section 8.6)
    near_miss_margin_threshold_m: float = 1.0

    # Crossing-conflict TTC warning: |relative heading| within this band of 90deg
    crossing_conflict_half_band_deg: float = 45.0

    # Failure taxonomy (inadequate lateral): headings within this band of 0° or 180° are
    # treated as longitudinal/head-on; all other oblique angles count as lateral conflicts.
    longitudinal_conflict_half_band_deg: float = 35.0
    # Primary evidence strength for inadequate_lateral_avoidance (must exceed delayed_response
    # at the default latency threshold and maladaptive_response in competing lateral scenes).
    inadequate_lateral_evidence_strength: float = 0.91

    # Comfort / stability flags used by failure taxonomy (section 8.8 / 九)
    severe_jerk_mps3: float = 5.0
    severe_lat_accel_mps2: float = 3.0
    severe_decel_mps2: float = -4.0
    overly_conservative_decel_mps2: float = -3.0
    overly_conservative_progress_ratio: float = 0.6

    # Collision gap threshold used by the geometry checker (for provenance only)
    collision_gap_threshold_m: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]]) -> "PostRolloutThresholds":
        if not mapping:
            return cls()
        known = {f: mapping[f] for f in cls.__dataclass_fields__ if f in mapping}  # type: ignore[attr-defined]
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Array helpers (torch-optional)
# ---------------------------------------------------------------------------

def _to_np(value: Any) -> Optional[np.ndarray]:
    """Convert a torch tensor / numpy array / nested list to float64 ndarray.

    Returns ``None`` for ``None`` inputs. Does not import torch; instead it
    duck-types the ``detach``/``cpu``/``numpy`` chain so callers may pass either
    torch tensors or plain numpy without this module depending on torch.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    # torch.Tensor duck-typing
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().numpy().astype(np.float64)
        except Exception:
            pass
    return np.asarray(value, dtype=np.float64)


def _heading(state_xyhh: np.ndarray) -> np.ndarray:
    """Return heading angle (rad) from [..., hcos, hsin]."""
    return np.arctan2(state_xyhh[..., 3], state_xyhh[..., 2])


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _velocity_from_positions(pos: np.ndarray, dt: float) -> np.ndarray:
    """Backward finite-difference velocity; step 0 copies step 1.

    ``pos`` shape (..., T, 2). Returns same shape.
    """
    dt = max(1e-6, float(dt))
    vel = np.full_like(pos, np.nan)
    if pos.shape[-2] >= 2:
        vel[..., 1:, :] = (pos[..., 1:, :] - pos[..., :-1, :]) / dt
        vel[..., 0, :] = vel[..., 1, :]
    return vel


def _first_consecutive_true(flags: np.ndarray, min_run: int) -> Optional[int]:
    """Index of the first element that begins a run of >= min_run True values."""
    flags = np.asarray(flags, dtype=bool)
    n = flags.shape[0]
    run = 0
    start = None
    for i in range(n):
        if flags[i]:
            if run == 0:
                start = i
            run += 1
            if run >= min_run:
                return int(start)
        else:
            run = 0
            start = None
    return None


# ---------------------------------------------------------------------------
# SAT (separating-axis) oriented-box margin
# ---------------------------------------------------------------------------

def sat_margin(a_state: np.ndarray, a_lw: np.ndarray, b_state: np.ndarray, b_lw: np.ndarray) -> float:
    """Signed separating-axis margin between two oriented rectangles.

    Positive => separated on at least one axis (max over the 4 candidate axes of
    the projected gap); non-positive => overlapping. ``lw`` is (length, width).
    Mirrors the convention used by ``crashsim_planner._compute_planner_metrics``.
    """
    a_h = np.asarray(a_state[2:4], dtype=np.float64)
    b_h = np.asarray(b_state[2:4], dtype=np.float64)
    a_norm, b_norm = np.linalg.norm(a_h), np.linalg.norm(b_h)
    if a_norm <= 1e-9 or b_norm <= 1e-9:
        return float("nan")
    a_u, b_u = a_h / a_norm, b_h / b_norm
    a_v = np.array([-a_u[1], a_u[0]], dtype=np.float64)
    b_v = np.array([-b_u[1], b_u[0]], dtype=np.float64)
    delta = np.asarray(b_state[:2] - a_state[:2], dtype=np.float64)
    a_ext = np.asarray(a_lw[:2], dtype=np.float64) * 0.5
    b_ext = np.asarray(b_lw[:2], dtype=np.float64) * 0.5
    seps: List[float] = []
    for axis in (a_u, a_v, b_u, b_v):
        ra = a_ext[0] * abs(float(np.dot(a_u, axis))) + a_ext[1] * abs(float(np.dot(a_v, axis)))
        rb = b_ext[0] * abs(float(np.dot(b_u, axis))) + b_ext[1] * abs(float(np.dot(b_v, axis)))
        seps.append(abs(float(np.dot(delta, axis))) - ra - rb)
    return float(max(seps))


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

@dataclass
class EgoKinematics:
    speed: np.ndarray            # (T,)
    accel: np.ndarray            # (T,) longitudinal (d speed / dt)
    jerk: np.ndarray             # (T,)
    yaw: np.ndarray              # (T,)
    yaw_rate: np.ndarray         # (T,)
    lateral_accel: np.ndarray    # (T,) = speed * yaw_rate
    valid: np.ndarray            # (T,) bool


def compute_ego_kinematics(ego_np: np.ndarray, dt: float) -> EgoKinematics:
    dt = max(1e-6, float(dt))
    FT = int(ego_np.shape[0])
    valid = np.isfinite(ego_np[:, :4]).all(axis=1)

    speed = np.full((FT,), np.nan)
    accel = np.full((FT,), np.nan)
    jerk = np.full((FT,), np.nan)
    yaw = _heading(ego_np)
    yaw_rate = np.full((FT,), np.nan)
    lateral_accel = np.full((FT,), np.nan)

    if FT >= 2:
        step = np.diff(ego_np[:, :2], axis=0)
        seg = np.linalg.norm(step, axis=1)
        sp = seg / dt
        speed[1:] = sp
        speed[0] = sp[0] if sp.size else np.nan
        a = np.diff(speed) / dt
        accel[1:] = a
        accel[0] = accel[1] if FT >= 2 else np.nan
        j = np.diff(accel) / dt
        jerk[1:] = j
        jerk[0] = jerk[1] if jerk.size else np.nan
        dyaw = _wrap_angle(np.diff(yaw))
        yr = dyaw / dt
        yaw_rate[1:] = yr
        yaw_rate[0] = yaw_rate[1]
        lateral_accel = speed * yaw_rate

    return EgoKinematics(
        speed=speed, accel=accel, jerk=jerk, yaw=yaw,
        yaw_rate=yaw_rate, lateral_accel=lateral_accel, valid=valid,
    )


# ---------------------------------------------------------------------------
# Relative motion, TTC, per-step critical-agent traces
# ---------------------------------------------------------------------------

@dataclass
class RelativeMotion:
    min_center_distance_ts: np.ndarray       # (T,)
    min_box_separation_margin_ts: np.ndarray # (T,)
    closing_speed_ts: np.ndarray             # (T,) closing speed to nearest agent
    ttc_ts: np.ndarray                       # (T,) TTC to nearest closing agent (nan if not closing)
    critical_agent_ts: np.ndarray            # (T,) int index into other agents (-1 if none)
    ttc_method_ts: List[str]                 # (T,) per-step method label
    n_agents: int


def compute_relative_motion(
    ego_np: np.ndarray,
    ego_lw_np: np.ndarray,
    other_np: Optional[np.ndarray],
    other_lw_np: Optional[np.ndarray],
    dt: float,
    thresholds: PostRolloutThresholds,
) -> RelativeMotion:
    dt = max(1e-6, float(dt))
    FT = int(ego_np.shape[0])
    min_center = np.full((FT,), np.nan)
    min_margin = np.full((FT,), np.nan)
    closing = np.full((FT,), np.nan)
    ttc = np.full((FT,), np.nan)
    crit = np.full((FT,), -1, dtype=np.int64)
    method: List[str] = [TTC_METHOD_RANGE_RATE] * FT

    if other_np is None or other_np.size == 0 or other_np.shape[0] == 0:
        return RelativeMotion(min_center, min_margin, closing, ttc, crit, method, 0)

    N = int(other_np.shape[0])
    ego_valid = np.isfinite(ego_np[:, :4]).all(axis=1)
    oth_valid = np.isfinite(other_np[:, :, :4]).all(axis=-1)

    ego_vel = _velocity_from_positions(ego_np[None, :, :2], dt)[0]      # (T,2)
    oth_vel = _velocity_from_positions(other_np[:, :, :2], dt)          # (N,T,2)

    for t in range(FT):
        if not ego_valid[t]:
            continue
        best_d = np.inf
        best_a = -1
        best_margin = np.inf
        for a in range(N):
            if not oth_valid[a, t]:
                continue
            d = float(np.linalg.norm(other_np[a, t, :2] - ego_np[t, :2]))
            if d < best_d:
                best_d = d
                best_a = a
            m = sat_margin(ego_np[t], ego_lw_np, other_np[a, t], other_lw_np[a])
            if math.isfinite(m) and m < best_margin:
                best_margin = m
        if best_a >= 0:
            min_center[t] = best_d
            crit[t] = best_a
            if math.isfinite(best_margin):
                min_margin[t] = best_margin
            # closing speed & TTC to nearest agent (range-rate model)
            rel_pos = other_np[best_a, t, :2] - ego_np[t, :2]
            rng = float(np.linalg.norm(rel_pos))
            rel_vel = oth_vel[best_a, t] - ego_vel[t]
            if rng > 1e-6 and np.all(np.isfinite(rel_vel)):
                range_rate = float(np.dot(rel_pos, rel_vel) / rng)  # >0 => separating
                closing_speed = -range_rate                        # >0 => closing
                closing[t] = closing_speed
                if closing_speed > 1e-6:
                    ttc_val = rng / closing_speed
                    if ttc_val >= 0:
                        ttc[t] = ttc_val
                    # crossing-conflict warning
                    ego_yaw = math.atan2(ego_np[t, 3], ego_np[t, 2])
                    oth_yaw = math.atan2(other_np[best_a, t, 3], other_np[best_a, t, 2])
                    rel_heading_deg = abs(math.degrees(_wrap_angle(np.array([oth_yaw - ego_yaw]))[0]))
                    band = float(thresholds.crossing_conflict_half_band_deg)
                    if abs(rel_heading_deg - 90.0) <= band:
                        method[t] = TTC_METHOD_CROSSING_WARNING

    return RelativeMotion(min_center, min_margin, closing, ttc, crit, method, N)


# ---------------------------------------------------------------------------
# Risk onset (section 8.3)
# ---------------------------------------------------------------------------

def detect_risk_onset(
    rel: RelativeMotion,
    thresholds: PostRolloutThresholds,
    dt: float,
) -> Dict[str, Any]:
    FT = rel.ttc_ts.shape[0]
    ttc_risk = np.zeros((FT,), dtype=bool)
    margin_risk = np.zeros((FT,), dtype=bool)

    for t in range(FT):
        ttc = rel.ttc_ts[t]
        cs = rel.closing_speed_ts[t]
        if math.isfinite(ttc) and ttc <= thresholds.ttc_risk_threshold_sec and math.isfinite(cs) and cs > 0:
            ttc_risk[t] = True

    # margin decreasing
    margin = rel.min_box_separation_margin_ts
    for t in range(1, FT):
        if (
            math.isfinite(margin[t])
            and margin[t] <= thresholds.margin_risk_threshold_m
            and math.isfinite(margin[t - 1])
            and margin[t] < margin[t - 1]
        ):
            margin_risk[t] = True

    combined = ttc_risk | margin_risk
    onset = _first_consecutive_true(combined, thresholds.risk_min_consecutive_steps)

    trigger = None
    if onset is not None:
        if ttc_risk[onset] and margin_risk[onset]:
            trigger = "ttc_and_margin"
        elif ttc_risk[onset]:
            trigger = "ttc"
        else:
            trigger = "margin"

    return {
        "risk_flag_ts": combined,
        "risk_onset_step": onset,
        "risk_onset_time_sec": (float(onset * dt) if onset is not None else None),
        "risk_trigger_type": trigger,
        "risk_detection_version": RISK_DETECTION_VERSION,
    }


# ---------------------------------------------------------------------------
# Response onset (section 8.4)
# ---------------------------------------------------------------------------

def detect_response_onset(
    kin: EgoKinematics,
    risk: Dict[str, Any],
    thresholds: PostRolloutThresholds,
    dt: float,
) -> Dict[str, Any]:
    FT = kin.accel.shape[0]

    # Longitudinal: accel <= threshold for >= min consecutive steps
    long_flag = np.zeros((FT,), dtype=bool)
    for t in range(FT):
        a = kin.accel[t]
        if math.isfinite(a) and a <= thresholds.longitudinal_decel_threshold_mps2:
            long_flag[t] = True
    long_onset = _first_consecutive_true(long_flag, thresholds.response_min_consecutive_steps)

    # Lateral: combine yaw-rate change, lateral accel, and lateral displacement
    # relative to a pre-risk heading trend (avoid flagging normal turning).
    risk_onset = risk.get("risk_onset_step")
    lat_flag = np.zeros((FT,), dtype=bool)

    # pre-risk yaw-rate baseline (median over steps before risk onset)
    if risk_onset is not None and risk_onset >= 1:
        base_window = kin.yaw_rate[:risk_onset]
    else:
        base_window = kin.yaw_rate[: max(1, FT // 3)]
    base_window = base_window[np.isfinite(base_window)]
    base_yaw_rate = float(np.median(base_window)) if base_window.size else 0.0

    for t in range(FT):
        yr = kin.yaw_rate[t]
        la = kin.lateral_accel[t]
        cond_yaw = math.isfinite(yr) and abs(yr - base_yaw_rate) >= thresholds.lateral_yaw_rate_threshold_radps
        cond_lat = math.isfinite(la) and abs(la) >= thresholds.lateral_accel_threshold_mps2
        # only count as evasive if it emerges at/after risk onset (if known)
        after_risk = (risk_onset is None) or (t >= risk_onset)
        if after_risk and (cond_yaw and cond_lat):
            lat_flag[t] = True
    lat_onset = _first_consecutive_true(lat_flag, thresholds.response_min_consecutive_steps)

    long_time = float(long_onset * dt) if long_onset is not None else None
    lat_time = float(lat_onset * dt) if lat_onset is not None else None

    # combined response = earliest of the two
    candidates = [c for c in (long_onset, lat_onset) if c is not None]
    if candidates:
        resp_onset = min(candidates)
        if long_onset is not None and lat_onset is not None and long_onset == lat_onset:
            resp_type = "combined"
        elif resp_onset == long_onset:
            resp_type = "longitudinal"
        else:
            resp_type = "lateral"
    else:
        resp_onset = None
        resp_type = None

    resp_time = float(resp_onset * dt) if resp_onset is not None else None
    risk_time = risk.get("risk_onset_time_sec")

    latency = None
    event_order_warning = None
    if resp_time is not None and risk_time is not None:
        latency = float(resp_time - risk_time)
        if latency < 0:
            event_order_warning = "response_before_risk"
            latency = None  # do not allow negative latency

    return {
        "longitudinal_response_flag_ts": long_flag,
        "lateral_response_flag_ts": lat_flag,
        "longitudinal_response_onset_step": long_onset,
        "longitudinal_response_onset_time_sec": long_time,
        "lateral_response_onset_step": lat_onset,
        "lateral_response_onset_time_sec": lat_time,
        "response_onset_step": resp_onset,
        "response_onset_time_sec": resp_time,
        "response_type": resp_type,
        "response_latency_sec": latency,
        "event_order_warning": event_order_warning,
    }


# ---------------------------------------------------------------------------
# Collision severity (section 8.5) & outcome classification (section 8.6)
# ---------------------------------------------------------------------------

def compute_collision_severity(
    ego_np: np.ndarray,
    other_np: Optional[np.ndarray],
    kin: EgoKinematics,
    collision_result: Dict[str, Any],
    dt: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "collision_partner_idx": None,
        "collision_step": None,
        "collision_time_sec": None,
        "ego_speed_at_collision_mps": None,
        "relative_speed_at_collision_mps": None,
        "relative_heading_at_collision_rad": None,
        "relative_long_gap_at_collision_m": None,
        "relative_lat_gap_at_collision_m": None,
    }
    veh_coll = collision_result.get("veh_coll")
    coll_time = collision_result.get("coll_time")
    did_coll = bool(collision_result.get("did_coll", False))
    if not did_coll or veh_coll is None or coll_time is None:
        return out

    veh_coll = np.asarray(veh_coll).astype(bool)
    coll_time = np.asarray(coll_time)
    if not np.any(veh_coll):
        return out

    coll_agents = np.flatnonzero(veh_coll)
    coll_steps = coll_time[veh_coll]
    local = int(np.argmin(coll_steps))
    cs = int(coll_steps[local])
    partner = int(coll_agents[local])

    out["collision_partner_idx"] = partner + 1  # 1-based (agent 0 is ego)
    out["collision_step"] = cs
    out["collision_time_sec"] = float(cs * dt)
    if 0 <= cs < kin.speed.shape[0] and math.isfinite(kin.speed[cs]):
        out["ego_speed_at_collision_mps"] = float(kin.speed[cs])

    FT = int(ego_np.shape[0])
    if other_np is not None and 0 <= cs < FT and partner < other_np.shape[0]:
        if np.isfinite(other_np[partner, cs, :4]).all() and np.isfinite(ego_np[cs, :4]).all():
            rel_xy = other_np[partner, cs, :2] - ego_np[cs, :2]
            ego_h = ego_np[cs, 2:4]
            h_norm = float(np.linalg.norm(ego_h))
            if h_norm > 1e-6:
                h_unit = ego_h / h_norm
                left = np.array([-h_unit[1], h_unit[0]])
                out["relative_long_gap_at_collision_m"] = float(np.dot(rel_xy, h_unit))
                out["relative_lat_gap_at_collision_m"] = float(np.dot(rel_xy, left))
            ego_yaw = math.atan2(ego_np[cs, 3], ego_np[cs, 2])
            oth_yaw = math.atan2(other_np[partner, cs, 3], other_np[partner, cs, 2])
            out["relative_heading_at_collision_rad"] = float(_wrap_angle(np.array([oth_yaw - ego_yaw]))[0])
            if cs >= 1 and np.isfinite(other_np[partner, cs - 1, :2]).all() and np.isfinite(ego_np[cs - 1, :2]).all():
                ego_v = (ego_np[cs, :2] - ego_np[cs - 1, :2]) / max(1e-6, dt)
                oth_v = (other_np[partner, cs, :2] - other_np[partner, cs - 1, :2]) / max(1e-6, dt)
                out["relative_speed_at_collision_mps"] = float(np.linalg.norm(ego_v - oth_v))
    return out


def classify_outcome(
    collision_result: Dict[str, Any],
    min_box_margin: Optional[float],
    thresholds: PostRolloutThresholds,
) -> Dict[str, Any]:
    """section 8.6 near-miss vs collision consistency logic."""
    checker_collision = bool(collision_result.get("did_coll", False))
    geometry_warning = False
    if checker_collision:
        outcome = OUTCOME_COLLISION
    elif min_box_margin is not None and min_box_margin <= 0:
        outcome = OUTCOME_GEOMETRY_WARNING
        geometry_warning = True
    elif min_box_margin is not None and 0 < min_box_margin <= thresholds.near_miss_margin_threshold_m:
        outcome = OUTCOME_NEAR_MISS
    else:
        outcome = OUTCOME_SAFE

    return {
        "outcome": outcome,
        "collision": checker_collision,
        "near_miss": outcome == OUTCOME_NEAR_MISS,
        "collision_checker_result": checker_collision,
        "sat_overlap_result": (bool(min_box_margin <= 0) if min_box_margin is not None else None),
        "geometry_consistency_warning": geometry_warning,
    }


# ---------------------------------------------------------------------------
# Comfort (section 8.8) & progress (section 8.7)
# ---------------------------------------------------------------------------

def compute_comfort(kin: EgoKinematics) -> Dict[str, Any]:
    def stat(arr: np.ndarray, reducer) -> Optional[float]:
        a = arr[np.isfinite(arr)]
        return float(reducer(a)) if a.size else None

    jerk = kin.jerk[np.isfinite(kin.jerk)]
    lat = kin.lateral_accel[np.isfinite(kin.lateral_accel)]
    return {
        "ego_max_decel_mps2": stat(kin.accel, np.min),
        "ego_max_accel_mps2": stat(kin.accel, np.max),
        "ego_mean_abs_accel_mps2": stat(np.abs(kin.accel), np.mean),
        "ego_jerk_rms_mps3": (float(math.sqrt(float(np.mean(np.square(jerk))))) if jerk.size else None),
        "ego_max_abs_jerk_mps3": stat(np.abs(kin.jerk), np.max),
        "ego_lateral_accel_max_mps2": stat(np.abs(kin.lateral_accel), np.max),
        "ego_lateral_accel_rms_mps2": (float(math.sqrt(float(np.mean(np.square(lat))))) if lat.size else None),
    }


def compute_progress(ego_np: np.ndarray, ego_gt_np: Optional[np.ndarray]) -> Dict[str, Any]:
    """Displacement-based progress proxy (section 8.7).

    Reference-path projection requires a route/lane reference that is not
    available from rollout-only data here, so we expose the displacement ratio
    proxy and name it explicitly.
    """
    out: Dict[str, Any] = {
        "progress_metric_name": "displacement_based_progress_proxy",
        "progress_metric_value": None,
        "ego_net_displacement_m": None,
        "ego_gt_net_displacement_m": None,
        "ego_displacement_ratio": None,
        "route_progress_m": None,
        "normalized_route_progress": None,
    }
    valid = np.flatnonzero(np.isfinite(ego_np[:, :2]).all(axis=1))
    if valid.size >= 2:
        net = float(np.linalg.norm(ego_np[valid[-1], :2] - ego_np[valid[0], :2]))
        out["ego_net_displacement_m"] = net
    if ego_gt_np is not None and ego_gt_np.shape[0] == ego_np.shape[0]:
        gvalid = np.flatnonzero(np.isfinite(ego_gt_np[:, :2]).all(axis=1))
        if gvalid.size >= 2:
            gnet = float(np.linalg.norm(ego_gt_np[gvalid[-1], :2] - ego_gt_np[gvalid[0], :2]))
            out["ego_gt_net_displacement_m"] = gnet
            if out["ego_net_displacement_m"] is not None and gnet > 1e-3:
                ratio = float(out["ego_net_displacement_m"] / gnet)
                out["ego_displacement_ratio"] = ratio
                out["progress_metric_value"] = ratio
    return out


def compute_road_compliance() -> Dict[str, Any]:
    """Road-compliance flags (section 8.9).

    Drivable-area / lane / curb reasoning needs the map environment, which is
    intentionally not passed into this rollout-only evaluator. All flags are
    therefore ``None`` and flagged as unavailable so downstream code never
    mistakes ``None`` for ``False``.
    """
    return {
        "offroad": None,
        "offroad_duration_sec": None,
        "drivable_area_violation": None,
        "drivable_area_violation_duration_sec": None,
        "lane_departure": None,
        "curb_contact": None,
        "secondary_collision": None,
        "road_compliance_available": False,
    }


# ---------------------------------------------------------------------------
# Failure taxonomy (section 九)
# ---------------------------------------------------------------------------

def _candidate(label: str, strength: float, fields: Sequence[str]) -> Dict[str, Any]:
    return {
        "label": label,
        "evidence_strength": round(float(max(0.0, min(1.0, strength))), 4),
        "supporting_fields": list(fields),
    }


def generate_failure_candidates(
    scene: Dict[str, Any],
    thresholds: PostRolloutThresholds,
) -> Dict[str, Any]:
    """Produce ranked candidate failure mechanisms.

    Emits multiple candidates with evidence strengths; never uses causal
    language ("root cause", "would have avoided"), only evidence-linked labels.

    taxonomy-v4 strategy (keeps the same seven labels):
    - Every failure pattern requires an adverse outcome (collision or near miss),
      except ``overly_conservative_response`` which is the no-outcome over-caution
      failure. This removes benign safe rollouts that previously flooded
      ``no_evident_response`` and ``unstable_evasive_maneuver``.
    - ``no_evident_response`` fires only when no response was detected; every
      "how the response failed" label (delayed / maladaptive / inadequate lateral /
      unstable / insufficient longitudinal) requires a detected response, so they
      are mutually exclusive with it and describe response *quality* instead.
    - ``unstable_evasive_maneuver`` requires an actual maneuver plus BOTH severe
      jerk and severe lateral accel, so coarse-grid jerk noise alone cannot trigger it.
    - ``overly_conservative_response`` requires no risk and BOTH a hard brake and a
      real progress loss, so ordinary slow/short rollouts are not mislabeled.
    """
    candidates: List[Dict[str, Any]] = []

    collision = bool(scene.get("collision"))
    near_miss = bool(scene.get("near_miss"))
    risk_onset = scene.get("risk_onset_time_sec")
    response_onset = scene.get("response_onset_time_sec")
    latency = scene.get("response_latency_sec")
    resp_type = scene.get("response_type")
    max_decel = _finite(scene.get("ego_max_decel_mps2"))
    max_jerk = _finite(scene.get("ego_max_abs_jerk_mps3"))
    lat_accel_max = _finite(scene.get("ego_lateral_accel_max_mps2"))
    progress_ratio = _finite(scene.get("ego_displacement_ratio"))
    min_margin = _finite(scene.get("min_box_separation_margin_m"))
    rel_heading = _finite(scene.get("relative_heading_at_collision_rad"))
    secondary = scene.get("secondary_collision")

    risk_present = risk_onset is not None
    response_present = response_onset is not None
    # A scene only exhibits an actionable *failure* pattern when it ended badly
    # (collision or near miss) — with the single exception of the no-outcome
    # over-caution failure below. Historically every pattern could fire on safe
    # scenes, which flooded ``no_evident_response`` and ``unstable_evasive_maneuver``
    # with benign rollouts (e.g. coarse-grid jerk spikes on collision-free
    # trajectories). Gating on an adverse outcome keeps each of the seven labels
    # a genuine failure descriptor and de-concentrates the fingerprint.
    adverse_outcome = collision or near_miss

    longitudinal_response = resp_type in ("longitudinal", "combined")

    # non-failure diagnostic buckets (excluded from the failure fingerprint)
    if not adverse_outcome and risk_present and response_present:
        candidates.append(_candidate("safe_resolution", 0.7,
                                     ["risk_onset_time_sec", "response_onset_time_sec", "collision"]))
    if near_miss and not collision:
        candidates.append(_candidate("near_miss_resolution", 0.75,
                                     ["near_miss", "min_box_separation_margin_m"]))

    if adverse_outcome:
        if risk_present and not response_present:
            # No evident response: risk emerged but no braking/steering was detected.
            candidates.append(_candidate("no_evident_response", 0.85,
                                         ["collision", "risk_onset_time_sec", "response_onset_time_sec"]))
        else:
            # A response occurred (or risk was undetected): diagnose *how* the
            # response failed. These labels are mutually exclusive with
            # ``no_evident_response`` because they all require a detected response.

            # delayed response: reacted, but too late relative to risk onset
            if risk_present and latency is not None and latency >= thresholds.delayed_latency_threshold_sec:
                candidates.append(_candidate("delayed_response", min(0.9, 0.5 + 0.2 * latency),
                                             ["risk_onset_time_sec", "response_onset_time_sec", "response_latency_sec"]))

            # maladaptive response: reacted in a counterproductive direction
            if collision and scene.get("accelerate_into_collision") is True:
                candidates.append(_candidate("maladaptive_response", 0.82,
                                             ["accelerate_into_collision", "collision"]))
            if collision and rel_heading is not None and resp_type == "lateral":
                is_longitudinal_conflict = (
                    abs(rel_heading) <= math.radians(45)
                    or abs(abs(rel_heading) - math.pi) <= math.radians(45)
                )
                if is_longitudinal_conflict:
                    candidates.append(_candidate("maladaptive_response", 0.7,
                                                 ["relative_heading_at_collision_rad", "response_type", "collision"]))

            # inadequate lateral avoidance: lateral/oblique conflict handled with a
            # longitudinal-only response (braked when steering-out was needed).
            # v4.1: oblique headings count; lane-change/lateral-intrusion category (P04/P10/P11/P12)
            # counts even without a crossing angle.
            if collision and longitudinal_response:
                is_lateral, lat_fields = _is_lateral_conflict_context(scene, rel_heading, thresholds)
                if is_lateral:
                    strength = float(thresholds.inadequate_lateral_evidence_strength)
                    if (
                        "crash_category_high" in lat_fields
                        and "relative_heading_at_collision_rad" in lat_fields
                    ):
                        strength = min(0.95, strength + 0.03)
                    candidates.append(_candidate(
                        "inadequate_lateral_avoidance",
                        strength,
                        lat_fields + ["response_type", "collision"],
                    ))

            # unstable evasive maneuver: a maneuver did occur and was kinematically
            # unstable. Require BOTH severe jerk AND severe lateral accel so that
            # coarse-grid jerk noise alone cannot trigger the label.
            if response_present:
                unstable = 0.0
                fields: List[str] = []
                if (max_jerk is not None and max_jerk >= thresholds.severe_jerk_mps3
                        and lat_accel_max is not None and lat_accel_max >= thresholds.severe_lat_accel_mps2):
                    unstable = 0.72
                    fields = ["ego_max_abs_jerk_mps3", "ego_lateral_accel_max_mps2"]
                if secondary is True:
                    unstable = max(unstable, 0.7)
                    fields = fields + ["secondary_collision"]
                if unstable > 0:
                    candidates.append(_candidate("unstable_evasive_maneuver", unstable, fields))

            # insufficient longitudinal response: braked, but not an emergency-level
            # brake, and still collided
            if collision and longitudinal_response and (max_decel is None or max_decel > thresholds.severe_decel_mps2):
                strength = 0.68 if (max_decel is not None and max_decel > thresholds.longitudinal_decel_threshold_mps2) else 0.6
                candidates.append(_candidate("insufficient_longitudinal_response", strength,
                                             ["response_type", "ego_max_decel_mps2", "collision"]))

    # overly conservative response: no adverse outcome and no genuine risk, yet the
    # planner braked hard AND surrendered substantial progress (unjustified caution).
    # Both conditions are required so ordinary slow/short rollouts are not flagged.
    if not adverse_outcome and not risk_present:
        heavy_brake = max_decel is not None and max_decel <= thresholds.overly_conservative_decel_mps2
        progress_loss = progress_ratio is not None and progress_ratio <= thresholds.overly_conservative_progress_ratio
        if heavy_brake and progress_loss:
            candidates.append(_candidate("overly_conservative_response", 0.62,
                                         ["ego_max_decel_mps2", "ego_displacement_ratio"]))

    if not candidates:
        # Split the former catch-all Inconclusive bucket:
        #  - no_risk_signal: routine safe interaction with no detected risk onset
        #  - no_clear_diagnostic_conclusion: outcome/risk evidence present but ambiguous
        if not collision and not near_miss and not risk_present:
            candidates.append(_candidate("no_risk_signal", 0.55, []))
        else:
            candidates.append(_candidate("no_clear_diagnostic_conclusion", 0.5, []))

    # sort by evidence strength (desc)
    candidates.sort(key=lambda c: c["evidence_strength"], reverse=True)

    primary = candidates[0]
    alternatives = [c["label"] for c in candidates[1:]]

    warning = None
    if len(candidates) >= 2 and abs(candidates[0]["evidence_strength"] - candidates[1]["evidence_strength"]) < 0.1:
        warning = "multiple_candidate_mechanisms_with_similar_evidence_strength"
    if primary["label"] == "no_clear_diagnostic_conclusion":
        warning = "insufficient_evidence_for_confident_label"
    elif primary["label"] == "no_risk_signal":
        warning = "no_risk_onset_detected"

    return {
        "candidate_failure_mechanisms": candidates,
        "primary_rule_based_failure_label": primary["label"],
        "primary_rule_based_failure_confidence": primary["evidence_strength"],
        "alternative_failure_labels": alternatives,
        "failure_label_warning": warning,
        "failure_taxonomy_version": FAILURE_TAXONOMY_VERSION,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_rollout(
    ego_traj: Any,
    ego_traj_gt: Any,
    ego_lw: Any,
    other_traj: Any,
    other_lw: Any,
    collision_result: Dict[str, Any],
    scene_metadata: Optional[Dict[str, Any]] = None,
    planner_metadata: Optional[Dict[str, Any]] = None,
    dt: float = 0.5,
    thresholds: Optional[Any] = None,
    kinematics_upsample_dt: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate a single completed planner rollout.

    Parameters mirror the trajectories assembled by ``crashsim_planner.py`` after
    the planner produced ``ego_plan_fut``. All trajectories must be unnormalized
    in the global frame with state ``[x, y, hcos, hsin]``.

    ``collision_result`` should carry the geometry checker output::

        {"did_coll": bool, "veh_coll": (N,) bool, "coll_time": (N,) int,
         "gap_thresh": float}

    Returns ``{"scene_record", "trace_data", "data_quality_flags"}``.
    """
    if isinstance(thresholds, PostRolloutThresholds):
        th = thresholds
    elif isinstance(thresholds, dict):
        th = PostRolloutThresholds.from_mapping(thresholds)
    else:
        th = PostRolloutThresholds()

    scene_metadata = dict(scene_metadata or {})
    planner_metadata = dict(planner_metadata or {})
    collision_result = dict(collision_result or {})

    data_quality_flags: List[str] = []

    ego_np = _to_np(ego_traj)
    ego_gt_np = _to_np(ego_traj_gt)
    ego_lw_np = _to_np(ego_lw)
    other_np = _to_np(other_traj)
    other_lw_np = _to_np(other_lw)

    if ego_np is None or ego_np.ndim != 2 or ego_np.shape[0] < 1:
        raise ValueError("ego_traj must be a (FT, 4) array with at least one step")
    FT = int(ego_np.shape[0])
    dt = float(dt)

    if ego_lw_np is None:
        ego_lw_np = np.array([4.0, 2.0], dtype=np.float64)
        data_quality_flags.append("ego_lw_missing_defaulted")
    ego_lw_np = ego_lw_np.reshape(-1)

    has_other = other_np is not None and other_np.ndim == 3 and other_np.shape[0] > 0
    if not has_other:
        other_np = None
        other_lw_np = None
        data_quality_flags.append("no_other_agents")

    ego_valid_frac = float(np.mean(np.isfinite(ego_np[:, :4]).all(axis=1)))
    if ego_valid_frac < 1.0:
        data_quality_flags.append("ego_trajectory_has_invalid_steps")

    # --- kinematics, relative motion ---
    # Safety / interaction / response metrics always use the rollout coarse grid
    # (``ego_np``, ``dt`` — typically 0.5 s): TTC, clearance, risk onset,
    # response latency, collision severity.  Fine-grid upsampling below is
    # opt-in for comfort-only derivatives and temporal traces.
    kin = compute_ego_kinematics(ego_np, dt)
    kin_comfort = kin
    trace_dt = dt
    if kinematics_upsample_dt is not None:
        ego_kin_np, kin_dt = upsample_trajectory_waypoints(ego_np, dt, float(kinematics_upsample_dt))
        if ego_kin_np.shape[0] > ego_np.shape[0]:
            kin_comfort = compute_ego_kinematics(ego_kin_np, kin_dt)
            trace_dt = kin_dt
            data_quality_flags.append(
                f"kinematics_upsampled:dt={dt:.3g}->dt={kin_dt:.3g},steps={ego_kin_np.shape[0]}"
            )
    rel = compute_relative_motion(ego_np, ego_lw_np, other_np, other_lw_np, dt, th)

    min_center = _finite(np.nanmin(rel.min_center_distance_ts)) if np.any(np.isfinite(rel.min_center_distance_ts)) else None
    min_margin = _finite(np.nanmin(rel.min_box_separation_margin_ts)) if np.any(np.isfinite(rel.min_box_separation_margin_ts)) else None

    # genuine TTC
    ttc_vals = rel.ttc_ts[np.isfinite(rel.ttc_ts)]
    min_ttc = float(np.min(ttc_vals)) if ttc_vals.size else None
    ttc_at_min_clearance = None
    ttc_method = TTC_METHOD_RANGE_RATE
    if np.any(np.isfinite(rel.min_box_separation_margin_ts)):
        step_min_margin = int(np.nanargmin(rel.min_box_separation_margin_ts))
        if math.isfinite(rel.ttc_ts[step_min_margin]):
            ttc_at_min_clearance = float(rel.ttc_ts[step_min_margin])
        ttc_method = rel.ttc_method_ts[step_min_margin]
    if any(m == TTC_METHOD_CROSSING_WARNING for m in rel.ttc_method_ts):
        if "crossing_conflict_ttc_method_warning" not in data_quality_flags:
            data_quality_flags.append("crossing_conflict_ttc_method_warning")

    # --- risk & response ---
    risk = detect_risk_onset(rel, th, dt)
    response = detect_response_onset(kin, risk, th, dt)
    if response.get("event_order_warning"):
        data_quality_flags.append(response["event_order_warning"])

    # --- collision severity & outcome ---
    severity = compute_collision_severity(ego_np, other_np, kin, collision_result, dt)
    outcome = classify_outcome(collision_result, min_margin, th)
    if outcome["geometry_consistency_warning"]:
        data_quality_flags.append("geometry_consistency_warning")

    # response window (collision scenes only)
    response_window = None
    if outcome["collision"] and severity["collision_time_sec"] is not None and response["response_onset_time_sec"] is not None:
        response_window = float(severity["collision_time_sec"] - response["response_onset_time_sec"])

    # --- comfort, progress, road compliance ---
    comfort = compute_comfort(kin_comfort)
    progress = compute_progress(ego_np, ego_gt_np)
    road = compute_road_compliance()
    for k in ("offroad", "drivable_area_violation", "lane_departure", "curb_contact", "secondary_collision"):
        if road.get(k) is None:
            data_quality_flags.append(f"metric_unavailable:{k}")

    # pre-collision accel pattern (for incorrect_interaction_handling)
    accelerate_into = None
    brake_before = None
    cs = severity.get("collision_step")
    if isinstance(cs, int) and cs >= 1:
        window = kin.accel[max(0, cs - 4):cs]
        window = window[np.isfinite(window)]
        if window.size:
            brake_before = bool(np.any(window <= th.longitudinal_decel_threshold_mps2))
            accelerate_into = bool(np.mean(window) >= 0.5)

    # --- assemble scene record ---
    scene_record: Dict[str, Any] = {
        "schema_version": SCENE_RECORD_SCHEMA_VERSION,
        # identity / provenance (filled from metadata)
        "experiment_id": scene_metadata.get("experiment_id"),
        "dataset_kind": scene_metadata.get("dataset_kind"),
        "dataset_version": scene_metadata.get("dataset_version"),
        "planner": planner_metadata.get("planner"),
        "actual_planner_cfg": planner_metadata.get("actual_planner_cfg"),
        "seed": planner_metadata.get("seed"),
        "repeat_id": planner_metadata.get("repeat_id"),
        "scene_token": scene_metadata.get("scene_token"),
        "sample_token": scene_metadata.get("sample_token"),
        "source_scene_token": scene_metadata.get("source_scene_token"),
        "source_sample_token": scene_metadata.get("source_sample_token"),
        "scene_index": scene_metadata.get("scene_index"),
        "sidx": scene_metadata.get("sidx"),
        "crash_type_fine": scene_metadata.get("crash_type_fine"),
        # Prefer an authoritative category; otherwise derive it from behavior_tag
        # (nuScenes-Crash scenes only carry behavior_tag, not crash_category_high).
        "crash_category_high": (
            scene_metadata.get("crash_category_high")
            or behavior_tag_to_crash_category(scene_metadata.get("behavior_tag"))
        ),
        "behavior_tag": scene_metadata.get("behavior_tag"),
        "critical_agent_idx": (severity.get("collision_partner_idx")
                               if severity.get("collision_partner_idx") is not None
                               else _critical_agent(rel)),
        # outcome
        "collision": outcome["collision"],
        "near_miss": outcome["near_miss"],
        "outcome_class": outcome["outcome"],
        "collision_checker_result": outcome["collision_checker_result"],
        "sat_overlap_result": outcome["sat_overlap_result"],
        "geometry_consistency_warning": outcome["geometry_consistency_warning"],
        # timing (collision_time_sec is a collision time, NOT a TTC)
        "collision_time_sec": severity["collision_time_sec"],
        "collision_step": severity["collision_step"],
        "min_ttc_sec": min_ttc,
        "ttc_at_min_clearance_sec": ttc_at_min_clearance,
        "ttc_calculation_method": ttc_method,
        "min_center_distance_m": min_center,
        "min_box_separation_margin_m": min_margin,
        # risk / response process
        "risk_onset_step": risk["risk_onset_step"],
        "risk_onset_time_sec": risk["risk_onset_time_sec"],
        "risk_trigger_type": risk["risk_trigger_type"],
        "risk_detection_version": risk["risk_detection_version"],
        "longitudinal_response_onset_step": response["longitudinal_response_onset_step"],
        "longitudinal_response_onset_time_sec": response["longitudinal_response_onset_time_sec"],
        "lateral_response_onset_step": response["lateral_response_onset_step"],
        "lateral_response_onset_time_sec": response["lateral_response_onset_time_sec"],
        "response_onset_step": response["response_onset_step"],
        "response_onset_time_sec": response["response_onset_time_sec"],
        "response_type": response["response_type"],
        "response_latency_sec": response["response_latency_sec"],
        "response_window_sec": response_window,
        "event_order_warning": response["event_order_warning"],
        # collision severity
        "collision_partner_idx": severity["collision_partner_idx"],
        "ego_speed_at_collision_mps": severity["ego_speed_at_collision_mps"],
        "relative_speed_at_collision_mps": severity["relative_speed_at_collision_mps"],
        "relative_heading_at_collision_rad": severity["relative_heading_at_collision_rad"],
        "relative_long_gap_at_collision_m": severity["relative_long_gap_at_collision_m"],
        "relative_lat_gap_at_collision_m": severity["relative_lat_gap_at_collision_m"],
        "brake_before_collision": brake_before,
        "accelerate_into_collision": accelerate_into,
        # comfort
        **comfort,
        # progress
        **progress,
        # road compliance
        **road,
        # data quality
        "ego_valid_fraction": ego_valid_frac,
        "num_other_agents": (int(other_np.shape[0]) if other_np is not None else 0),
    }

    failure = generate_failure_candidates(scene_record, th)
    scene_record.update(failure)
    scene_record["thresholds"] = th.to_dict()
    scene_record["data_quality_flags"] = list(data_quality_flags)

    # --- per-step traces (section 8.2) ---
    trace_data: Dict[str, Any] = {
        "trace_dt": trace_dt,
        "trace_num_steps": int(kin_comfort.speed.shape[0]),
        "ego_speed_ts": kin_comfort.speed,
        "ego_accel_ts": kin_comfort.accel,
        "ego_jerk_ts": kin_comfort.jerk,
        "ego_yaw_rate_ts": kin_comfort.yaw_rate,
        "ego_lateral_accel_ts": kin_comfort.lateral_accel,
        "min_center_distance_ts": rel.min_center_distance_ts,
        "min_box_separation_margin_ts": rel.min_box_separation_margin_ts,
        "closing_speed_ts": rel.closing_speed_ts,
        "ttc_ts": rel.ttc_ts,
        "risk_flag_ts": risk["risk_flag_ts"].astype(np.int8),
        "longitudinal_response_flag_ts": response["longitudinal_response_flag_ts"].astype(np.int8),
        "lateral_response_flag_ts": response["lateral_response_flag_ts"].astype(np.int8),
    }

    return {
        "scene_record": _jsonable(scene_record),
        "trace_data": trace_data,
        "data_quality_flags": list(data_quality_flags),
    }


def _critical_agent(rel: RelativeMotion) -> Optional[int]:
    if not np.any(np.isfinite(rel.min_box_separation_margin_ts)):
        return None
    step = int(np.nanargmin(rel.min_box_separation_margin_ts))
    a = int(rel.critical_agent_ts[step])
    return (a + 1) if a >= 0 else None


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays and non-finite floats to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Trace serialization (section 8.2)
# ---------------------------------------------------------------------------

def save_trace(trace_data: Dict[str, Any], trace_path: str) -> Dict[str, Any]:
    """Persist per-step traces to a compressed ``.npz`` file.

    Returns the light-weight reference stored in ``results.jsonl``
    (``trace_path``, ``trace_num_steps``, ``trace_dt``).
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    scalars: Dict[str, Any] = {}
    for k, v in trace_data.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            scalars[k] = v
    np.savez_compressed(trace_path, **arrays, _scalars=np.array([repr(scalars)], dtype=object))
    return {
        "trace_path": trace_path,
        "trace_num_steps": int(trace_data.get("trace_num_steps", 0)),
        "trace_dt": float(trace_data.get("trace_dt", 0.0)),
    }


# ---------------------------------------------------------------------------
# Single-planner aggregation (section 六: single-planner report)
# ---------------------------------------------------------------------------

def aggregate_single_planner(scene_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-scene records for one planner into a compact summary.

    Deterministic descriptive statistics only; no LLM, no causal claims.
    """
    records = list(scene_records)
    n = len(records)
    interactive = [r for r in records if int(r.get("num_other_agents", 0)) > 0]
    n_int = len(interactive)
    n_coll = sum(1 for r in interactive if bool(r.get("collision")))
    n_near = sum(1 for r in interactive if bool(r.get("near_miss")))

    def med(key: str, subset: Sequence[Dict[str, Any]]) -> Optional[float]:
        vals = [_finite(r.get(key)) for r in subset]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    coll_records = [r for r in interactive if bool(r.get("collision"))]

    label_counts: Dict[str, int] = {lbl: 0 for lbl in FAILURE_TAXONOMY}
    for r in records:
        lbl = r.get("primary_rule_based_failure_label")
        if lbl in label_counts:
            label_counts[lbl] += 1

    category_counts: Dict[str, int] = {}
    for r in records:
        c = r.get("crash_category_high")
        if c:
            category_counts[c] = category_counts.get(c, 0) + 1

    return {
        "planner": records[0].get("planner") if records else None,
        "dataset_kind": records[0].get("dataset_kind") if records else None,
        "n_scenes": n,
        "n_interactive": n_int,
        "n_collision": n_coll,
        "n_near_miss": n_near,
        "collision_rate": (n_coll / n_int if n_int else None),
        "near_miss_rate": (n_near / n_int if n_int else None),
        "median_min_ttc_sec": med("min_ttc_sec", interactive),
        "median_min_box_separation_margin_m": med("min_box_separation_margin_m", interactive),
        "median_response_latency_sec": med("response_latency_sec", interactive),
        "median_impact_speed_mps": med("ego_speed_at_collision_mps", coll_records),
        "median_max_abs_jerk_mps3": med("ego_max_abs_jerk_mps3", interactive),
        "median_progress_proxy": med("progress_metric_value", interactive),
        "failure_label_counts": label_counts,
        "crash_category_counts": category_counts,
    }
