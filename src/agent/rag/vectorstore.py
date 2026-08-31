import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from sentence_transformers import SentenceTransformer

from .crash_data_processor import (
    DT_CTRL,
    NHTSA_CODE_NAMES,
    STAGE_EMERGENCY_RESPONSE,
    STAGE_RISK_AWARENESS,
    STAGE_TIME_INDEX_NAMES,
    _stage_time_indices,
    classify_interaction_stages,
    sanitize_stage_time_indices_vs_impact,
)


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_EPS = 1e-6
# Layer-2 (variant): max distinct (source_event_id, target_id) per matched precrash_code.
_VARIANT_LAYER_CAP_PER_TYPE = 50
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(str(text or ""))]


def _tf(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, av in a.items():
        dot += av * b.get(key, 0.0)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _to_open_unit_interval(value: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(value)))


def _angle_diff_wrapped(a: float, b: float) -> float:
    """Shortest signed angle from ``b`` to ``a``, in radians, in ``[-pi, pi]``."""
    d = float(a) - float(b)
    return math.atan2(math.sin(d), math.cos(d))


def _wrap_angle_rad(theta: float) -> float:
    """Coerce one radian-valued angle into ``[-pi, pi]`` (handles stray 2π jumps)."""
    t = float(theta)
    return math.atan2(math.sin(t), math.cos(t))


def _dense_cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for av, bv in zip(a, b):
        dot += float(av) * float(bv)
        na += float(av) * float(av)
        nb += float(bv) * float(bv)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _dense_cosine_weighted_ego_target(
    a: List[float],
    b: List[float],
    *,
    ego_weight: float,
    target_weight: float,
    ego_dims: int = 32,
    target_dims: int = 8,
) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    expected = int(ego_dims) + int(target_dims)
    if len(a) != expected:
        return _dense_cosine(a, b)

    ew = float(ego_weight)
    tw = float(target_weight)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i, (av, bv) in enumerate(zip(a, b)):
        w = ew if i < ego_dims else tw
        avw = float(av) * w
        bvw = float(bv) * w
        dot += avw * bvw
        na += avw * avw
        nb += bvw * bvw
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _cosine_to_unit(value: float) -> float:
    # Signed vectors (e.g. event_feature_vector / control_vector) may legitimately
    # point in opposite directions, so cosine can fall in [-1, 1]. Linearly remap
    # to [0, 1] for blending.
    return _to_open_unit_interval((float(value) + 1.0) / 2.0)


def _sentence_embedding_cosine_to_unit(value: float) -> float:
    """Map sentence-embedding cosine to [0, 1] without the ``(x+1)/2`` squeeze.

    Sentence encoders like ``all-MiniLM-L6-v2`` yield essentially non-negative
    cosine values on real text (typically ``[0.3, 0.9]``). Reusing
    ``_cosine_to_unit`` compresses that useful range into ``[0.65, 0.95]`` and
    wipes out discriminative power — the root cause of ``embedding_similarity``
    clustering around 0.8 regardless of the record. We clamp the (rare)
    negative cosines to 0 and return the raw cosine otherwise, leaving any
    further contrast stretching to the caller (see batch normalisation in
    ``JsonVectorStore.query``).
    """
    return _to_open_unit_interval(max(0.0, float(value)))


def _numeric_similarity(
    query_meta: Dict,
    record_meta: Dict,
    velocity_scale: float = 35.0,
    acceleration_scale: float = 15.0,
    yaw_rate_scale: float = 5.0,
    displacement_x_scale: float = 25.0,
    displacement_y_scale: float = 5.0,
) -> float:
    """Compute numeric similarity for Layer 3 sample matching.

    Compares: ego_v_mean, target_v_mean, ego_current_state.{heading,speed},
              target_current_state.{heading,speed}.

    Returns similarity in [0, 1] based on L2 distance, normalized by scales.
    Heading differences use the shortest arc (wrapped to ``[-pi, pi]``), not
    linear subtraction.

    ``heading_scale`` together with ``ego_heading_sq_weight`` /
    ``target_heading_sq_weight`` jointly set how much each heading mismatch
    lowers the score versus speed / acc / yaw_rate. ``target_heading_sq_weight``
    is deliberately larger than ego's so that a wrong-oriented target vehicle
    is penalised more strongly than a small ego-yaw drift (the matched sample
    should share the target's approach angle). Headings must be radians; if
    metadata stores degrees, convert before indexing.
    """
    # Extract nested state fields
    query_ego_state = query_meta.get("ego_current_state", {})
    record_ego_state = record_meta.get("ego_current_state", {})
    query_target_state = query_meta.get("target_current_state", {})
    record_target_state = record_meta.get("target_current_state", {})

    # Fourth element encodes heading-ness:
    #   None     -> scalar (linear difference)
    #   "target" -> circular difference weighted by target_heading_sq_weight
    #
    # NOTE: target heading is intentionally excluded here and is handled by a
    # dedicated feature match (`_target_heading_similarity`) so it can be
    # weighted/inspected independently from other numeric fields.
    state_fields: List[Tuple[float, float, float, Optional[str]]] = [
        (query_ego_state.get("speed", 0.0), record_ego_state.get("speed", 0.0), velocity_scale, None),
        (query_ego_state.get("acc", 0.0), record_ego_state.get("acc", 0.0), acceleration_scale, None),
        (query_ego_state.get("yaw_rate", 0.0), record_ego_state.get("yaw_rate", 0.0), yaw_rate_scale, None),
        (query_target_state.get("speed", 0.0), record_target_state.get("speed", 0.0), velocity_scale, None),
        (query_target_state.get("acc", 0.0), record_target_state.get("acc", 0.0), acceleration_scale, None),
        (query_target_state.get("yaw_rate", 0.0), record_target_state.get("yaw_rate", 0.0), yaw_rate_scale, None),
        (query_target_state.get("delta_x", 0.0), record_target_state.get("delta_x", 0.0), displacement_x_scale, None),
        (query_target_state.get("delta_y", 0.0), record_target_state.get("delta_y", 0.0), displacement_y_scale, None),
        (query_target_state.get("relative_speed", 0.0), record_target_state.get("relative_speed", 0.0), velocity_scale, None),
        (query_target_state.get("relative_acc", 0.0), record_target_state.get("relative_acc", 0.0), acceleration_scale, None),
    ]

    squared_diffs = []

    # State fields
    for q_val, r_val, scale, heading_kind in state_fields:
        if heading_kind is not None:
            diff = _angle_diff_wrapped(q_val, r_val) / scale
        else:
            diff = (float(q_val) - float(r_val)) / scale
        sq = diff * diff
        squared_diffs.append(sq)

    if not squared_diffs:
        return 0.5  # neutral

    # L2 distance → similarity via exp(-d^2)
    # Use mean squared normalized error so the score is less sensitive
    # to the number of active numeric fields.
    l2_dist_sq = sum(squared_diffs) / max(1.0, float(len(squared_diffs)))
    similarity = math.exp(-l2_dist_sq)
    return _to_open_unit_interval(similarity)


def _target_heading_similarity(
    query_meta: Dict,
    record_meta: Dict,
    heading_scale: float = math.pi / 4,
) -> Tuple[float, bool]:
    """Independent feature match for target heading (Layer 3 sample matching).

    Returns (similarity, has_heading). Similarity is in [0, 1] and uses the
    shortest wrapped angular difference.
    """
    query_target_state = query_meta.get("target_current_state", {}) or {}
    record_target_state = record_meta.get("target_current_state", {}) or {}

    qh = _safe_finite_float(query_target_state.get("heading", None))
    rh = _safe_finite_float(record_target_state.get("heading", None))
    if qh is None or rh is None:
        return 0.0, False

    if heading_scale <= 0.0:
        return 0.5, True

    diff = _angle_diff_wrapped(qh, rh) / float(heading_scale)
    l2_dist_sq = (diff * diff)
    sim = math.exp(-l2_dist_sq)
    return _to_open_unit_interval(sim), True


def _normalize_control_value(value: float, low: float, high: float) -> float:
    span = high - low
    if span <= 0:
        return 0.5
    return max(0.0, min(1.0, (float(value) - low) / span))


def _normalize_signed_control_value(value: float, scale: float) -> float:
    """Zero-centered normalization of a control / diff scalar to ``[-1, 1]``.

    Unlike ``_normalize_control_value`` this preserves sign, so that:
      * a true ``0`` input maps to ``0`` (not ``0.5``);
      * padding with ``0`` is semantically "no signal" rather than "mid-range";
      * cosine similarity over the resulting vector has real directional meaning
        (vectors are no longer squeezed into the non-negative orthant).
    """
    if scale <= 0.0:
        return 0.0
    v = float(value) / float(scale)
    return max(-1.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Event-feature vector (event-level, fixed length)
# ---------------------------------------------------------------------------
#
# ``event_feature`` is a per-event impact-centered summary produced by
# ``crash_data_processor.build_event_feature``. It is dict-structured and
# fixed in schema. Encoding it into a compact, normalized vector lets us
# plug it into cosine-based similarity alongside text / embedding signals,
# mainly at the ``variant`` retrieval layer.
#
# Each field is declared with a kind controlling how values are mapped:
#   - "signed":   x / scale  clipped to [-1, 1] (e.g. acceleration).
#   - "unsigned": x / scale  clipped to [ 0, 1] (e.g. speed, yaw rate).
#   - "circular": encoded as (cos(x), sin(x)); no scale, handles wrap-around.
#
# Missing values (``None`` / NaN / inf) fall back to neutral encodings
# (0 for signed & circular components, 0.5 for unsigned) so that they do
# not dominate cosine similarity by accident.

_EventFeatureSpec = Tuple[str, str, float]

_DEFAULT_EVENT_FEATURE_SPECS: List[_EventFeatureSpec] = [
    # Group A: means over event window
    ("ego_mean_v", "unsigned", 50.0),
    ("ego_mean_acc", "signed", 8.0),
    ("ego_mean_delta_heading", "unsigned", 0.3),
    ("ego_mean_yaw_rate", "unsigned", 2.0),
    ("sur_mean_v", "unsigned", 50.0),
    ("sur_mean_acc", "signed", 8.0),
    ("sur_mean_delta_heading", "unsigned", 0.3),
    ("sur_mean_yaw_rate", "unsigned", 2.0),
    # Group B: speed start/end/min/max/delta
    ("ego_v_start", "unsigned", 50.0),
    ("ego_v_end", "unsigned", 50.0),
    ("ego_v_min", "unsigned", 50.0),
    ("ego_v_max", "unsigned", 50.0),
    ("ego_delta_v", "signed", 20.0),
    ("sur_v_start", "unsigned", 50.0),
    ("sur_v_end", "unsigned", 50.0),
    ("sur_v_min", "unsigned", 50.0),
    ("sur_v_max", "unsigned", 50.0),
    ("sur_delta_v", "signed", 20.0),
    # Group C: acceleration extremes and ratios
    ("ego_acc_min", "signed", 8.0),
    ("ego_acc_max", "signed", 8.0),
    ("ego_decel_ratio", "unsigned", 1.0),
    ("ego_accel_ratio", "unsigned", 1.0),
    ("ego_hard_decel_ratio", "unsigned", 1.0),
    ("sur_acc_min", "signed", 8.0),
    ("sur_acc_max", "signed", 8.0),
    ("sur_decel_ratio", "unsigned", 1.0),
    ("sur_accel_ratio", "unsigned", 1.0),
    ("sur_hard_decel_ratio", "unsigned", 1.0),
    # Group D: heading change, yaw, lateral displacement
    ("ego_total_heading_change", "circular", 0.0),
    ("sur_total_heading_change", "circular", 0.0),
    ("ego_abs_total_heading_change", "unsigned", 15.0),
    ("sur_abs_total_heading_change", "unsigned", 15.0),
    ("ego_max_abs_yaw_rate", "unsigned", 5.0),
    ("sur_max_abs_yaw_rate", "unsigned", 5.0),
    ("ego_yaw_rate_sign_changes", "unsigned", 30.0),
    ("sur_yaw_rate_sign_changes", "unsigned", 30.0),
    ("ego_lateral_disp", "signed", 30.0),
    ("sur_lateral_disp", "signed", 30.0),
    ("ego_lateral_disp_abs", "unsigned", 30.0),
    ("sur_lateral_disp_abs", "unsigned", 30.0),
    # Group E: relative geometry over event window
    ("rel_x_start", "signed", 50.0),
    ("rel_y_start", "signed", 50.0),
    ("rel_x_end", "signed", 50.0),
    ("rel_y_end", "signed", 50.0),
    ("rel_x_mean", "signed", 50.0),
    ("rel_y_mean", "signed", 50.0),
    ("rel_dist_start", "unsigned", 80.0),
    ("rel_dist_end", "unsigned", 80.0),
    ("rel_dist_min", "unsigned", 80.0),
    ("rel_dist_slope", "signed", 25.0),
    ("rel_bearing_start", "circular", 0.0),
    ("rel_bearing_end", "circular", 0.0),
    ("rel_bearing_mean", "circular", 0.0),
    # Group F: relative dynamics over event window
    ("relative_heading_start", "circular", 0.0),
    ("relative_heading_end", "circular", 0.0),
    ("relative_heading_mean", "circular", 0.0),
    ("relative_heading_change", "circular", 0.0),
    ("relative_speed_start", "signed", 30.0),
    ("relative_speed_end", "signed", 30.0),
    ("relative_speed_mean", "signed", 30.0),
    ("closing_speed_start", "signed", 30.0),
    ("closing_speed_end", "signed", 30.0),
    ("closing_speed_mean", "signed", 30.0),
    ("closing_speed_max", "signed", 30.0),
    ("closing_acc_mean", "signed", 20.0),
    # Group G: TTC / risk evolution
    ("ttc_min", "unsigned", 30.0),
    ("ttc_end", "unsigned", 30.0),
    ("ttc_slope", "signed", 5.0),
    ("inv_ttc_max", "unsigned", 8.0),
    ("time_to_min_dist", "unsigned", 15.0),
    # Group H: vehicle dimensions (meters; from variant metadata)
    ("ego_width", "unsigned", 6.0),
    ("ego_length", "unsigned", 12.0),
    ("target_width", "unsigned", 6.0),
    ("target_length", "unsigned", 20.0),
]


def _safe_finite_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _encode_event_feature_value(value, kind: str, scale: float) -> List[float]:
    """Encode a single ``event_feature`` field into vector components."""
    v = _safe_finite_float(value)

    if kind == "circular":
        if v is None:
            return [0.0, 0.0]
        heading_weight = 5
        return [heading_weight * math.cos(v), heading_weight * math.sin(v)]

    if v is None or scale <= 0.0:
        return [0.0] if kind == "signed" else [0.5]

    normed = v / scale
    if kind == "signed":
        return [max(-1.0, min(1.0, normed))]
    # "unsigned"
    return [max(0.0, min(1.0, normed))]


def build_event_feature_vector(
    event_feature: Optional[Dict],
    field_specs: Optional[List[_EventFeatureSpec]] = None,
) -> List[float]:
    """Flatten an ``event_feature`` dict into a fixed-length normalized vector.

    The output length is deterministic: one component per non-circular field
    and two per circular field.  This vector is intended for cosine-based
    similarity against the query event summary (see ``JsonVectorStore.query``
    and ``query_event_feature_vector``).
    """
    specs = field_specs if field_specs is not None else _DEFAULT_EVENT_FEATURE_SPECS
    ef = event_feature or {}
    vec: List[float] = []
    for name, kind, scale in specs:
        vec.extend(_encode_event_feature_value(ef.get(name), kind, scale))
    return vec


def build_control_vector(
    ego_controls: List[Dict],
    target_controls: List[Dict],
    use_control: bool = False,
    n_steps_ego: Optional[int] = None,
    n_steps_target: Optional[int] = None,
    acc_scale: float = 8.0,
    steer_scale: float = 35.0,
    delta_s_scale: float = 15.0,
    delta_heading_scale: float = 0.8,
) -> List[float]:
    """Flatten ego + target controls into a single zero-centered vector, allowing
    asymmetric step counts for the two agents.

    Layout:
        [ego_val_0a, ego_val_0b, ..., ego_val_{Ne-1}a, ego_val_{Ne-1}b,
         target_val_0a, target_val_0b, ..., target_val_{Nt-1}a, target_val_{Nt-1}b]

    where (a, b) is (acceleration, steering_angle) when ``use_control`` is True
    and (delta_s, delta_heading) otherwise.

    Encoding (v2, zero-centered):
        Each scalar is mapped by ``value / scale`` and clipped to ``[-1, 1]``.
        A true zero input maps to ``0``, and missing / padded steps are also
        filled with ``0``. This keeps the ``0`` vector as "no signal" instead
        of the previous ``0.5``-midpoint convention that artificially aligned
        every vector into the non-negative orthant and inflated cosine scores
        close to 1. Scales default to values tuned for the real distributions
        (heading changes per 0.5 s step are typically ``|Δh| ≤ 0.5 rad``,
        per-step longitudinal displacements ``Δs ≤ 10 m``, etc.).

    ``n_steps_ego`` / ``n_steps_target`` default to the full length of the
    corresponding control list, so the returned vector can model e.g. 16 ego
    steps together with 4 target steps. For consistency across the corpus and
    the query, callers SHOULD pass explicit values here.
    """
    if n_steps_ego is None:
        n_steps_ego = len(ego_controls)
    if n_steps_target is None:
        n_steps_target = len(target_controls)

    vec: List[float] = []
    for controls, n_steps in (
        (ego_controls, n_steps_ego),
        (target_controls, n_steps_target),
    ):
        for i in range(n_steps):
            if i < len(controls):
                step = controls[i]
                if use_control:
                    a_val = _normalize_signed_control_value(
                        float(step.get("acceleration", 0.0)), acc_scale
                    )
                    b_val = _normalize_signed_control_value(
                        float(step.get("steering_angle", 0.0)), steer_scale
                    )
                else:
                    a_val = _normalize_signed_control_value(
                        float(step.get("delta_s", 0.0)), delta_s_scale
                    )
                    b_val = _normalize_signed_control_value(
                        float(step.get("delta_heading", 0.0)), delta_heading_scale
                    )
            else:
                # Padding: zero = "no signal" under signed normalization.
                a_val = 0.0
                b_val = 0.0
            vec.extend([a_val, b_val])
    return vec


# ---------------------------------------------------------------------------
# Trajectory vector (temporal-aware encoding)
# ---------------------------------------------------------------------------

def _traj_point(traj: List[Dict], idx: int) -> Tuple[float, float]:
    """Return (x, y) for *idx*, or (0, 0) if out of range."""
    if 0 <= idx < len(traj):
        return float(traj[idx].get("x", 0.0)), float(traj[idx].get("y", 0.0))
    return 0.0, 0.0


def build_trajectory_vector(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    n_steps = None,
    dt: float = 0.5,
    disp_scale: float = 25.0,
    vel_scale: float = 50.0,
    curv_scale: float = 0.5,
) -> List[float]:
    """Encode first *n_steps* of ego + target trajectories into a vector
    that captures temporal dynamics — not just raw positions.

    For each agent and each consecutive step pair the vector encodes:
      - dx, dy:        displacement delta (captures direction of motion)
      - speed:         instantaneous speed = sqrt(dx² + dy²) / dt
      - heading:       atan2(dy, dx) encoded as (cos, sin) to avoid wrapping
      - curvature:     heading change rate  (Δheading / ds)

    All values are scaled to roughly [-1, 1] via known physical ranges so
    that cosine similarity is meaningful.

    Per agent per step: 6 features  →  total = 2 agents × n_steps × 6.

    Parameters
    ----------
    ego_traj, target_traj : list of {"t", "x", "y"} dicts
    n_steps : number of consecutive displacement steps to encode
        (needs n_steps + 1 trajectory points)
    dt : time delta between consecutive points (seconds)
    disp_scale : max expected displacement per step (metres)
    vel_scale : max expected speed (m/s)
    curv_scale : max expected curvature (rad/m)
    """
    if n_steps is None:
        n_steps = max(len(ego_traj), len(target_traj)) - 1
    vec: List[float] = []
    for traj in (ego_traj, target_traj):
        prev_heading = 0.0
        for i in range(n_steps):
            x0, y0 = _traj_point(traj, i)
            x1, y1 = _traj_point(traj, i + 1)

            dx = x1 - x0
            dy = y1 - y0
            ds = math.sqrt(dx * dx + dy * dy)
            speed = ds / dt if dt > 0 else 0.0
            heading = math.atan2(dy, dx) if ds > _EPS else prev_heading

            # Curvature ≈ heading_change / arc_length
            d_heading = heading - prev_heading
            # Normalise angle difference to [-π, π]
            d_heading = (d_heading + math.pi) % (2 * math.pi) - math.pi
            curvature = d_heading / ds if ds > _EPS else 0.0
            prev_heading = heading

            vec.extend([
                dx / disp_scale,
                dy / disp_scale,
                speed / vel_scale,
                math.cos(heading),
                math.sin(heading),
                max(-1.0, min(1.0, curvature / curv_scale)),
            ])

    return vec


# ---------------------------------------------------------------------------
# DTW-based trajectory similarity
# ---------------------------------------------------------------------------

_TRAJ_FEATURES_PER_AGENT = 6  # dx, dy, speed, cos_h, sin_h, curvature


def _dtw_similarity(
    vec_a: List[float],
    vec_b: List[float],
    features_per_agent: int = _TRAJ_FEATURES_PER_AGENT,
    n_agents: int = 2,
    band: Optional[int] = None,
    sigma: float = 2.0,
) -> float:
    """DTW-based similarity for trajectory sequence vectors.

    The flat vector layout from ``build_trajectory_vector`` is::

        [ego_step0(6), ego_step1(6), ..., target_step0(6), ..., target_stepN(6)]

    This function reshapes each vector into ``(n_steps, n_agents * features)``
    by interleaving agents at each time step, runs DTW with per-step Euclidean
    cost and a Sakoe-Chiba band, then converts to a similarity in [0, 1].

    Parameters
    ----------
    vec_a, vec_b : flat trajectory vectors (may differ in length / n_steps).
    features_per_agent : features per agent per step (default 6).
    n_agents : number of agents encoded (default 2 — ego + target).
    band : Sakoe-Chiba band width in steps.  ``None`` → auto (max(2, min_len//3)).
    sigma : Gaussian decay width for dist → similarity conversion.
    """
    features_per_step = features_per_agent * n_agents  # 12

    if not vec_a or not vec_b:
        return 0.0

    na = len(vec_a) // (features_per_agent * n_agents)
    nb = len(vec_b) // (features_per_agent * n_agents)
    if na <= 0 or nb <= 0:
        return 0.0

    # --- Reshape: (n_steps, features_per_step) ---
    # Original layout per vector: [agent0_all_steps | agent1_all_steps]
    # Need: [[agent0_step_i | agent1_step_i] for i in range(n_steps)]
    def _reshape(vec: List[float], n_steps: int) -> List[List[float]]:
        block_size = n_steps * features_per_agent
        rows: List[List[float]] = []
        for i in range(n_steps):
            row: List[float] = []
            for a in range(n_agents):
                offset = a * block_size + i * features_per_agent
                for k in range(features_per_agent):
                    idx = offset + k
                    row.append(vec[idx] if idx < len(vec) else 0.0)
            rows.append(row)
        return rows

    steps_a = _reshape(vec_a, na)
    steps_b = _reshape(vec_b, nb)

    # --- Feature-wise normalisation (jointly over both sequences) ---
    # This avoids one feature scale dominating DTW distance.
    all_steps = steps_a + steps_b
    if not all_steps:
        return 0.0
    dim = len(all_steps[0])

    means = [0.0] * dim
    for row in all_steps:
        for k, v in enumerate(row):
            means[k] += float(v)
    inv_n = 1.0 / max(len(all_steps), 1)
    for k in range(dim):
        means[k] *= inv_n

    stds = [0.0] * dim
    for row in all_steps:
        for k, v in enumerate(row):
            diff = float(v) - means[k]
            stds[k] += diff * diff
    for k in range(dim):
        stds[k] = math.sqrt(stds[k] * inv_n)
        if stds[k] < _EPS:
            stds[k] = 1.0

    def _normalise_rows(rows: List[List[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for row in rows:
            out.append([(float(v) - means[k]) / stds[k] for k, v in enumerate(row)])
        return out

    steps_a = _normalise_rows(steps_a)
    steps_b = _normalise_rows(steps_b)

    # --- Step-wise Euclidean distance ---
    def _step_dist(sa: List[float], sb: List[float]) -> float:
        d = 0.0
        for va, vb in zip(sa, sb):
            diff = va - vb
            d += diff * diff
        return math.sqrt(d)

    # --- DTW with Sakoe-Chiba band ---
    if band is None:
        band = max(2, min(na, nb) // 3)
    INF = float("inf")
    # Use 1-indexed cost matrix; cost[0][0] = 0, rest of row/col 0 = INF
    cost = [[INF] * (nb + 1) for _ in range(na + 1)]
    cost[0][0] = 0.0

    for i in range(1, na + 1):
        j_lo = max(1, i - band)
        j_hi = min(nb, i + band)
        for j in range(j_lo, j_hi + 1):
            d = _step_dist(steps_a[i - 1], steps_b[j - 1])
            cost[i][j] = d + min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])

    dtw_dist = cost[na][nb]
    if dtw_dist >= INF:
        return 0.0

    # --- Normalise to per-step average, then Gaussian decay → [0, 1] ---
    n_mean = (na + nb) / 2.0
    avg_dist = dtw_dist / max(n_mean, 1.0)
    similarity = math.exp(-(avg_dist * avg_dist) / (sigma * sigma))
    return _to_open_unit_interval(similarity)


_METADATA_TAG_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "intersection": ("intersection", "junction", "crossing"),
    "non_junction": ("non-junction", "non junction", "midblock", "mid-block"),
    "urban": ("urban", "city", "downtown"),
    "rural": ("rural", "country road"),
    "night": ("night", "nighttime", "dark"),
    "daylight": ("daylight", "daytime", "day light"),
    "high_speed": ("high-speed", "high speed", "55mph", "65mph", "highway", "freeway"),
    "low_speed": ("low-speed", "low speed", "25mph", "35mph", "<=35"),
    "adverse": ("adverse", "rain", "snow", "fog", "wet", "slippery"),
    "clear": ("clear", "dry road"),
}

_INTERACTION_TAG_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "left_turn": ("left turn across", "unprotected left turn", "left turn", "turn left"),
    "right_turn": ("right turn", "turn right", "right-turn"),
    "same_direction": ("same-direction", "same direction", "turn-in across", "turn in across"),
    "lane_change_cut_in": ("lane change", "cut-in", "cut in", "merge into"),
    "rear_end": ("rear-end", "rear end", "following vehicle", "lead vehicle"),
    "crossing_path": (
        "crossing path",
        "intersecting path",
        "cross traffic",
        "cross into the ego",
        "crossing into the ego",
        "cross the intersection",
        "cross the junction",
        "crossing front",
        "fail to yield",
    ),
    "opposite_direction": ("oncoming", "opposite direction", "head-on", "head on", "opposing", "opposite"),
    "drifting": ("drift", "encroach"),
    "evasive": ("evasive action", "sudden maneuver", "sharp steering", "evasive"),
    "reversing": ("backing up", "reverse", "reversing"),
    "acceleration": ("accelerate", "acceleration", "speed up"),
    "deceleration": (
        "decelerate",
        "deceleration",
        "slow down",
        "slowing down",
        "brake",
        "braking",
        "lead vehicle decelerating",
        "stop suddenly",
    ),
    "stop": (
        "lead vehicle stopped",
        "come to a stop",
        "full stop",
        "stopped vehicle",
        "stopped car",
        "stationary",
        "vehicle stopped",
    ),
    "parking": ("parking", "park-in", "parked"),
    "pedestrian": ("pedestrian", "cyclist", "non-motorized", "non motorized"),
}

_BEHAVIOR_PRECRASH_INFERENCE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "2",
        (
            "lead vehicle decelerating",
            "lead vehicle stopped or decelerating",
            "lead vehicle decelerating or stopping",
            "decelerating",
            "deceleration",
            "slowing down",
            "slow down",
            "slow sharply",
            "brake",
            "braking",
            "hard braking",
            "hard brake",
            "brake hard",
            "sudden braking",
            "brake suddenly",
            "slam on the brakes",
            "abrupt deceleration",
            "stop suddenly",
        ),
    ),
    (
        "1",
        (
            "lead vehicle stopped",
            "stopped ahead",
            "stopped in lane",
            "stopped in the lane",
            "stopped in ego lane",
            "stopped in the ego lane",
            "come to a stop",
            "comes to a stop",
            "coming to a stop",
            "full stop",
            "at a full stop",
            "at a stop",
            "vehicle stopped",
            "target stopped",
            "stationary",
            "at a standstill",
            "standstill",
            "not moving",
        ),
    ),
    ("15", ("lead vehicle accelerating", "accelerating", "speed up", "speeding up")),
    ("7", ("lower constant speed", "moving slowly", "crawling")),
    ("4", ("changing lanes", "lane change", "cut-in", "cut in", "merge into")),
    (
        "3",
        (
            "left turn across path",
            "turn across path from opposite",
            "left turn across",
            "turn left across",
            "turn across the ego",
            "turn across the ego's path",
            "turn across path",
            "unprotected left",
            "left turn at the junction",
            "left turn at the intersection",
        ),
    ),
    (
        "5",
        (
            "straight crossing paths",
            "straight crossing",
            "crossing paths at junction",
            "cross into the ego",
            "cross into the ego's path",
            "crossing into the ego",
            "fail to yield at the intersection",
            "fail to yield at intersection",
        ),
    ),
    (
        "8",
        (
            "opposite direction",
            "opposite-direction",
            "oncoming",
            "oncoming vehicle",
            "oncoming traffic",
            "head-on",
            "head on",
            "head-on collision",
            "head on collision",
            "wrong-way",
            "wrong way",
            "driving the wrong way",
            "coming toward the ego",
            "coming towards the ego",
            "approaching head-on",
            "approaching from the opposite direction",
            "opposing lane",
            "opposing front",
        ),
    ),
    (
        "6",
        (
            "turning - same direction",
            "turn in same direction",
            "same-direction turn",
            "same direction turning",
            "turn right at junction",
            "turn right at the junction",
            "turn right at intersection",
            "turn right at the intersection",
            "right turn at junction",
            "right turn at intersection",
            "turning right",
            "right-turning",
            "right turn",
            "turn right",
            "right turn ahead",
        ),
    ),
    ("10", ("drifting", "drift")),
    ("11", ("following vehicle making a maneuver",)),
    ("12", ("evasive action", "evasive")),
    ("13", ("parking", "parked")),
    ("14", ("turning right at junction",)),
)

# When both match, keep the more specific junction code.
_PRECRASH_INFERENCE_SUPPRESS: Tuple[Tuple[str, str], ...] = (
    ("3", "6"),
    ("5", "8"),
    ("5", "6"),
)


def _resolve_precrash_inference_conflicts(codes: Set[str]) -> Set[str]:
    out = set(codes)
    for keep, drop in _PRECRASH_INFERENCE_SUPPRESS:
        if keep in out:
            out.discard(drop)
    return out


def infer_precrash_codes_from_interaction_tags(
    interaction_tags: Optional[Iterable[str]] = None,
    metadata_tags: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Infer junction precrash codes from structured interaction / scene tags."""
    itags = {str(t).strip().lower() for t in (interaction_tags or []) if str(t).strip()}
    mtags = {str(t).strip().lower() for t in (metadata_tags or []) if str(t).strip()}
    if not itags and not mtags:
        return set()
    codes: Set[str] = set()
    at_junction = "intersection" in mtags or "intersection" in itags
    if "left_turn" in itags and "opposite_direction" in itags:
        codes.add("3")
    if at_junction and "crossing_path" in itags:
        codes.add("5")
    return codes


def infer_precrash_codes_from_behavior(text: str) -> Set[str]:
    """Map free-text adversarial behavior to NHTSA ``precrash_code`` hints."""
    s = str(text or "").strip().lower()
    if not s:
        return set()
    codes: Set[str] = set()
    for code, name in NHTSA_CODE_NAMES.items():
        if name.lower() in s:
            codes.add(str(code))
    # if codes:
    #     return _resolve_precrash_inference_conflicts(codes)
    for code, patterns in _BEHAVIOR_PRECRASH_INFERENCE:
        if any(p in s for p in patterns):
            codes.add(code)
    return _resolve_precrash_inference_conflicts(codes)


def infer_precrash_code_hints(
    behavior_text: str = "",
    interaction_tags: Optional[Iterable[str]] = None,
    metadata_tags: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Merge behavior-text and tag-based NHTSA precrash code hints."""
    codes = infer_precrash_codes_from_behavior(behavior_text)
    codes |= infer_precrash_codes_from_interaction_tags(interaction_tags, metadata_tags)
    return _resolve_precrash_inference_conflicts(codes)


def _behavior_type_match_score(query_codes: Set[str], rec_code: str) -> float:
    """1.0 if record type is among inferred query types; 0.5 if no query hint."""
    rec = str(rec_code or "").strip()
    if not query_codes:
        return 0.5
    if rec and rec in query_codes:
        return 1.0
    return 0.0


_TAG_CONFLICT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("urban", "rural"),
    ("intersection", "non_junction"),
    ("signalized", "non_signalized"),
    ("high_speed", "low_speed"),
    ("left_turn", "right_turn"),
    ("acceleration", "deceleration"),
    ("acceleration", "stop"),
    ("same_direction", "opposite_direction"),
    ("daylight", "night"),
    ("clear", "adverse"),
)


def _extract_tags(text: str, tag_patterns: Dict[str, Tuple[str, ...]]) -> Set[str]:
    s = str(text or "").strip().lower()
    if not s:
        return set()
    tags: Set[str] = set()
    for tag, patterns in tag_patterns.items():
        if any(pattern in s for pattern in patterns):
            tags.add(tag)
    return tags


def _tag_score(query_tags: Set[str], record_tags: Set[str]) -> float:
    if not query_tags:
        return 0.0
    if not record_tags:
        return _to_open_unit_interval(0.1)

    overlap = len(query_tags & record_tags)
    coverage = overlap / max(1, len(query_tags))
    precision = overlap / max(1, len(record_tags))

    conflicts = 0
    for a, b in _TAG_CONFLICT_PAIRS:
        if (a in query_tags and b in record_tags) or (b in query_tags and a in record_tags):
            conflicts += 1
    conflict_penalty = conflicts / max(1, len(query_tags))

    raw_score = (0.55 * coverage) + (0.35 * precision) - (0.35 * conflict_penalty) + 0.1
    return _to_open_unit_interval(raw_score)


def _reconstruct_adv_future_from_controls(
    adv_current_state: Dict,
    anchor_control: List[Dict],
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Reconstruct adversary future (x, y) and per-step heading from current state +
    per-step ``delta_s`` / ``delta_heading`` controls.

    Mirrors the mid-heading arc-length integrator in
    ``crash_visualizer.reconstruct_trajectory_from_deltas`` so the resulting
    points are directly comparable with the predicted ego future trajectory
    (both in the ego-local frame).

    Returns parallel lists of positions and yaw (rad) at each integrated step.
    """
    if not adv_current_state or not anchor_control:
        return [], []

    x = float(adv_current_state.get("x", 0.0))
    y = float(adv_current_state.get("y", 0.0))
    yaw = float(adv_current_state.get("heading", 0.0))

    pts: List[Tuple[float, float]] = []
    yaws: List[float] = []
    for step in anchor_control:
        ds = float(step.get("delta_s", 0.0) or 0.0)
        d_yaw = float(step.get("delta_heading", 0.0) or 0.0)
        yaw_mid = yaw + d_yaw * 0.5
        x += ds * math.cos(yaw_mid)
        y += ds * math.sin(yaw_mid)
        yaw += d_yaw
        pts.append((x, y))
        yaws.append(float(yaw))
    return pts, yaws


def _headings_for_polyline(
    anchor_xy: Tuple[float, float],
    points: List[Tuple[float, float]],
) -> List[float]:
    """Instantaneous heading at each point from successive chord directions."""
    if not points:
        return []
    px, py = float(anchor_xy[0]), float(anchor_xy[1])
    out: List[float] = []
    for x, y in points:
        xf, yf = float(x), float(y)
        dx, dy = xf - px, yf - py
        if math.hypot(dx, dy) > 1e-9:
            h = math.atan2(dy, dx)
        elif out:
            h = out[-1]
        else:
            h = 0.0
        out.append(h)
        px, py = xf, yf
    return out


def _headings_polyline_unanchored(points: List[Tuple[float, float]]) -> List[float]:
    """Heading at each vertex: first segment uses p[0]→p[1], else p[i-1]→p[i]."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    out: List[float] = []
    for i in range(n):
        if i == 0:
            dx = float(points[1][0]) - float(points[0][0])
            dy = float(points[1][1]) - float(points[0][1])
        else:
            dx = float(points[i][0]) - float(points[i - 1][0])
            dy = float(points[i][1]) - float(points[i - 1][1])
        if math.hypot(dx, dy) > 1e-9:
            out.append(math.atan2(dy, dx))
        else:
            out.append(out[-1] if out else 0.0)
    return out


def _cross_z(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _oriented_box_corners(
    cx: float,
    cy: float,
    heading_rad: float,
    length_m: float,
    width_m: float,
) -> List[Tuple[float, float]]:
    """Return 4 corners for an oriented rectangle centered at ``(cx, cy)``."""
    l = max(0.0, float(length_m))
    w = max(0.0, float(width_m))
    c = math.cos(float(heading_rad))
    s = math.sin(float(heading_rad))
    local = [
        (-l / 2.0, -w / 2.0),
        (l / 2.0, -w / 2.0),
        (l / 2.0, w / 2.0),
        (-l / 2.0, w / 2.0),
    ]
    corners: List[Tuple[float, float]] = []
    for lx, ly in local:
        rx = lx * c + ly * (-s)
        ry = lx * s + ly * c
        corners.append((cx + rx, cy + ry))
    return corners


def _oriented_front_half_corners(
    cx: float,
    cy: float,
    heading_rad: float,
    length_m: float,
    width_m: float,
) -> List[Tuple[float, float]]:
    """Return ego front-half rectangle corners in world frame."""
    l = max(0.0, float(length_m))
    w = max(0.0, float(width_m))
    c = math.cos(float(heading_rad))
    s = math.sin(float(heading_rad))
    local = [
        (0.0, -w / 2.0),
        (l / 2.0, -w / 2.0),
        (l / 2.0, w / 2.0),
        (0.0, w / 2.0),
    ]
    corners: List[Tuple[float, float]] = []
    for lx, ly in local:
        rx = lx * c + ly * (-s)
        ry = lx * s + ly * c
        corners.append((cx + rx, cy + ry))
    return corners


def _segment_segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
    eps: float = 1e-12,
) -> Optional[Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < eps:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-18:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _segment_segment_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> float:
    if _segment_segment_intersection(p1, p2, p3, p4) is not None:
        return 0.0
    return min(
        _point_to_segment_distance(p1[0], p1[1], p3[0], p3[1], p4[0], p4[1]),
        _point_to_segment_distance(p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]),
        _point_to_segment_distance(p3[0], p3[1], p1[0], p1[1], p2[0], p2[1]),
        _point_to_segment_distance(p4[0], p4[1], p1[0], p1[1], p2[0], p2[1]),
    )


def _point_in_convex_polygon_ccw(
    px: float, py: float, poly: List[Tuple[float, float]], eps: float = 1e-9
) -> bool:
    m = len(poly)
    if m < 3:
        return False
    for i in range(m):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % m]
        if _cross_z(bx - ax, by - ay, px - ax, py - ay) < -eps:
            return False
    return True


def _vehicle_boxes_min_distance_pure(
    corners_a: List[Tuple[float, float]],
    corners_b: List[Tuple[float, float]],
) -> float:
    """Minimum Euclidean distance between boundaries of two convex rectangles."""
    # Fast overlap test for convex polygons.
    for p in corners_a:
        if _point_in_convex_polygon_ccw(p[0], p[1], corners_b):
            return 0.0
    for p in corners_b:
        if _point_in_convex_polygon_ccw(p[0], p[1], corners_a):
            return 0.0
    na = len(corners_a)
    nb = len(corners_b)
    for i in range(na):
        a1 = corners_a[i]
        a2 = corners_a[(i + 1) % na]
        for j in range(nb):
            b1 = corners_b[j]
            b2 = corners_b[(j + 1) % nb]
            if _segment_segment_intersection(a1, a2, b1, b2) is not None:
                return 0.0
    min_d = float("inf")
    for i in range(na):
        a1 = corners_a[i]
        a2 = corners_a[(i + 1) % na]
        for j in range(nb):
            b1 = corners_b[j]
            b2 = corners_b[(j + 1) % nb]
            min_d = min(min_d, _segment_segment_distance(a1, a2, b1, b2))
    return float(min_d)


def _vehicle_boxes_min_distance_m(
    corners_a: List[Tuple[float, float]],
    corners_b: List[Tuple[float, float]],
) -> float:
    try:
        from shapely.geometry import Polygon as ShapelyPolygon  # type: ignore[import-untyped]

        pa = ShapelyPolygon(corners_a)
        pb = ShapelyPolygon(corners_b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        return float(pa.distance(pb))
    except Exception:
        return _vehicle_boxes_min_distance_pure(corners_a, corners_b)


def _vehicle_boxes_contact_points_world(
    ego_corners: List[Tuple[float, float]],
    adv_corners: List[Tuple[float, float]],
    *,
    ego_center_xy: Tuple[float, float],
    adv_center_xy: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return contact proxy points on ego/adv boundaries in world frame.

    Reference behavior follows evaluation.ipynb:
    - If overlap area > 0: use overlap centroid, then nearest boundary points
      from each box to the centroid.
    - Else: use boundary-to-boundary nearest points.
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon  # type: ignore[import-untyped]
        from shapely.ops import nearest_points  # type: ignore[import-untyped]

        ego_poly = ShapelyPolygon(ego_corners)
        adv_poly = ShapelyPolygon(adv_corners)
        if not ego_poly.is_valid:
            ego_poly = ego_poly.buffer(0)
        if not adv_poly.is_valid:
            adv_poly = adv_poly.buffer(0)

        inter = ego_poly.intersection(adv_poly)
        if (not inter.is_empty) and float(inter.area) > 1e-12:
            c = inter.centroid
            p_ego, _ = nearest_points(ego_poly.boundary, c)
            p_adv, _ = nearest_points(adv_poly.boundary, c)
            return (float(p_ego.x), float(p_ego.y)), (float(p_adv.x), float(p_adv.y))

        p_ego, p_adv = nearest_points(ego_poly.boundary, adv_poly.boundary)
        return (float(p_ego.x), float(p_ego.y)), (float(p_adv.x), float(p_adv.y))
    except Exception:
        # Lightweight fallback without shapely: nearest vertices to opposite center.
        ecx, ecy = float(ego_center_xy[0]), float(ego_center_xy[1])
        acx, acy = float(adv_center_xy[0]), float(adv_center_xy[1])

        best_ego = min(
            ego_corners,
            key=lambda p: math.hypot(float(p[0]) - acx, float(p[1]) - acy),
        )
        best_adv = min(
            adv_corners,
            key=lambda p: math.hypot(float(p[0]) - ecx, float(p[1]) - ecy),
        )
        return (float(best_ego[0]), float(best_ego[1])), (float(best_adv[0]), float(best_adv[1]))


def _ego_contact_frontness_score(
    ego_contact_xy: Tuple[float, float],
    *,
    ego_center_xy: Tuple[float, float],
    ego_heading_rad: float,
    ego_length_m: float,
    min_distance_m: float,
) -> float:
    """Map ego contact point frontness into [0,1], front > rear.

    Front half of the vehicle maps to 1.0; rear half uses linear mapping from
    rear bumper (0) to center (1.0).
    """
    if min_distance_m >= 0.5:
        return 0.0
    l = max(float(ego_length_m), 1e-6)
    cx, cy = float(ego_center_xy[0]), float(ego_center_xy[1])
    px, py = float(ego_contact_xy[0]), float(ego_contact_xy[1])
    dx, dy = px - cx, py - cy
    c = math.cos(float(ego_heading_rad))
    s = math.sin(float(ego_heading_rad))
    local_x = dx * c + dy * s  # ego-forward axis in local frame
    if local_x >= 0.0:
        return _to_open_unit_interval(1.0)
    linear = (2.0 * local_x + l) / l
    return _to_open_unit_interval(max(0.0, min(1.0, linear)))


def _stage_trajectory_dicts_from_future_pts(
    ego_future: List[Tuple[float, float]],
    adv_future: List[Tuple[float, float]],
    *,
    dt: float = DT_CTRL,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Build aligned ego/adv trajectory dicts from the same 12-step future pair
    used by ``_min_distance_between_trajectories`` (``predicted_ego_future`` +
    adversary points from ``_reconstruct_adv_future_from_controls``).
    """
    n = min(len(ego_future), len(adv_future))
    if n < 3:
        return [], []
    ego_traj = [
        {
            "t": round((i + 1) * dt, 1),
            "x": float(ego_future[i][0]),
            "y": float(ego_future[i][1]),
        }
        for i in range(n)
    ]
    tgt_traj = [
        {
            "t": round((i + 1) * dt, 1),
            "x": float(adv_future[i][0]),
            "y": float(adv_future[i][1]),
        }
        for i in range(n)
    ]
    return ego_traj, tgt_traj


def _compute_stage_time_indices(
    ego_traj: List[Dict],
    target_traj: List[Dict],
    *,
    ego_length: float,
    ego_width: float,
    target_length: float,
    target_width: float,
    dt_default: float = DT_CTRL,
    impact_time_index: Optional[int] = None,
) -> Optional[Dict[str, Optional[int]]]:
    """Classify interaction stages and return first local index per stage."""
    if len(ego_traj) < 3 or len(target_traj) < 3:
        return None
    stage_out = classify_interaction_stages(
        ego_traj,
        target_traj,
        dt_default=dt_default,
        ego_length=ego_length,
        ego_width=ego_width,
        target_length=target_length,
        target_width=target_width,
    )
    if int(stage_out.get("n_steps", 0)) <= 0:
        return None
    return sanitize_stage_time_indices_vs_impact(
        _stage_time_indices(stage_out["stage_ids"]),
        impact_time_index,
    )


def _normalize_stage_time_indices(
    raw: object,
    *,
    impact_time_index: Optional[int] = None,
) -> Optional[Dict[str, Optional[int]]]:
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Optional[int]] = {}
    for name in STAGE_TIME_INDEX_NAMES:
        val = raw.get(name)
        if val is None:
            out[name] = None
            continue
        try:
            out[name] = int(val)
        except (TypeError, ValueError):
            out[name] = None
    return sanitize_stage_time_indices_vs_impact(out, impact_time_index)


# Relative importance when blending per-stage onset-alignment scores.
# Emergency response is closer to collision and more discriminative for retrieval.
_STAGE_TIME_SCORE_WEIGHTS: Dict[str, float] = {
    STAGE_RISK_AWARENESS: 0.20,
    STAGE_EMERGENCY_RESPONSE: 0.80,
}
# Full 12-step horizon gap when record has a stage onset but query does not.
_STAGE_TIME_MISSING_QUERY_PENALTY_DIFF = 12.0


def _stage_time_indices_abs_diffs(
    query_indices: Dict[str, Optional[int]],
    record_indices: Dict[str, Optional[int]],
) -> Dict[str, float]:
    """Absolute index gap per stage; missing query onset → strong penalty."""
    out: Dict[str, float] = {}
    for name in STAGE_TIME_INDEX_NAMES:
        q_idx = query_indices.get(name)
        r_idx = record_indices.get(name)
        if r_idx is None:
            continue
        if q_idx is None:
            out[name] = _STAGE_TIME_MISSING_QUERY_PENALTY_DIFF
            continue
        out[name] = float(abs(int(q_idx) - int(r_idx)))
    return out


def _impact_ts_abs_diff(
    query_ts: Optional[int],
    record_impact_index: Optional[int],
) -> Optional[float]:
    """Gap between query mdc timestep and record impact index; missing query → penalty."""
    if record_impact_index is None:
        return None
    if query_ts is None:
        return _STAGE_TIME_MISSING_QUERY_PENALTY_DIFF
    return float(abs(int(query_ts) - int(record_impact_index)))


def _stage_time_indices_mean_abs_diff(
    query_indices: Dict[str, Optional[int]],
    record_indices: Dict[str, Optional[int]],
) -> Optional[float]:
    """Mean absolute index gap across stages present in both query and record."""
    diffs = _stage_time_indices_abs_diffs(query_indices, record_indices)
    if not diffs:
        return None
    return float(sum(diffs.values())) / float(len(diffs))


def _single_stage_time_decay_score(diff: float, diff_max: float) -> float:
    """Map one stage's absolute index gap to (0, 1]; diff=0 → 1.0."""
    if diff <= 0.0:
        return 1.0
    denom = float(max(1.0, diff_max))
    x = float(diff) / denom
    decay_k = 10.0
    exp_term = float(math.exp(-decay_k * x))
    linear_term = max(0.0, 1.0 - x)
    base = 0.45 * exp_term + 0.55 * linear_term
    return _to_open_unit_interval(base)


def _stage_time_score_from_diff(
    stage_diffs: Dict[str, float],
    stage_diff_max: Dict[str, float],
    stage_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted blend of per-stage onset alignment scores.

    Each stage in ``stage_diffs`` is scored independently (same decay shape as
    ``impact_ts``), then combined using ``stage_weights``. Stages absent from
    either dict are skipped; remaining weights are re-normalised.
    """
    if not stage_diffs or not stage_diff_max:
        return 0.0
    weights = stage_weights or _STAGE_TIME_SCORE_WEIGHTS
    weighted_sum = 0.0
    weight_total = 0.0
    for name, diff in stage_diffs.items():
        w = float(weights.get(name, 0.0))
        if w <= 0.0:
            continue
        diff_max = stage_diff_max.get(name)
        if diff_max is None:
            continue
        score = _single_stage_time_decay_score(float(diff), float(diff_max))
        weighted_sum += w * score
        weight_total += w
    if weight_total <= 0.0:
        return 0.0
    return _to_open_unit_interval(weighted_sum / weight_total)


def _resolve_one_vehicle_dim_m(
    query_numeric_meta: Optional[Dict],
    *,
    top_keys: Tuple[str, ...],
    state_key: str,
    nested_keys: Tuple[str, ...],
    default: float,
) -> Tuple[float, str]:
    """Resolve one scalar dimension; source is ``top_level:*``, ``nested:*``, or ``default``."""
    if not isinstance(query_numeric_meta, dict):
        return default, "default"
    state = query_numeric_meta.get(state_key)
    if not isinstance(state, dict):
        state = {}
    for key in top_keys:
        raw = query_numeric_meta.get(key)
        if raw is None:
            continue
        try:
            return float(raw), f"top_level:{key}"
        except (TypeError, ValueError):
            continue
    for nk in nested_keys:
        raw = state.get(nk)
        if raw is None:
            continue
        try:
            return float(raw), f"nested:{state_key}.{nk}"
        except (TypeError, ValueError):
            continue
    return default, "default"


def _resolve_vehicle_lengths_m(
    query_numeric_meta: Optional[Dict],
) -> Tuple[float, float, str, str]:
    """Ego / adversary length in meters for bumper-offset distance; default sedan-scale."""
    ego_len, ego_src = _resolve_one_vehicle_dim_m(
        query_numeric_meta,
        top_keys=("ego_length_m", "ego_vehicle_length"),
        state_key="ego_current_state",
        nested_keys=("length", "vehicle_length"),
        default=4.5,
    )
    adv_len, adv_src = _resolve_one_vehicle_dim_m(
        query_numeric_meta,
        top_keys=("adv_length_m", "adversary_vehicle_length"),
        state_key="target_current_state",
        nested_keys=("length", "vehicle_length"),
        default=4.5,
    )
    return ego_len, adv_len, ego_src, adv_src


def _resolve_vehicle_widths_m(
    query_numeric_meta: Optional[Dict],
) -> Tuple[float, float, str, str]:
    """Ego / adversary width in meters; default sedan-scale."""
    ego_w, ego_src = _resolve_one_vehicle_dim_m(
        query_numeric_meta,
        top_keys=("ego_width_m", "ego_vehicle_width"),
        state_key="ego_current_state",
        nested_keys=("width", "vehicle_width"),
        default=1.8,
    )
    adv_w, adv_src = _resolve_one_vehicle_dim_m(
        query_numeric_meta,
        top_keys=("adv_width_m", "adversary_vehicle_width"),
        state_key="target_current_state",
        nested_keys=("width", "vehicle_width"),
        default=1.8,
    )
    return ego_w, adv_w, ego_src, adv_src


def _log_sample_vehicle_dims_resolution(
    query_numeric_meta: Optional[Dict],
    ego_len_m: float,
    adv_len_m: float,
    ego_len_src: str,
    adv_len_src: str,
    ego_wid_m: float,
    adv_wid_m: float,
    ego_wid_src: str,
    adv_wid_src: str,
) -> None:
    meta_kind = type(query_numeric_meta).__name__
    meta_keys = (
        sorted(query_numeric_meta.keys())
        if isinstance(query_numeric_meta, dict)
        else []
    )
    print(
        "RAG sample vehicle dims: "
        f"ego_len={ego_len_m:.3f}m({ego_len_src}), "
        f"adv_len={adv_len_m:.3f}m({adv_len_src}), "
        f"ego_wid={ego_wid_m:.3f}m({ego_wid_src}), "
        f"adv_wid={adv_wid_m:.3f}m({adv_wid_src}); "
        f"query_numeric_meta={meta_kind}"
        + (f" keys={meta_keys}" if meta_keys else ""),
        flush=True,
    )


def _min_distance_between_trajectories_euclidean(
    ego_future: List[Tuple[float, float]],
    adv_future: List[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[int]]:
    """Per-step minimum Euclidean distance between aligned trajectories.

    Both sequences are assumed to share the same dt step (e.g. 0.5s) starting
    one step after the current frame. Returns ``None`` if either sequence is
    empty.
    """
    n = min(len(ego_future), len(adv_future))
    if n <= 0:
        return None, None
    best: Optional[float] = None
    best_i: Optional[int] = None
    for i in range(n):
        ex, ey = float(ego_future[i][0]), float(ego_future[i][1])
        ax, ay = float(adv_future[i][0]), float(adv_future[i][1])
        d = math.hypot(ex - ax, ey - ay)
        if best is None or d < best:
            best = d
            best_i = i
    return best, best_i


def _min_distance_between_trajectories(
    ego_future: List[Tuple[float, float]],
    adv_future: List[Tuple[float, float]],
    *,
    ego_headings: Optional[List[float]] = None,
    adv_headings: Optional[List[float]] = None,
    ego_anchor_xy: Tuple[float, float] = (0.0, 0.0),
    ego_length_m: float = 4.5,
    ego_width_m: float = 2.0,
    adv_length_m: float = 4.5,
    adv_width_m: float = 2.0,
) -> Tuple[Optional[float], Optional[int], float]:
    """Per-step min distance between adv body and ego body.

    Each step constructs oriented rectangles using center pose + (length, width):
    - Ego uses full body.
    - Adv uses its full body.
    Distance is boundary-to-boundary box separation (0 when overlapping).
    Also computes a contact-proxy frontness score on ego:
    - front contact -> higher score
    - rear contact  -> lower score
    Returns ``(best_distance, best_step_idx_1based, frontness_score)``.
    Collision uses minimum box distance <= 0.1m; IoU is not used.
    """
    n = min(len(ego_future), len(adv_future))
    if n <= 0:
        return None, None, 0.0

    if ego_headings is None:
        ego_headings = _headings_for_polyline(ego_anchor_xy, ego_future[:n])
    if adv_headings is None:
        adv_headings = _headings_polyline_unanchored(adv_future[:n])

    if len(ego_headings) < n:
        ego_headings = _headings_for_polyline(ego_anchor_xy, ego_future[:n])
    if len(adv_headings) < n:
        adv_headings = _headings_polyline_unanchored(adv_future[:n])

    best: Optional[float] = None
    best_i: Optional[int] = None
    best_frontness = 0.0
    for i in range(n):
        ecx, ecy = float(ego_future[i][0]), float(ego_future[i][1])
        acx, acy = float(adv_future[i][0]), float(adv_future[i][1])
        he = float(ego_headings[i])
        ha = float(adv_headings[i])

        ego_body = _oriented_box_corners(ecx, ecy, he, ego_length_m, ego_width_m)
        adv_body = _oriented_box_corners(
            acx, acy, ha, adv_length_m, adv_width_m
        )
        d = _vehicle_boxes_min_distance_m(ego_body, adv_body)

        ego_contact_w, _ = _vehicle_boxes_contact_points_world(
            ego_body,
            adv_body,
            ego_center_xy=(ecx, ecy),
            adv_center_xy=(acx, acy),
        )
        frontness = _ego_contact_frontness_score(
            ego_contact_w,
            ego_center_xy=(ecx, ecy),
            ego_heading_rad=he,
            ego_length_m=ego_length_m,
            min_distance_m=d,
        )

        if best is None or d < best or (abs(d - float(best)) <= 1e-9 and frontness > best_frontness):
            best = d
            best_i = i
            best_frontness = float(frontness)
    return best, best_i, _to_open_unit_interval(best_frontness)


def _metadata_is_crash_unit(metadata: object) -> float:
    """Map ``metadata['is_crash']`` to 1.0 (crash) or 0.0 (non-crash / unknown)."""
    if not isinstance(metadata, dict):
        return 0.0
    v = metadata.get("is_crash")
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return 1.0 if float(v) != 0.0 else 0.0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return 1.0
        if s in ("false", "0", "no", ""):
            return 0.0
    return 0.0


def _weighted_blend(components: Dict[str, float], weights: Dict[str, float]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for key, value in components.items():
        weight = float(weights.get(key, 0.0))
        if weight <= 0.0:
            continue
        weighted_sum += weight * float(value)
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return _to_open_unit_interval(weighted_sum / total_weight)


class JsonVectorStore:
    """
    Lightweight on-disk vector store backed by sentence embeddings.
    """

    def __init__(self, db_path: str, embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL) -> None:
        self.db_path = Path(db_path)
        self.entries: List[Dict] = []
        self.embedding_model_name = str(embedding_model_name or _DEFAULT_EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(self.embedding_model_name)

    def _encode(self, text: str) -> List[float]:
        vector = self._embedder.encode(str(text or ""), normalize_embeddings=True)
        return [float(x) for x in vector]

    def build(self, records: List[Dict]) -> None:
        self.entries = []
        for r in records:
            text = str(r.get("document", ""))
            tokens = _tokenize(text)
            metadata = dict(r.get("metadata", {}))
            anchor_trajectory = metadata.get("anchor_trajectory", r.get("anchor_trajectory", []))
            anchor_control = metadata.get("anchor_control", r.get("anchor_control", []))
            metadata["anchor_trajectory"] = anchor_trajectory
            metadata["anchor_control"] = anchor_control
            metadata_blob = " ".join(
                [
                    str(metadata.get("environment", "")),
                    str(metadata.get("ego_maneuver", "")),
                    str(metadata.get("pre_crash_description", "")),
                    str(metadata.get("contributing_factors", "")),
                ]
            )
            metadata_tags = sorted(_extract_tags(metadata_blob, _METADATA_TAG_PATTERNS))
            interaction_tags = sorted(
                _extract_tags(
                    " ".join(
                        [
                            str(metadata.get("scenario_type", "")),
                            str(metadata.get("adversarial_maneuver", "")),
                        ]
                    ),
                    _INTERACTION_TAG_PATTERNS,
                )
            )
            event_feature = metadata.get("event_feature") or r.get("event_feature") or {}
            event_feature_vector = r.get("event_feature_vector") or []
            metadata["event_feature"] = event_feature

            self.entries.append(
                {
                    "id": r["id"],
                    "metadata": metadata,
                    "document": text,
                    "tf": dict(_tf(tokens)),
                    "embedding": self._encode(text),
                    "metadata_tags": metadata_tags,
                    "interaction_tags": interaction_tags,
                    "anchor_trajectory": anchor_trajectory,
                    "anchor_control": anchor_control,
                    "control_vector": r.get("control_vector", []),
                    "trajectory_vector": r.get("trajectory_vector", []),
                    "event_feature_vector": event_feature_vector,
                }
            )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db_path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.db_path.exists():
            self.entries = []
            return
        with self.db_path.open("r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def query(
        self,
        query_text: str,
        top_k: int = 1,
        env: str = "",
        scene_text: str = "",
        behavior_text: str = "",
        metadata_tags: Optional[Iterable[str]] = None,
        interaction_tags: Optional[Iterable[str]] = None,
        query_sequence_vector: Optional[List[float]] = None,
        filter_codes: Optional[Set[str]] = None,
        filter_variants: Optional[Set[Tuple[int, int]]] = None,  # (source_event_id, target_id)
        layer: str = "sample",  # "mode" | "variant" | "sample"
        query_numeric_meta: Optional[Dict] = None,
        query_event_feature_vector: Optional[List[float]] = None,
        type_prior_weights: Optional[Dict[str, float]] = None,
        type_score_weights: Optional[Dict[str, float]] = None,
        variant_score_weights: Optional[Dict[Tuple[int, int], float]] = None,
        use_trajectory: bool = False,
        predicted_ego_future: Optional[List[Tuple[float, float]]] = None,
        adv_current_state: Optional[Dict] = None,
    ) -> List[Dict]:
        query_text = str(query_text or "")
        scene_text = str(scene_text or "")
        behavior_text = str(behavior_text or "")
        env = str(env or "").strip().lower()

        qtf = _tf(_tokenize(query_text))
        query_embedding = self._encode(" ".join([query_text]))
        query_metadata_tags = set(metadata_tags or [])
        query_metadata_tags.update(_extract_tags(" ".join([env, scene_text]), _METADATA_TAG_PATTERNS))
        query_interaction_tags = set(interaction_tags or [])
        query_interaction_tags.update(_extract_tags(" ".join([behavior_text]), _INTERACTION_TAG_PATTERNS))
        query_precrash_code_hints = infer_precrash_code_hints(
            behavior_text=behavior_text,
            interaction_tags=query_interaction_tags,
            metadata_tags=query_metadata_tags,
        )

        ego_len_m, adv_len_m, ego_len_src, adv_len_src = _resolve_vehicle_lengths_m(
            query_numeric_meta
        )
        ego_wid_m, adv_wid_m, ego_wid_src, adv_wid_src = _resolve_vehicle_widths_m(
            query_numeric_meta
        )
        if layer == "sample":
            _log_sample_vehicle_dims_resolution(
                query_numeric_meta,
                ego_len_m,
                adv_len_m,
                ego_len_src,
                adv_len_src,
                ego_wid_m,
                adv_wid_m,
                ego_wid_src,
                adv_wid_src,
            )

        # --- Pass 1: compute per-entry signals, defer embedding stretch ---
        pending: List[Dict] = []
        raw_embedding_cos: List[float] = []
        for e in self.entries:
            metadata = dict(e.get("metadata", {}))

            # Pre-filter by precrash_code if requested
            if filter_codes is not None:
                rec_code = str(metadata.get("precrash_code", "")).strip()
                if rec_code not in filter_codes:
                    continue

            # Pre-filter by (source_event_id, target_id) if requested (Layer 3)
            if filter_variants is not None:
                rec_eid = metadata.get("source_event_id")
                rec_tid = metadata.get("target_id")
                if (rec_eid, rec_tid) not in filter_variants:
                    continue

            rec_env = str(metadata.get("environment", "")).lower()
            rec_doc = str(e.get("document", ""))
            etf = Counter(e.get("tf", {}))
            rec_embedding = e.get("embedding")
            if not isinstance(rec_embedding, list) or not rec_embedding:
                rec_embedding = self._encode(rec_doc)

            text_similarity = _to_open_unit_interval(_cosine(qtf, etf))
            raw_emb_cos = max(0.0, _dense_cosine(query_embedding, rec_embedding))

            record_metadata_tags = set(e.get("metadata_tags", []))
            if not record_metadata_tags:
                record_metadata_tags = _extract_tags(
                    " ".join(
                        [
                            rec_env,
                            str(metadata.get("ego_maneuver", "")),
                            str(metadata.get("pre_crash_description", "")),
                            str(metadata.get("contributing_factors", "")),
                        ]
                    ),
                    _METADATA_TAG_PATTERNS,
                )
            metadata_match = _tag_score(query_metadata_tags, record_metadata_tags)

            record_interaction_tags = set(e.get("interaction_tags", []))
            if not record_interaction_tags:
                record_interaction_tags = _extract_tags(
                    " ".join(
                        [
                            str(metadata.get("scenario_type", "")),
                            str(metadata.get("adversarial_maneuver", "")),
                        ]
                    ),
                    _INTERACTION_TAG_PATTERNS,
                )
            interaction_match = _tag_score(query_interaction_tags, record_interaction_tags)

            rec_precrash_code = str(metadata.get("precrash_code", "")).strip()
            behavior_type_match = _behavior_type_match_score(
                query_precrash_code_hints, rec_precrash_code
            )

            # Event-feature similarity (event-level impact-centered summary).
            # Useful mainly at the variant layer, and optionally as a weak
            # supplementary signal at the sample layer.
            event_feature_sim = 0.0
            has_event_feature = False
            if query_event_feature_vector:
                rec_efv = e.get("event_feature_vector") or []
                if rec_efv and len(rec_efv) == len(query_event_feature_vector):
                    event_feature_sim = _cosine_to_unit(
                        _dense_cosine(query_event_feature_vector, rec_efv)
                    )
                    has_event_feature = True
                else:
                    print(f"Event feature vector mismatch: {len(rec_efv)} != {len(query_event_feature_vector)}")

            # Kinematic similarity (sequence_vector)
            sequence_sim = 0.0
            has_sequence = False
            if query_sequence_vector:
                if use_trajectory:
                    rec_tv = e.get("trajectory_vector", [])
                    # DTW: handles variable length, temporal warping
                    sequence_sim = _dtw_similarity(query_sequence_vector, rec_tv)
                else:
                    ego_weight = 1.0
                    target_weight = 2.0
                    rec_cv = e.get("control_vector", [])
                    sequence_sim = _cosine_to_unit(
                        _dense_cosine_weighted_ego_target(
                            query_sequence_vector,
                            rec_cv,
                            ego_weight=ego_weight,
                            target_weight=target_weight,
                        )
                    )
                has_sequence = True

            # Numeric similarity (Layer 3 only — velocity means + current states)
            numeric_sim = 0.0
            has_numeric = False
            if layer == "sample" and query_numeric_meta:
                numeric_sim = _numeric_similarity(query_numeric_meta, metadata)
                has_numeric = True

            # Target heading similarity (independent feature match)
            target_heading_sim = 0.0
            has_target_heading = False
            if layer == "sample" and query_numeric_meta:
                target_heading_sim, has_target_heading = _target_heading_similarity(
                    query_numeric_meta, metadata
                )

            # Minimum distance between predicted ego future and the adversary
            # future trajectory reconstructed from the record's anchor_control
            # (delta_s + delta_heading integration starting at adv_current_state).
            # Smaller mdc → adversary gets closer to ego → more threatening →
            # higher score (assigned in Pass 2 via batch min-max normalisation).
            mdc: Optional[float] = None
            mdc_ts: Optional[int] = None
            mdc_contact_frontness: float = 0.0
            stage_time_diffs: Optional[Dict[str, float]] = None
            rec_anchor_control = e.get("anchor_control") or metadata.get("anchor_control") or []
            if (
                predicted_ego_future
                and adv_current_state
                and rec_anchor_control
            ):
                adv_future_pts, adv_yaws = _reconstruct_adv_future_from_controls(
                    adv_current_state, rec_anchor_control
                )
                mdc, mdc_ts, mdc_contact_frontness = _min_distance_between_trajectories(
                    predicted_ego_future,
                    adv_future_pts,
                    ego_headings=None,
                    adv_headings=adv_yaws,
                    ego_anchor_xy=(0.0, 0.0),
                    ego_length_m=ego_len_m,
                    ego_width_m=ego_wid_m,
                    adv_length_m=adv_len_m,
                    adv_width_m=adv_wid_m,
                )
                q_ego_traj, q_tgt_traj = _stage_trajectory_dicts_from_future_pts(
                    predicted_ego_future,
                    adv_future_pts,
                )
                query_sti = _compute_stage_time_indices(
                    q_ego_traj,
                    q_tgt_traj,
                    ego_length=ego_len_m,
                    ego_width=ego_wid_m,
                    target_length=adv_len_m,
                    target_width=adv_wid_m,
                    impact_time_index=mdc_ts,
                )
                rec_impact_idx: Optional[int] = None
                try:
                    raw_imp = metadata.get("impact_time_index")
                    if raw_imp is not None:
                        rec_impact_idx = int(raw_imp)
                except (TypeError, ValueError):
                    rec_impact_idx = None
                rec_sti = _normalize_stage_time_indices(
                    metadata.get("stage_time_indices"),
                    impact_time_index=rec_impact_idx,
                )
                if query_sti and rec_sti:
                    diffs = _stage_time_indices_abs_diffs(query_sti, rec_sti)
                    stage_time_diffs = diffs or None
            # Hard filter: discard candidates whose predicted-future min distance
            # is too large (less threatening / less relevant).
            # if mdc is not None and float(mdc) > 10.0:
            #     continue

            raw_embedding_cos.append(raw_emb_cos)
            pending.append(
                {
                    "entry": e,
                    "text_similarity": text_similarity,
                    "raw_emb_cos": raw_emb_cos,
                    "metadata_match": metadata_match,
                    "interaction_match": interaction_match,
                    "behavior_type_match_score": behavior_type_match,
                    "event_feature_sim": event_feature_sim,
                    "has_event_feature": has_event_feature,
                    "sequence_sim": sequence_sim,
                    "has_sequence": has_sequence,
                    "numeric_sim": numeric_sim,
                    "has_numeric": has_numeric,
                    "target_heading_sim": target_heading_sim,
                    "has_target_heading": has_target_heading,
                    "mdc": mdc,
                    "mdc_ts": mdc_ts,
                    "mdc_contact_frontness": mdc_contact_frontness,
                    "stage_time_diffs": stage_time_diffs,
                }
            )

        # --- Batch min-max for predicted-future mdc (variant / sample layers) ---
        # Smaller mdc → adversary is closer to ego → higher score. We normalise
        # across the candidate batch and invert so that the minimum distance
        # maps to 1.0 and the maximum maps to 0.0.
        _all_mdc = [row["mdc"] for row in pending if row["mdc"] is not None]
        if _all_mdc:
            _mdc_min = min(_all_mdc)
            _mdc_max = max(_all_mdc)
        else:
            _mdc_min = _mdc_max = None

        # --- Batch min-max for impact-time alignment (sample layer only) ---
        # We want the timestep `ts` at which the min-distance occurs to be close
        # to the record metadata's `impact_time_index` (smaller index distance
        # → higher score). We normalise abs(ts - impact_time_index) across the
        # candidate batch and invert it so min diff maps to 1.0.
        _all_impact_ts_diff: List[float] = []
        if layer == "sample":
            for row in pending:
                ts = row.get("mdc_ts", None)
                try:
                    impact_idx = int(
                        (row.get("entry", {}) or {})
                        .get("metadata", {})
                        .get("impact_time_index")
                    )
                except Exception:
                    impact_idx = None
                diff = _impact_ts_abs_diff(ts, impact_idx)
                if diff is not None:
                    _all_impact_ts_diff.append(diff)

        if _all_impact_ts_diff:
            _impact_diff_min = min(_all_impact_ts_diff)
            _impact_diff_max = max(_all_impact_ts_diff)
        else:
            _impact_diff_min = _impact_diff_max = None

        # --- Batch min-max for stage-time alignment (sample layer only) ---
        # Per-stage max diff so risk_awareness / emergency_response normalise independently.
        _stage_time_diff_max: Dict[str, float] = {}
        if layer == "sample":
            for name in STAGE_TIME_INDEX_NAMES:
                stage_diffs_batch: List[float] = []
                for row in pending:
                    diffs = row.get("stage_time_diffs")
                    if not isinstance(diffs, dict):
                        continue
                    val = diffs.get(name)
                    if val is not None:
                        stage_diffs_batch.append(float(val))
                if stage_diffs_batch:
                    _stage_time_diff_max[name] = max(stage_diffs_batch)

        # Sample layer: when trajectory context is available, every candidate
        # participates in stage_time weighting; missing/invalid diff → score 0.
        has_stage_time_context = bool(
            layer == "sample" and predicted_ego_future and adv_current_state
        )

        # --- Batch contrast-stretch embedding cosines across candidates ---
        # Sentence-embedding cosine distributions are narrow on thematically
        # similar corpora (e.g. all entries describe car crashes), so absolute
        # values cluster (~0.5-0.8) and lose discriminative power. We blend
        # the clamped raw cosine with its in-batch min-max rescaled version so
        # both absolute quality and relative ranking contribute.
        if raw_embedding_cos:
            lo = min(raw_embedding_cos)
            hi = max(raw_embedding_cos)
        else:
            lo, hi = 0.0, 0.0
        span = hi - lo

        # --- Pass 2: blend weights and produce final scores ---
        scored: List[Tuple[float, Dict]] = []
        for row in pending:
            e = row["entry"]
            raw_emb_cos = row["raw_emb_cos"]
            if span > _EPS:
                stretched = (raw_emb_cos - lo) / span
                embedding_similarity = _to_open_unit_interval(
                    0.5 * raw_emb_cos + 0.5 * stretched
                )
            else:
                embedding_similarity = _sentence_embedding_cosine_to_unit(raw_emb_cos)

            has_sequence = row["has_sequence"]
            has_numeric = row["has_numeric"]
            has_event_feature = row["has_event_feature"]
            has_target_heading = row.get("has_target_heading", False)

            # mdc score: only for variant / sample layers. Smaller min-distance
            # between predicted ego future and reconstructed adv future → more
            # threatening scenario → higher score.
            mdc = row["mdc"]
            has_mdc = (
                mdc is not None
                and _mdc_min is not None
                and layer == "sample"
            )
            if has_mdc:
                if mdc <= 0.1:
                    mdc_score = 1.0
                else:
                    _mdc_span = _mdc_max - _mdc_min
                    if _mdc_span > _EPS:
                        mdc_score = _to_open_unit_interval(
                            1.0 - (mdc - _mdc_min) / _mdc_span
                        )
                    else:
                        mdc_score = 0.5
            else:
                mdc_score = 0.0

            # Impact-time alignment score (sample layer only):
            # prefer min-distance timestep `ts` close to metadata impact_time_index.
            impact_ts_score = 0.0
            has_impact_ts = False
            if layer == "sample" and _impact_diff_min is not None:
                ts = row.get("mdc_ts", None)
                try:
                    impact_idx = int((e.get("metadata", {}) or {}).get("impact_time_index"))
                except Exception:
                    impact_idx = None
                diff = _impact_ts_abs_diff(ts, impact_idx)
                if diff is not None:
                    has_impact_ts = True
                    # metadata impact_time_index is 0-based on 12-step anchor;
                    if diff == 0.0:
                        impact_ts_score = 1.0
                    else:
                        # Gentler decay: blend mild exponential with linear so
                        # moderate |ts - impact_idx| gaps still get usable scores.
                        # Normalise by batch max to keep scale stable.
                        _impact_denom = float(max(1, int(_impact_diff_max or 1)))
                        _impact_x = float(diff) / _impact_denom  # in [0, 1] (typically)
                        _impact_decay_k = 10.0
                        _impact_exp = float(math.exp(-_impact_decay_k * _impact_x))
                        _impact_linear = max(0.0, 1.0 - _impact_x)
                        _impact_base = 0.45 * _impact_exp + 0.55 * _impact_linear
                        impact_ts_score = _to_open_unit_interval(_impact_base)

            # Stage-time alignment (sample layer): query stage onset indices are
            # derived from the same 12-step ``predicted_ego_future`` +
            # ``_reconstruct_adv_future_from_controls`` pair as mdc; compared
            # against the record metadata ``stage_time_indices``.
            stage_time_score = 0.0
            if has_stage_time_context:
                stage_diffs = row.get("stage_time_diffs")
                if isinstance(stage_diffs, dict) and stage_diffs and _stage_time_diff_max:
                    stage_time_score = _stage_time_score_from_diff(
                        stage_diffs, _stage_time_diff_max
                    )

            components = {
                "text_similarity": row["text_similarity"],
                "embedding_similarity": embedding_similarity,
                "metadata_match": row["metadata_match"],
                "interaction_match": row["interaction_match"],
                "behavior_type_match_score": float(row.get("behavior_type_match_score", 0.0) or 0.0),
                "sequence_similarity": row["sequence_sim"],
                "numeric_similarity": row["numeric_sim"],
                "target_heading_similarity": row.get("target_heading_sim", 0.0),
                "event_feature_similarity": row["event_feature_sim"],
                "mdc_score": mdc_score,
                "mdc_contact_frontness": float(row.get("mdc_contact_frontness", 0.0) or 0.0),
                "impact_ts_score": impact_ts_score,
                "stage_time_score": stage_time_score,
                "is_crash_score": (
                    _metadata_is_crash_unit(e.get("metadata", {}) or {})
                    if layer == "variant"
                    else 0.0
                ),
            }

            # def _too_low(v: object, thr: float = 0.1) -> bool:
            #     if v is None:
            #         return True
            #     try:
            #         fv = float(v)
            #     except Exception:
            #         return True
            #     if math.isnan(fv):
            #         return True
            #     return fv < thr

            # if layer == "mode":
            #     if _too_low(components["interaction_match"], 0.1):
            #         continue
            # elif layer == "variant":
            #     if _too_low(components["event_feature_similarity"], 0.1):
            #         continue
            # elif layer == "sample":
            #     print("numeric_similarity:", components["numeric_similarity"])
            #     if  _too_low(components["sequence_similarity"], 0.1) or _too_low(components["mdc_score"], 0.1):
            #         continue

            # Layer-aware weight profiles
            if layer == "mode":
                # Semantic-only: text + embedding; no sequence/numeric
                weights = {
                    "text_similarity": 0.10,
                    "embedding_similarity": 0.30,
                    "metadata_match": 0.30 if query_metadata_tags else 0.0,
                    "interaction_match": 0.30 if query_interaction_tags else 0.0,
                    "behavior_type_match_score": 0.20 if query_precrash_code_hints else 0.0,
                    "sequence_similarity": 0.0,
                    "numeric_similarity": 0.0,
                }
            elif layer == "variant":
                # Balanced: semantic + optional kinematic
                if has_sequence:
                    weights = {
                        "text_similarity": 0.05,
                        "embedding_similarity": 0.05,
                        "metadata_match": 0.10 if query_metadata_tags else 0.0,
                        "interaction_match": 0.20 if query_interaction_tags else 0.0,
                        "behavior_type_match_score": 0.10 if query_precrash_code_hints else 0.0,
                        "sequence_similarity": 0.50,
                        "numeric_similarity": 0.0,
                    }
                else:
                    weights = {
                        "text_similarity": 0.05,
                        "embedding_similarity": 0.05,
                        "metadata_match": 0.15 if query_metadata_tags else 0.0,
                        "interaction_match": 0.20 if query_interaction_tags else 0.0,
                        "behavior_type_match_score": 0.10 if query_precrash_code_hints else 0.0,
                        "sequence_similarity": 0.0,
                        "numeric_similarity": 0.0,
                    }
            else:
                # sample: sequence-vector + numeric-state dominant, with a
                if has_sequence and has_numeric:
                    weights = {
                        "text_similarity": 0.05,
                        "embedding_similarity": 0.05,
                        "metadata_match": 0.10 if query_metadata_tags else 0.0,
                        "interaction_match": 0.10,
                        "behavior_type_match_score": 0.10 if query_precrash_code_hints else 0.0,
                        "sequence_similarity": 0.40,
                        "numeric_similarity": 0.20,
                    }
                elif has_sequence:
                    weights = {
                        "text_similarity": 0.05,
                        "embedding_similarity": 0.05,
                        "metadata_match": 0.05 if query_metadata_tags else 0.0,
                        "interaction_match": 0.10,
                        "behavior_type_match_score": 0.10 if query_precrash_code_hints else 0.0,
                        "sequence_similarity": 0.45,
                        "numeric_similarity": 0.0,
                    }
                elif has_numeric:
                    weights = {
                        "text_similarity": 0.05,
                        "embedding_similarity": 0.05,
                        "metadata_match": 0.05 if query_metadata_tags else 0.0,
                        "interaction_match": 0.10,
                        "behavior_type_match_score": 0.10 if query_precrash_code_hints else 0.0,
                        "sequence_similarity": 0.0,
                        "numeric_similarity": 0.40,
                    }
                else:
                    weights = {
                        "text_similarity": 0.25,
                        "embedding_similarity": 0.30,
                        "metadata_match": 0.15 if query_metadata_tags else 0.0,
                        "interaction_match": 0.30 if query_interaction_tags else 0.0,
                        "behavior_type_match_score": 0.10 if query_precrash_code_hints else 0.0,
                        "sequence_similarity": 0.0,
                        "numeric_similarity": 0.0,
                    }

            # Independent target heading feature weight (sample layer only).
            # `_weighted_blend` renormalises by the sum of active weights.
            if layer == "sample" and has_target_heading:
                weights["target_heading_similarity"] = 0.10
            else:
                weights["target_heading_similarity"] = 0.0

            # Add event-feature weight. ``_weighted_blend`` normalises by the
            # total of active weights, so inserting a non-zero value here
            # redistributes existing components proportionally.
            if has_event_feature:
                if layer == "variant":
                    weights["event_feature_similarity"] = 0.60
                elif layer == "sample":
                    weights["event_feature_similarity"] = 0.30
                else:
                    weights["event_feature_similarity"] = 0.0
            else:
                weights["event_feature_similarity"] = 0.0

            # mdc score: smaller predicted-future adv-ego distance → more
            # threatening → higher score. Active only for variant / sample
            # layers when predicted_ego_future + adv_current_state + the
            # record's anchor_control are all available.
            if has_mdc:
                weights["mdc_score"] = 0.3
            else:
                weights["mdc_score"] = 0.0

            # Contact frontness (sample layer): same min-distance timestep,
            # but collisions/contacts on ego front are preferred over rear.
            if layer == "sample" and has_mdc:
                weights["mdc_contact_frontness"] = 0.15
            else:
                weights["mdc_contact_frontness"] = 0.0

            # impact-time alignment: active when record has impact_time_index;
            # missing query mdc_ts is penalised instead of skipped.
            if has_impact_ts:
                weights["impact_ts_score"] = 0.4
            else:
                weights["impact_ts_score"] = 0.0

            if has_stage_time_context:
                weights["stage_time_score"] = 0.6
            else:
                weights["stage_time_score"] = 0.0

            # Crash vs near-miss prior from record metadata (variant layer only).
            if layer == "variant":
                weights["is_crash_score"] = 0.5
            else:
                weights["is_crash_score"] = 0.0

            score = _weighted_blend(components, weights)

            # Cross-layer prior multiplier:
            # - mode layer: multiply by optional dataset ``type_prior_weights`` (precrash_code)
            # - variant layer: multiply by type rank ``type_score_weights`` (from layer-1)
            # - sample layer: multiply by variant weight (from layer-2 variant rank)
            layer_prior_weight = 1.0
            if layer == "mode" and type_prior_weights:
                precrash_code = str((e.get("metadata", {}) or {}).get("precrash_code", "")).strip()
                if precrash_code:
                    layer_prior_weight = float(type_prior_weights.get(precrash_code, 1.0))
            elif layer == "variant" and type_score_weights:
                precrash_code = str((e.get("metadata", {}) or {}).get("precrash_code", "")).strip()
                if precrash_code:
                    layer_prior_weight = float(type_score_weights.get(precrash_code, 1.0))
            elif layer == "sample" and variant_score_weights:
                metadata = e.get("metadata", {}) or {}
                rec_eid = metadata.get("source_event_id")
                rec_tid = metadata.get("target_id")
                if rec_eid is not None and rec_tid is not None:
                    layer_prior_weight = float(variant_score_weights.get((rec_eid, rec_tid), 1.0))
            score *= layer_prior_weight

            scored.append((score, {**e, "__score_components": components}))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Variant layer + matched types: scan every candidate per precrash_code, keep the
        # best-scoring row per variant key, then take top ``_VARIANT_LAYER_CAP_PER_TYPE``
        # variants per type; merge and re-rank globally for ``top_k``.
        if layer == "variant" and filter_codes:

            def _variant_entry_precrash(entry: Dict) -> str:
                return str((entry.get("metadata") or {}).get("precrash_code", "")).strip()

            def _variant_entry_key(entry: Dict) -> Tuple:
                md = entry.get("metadata", {}) or {}
                eid, tid = md.get("source_event_id"), md.get("target_id")
                if eid is not None and tid is not None:
                    return ("et", eid, tid)
                return ("id", entry.get("id"))

            pruned: List[Tuple[float, Dict]] = []
            for code in sorted(filter_codes):
                rows = [(s, e) for s, e in scored if _variant_entry_precrash(e) == code]
                best_by_key: Dict[Tuple, Tuple[float, Dict]] = {}
                for s, e in rows:
                    vk = _variant_entry_key(e)
                    prev = best_by_key.get(vk)
                    if prev is None or s > prev[0]:
                        best_by_key[vk] = (s, e)
                ranked = sorted(best_by_key.values(), key=lambda t: t[0], reverse=True)
                pruned.extend(ranked[:_VARIANT_LAYER_CAP_PER_TYPE])
            pruned.sort(key=lambda x: x[0], reverse=True)
            scored = pruned

        # Debug: variant / sample layers — print top scored breakdown (not written to ``out``).
        if layer == "variant":
            _n = min(50, len(scored))
            print("\n[VectorStore][DEBUG] layer=variant top scored components:")
            for i, (score, e) in enumerate(scored[:_n], start=1):
                components = e.get("__score_components", {}) or {}
                print(
                    (
                        "  "
                        + f"#{i} id={e.get('id')} score={float(score):.4f} "
                        + "text={text:.4f} emb={emb:.4f} meta={meta:.4f} inter={inter:.4f} "
                        + "event={event:.4f} is_crash={is_crash:.4f} behavior={behavior:.4f}"
                    ).format(
                        text=float(components.get("text_similarity", 0.0) or 0.0),
                        emb=float(components.get("embedding_similarity", 0.0) or 0.0),
                        meta=float(components.get("metadata_match", 0.0) or 0.0),
                        inter=float(components.get("interaction_match", 0.0) or 0.0),
                        event=float(components.get("event_feature_similarity", 0.0) or 0.0),
                        is_crash=float(components.get("is_crash_score", 0.0) or 0.0),
                        behavior=float(components.get("behavior_type_match_score", 0.0) or 0.0),
                    )
                )
        if layer == "sample":
            _n = min(50, len(scored))
            print("\n[VectorStore][DEBUG] layer=sample top scored components:")
            for i, (score, e) in enumerate(scored[:_n], start=1):
                components = e.get("__score_components", {}) or {}
                print(
                    (
                        "  "
                        + f"#{i} id={e.get('id')} score={float(score):.4f} "
                        + "text={text:.4f} emb={emb:.4f} meta={meta:.4f} inter={inter:.4f} "
                        + "seq={seq:.4f} num={num:.4f} th={th:.4f} event={event:.4f} "
                        + "mdc={mdc:.4f} front={front:.4f} impact_ts={impact_ts:.4f} "
                        + "stage_time={stage_time:.4f} behavior={behavior:.4f}"
                    ).format(
                        text=float(components.get("text_similarity", 0.0) or 0.0),
                        emb=float(components.get("embedding_similarity", 0.0) or 0.0),
                        meta=float(components.get("metadata_match", 0.0) or 0.0),
                        inter=float(components.get("interaction_match", 0.0) or 0.0),
                        seq=float(components.get("sequence_similarity", 0.0) or 0.0),
                        num=float(components.get("numeric_similarity", 0.0) or 0.0),
                        th=float(components.get("target_heading_similarity", 0.0) or 0.0),
                        event=float(components.get("event_feature_similarity", 0.0) or 0.0),
                        mdc=float(components.get("mdc_score", 0.0) or 0.0),
                        front=float(components.get("mdc_contact_frontness", 0.0) or 0.0),
                        impact_ts=float(components.get("impact_ts_score", 0.0) or 0.0),
                        stage_time=float(components.get("stage_time_score", 0.0) or 0.0),
                        behavior=float(components.get("behavior_type_match_score", 0.0) or 0.0),
                    )
                )
        out: List[Dict] = []
        for score, e in scored[:top_k]:
            metadata = dict(e.get("metadata", {}))
            if "anchor_trajectory" not in metadata:
                metadata["anchor_trajectory"] = e.get("anchor_trajectory", [])
            if "anchor_control" not in metadata:
                metadata["anchor_control"] = e.get("anchor_control", [])
            components = e.get("__score_components", {})
            item = {
                "id": e.get("id"),
                "document": e.get("document"),
                "similarity_score": round(float(score), 4),
                "text_similarity": round(float(components.get("text_similarity", 0.0)), 4),
                "embedding_similarity": round(float(components.get("embedding_similarity", 0.0)), 4),
                "metadata_match": round(float(components.get("metadata_match", 0.0)), 4),
                "interaction_match": round(float(components.get("interaction_match", 0.0)), 4),
                "sequence_similarity": round(float(components.get("sequence_similarity", 0.0)), 4),
                "numeric_similarity": round(float(components.get("numeric_similarity", 0.0)), 4),
                "event_feature_similarity": round(float(components.get("event_feature_similarity", 0.0)), 4),
                "mdc_score": round(float(components.get("mdc_score", 0.0)), 4),
                "mdc_contact_frontness": round(float(components.get("mdc_contact_frontness", 0.0)), 4),
                "is_crash_score": round(float(components.get("is_crash_score", 0.0)), 4),
                "behavior_type_match_score": round(float(components.get("behavior_type_match_score", 0.0)), 4),
                "stage_time_score": round(float(components.get("stage_time_score", 0.0)), 4),
                "anchor_trajectory": metadata.get("anchor_trajectory", []),
                "anchor_control": metadata.get("anchor_control", []),
            }
            item.update(metadata)
            out.append(item)
        return out
