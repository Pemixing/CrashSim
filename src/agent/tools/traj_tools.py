import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.patches import Polygon
    from matplotlib import transforms as mtransforms
except ImportError:  # pragma: no cover - debug-only dependency
    plt = None
    Rectangle = None
    Polygon = None
    mtransforms = None

BOX_COLLISION_DISTANCE_THRESH_M = 0.1


def extract_agent_points(scene_text: str) -> Dict[int, List[Tuple[float, float]]]:
    result: Dict[int, List[Tuple[float, float]]] = {}
    current_id: Optional[int] = None
    id_pat = re.compile(r"^\s*id:\s*(\d+),", re.IGNORECASE)
    agent_pat = re.compile(r"^\s*Agent\s+(\d+)\s*:", re.IGNORECASE)
    xy_pat = re.compile(r"x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m")
    hist_pat = re.compile(r"Historical trajectory.*?:\s*(\[[^\]]*\])", re.IGNORECASE)
    tuple_pat = re.compile(
        r"\(\s*([-+]?(?:\d+(?:\.\d+)?|nan))\s*,\s*([-+]?(?:\d+(?:\.\d+)?|nan))\s*\)",
        re.IGNORECASE,
    )

    for line in scene_text.splitlines():
        id_match = id_pat.search(line)
        agent_match = agent_pat.search(line)
        if id_match or agent_match:
            current_id = int((id_match or agent_match).group(1))
            result.setdefault(current_id, [])
            continue

        if current_id is None:
            continue

        # Prefer historical trajectory because it contains ordered multi-frame points.
        hist_match = hist_pat.search(line)
        if hist_match:
            hist_points: List[Tuple[float, float]] = []
            for x_text, y_text in tuple_pat.findall(hist_match.group(1)):
                if x_text.lower() == "nan" or y_text.lower() == "nan":
                    continue
                hist_points.append((float(x_text), float(y_text)))
            if hist_points:
                result[current_id] = hist_points
            continue

        # Fallback when historical trajectory line is missing.
        if "Current state:" in line and not result[current_id]:
            xy = xy_pat.search(line)
            if xy:
                result[current_id].append((float(xy.group(1)), float(xy.group(2))))
    return result


def extract_scene_headings(scene_text: str) -> Tuple[float, Dict[int, float]]:
    """Extract ego and surrounding-agent headings (rad) from scene text."""
    ego_heading = 0.0
    agent_headings: Dict[int, float] = {}
    current_id: Optional[int] = None

    id_pat = re.compile(r"^\s*id:\s*(\d+),", re.IGNORECASE)
    agent_pat = re.compile(r"^\s*Agent\s+(\d+)\s*:", re.IGNORECASE)
    heading_pat = re.compile(r"heading=([-+]?(?:\d+(?:\.\d+)?|nan))", re.IGNORECASE)

    ego_block_match = re.search(
        r"EGO:\s*(.*?)(?:\n\s*\n|\nSURROUNDING AGENTS|\Z)",
        scene_text,
        re.IGNORECASE | re.DOTALL,
    )
    if ego_block_match:
        ego_heading_match = heading_pat.search(ego_block_match.group(1))
        if ego_heading_match:
            heading_text = ego_heading_match.group(1)
            if heading_text.lower() != "nan":
                ego_heading = float(heading_text)

    for line in scene_text.splitlines():
        id_match = id_pat.search(line)
        agent_match = agent_pat.search(line)
        if id_match or agent_match:
            current_id = int((id_match or agent_match).group(1))
            continue

        if current_id is None or "Current state:" not in line:
            continue

        heading_match = heading_pat.search(line)
        if not heading_match:
            continue
        heading_text = heading_match.group(1)
        if heading_text.lower() == "nan":
            continue
        agent_headings[current_id] = float(heading_text)

    return ego_heading, agent_headings


def fallback_linear_traj(points: List[Tuple[float, float]]) -> List[Dict]:
    print("fallback_linear_traj points")
    if not points:
        points = [(0.0, 0.0)]
    cur_x, cur_y = points[-1]
    if len(points) >= 2:
        vx = (points[-1][0] - points[-2][0]) / 0.5
        vy = (points[-1][1] - points[-2][1]) / 0.5
    else:
        vx, vy = 0.0, 0.0

    traj = []
    for i in range(12):
        t = (i + 1) * 0.5
        traj.append(
            {
                "t": round(t, 1),
                "x": round(float(cur_x + vx * t), 3),
                "y": round(float(cur_y + vy * t), 3),
            }
        )
    return traj


def normalize_trajectory(raw_traj: object) -> Optional[List[Dict]]:
    if not isinstance(raw_traj, list):
        return None

    parsed: List[Tuple[float, float]] = []
    for item in raw_traj:
        if isinstance(item, dict) and "x" in item and "y" in item:
            try:
                parsed.append((float(item["x"]), float(item["y"])))
            except (TypeError, ValueError):
                continue
        elif isinstance(item, list) and len(item) >= 2:
            try:
                parsed.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue

    if not parsed:
        return None

    if len(parsed) >= 12:
        parsed = parsed[:12]
    else:
        while len(parsed) < 12:
            parsed.append(parsed[-1])

    out: List[Dict] = []
    for i, (x, y) in enumerate(parsed):
        out.append({"t": round((i + 1) * 0.5, 1), "x": round(x, 3), "y": round(y, 3)})
    return out


def emergency_brake_stop_trajectory(
    template: object,
    start_xy: Tuple[float, float],
    heading_rad: float,
    start_speed_mps: float,
    *,
    default_dt: float = 0.5,
    max_accel_mps2: float = 6.0,
    max_speed_mps: float = 55.0,
) -> List[Dict[str, float]]:
    """
    Constant deceleration (emergency brake) along heading until v=0.

    - Keeps the same point count and (when available) timestamps as `template`.
    - Output dict contains at least: {t, x, y, v}.
    """
    if not isinstance(template, list) or not template:
        return []

    dt = float(default_dt)
    a_brake = max(1e-3, float(max_accel_mps2))
    v = min(max(0.0, float(max_speed_mps)), max(0.0, float(start_speed_mps)))

    h = float(heading_rad)
    ux = math.cos(h)
    uy = math.sin(h)

    x, y = float(start_xy[0]), float(start_xy[1])
    out: List[Dict[str, float]] = []
    for idx in range(len(template)):
        # Use the same integration convention as opt_anchor_kin's emergency brake path.
        x = x + v * dt * ux
        y = y + v * dt * uy
        v = max(0.0, v - a_brake * dt)

        t_val = (idx + 1) * dt
        if isinstance(template[idx], dict) and "t" in template[idx]:
            try:
                t_val = float(template[idx]["t"])
            except (TypeError, ValueError):
                pass

        out.append(
            {
                "t": float(t_val),
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "v": round(float(v), 3),
            }
        )
    return out


# Post-collision hard-brake deceleration (m/s^2); above typical comfort braking (~3–4)
# and aligned with aggressive emergency / crash-aftermath stopping.
_POST_COLLISION_HARD_BRAKE_MPS2 = 10.0


def truncate_trajectory_after_collision_to_hard_stop(
    traj: object,
    closest_t_index: object,
) -> object:
    """Rewrite the trajectory **after** the collision instant as a hard emergency stop.

    Pre-collision points are left unchanged. From the first post-collision step the
    adversary continues along the impact heading at the collision speed, then
    decelerates at a constant aggressive rate (``_POST_COLLISION_HARD_BRAKE_MPS2``)
    until ``v=0`` and remains stationary thereafter. Steering / yaw rates are
    zeroed on the post-collision suffix.
    """
    if not isinstance(traj, list):
        return traj
    try:
        collision_idx = int(closest_t_index)
    except (TypeError, ValueError):
        return traj
    if not (0 <= collision_idx < (len(traj) - 1)):
        return traj
    if not isinstance(traj[collision_idx], dict):
        return traj

    collision_pt = traj[collision_idx]
    cx = collision_pt.get("x")
    cy = collision_pt.get("y")
    if cx is None or cy is None:
        return traj
    try:
        cx_f = float(cx)
        cy_f = float(cy)
    except (TypeError, ValueError):
        return traj

    suffix_template = traj[collision_idx + 1 :]
    if not suffix_template:
        return traj

    default_dt = 0.5
    impact_heading = 0.0
    v0 = 0.0
    if collision_idx >= 1 and isinstance(traj[collision_idx - 1], dict):
        px = traj[collision_idx - 1].get("x")
        py = traj[collision_idx - 1].get("y")
        try:
            px_f = float(px)
            py_f = float(py)
            impact_heading = math.atan2(cy_f - py_f, cx_f - px_f)
            dist = math.hypot(cx_f - px_f, cy_f - py_f)
            if (
                "t" in traj[collision_idx]
                and "t" in traj[collision_idx - 1]
            ):
                try:
                    dt_est = float(traj[collision_idx]["t"]) - float(
                        traj[collision_idx - 1]["t"]
                    )
                    if math.isfinite(dt_est) and dt_est > 1e-3:
                        default_dt = dt_est
                except (TypeError, ValueError):
                    pass
            v0 = dist / max(1e-3, default_dt)
        except (TypeError, ValueError):
            pass
    if v0 <= 0.0:
        for key in ("speed", "v"):
            if key in collision_pt:
                try:
                    v0 = max(v0, float(collision_pt[key]))
                except (TypeError, ValueError):
                    pass
    if impact_heading == 0.0:
        for key in ("heading", "yaw", "theta"):
            if key in collision_pt:
                try:
                    impact_heading = float(collision_pt[key])
                    break
                except (TypeError, ValueError):
                    pass

    a_brake = float(_POST_COLLISION_HARD_BRAKE_MPS2)
    brake_suffix = emergency_brake_stop_trajectory(
        suffix_template,
        (cx_f, cy_f),
        impact_heading,
        v0,
        default_dt=default_dt,
        max_accel_mps2=a_brake,
    )

    new_traj: List[Any] = list(traj[: collision_idx + 1])
    for orig_pt, brake_pt in zip(suffix_template, brake_suffix):
        if isinstance(orig_pt, dict):
            new_pt = dict(orig_pt)
            new_pt["t"] = brake_pt["t"]
            new_pt["x"] = brake_pt["x"]
            new_pt["y"] = brake_pt["y"]
            v_k = float(brake_pt["v"])
            if "speed" in new_pt:
                new_pt["speed"] = v_k
            if "v" in new_pt:
                new_pt["v"] = v_k
            if "vx" in new_pt or "vy" in new_pt:
                vx = v_k * math.cos(impact_heading)
                vy = v_k * math.sin(impact_heading)
                if "vx" in new_pt:
                    new_pt["vx"] = round(vx, 3)
                if "vy" in new_pt:
                    new_pt["vy"] = round(vy, 3)
            if "acceleration" in new_pt:
                new_pt["acceleration"] = -a_brake if v_k > 1e-6 else 0.0
            if "ax" in new_pt:
                new_pt["ax"] = round(-a_brake * math.cos(impact_heading), 3) if v_k > 1e-6 else 0.0
            if "ay" in new_pt:
                new_pt["ay"] = round(-a_brake * math.sin(impact_heading), 3) if v_k > 1e-6 else 0.0
            if "heading" in new_pt:
                new_pt["heading"] = round(float(impact_heading), 4)
            if "yaw" in new_pt:
                new_pt["yaw"] = round(float(impact_heading), 4)
            if "theta" in new_pt:
                new_pt["theta"] = round(float(impact_heading), 4)
            for k in ("steering_angle", "steer", "heading_rate", "yaw_rate"):
                if k in new_pt:
                    new_pt[k] = 0.0
            new_traj.append(new_pt)
        else:
            new_traj.append(orig_pt)
    return new_traj


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _estimate_initial_speed(
    history_points: Optional[List[Tuple[float, float]]],
    current_xy: Tuple[float, float],
    first_future_xy: Tuple[float, float],
    default_dt: float,
) -> float:
    if history_points and len(history_points) >= 2:
        dx = float(history_points[-1][0]) - float(history_points[-2][0])
        dy = float(history_points[-1][1]) - float(history_points[-2][1])
        speed = math.hypot(dx, dy) / max(1e-6, default_dt)
        if math.isfinite(speed):
            return speed

    speed = math.hypot(first_future_xy[0] - current_xy[0], first_future_xy[1] - current_xy[1])
    return speed / max(1e-6, default_dt) if math.isfinite(speed) else 0.0


def _cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    u: float,
) -> Tuple[float, float]:
    inv = 1.0 - u
    b0 = inv * inv * inv
    b1 = 3.0 * inv * inv * u
    b2 = 3.0 * inv * u * u
    b3 = u * u * u
    return (
        b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
    )


def _bezier_heading_bridge(
    points: List[Tuple[float, float]],
    current_xy: Tuple[float, float],
    current_heading: float,
    bridge_steps: int,
) -> List[Tuple[float, float]]:
    bridge_steps = max(1, min(bridge_steps, len(points)))
    target = points[bridge_steps - 1]
    distance = max(1e-6, math.hypot(target[0] - current_xy[0], target[1] - current_xy[1]))

    if bridge_steps < len(points):
        next_target = points[bridge_steps]
        target_heading = math.atan2(next_target[1] - target[1], next_target[0] - target[0])
    elif bridge_steps >= 2:
        prev_target = points[bridge_steps - 2]
        target_heading = math.atan2(target[1] - prev_target[1], target[0] - prev_target[0])
    else:
        target_heading = current_heading

    handle = min(distance * 0.45, 8.0)
    p1 = (
        current_xy[0] + handle * math.cos(current_heading),
        current_xy[1] + handle * math.sin(current_heading),
    )
    p2 = (
        target[0] - handle * math.cos(target_heading),
        target[1] - handle * math.sin(target_heading),
    )

    bridged = list(points)
    for idx in range(bridge_steps):
        u = float(idx + 1) / float(bridge_steps)
        bridged[idx] = _cubic_bezier(current_xy, p1, p2, target, u)
    return bridged


def optimize_anchor_trajectory_kinematics(
    anchor_trajectory_6s: object,
    current_state: Optional[Dict[str, float]],
    history_points: Optional[List[Tuple[float, float]]] = None,
    initial_speed_mps: Optional[float] = None,
    default_dt: float = 0.5,
    max_yaw_rate_rad_s: float = 0.6,
    max_accel_mps2: float = 6.0,
    max_speed_mps: float = 55.0,
    first_point_lateral_offset_threshold_m: float = 3.0,
    heading_mismatch_threshold_rad: float = math.radians(35.0),
    keep_original_if_feasible: bool = True,
    feasibility_slack: float = 1.10,
    max_deviation_m: float = 4.0,
    max_pull_per_step_m: float = 0.3,
    warm_start_steps: int = 4,
    transition_steps: int = 4,
    reverse_detection_horizon: int = 5,
) -> List[Dict]:
    """
    Smooth anchor trajectory with kinematic constraints.

    Difference from previous version:
    - No Bezier bridge.
    - No direct hard splice: new_pts + pts[steps:].
    - Uses heading-constrained warm start + transition rollout before merging
      back to the original anchor.

    If cumulative longitudinal displacement over the first few anchor segments
    (see reverse_detection_horizon) is backward relative to vehicle heading,
    replace with a straight emergency-brake deceleration along that net motion
    direction until stopped.
    """
    norm = normalize_trajectory(anchor_trajectory_6s)
    if norm is None:
        print("norm is None, returning fallback_linear_traj")
        return fallback_linear_traj(history_points or [])

    if not current_state:
        print("current_state is None, returning norm")
        return norm

    try:
        cur_x = float(current_state["x"])
        cur_y = float(current_state["y"])
        heading0 = float(current_state["heading"])
    except (KeyError, TypeError, ValueError):
        return norm

    if not all(math.isfinite(v) for v in (cur_x, cur_y, heading0)):
        return norm

    if not math.isfinite(float(default_dt)) or float(default_dt) <= 1e-6:
        default_dt = 0.5

    max_yaw_rate_rad_s = max(1e-3, float(max_yaw_rate_rad_s))
    max_accel_mps2 = max(1e-3, float(max_accel_mps2))
    max_speed_mps = max(1e-3, float(max_speed_mps))
    first_point_lateral_offset_threshold_m = max(0.0, float(first_point_lateral_offset_threshold_m))
    heading_mismatch_threshold_rad = max(0.0, float(heading_mismatch_threshold_rad))
    feasibility_slack = max(1.0, float(feasibility_slack))
    max_deviation_m = max(0.0, float(max_deviation_m))
    max_pull_per_step_m = max(0.0, float(max_pull_per_step_m))
    warm_start_steps = max(0, int(warm_start_steps))
    transition_steps = max(0, int(transition_steps))
    reverse_detection_horizon = max(1, int(reverse_detection_horizon))

    points = [(float(pt["x"]), float(pt["y"])) for pt in norm]
    if not points:
        return norm

    current_xy = (cur_x, cur_y)

    def _ego_local_offset(
        p: Tuple[float, float],
        origin: Tuple[float, float],
        h: float,
    ) -> Tuple[float, float]:
        dx = float(p[0]) - float(origin[0])
        dy = float(p[1]) - float(origin[1])
        longitudinal = math.cos(h) * dx + math.sin(h) * dy
        lateral = -math.sin(h) * dx + math.cos(h) * dy
        return longitudinal, lateral

    def _first_point_consistency(
        pts: List[Tuple[float, float]],
        start_xy: Tuple[float, float],
        start_heading: float,
    ) -> Tuple[float, float, float]:
        first = pts[0]
        dx = first[0] - start_xy[0]
        dy = first[1] - start_xy[1]
        first_dist = math.hypot(dx, dy)

        if first_dist > 1e-9:
            first_heading = math.atan2(dy, dx)
        else:
            first_heading = start_heading

        heading_error = abs(_wrap_angle(first_heading - start_heading))
        _, lateral_offset = _ego_local_offset(first, start_xy, start_heading)
        return first_dist, abs(lateral_offset), heading_error

    def _anchor_speeds(
        pts: List[Tuple[float, float]],
        start_xy: Tuple[float, float],
    ) -> List[float]:
        speeds: List[float] = []
        prev = start_xy

        for p in pts:
            dist = math.hypot(float(p[0]) - float(prev[0]), float(p[1]) - float(prev[1]))
            speeds.append(dist / default_dt)
            prev = p

        return speeds

    def _estimate_start_speed() -> float:
        if initial_speed_mps is not None:
            try:
                v = float(initial_speed_mps)
                if math.isfinite(v):
                    return min(max_speed_mps, max(0.0, v))
            except (TypeError, ValueError):
                pass

        try:
            v = _estimate_initial_speed(history_points, current_xy, points[0], default_dt)
            if math.isfinite(float(v)):
                return min(max_speed_mps, max(0.0, float(v)))
        except Exception:
            pass

        speeds = _anchor_speeds(points, current_xy)
        if speeds:
            return min(max_speed_mps, max(0.0, speeds[0]))

        return 0.0


    def _is_reverse_anchor_motion(angle_threshold_deg=135, lon_tol_ratio=0.05, lateral_tol_ratio=0.3):
        """
        Determine if a polyline segment is moving roughly opposite to vehicle heading.
        - Uses cumulative longitudinal displacement along vehicle heading.
        - Uses average motion direction to prevent misjudging curves as reverse.
        - Ignores small lateral offsets within lateral_tol_ratio of path length.
        """
        if not points:
            return False

        k = min(reverse_detection_horizon, len(points))
        h = float(heading0)
        ch, sh = math.cos(h), math.sin(h)

        net_lon = 0.0
        path_dx = 0.0
        path_dy = 0.0
        path_len = 0.0
        max_lat_offset = 0.0

        prev_x, prev_y = float(current_xy[0]), float(current_xy[1])

        for i in range(k):
            px, py = float(points[i][0]), float(points[i][1])
            dx = px - prev_x
            dy = py - prev_y
            segment_len = math.hypot(dx, dy)
            path_len += segment_len

            # Longitudinal and lateral projections along heading
            lon = ch * dx + sh * dy
            lat = -sh * dx + ch * dy  # perpendicular to heading

            net_lon += lon
            max_lat_offset = max(max_lat_offset, abs(lat))

            path_dx += dx
            path_dy += dy
            prev_x, prev_y = px, py

        if path_len < 1e-6:
            return False

        # Average motion direction
        avg_angle = math.atan2(path_dy, path_dx)
        heading_diff = abs((avg_angle - h + math.pi) % (2 * math.pi) - math.pi)

        # Reverse if:
        # 1. Avg motion opposite heading (> angle_threshold)
        # 2. Cumulative longitudinal displacement negative
        # But only if lateral offset not too large (i.e., ignore curves)
        lateral_ratio = max_lat_offset / path_len if path_len > 1e-6 else 0.0

        if lateral_ratio > lateral_tol_ratio:
            # Path mostly lateral → likely a curve, ignore net_lon
            if heading_diff > math.radians(angle_threshold_deg):
                return True
            return False
        else:
            # Mostly longitudinal
            if heading_diff > math.radians(angle_threshold_deg):
                return True
            if net_lon < -max(lon_tol_ratio, lon_tol_ratio * path_len):
                return True

        return False

    def _is_original_feasible() -> bool:
        speeds = _anchor_speeds(points, current_xy)

        prev_xy = current_xy
        prev_h = heading0
        prev_v = _estimate_start_speed()

        for p, v in zip(points, speeds):
            dx = p[0] - prev_xy[0]
            dy = p[1] - prev_xy[1]
            dist = math.hypot(dx, dy)

            if dist > 1e-9:
                h = math.atan2(dy, dx)
            else:
                h = prev_h

            yaw_rate = abs(_wrap_angle(h - prev_h)) / default_dt
            if yaw_rate > max_yaw_rate_rad_s * feasibility_slack:
                print(
                    "opt_anchor_kin: original infeasible (yaw_rate) "
                    f"yaw_rate={yaw_rate:.3f} > limit={(max_yaw_rate_rad_s * feasibility_slack):.3f} "
                    f"(max_yaw_rate={max_yaw_rate_rad_s:.3f}, slack={feasibility_slack:.3f}, dt={default_dt:.3f})"
                )
                return False

            prev_xy = p
            prev_h = h
            prev_v = v

        first_dist, first_lateral_offset, heading_error = _first_point_consistency(
            points, current_xy, heading0
        )

        if first_lateral_offset > first_point_lateral_offset_threshold_m * feasibility_slack:
            print(
                "opt_anchor_kin: original infeasible (first_lateral_offset) "
                f"lat={first_lateral_offset:.3f} > limit={(first_point_lateral_offset_threshold_m * feasibility_slack):.3f} "
                f"(thr={first_point_lateral_offset_threshold_m:.3f}, slack={feasibility_slack:.3f})"
            )
            return False

        if heading_error > heading_mismatch_threshold_rad * feasibility_slack:
            print(
                "opt_anchor_kin: original infeasible (heading_error) "
                f"heading_error_deg={math.degrees(heading_error):.2f} > limit_deg={math.degrees(heading_mismatch_threshold_rad * feasibility_slack):.2f} "
                f"(thr_deg={math.degrees(heading_mismatch_threshold_rad):.2f}, slack={feasibility_slack:.3f})"
            )
            return False

        return True

    def _emergency_brake_stop_trajectory(v0: float) -> List[Dict]:
        """
        Constant deceleration (emergency brake) along current heading until v=0.
        It keeps the same point count and timestamps as `norm`.
        """
        out = emergency_brake_stop_trajectory(
            norm,
            (float(cur_x), float(cur_y)),
            float(heading0),
            float(v0),
            default_dt=float(default_dt),
            max_accel_mps2=float(max_accel_mps2),
            max_speed_mps=float(max_speed_mps),
        )
        # Keep the original schema (t, x, y) for the optimizer.
        return [{"t": p["t"], "x": p["x"], "y": p["y"]} for p in out]

    def _rollout_one_step(
        x: float,
        y: float,
        h: float,
        v: float,
        target_xy: Tuple[float, float],
        desired_v: float,
    ) -> Tuple[float, float, float, float]:
        max_dh = max_yaw_rate_rad_s * default_dt
        max_dv = max_accel_mps2 * default_dt

        dx = float(target_xy[0]) - float(x)
        dy = float(target_xy[1]) - float(y)
        dist = math.hypot(dx, dy)

        if dist > 1e-9:
            desired_h = math.atan2(dy, dx)
        else:
            desired_h = h

        dh = _wrap_angle(desired_h - h)
        dh = max(-max_dh, min(max_dh, dh))
        h = _wrap_angle(h + dh)

        desired_v = min(max_speed_mps, max(0.0, float(desired_v)))
        dv = desired_v - v
        dv = max(-max_dv, min(max_dv, dv))
        v = min(max_speed_mps, max(0.0, v + dv))

        x = float(x) + v * default_dt * math.cos(h)
        y = float(y) + v * default_dt * math.sin(h)

        return x, y, h, v

    def _heading_constrained_warm_start_with_transition(
        pts: List[Tuple[float, float]],
        start_xy: Tuple[float, float],
        start_heading: float,
        start_speed: float,
        warm_steps: int,
        trans_steps: int,
    ) -> List[Tuple[float, float]]:
        """
        Warm start + smooth transition before merging back to the original anchor.

        It only replaces the infeasible prefix. The suffix of the original anchor
        is preserved, but the merge point is delayed until yaw-rate feasibility
        is approximately satisfied.
        """
        if not pts or warm_steps <= 0:
            return pts

        n = len(pts)
        warm_steps = min(max(1, int(warm_steps)), n)
        trans_steps = min(max(0, int(trans_steps)), max(0, n - warm_steps))

        anchor_speeds = _anchor_speeds(pts, start_xy)

        x, y = float(start_xy[0]), float(start_xy[1])
        h = float(start_heading)
        v = min(max_speed_mps, max(0.0, float(start_speed)))

        fixed_prefix: List[Tuple[float, float]] = []

        # ------------------------------------------------------------
        # 1. Heading-constrained warm start.
        # ------------------------------------------------------------
        for i in range(warm_steps):
            lookahead_idx = min(i + 2, n - 1)
            desired_v = anchor_speeds[i] if i < len(anchor_speeds) else v

            x, y, h, v = _rollout_one_step(
                x=x,
                y=y,
                h=h,
                v=v,
                target_xy=pts[lookahead_idx],
                desired_v=desired_v,
            )
            fixed_prefix.append((x, y))

        # ------------------------------------------------------------
        # 2. Transition rollout.
        #    Still generated by constrained rollout, not direct interpolation.
        # ------------------------------------------------------------
        merge_start = warm_steps
        merge_end = min(n, warm_steps + trans_steps)

        for k, i in enumerate(range(merge_start, merge_end)):
            # alpha increases from small to large:
            # early transition tracks lookahead,
            # late transition tracks original anchor more strongly.
            denom = max(1, merge_end - merge_start + 1)
            alpha = float(k + 1) / float(denom)

            lookahead_idx = min(i + 2, n - 1)
            lookahead_target = pts[lookahead_idx]
            anchor_target = pts[i]

            target_x = (1.0 - alpha) * float(lookahead_target[0]) + alpha * float(anchor_target[0])
            target_y = (1.0 - alpha) * float(lookahead_target[1]) + alpha * float(anchor_target[1])
            target_xy = (target_x, target_y)

            desired_v = anchor_speeds[i] if i < len(anchor_speeds) else v

            x, y, h, v = _rollout_one_step(
                x=x,
                y=y,
                h=h,
                v=v,
                target_xy=target_xy,
                desired_v=desired_v,
            )
            fixed_prefix.append((x, y))

        # ------------------------------------------------------------
        # 3. Adaptive merge check.
        #    If directly attaching the next raw anchor creates a sharp turn,
        #    replace more prefix points by constrained rollout until reachable
        #    or until trajectory is exhausted.
        # ------------------------------------------------------------
        while merge_end < n:
            next_anchor = pts[merge_end]
            dx = float(next_anchor[0]) - float(x)
            dy = float(next_anchor[1]) - float(y)
            dist = math.hypot(dx, dy)

            if dist > 1e-9:
                next_h = math.atan2(dy, dx)
            else:
                next_h = h

            required_yaw_rate = abs(_wrap_angle(next_h - h)) / default_dt

            # Also avoid unrealistically large spatial jump at the merge point.
            max_reachable_dist = min(max_speed_mps, max(0.0, v + max_accel_mps2 * default_dt)) * default_dt
            distance_ok = dist <= max_reachable_dist * feasibility_slack
            yaw_ok = required_yaw_rate <= max_yaw_rate_rad_s * feasibility_slack

            if yaw_ok and distance_ok:
                break

            lookahead_idx = min(merge_end + 1, n - 1)
            desired_v = anchor_speeds[merge_end] if merge_end < len(anchor_speeds) else v

            x, y, h, v = _rollout_one_step(
                x=x,
                y=y,
                h=h,
                v=v,
                target_xy=pts[lookahead_idx],
                desired_v=desired_v,
            )
            fixed_prefix.append((x, y))
            merge_end += 1

        return fixed_prefix + pts[merge_end:]

    start_speed = _estimate_start_speed()

    if _is_reverse_anchor_motion():
        print(
            "opt_anchor_kin: reverse anchor -> emergency brake stop "
            f"(start_speed={start_speed:.3f}, decel={max_accel_mps2:.3f})"
        )
        return _emergency_brake_stop_trajectory(start_speed)

    first_dist, first_lateral_offset, heading_error = _first_point_consistency(
        points, current_xy, heading0
    )

    need_warm_start = (
        heading_error > heading_mismatch_threshold_rad
        or first_lateral_offset > first_point_lateral_offset_threshold_m
    )

    print(
        "opt_anchor_kin: precheck "
        f"need_warm_start={need_warm_start} "
        f"first_dist={first_dist:.3f} "
        f"first_lateral_offset={first_lateral_offset:.3f} (thr={first_point_lateral_offset_threshold_m:.3f}) "
        f"heading_error_deg={math.degrees(heading_error):.2f} (thr_deg={math.degrees(heading_mismatch_threshold_rad):.2f}) "
        f"dt={default_dt:.3f}"
    )

    if keep_original_if_feasible and not need_warm_start and _is_original_feasible():
        print("opt_anchor_kin: keeping original (feasible, no warm start)")
        return [
            {
                "t": norm[idx]["t"],
                "x": round(float(points[idx][0]), 3),
                "y": round(float(points[idx][1]), 3),
            }
            for idx in range(len(points))
        ]

    print(
        "opt_anchor_kin: optimizing "
        f"start_speed={start_speed:.3f} "
        f"(max_speed={max_speed_mps:.3f}, max_yaw_rate={max_yaw_rate_rad_s:.3f}, max_accel={max_accel_mps2:.3f}, slack={feasibility_slack:.3f})"
    )

    if need_warm_start:
        print(
            "applying heading-constrained warm start with transition, "
            f"first_dist={first_dist:.2f}, "
            f"first_lateral_offset={first_lateral_offset:.2f}, "
            f"heading_error_deg={math.degrees(heading_error):.1f}"
        )

        points = _heading_constrained_warm_start_with_transition(
            pts=points,
            start_xy=current_xy,
            start_heading=heading0,
            start_speed=start_speed,
            warm_steps=min(warm_start_steps, len(points)),
            trans_steps=transition_steps,
        )

    # After replacing the prefix, run the original constrained smoothing logic.
    anchor_speeds = _anchor_speeds(points, current_xy)

    x, y = float(cur_x), float(cur_y)
    h = float(heading0)
    v = min(max_speed_mps, max(0.0, start_speed))

    optimized: List[Dict] = []

    for idx, anchor_xy in enumerate(points):
        desired_v = anchor_speeds[idx] if idx < len(anchor_speeds) else v

        x, y, h, v = _rollout_one_step(
            x=x,
            y=y,
            h=h,
            v=v,
            target_xy=anchor_xy,
            desired_v=desired_v,
        )

        # Pullback should be weak near the modified prefix.
        if need_warm_start:
            if idx < warm_start_steps:
                pull_scale = 0.0
            elif idx < warm_start_steps + transition_steps:
                pull_scale = 0.3
            else:
                pull_scale = 1.0
        else:
            pull_scale = 1.0

        effective_max_pull = max_pull_per_step_m * pull_scale

        dx = float(anchor_xy[0]) - float(x)
        dy = float(anchor_xy[1]) - float(y)
        dev = math.hypot(dx, dy)

        if dev > 1e-9 and pull_scale > 0.0:
            pull = 0.0

            if effective_max_pull > 0.0:
                pull = min(effective_max_pull, dev)

            if max_deviation_m > 0.0 and dev > max_deviation_m:
                pull = max(pull, (dev - max_deviation_m) * pull_scale)

            if pull > 0.0:
                x += pull * dx / dev
                y += pull * dy / dev

        optimized.append(
            {
                "t": norm[idx]["t"],
                "x": round(float(x), 3),
                "y": round(float(y), 3),
            }
        )

    return optimized


def _cross_z(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _headings_along_polyline_xy(
    points: List[Tuple[float, float]],
) -> List[float]:
    """Heading (rad, atan2 dy,dx) at each vertex; tangent from consecutive segments."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    out: List[float] = []
    for i in range(n):
        if i < n - 1:
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
        else:
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
        if dx == 0.0 and dy == 0.0:
            out.append(out[-1] if out else 0.0)
        else:
            out.append(math.atan2(dy, dx))
    return out


def _oriented_box_corners_nusc(
    cx: float,
    cy: float,
    heading_rad: float,
    length_m: float,
    width_m: float,
) -> List[Tuple[float, float]]:
    """Rectangle corners in scene frame; matches ``datasets.nuscenes_utils.get_corners`` layout."""
    l = float(length_m)
    w = float(width_m)
    c = math.cos(heading_rad)
    s = math.sin(heading_rad)
    # Rows of rot match get_rot(h): [[cos, sin], [-sin, cos]]; corners are rows @ rot.
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


def _polygon_signed_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        s += poly[i][0] * poly[j][1]
        s -= poly[j][0] * poly[i][1]
    return s * 0.5


def _polygon_area_shoelace(poly: List[Tuple[float, float]]) -> float:
    return abs(_polygon_signed_area(poly))


def _as_ccw_polygon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(poly) < 3:
        return poly
    if _polygon_signed_area(poly) < 0.0:
        return list(reversed(poly))
    return poly


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


def _dedupe_points(
    pts: List[Tuple[float, float]], eps: float = 1e-7
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in pts:
        ok = True
        for ox, oy in out:
            if math.hypot(x - ox, y - oy) <= eps:
                ok = False
                break
        if ok:
            out.append((x, y))
    return out


def _convex_hull_monotone_chain(
    points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return _cross_z(a[0] - o[0], a[1] - o[1], b[0] - o[0], b[1] - o[1])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


def _convex_polygons_intersection_area(
    poly_a: List[Tuple[float, float]],
    poly_b: List[Tuple[float, float]],
) -> float:
    """Intersection area of two convex polygons."""
    poly_a = _as_ccw_polygon(poly_a)
    poly_b = _as_ccw_polygon(poly_b)
    cand: List[Tuple[float, float]] = []
    for p in poly_a:
        if _point_in_convex_polygon_ccw(p[0], p[1], poly_b):
            cand.append(p)
    for p in poly_b:
        if _point_in_convex_polygon_ccw(p[0], p[1], poly_a):
            cand.append(p)
    na, nb = len(poly_a), len(poly_b)
    for i in range(na):
        a1, a2 = poly_a[i], poly_a[(i + 1) % na]
        for j in range(nb):
            b1, b2 = poly_b[j], poly_b[(j + 1) % nb]
            ipt = _segment_segment_intersection(a1, a2, b1, b2)
            if ipt is not None:
                cand.append(ipt)
    cand = _dedupe_points(cand)
    if len(cand) < 3:
        return 0.0
    hull = _convex_hull_monotone_chain(cand)
    if len(hull) < 3:
        return 0.0
    return _polygon_area_shoelace(hull)


def _vehicle_box_iou_pure(
    corners_a: List[Tuple[float, float]],
    corners_b: List[Tuple[float, float]],
) -> float:
    area_a = _polygon_area_shoelace(corners_a)
    area_b = _polygon_area_shoelace(corners_b)
    inter = _convex_polygons_intersection_area(corners_a, corners_b)
    union = area_a + area_b - inter
    if union <= 1e-12:
        return 0.0
    return inter / union


def _vehicle_box_iou(
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
        inter = float(pa.intersection(pb).area)
        union = float(pa.union(pb).area)
        if union <= 1e-12:
            return 0.0
        return inter / union
    except Exception:
        return _vehicle_box_iou_pure(corners_a, corners_b)


def _vehicle_boxes_min_distance_pure(
    corners_a: List[Tuple[float, float]],
    corners_b: List[Tuple[float, float]],
) -> float:
    """Minimum Euclidean distance between boundaries of two convex vehicle rectangles."""
    if _convex_polygons_intersection_area(corners_a, corners_b) > 1e-9:
        return 0.0
    na, nb = len(corners_a), len(corners_b)
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


def _extract_xy_heading_from_traj_point(
    point: Dict[str, object],
) -> Optional[Tuple[float, float, Optional[float]]]:
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        return None
    try:
        x = float(point["x"])
        y = float(point["y"])
    except (TypeError, ValueError):
        return None
    h: Optional[float] = None
    try:
        if "hx" in point and "hy" in point:
            hx = float(point["hx"])
            hy = float(point["hy"])
            if math.hypot(hx, hy) > 1e-9:
                h = math.atan2(hy, hx)
        elif "heading" in point:
            h = float(point["heading"])
    except (TypeError, ValueError):
        h = None
    return (x, y, h)


def _parse_agent_length_width_entry(
    entry: object,
    *,
    default_length: float = 4.5,
    default_width: float = 1.8,
) -> Tuple[float, float]:
    """Parse one ``agent_length_width`` row: ``{length, width}`` in meters."""
    if not isinstance(entry, dict):
        return default_length, default_width
    try:
        length_f = float(entry.get("length"))
    except (TypeError, ValueError):
        length_f = float("nan")
    try:
        width_f = float(entry.get("width"))
    except (TypeError, ValueError):
        width_f = float("nan")
    if not math.isfinite(length_f) or length_f <= 0.0:
        length_f = default_length
    if not math.isfinite(width_f) or width_f <= 0.0:
        width_f = default_width
    return length_f, width_f


def load_vehicle_dims_from_trajectory_json(
    scene_text_path: Optional[str],
    adv_row_id: int,
) -> Dict[str, float]:
    """Ego/adversarial L×W from ``*_trajectory.json`` ``agent_length_width`` (row 0 = ego)."""
    defaults = {
        "ego_length": 4.5,
        "ego_width": 1.8,
        "target_length": 4.5,
        "target_width": 1.8,
    }
    if not scene_text_path:
        return dict(defaults)
    txt_path = Path(scene_text_path)
    if not txt_path.is_file():
        return dict(defaults)
    json_path = txt_path.parent / f"{txt_path.stem}_trajectory.json"
    if not json_path.is_file():
        return dict(defaults)
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return dict(defaults)
    if not isinstance(payload, dict):
        return dict(defaults)
    alw = payload.get("agent_length_width")
    if not isinstance(alw, list) or not alw:
        return dict(defaults)

    ego_length, ego_width = _parse_agent_length_width_entry(
        alw[0],
        default_length=defaults["ego_length"],
        default_width=defaults["ego_width"],
    )
    if adv_row_id < 0 or adv_row_id >= len(alw):
        target_length, target_width = defaults["target_length"], defaults["target_width"]
    else:
        target_length, target_width = _parse_agent_length_width_entry(
            alw[adv_row_id],
            default_length=defaults["target_length"],
            default_width=defaults["target_width"],
        )
    return {
        "ego_length": ego_length,
        "ego_width": ego_width,
        "target_length": target_length,
        "target_width": target_width,
    }


def trajectory_collision_check(
    ego_future_trajectory: List[Dict],
    adv_future_trajectory: List[Dict],
    collision_distance_m: float = 3.5,
    *,
    ego_length_m: float = 4.5,
    ego_width_m: float = 2.0,
    target_length_m: float = 4.5,
    target_width_m: float = 2.0,
    vehicle_collision_iou_thresh: float = 0.02,
) -> Dict[str, object]:
    """Check ego vs. target future trajectories at matched timestamps using oriented vehicle boxes.

    Collision is detected from oriented-box minimum boundary distance:
    ``min_distance_m <= BOX_COLLISION_DISTANCE_THRESH_M``.

    Headings default to tangents along each polyline when not provided per point (``hx``/``hy``
    or ``heading``).

    The third positional argument ``collision_distance_m`` is kept for backward compatibility
    with older call sites. ``vehicle_collision_iou_thresh`` is also kept for compatibility
    with existing call signatures, but IoU is no longer used in this detector.

    ``min_distance_m`` is the minimum surface separation between the two oriented vehicle boxes
    (0 when overlapping). ``closest_t_index`` is always 0-based. On collision, it is the first
    timestep satisfying the distance threshold; otherwise it is the timestep with minimum separation.
    """
    _ = collision_distance_m
    _ = vehicle_collision_iou_thresh
    ego_states: List[Tuple[float, float, Optional[float]]] = []
    adv_states: List[Tuple[float, float, Optional[float]]] = []

    for point in ego_future_trajectory:
        ex = _extract_xy_heading_from_traj_point(point)
        if ex is not None:
            ego_states.append(ex)
    for point in adv_future_trajectory:
        ax = _extract_xy_heading_from_traj_point(point)
        if ax is not None:
            adv_states.append(ax)

    n = min(len(ego_states), len(adv_states))
    if n == 0:
        return {
            "collision": False,
            "min_distance_m": None,
            "closest_t": None,
            "closest_t_index": None,
        }

    ego_xy = [(s[0], s[1]) for s in ego_states]
    adv_xy = [(s[0], s[1]) for s in adv_states]
    ego_headings_from_traj = _headings_along_polyline_xy(ego_xy)
    adv_headings_from_traj = _headings_along_polyline_xy(adv_xy)

    min_box_sep_m = float("inf")
    min_box_sep_idx = 0
    first_collision_idx: Optional[int] = None

    for i in range(n):
        ex, ey, eh = ego_states[i]
        ax, ay, ah = adv_states[i]

        he = eh if eh is not None else ego_headings_from_traj[i]
        ha = ah if ah is not None else adv_headings_from_traj[i]

        ego_corners = _oriented_box_corners_nusc(ex, ey, he, ego_length_m, ego_width_m)
        adv_corners = _oriented_box_corners_nusc(ax, ay, ha, target_length_m, target_width_m)
        sep_m = _vehicle_boxes_min_distance_m(ego_corners, adv_corners)
        if sep_m < min_box_sep_m:
            min_box_sep_m = sep_m
            min_box_sep_idx = i
        if first_collision_idx is None and sep_m <= float(BOX_COLLISION_DISTANCE_THRESH_M):
            first_collision_idx = i

    collision = first_collision_idx is not None
    highlight_idx = first_collision_idx if collision else min_box_sep_idx
    closest_idx = int(highlight_idx)

    return {
        "collision": collision,
        "min_distance_m": round(float(min_box_sep_m), 3),
        "closest_t": round(closest_idx * 0.5, 1),
        "closest_t_index": closest_idx,
    }


def _first_xy_from_future_traj(traj: List[Dict]) -> Optional[Tuple[float, float]]:
    for p in traj:
        if isinstance(p, dict) and "x" in p and "y" in p:
            try:
                return (float(p["x"]), float(p["y"]))
            except (TypeError, ValueError):
                continue
    return None


def _world_to_ego_current_local(
    x: float,
    y: float,
    origin_xy: Tuple[float, float],
    theta_ref_rad: float,
) -> Tuple[float, float]:
    """Scene/world (x forward, y right) -> local frame at ego_current with +x along theta_ref."""
    x0, y0 = origin_xy
    dx = float(x) - x0
    dy = float(y) - y0
    c = math.cos(theta_ref_rad)
    s = math.sin(theta_ref_rad)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return lx, ly


def _traj_as_xy_tuples(traj: object) -> List[Tuple[float, float]]:
    """Flatten trajectory points to (x, y), supporting dict or length-2 sequence points."""
    out: List[Tuple[float, float]] = []
    if not isinstance(traj, list) or not traj:
        return out
    for p in traj:
        try:
            if isinstance(p, dict) and "x" in p and "y" in p:
                out.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
        except (TypeError, ValueError):
            continue
    return out


def _traj_max_abs_xy(traj: object) -> float:
    m = 0.0
    for x, y in _traj_as_xy_tuples(traj):
        m = max(m, abs(x), abs(y))
    return m


def _future_traj_to_ego_current_local(
    traj: object,
    origin_xy: Tuple[float, float],
    theta_ref_rad: float,
) -> List:
    """Map each trajectory point from scene/world to ego-current local; supports dict or [x, y] points."""
    out: List = []
    if not isinstance(traj, list):
        return out
    for p in traj:
        if isinstance(p, dict):
            q = dict(p)
            if "x" in q and "y" in q:
                try:
                    lx, ly = _world_to_ego_current_local(
                        float(q["x"]), float(q["y"]), origin_xy, theta_ref_rad
                    )
                    q["x"] = round(lx, 3)
                    q["y"] = round(ly, 3)
                except (TypeError, ValueError):
                    pass
            out.append(q)
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                lx, ly = _world_to_ego_current_local(
                    float(p[0]), float(p[1]), origin_xy, theta_ref_rad
                )
                out.append([round(lx, 3), round(ly, 3)])
            except (TypeError, ValueError):
                out.append(list(p) if isinstance(p, tuple) else p)
        else:
            out.append(p)
    return out


def _maybe_world_traj_to_ego_local(
    traj: object,
    origin_xy: Tuple[float, float],
    theta_ref_rad: float,
    *,
    extent_thresh_m: float = 100.0,
) -> object:
    """
    If trajectory coordinates look like scene/world (large map extent), convert to ego-current local
    so polylines align with current-state boxes from scene_text. Typical ego-centric futures
    (small extent) are left unchanged.
    """
    if not traj:
        return traj
    if _traj_max_abs_xy(traj) <= float(extent_thresh_m):
        return traj
    return _future_traj_to_ego_current_local(traj, origin_xy, theta_ref_rad)


def visualize_trajectory_validation(
    sample_id: str,
    ego_history: List[Tuple[float, float]],
    adv_vehicle_id: int,
    ego_future_trajectory,
    adv_future_trajectory,
    adv_future_trajectory_before_opt=None,
    scene_text: str = "",
    collision_info: Optional[Dict[str, object]] = None,
    source: str = "unknown",
) -> Optional[str]:
    """Plot trajectory debug image for Step-3 collision validation."""

    if plt is None or Rectangle is None or mtransforms is None:
        print("[traj_tools] matplotlib not installed, skip debug visualization.")
        return None

    vis_dir = Path(os.getenv("TRAJ_DEBUG_VIS_DIR", "out/debug_traj_vis"))
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    vehicle_width = 2.0
    vehicle_length = 4.5

    def to_plot_xy(x_raw: float, y_raw: float) -> Tuple[float, float]:
        return -float(y_raw), float(x_raw)

    def scene_heading_to_plot_heading(scene_heading_rad: float) -> float:
        return (math.pi / 2.0) + float(scene_heading_rad)

    def _add_oriented_box(x, y, heading_rad, color, linewidth, alpha):
        rect = Rectangle(
            (x - vehicle_length / 2.0, y - vehicle_width / 2.0),
            vehicle_length,
            vehicle_width,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            alpha=alpha,
        )
        rect.set_transform(mtransforms.Affine2D().rotate_around(x, y, heading_rad) + ax.transData)
        ax.add_patch(rect)

    def _scene_headings_along_traj(raw_points):
        n = len(raw_points)
        if n == 0:
            return []
        if n == 1:
            return [0.0]
        out = []
        for i in range(n):
            if i < n - 1:
                dx = raw_points[i + 1][0] - raw_points[i][0]
                dy = raw_points[i + 1][1] - raw_points[i][1]
            else:
                dx = raw_points[i][0] - raw_points[i - 1][0]
                dy = raw_points[i][1] - raw_points[i - 1][1]
            if dx == 0.0 and dy == 0.0:
                out.append(out[-1] if out else 0.0)
            else:
                out.append(math.atan2(dy, dx))
        return out

    def draw_future(traj, color, label):
        if not traj:
            return []

        raw_points = [(float(p[0]), float(p[1])) for p in traj]
        points = [to_plot_xy(x, y) for x, y in raw_points]

        tangents = _scene_headings_along_traj(raw_points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        ax.plot(xs, ys, color=color, linewidth=2.0, label=label)

        for i, (x, y) in enumerate(points):
            fade = 0.25 + 0.75 * (i + 1) / max(1, len(points))
            heading_rad = scene_heading_to_plot_heading(tangents[i])
            _add_oriented_box(x, y, heading_rad, color, 1.5, fade)

        return points

    scene_text = scene_text or ""
    ego_heading_scene, agent_headings_scene = extract_scene_headings(scene_text)

    def _extract_current_states_xy_heading(
        text: str,
    ) -> Tuple[Optional[Tuple[float, float, Optional[float]]], Dict[int, Tuple[float, float, Optional[float]]]]:
        """
        Extract current-state (x, y, heading) from scene_text.

        Returns:
          - ego: (x, y, heading) in world/scene frame, heading may be None
          - agents: {agent_id: (x, y, heading)} heading may be None
        """
        ego_state: Optional[Tuple[float, float, Optional[float]]] = None
        agent_states: Dict[int, Tuple[float, float, Optional[float]]] = {}

        ego_block_match = re.search(
            r"EGO:\s*(.*?)(?:\n\s*\n|\nSURROUNDING AGENTS|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if ego_block_match:
            block = ego_block_match.group(1)
            m_xy = re.search(
                r"Current state:.*?x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m",
                block,
                re.IGNORECASE | re.DOTALL,
            )
            if m_xy:
                x = float(m_xy.group(1))
                y = float(m_xy.group(2))
                m_h = re.search(r"heading=([-+]?(?:\d+(?:\.\d+)?|nan))", block, re.IGNORECASE)
                heading: Optional[float] = None
                if m_h and m_h.group(1).lower() != "nan":
                    heading = float(m_h.group(1))
                ego_state = (x, y, heading)

        current_id: Optional[int] = None
        id_pat = re.compile(r"^\s*id:\s*(\d+),", re.IGNORECASE)
        agent_pat = re.compile(r"^\s*Agent\s+(\d+)\s*:", re.IGNORECASE)
        xy_pat = re.compile(
            r"Current state:.*?x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m",
            re.IGNORECASE,
        )
        heading_pat = re.compile(r"heading=([-+]?(?:\d+(?:\.\d+)?|nan))", re.IGNORECASE)

        for line in text.splitlines():
            id_match = id_pat.search(line)
            agent_match = agent_pat.search(line)
            if id_match or agent_match:
                current_id = int((id_match or agent_match).group(1))
                continue

            if current_id is None:
                continue

            if "Current state:" not in line:
                continue

            m_xy = xy_pat.search(line)
            if not m_xy:
                continue
            x = float(m_xy.group(1))
            y = float(m_xy.group(2))
            m_h = heading_pat.search(line)
            heading: Optional[float] = None
            if m_h and m_h.group(1).lower() != "nan":
                heading = float(m_h.group(1))
            agent_states[current_id] = (x, y, heading)

        return ego_state, agent_states

    ego_state_scene, agent_states_scene = _extract_current_states_xy_heading(scene_text)

    ego_pts_world = _traj_as_xy_tuples(ego_future_trajectory)

    ego_current_origin: Optional[Tuple[float, float]] = None
    if ego_state_scene is not None:
        ego_current_origin = (float(ego_state_scene[0]), float(ego_state_scene[1]))
    elif ego_history:
        ego_current_origin = (float(ego_history[-1][0]), float(ego_history[-1][1]))

    # Use heading from scene_text current state; avoid inferring from history for "current" visualization.
    theta_ego_current = float(ego_heading_scene)
    if (not theta_ego_current) and ego_state_scene is not None and ego_state_scene[2] is not None:
        theta_ego_current = float(ego_state_scene[2])
    if (not theta_ego_current) and len(ego_pts_world) >= 2:
        theta_ego_current = math.atan2(
            ego_pts_world[1][1] - ego_pts_world[0][1],
            ego_pts_world[1][0] - ego_pts_world[0][0],
        )

    extent_world = float(os.getenv("TRAJ_VIZ_WORLD_EXTENT_THRESH_M", "100.0"))

    def normalize_future_trajectory_format(traj):
        if not traj:
            return traj
        if isinstance(traj[0], dict):
            return [
                [float(p["x"]), float(p["y"])]
                for p in traj
                if "x" in p and "y" in p
            ]
        return traj

    def _traj_for_draw(raw):
        if not raw:
            return raw
        t = raw
        if ego_current_origin is not None:
            t = _maybe_world_traj_to_ego_local(
                t, ego_current_origin, theta_ego_current, extent_thresh_m=extent_world
            )
        return normalize_future_trajectory_format(t)

    ego_future_for_draw = _traj_for_draw(ego_future_trajectory)
    adv_future_for_draw = _traj_for_draw(adv_future_trajectory)
    adv_before_for_draw = (
        _traj_for_draw(adv_future_trajectory_before_opt) if adv_future_trajectory_before_opt else []
    )

    ego_future_points = draw_future(ego_future_for_draw, "green", "ego_future_8s")
    adv_future_points = draw_future(adv_future_for_draw, "red", "adversarial_future_8s")
    if adv_before_for_draw:
        # Draw pre-optimization adversarial path as dashed blue for comparison.
        raw_points = [(float(p[0]), float(p[1])) for p in adv_before_for_draw]
        points = [to_plot_xy(x, y) for x, y in raw_points]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            color="blue",
            linewidth=2.0,
            linestyle="--",
            label="adversarial_future_before_opt",
            alpha=0.9,
        )

    def _heading_from_traj_points(traj_points) -> float:
        if len(traj_points) >= 2:
            dx = float(traj_points[1][0]) - float(traj_points[0][0])
            dy = float(traj_points[1][1]) - float(traj_points[0][1])
            if dx != 0.0 or dy != 0.0:
                return math.atan2(dy, dx)
        return 0.0

    def _draw_box_arrow_at_plot_frame(x_plot: float, y_plot: float, heading_plot: float, arrow_color: str) -> None:
        _add_oriented_box(x_plot, y_plot, heading_plot, "black", 2.2, 1.0)
        arrow_len = max(1.0, vehicle_length * 0.75)
        dx = arrow_len * math.cos(heading_plot)
        dy = arrow_len * math.sin(heading_plot)
        ax.arrow(
            x_plot,
            y_plot,
            dx,
            dy,
            width=0.08,
            head_width=0.6,
            head_length=0.8,
            length_includes_head=True,
            color=arrow_color,
            alpha=0.95,
            zorder=5,
        )

    def _draw_current_state_box_from_scene(
        current_world_xy: Optional[Tuple[float, float]],
        heading_scene_rad: Optional[float],
        arrow_color: str,
    ) -> None:
        if current_world_xy is None:
            return

        if ego_current_origin is not None:
            local_xy = _world_to_ego_current_local(
                current_world_xy[0], current_world_xy[1], ego_current_origin, theta_ego_current
            )
            heading_local_scene = (
                float(heading_scene_rad) - theta_ego_current
                if heading_scene_rad is not None
                else 0.0
            )
        else:
            local_xy = current_world_xy
            heading_local_scene = float(heading_scene_rad) if heading_scene_rad is not None else 0.0

        x_plot, y_plot = to_plot_xy(local_xy[0], local_xy[1])
        heading_plot = scene_heading_to_plot_heading(heading_local_scene)
        _draw_box_arrow_at_plot_frame(x_plot, y_plot, heading_plot, arrow_color)

    ego_heading_for_current = float(ego_heading_scene)
    if ego_heading_for_current == 0.0 and ego_pts_world:
        ego_heading_for_current = _heading_from_traj_points(ego_pts_world)
    adv_heading_for_current = agent_headings_scene.get(adv_vehicle_id)
    if adv_heading_for_current is None and adv_future_trajectory:
        adv_xy_pts = _traj_as_xy_tuples(adv_future_trajectory)
        if len(adv_xy_pts) >= 2:
            adv_heading_for_current = _heading_from_traj_points(adv_xy_pts)

    ego_xy_for_current: Optional[Tuple[float, float]] = (
        (float(ego_state_scene[0]), float(ego_state_scene[1])) if ego_state_scene is not None else None
    )
    adv_state = agent_states_scene.get(adv_vehicle_id)
    adv_xy_for_current: Optional[Tuple[float, float]] = (
        (float(adv_state[0]), float(adv_state[1])) if adv_state is not None else None
    )

    def _adv_scene_xy_in_traj_frame() -> Optional[Tuple[float, float]]:
        """Same (x,y) plane as adversarial polyline after _traj_for_draw (ego-local or mapped)."""
        if adv_xy_for_current is None:
            return None
        if ego_current_origin is not None:
            return _world_to_ego_current_local(
                adv_xy_for_current[0], adv_xy_for_current[1], ego_current_origin, theta_ego_current
            )
        return (float(adv_xy_for_current[0]), float(adv_xy_for_current[1]))

    def _draw_adversary_current_pose() -> None:
        """
        RAG / anchor trajectories often live in the anchor's own local frame while scene_text gives
        the query scene's ego-local coordinates — both can have modest extent (<100 m) so
        _maybe_world_traj_to_ego_local does not fire. When scene agent position disagrees with the
        first waypoint of the trajectory used for collision, draw the adversary box at that first
        waypoint so the debug figure matches trajectory_collision_check.
        """
        align_max_m = float(os.getenv("TRAJ_VIZ_ADV_SCENE_ALIGN_MAX_M", "10.0"))
        scene_xy_traj_plane = _adv_scene_xy_in_traj_frame()
        traj0: Optional[Tuple[float, float]] = None
        if adv_future_for_draw and isinstance(adv_future_for_draw[0], (list, tuple)):
            p0 = adv_future_for_draw[0]
            if len(p0) >= 2:
                try:
                    traj0 = (float(p0[0]), float(p0[1]))
                except (TypeError, ValueError):
                    traj0 = None
        snap = False
        if traj0 is not None:
            if scene_xy_traj_plane is None:
                snap = True
            else:
                sep = math.hypot(scene_xy_traj_plane[0] - traj0[0], scene_xy_traj_plane[1] - traj0[1])
                snap = sep > align_max_m
        if snap and traj0 is not None:
            raw_points = [(float(p[0]), float(p[1])) for p in adv_future_for_draw]
            tangents = _scene_headings_along_traj(raw_points)
            heading_plot = scene_heading_to_plot_heading(tangents[0])
            x_plot, y_plot = to_plot_xy(traj0[0], traj0[1])
            _draw_box_arrow_at_plot_frame(x_plot, y_plot, heading_plot, "red")
            return
        _draw_current_state_box_from_scene(adv_xy_for_current, adv_heading_for_current, "red")

    _draw_current_state_box_from_scene(ego_xy_for_current, ego_heading_for_current, "green")
    _draw_adversary_current_pose()

    if collision_info and collision_info.get("closest_t_index") is not None:
        idx = int(collision_info["closest_t_index"])
        if idx < len(ego_future_points) and idx < len(adv_future_points):
            ex, ey = ego_future_points[idx]
            axx, ayy = adv_future_points[idx]
            ax.plot([ex, axx], [ey, ayy], color="orange", linewidth=2.2, linestyle=":")
            ax.scatter([ex, axx], [ey, ayy], color="orange", s=35)

    all_xy = []
    all_xy.extend(to_plot_xy(p[0], p[1]) for p in ego_future_for_draw)
    all_xy.extend(to_plot_xy(p[0], p[1]) for p in adv_future_for_draw)
    all_xy.extend(to_plot_xy(p[0], p[1]) for p in adv_before_for_draw)

    if all_xy:
        xs = [p[0] for p in all_xy]
        ys = [p[1] for p in all_xy]
        pad = 8.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Step3 Validation | sample={sample_id} | adv={adv_vehicle_id}")

    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    is_collision = collision_info.get("collision") if collision_info else False
    # Keep full sample_id for on-plot text, but shorten it for file naming.
    # Example: "boston-seaport_test_0_000010" -> "0_000010"
    sample_id_for_file = str(sample_id)
    m = re.search(r"(\d+_\d+)$", sample_id_for_file)
    if m:
        sample_id_for_file = m.group(1)

    save_path = (
        vis_dir / f"{sample_id_for_file}_adv{adv_vehicle_id}_{source}_{is_collision}.png"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

    return str(save_path)
