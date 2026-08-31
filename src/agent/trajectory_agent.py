import json
import os
import math
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sympy import false

from .base_agent import AgentMessage, BaseAgent
from .memory import SceneContext
from .rag.crash_data_processor import (
    DT_CTRL,
    _stage_time_indices,
    classify_interaction_stages,
    sanitize_stage_time_indices_vs_impact,
)
from .retry_agent import RetryAgent
from .tools.api_tools import call_openai_chat, extract_json
from .tools.traj_tools import (
    extract_agent_points,
    fallback_linear_traj,
    load_vehicle_dims_from_trajectory_json,
    normalize_trajectory,
    optimize_anchor_trajectory_kinematics,
    truncate_trajectory_after_collision_to_hard_stop,
    trajectory_collision_check,
    visualize_trajectory_validation,
)


def _build_system_prompt(output_xy_trajectory: bool) -> str:
    if output_xy_trajectory:
        output_spec = (
            '  "anchor_trajectory_6s": [{"t":0.5,"x":float,"y":float}, ... 12 points],\n'
        )
        step2_format = (
            "Step 2 output format:\n"
            "- Return 12 trajectory points from t=0.5 to t=6.0 (step=0.5s)\n"
            "- Coordinates in ego-centered BEV system\n\n"
        )
    else:
        output_spec = (
            '  "anchor_trajectory_6s": [{"t":0.5,"acceleration":float,"steering_angle":float}, ... 12 control steps],\n'
        )
        step2_format = (
            "Step 2 output format:\n"
            "- Return 12 control steps from t=0.5 to t=6.0 (step=0.5s)\n"
            "- Each step uses bicycle-model controls in ego-centered BEV conventions\n"
            "- acceleration means longitudinal acceleration (m/s^2), steering_angle means steering angle (deg)\n\n"
        )

    return (
        "You are an autonomous-driving adversarial trajectory generator for safety-critical scenario synthesis. "
        "Your task is to generate one reference anchor for a specified adversarial vehicle. "
        "This anchor will be used as guidance for a diffusion model to generate adversarial safety-critical driving scenarios.\n\n"
        "The input includes:\n"
        "- Ego vehicle intention and predicted future trajectory\n"
        "- Historical trajectories of ego and adversarial vehicles\n"
        "- The adversarial vehicle ID\n"
        "- The inferred potential adversarial behavior\n\n"
        "Goal:\n"
        "Generate a physically plausible adversarial plan that COLLIDES with the ego vehicle "
        "within the next 6 seconds. The trajectory must be human-aware: it should include "
        "strategic maneuvers, small hesitations, and emergency avoidance or braking actions, "
        "but ultimately still result in a collision, mimicking real-world accident behavior.\n\n"
        "Requirements:\n"
        "1. The predicted ego future trajectory is GIVEN and must be treated as ground truth. Do NOT modify or re-predict it.\n"
        "2. The adversarial plan must intersect with the ego trajectory within 6 seconds.\n"
        "3. Unless the behavior explicitly involves reversing, the adversarial motion should move forward along current heading.\n"
        f"{step2_format}"
        "Output rules:\n"
        "- Always output valid JSON only.\n"
        "- Do not include explanations, markdown, or additional text outside the JSON.\n\n"
        "The output must follow this schema exactly:\n"
        "{\n"
        '  "adversarial_vehicle_id": int,\n'
        '  "reasoning_steps": [\n'
        '    {"step 1 Analyze ego trajectory and collision target region": "..."},\n'
        '    {"step 2 Generate adversarial plan with behavior reasoning": "..."},\n'
        '    {"step 3 Collision check": "True or False"}\n'
        "  ],\n"
        f"{output_spec}"
        '  "anchor_confidence": float,\n'
        '  "closest_t_index": int,\n'
        '  "behavior_tag": "string"\n'
        "}"
    )


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _mode_adversarial_behavior(mode: Dict) -> str:
    """NHTSA Layer-1 mode text for LLM reference (scenario + adversarial maneuver)."""
    if not isinstance(mode, dict):
        return ""
    parts: List[str] = []
    for key in ("scenario_type", "adversarial_maneuver", "description"):
        val = str(mode.get(key) or "").strip()
        if val and val not in parts:
            parts.append(val)
    return " | ".join(parts)


def _format_reference_trajectory(reference_trajectory: object) -> str:
    if not reference_trajectory:
        return ""
    if isinstance(reference_trajectory, str):
        return reference_trajectory.strip()
    try:
        return json.dumps(reference_trajectory, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(reference_trajectory)


def _rag_reference_prompt_fields(ctx: SceneContext) -> Tuple[str, str]:
    """Resolve mode-level adversarial_behavior and reference_trajectory for the user prompt."""
    mode = ctx.retrieved_mode if isinstance(ctx.retrieved_mode, dict) else {}
    if not mode:
        anchor = ctx.retrieved_anchor if isinstance(ctx.retrieved_anchor, dict) else {}
        if str(anchor.get("rag_layer") or "").strip() == "mode" or anchor.get("precrash_code"):
            mode = anchor
    ref = ctx.reference_trajectory if isinstance(ctx.reference_trajectory, dict) else {}
    if not ref and isinstance(mode, dict):
        ref = mode.get("reference_trajectory") or {}
        if not ref and isinstance(mode.get("metadata"), dict):
            ref = mode["metadata"].get("reference_trajectory") or {}
    return _mode_adversarial_behavior(mode), _format_reference_trajectory(ref)


def _build_user_prompt(
    ctx: SceneContext,
    ego_history: List[Tuple[float, float]],
    adv_history_points: List[Tuple[float, float]],
    adv_current_state: Dict[str, float],
    predicted_ego: List[Tuple[float, float]],
    output_xy_trajectory: bool,
    adversarial_behavior: str = "",
    reference_trajectory: str = "",
) -> str:
    if output_xy_trajectory:
        step2_requirement = (
            f"Step 2: Generate a 12-point human-aware adversarial trajectory (t=0.5 to 6.0) starting from current state {adv_current_state}. "
            "The trajectory must:\n"
            "- Lead to collision with the ego trajectory within 6 seconds, but not by naive straight-line acceleration\n"
            "- Include realistic human driving behavior: strategic maneuvers, small hesitations\n"
            "- Incorporate emergency braking or avoidance actions close to collision, but collision still occurs\n"
            "- Be smooth and physically plausible\n"
            "- Reflect the specified adversarial behavior\n"
        )
    else:
        step2_requirement = (
            f"Step 2: Generate a 12-step human-aware adversarial control sequence (t=0.5 to 6.0) starting from current state {adv_current_state}. "
            "Each step must follow format {t, acceleration, steering_angle}. The controls must:\n"
            "- Lead to collision with the ego trajectory within 6 seconds, but not via direct straight-line crash\n"
            "- Include realistic human driving behavior: strategic maneuvers, small hesitations\n"
            "- Incorporate emergency braking or avoidance actions close to collision, but collision still occurs\n"
            "- Be smooth and physically plausible\n"
            "- Reflect the specified adversarial behavior\n"
        )

    return (
        "Use the following information to generate one human-aware adversarial anchor trajectory.\n\n"
        f"- ego_driving_intention: {ctx.ego_driving_intention}\n"
        f"- ego_history: {ego_history}\n"
        f"- predicted_ego_future: {predicted_ego}\n"
        f"- adversarial_vehicle_id: {ctx.adversarial_vehicle_ids}\n"
        f"- adversarial_vehicle_history: {adv_history_points}\n"
        f"- 【IMPORTANT】potential_adversarial_behavior: {ctx.potential_adversarial_behaviors}\n\n"

        "Reasoning steps:\n"
        "Step 1: Analyze the predicted ego future trajectory to identify a collision target point "
        "(i.e., where and when collision should occur within 6 seconds).\n"
        f"{step2_requirement}"

        "Step 3: Check if the adversarial trajectory intersects with the ego trajectory.\n"
        "If no collision occurs, revise Step 2 until collision is achieved, ensuring the trajectory remains human-aware.\n\n"
        + (
            "Reference example from a real-world collision event:\n"
            f"- adversarial_behavior (retrieved mode): {adversarial_behavior}\n"
            f"- reference_trajectory: {reference_trajectory}\n"
            if (adversarial_behavior or reference_trajectory)
            else ""
        )
    )


def _update_step3_reasoning(parsed: Dict, collision_info: Dict[str, object], retry_count: int) -> None:
    reasoning_steps = parsed.get("reasoning_steps")
    if not isinstance(reasoning_steps, list):
        print("reasoning_steps is not a list, cannot update step 3 reasoning.")
        return
    step3_text = (
        "Collision check (tool validated): "
        f"{collision_info.get('collision')} | "
        f"min_distance_m={collision_info.get('min_distance_m')} | "
        f"closest_t={collision_info.get('closest_t')} | "
        f"retry_count={retry_count}"
    )
    if len(reasoning_steps) >= 3 and isinstance(reasoning_steps[2], dict):
        reasoning_steps[2] = {"step 3": step3_text}
    elif len(reasoning_steps) >= 3:
        reasoning_steps[2] = step3_text
    else:
        reasoning_steps.append({"step 3": step3_text})


def _extract_ego_future_trajectory(parsed: Dict, ego_history: List[Tuple[float, float]]) -> List[Dict]:
    direct_keys = [
        "ego_trajectory_6s",
        "ego_future_trajectory_6s",
        "ego_trajectory",
        "future_ego_trajectory",
    ]
    for key in direct_keys:
        norm = normalize_trajectory(parsed.get(key))
        if norm is not None:
            return norm

    reasoning_steps = parsed.get("reasoning_steps", [])
    pair_pattern = re.compile(
        r"x\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\s*[,，]\s*y\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    if isinstance(reasoning_steps, list):
        for step in reasoning_steps:
            step_text = ""
            if isinstance(step, dict):
                step_text = " ".join(str(v) for v in step.values())
            elif isinstance(step, str):
                step_text = step
            if "ego" not in step_text.lower():
                continue

            pairs = pair_pattern.findall(step_text)
            if not pairs:
                continue
            raw = [{"x": float(x), "y": float(y)} for x, y in pairs]
            norm = normalize_trajectory(raw)
            if norm is not None:
                return norm

    return fallback_linear_traj(ego_history)


def _points_from_traj_segment(raw_segment: object) -> List[Tuple[float, float]]:
    if not isinstance(raw_segment, list):
        return []
    out: List[Tuple[float, float]] = []
    for pt in raw_segment:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x, y = pt[0], pt[1]
        if x is None or y is None:
            continue
        try:
            out.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    return out


def _collision_check_with_vehicle_dims(
    predicted_ego: List[Dict],
    adv_trajectory: List[Dict],
    scene_text_path: Optional[str],
    adv_row_id: int,
) -> Dict[str, object]:
    dims = load_vehicle_dims_from_trajectory_json(scene_text_path, adv_row_id)
    return trajectory_collision_check(
        predicted_ego,
        adv_trajectory,
        ego_length_m=dims["ego_length"],
        ego_width_m=dims["ego_width"],
        target_length_m=dims["target_length"],
        target_width_m=dims["target_width"],
    )


def _compute_stage_time_indices(
    predicted_ego: List[Dict],
    adv_trajectory: List[Dict],
    scene_text_path: Optional[str],
    adv_row_id: int,
    *,
    impact_time_index: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    dims = load_vehicle_dims_from_trajectory_json(scene_text_path, adv_row_id)
    stage_out = classify_interaction_stages(
        predicted_ego,
        adv_trajectory,
        dt_default=DT_CTRL,
        ego_length=dims["ego_length"],
        ego_width=dims["ego_width"],
        target_length=dims["target_length"],
        target_width=dims["target_width"],
    )
    return sanitize_stage_time_indices_vs_impact(
        _stage_time_indices(stage_out["stage_ids"]),
        impact_time_index,
    )


def _load_ego_global_current(scene_text_path: Optional[str]) -> Optional[Dict]:
    if not scene_text_path:
        return None
    txt_path = Path(scene_text_path)
    if not txt_path.is_file():
        return None
    json_path = txt_path.parent / f"{txt_path.stem}_trajectory.json"
    if not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data.get("ego_current_state")


def _try_load_history_trajectories(scene_text_path: Optional[str]) -> Optional[List[List[Tuple[float, float]]]]:
    if not scene_text_path:
        return None
    txt_path = Path(scene_text_path)
    if not txt_path.is_file():
        return None
    json_path = txt_path.parent / f"{txt_path.stem}_trajectory.json"
    if not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    ht = data.get("history_trajectory_local")
    if not isinstance(ht, list):
        return None
    ft = data.get("predicted_ego_future_trajectory_local")
    ego_global_current = data.get("ego_current_state")
    return [_points_from_traj_segment(seg) for seg in ht], ft, ego_global_current


def _anchor_traj_local_to_global(
    traj: List[Dict],
    ego_global_current: Any,
) -> List[Dict]:
    """
    Map BEV points from ego-current local frame (history_trajectory_local) to global map frame,
    using ego_current_state (x, y, heading rad) from *_trajectory.json. Matches
    datasets.nuscenes_utils.objects2frame(..., toworld=True) for (x, y).
    """
    if not traj or ego_global_current is None or not isinstance(ego_global_current, dict):
        return traj
    try:
        cx = float(ego_global_current["x"])
        cy = float(ego_global_current["y"])
        theta = float(ego_global_current["heading"])
    except (KeyError, TypeError, ValueError):
        return traj
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # (lx, ly) @ get_rot(theta) + (cx, cy), same as objects2frame toworld branch
    out: List[Dict] = []
    for pt in traj:
        if not isinstance(pt, dict):
            out.append(pt)
            continue
        try:
            lx = float(pt["x"])
            ly = float(pt["y"])
        except (KeyError, TypeError, ValueError):
            out.append(dict(pt))
            continue
        gx = lx * cos_t - ly * sin_t + cx
        gy = lx * sin_t + ly * cos_t + cy
        new_pt = dict(pt)
        new_pt["x"] = gx
        new_pt["y"] = gy
        out.append(new_pt)
    return out


def _resolve_ego_and_adv_history(
    scene_text_path: Optional[str],
    scene_text: str,
    adv_vehicle_id: int,
    agent_points: Dict[int, List[Tuple[float, float]]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Any]:
    trajs, predicted_ego_future, ego_global_current = _try_load_history_trajectories(scene_text_path)
    predicted_ego_future = normalize_trajectory(predicted_ego_future)
    if trajs:
        ego_history = trajs[0] if len(trajs) > 0 else []
        adv_history_points: List[Tuple[float, float]] = []
        if 0 <= adv_vehicle_id < len(trajs):
            adv_history_points = trajs[adv_vehicle_id]
        if ego_history or adv_history_points:
            return ego_history, adv_history_points, ego_global_current, predicted_ego_future
    ego_history = _extract_ego_history(scene_text)
    adv_history_points = _resolve_agent_history_points(agent_points, adv_vehicle_id)
    return ego_history, adv_history_points, ego_global_current, predicted_ego_future


def _parse_history_tuple_list(raw_text: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    pair_pattern = re.compile(
        r"\(\s*([-+]?(?:\d+(?:\.\d+)?|nan))\s*,\s*([-+]?(?:\d+(?:\.\d+)?|nan))\s*\)",
        re.IGNORECASE,
    )
    for x_text, y_text in pair_pattern.findall(raw_text):
        if x_text.lower() == "nan" or y_text.lower() == "nan":
            continue
        points.append((float(x_text), float(y_text)))
    return points


def _extract_ego_history(scene_text: str) -> List[Tuple[float, float]]:
    if scene_text:
        ego_block_match = re.search(
            r"EGO:\s*(.*?)(?:\n\s*\n|\nSURROUNDING AGENTS|\Z)",
            scene_text,
            re.IGNORECASE | re.DOTALL,
        )
        ego_block = ego_block_match.group(1) if ego_block_match else scene_text
        hist_match = re.search(
            r"Historical trajectory.*?:\s*(\[[^\]]*\])",
            ego_block,
            re.IGNORECASE | re.DOTALL,
        )
        if hist_match:
            points = _parse_history_tuple_list(hist_match.group(1))
            if points:
                return points

        current_match = re.search(
            r"Current state:\s*\(x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m",
            ego_block,
            re.IGNORECASE,
        )
        if current_match:
            return [(float(current_match.group(1)), float(current_match.group(2)))]

    return []


def _is_control_trajectory(raw_traj: object) -> bool:
    if not isinstance(raw_traj, list) or not raw_traj:
        return False
    first = raw_traj[0]
    if not isinstance(first, dict):
        return False
    has_acc = "acceleration" in first
    has_steer = "steering_angle" in first
    return has_acc and has_steer


def _is_diff_trajectory(raw_traj: object) -> bool:
    if not isinstance(raw_traj, list) or not raw_traj:
        return False
    first = raw_traj[0]
    if not isinstance(first, dict):
        return False
    return "delta_s" in first and "delta_heading" in first


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _diff_to_xy_trajectory(
    diff_steps: List[Dict],
    history_points: List[Tuple[float, float]],
    current_state: Optional[Dict[str, float]] = None,
    default_dt: float = 0.5,
) -> List[Dict]:
    if not diff_steps:
        return fallback_linear_traj(history_points)

    has_current_xy = (
        isinstance(current_state, dict)
        and "x" in current_state
        and "y" in current_state
        and math.isfinite(float(current_state["x"]))
        and math.isfinite(float(current_state["y"]))
    )
    has_current_heading = (
        isinstance(current_state, dict)
        and "heading" in current_state
        and math.isfinite(float(current_state["heading"]))
    )

    if has_current_xy:
        x = float(current_state["x"])
        y = float(current_state["y"])
    elif history_points:
        x, y = float(history_points[-1][0]), float(history_points[-1][1])
    else:
        x, y = 0.0, 0.0

    if has_current_heading:
        heading = float(current_state["heading"])
    elif len(history_points) >= 2:
        dx = float(history_points[-1][0] - history_points[-2][0])
        dy = float(history_points[-1][1] - history_points[-2][1])
        heading = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else 0.0
    else:
        heading = 0.0

    out: List[Dict] = []
    prev_t = 0.0
    for i, step in enumerate(diff_steps):
        if not isinstance(step, dict):
            continue

        raw_t = step.get("t", round((i + 1) * default_dt, 1))
        try:
            target_t = float(raw_t)
        except (TypeError, ValueError):
            target_t = round((i + 1) * default_dt, 1)
        if not math.isfinite(target_t):
            target_t = round((i + 1) * default_dt, 1)
        if target_t <= prev_t:
            target_t = prev_t + default_dt
        prev_t = target_t

        try:
            delta_s = float(step.get("delta_s", 0.0))
        except (TypeError, ValueError):
            delta_s = 0.0
        try:
            delta_heading = float(step.get("delta_heading", 0.0))
        except (TypeError, ValueError):
            delta_heading = 0.0

        # Heuristic: if looks like degrees, convert to radians.
        if abs(delta_heading) > 2 * math.pi + 1e-3:
            delta_heading = math.radians(delta_heading)

        heading_mid = heading + 0.5 * delta_heading

        x += delta_s * math.cos(heading_mid)
        y += delta_s * math.sin(heading_mid)

        heading = _wrap_pi(heading + delta_heading)

        out.append({"t": round(target_t, 1), "x": round(x, 3), "y": round(y, 3)})

    norm = normalize_trajectory(out)
    return norm if norm is not None else fallback_linear_traj(history_points)


def _controls_to_xy_trajectory(
    controls: List[Dict],
    history_points: List[Tuple[float, float]],
    current_state: Optional[Dict[str, float]] = None,
    default_dt: float = 0.5,
    wheelbase: float = 2.8,
) -> List[Dict]:
    if not controls:
        return fallback_linear_traj(history_points)

    has_current_xy = (
        isinstance(current_state, dict)
        and "x" in current_state
        and "y" in current_state
        and math.isfinite(float(current_state["x"]))
        and math.isfinite(float(current_state["y"]))
    )
    has_current_heading = (
        isinstance(current_state, dict)
        and "heading" in current_state
        and math.isfinite(float(current_state["heading"]))
    )

    # Start from adversarial latest observed state when available.
    if has_current_xy:
        x = float(current_state["x"])
        y = float(current_state["y"])
        print("current_state", x, y)
    elif history_points:
        x, y = float(history_points[-1][0]), float(history_points[-1][1])
        print("history_points", x, y)
    else:
        x, y = 0.0, 0.0
        print("history_points is empty", x, y)

    if len(history_points) >= 2:
        dx = float(history_points[-1][0] - history_points[-2][0])
        dy = float(history_points[-1][1] - history_points[-2][1])
        speed = math.hypot(dx, dy) / max(1e-6, default_dt)
        yaw = float(current_state["heading"]) if has_current_heading else math.atan2(dy, dx)
    else:
        speed = float(os.getenv("TRAJ_RAG_INITIAL_SPEED", "8.0"))
        # scene_text convention: heading=0 is +x (ego-forward, vertical-up in BEV).
        yaw = float(current_state["heading"]) if has_current_heading else 0.0

    parsed_steps: List[Dict[str, float]] = []
    for i, step in enumerate(controls):
        if not isinstance(step, dict):
            continue
        raw_t = float(step.get("t", round((i + 1) * default_dt, 1)))
        if not math.isfinite(raw_t):
            raw_t = round((i + 1) * default_dt, 1)
        parsed_steps.append(
            {
                "raw_t": raw_t,
                "acceleration": float(step.get("acceleration", 0.0)),
                "steering_angle": float(step.get("steering_angle", 0.0)),
            }
        )

    if not parsed_steps:
        return fallback_linear_traj(history_points)

    # Rebase controls onto local future horizon so first valid control maps to t=default_dt.
    first_raw_t = parsed_steps[0]["raw_t"]
    t_offset = first_raw_t - default_dt
    max_dt = max(default_dt, float(os.getenv("TRAJ_RAG_MAX_DT", str(default_dt * 1.5))))

    out: List[Dict] = []
    prev_t = 0.0
    for i, step in enumerate(parsed_steps):
        target_t = step["raw_t"] - t_offset
        if not math.isfinite(target_t):
            target_t = (i + 1) * default_dt
        if target_t <= prev_t:
            target_t = prev_t + default_dt
        dt = min(max_dt, max(1e-3, target_t - prev_t))
        cur_t = prev_t + dt
        prev_t = cur_t
        acc = step["acceleration"]
        steer_deg = step["steering_angle"]
        steer_rad = math.radians(steer_deg)
        print("acc", acc)
        print("steer_rad", steer_rad)
        speed = max(0.0, speed + acc * dt)
        print("speed", speed)
        yaw += (speed / max(1e-6, wheelbase)) * math.tan(steer_rad) * dt
        print("yaw", yaw)
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
        print("x", x)
        print("y", y)
        out.append({"t": round(cur_t, 1), "x": round(x, 3), "y": round(y, 3)})

    print("out", out)
    norm = normalize_trajectory(out)
    print("norm out", norm)
    return norm if norm is not None else fallback_linear_traj(history_points)


def _resolve_anchor_xy_trajectory(
    anchor_source: Optional[str],
    anchor_trajectory_6s: object,
    history_points: List[Tuple[float, float]],
    current_state: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    print("control trajectory", anchor_trajectory_6s)
    if _is_control_trajectory(anchor_trajectory_6s):
        print("control trajectory is true")
        return _controls_to_xy_trajectory(
            anchor_trajectory_6s,
            history_points=history_points,
            current_state=current_state,
        )

    if _is_diff_trajectory(anchor_trajectory_6s):
        return _diff_to_xy_trajectory(
            anchor_trajectory_6s,
            history_points=history_points,
            current_state=current_state,
        )

    norm = normalize_trajectory(anchor_trajectory_6s)
    if norm is not None:
        return norm
    return fallback_linear_traj(history_points)


def _resolve_agent_history_points(
    agent_points: Dict[int, List[Tuple[float, float]]],
    adv_vehicle_id: int,
) -> List[Tuple[float, float]]:
    print("resolve_agent_history_points")
    # Primary key is the exact agent id in scene_text.
    if adv_vehicle_id in agent_points:
        return agent_points[adv_vehicle_id]
    # Backward compatibility for old 0-based indexing assumptions.
    if (adv_vehicle_id - 1) in agent_points:
        return agent_points[adv_vehicle_id - 1]
    return []


def _extract_adv_current_state(
    scene_text: str,
    adv_vehicle_id: int,
    history_points: List[Tuple[float, float]],
) -> Optional[Dict[str, float]]:
    if not scene_text:
        return None

    id_pattern = re.compile(
        rf"(?:^\s*Agent\s+{adv_vehicle_id}\s*:|^\s*id:\s*{adv_vehicle_id}\s*,)",
        re.IGNORECASE | re.MULTILINE,
    )
    current_state_pattern = re.compile(
        r"Current state:\s*\(\s*x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m,\s*heading=([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    x_val: Optional[float] = None
    y_val: Optional[float] = None
    heading_val: Optional[float] = None

    id_match = id_pattern.search(scene_text)
    if id_match:
        block_start = id_match.start()
        next_id_match = re.search(
            r"(?:^\s*Agent\s+\d+\s*:|^\s*id:\s*\d+\s*,)",
            scene_text[id_match.end():],
            re.IGNORECASE | re.MULTILINE,
        )
        if next_id_match:
            block_end = id_match.end() + next_id_match.start()
            agent_block = scene_text[block_start:block_end]
        else:
            agent_block = scene_text[block_start:]
        cur_match = current_state_pattern.search(agent_block)
        if cur_match:
            x_val = float(cur_match.group(1))
            y_val = float(cur_match.group(2))
            heading_val = float(cur_match.group(3))

    if (x_val is None or y_val is None) and history_points:
        x_val = float(history_points[-1][0])
        y_val = float(history_points[-1][1])

    if heading_val is None and len(history_points) >= 2:
        dx = float(history_points[-1][0] - history_points[-2][0])
        dy = float(history_points[-1][1] - history_points[-2][1])
        if dx != 0.0 or dy != 0.0:
            heading_val = math.atan2(dy, dx)

    if x_val is None or y_val is None or heading_val is None:
        return None
    return {"x": x_val, "y": y_val, "heading": heading_val}


def _extract_adv_initial_speed_mps(
    scene_text: str,
    adv_vehicle_id: int,
    history_points: List[Tuple[float, float]],
    default_dt: float = 0.5,
) -> Optional[float]:
    if not scene_text:
        return None

    id_pattern = re.compile(
        rf"(?:^\s*Agent\s+{adv_vehicle_id}\s*:|^\s*id:\s*{adv_vehicle_id}\s*,)",
        re.IGNORECASE | re.MULTILINE,
    )
    id_match = id_pattern.search(scene_text)
    if not id_match:
        # Backward compatibility for old 0-based indexing assumptions.
        id_pattern_alt = re.compile(
            rf"(?:^\s*Agent\s+{adv_vehicle_id - 1}\s*:|^\s*id:\s*{adv_vehicle_id - 1}\s*,)",
            re.IGNORECASE | re.MULTILINE,
        )
        id_match = id_pattern_alt.search(scene_text)
    if not id_match:
        id_match = None

    agent_block = scene_text
    if id_match:
        block_start = id_match.start()
        next_id_match = re.search(
            r"(?:^\s*Agent\s+\d+\s*:|^\s*id:\s*\d+\s*,)",
            scene_text[id_match.end() :],
            re.IGNORECASE | re.MULTILINE,
        )
        if next_id_match:
            block_end = id_match.end() + next_id_match.start()
            agent_block = scene_text[block_start:block_end]
        else:
            agent_block = scene_text[block_start:]

    m_speed = re.search(
        r"(?:^|[,\s])speed=([-+]?(?:\d+(?:\.\d+)?|nan))",
        agent_block,
        re.IGNORECASE | re.MULTILINE,
    )
    if m_speed:
        s = m_speed.group(1)
        if s.lower() != "nan":
            try:
                v = float(s)
                if math.isfinite(v) and v >= 0.0:
                    return v
            except (TypeError, ValueError):
                pass

    m_vx = re.search(
        r"(?:^|[,\s])vx=([-+]?(?:\d+(?:\.\d+)?|nan))",
        agent_block,
        re.IGNORECASE | re.MULTILINE,
    )
    m_vy = re.search(
        r"(?:^|[,\s])vy=([-+]?(?:\d+(?:\.\d+)?|nan))",
        agent_block,
        re.IGNORECASE | re.MULTILINE,
    )
    if m_vx and m_vy:
        vx_s = m_vx.group(1)
        vy_s = m_vy.group(1)
        if vx_s.lower() != "nan" and vy_s.lower() != "nan":
            try:
                vx = float(vx_s)
                vy = float(vy_s)
                v = math.hypot(vx, vy)
                if math.isfinite(v):
                    return max(0.0, v)
            except (TypeError, ValueError):
                pass

    if len(history_points) >= 2:
        dx = float(history_points[-1][0]) - float(history_points[-2][0])
        dy = float(history_points[-1][1]) - float(history_points[-2][1])
        v = math.hypot(dx, dy) / max(1e-6, float(default_dt))
        if math.isfinite(v):
            return max(0.0, v)

    return None


def _min_distance_m_from_collision_info(collision_info: object) -> Optional[float]:
    if not isinstance(collision_info, dict):
        return None
    raw_min = collision_info.get("min_distance_m")
    if isinstance(raw_min, (int, float)):
        return float(raw_min)
    return None


def _collision_info_from_reply_payload(payload: Dict[str, Any]) -> Dict[str, object]:
    """Rebuild collision_info dict for visualization from run() reply payload."""
    closest_t_index = payload.get("closest_t_index")
    closest_t: Optional[float] = None
    if isinstance(closest_t_index, int):
        closest_t = round(closest_t_index * 0.5, 1)
    min_dist = payload.get("min_distance_m")
    if not isinstance(min_dist, (int, float)):
        min_dist = None
    return {
        "collision": bool(payload.get("is_collision")),
        "min_distance_m": float(min_dist) if min_dist is not None else None,
        "closest_t": closest_t,
        "closest_t_index": closest_t_index,
    }


class TrajectoryAgent(BaseAgent):
    agent_name = "TrajectoryAgent"

    def __init__(
        self,
        api_base: str = "https://your-api-endpoint/v1",
        api_key: str = "",
        model: str = "",
        use_rag: bool = True,
        rag_retry: bool = False,
        llm_output_trajectory: bool = True,
        optimize_anchor_trajectory: bool = False,
        use_llm: bool = False,
        collision_truncation: bool = True,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model = model or "gpt-4o"
        self.use_rag = use_rag
        self.rag_retry = rag_retry
        self.llm_output_trajectory = llm_output_trajectory
        self.optimize_anchor_trajectory = optimize_anchor_trajectory
        self.use_llm = use_llm
        self.collision_truncation = collision_truncation
        if not self.api_key:
            raise ValueError("LLM api_key is required.")
        self.retry_agent = RetryAgent(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            llm_output_trajectory=self.llm_output_trajectory,
        )

    def _retry_trajectory(
        self,
        ctx: SceneContext,
        predicted_ego: List[Dict],
        adv_trajectory: List[Dict],
        adv_history_points: List[Tuple[float, float]],
        adv_current_state: Optional[Dict[str, float]],
        collision_info: Dict[str, Any],
        trace_id: str,
        adv_vehicle_id: int,
        parsed: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Delegate collision retry to :class:`RetryAgent` and return its payload."""
        retry_reply = self.retry_agent.run(
            AgentMessage(
                sender=self.agent_name,
                receiver=self.retry_agent.agent_name,
                msg_type="request",
                payload={
                    "context": ctx,
                    "predicted_ego": predicted_ego,
                    "adv_trajectory": adv_trajectory,
                    "adv_history_points": adv_history_points,
                    "adv_current_state": adv_current_state,
                    "collision_info": collision_info,
                    "parsed": parsed or {},
                    "adv_row_id": adv_vehicle_id,
                },
                trace_id=trace_id,
            )
        )
        return retry_reply.payload

    def describe(self) -> str:
        return "Generate adversarial anchor trajectory for diffusion guidance."

    def run(self, message: AgentMessage) -> AgentMessage:
        ctx: SceneContext = message.payload["context"]
        debug_vis_enabled = os.getenv("TRAJ_DEBUG_VIS", "0").strip().lower() in {"1", "true", "yes", "y"}

        scene_text = ctx.scene_text
        if scene_text is None and ctx.scene_text_path:
            scene_text = _read_text(Path(ctx.scene_text_path))
            ctx.scene_text = scene_text

        agent_points = extract_agent_points(scene_text or "")
        adv_vehicle_id = ctx.adversarial_vehicle_ids[0] if ctx.adversarial_vehicle_ids else 0
        if adv_vehicle_id not in agent_points and agent_points:
            adv_vehicle_id = sorted(agent_points.keys())[0]
        ego_history, adv_history_points, ego_global_current, predicted_ego = _resolve_ego_and_adv_history(
            ctx.scene_text_path, scene_text or "", adv_vehicle_id, agent_points
        )
        print("agent_points", agent_points)
        print("adv_vehicle_id", adv_vehicle_id)
        print("ego_history", ego_history)
        print("adv_history_points", adv_history_points)
        print("predicted_ego", predicted_ego)
        adv_current_state = _extract_adv_current_state(scene_text or "", adv_vehicle_id, adv_history_points)
        print("adv_current_state", adv_current_state)

        if self.use_rag and ctx.anchor_source == "rag" and ctx.anchor_trajectory_6s:
            adv_future_trajectory = _resolve_anchor_xy_trajectory(
                anchor_source=ctx.anchor_source,
                anchor_trajectory_6s=ctx.anchor_trajectory_6s,
                history_points=adv_history_points,
                current_state=adv_current_state,
            )
            print("adv_future_trajectory", adv_future_trajectory)
            collision_info = _collision_check_with_vehicle_dims(
                predicted_ego, adv_future_trajectory, ctx.scene_text_path, adv_vehicle_id
            )
            if collision_info is None:
                print("collision_info is None, defaulting to no collision.")
            if collision_info.get("closest_t_index") is None:
                print("closest_t_index is None in collision_info, setting to -1.")

            retry_count = 0
            source_tag = "rag"
            if self.rag_retry:
                if not collision_info.get("collision"):
                    print("[TrajectoryAgent] RAG anchor failed collision check, delegating to RetryAgent.")
                    retry_payload = self._retry_trajectory(
                        ctx=ctx,
                        predicted_ego=predicted_ego,
                        adv_trajectory=adv_future_trajectory,
                        adv_history_points=adv_history_points,
                        adv_current_state=adv_current_state,
                        collision_info=collision_info,
                        trace_id=message.trace_id,
                        adv_vehicle_id=adv_vehicle_id,
                    )
                    adv_future_trajectory = retry_payload["adv_trajectory"]
                    collision_info = retry_payload["collision_info"]
                    retry_count = retry_payload.get("retry_count", 0)
                    if retry_count > 0:
                        source_tag = "rag_retry"

            if debug_vis_enabled:
                save_path = visualize_trajectory_validation(
                    sample_id=ctx.sample_id,
                    ego_history=ego_history,
                    adv_vehicle_id=adv_vehicle_id,
                    ego_future_trajectory=predicted_ego,
                    adv_future_trajectory=adv_future_trajectory,
                    scene_text=scene_text or "",
                    collision_info=collision_info,
                    source=source_tag,
                )
                if save_path:
                    print(f"[TrajectoryAgent] step3 validation image saved: {save_path}")
            print(
                f"[TrajectoryAgent] step3 collision check (RAG): {collision_info} | "
                f"retry_count={retry_count}"
            )

            confidence = ctx.retrieval_similarity if ctx.retrieval_similarity is not None else 1.0
            behavior_tag = str((ctx.retrieved_anchor or {}).get("scenario_type", "")).strip() or "rag_anchor"

            return AgentMessage(
                sender=self.agent_name,
                receiver=message.sender,
                msg_type="response",
                payload={
                    "adversarial_vehicle_id": adv_vehicle_id,
                    "anchor_trajectory_6s": adv_future_trajectory,
                    "closest_t_index": collision_info.get("closest_t_index"),
                    "is_collision": collision_info.get("collision"),
                    "anchor_confidence": float(confidence),
                    "behavior_tag": behavior_tag,
                    "min_distance_m": _min_distance_m_from_collision_info(collision_info),
                },
                trace_id=message.trace_id,
            )

        if not self.use_llm:
            return AgentMessage(
                sender=self.agent_name,
                receiver=message.sender,
                msg_type="response",
                payload={
                    "adversarial_vehicle_id": adv_vehicle_id,
                    "anchor_trajectory_6s": [],
                    "closest_t_index": None,
                    "is_collision": None,
                    "anchor_confidence": 0.0,
                    "behavior_tag": "",
                    "min_distance_m": None,
                },
                trace_id=message.trace_id,
            )

        rag_adv_behavior, rag_ref_trajectory = _rag_reference_prompt_fields(ctx)
        raw = call_openai_chat(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            system_prompt=_build_system_prompt(self.llm_output_trajectory),
            user_content=_build_user_prompt(
                ctx,
                ego_history,
                adv_history_points,
                adv_current_state,
                predicted_ego,
                output_xy_trajectory=self.llm_output_trajectory,
                adversarial_behavior=rag_adv_behavior,
                reference_trajectory=rag_ref_trajectory,
            ),
            temperature=0.3,
            timeout=600.0,
        )
        print("system prompt:", _build_system_prompt(self.llm_output_trajectory))
        print("user prompt:", _build_user_prompt(
                ctx,
                ego_history,
                adv_history_points,
                adv_current_state,
                predicted_ego,
                output_xy_trajectory=self.llm_output_trajectory,
                adversarial_behavior=rag_adv_behavior,
                reference_trajectory=rag_ref_trajectory,
            ))
        print("llm response:", raw)
        parsed = extract_json(raw)

        candidate_id = parsed.get("adversarial_vehicle_id")
        try:
            adv_vehicle_id = int(candidate_id)
        except (TypeError, ValueError):
            adv_vehicle_id = ctx.adversarial_vehicle_ids[0] if ctx.adversarial_vehicle_ids else 1

        norm_traj = _resolve_anchor_xy_trajectory(
            anchor_source="llm",
            anchor_trajectory_6s=parsed.get("anchor_trajectory_6s"),
            history_points=adv_history_points,
            current_state=adv_current_state,
        )
        collision_info = _collision_check_with_vehicle_dims(
            predicted_ego, norm_traj, ctx.scene_text_path, adv_vehicle_id
        )
        print("collision_info", collision_info.get("collision"))

        retry_count = 0
        max_retry = RetryAgent._resolve_max_retry()
        if not collision_info.get("collision"):
            print("[TrajectoryAgent] LLM anchor failed collision check, delegating to RetryAgent.")
            retry_payload = self._retry_trajectory(
                ctx=ctx,
                predicted_ego=predicted_ego,
                adv_trajectory=norm_traj,
                adv_history_points=adv_history_points,
                adv_current_state=adv_current_state,
                collision_info=collision_info,
                trace_id=message.trace_id,
                adv_vehicle_id=adv_vehicle_id,
                parsed=parsed,
            )
            norm_traj = retry_payload["adv_trajectory"]
            collision_info = retry_payload["collision_info"]
            retry_count = retry_payload.get("retry_count", 0)
            max_retry = retry_payload.get("max_retry", max_retry)
            retry_parsed = retry_payload.get("parsed")
            if isinstance(retry_parsed, dict) and retry_parsed:
                parsed = retry_parsed

        _update_step3_reasoning(parsed, collision_info, retry_count)

        if debug_vis_enabled:
            save_path = visualize_trajectory_validation(
                sample_id=ctx.sample_id,
                ego_history=ego_history,
                adv_vehicle_id=adv_vehicle_id,
                ego_future_trajectory=predicted_ego,
                adv_future_trajectory=norm_traj,
                scene_text=scene_text or "",
                collision_info=collision_info,
                source="llm_retry" if retry_count > 0 else "llm"
            )
            if save_path:
                print(f"[TrajectoryAgent] step3 validation image saved: {save_path}")
        print(
            f"[TrajectoryAgent] step3 collision check (LLM): {collision_info} | "
            f"retry_count={retry_count}/{max_retry}"
        )

        confidence = parsed.get("anchor_confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))

        behavior_tag = str(parsed.get("behavior_tag", "")).strip()
        if not behavior_tag and ctx.potential_adversarial_behaviors:
            behavior_tag = ctx.potential_adversarial_behaviors[0]
        if not behavior_tag:
            behavior_tag = "unknown"

        # Keep trajectory_6s for compatibility with existing downstream scripts.
        return AgentMessage(
            sender=self.agent_name,
            receiver=message.sender,
            msg_type="response",
            payload={
                "adversarial_vehicle_id": adv_vehicle_id,
                "anchor_trajectory_6s": norm_traj,
                "closest_t_index": collision_info.get("closest_t_index"),
                "is_collision": collision_info.get("collision"),
                "anchor_confidence": confidence,
                "behavior_tag": behavior_tag,
                "min_distance_m": _min_distance_m_from_collision_info(collision_info),
            },
            trace_id=message.trace_id,
        )

    def run_rows(self, rows: List[Dict], max_samples: int = -1) -> List[Dict]:
        use_rows = rows[:max_samples] if max_samples > 0 else rows
        out_rows: List[Dict] = []
        for idx, row in enumerate(use_rows):
            sample_id = str(row.get("sample_id", "")).strip()
            image_path_in_row = row.get("image_path")
            if isinstance(image_path_in_row, str) and image_path_in_row.strip():
                sample_id = Path(image_path_in_row).stem
            if not sample_id:
                sample_id = f"row_{idx}"

            # IMPORTANT: batch jobs should not crash due to a single bad sample.
            try:
                txt_path: Optional[Path] = None
                txt_path_from_vlm = row.get("text_path")
                if txt_path_from_vlm and Path(txt_path_from_vlm).exists():
                    txt_path = Path(txt_path_from_vlm)
                elif isinstance(image_path_in_row, str) and image_path_in_row.strip():
                    image_path = Path(image_path_in_row)
                    candidate = image_path.parent / f"{image_path.stem}.txt"
                    if candidate.exists():
                        txt_path = candidate
                if txt_path is None:
                    raise FileNotFoundError(f"Cannot find scene text for sample_id={sample_id}")

                scene_text = _read_text(txt_path)
                adversarial_vehicle_ids = row.get("adversarial_vehicle_ids", []) or []
                potential_behaviors = row.get("potential_adversarial_behaviors", []) or []
                spatial_relationships = row.get("spatial_relationships_with_ego", []) or []
                retrieval_messages = row.get("retrieval_agent_messages", []) or []
            except Exception as e:
                out_rows.append(
                    {
                        "sample_id": sample_id,
                        "trajectory_agent_messages": [],
                        "error": {
                            "stage": "prepare_sample",
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(limit=50),
                        },
                    }
                )
                continue

            if len(adversarial_vehicle_ids) == 0:
                print(f"[TrajectoryAgent] skip sample_id={sample_id}: adversarial_vehicle_ids is empty")
                out_rows.append({"sample_id": sample_id, "trajectory_agent_messages": []})
                continue

            num_candidates = max(
                len(adversarial_vehicle_ids),
                len(potential_behaviors),
                len(retrieval_messages),
                1,
            )
            out: List[Dict] = []
            for cand_idx in range(num_candidates):
                try:
                    adv_vehicle_id = (
                        adversarial_vehicle_ids[cand_idx]
                        if cand_idx < len(adversarial_vehicle_ids)
                        else (adversarial_vehicle_ids[0] if adversarial_vehicle_ids else None)
                    )
                    behavior = (
                        potential_behaviors[cand_idx]
                        if cand_idx < len(potential_behaviors)
                        else (potential_behaviors[0] if potential_behaviors else "")
                    )
                    spatial_rel = (
                        spatial_relationships[cand_idx] if cand_idx < len(spatial_relationships) else ""
                    )
                    retrieval_payload = (
                        retrieval_messages[cand_idx] if cand_idx < len(retrieval_messages) else {}
                    )
                    if not isinstance(retrieval_payload, dict):
                        retrieval_payload = {}

                    ctx = SceneContext(
                        sample_id=sample_id,
                        bev_image_path=row.get("image_path"),
                        scene_text_path=str(txt_path),
                        scene_text=scene_text,
                        scene_understanding=row.get("scene_understanding"),
                        ego_driving_intention=row.get("ego_driving_intention"),
                        adversarial_vehicle_ids=[adv_vehicle_id] if adv_vehicle_id is not None else [],
                        spatial_relationships_with_ego=[spatial_rel] if spatial_rel else [],
                        potential_adversarial_behaviors=[behavior] if behavior else [],
                        retrieved_anchor=retrieval_payload.get("retrieved_anchor", {}),
                        retrieved_mode=retrieval_payload.get("retrieved_mode", {}),
                        reference_trajectory=retrieval_payload.get("reference_trajectory", {}),
                        retrieval_similarity=retrieval_payload.get("retrieval_similarity"),
                        anchor_source=retrieval_payload.get("anchor_source"),
                        anchor_trajectory_6s=retrieval_payload.get("anchor_trajectory_6s"),
                    )
                    # When LLM is disabled, only the RAG branch in run() can produce a real anchor.
                    # Otherwise run() returns [] and optimize_anchor_trajectory_kinematics would
                    # replace it with fallback_linear_traj(history), falsely filling anchor_trajectory_6s.
                    has_rag_anchor = (
                        self.use_rag
                        and ctx.anchor_source == "rag"
                        and ctx.anchor_trajectory_6s
                    )
                    if not self.use_llm and not has_rag_anchor:
                        continue

                    reply = self.run(
                        AgentMessage(
                            sender="driver",
                            receiver=self.agent_name,
                            msg_type="request",
                            payload={"context": ctx},
                            trace_id=sample_id,
                        )
                    )

                    reply_adv_vehicle_id = reply.payload["adversarial_vehicle_id"]
                    try:
                        reply_adv_vehicle_id_int = int(reply_adv_vehicle_id)
                    except (TypeError, ValueError):
                        reply_adv_vehicle_id_int = adv_vehicle_id if adv_vehicle_id is not None else 0

                    agent_points_for_opt = extract_agent_points(scene_text or "")
                    adv_history_for_opt = _resolve_agent_history_points(
                        agent_points_for_opt, reply_adv_vehicle_id_int
                    )
                    adv_current_state_for_opt = _extract_adv_current_state(
                        scene_text or "", reply_adv_vehicle_id_int, adv_history_for_opt
                    )
                    adv_initial_speed_for_opt = _extract_adv_initial_speed_mps(
                        scene_text or "",
                        reply_adv_vehicle_id_int,
                        adv_history_for_opt,
                    )
                    anchor_traj_before_opt = reply.payload["anchor_trajectory_6s"]
                    optimized_anchor_trajectory_6s = anchor_traj_before_opt
                    ego_global_current = _load_ego_global_current(txt_path)
                    if self.optimize_anchor_trajectory:
                        optimized_anchor_trajectory_6s = optimize_anchor_trajectory_kinematics(
                            anchor_traj_before_opt,
                            current_state=adv_current_state_for_opt,
                            history_points=adv_history_for_opt,
                            initial_speed_mps=adv_initial_speed_for_opt,
                        )

                        debug_vis_enabled = os.getenv("TRAJ_DEBUG_VIS", "0").strip().lower() in {
                            "1",
                            "true",
                            "yes",
                            "y",
                        }
                        if debug_vis_enabled:
                            ego_history, _, _, predicted_ego = _resolve_ego_and_adv_history(
                                str(txt_path) if txt_path is not None else None,
                                scene_text or "",
                                reply_adv_vehicle_id_int,
                                agent_points_for_opt,
                            )
                            collision_info = _collision_check_with_vehicle_dims(
                                predicted_ego,
                                optimized_anchor_trajectory_6s,
                                str(txt_path) if txt_path is not None else None,
                                reply_adv_vehicle_id_int,
                            )
                            save_path = visualize_trajectory_validation(
                                sample_id=ctx.sample_id,
                                ego_history=ego_history,
                                adv_vehicle_id=reply_adv_vehicle_id_int,
                                ego_future_trajectory=predicted_ego,
                                adv_future_trajectory=optimized_anchor_trajectory_6s,
                                adv_future_trajectory_before_opt=anchor_traj_before_opt,
                                scene_text=scene_text or "",
                                collision_info=collision_info,
                                source="opt",
                            )
                            if save_path:
                                print(f"[TrajectoryAgent] optimization visualization saved: {save_path}")

                    if self.collision_truncation and reply.payload.get("is_collision"):
                        optimized_anchor_trajectory_6s = truncate_trajectory_after_collision_to_hard_stop(
                            optimized_anchor_trajectory_6s,
                            reply.payload.get("closest_t_index"),
                        )
                        print("Trajectory truncated after collision for hard stop behavior.")

                        debug_vis_enabled = os.getenv("TRAJ_DEBUG_VIS", "0").strip().lower() in {
                            "1",
                            "true",
                            "yes",
                            "y",
                        }
                        if debug_vis_enabled:
                            ego_history, _, _, predicted_ego = _resolve_ego_and_adv_history(
                                str(txt_path) if txt_path is not None else None,
                                scene_text or "",
                                reply_adv_vehicle_id_int,
                                agent_points_for_opt,
                            )
                            collision_info = _collision_info_from_reply_payload(reply.payload)
                            save_path = visualize_trajectory_validation(
                                sample_id=ctx.sample_id,
                                ego_history=ego_history,
                                adv_vehicle_id=reply_adv_vehicle_id_int,
                                ego_future_trajectory=predicted_ego,
                                adv_future_trajectory=optimized_anchor_trajectory_6s,
                                adv_future_trajectory_before_opt=anchor_traj_before_opt,
                                scene_text=scene_text or "",
                                collision_info=collision_info,
                                source="trunc",
                            )
                            if save_path:
                                print(f"[TrajectoryAgent] truncation visualization saved: {save_path}")

                    anchor_traj_out = _anchor_traj_local_to_global(
                        optimized_anchor_trajectory_6s, ego_global_current
                    )

                    _, _, _, predicted_ego_for_stage = _resolve_ego_and_adv_history(
                        str(txt_path) if txt_path is not None else None,
                        scene_text or "",
                        reply_adv_vehicle_id_int,
                        agent_points_for_opt,
                    )
                    closest_t_index = reply.payload.get("closest_t_index")
                    impact_idx: Optional[int] = None
                    if closest_t_index is not None:
                        try:
                            impact_idx = int(closest_t_index)
                        except (TypeError, ValueError):
                            impact_idx = None
                    stage_time_indices = _compute_stage_time_indices(
                        predicted_ego_for_stage,
                        optimized_anchor_trajectory_6s,
                        str(txt_path) if txt_path is not None else None,
                        reply_adv_vehicle_id_int,
                        impact_time_index=impact_idx,
                    )

                    out.append(
                        {
                            "candidate_index": cand_idx,
                            "adversarial_vehicle_id_input": adv_vehicle_id,
                            "behavior_input": behavior,
                            "spatial_relationship_input": spatial_rel,
                            "anchor_source_input": ctx.anchor_source,
                            "retrieval_similarity_input": ctx.retrieval_similarity,
                            "adversarial_vehicle_id": reply.payload["adversarial_vehicle_id"],
                            "anchor_trajectory_6s": anchor_traj_out,
                            "closest_t_index": reply.payload["closest_t_index"],
                            "is_collision": reply.payload.get("is_collision"),
                            "anchor_confidence": reply.payload["anchor_confidence"],
                            "behavior_tag": reply.payload["behavior_tag"],
                            "min_distance_m": reply.payload["min_distance_m"],
                            "stage_time_indices": stage_time_indices,
                        }
                    )
                except Exception as e:
                    out.append(
                        {
                            "candidate_index": cand_idx,
                            "adversarial_vehicle_id_input": adversarial_vehicle_ids[cand_idx]
                            if cand_idx < len(adversarial_vehicle_ids)
                            else (adversarial_vehicle_ids[0] if adversarial_vehicle_ids else None),
                            "behavior_input": potential_behaviors[cand_idx]
                            if cand_idx < len(potential_behaviors)
                            else (potential_behaviors[0] if potential_behaviors else ""),
                            "spatial_relationship_input": spatial_relationships[cand_idx]
                            if cand_idx < len(spatial_relationships)
                            else "",
                            "error": {
                                "stage": "run_candidate",
                                "type": type(e).__name__,
                                "message": str(e),
                                "traceback": traceback.format_exc(limit=50),
                            },
                        }
                    )
            out_rows.append({"sample_id": sample_id, "trajectory_agent_messages": out})
        return out_rows
