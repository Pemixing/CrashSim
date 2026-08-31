import base64
import json
import os
from pathlib import Path
import time
import traceback
from typing import Dict, List, Tuple, Optional
import io
import math
import re
import numpy as np
from PIL import Image

from .base_agent import AgentMessage, BaseAgent
from .memory import SceneContext
from .tools.api_tools import call_openai_chat, extract_json
from .tools.traj_tools import extract_agent_points, extract_scene_headings

SYSTEM_PROMPT = (
    "You are an expert analyst for safety-critical autonomous driving scenarios. "
    "Your task is to analyze Bird's-Eye-View (BEV) driving scenes and identify adversarial vehicles "
    "whose potential behavior forces the ego vehicle into a safety-critical state or requires urgent reaction.\n\n"

    "You must reason using visual scene understanding, agent motion, lane topology, and multi-agent interactions, "
    "with a strong focus on physically plausible future conflicts within drivable regions.\n\n"

    "Input:\n"
    "  1)【MOST IMPORTANT】BEV image:\n"
    "     - Map structure includes lanes, lane boundaries, intersections, crosswalks, and drivable areas, and white regions indicating non-drivable areas\n"
    "     - Vehicles are shown as boxes with numeric IDs and heading directions\n"
    "     - **Black arrows on each vehicle indicate heading directions (VERY IMPORTANT, as they show whether the vehicle is moving toward the ego vehicle or away from the ego vehicle)**\n"
    "     - Ego vehicle is shown in GREEN\n"
    "     - Surrounding vehicles are shown in BLUE\n\n"

    "  2) Scene text (ego-centric coordinate system):\n"
    "     - Current states of ego and surrounding agents (position, velocity, heading)\n"
    "     - Spatial relation with ego:\n"
    "         • same-direction front / rear\n"
    "         • opposing front / rear\n"
    "         • crossing front / rear\n"
    "         • left-front / left-rear\n"
    "         • right-front / right-rear\n"
    "     - Lane relation w.r.t ego:\n"
    "         • same lane\n"
    "         • same-direction adjacent lane\n"
    "         • opposing adjacent lane\n"
    "         • non-adjacent lane\n\n"

    "【IMPORTANT】Crash Pattern Priors:\n"
    "The percentages represent the prior probability distribution of each crash pattern.\n"
    "  1. Lead Vehicle Stopped — **27.85%**\n"
    "  2. Lead Vehicle Decelerating — **12.22%**\n"
    "  3. Left Turn Across Path From Opposite Directions at Junction — **11.71%**\n"
    "  4. Vehicle(s) Changing Lanes - Same Direction — **9.65%**\n"
    "  5. Straight Crossing Paths at Junction — **7.53%**\n"
    "  6. Vehicle(s) Turning - Same Direction — **6.33%**\n"
    "  7. Lead Vehicle Moving at Lower Constant Speed — **5.99%**\n"
    "  8. Vehicle(s) - Opposite Direction — **3.97%**\n"
    "  9. Backing Up Into Another Vehicle — **3.73%**\n"
    " 10. Vehicle(s) Drifting - Same Direction — **2.80%**\n"
    " 11. Following Vehicle Making a Maneuver — **2.44%**\n"
    " 12. Evasive Action — **1.98%**\n"
    " 13. Vehicle(s) Parking - Same Direction — **1.37%**\n"
    " 14. Vehicle Turning Right at Junction — **1.00%**\n"
    " 15. Lead Vehicle Accelerating — **0.39%**\n"
    " 16. Other — **1.03%**\n\n"

    "**Crash Pattern Prior Guidelines:**\n"
    "  - Treat the crash pattern distribution as a probabilistic prior over the potential behaviors of candidate adversarial agents (not the ego vehicle).\n"
    "  - When selecting adversarial vehicles, systematically check high-likelihood patterns first if they are geometrically feasible and consistent with the scene understanding.\n"
    "  - Use the prior to guide both candidate generation and ranking, especially when multiple interaction types are possible.\n"

    "Output Requirements:\n"
    "- scene_understanding: must adhere to the Reasoning Steps for internal analysis and Summarize the Reasoning; must cover each surrounding "
    "vehicle's behavior pattern and inferred goal, and the ego vehicle's inferred goal.\n"
    "- adversarial_vehicle_ids: list at most 3 vehicle IDs that pose the most plausible future risks; [] if none.\n"
    "- Order adversarial_vehicle_ids from most likely to least likely (scene evidence first, prior second).\n"
    "- spatial_relationships_with_ego (from scene text): a list of spatial relationships, one for each vehicle ID in adversarial_vehicle_ids.\n"
    "- lane_relationships_with_ego (from scene text): a list of lane relationships, one for each vehicle ID in adversarial_vehicle_ids.\n"
    "- potential_adversarial_behaviors: a list of behavior descriptions, one for each vehicle ID in adversarial_vehicle_ids.\n"
    "- Vehicle IDs must match the scene.\n\n"

    "Always output valid JSON only. No explanations.\n\n"

    "The output must follow this schema exactly:\n"
    "{\n"
    '  "scene_understanding": "string",\n'
    '  "ego_driving_intention": "string",\n'
    '  "adversarial_vehicle_ids": [int],\n'
    '  "spatial_relationships_with_ego": ["string"],\n'
    '  "lane_relationships_with_ego": ["string"],\n'
    '  "potential_adversarial_behaviors": ["string"]\n'
    "}"
)


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _encode_image_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"



def _encode_image_obj_data_url(img_obj) -> str:
    # img_obj: PIL.Image.Image 或 numpy.ndarray(H,W,3/4)
    if img_obj is None:
        raise ValueError("Image obs is None; expected PIL.Image.Image or numpy.ndarray.")
    if isinstance(img_obj, np.ndarray):
        arr = img_obj
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # 如果是 OpenCV 的 BGR，可先 arr = arr[..., ::-1]
        if arr.ndim == 3 and arr.shape[2] == 4:
            pil_img = Image.fromarray(arr, mode="RGBA")
            fmt, mime = "PNG", "image/png"
        else:
            pil_img = Image.fromarray(arr, mode="RGB")
            fmt, mime = "PNG", "image/png"
    elif isinstance(img_obj, Image.Image):
        pil_img = img_obj
        fmt = "JPEG" if pil_img.mode == "RGB" else "PNG"
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    else:
        raise TypeError(f"Unsupported image object type: {type(img_obj)}")
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _encode_image_input_data_url(image_input) -> str:
    """
    Encode image as `data:*;base64,...` for VLM input.

    Rule:
    - image_path (Path/str) -> use `_encode_image_data_url()`
    - image_obs  (PIL/numpy) -> use `_encode_image_obj_data_url()`
    """
    if isinstance(image_input, (Path, str)):
        # print("image_input is Path/str", image_input)
        return _encode_image_data_url(Path(str(image_input)))
    if isinstance(image_input, (Image.Image, np.ndarray)):
        # print("image_input is PIL/numpy")
        return _encode_image_obj_data_url(image_input)
    if image_input is None:
        raise ValueError("Image input is None; expected image_path (Path/str) or image_obs (PIL/numpy).")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _list_sample_dirs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_dir()], key=lambda x: x.name)


def _find_paired_png_txt(sample_dir: Path) -> Tuple[Path, Path]:
    png_files = sorted(sample_dir.glob("*.png"))
    txt_files = sorted(sample_dir.glob("*.txt"))
    if not png_files:
        raise FileNotFoundError(f"No png file found in sample folder: {sample_dir}")
    if not txt_files:
        raise FileNotFoundError(f"No txt file found in sample folder: {sample_dir}")

    txt_by_stem = {p.stem: p for p in txt_files}
    for png in png_files:
        if png.stem in txt_by_stem:
            return png, txt_by_stem[png.stem]
    raise FileNotFoundError(f"No matched png/txt stem pair found in sample folder: {sample_dir}")


def _normalize_vlm_output(v: Dict, sample_id: str) -> Dict:
    scene_understanding = str(v.get("scene_understanding", "")).strip()
    ego_intention = str(v.get("ego_driving_intention", "")).strip()
    adv_ids = v.get("adversarial_vehicle_ids", [])
    spatial_relationships = v.get("spatial_relationships_with_ego", [])
    behaviors = v.get("potential_adversarial_behaviors", [])

    if not isinstance(adv_ids, list):
        adv_ids = [adv_ids]
    clean_ids = []
    for item in adv_ids:
        try:
            clean_ids.append(int(item))
        except (TypeError, ValueError):
            continue

    if not isinstance(spatial_relationships, list):
        spatial_relationships = [spatial_relationships]
    clean_spatial_relationships = [str(x).strip() for x in spatial_relationships if str(x).strip()]

    if not isinstance(behaviors, list):
        behaviors = [behaviors]
    clean_behaviors = [str(x).strip() for x in behaviors if str(x).strip()]

    if not scene_understanding:
        scene_understanding = "No reliable scene understanding from model; fallback summary."
    if not ego_intention:
        ego_intention = "Maintain lane and drive safely with collision avoidance."

    return {
        "sample_id": sample_id,
        "scene_understanding": scene_understanding,
        "ego_driving_intention": ego_intention,
        "adversarial_vehicle_ids": clean_ids,
        "spatial_relationships_with_ego": clean_spatial_relationships,
        "potential_adversarial_behaviors": clean_behaviors,
    }


def _build_reasoning_prompt(scene_text: str) -> str:
    return (
        "You are given:\n"
        "1)【MOST IMPORTANT】A BEV image of the scene.\n"
        "2) A textual scene description:\n"
        f"{scene_text}\n\n"        

        "Reasoning steps:\n"
        "Step 1: Visual Scene Understanding (from BEV image):\n"
        "   - Analyze the road layout with a focus on **drivable regions**: identify lane geometry, road boundaries, "
        "and explicitly distinguish between drivable and non-drivable areas (e.g., medians, sidewalks, barriers).\n"
        "   - Identify the lane-level structure: number of lanes, lane directions, "
        "connectivity at intersections, and lane-change feasibility zones.\n"
        "   - Locate each labeled vehicle by its ID and determine which lane it occupies, "
        "with particular attention to vehicles in the ego lane or adjacent lanes that could potentially interact with the ego vehicle through lane changes, merging, yielding, or sudden braking behaviors.\n"
        "Step 2: Agent Behavior Analysis (integrating BEV image, scene text, and crash pattern priors):\n"
        "   - Carefully extract each agent's spatial relationship with the ego vehicle from the scene text, "
        "with primary attention on vehicles ahead of the ego or converging into its path. "
        "Vehicles that are clearly behind the ego and not exhibiting approaching, accelerating, or lane-changing behavior toward the ego "
        "can be deprioritized or safely ignored.\n"
        "   - Incorporate the lane-level relation of each surrounding agent with respect to the ego vehicle to understand interaction context:\n"
        "       • same lane **front** → mainly involves longitudinal interaction, where risk depends on relative distance and speed (e.g., deceleration or stop behavior)\n"
        "       • same-direction **adjacent lane** → enables lateral interaction such as cut-in or merge into the ego's path\n"
        "       • **opposing adjacent lane** → may introduce conflicts from oncoming or turning vehicles\n"
        "       • **non-adjacent lane** → generally implies weaker interaction unless future maneuvers bring the agent closer to the ego's path\n"
        "   - Incorporate the crash pattern prior probabilities to guide behavior inference:\n"
        "       • Start with high-likelihood patterns: Lead Vehicle Stopped (27.85%), Lead Vehicle Decelerating (12.22%), Left Turn Across Path From Opposite (11.71%), Vehicle Changing Lanes - Same Direction (9.65%), etc.\n"
        "       • Evaluate each pattern's geometric feasibility given current agent heading, speed, lane, and proximity to the ego\n"
        "   - Infer ego's intention and future ~6s goal.\n"
        "   - Infer each agent's behavior pattern and future ~6s goal.\n"
        "Step 3: Safety-Critical Interaction Prediction:\n"
        "   - For each surrounding agent, verify whether its inferred future trajectory intersects "
        "with the ego vehicle's inferred future trajectory within the prediction horizon.\n"
        "   - Mark agents as adversarial candidates if trajectories intersect, merge into the same space, or approach with insufficient safety margin.\n\n "

        "Reasoning Guidelines:\n"
        "- **Never ignore any nearby front vehicle, even if it is currently moving fast or steadily; always account for possible future deceleration, sudden braking, or stopping.**\n"
        "- **Do not ignore stationary or slow-moving vehicles; "
        "they may still pose future interaction risks with the ego vehicle.**\n"
        "- **Only consider future interaction risks: assess whether vehicles' predicted future trajectories "
        "will intersect or conflict with the ego vehicle's future path. "
        "Do not treat past trajectory crossings as current risks.**\n"
        "- **Ensure all predicted future trajectories strictly lie within drivable areas inferred from the BEV image. "
        "Trajectories must respect road topology and cannot pass through non-drivable regions such as sidewalks, buildings, or off-road areas.**\n\n"
    )


def _surrounding_agents_is_none(scene_text: str) -> bool:
    """
    Return True if the text indicates there are no surrounding agents.

    Expected pattern examples:
    - "SURROUNDING AGENTS: none"
    - "SURROUNDING AGENTS: None"
    """
    lines = scene_text.splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("surrounding agents"):
            # tolerate "SURROUNDING AGENTS: none" and variants with extra spaces
            _, sep, tail = line.partition(":")
            if not sep:
                continue
            return tail.strip().lower() in {"none", "null", "n/a", "na", "no", "0", "[]", "{}"}
    return False


def _angle_wrap_pi(rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    x = float(rad)
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def _parse_ego_current_xy(scene_text: str) -> Optional[Tuple[float, float]]:
    ego_block_match = re.search(
        r"EGO:\s*(.*?)(?:\n\s*\n|\nSURROUNDING AGENTS|\Z)",
        scene_text or "",
        re.IGNORECASE | re.DOTALL,
    )
    ego_block = ego_block_match.group(1) if ego_block_match else (scene_text or "")
    m = re.search(
        r"Current state:\s*\(x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m",
        ego_block,
        re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except (TypeError, ValueError):
        return None


def _parse_ego_current_state(scene_text: str) -> Optional[Dict[str, float]]:
    """
    Parse ego current state from scene_text.

    Returns dict with available keys among: x, y, vx, vy, speed, heading.
    """
    ego_block_match = re.search(
        r"EGO:\s*(.*?)(?:\n\s*\n|\nSURROUNDING AGENTS|\Z)",
        scene_text or "",
        re.IGNORECASE | re.DOTALL,
    )
    ego_block = ego_block_match.group(1) if ego_block_match else (scene_text or "")
    m_xy = re.search(
        r"Current state:\s*\(x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m",
        ego_block,
        re.IGNORECASE,
    )
    if not m_xy:
        return None

    out: Dict[str, float] = {}
    try:
        out["x"] = float(m_xy.group(1))
        out["y"] = float(m_xy.group(2))
    except (TypeError, ValueError):
        return None

    def _maybe_float(pat: str) -> Optional[float]:
        mm = re.search(pat, ego_block, re.IGNORECASE)
        if not mm:
            return None
        txt = mm.group(1)
        if str(txt).lower() == "nan":
            return None
        try:
            v = float(txt)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    vx = _maybe_float(r"vx=([-+]?(?:\d+(?:\.\d+)?|nan))")
    vy = _maybe_float(r"vy=([-+]?(?:\d+(?:\.\d+)?|nan))")
    speed = _maybe_float(r"speed=([-+]?(?:\d+(?:\.\d+)?|nan))")
    heading = _maybe_float(r"heading=([-+]?(?:\d+(?:\.\d+)?|nan))")

    if vx is not None:
        out["vx"] = float(vx)
    if vy is not None:
        out["vy"] = float(vy)
    if speed is not None:
        out["speed"] = float(speed)
    if heading is not None:
        out["heading"] = float(heading)

    if ("vx" not in out or "vy" not in out) and ("speed" in out) and ("heading" in out):
        s = float(out["speed"])
        h = float(out["heading"])
        out.setdefault("vx", s * math.cos(h))
        out.setdefault("vy", s * math.sin(h))
    return out


def _parse_agent_current_state(scene_text: str, agent_id: int) -> Optional[Dict[str, float]]:
    """
    Parse one surrounding agent's current state from scene_text.

    Expected patterns (tolerant):
    - "Agent {id}:" block with a "Current state:" line
    - Or "id: {id}," block with a "Current state:" line

    Returns dict with available keys among: x, y, vx, vy, speed, heading.
    """
    # Capture a block for this agent until blank line or next agent header.
    # We keep it tolerant to both "Agent 12:" and "id: 12," styles.
    block_pat = re.compile(
        rf"(?:^\s*Agent\s+{int(agent_id)}\s*:|^\s*id:\s*{int(agent_id)}\s*,)"
        r"(.*?)(?=^\s*Agent\s+\d+\s*:|^\s*id:\s*\d+\s*,|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    m_block = block_pat.search(scene_text or "")
    if not m_block:
        return None
    block = m_block.group(1)

    # Current state line (may include vx/vy/speed/heading, order may vary).
    m_xy = re.search(
        r"Current state:\s*\(x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m",
        block,
        re.IGNORECASE,
    )
    if not m_xy:
        return None

    out: Dict[str, float] = {}
    try:
        out["x"] = float(m_xy.group(1))
        out["y"] = float(m_xy.group(2))
    except (TypeError, ValueError):
        return None

    # Optional fields.
    def _maybe_float(pat: str) -> Optional[float]:
        mm = re.search(pat, block, re.IGNORECASE)
        if not mm:
            return None
        txt = mm.group(1)
        if str(txt).lower() == "nan":
            return None
        try:
            v = float(txt)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    vx = _maybe_float(r"vx=([-+]?(?:\d+(?:\.\d+)?|nan))")
    vy = _maybe_float(r"vy=([-+]?(?:\d+(?:\.\d+)?|nan))")
    speed = _maybe_float(r"speed=([-+]?(?:\d+(?:\.\d+)?|nan))")
    heading = _maybe_float(r"heading=([-+]?(?:\d+(?:\.\d+)?|nan))")

    if vx is not None:
        out["vx"] = float(vx)
    if vy is not None:
        out["vy"] = float(vy)
    if speed is not None:
        out["speed"] = float(speed)
    if heading is not None:
        out["heading"] = float(heading)

    # If vx/vy missing but speed+heading available, synthesize vx/vy in scene frame.
    if ("vx" not in out or "vy" not in out) and ("speed" in out) and ("heading" in out):
        s = float(out["speed"])
        h = float(out["heading"])
        out.setdefault("vx", s * math.cos(h))
        out.setdefault("vy", s * math.sin(h))

    return out


def _filter_adversarial_vehicle_ids_posterior(
    scene_text: str,
    adversarial_vehicle_ids: List[int],
    *,
    txt_parent_dir: Optional[str] = None,
    max_keep: int = 3,
    max_range_m: float = 60.0,
    behind_x_m: float = 1.0,
    max_behind_dist_m: float = 35.0,
    moving_away_rel_speed_mps: float = 1.0,
    horizon_s: float = 6.0,
    reaction_time_s: float = 1.0,
    max_decel_mps2: float = 6.0,
    lane_like_half_width_m: float = 2.0,
    max_lateral_far_m: float = 25.0,
    min_heading_diff_rad: float = math.pi / 4,
    debug: bool = True,
) -> List[int]:
    """
    Multi-rule posterior filter for VLM-picked adversarial IDs.

    This function uses the scene text (current state + historical trajectories) as a
    commonsense constraint to remove agents that are very unlikely to interact with ego.

    Implemented rules (extendable):
    - R0: The ID must exist in the scene text (we can parse current state or history points)
    - R1: Dynamic range gate based on current speed, horizon, and max decel
    - R2: Filter if "behind ego + same lane" (follower / no short-horizon cut-in)
    - R3: Filter if "far behind ego" (rearward distance exceeds max_behind_dist_m)
    - R4: Filter if "behind ego + moving away" (non-same-lane rear traffic leaving)
    - R5: Filter if "lateral far OR same-dir diverging away >=45deg OR opposite-dir radially away >=45deg"
    """
    if not adversarial_vehicle_ids:
        return []

    ego_xy = _parse_ego_current_xy(scene_text)
    if ego_xy is None:
        # If ego pose is missing, fall back conservatively (dedupe + truncate only).
        uniq: List[int] = []
        for vid in adversarial_vehicle_ids:
            if isinstance(vid, int) and vid not in uniq:
                uniq.append(vid)
        return uniq[:max_keep]

    # NOTE: scene_text is already in ego-centric local frame:
    # +x forward along ego heading, +y left (right-handed).
    # Therefore, ego heading in this frame is effectively 0 for relative-angle checks,
    # and we do NOT need to rotate (dx,dy) into an ego-aligned frame.
    _, agent_headings_scene = extract_scene_headings(scene_text or "")
    agent_points = extract_agent_points(scene_text or "")

    ex, ey = float(ego_xy[0]), float(ego_xy[1])
    ego_heading = 0.0

    ego_st = _parse_ego_current_state(scene_text) or {}
    ego_speed = float(ego_st.get("speed", 0.0) or 0.0)
    if (not math.isfinite(ego_speed)) or (ego_speed < 0.0):
        ego_speed = 0.0

    kept: List[int] = []
    seen = set()

    def _dbg(msg: str) -> None:
        if debug:
            tag = f"[posterior_filter sample={txt_parent_dir}]" if txt_parent_dir else "[posterior_filter]"
            print(f"{tag} {msg}")

    # Dynamic interaction distance gate for short horizon.
    # Intuition: within horizon, ego can travel ~ v * (reaction + horizon) and also has a braking distance term.
    # We cap by max_range_m to avoid exploding ranges when parsing is noisy.
    reaction_s_f = max(0.0, float(reaction_time_s))
    amax = float(max_decel_mps2)
    if (not math.isfinite(amax)) or (amax <= 1e-3):
        amax = 6.0
    # Time to decelerate from current speed to 0 under max decel.
    break_s_f = ego_speed / amax
    if (not math.isfinite(break_s_f)) or (break_s_f < 0.0):
        break_s_f = 0.0
    horizon_s_f = max(0.0, float(horizon_s) - reaction_s_f - break_s_f)
    # braking distance (to stop) + reaction distance + horizon travel distance
    ego_brake_dist = (ego_speed * ego_speed) / (2.0 * amax)
    ego_react_dist = ego_speed * reaction_s_f
    ego_horizon_dist = ego_speed * horizon_s_f
    base_dynamic_range = 10.0 + ego_react_dist + ego_horizon_dist + ego_brake_dist
    if not math.isfinite(base_dynamic_range):
        base_dynamic_range = 50.0
    base_dynamic_range = float(np.clip(base_dynamic_range, 15.0, float(max_range_m)))

    for vid in adversarial_vehicle_ids:
        if not isinstance(vid, int):
            _dbg(f"skip non-int vid={vid!r}")
            continue
        if vid in seen:
            _dbg(f"skip duplicate vid={vid}")
            continue
        seen.add(vid)

        # R0: must exist in parsed points or at least parse current state.
        st = _parse_agent_current_state(scene_text, vid)
        hist = agent_points.get(vid, [])
        if st is None and not hist:
            _dbg(f"R0 filtered vid={vid} (no current state and no history)")
            continue

        # Prefer current xy from parsed state, else last history point.
        if st is not None:
            ax, ay = float(st["x"]), float(st["y"])
        else:
            ax, ay = float(hist[-1][0]), float(hist[-1][1])

        dx, dy = ax - ex, ay - ey
        dist = math.hypot(dx, dy)
        if not math.isfinite(dist):
            _dbg(f"filtered vid={vid} (non-finite dist), dx={dx}, dy={dy}")
            continue

        # In ego-local coordinates already.
        lx, ly = dx, dy

        # Compute radial speed away from ego (positive => moving away).
        # Some datasets only provide speed (no vx/vy). We handle that here.
        rel_radial = None
        rvx = None
        rvy = None

        # Heading difference if available.
        agent_h = None
        if st is not None and "heading" in st:
            agent_h = float(st["heading"])
        elif vid in agent_headings_scene:
            agent_h = float(agent_headings_scene[vid])

        # Prefer explicit vx/vy if present.
        if st is not None and ("vx" in st) and ("vy" in st):
            rvx = float(st.get("vx", 0.0))
            rvy = float(st.get("vy", 0.0))

        # If only speed is available, synthesize (vx,vy) using heading (ego-local).
        if (rvx is None or rvy is None) and st is not None and ("speed" in st):
            spd = float(st.get("speed", 0.0))
            if agent_h is not None and math.isfinite(float(agent_h)):
                h = float(agent_h)
                rvx = spd * math.cos(h)
                rvy = spd * math.sin(h)

        if rvx is not None and rvy is not None and dist > 1e-3:
            rel_radial = (dx * float(rvx) + dy * float(rvy)) / dist

        # Agent speed magnitude (fallback 0).
        agent_speed = 0.0
        if st is not None and "speed" in st and math.isfinite(float(st.get("speed", 0.0))):
            agent_speed = max(0.0, float(st.get("speed", 0.0)))
        elif rvx is not None and rvy is not None:
            agent_speed = math.hypot(float(rvx), float(rvy))

        # R1: dynamic range gate (cap by max_range_m).
        # Add what the agent can traverse in horizon as well (conservative).
        dyn_range = base_dynamic_range + agent_speed * horizon_s
        if not math.isfinite(dyn_range):
            dyn_range = float(max_range_m)
        dyn_range = float(np.clip(dyn_range, 30.0, float(max_range_m)))
        if dist > dyn_range:
            _dbg(
                f"R1 filtered vid={vid} dist={dist:.2f}m > dyn_range={dyn_range:.2f}m "
                f"(ego_v={ego_speed:.2f}, agent_v={agent_speed:.2f}, horizon={horizon_s_f:.2f})"
            )
            continue

        is_behind = lx < -float(behind_x_m)
        same_lane = abs(ly) <= float(lane_like_half_width_m)

        # R2: same-lane follower behind ego — unlikely adversarial within horizon.
        if is_behind and same_lane:
            _dbg(
                f"R2 filtered vid={vid} (behind+same_lane), lx={lx:.2f}, ly={ly:.2f}, "
                f"lane_half_w={float(lane_like_half_width_m):.2f}"
            )
            continue

        # R3: far behind ego — rearward gap too large for short-horizon interaction.
        behind_dist_m = max(float(max_behind_dist_m), float(behind_x_m) + 1.0)
        if lx < -behind_dist_m:
            _dbg(
                f"R3 filtered vid={vid} (far_behind), lx={lx:.2f}, "
                f"max_behind_dist={behind_dist_m:.2f}"
            )
            continue

        # R4: behind + moving away (e.g. adjacent-lane rear traffic leaving).
        is_moving_away = (rel_radial is not None) and (rel_radial >= float(moving_away_rel_speed_mps))
        if is_behind and is_moving_away:
            _dbg(
                f"R4 filtered vid={vid} (behind+away), lx={lx:.2f}, ly={ly:.2f}, "
                f"rel_radial={rel_radial:.2f}"
            )
            continue

        # R5: lateral far OR heading-band "moving away" (unlikely to cut in).
        # - same_dir_away_45: heading within ±45° of ego AND velocity diverges >=45° from ego +x
        #   AND radially moving away (filters same-direction traffic peeling off).
        # - opp_dir_away_45: heading within ±45° of reverse AND radially moving away
        #   (filters opposite-lane traffic leaving; does NOT filter slow/stationary oncoming).
        lateral_far = abs(ly) > float(max_lateral_far_m)

        agent_heading_rad = None
        if agent_h is not None and math.isfinite(float(agent_h)):
            agent_heading_rad = float(agent_h)
        elif rvx is not None and rvy is not None:
            rvx_f_h, rvy_f_h = float(rvx), float(rvy)
            if math.hypot(rvx_f_h, rvy_f_h) > 1e-3:
                agent_heading_rad = math.atan2(rvy_f_h, rvx_f_h)

        heading_diff_rad = None
        if agent_heading_rad is not None:
            heading_diff_rad = abs(_angle_wrap_pi(agent_heading_rad - ego_heading))

        band = float(min_heading_diff_rad)
        same_dir = heading_diff_rad is not None and heading_diff_rad <= band
        opp_dir = heading_diff_rad is not None and heading_diff_rad >= (math.pi - band)

        is_radially_away = (
            rel_radial is not None and rel_radial >= float(moving_away_rel_speed_mps)
        )

        vel_angle_rad = None
        if rvx is not None and rvy is not None and math.hypot(float(rvx), float(rvy)) > 1e-3:
            vel_angle_rad = math.atan2(float(rvy), float(rvx))

        same_dir_away_45 = False
        if same_dir and is_radially_away and vel_angle_rad is not None:
            vel_vs_ego = abs(_angle_wrap_pi(vel_angle_rad - ego_heading))
            same_dir_away_45 = vel_vs_ego >= band

        opp_dir_away_45 = bool(opp_dir and is_radially_away)

        if lateral_far or same_dir_away_45 or opp_dir_away_45:
            reasons = []
            if lateral_far:
                reasons.append("lateral_far")
            if same_dir_away_45:
                reasons.append("same_dir_away_45")
            if opp_dir_away_45:
                reasons.append("opp_dir_away_45")
            _dbg(
                f"R5 filtered vid={vid} ({'+'.join(reasons)}), ly={ly:.2f}, "
                f"rel_radial={rel_radial if rel_radial is not None else float('nan'):.2f}, "
                f"heading_diff={math.degrees(heading_diff_rad) if heading_diff_rad is not None else float('nan'):.1f}deg, "
                f"vel_vs_ego={math.degrees(abs(_angle_wrap_pi(vel_angle_rad - ego_heading))) if vel_angle_rad is not None else float('nan'):.1f}deg, "
                f"band={math.degrees(band):.1f}deg, max_lateral_far={float(max_lateral_far_m):.2f}"
            )
            continue

        kept.append(vid)
        if len(kept) >= int(max_keep):
            break

    return kept


def _filter_norm_fields_by_ids(norm: Dict, kept_ids: List[int]) -> Dict:
    """Keep norm JSON schema while aligning per-id lists if lengths match."""
    out = dict(norm)
    old_ids: List[int] = list(out.get("adversarial_vehicle_ids", []) or [])
    out["adversarial_vehicle_ids"] = list(kept_ids)

    def _filter_parallel_list(key: str) -> None:
        lst = out.get(key, None)
        if not isinstance(lst, list):
            return
        if len(lst) != len(old_ids):
            return
        idx = {vid: i for i, vid in enumerate(old_ids)}
        new_lst = []
        for vid in kept_ids:
            if vid in idx and 0 <= idx[vid] < len(lst):
                new_lst.append(lst[idx[vid]])
        out[key] = new_lst

    _filter_parallel_list("spatial_relationships_with_ego")
    _filter_parallel_list("lane_relationships_with_ego")
    _filter_parallel_list("potential_adversarial_behaviors")
    return out


class SceneAgent(BaseAgent):
    agent_name = "SceneAgent"

    def __init__(
        self,
        api_base: str = "https://your-api-endpoint/v1",
        api_key: str = "",
        model: str = "",
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model = model or "gpt-4o"
        if not self.api_key:
            raise ValueError("VLM api_key is required.")

    def describe(self) -> str:
        return "Analyze BEV scene and produce adversarial candidates."

    def run(self, message: AgentMessage) -> AgentMessage:
        ctx: SceneContext = message.payload["context"]
        txt_path = Path(str(ctx.scene_text_path))
        scene_text = ctx.scene_text if ctx.scene_text is not None else _read_text(txt_path)

        user_content = [
            {"type": "image_url", "image_url": {"url": _encode_image_input_data_url(ctx.bev_image_path)}},
            {"type": "text", "text": _build_reasoning_prompt(scene_text)},
        ]
        raw = call_openai_chat(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.0,
            timeout=600.0,
        )
        parsed = extract_json(raw)
        norm = _normalize_vlm_output(parsed, sample_id=ctx.sample_id)

        # Posterior sanity filter for VLM-picked adversarial IDs using scene trajectories/states.
        kept_ids = _filter_adversarial_vehicle_ids_posterior(
            scene_text,
            list(norm.get("adversarial_vehicle_ids", []) or []),
            txt_parent_dir=str(txt_path.parent),
        )
        norm = _filter_norm_fields_by_ids(norm, kept_ids)

        return AgentMessage(
            sender=self.agent_name,
            receiver=message.sender,
            msg_type="response",
            payload=norm,
            trace_id=message.trace_id,
        )

    def run_directory(
        self,
        input_dir: str,
        max_samples: int = -1,
        *,
        start_index: int = 0,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 0,
        initial_rows: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Process a directory of scene samples.

        Parameters
        ----------
        initial_rows : list of dicts, optional
            Rows already processed in a previous run (e.g. loaded from a
            checkpoint by the caller).  These are prepended to every checkpoint
            write so that the on-disk file always represents the *full* progress
            and a subsequent resume will compute the correct start_index.
            The return value of this method still contains only the *new* rows
            produced in this call; the caller is responsible for merging.
        """
        root = Path(input_dir)
        sample_dirs = _list_sample_dirs(root)
        if start_index and int(start_index) > 0:
            sample_dirs = sample_dirs[int(start_index):]
        if max_samples > 0:
            sample_dirs = sample_dirs[:max_samples]
        rows: List[Dict] = []
        _initial_rows: List[Dict] = list(initial_rows) if initial_rows else []

        def _checkpoint_rows(reason: str) -> None:
            """
            Save full progress (initial_rows + new rows) to a jsonl checkpoint.
            Best-effort: never raises to avoid masking the original error.
            """
            try:
                ckpt_path = Path(checkpoint_path) if checkpoint_path else (root / "scene_agent_rows.jsonl")
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
                all_rows = _initial_rows + rows
                with tmp_path.open("w", encoding="utf-8") as f:
                    for r in all_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                tmp_path.replace(ckpt_path)
                print(
                    f"[SceneAgent.run_directory] checkpoint saved: {ckpt_path} "
                    f"({len(all_rows)} rows, {len(_initial_rows)} prev + {len(rows)} new) "
                    f"reason={reason}"
                )
            except Exception as e:
                print(f"[SceneAgent.run_directory] checkpoint FAILED reason={reason}: {e!r}")

        for sample_dir in sample_dirs:
            # Bug fix: wrap file-IO before the VLM call in try-except so that a
            # missing/corrupt sample does not crash the whole run without saving.
            try:
                image_path, txt_path = _find_paired_png_txt(sample_dir)
                sample_id = image_path.stem
                scene_text = _read_text(txt_path)
            except Exception as e:
                err_sample_id = sample_dir.name
                rows.append({
                    "sample_id": err_sample_id,
                    "sample_dir": str(sample_dir),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "ts": int(time.time()),
                })
                _checkpoint_rows(reason=f"file_error@{err_sample_id}")
                print(f"[SceneAgent.run_directory] skipping {sample_dir}: {e!r}")
                continue

            if _surrounding_agents_is_none(scene_text):
                out: Dict = {}
                out["sample_id"] = sample_id
                out["text_path"] = str(txt_path)
                out["image_path"] = str(image_path)
                rows.append(out)
                continue
            ctx = SceneContext(
                sample_id=sample_id,
                bev_image_path=str(image_path),
                scene_text_path=str(txt_path),
                scene_text=scene_text,
            )
            try:
                reply = self.run(
                    AgentMessage(
                        sender="driver",
                        receiver=self.agent_name,
                        msg_type="request",
                        payload={"context": ctx},
                        trace_id=sample_id,
                    )
                )
                out = dict(reply.payload)
                out["text_path"] = str(txt_path)
                out["image_path"] = str(image_path)
                rows.append(out)
            except Exception as e:
                err = {
                    "sample_id": sample_id,
                    "text_path": str(txt_path),
                    "image_path": str(image_path),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "ts": int(time.time()),
                }
                rows.append({
                    "sample_id": sample_id,
                    "text_path": str(txt_path),
                    "image_path": str(image_path),
                    "error": repr(e),
                })
                _checkpoint_rows(reason=f"exception@{sample_id}")
                try:
                    err_path = root / f"scene_agent_error_{sample_id}_{int(time.time())}.json"
                    with err_path.open("w", encoding="utf-8") as f:
                        json.dump(err, f, ensure_ascii=False, indent=2)
                    print(f"[SceneAgent.run_directory] error saved: {err_path}")
                except Exception as e2:
                    print(f"[SceneAgent.run_directory] error save FAILED for {sample_id}: {e2!r}")
                continue

            if checkpoint_every and (len(rows) % int(checkpoint_every) == 0):
                _checkpoint_rows(reason=f"periodic@{len(rows)}")

        # Always persist final state so the checkpoint reflects completion.
        _checkpoint_rows(reason="done")
        return rows
