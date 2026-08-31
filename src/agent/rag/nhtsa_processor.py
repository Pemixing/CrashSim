from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import json
import math
import os
import random


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _interpolate_keyframes(
    keyframes: Sequence[Tuple[float, float, float]],
    t: float,
) -> Tuple[float, float]:
    if not keyframes:
        return 0.0, 0.0
    if t <= keyframes[0][0]:
        return float(keyframes[0][1]), float(keyframes[0][2])
    if t >= keyframes[-1][0]:
        return float(keyframes[-1][1]), float(keyframes[-1][2])

    for i in range(len(keyframes) - 1):
        t0, a0, s0 = keyframes[i]
        t1, a1, s1 = keyframes[i + 1]
        if t0 <= t <= t1:
            span = max(1e-6, t1 - t0)
            ratio = (t - t0) / span
            return (
                float(a0 + (a1 - a0) * ratio),
                float(s0 + (s1 - s0) * ratio),
            )
    return 0.0, 0.0


def _human_control_sequence(
    keyframes: Sequence[Tuple[float, float, float]],
    steps: int = 12,
    dt: float = 0.5,
    max_acc_step: float = 0.9,
    max_steer_step: float = 6.0,
    micro_adjust: float = 0.25,
) -> List[Dict]:
    """
    Build a smooth, human-like control sequence from keyframes.
    Keyframe tuple = (t_sec, acceleration, steering_angle).
    """
    seq: List[Dict] = []
    cur_acc = 0.0
    cur_steer = 0.0
    for i in range(steps):
        t = round((i + 1) * dt, 1)
        target_acc, target_steer = _interpolate_keyframes(keyframes, t)

        # Small steering oscillation simulates natural micro-corrections.
        target_steer += micro_adjust * math.sin(t * 1.3)

        acc_delta = _clamp(target_acc - cur_acc, -max_acc_step, max_acc_step)
        steer_delta = _clamp(target_steer - cur_steer, -max_steer_step, max_steer_step)
        cur_acc += acc_delta
        cur_steer += steer_delta

        seq.append({"t": t, "acceleration": round(cur_acc, 3), "steering_angle": round(cur_steer, 3)})
    return seq


def _control_list_from_event_field(ecs: Any) -> Optional[List[Dict]]:
    """
    Normalize event_control_sequence: legacy list of steps, or dict with
    ego_control_sequence / target_control_sequence (prefers target if present,
    else ego).
    """
    if ecs is None:
        return None
    if isinstance(ecs, dict):
        tgt = ecs.get("target_control_sequence") or []
        ego = ecs.get("ego_control_sequence") or []
        if tgt:
            return [dict(s) for s in tgt]
        if ego:
            return [dict(s) for s in ego]
        return None
    if isinstance(ecs, list):
        if not ecs:
            return None
        return [dict(step) for step in ecs]
    return None


def _build_controls_from_profile(
    profile: Dict,
    steps: int = 12,
    dt: float = 0.5,
) -> List[Dict]:
    raw = profile.get("event_control_sequence")
    if raw is None:
        raw = profile.get("control_sequence")
    seq = _control_list_from_event_field(raw)
    if seq:
        return seq
    keyframes = profile.get("keyframes")
    if keyframes:
        return _human_control_sequence(
            keyframes=keyframes,
            steps=steps,
            dt=dt,
            max_acc_step=float(profile.get("max_acc_step", 0.9)),
            max_steer_step=float(profile.get("max_steer_step", 6.0)),
            micro_adjust=float(profile.get("micro_adjust", 0.25)),
        )
    return _human_control_sequence([(0.0, 0.0, 0.0), (8.0, 0.0, 0.0)])


TRAJECTORY_PROFILES_BY_CODE: Dict[str, Dict] = {
    # P01 – Backing Up Into Another Vehicle: reverse motion with slight steering correction
    "01": {
        "keyframes": [
            (0.0, -0.5, 0.0),
            (1.5, -0.8, 0.0),
            (3.5, -0.6, -4.0),
            (5.5, -0.4, -7.0),
            (7.0, -0.2, -3.0),
            (8.0, 0.0, 0.0),
        ],
        "max_acc_step": 0.45,
        "max_steer_step": 4.0,
        "micro_adjust": 0.16,
    },

    # P02 – Vehicle(s) Turning - Same Direction: coordinated turn, slight propulsion
    "02": {
        "keyframes": [
            (0.0, 0.1, 2.0),
            (1.5, 0.2, 8.0),
            (3.5, 0.1, 14.0),
            (5.5, 0.0, 12.0),
            (7.0, 0.0, 5.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P03 – Vehicle(s) Parking - Same Direction: decelerate into parking slot
    "03": {
        "keyframes": [
            (0.0, -0.2, 0.0),
            (1.5, -0.5, -3.0),
            (3.5, -0.7, -7.0),
            (5.5, -0.4, -10.0),
            (7.0, -0.2, -4.0),
            (8.0, 0.0, 0.0),
        ],
        "max_acc_step": 0.6,
        "max_steer_step": 4.5,
        "micro_adjust": 0.14,
    },

    # P04 – Vehicle(s) Changing Lanes - Same Direction: assertive lane change, mild acceleration
    "04": {
        "keyframes": [
            (0.0, 0.2, 0.0),
            (1.2, 0.5, -5.0),
            (2.8, 0.4, -10.0),
            (4.5, 0.2, -6.0),
            (6.5, 0.1, -1.5),
            (8.0, 0.0, 0.0),
        ]
    },

    # P05 – Vehicle(s) Drifting - Same Direction: low longitudinal change, progressive lateral drift
    "05": {
        "keyframes": [
            (0.0, 0.0, 0.0),
            (2.5, 0.0, 2.5),
            (4.5, 0.0, 5.5),
            (6.0, 0.0, 8.0),
            (8.0, 0.0, 10.0),
        ]
    },

    # P06 – Vehicle(s) Making a Maneuver - Opposite Direction: oncoming vehicle turns across path, committed maneuver
    "06": {
        "keyframes": [
            (0.0, 0.2, 0.0),
            (1.8, 0.4, 5.0),
            (3.5, 0.3, 11.0),
            (5.0, 0.1, 14.0),
            (6.8, 0.0, 7.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P07 – Vehicle(s) Not Making a Maneuver - Opposite Direction: steady oncoming straight
    "07": {
        "keyframes": [
            (0.0, 0.0, 0.0),
            (2.0, 0.1, 0.0),
            (5.0, 0.1, 0.0),
            (6.5, 0.0, 0.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P08 – Following Vehicle Making a Maneuver: follower closes in, then makes evasive/overtaking-style maneuver
    "08": {
        "keyframes": [
            (0.0, 0.4, 0.0),
            (2.0, 0.6, 0.0),
            (3.8, 0.5, -5.0),
            (5.3, 0.3, -10.0),
            (6.8, 0.1, -6.0),
            (8.0, 0.0, -1.0),
        ]
    },

    # P09 – Lead Vehicle Accelerating: straight, progressive acceleration
    "09": {
        "keyframes": [
            (0.0, 0.3, 0.0),
            (2.0, 0.6, 0.0),
            (4.5, 0.7, 0.0),
            (6.5, 0.4, 0.0),
            (8.0, 0.2, 0.0),
        ]
    },

    # P10 – Lead Vehicle Moving at Lower Constant Speed: settle into low-speed cruising
    "10": {
        "keyframes": [
            (0.0, -0.2, 0.0),
            (1.5, -0.4, 0.0),
            (3.5, -0.1, 0.0),
            (5.5, 0.0, 0.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P11 – Lead Vehicle Decelerating: progressive braking straight
    "11": {
        "keyframes": [
            (0.0, -0.2, 0.0),
            (2.0, -0.7, 0.0),
            (4.0, -1.1, 0.0),
            (5.8, -0.8, 0.0),
            (7.2, -0.3, 0.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P12 – Lead Vehicle Stopped: firm braking to full stop
    "12": {
        "keyframes": [
            (0.0, -0.6, 0.0),
            (1.8, -1.2, 0.0),
            (3.5, -1.8, 0.0),
            (5.5, -1.0, 0.0),
            (7.0, -0.2, 0.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P13 – Left Turn Across Path at Junction: committed left crossing, slight positive drive
    "13": {
        "keyframes": [
            (0.0, 0.1, 2.0),
            (1.8, 0.3, 9.0),
            (3.5, 0.2, 17.0),
            (5.0, 0.1, 14.0),
            (6.8, 0.0, 6.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P14 – Vehicle Turning Right at Signalized Junction: controlled right turn
    "14": {
        "keyframes": [
            (0.0, 0.1, -2.0),
            (1.5, 0.2, -8.0),
            (3.2, 0.1, -15.0),
            (5.0, 0.0, -13.0),
            (6.8, 0.0, -5.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P15 – Straight Crossing Paths at Junction: slight hesitation then continue
    "15": {
        "keyframes": [
            (0.0, 0.1, 0.0),
            (2.0, 0.0, 0.0),
            (3.8, -0.3, 0.0),
            (5.2, 0.2, 0.0),
            (6.8, 0.1, 0.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P16 – Evasive Action: brake + steer S-shape avoidance
    "16": {
        "keyframes": [
            (0.0, -0.2, 0.0),
            (1.8, -0.7, -3.0),
            (3.5, -1.1, -10.0),
            (5.0, -0.7, -8.0),
            (6.5, -0.2, -3.0),
            (8.0, 0.0, 0.0),
        ]
    },

    # P17 – Other: mild maneuver → instability → recovery or loss-of-control
    "17": {
        "keyframes": [
            (0.0,  0.1,   0.0),    # slight forward motion (normal driving)
            (1.5,  0.3,   2.0),    # mild maneuver (e.g., turn / lane adjust)
            (3.0, -0.4,  -6.0),    # instability begins (over-correction)
            (4.5, -0.8, -12.0),    # stronger loss of control (skid / sharp steer)
            (6.0, -0.5,  -8.0),    # continued instability (partial recovery attempt)
            (8.0,  0.0,   0.0),    # stabilize or end state
        ]
    }
}


# ---------------------------------------------------------------------------
# Real crash profiles (data-driven replacements for hand-crafted keyframes)
# ---------------------------------------------------------------------------

_CRASH_PROFILES_CACHE: Dict[str, Dict] = {}
_CRASH_PROFILES_LOADED: bool = False

_DEFAULT_CRASH_PROFILES_PATHS = [
    "src/out/nhtsa_rag/crash_profiles.json",
    "out/nhtsa_rag/crash_profiles.json",
]


def load_crash_profiles(path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load real crash trajectory profiles from JSON.
    Profiles are keyed by NHTSA precrash code ("1"-"21").
    Each entry has 'event_control_sequence' (primary) and 'variants' (list of alternatives).
    Legacy JSON may use 'control_sequence' or 'keyframes'.
    """
    global _CRASH_PROFILES_CACHE, _CRASH_PROFILES_LOADED

    if path is None:
        path = os.environ.get("CRASH_PROFILES_PATH", "")

    candidates = [path] if path else []
    candidates.extend(_DEFAULT_CRASH_PROFILES_PATHS)

    for p in candidates:
        if not p:
            continue
        fp = Path(p)
        if fp.is_file():
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _CRASH_PROFILES_CACHE = data.get("profiles", {})
            _CRASH_PROFILES_LOADED = True
            return _CRASH_PROFILES_CACHE

    _CRASH_PROFILES_LOADED = True
    return _CRASH_PROFILES_CACHE


def get_crash_profile(
    precrash_code: str,
    sample_variant: bool = True,
) -> Optional[Dict]:
    """
    Look up a real crash profile by precrash code.
    If sample_variant=True and variants exist, randomly pick one.
    Returns a dict with 'control_sequence' (from event_control_sequence: prefers
    target_control_sequence then ego_control_sequence, or legacy list) or
    legacy 'keyframes', plus optional tuning params.
    """
    global _CRASH_PROFILES_LOADED
    if not _CRASH_PROFILES_LOADED:
        load_crash_profiles()

    code = str(precrash_code).strip()
    entry = _CRASH_PROFILES_CACHE.get(code)
    if entry is None:
        return None

    def _merge_tuning(base: Dict) -> Dict:
        for k in ("max_acc_step", "max_steer_step", "micro_adjust"):
            if k in entry:
                base[k] = entry[k]
        return base

    if sample_variant and entry.get("variants"):
        variant = random.choice(entry["variants"])
        result: Dict = {}
        raw_ecs = variant.get("event_control_sequence")
        if raw_ecs is None:
            raw_ecs = variant.get("control_sequence")
        ecs_list = _control_list_from_event_field(raw_ecs)
        if ecs_list:
            result["control_sequence"] = list(ecs_list)
        elif variant.get("keyframes"):
            result["keyframes"] = [tuple(kf) for kf in variant["keyframes"]]
        _merge_tuning(result)
        return result if ("control_sequence" in result or "keyframes" in result) else None

    result = {}
    raw_ecs = entry.get("event_control_sequence")
    if raw_ecs is None:
        raw_ecs = entry.get("control_sequence")
    ecs_list = _control_list_from_event_field(raw_ecs)
    if ecs_list:
        result["control_sequence"] = list(ecs_list)
    elif entry.get("keyframes"):
        result["keyframes"] = [tuple(kf) for kf in entry["keyframes"]]
    _merge_tuning(result)
    result.setdefault("max_acc_step", entry.get("max_acc_step", 0.9))
    result.setdefault("max_steer_step", entry.get("max_steer_step", 6.0))
    result.setdefault("micro_adjust", entry.get("micro_adjust", 0.25))
    return result if ("control_sequence" in result or "keyframes" in result) else None


def _integrate_controls_to_xy(
    controls: Sequence[Dict],
    dt: float = 0.5,
    initial_speed: float = 8.0,
    wheelbase: float = 2.8,
) -> List[Tuple[float, float]]:
    """
    Convert acceleration + steering controls to 2D trajectory points.
    Simple kinematic bicycle integration with fixed wheelbase.
    """
    x = 0.0
    y = 0.0
    yaw = 0.0
    speed = max(0.0, float(initial_speed))
    points: List[Tuple[float, float]] = [(x, y)]

    for step in controls:
        acc = float(step.get("acceleration", 0.0))
        steer_deg = float(step.get("steering_angle", 0.0))
        steer_rad = math.radians(steer_deg)

        speed = max(0.0, speed + acc * dt)
        yaw += (speed / max(1e-6, wheelbase)) * math.tan(steer_rad) * dt
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
        points.append((x, y))

    return points


def _controls_to_anchor_trajectory(
    controls: Sequence[Dict],
    dt: float = 0.5,
    initial_speed: float = 8.0,
    wheelbase: float = 2.8,
) -> List[Dict]:
    """
    Convert a control sequence [{t, acceleration, steering_angle}, ...] into an
    anchor trajectory [{t, x, y}, ...] in a local frame.
    """
    if not controls:
        return []
    xy = _integrate_controls_to_xy(
        controls=controls,
        dt=dt,
        initial_speed=initial_speed,
        wheelbase=wheelbase,
    )
    out: List[Dict] = []
    for i, (x, y) in enumerate(xy):
        t = float(controls[i - 1].get("t", i * dt)) if i > 0 else 0.0
        out.append({"t": round(t, 3), "x": round(float(x), 4), "y": round(float(y), 4)})
    return out


@dataclass
class NHTSARecord:
    record_id: str
    scenario_type: str
    pre_crash_description: str
    ego_maneuver: str
    adversarial_maneuver: str
    contributing_factors: List[str]
    severity: str
    environment: str
    anchor_trajectory: List[Dict]
    anchor_control: List[Dict]
    precrash_code: str = ""

    def to_rag_text(self) -> str:
        return (
            f"[scenario] {self.scenario_type}\n"
            f"[description] {self.pre_crash_description}\n"
            f"[ego_maneuver] {self.ego_maneuver}\n"
            f"[adversarial_maneuver] {self.adversarial_maneuver}\n"
            f"[environment] {self.environment}"
        )

    def to_metadata(self) -> Dict:
        return {
            "record_id": self.record_id,
            "description": self.pre_crash_description,
            "ego_maneuver": self.ego_maneuver,
            "scenario_type": self.scenario_type,
            "adversarial_maneuver": self.adversarial_maneuver,
            "factors": self.contributing_factors,
            "severity": self.severity,
            "environment": self.environment,
            "anchor_trajectory": self.anchor_trajectory,
            "anchor_control": self.anchor_control,
            "precrash_code": self.precrash_code,
        }


class NHTSAProcessor:
    """NHTSA pre-crash records with trajectory anchors for RAG."""

    BUILTIN_SCENARIOS = [
        {
            "id": "P01",
            "type": "Lead Vehicle Stopped",
            "description": "Vehicle A travels along a lane while approaching slower traffic ahead. Vehicle B, positioned ahead in the same lane, comes to a complete stop due to traffic conditions or intersection control, leaving Vehicle A with insufficient time to brake.",
            "ego_maneuver": "Following in lane while approaching slowed or stopped traffic",
            "adversarial_maneuver": "Lead vehicle comes to a full stop in travel lane",
            "factors": ["rural area", "intersection-related", "inattention", "speeding", "younger driver"],
            "severity": "frequency: very_high (16.41%); economic_cost: very_high (12.84%); functional_years_lost: very_high (8.69%); mais3_plus: medium (0.50%)",
            "env": "urban daylight clear intersection-related 35mph",
            "precrash_code": "1",
        },
        {
            "id": "P02",
            "type": "Lead Vehicle Decelerating",
            "description": "Vehicle A follows behind Vehicle B in the same lane with a decreasing headway. Vehicle B begins to decelerate due to traffic or roadway conditions, and Vehicle A reacts late, creating a rear-end collision risk.",
            "ego_maneuver": "Following in lane with a closing gap",
            "adversarial_maneuver": "Lead vehicle applies progressive braking that triggers rear-end conflict",
            "factors": ["daylight", "adverse weather", "rural area", "intersection-related", "high-speed road", "inattention", "speeding", "younger driver"],
            "severity": "frequency: high (7.20%); economic_cost: high (5.33%); functional_years_lost: medium (3.62%); mais3_plus: low (0.49%)",
            "env": "rural daylight clear non-junction >=55mph",
            "precrash_code": "2",
        },
        {
            "id": "P03",
            "type": "Left Turn Across Path From Opposite Directions at Junction",
            "description": "Vehicle A travels straight through an intersection from the opposite direction. Vehicle B initiates a left turn across the intersection and either misjudges the available gap or fails to yield to oncoming traffic, causing both vehicles to enter intersecting paths and creating a crossing-path conflict.",
            "ego_maneuver": "Oncoming straight-through movement",
            "adversarial_maneuver": "Executes left turn across oncoming path with insufficient gap or without yielding",
            "factors": [
                "intersection",
                "low-speed road",
                "vision obscured",
                "inattention",
                "mixed driver age groups",
                "urban and rural environments"
            ],
            "severity": "frequency: medium (~3.4%); economic_cost: high (~4.5%); functional_years_lost: high (~4.2%); mais3_plus: medium-high (~1.2%)",
            "env": "daylight clear junction (signalized or non-signalized) ~35mph",
            "precrash_code": "3"
        },
        {
            "id": "P04",
            "type": "Vehicle(s) Changing Lanes - Same Direction",
            "description": "Vehicle A maintains its lane and speed in a traffic lane. Vehicle B travels in an adjacent lane and performs a lane change into Vehicle A's lane without sufficient lateral gap, bringing the two vehicles into the same lane space.",
            "ego_maneuver": "Maintaining lane and speed",
            "adversarial_maneuver": "Unsafe lane change/cut-in with insufficient lateral gap",
            "factors": ["non-junction area", "high-speed road", "inattention", "younger driver"],
            "severity": "frequency: high (5.69%); economic_cost: medium (3.54%); functional_years_lost: medium (2.57%); mais3_plus: low (0.42%)",
            "env": "urban daylight clear non-junction >=55mph",
            "precrash_code": "4",
        },
        {
            "id": "P05",
            "type": "Straight Crossing Paths at Junction",
            "description": "Vehicle A approaches a intersection from one direction and proceeds straight. Vehicle B approaches from a perpendicular road and also continues straight without yielding, causing both vehicles to enter the intersection simultaneously.",
            "ego_maneuver": "Straight crossing at a junction",
            "adversarial_maneuver": "Fails to yield at crossing",
            "factors": ["rural area", "low-speed road", "vision obscured", "female", "younger and older drivers"],
            "severity": "frequency: high (4.44%); economic_cost: high (6.08%); functional_years_lost: high (6.29%); mais3_plus: high (1.21%)",
            "env": "urban daylight clear stop-sign intersection 25mph",
            "precrash_code": "5",
        },
        {
            "id": "P06",
            "type": "Vehicle(s) Turning - Same Direction",
            "description": "Vehicle A travels straight in its lane in the same direction as surrounding traffic. Vehicle B, traveling ahead or adjacent in the same direction, initiates a turning maneuver across the lane, causing its path to intersect with Vehicle A.",
            "ego_maneuver": "Proceeding straight in the same direction lane",
            "adversarial_maneuver": "Unsafe same-direction turn-in across ego path",
            "factors": ["clear weather", "dry road", "low-speed road", "younger driver"],
            "severity": "frequency: medium (3.73%); economic_cost: medium (2.34%); functional_years_lost: medium (1.68%); mais3_plus: low (0.44%)",
            "env": "urban daylight clear intersection 35mph",
            "precrash_code": "6",
        },
        {
            "id": "P07",
            "type": "Lead Vehicle Moving at Lower Constant Speed",
            "description": "Vehicle A approaches from behind while traveling in the same lane. Vehicle B ahead maintains an unusually low constant speed in the travel lane, causing the gap between the vehicles to close and creating a rear-end conflict risk.",
            "ego_maneuver": "Following in lane with traffic flow",
            "adversarial_maneuver": "Lead vehicle sustains abnormally low speed in travel lane",
            "factors": ["non-junction location", "high-speed road", "inattention", "speeding", "younger driver"],
            "severity": "frequency: medium (3.53%); economic_cost: medium (3.26%); functional_years_lost: medium (2.81%); mais3_plus: medium (0.71%)",
            "env": "urban daylight clear non-junction >=55mph",
            "precrash_code": "7",
        },
        {
            "id": "P08",
            "type": "Vehicle(s) - Opposite Direction",
            "description": "Two vehicles approach each other from opposite directions. The oncoming vehicle intrudes into the ego vehicle's lane, either due to a maneuver (e.g., overtaking or turning across the centerline) or unintentional encroachment, leading to a potential head-on conflict.",
            "ego_maneuver": "Going straight in own lane",
            "adversarial_maneuver": "Oncoming vehicle intrudes into ego lane",
            "factors": [
                "dark",
                "adverse weather",
                "wet/slippery road",
                "non-level road",
                "rural area",
                "non-junction",
                "high-speed road",
                "alcohol",
                "vision obscured",
                "inattention",
                "speeding",
                "male and young driver"
            ],
            "severity": "frequency: low-to-medium (~2.34%); economic_cost: medium-to-high; functional_years_lost: medium-to-high; mais3_plus: very_high",
            "env": "rural daylight clear non-junction >=55mph",
            "precrash_code": "8"
        },
        {
            "id": "P09",
            "type": "Backing Up Into Another Vehicle",
            "description": "Vehicle A moves along a roadway or passes behind a driveway or parking area. Vehicle B begins reversing from a driveway, parking space, or alley without sufficient rearward observation and backs into the path of Vehicle A.",
            "ego_maneuver": "Stopped or slowly passing behind a reversing vehicle",
            "adversarial_maneuver": "Reverses into traffic without adequate rearward observation",
            "factors": ["daylight", "driveway/alley or intersection-related", "low-speed road", "vision obscured", "inattention", "younger driver"],
            "severity": "frequency: medium (2.20%); economic_cost: low (0.79%); functional_years_lost: low (0.32%); mais3_plus: low (0.13%)",
            "env": "urban daylight clear driveway/alley 25mph",
            "precrash_code": "9",
        },
        {
            "id": "P10",
            "type": "Vehicle(s) Drifting - Same Direction",
            "description": "Vehicle A travels steadily in its lane. Vehicle B, traveling in a neighboring lane in the same direction, gradually drifts laterally out of its lane due to distraction or loss of control, encroaching into Vehicle A's path.",
            "ego_maneuver": "Maintaining lane and speed",
            "adversarial_maneuver": "Drifts out of lane into ego trajectory",
            "factors": ["high-speed road", "speeding", "younger driver"],
            "severity": "frequency: medium (1.65%); economic_cost: medium (1.15%); functional_years_lost: medium (1.32%); mais3_plus: medium (0.58%)",
            "env": "urban daylight clear non-junction >=55mph",
            "precrash_code": "10",
        },
        {
            "id": "P11",
            "type": "Following Vehicle Making a Maneuver",
            "description": "Vehicle A travels ahead in a lane at a steady speed. Vehicle B follows behind and approaches with a closing gap, then performs a sudden maneuver such as a late lane change or evasive movement that brings it into conflict with Vehicle A.",
            "ego_maneuver": "Lead vehicle cruising steadily in lane",
            "adversarial_maneuver": "Following vehicle executes a late evasive lane maneuver into conflict",
            "factors": ["intersection-related location", "inattention", "speeding", "younger driver"],
            "severity": "frequency: medium (1.44%); economic_cost: medium (1.01%); functional_years_lost: low (0.67%); mais3_plus: medium (0.50%)",
            "env": "urban daylight clear non-junction 55mph",
            "precrash_code": "11",
        },
        {
            "id": "P12",
            "type": "Evasive Action",
            "description": "Vehicle A travels steadily along its lane. Vehicle B, either following a prior maneuver that creates a developing conflict or without any prior indication, suddenly performs an abrupt evasive action such as hard braking or sharp steering. This late or unexpected maneuver causes Vehicle B to intrude into Vehicle A's path, resulting in a trajectory intersection and conflict.",
            "ego_maneuver": "Lane keeping or steady cruising during normal approach",
            "adversarial_maneuver": "Performs sudden or late evasive braking and/or steering that intrudes into ego path",
            "factors": [
                "urban area",
                "intersection or driveway-related location",
                "low-speed road",
                "low visibility (e.g., dark)",
                "inattention or delayed reaction",
                "younger drivers"
            ],
            "env": "urban daylight or dark clear roadway (intersection or non-junction) ~35mph",
            "precrash_code": "12"
        },
        {
            "id": "P13",
            "type": "Vehicle(s) Parking - Same Direction",
            "description": "Vehicle A travels along a roadway near the curbside traffic flow. Vehicle B, traveling in the same direction, abruptly slows and begins a curbside parking maneuver without adequate surrounding checks, creating a conflict with Vehicle A approaching from behind.",
            "ego_maneuver": "Lane keeping near curbside traffic flow",
            "adversarial_maneuver": "Abrupt deceleration and park-in maneuver with insufficient clearance",
            "factors": ["adverse weather", "non-junction area", "low-speed road", "inattention", "younger driver"],
            "severity": "frequency: low (0.81%); economic_cost: low (0.52%); functional_years_lost: low (0.41%); mais3_plus: low (0.45%)",
            "env": "urban daylight clear non-junction 25mph",
            "precrash_code": "13",
        },
        {
            "id": "P14",
            "type": "Vehicle Turning Right at Junction",
            "description": "Vehicle A proceeds straight through a intersection. Vehicle B approaches the intersection and performs a right turn into or across Vehicle A's path without sufficient speed reduction or clearance.",
            "ego_maneuver": "Proceeding straight through the junction",
            "adversarial_maneuver": "Unsafe right-turn-in with insufficient speed reduction and clearance",
            "factors": ["adverse weather", "intersection/intersection-related", "low-speed road", "vision obscured", "younger and older drivers"],
            "severity": "frequency: low (0.59%); economic_cost: low (0.30%); functional_years_lost: low (0.15%); mais3_plus: low (0.27%)",
            "env": "urban daylight clear signalized intersection 35mph",
            "precrash_code": "14",
        },
        {
            "id": "P15",
            "type": "Lead Vehicle Accelerating",
            "description": "Vehicle A follows Vehicle B in the same lane under normal traffic flow. Vehicle B suddenly accelerates and changes the relative spacing dynamics, causing unstable speed adaptation and creating a potential interaction conflict.",
            "ego_maneuver": "Following in lane with nominal headway",
            "adversarial_maneuver": "Lead vehicle accelerates abruptly and destabilizes closing gap",
            "factors": ["dry road", "intersection-related location", "high-speed road", "traffic signal", "inattention", "speeding", "female and younger driver"],
            "severity": "frequency: low (0.32%); economic_cost: low (0.23%); functional_years_lost: low (0.15%); mais3_plus: medium (0.55%)",
            "env": "urban daylight clear intersection-related 45mph",
            "precrash_code": "15",
        },
        {
            "id": "P16",
            "type": "Other",
            "description": "Vehicle A is involved in a pre-crash event that does not involve a direct interaction with another motor vehicle. This includes collisions with vulnerable road users such as pedestrians or cyclists, as well as single-vehicle loss-of-control events such as roadway departure, rollover, or improper maneuvers (e.g., U-turn leading to crash). These scenarios typically arise from misjudgment, loss of control, or failure to detect non-vehicle agents or road boundaries.",
            "ego_maneuver": "Normal driving, turning (e.g., U-turn), or maneuvering with possible loss of control or perception failure",
            "adversarial_maneuver": "None (single-vehicle) or non-motorized agent behavior (e.g., pedestrian crossing, cyclist entering roadway)",
            "factors": [
                "pedestrian or cyclist presence",
                "loss of control",
                "roadway departure",
                "rollover",
                "visibility limitation",
                "inattention",
                "improper maneuver (e.g., unsafe U-turn)",
                "urban or rural mixed environment"
            ],
            "env": "urban or rural daylight or dark, roadway or roadside, variable speed",
            "precrash_code": "16"
        }
    ]

    def load_crash_samples(
        self,
        crash_profiles_path: Optional[str] = None,
    ) -> List[NHTSARecord]:
        """
        Load real crash samples from crash_profiles.json and convert to NHTSARecords.
        Each variant becomes a separate record for RAG retrieval diversity.
        """
        profiles = load_crash_profiles(crash_profiles_path)
        if not profiles:
            return []

        scenario_map = {
            item["precrash_code"]: item for item in self.BUILTIN_SCENARIOS
        }

        records: List[NHTSARecord] = []
        for code, entry in profiles.items():
            builtin = scenario_map.get(code, {})
            scenario_type = entry.get("nhtsa_type", builtin.get("type", "Unknown"))
            ego_maneuver = builtin.get("ego_maneuver", "Unknown")
            adversarial_maneuver = builtin.get("adversarial_maneuver", "Unknown")
            factors = list(builtin.get("factors", []))
            severity = builtin.get("severity", "unknown")
            environment = builtin.get("env", "urban daylight clear")

            for i, variant in enumerate(entry.get("variants", [])):
                desc = variant.get("description", "")
                eid = variant.get("source_event_id", i)
                tid = variant.get("target_id", 0)
                record_id = f"P{code.zfill(2)}_real_{eid}_t{tid}_{i}"
                records.append(
                    NHTSARecord(
                        record_id=record_id,
                        scenario_type=scenario_type,
                        pre_crash_description=desc,
                        ego_maneuver=ego_maneuver,
                        adversarial_maneuver=adversarial_maneuver,
                        contributing_factors=factors,
                        severity=severity,
                        environment=environment,
                        anchor_trajectory=[],
                        anchor_control=[],
                        precrash_code=str(code).strip(),
                    )
                )
        return records
