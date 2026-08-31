"""build_nusc_crash.py

Convert generated safety-critical scenarios (the ``scene_*.json`` produced by the adv_gen_* scripts)
back into the original nuScenes v1.0-* JSON format, to build a "nuScenes-Crash" dataset.

Inputs:
  - One or more directories (recursively scanned) containing scenario files named like
    ``scene_XXXX_YYY.json``.
    Example: ``./out/gen/val10/adv_gen_rule_based_*/scenario_results/adv_success``.

Outputs:
  - Standard nuScenes JSON tables under ``<out_dir>/v1.0-<version>/``:
        scene.json, sample.json, sample_annotation.json, ego_pose.json,
        instance.json, log.json, map.json,
        category.json, attribute.json, visibility.json,
        sensor.json, calibrated_sensor.json, sample_data.json
  - Optionally copy/symlink the original nuScenes ``maps/`` directory into the output so that
    APIs like ``NuScenesMap`` can load maps directly.

Each scenario is mapped to an independent "crash scene" (a nuScenes scene) containing
``T_past + T_fut`` keyframes (default: 4 + 12 = 16). Frame spacing is determined by the
scenario field ``dt`` (default: 0.5s, i.e., 2Hz):
  - First ``T_past`` frames: from ``past`` (real nuScenes history).
  - Last ``T_fut`` frames: from ``fut_adv`` (adversarial future).

Notes:
  - Training-time data loaders often use ``flip_singapore=True`` to mirror the y-axis for
    singapore-* maps; this script flips them back to the original nuScenes coordinate system
    by default.
  - Since there is no raw sensor data, ``sample_data.json``, ``calibrated_sensor.json``, and
    ``sensor.json`` are written as empty lists (the standard nuScenes API still loads fine).

Example:
    python src/build_nusc_crash.py \\
        --input ./out/gen/val_filter/adv_gen_rule_based_1778350662_4090node2_0/scenario_results/adv_success \\
        --out ./out/nusc_crash_val \\
        --version trainval-crash \\
        --source_nusc_dir /path/to/nuscenes/trainval \\
        --link_maps symlink
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

# Exactly the same as NUSC_MAP_SIZES in datasets/map_env.py. We copy it here so this script can
# run on machines without the full training environment (torch / nuscenes-devkit, etc.).
NUSC_MAP_SIZES: Dict[str, Tuple[float, float]] = {  # unit: meters, (H, W)
    "singapore-onenorth": (2025.0, 1585.6),
    "singapore-hollandvillage": (2922.9, 2808.3),
    "singapore-queenstown": (3687.1, 3228.6),
    "boston-seaport": (2118.1, 2979.5),
}

SUPPORTED_MAPS = tuple(NUSC_MAP_SIZES.keys())

# The 5-digit numeric index portion in nuScenes scene names.
_SCENE_NAME_TPL = "crash-{idx:05d}-{base}"

# Default nuScenes visibility levels (copied from the official dataset).
DEFAULT_VISIBILITY = [
    {"description": "visibility of whole object is between 0 and 40%",
     "token": "1", "level": "v0-40"},
    {"description": "visibility of whole object is between 40 and 60%",
     "token": "2", "level": "v40-60"},
    {"description": "visibility of whole object is between 60 and 80%",
     "token": "3", "level": "v60-80"},
    {"description": "visibility of whole object is between 80 and 100%",
     "token": "4", "level": "v80-100"},
]

# Reverse mapping consistent with NuScenesDataset's key2cat/cat2vec:
# map sem one-hot ([1,0]=vehicle / [0,1]=pedestrian) back to a nuScenes category name.
_SEM_TO_CATEGORY = {
    (1.0, 0.0): "vehicle.car",
    (0.0, 1.0): "human.pedestrian.adult",
}

# Default heights (meters) per category, only used to fill the 3rd dimension of nuScenes `size` (box height).
_CAT_DEFAULT_HEIGHT = {
    "vehicle.car": 1.6,
    "vehicle.truck": 3.0,
    "vehicle.bus.rigid": 3.5,
    "vehicle.bus.bendy": 3.5,
    "vehicle.bicycle": 1.4,
    "vehicle.motorcycle": 1.4,
    "vehicle.trailer": 3.5,
    "vehicle.construction": 3.0,
    "vehicle.emergency.police": 1.7,
    "vehicle.emergency.ambulance": 2.5,
    "human.pedestrian.adult": 1.7,
    "human.pedestrian.child": 1.3,
    "human.pedestrian.construction_worker": 1.7,
    "human.pedestrian.personal_mobility": 1.5,
    "human.pedestrian.police_officer": 1.7,
    "human.pedestrian.stroller": 1.5,
    "human.pedestrian.wheelchair": 1.5,
}

# Default categories written to category.json (covers all official nuScenes categories).
_DEFAULT_CATEGORIES = [
    ("vehicle.car",
     "Vehicle designed primarily for personal use, e.g. sedans, hatch-backs, wagons, vans, mini-vans, SUVs and jeeps."),
    ("vehicle.truck",
     "Vehicles primarily designed to haul cargo, e.g pick-up trucks, lorries, trucks and semi-tractors."),
    ("vehicle.bus.rigid", "Rigid bus."),
    ("vehicle.bus.bendy", "Bendy bus."),
    ("vehicle.bicycle", "Human or electric powered 2-wheeled vehicle designed to travel at lower speeds either on roads or sidewalks."),
    ("vehicle.motorcycle", "Gasoline or electric powered 2-wheeled vehicle designed to move rapidly on the road."),
    ("vehicle.trailer", "Any vehicle trailer, both for trucks, cars and bikes."),
    ("vehicle.construction", "Vehicles primarily designed for construction. Typically very slow moving or stationary."),
    ("vehicle.emergency.police", "All police vehicles, including police bicycles and motorcycles."),
    ("vehicle.emergency.ambulance", "All types of ambulances."),
    ("human.pedestrian.adult", "Adult subcategory."),
    ("human.pedestrian.child", "Child subcategory."),
    ("human.pedestrian.construction_worker", "Construction worker."),
    ("human.pedestrian.personal_mobility", "Personal mobility device."),
    ("human.pedestrian.police_officer", "Police officer."),
    ("human.pedestrian.stroller", "Stroller."),
    ("human.pedestrian.wheelchair", "Wheelchair."),
    ("animal", "All animals, e.g. cats, rats, dogs, deer, birds."),
    ("movable_object.barrier", "Temporary road barrier placed in the scene."),
    ("movable_object.debris", "Movable debris on the road."),
    ("movable_object.pushable_pullable", "Pushable / pullable object."),
    ("movable_object.trafficcone", "All types of traffic cones."),
    ("static_object.bicycle_rack", "Bicycle racks (typically permanent)."),
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _make_token(*parts: Any) -> str:
    """Generate a stable 32-hex token (md5-based) for reproducibility."""
    seed = "::".join(str(p) for p in parts)
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def _yaw_to_quaternion(yaw: float) -> List[float]:
    """nuScenes uses (w, x, y, z) quaternions; here we only encode yaw."""
    half = 0.5 * yaw
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _hcos_hsin_to_yaw(hcos: float, hsin: float) -> float:
    return math.atan2(hsin, hcos)


def _is_singapore(map_name: str) -> bool:
    return map_name.startswith("singapore-")


def _unflip_traj(traj_xyhh: np.ndarray, map_name: str, flip_singapore: bool) -> np.ndarray:
    """If training used `flip_singapore`, flip singapore y/hsin back to the original coordinates."""
    if not (flip_singapore and _is_singapore(map_name)):
        return traj_xyhh
    if map_name not in NUSC_MAP_SIZES:
        return traj_xyhh
    mh = float(NUSC_MAP_SIZES[map_name][0])
    out = traj_xyhh.copy()
    valid = ~np.isnan(out[..., 0])
    out[..., 1] = np.where(valid, mh - out[..., 1], out[..., 1])
    out[..., 3] = np.where(valid, -out[..., 3], out[..., 3])
    return out


def _stable_scenario_id(path: str, fallback_idx: int) -> str:
    """Build a stable string id from the scenario path and sample id."""
    base = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path)) or "root"
    return f"{parent}::{base}::{fallback_idx:06d}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Scenario -> nuScenes records
# ---------------------------------------------------------------------------

def _category_for_agent(sem_row: Sequence[float]) -> str:
    key = tuple(float(v) for v in sem_row)
    return _SEM_TO_CATEGORY.get(key, "vehicle.car")


def _bbox_size(lw_row: Sequence[float], category_name: str) -> List[float]:
    """nuScenes `size` = [width, length, height]; our `lw` = [length, width]."""
    length = float(lw_row[0])
    width = float(lw_row[1])
    height = _CAT_DEFAULT_HEIGHT.get(category_name, 1.6)
    return [width, length, height]


def _extract_behavior_tag(scenario: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the attacker's ``behavior_tag``.

    Generation pipelines (e.g. ``evaluate_agent.py``) typically write the chosen
    candidate's full payload back into ``trajectory_agent_message`` while keeping
    the full candidate list in ``trajectory_agent_messages``. We prefer the
    singular field; if missing, we look for the message whose ``candidate_index``
    matches ``scenario['candidate_index']``, and finally fall back to the first
    message that has a non-empty ``behavior_tag``.
    """

    def _safe_str(v: Any) -> Optional[str]:
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
        return None

    msg = scenario.get("trajectory_agent_message")
    if isinstance(msg, dict):
        tag = _safe_str(msg.get("behavior_tag"))
        if tag:
            return tag

    msgs = scenario.get("trajectory_agent_messages")
    cand_idx = scenario.get("candidate_index")
    if isinstance(msgs, list):
        if isinstance(cand_idx, int):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                if int(m.get("candidate_index", -1)) == int(cand_idx):
                    tag = _safe_str(m.get("behavior_tag"))
                    if tag:
                        return tag
        for m in msgs:
            if not isinstance(m, dict):
                continue
            tag = _safe_str(m.get("behavior_tag"))
            if tag:
                return tag

    return _safe_str(scenario.get("behavior_tag"))


def _build_scenario_records(
    scenario: Dict[str, Any],
    scenario_idx: int,
    scenario_id: str,
    base_timestamp_us: int,
    flip_singapore: bool,
    map_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Convert a single scenario dict into all nuScenes records for one scene.

    Returning ``None`` means the scenario is invalid (e.g., missing required fields) and is skipped.
    """

    map_name = scenario.get("map")
    if not isinstance(map_name, str) or map_name not in NUSC_MAP_SIZES:
        print(f"[skip] {scenario_id}: unknown / missing map {map_name!r}")
        return None

    if "past" not in scenario or "fut_adv" not in scenario:
        print(f"[skip] {scenario_id}: missing past/fut_adv")
        return None

    past = np.asarray(scenario["past"], dtype=np.float64)        # (N, T_past, 6)
    fut_adv = np.asarray(scenario["fut_adv"], dtype=np.float64)  # (N, T_fut, 4)

    if past.ndim != 3 or fut_adv.ndim != 3 or past.shape[0] != fut_adv.shape[0]:
        print(f"[skip] {scenario_id}: past/fut_adv shape mismatch "
              f"({past.shape} vs {fut_adv.shape})")
        return None

    N = int(scenario.get("N", past.shape[0]))
    T_past = int(past.shape[1])
    T_fut = int(fut_adv.shape[1])
    T_total = T_past + T_fut
    dt = float(scenario.get("dt", 0.5))
    dt_us = int(round(dt * 1e6))

    lw = np.asarray(scenario.get("lw", []), dtype=np.float64)  # (N, 2)
    sem = np.asarray(scenario.get("sem", []), dtype=np.float64)  # (N, 2)
    if lw.shape != (N, 2):
        print(f"[skip] {scenario_id}: lw shape {lw.shape} != ({N},2)")
        return None
    if sem.shape != (N, 2):
        sem = np.tile(np.array([1.0, 0.0]), (N, 1))  # default to vehicle.car

    # Take first 4 dims of past (x, y, hcos, hsin) and concat with fut_adv (4 dims).
    traj_xyhh = np.concatenate([past[:, :, :4], fut_adv[:, :, :4]], axis=1)
    # Undo singapore flips
    traj_xyhh = _unflip_traj(traj_xyhh, map_name, flip_singapore)

    valid = ~np.isnan(traj_xyhh[..., 0]) & ~np.isnan(traj_xyhh[..., 1])  # (N, T_total)

    # We need a continuous ego (agent 0) pose to create a scene.
    if not np.all(valid[0]):
        # Forward-fill ego (NaNs are rare, but keep it robust).
        last_valid = None
        for t in range(T_total):
            if valid[0, t]:
                last_valid = t
                break
        if last_valid is None:
            print(f"[skip] {scenario_id}: ego trajectory all-NaN")
            return None
        for t in range(T_total):
            if not valid[0, t]:
                src = last_valid if t < last_valid else t - 1
                traj_xyhh[0, t] = traj_xyhh[0, src]
                valid[0, t] = True
            else:
                last_valid = t

    sample_id = scenario.get("sample_id") or scenario_id
    safe_sample_id = sample_id.replace("/", "-").replace(":", "-")

    # ---- token generation ----
    log_token = _make_token("log", map_name, scenario_id)
    scene_token = _make_token("scene", scenario_id)
    # If source nuScenes map metadata is available, reuse its token/filename so NuScenesMap and
    # other APIs can load maps directly from this dataset.
    if map_lookup and map_name in map_lookup:
        map_token = map_lookup[map_name]["token"]
        map_filename = map_lookup[map_name].get("filename", f"maps/{map_name}.png")
    else:
        map_token = _make_token("map", map_name)
        map_filename = f"maps/{map_name}.png"
    sample_tokens = [_make_token("sample", scenario_id, t) for t in range(T_total)]
    ego_pose_tokens = [_make_token("ego_pose", scenario_id, t) for t in range(T_total)]

    # ---- log ----
    base_dt = datetime.fromtimestamp(base_timestamp_us / 1e6, tz=timezone.utc)
    log_record = {
        "token": log_token,
        "logfile": f"crash-{scenario_idx:05d}",
        "vehicle": "n0xx",
        "date_captured": base_dt.strftime("%Y-%m-%d"),
        "location": map_name,
    }

    behavior_tag = _extract_behavior_tag(scenario)

    # ---- scene ----
    scene_description = (
        f"Synthesized adversarial crash scenario: planner="
        f"{scenario.get('planner', 'unknown')}, attack_agt="
        f"{scenario.get('attack_agt')}, attack_t={scenario.get('attack_t')}, "
        f"adv_token={scenario.get('adv_token')}, adv_success="
        f"{scenario.get('adv_success', True)}."
    )
    if behavior_tag:
        scene_description = scene_description + f" behavior_tag={behavior_tag}."
    scene_record: Dict[str, Any] = {
        "token": scene_token,
        "log_token": log_token,
        "nbr_samples": T_total,
        "first_sample_token": sample_tokens[0],
        "last_sample_token": sample_tokens[-1],
        "name": _SCENE_NAME_TPL.format(idx=scenario_idx, base=safe_sample_id),
        "description": scene_description,
    }
    # ``behavior_tag`` is unknown to the official nuScenes devkit, so keeping it
    # at the top level of scene.json is safe and lets downstream code (e.g.
    # ``crashsim_planner.py``) cluster results by behavior pattern.
    if behavior_tag:
        scene_record["behavior_tag"] = behavior_tag

    # ---- sample ----
    sample_records: List[Dict[str, Any]] = []
    for t in range(T_total):
        sample_records.append({
            "token": sample_tokens[t],
            "timestamp": int(base_timestamp_us + t * dt_us),
            "prev": sample_tokens[t - 1] if t > 0 else "",
            "next": sample_tokens[t + 1] if t < T_total - 1 else "",
            "scene_token": scene_token,
        })

    # ---- ego_pose ----
    ego_pose_records: List[Dict[str, Any]] = []
    for t in range(T_total):
        x, y, hc, hs = traj_xyhh[0, t]
        yaw = _hcos_hsin_to_yaw(float(hc), float(hs))
        ego_pose_records.append({
            "token": ego_pose_tokens[t],
            "timestamp": int(base_timestamp_us + t * dt_us),
            "rotation": _yaw_to_quaternion(yaw),
            "translation": [float(x), float(y), 0.0],
        })

    # ---- instance / sample_annotation ----
    instance_records: List[Dict[str, Any]] = []
    annotation_records: List[Dict[str, Any]] = []

    attack_agt = scenario.get("attack_agt")
    adv_token_ext = scenario.get("adv_token")

    for n in range(1, N):
        category_name = _category_for_agent(sem[n])

        # In nuScenes semantics, instance.token must be globally unique.
        # We always generate a stable token from (scenario_id, agent_id). If cross-dataset tracing
        # is needed, store the external adv_token as an extra field (ignored by the official devkit).
        inst_token = _make_token("instance", scenario_id, n)
        external_adv_token: Optional[str] = None
        if attack_agt is not None and int(attack_agt) == n and isinstance(adv_token_ext, str):
            external_adv_token = adv_token_ext

        per_agent_anns: List[Tuple[int, str]] = []
        for t in range(T_total):
            if not valid[n, t]:
                continue
            ann_token = _make_token("annotation", scenario_id, n, t)
            per_agent_anns.append((t, ann_token))

        if not per_agent_anns:
            continue

        size_xyz = _bbox_size(lw[n], category_name)
        # Use z_center = height/2 so the box sits on the ground plane.
        z_center = size_xyz[2] / 2.0

        for i, (t, ann_token) in enumerate(per_agent_anns):
            x, y, hc, hs = traj_xyhh[n, t]
            yaw = _hcos_hsin_to_yaw(float(hc), float(hs))
            prev_tok = per_agent_anns[i - 1][1] if i > 0 else ""
            next_tok = per_agent_anns[i + 1][1] if i < len(per_agent_anns) - 1 else ""
            annotation_records.append({
                "token": ann_token,
                "sample_token": sample_tokens[t],
                "instance_token": inst_token,
                "visibility_token": "4",
                "attribute_tokens": [],
                "translation": [float(x), float(y), z_center],
                "size": size_xyz,
                "rotation": _yaw_to_quaternion(yaw),
                "prev": prev_tok,
                "next": next_tok,
                "num_lidar_pts": 0,
                "num_radar_pts": 0,
                **({"external_adv_token": external_adv_token} if external_adv_token else {}),
            })

        instance_rec: Dict[str, Any] = {
            "token": inst_token,
            "category_token": _make_token("category", category_name),
            "nbr_annotations": len(per_agent_anns),
            "first_annotation_token": per_agent_anns[0][1],
            "last_annotation_token": per_agent_anns[-1][1],
        }
        if external_adv_token:
            instance_rec["external_adv_token"] = external_adv_token
            # Attach the attacker's behavior_tag (if any) right next to
            # ``external_adv_token`` so the planner can cluster by behavior.
            if behavior_tag:
                instance_rec["behavior_tag"] = behavior_tag
        instance_records.append(instance_rec)

    # ---- map ----
    map_record = {
        "token": map_token,
        "log_tokens": [log_token],
        "category": "semantic_prior",
        "filename": map_filename,
    }

    return {
        "scene": [scene_record],
        "sample": sample_records,
        "sample_annotation": annotation_records,
        "ego_pose": ego_pose_records,
        "instance": instance_records,
        "log": [log_record],
        "map": [map_record],
    }


# ---------------------------------------------------------------------------
# Static tables (category / attribute / visibility / sensor / calibrated_sensor)
# ---------------------------------------------------------------------------

def _build_static_records() -> Dict[str, List[Dict[str, Any]]]:
    category_records = [
        {
            "token": _make_token("category", name),
            "name": name,
            "description": desc,
        }
        for name, desc in _DEFAULT_CATEGORIES
    ]

    return {
        "category.json": category_records,
        "attribute.json": [],
        "visibility.json": list(DEFAULT_VISIBILITY),
        "sensor.json": [],
        "calibrated_sensor.json": [],
        "sample_data.json": [],
    }


# ---------------------------------------------------------------------------
# Load map metadata from the source nuScenes dataset (to reuse map tokens / filenames)
# ---------------------------------------------------------------------------

def _detect_source_meta_dir(source_nusc_dir: str, hint: Optional[str]) -> Optional[str]:
    if hint:
        cand = os.path.join(source_nusc_dir, hint)
        if os.path.isdir(cand):
            return cand
    # Look for a v1.0-* subdirectory
    for entry in sorted(os.listdir(source_nusc_dir)):
        if entry.startswith("v1.0-") and os.path.isdir(os.path.join(source_nusc_dir, entry)):
            return os.path.join(source_nusc_dir, entry)
    return None


def _load_source_map_lookup(source_nusc_dir: str, source_version_subdir: Optional[str]
                            ) -> Optional[Dict[str, Dict[str, Any]]]:
    """Build a ``location -> {token, filename}`` mapping from source nuScenes metadata.

    In nuScenes, ``map.json`` names map images by token (``maps/<token>.png``). Each map is linked
    to logs via ``log_tokens``, and ``log.json`` provides the ``location`` (e.g., ``singapore-onenorth``).
    """
    meta_dir = _detect_source_meta_dir(source_nusc_dir, source_version_subdir)
    if meta_dir is None:
        return None

    map_path = os.path.join(meta_dir, "map.json")
    log_path = os.path.join(meta_dir, "log.json")
    if not os.path.exists(map_path) or not os.path.exists(log_path):
        print(f"[warn] could not locate map.json/log.json under {meta_dir}; "
              "skip source map lookup")
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    log_to_loc = {l["token"]: l["location"] for l in logs}

    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)
    location_to_record: Dict[str, Dict[str, Any]] = {}
    for m in maps:
        for lt in m.get("log_tokens", []):
            loc = log_to_loc.get(lt)
            if loc is None:
                continue
            location_to_record.setdefault(loc, {
                "token": m["token"],
                "filename": m.get("filename", f"maps/{loc}.png"),
            })
    if not location_to_record:
        return None
    print(f"[info] loaded source map lookup for {len(location_to_record)} maps "
          f"({sorted(location_to_record.keys())})")
    return location_to_record


# ---------------------------------------------------------------------------
# Copy/symlink maps directory
# ---------------------------------------------------------------------------

def _link_or_copy_maps(source_nusc_dir: str, out_dir: str, mode: str) -> None:
    """Expose ``source_nusc_dir/maps`` as ``out_dir/maps``.

    ``mode``:
      - ``"none"``: do nothing
      - ``"symlink"``: create a `maps` symlink (recommended, space-efficient)
      - ``"copy"``: copy the entire `maps` directory
    """
    if mode == "none":
        return
    src_maps = os.path.join(source_nusc_dir, "maps")
    if not os.path.isdir(src_maps):
        print(f"[warn] source maps dir not found: {src_maps}; skip linking")
        return
    dst_maps = os.path.join(out_dir, "maps")
    if os.path.exists(dst_maps) or os.path.islink(dst_maps):
        print(f"[info] maps already exists at {dst_maps}, skip")
        return
    if mode == "symlink":
        os.symlink(os.path.abspath(src_maps), dst_maps)
        print(f"[info] symlinked {src_maps} -> {dst_maps}")
    elif mode == "copy":
        import shutil
        shutil.copytree(src_maps, dst_maps)
        print(f"[info] copied {src_maps} -> {dst_maps}")
    else:
        print(f"[warn] unknown link_maps mode {mode!r}; skip")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _gather_scenario_paths(inputs: Sequence[str], only_adv_success: bool) -> List[str]:
    paths: List[str] = []
    for entry in inputs:
        entry = os.path.abspath(entry)
        if os.path.isdir(entry):
            if only_adv_success:
                # Only collect scenario json under adv_success to avoid pulling in unrelated json files.
                # Support two input styles:
                # - pass the adv_success directory directly
                # - pass an upper-level directory (e.g., ./out/gen/val10/) and recursively search
                #   for **/scenario_results/adv_success/*.json
                adv_dir = os.path.join(entry, "scenario_results", "adv_success")
                if os.path.isdir(adv_dir):
                    paths.extend(sorted(glob(os.path.join(adv_dir, "*.json"))))
                paths.extend(
                    sorted(
                        glob(
                            os.path.join(entry, "**", "scenario_results", "adv_success", "*.json"),
                            recursive=True,
                        )
                    )
                )
            else:
                paths.extend(sorted(glob(os.path.join(entry, "*.json"))))
                paths.extend(sorted(glob(os.path.join(entry, "**", "*.json"), recursive=True)))
        elif os.path.isfile(entry) and entry.endswith(".json"):
            paths.append(entry)
        else:
            print(f"[warn] skip non-json input: {entry}")
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _write_json(path: str, records: List[Dict[str, Any]], pretty: bool) -> None:
    """Write JSON in a nuScenes-like style: with ``indent=0`` each field goes on its own line."""
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(records, f, indent=0, ensure_ascii=False)
        else:
            json.dump(records, f, ensure_ascii=False)
        f.write("\n")


def build_dataset(args: argparse.Namespace) -> None:
    scenario_paths = _gather_scenario_paths(args.input, only_adv_success=args.only_adv_success)
    if not scenario_paths:
        print("[error] no scenarios collected; exiting")
        return
    print(f"[info] collected {len(scenario_paths)} scenarios")

    out_meta_dir = os.path.join(args.out, f"v1.0-{args.version}")
    _ensure_dir(out_meta_dir)

    aggregated: Dict[str, List[Dict[str, Any]]] = {
        "scene": [],
        "sample": [],
        "sample_annotation": [],
        "ego_pose": [],
        "instance": [],
        "log": [],
    }
    map_index: Dict[str, Dict[str, Any]] = {}  # token -> map record (merge log_tokens)

    base_timestamp_us = int(args.base_timestamp_us)
    scene_stride_us = int(args.scene_stride_seconds * 1e6)

    map_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    if args.source_nusc_dir and args.reuse_source_map_tokens:
        map_lookup = _load_source_map_lookup(
            args.source_nusc_dir, args.source_version_subdir
        )

    n_built = 0
    for sidx, sc_path in enumerate(scenario_paths):
        try:
            with open(sc_path, "r", encoding="utf-8") as f:
                scenario = json.load(f)
        except Exception as exc:
            print(f"[skip] failed to read {sc_path}: {exc}")
            continue

        scenario_id = _stable_scenario_id(sc_path, sidx)
        scene_base_ts = base_timestamp_us + sidx * scene_stride_us
        recs = _build_scenario_records(
            scenario=scenario,
            scenario_idx=sidx,
            scenario_id=scenario_id,
            base_timestamp_us=scene_base_ts,
            flip_singapore=args.flip_singapore,
            map_lookup=map_lookup,
        )
        if recs is None:
            continue

        for key in ("scene", "sample", "sample_annotation", "ego_pose", "instance", "log"):
            aggregated[key].extend(recs[key])
        for m in recs["map"]:
            cur = map_index.setdefault(m["token"], {**m, "log_tokens": []})
            for lt in m["log_tokens"]:
                if lt not in cur["log_tokens"]:
                    cur["log_tokens"].append(lt)

        n_built += 1
        if (sidx + 1) % 50 == 0:
            print(f"[info] built {n_built}/{sidx + 1} scenarios so far")

    print(f"[info] total scenes built: {n_built}")
    if n_built == 0:
        print("[error] nothing was built; exiting")
        return

    static = _build_static_records()

    final: Dict[str, List[Dict[str, Any]]] = {
        "scene.json": aggregated["scene"],
        "sample.json": aggregated["sample"],
        "sample_annotation.json": aggregated["sample_annotation"],
        "ego_pose.json": aggregated["ego_pose"],
        "instance.json": aggregated["instance"],
        "log.json": aggregated["log"],
        "map.json": list(map_index.values()),
    }
    final.update(static)

    for fname, records in final.items():
        out_path = os.path.join(out_meta_dir, fname)
        _write_json(out_path, records, pretty=not args.compact_json)
        print(f"[info] wrote {out_path} ({len(records)} records)")

    # Optional: expose the maps directory
    if args.source_nusc_dir:
        _link_or_copy_maps(
            source_nusc_dir=args.source_nusc_dir,
            out_dir=args.out,
            mode=args.link_maps,
        )

    # Simple sanity check: load with nuscenes-devkit (if available)
    if args.verify:
        try:
            from nuscenes.nuscenes import NuScenes
            print("[info] verifying with nuscenes-devkit ...")
            nusc = NuScenes(version=f"v1.0-{args.version}", dataroot=args.out, verbose=False)
            print(f"[ok] nusc loaded: {len(nusc.scene)} scenes, "
                  f"{len(nusc.sample)} samples, {len(nusc.sample_annotation)} annotations")
        except Exception as exc:
            print(f"[warn] verification failed: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert generated safety-critical scenarios back to nuScenes v1.0 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="One or more scenario directories, or a single .json file. Directories are scanned recursively.",
    )
    p.add_argument(
        "--out", "-o", required=True,
        help="Output directory. A v1.0-<version>/ metadata folder will be created under it.",
    )
    p.add_argument(
        "--version", default="trainval-crash",
        help="Output nuScenes version suffix (i.e., v1.0-<version>).",
    )
    p.add_argument(
        "--only_adv_success", action="store_true", default=False,
        help="Only collect **/scenario_results/adv_success/*.json under the input directories.",
    )
    p.add_argument(
        "--flip_singapore", dest="flip_singapore", action="store_true", default=True,
        help="(Enabled by default) Assume scenarios used training-time singapore y-flip and flip them back.",
    )
    p.add_argument(
        "--no_flip_singapore", dest="flip_singapore", action="store_false",
        help="Disable singapore y un-flip; use the coordinates stored in scenarios as-is.",
    )
    p.add_argument(
        "--source_nusc_dir", default=None,
        help="Source nuScenes root directory (contains maps/ and v1.0-* metadata). "
             "If provided, it can be used to reuse map.json token/filename and expose maps/.",
    )
    p.add_argument(
        "--source_version_subdir", default=None,
        help="Source nuScenes metadata subdir name (e.g., v1.0-trainval). If empty, auto-detect the first v1.0-*.",
    )
    p.add_argument(
        "--reuse_source_map_tokens", action="store_true", default=True,
        help="(Enabled by default) Reuse map token/filename from the source nuScenes dataset.",
    )
    p.add_argument(
        "--no_reuse_source_map_tokens", dest="reuse_source_map_tokens", action="store_false",
        help="Disable map token reuse; use locally generated tokens/filenames instead.",
    )
    p.add_argument(
        "--link_maps", choices=["none", "symlink", "copy"], default="symlink",
        help="How to expose source_nusc_dir/maps to the output directory.",
    )
    p.add_argument(
        "--base_timestamp_us", type=int, default=1_700_000_000_000_000,
        help="Timestamp of the first sample (microseconds).",
    )
    p.add_argument(
        "--scene_stride_seconds", type=float, default=120.0,
        help="Extra time offset between scenes (seconds) to avoid timestamp collisions.",
    )
    p.add_argument(
        "--compact_json", action="store_true",
        help="Compact JSON output (disable nuScenes-style per-field newlines).",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="After building, load once with nuscenes-devkit as a sanity check.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
