import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .base_agent import AgentMessage, BaseAgent
from .memory import SceneContext
from .rag.crash_data_processor import (
    DT_CTRL,
    EventTimeSeries,
    build_event_feature,
    compute_kinematics_from_trajectory,
)
from .rag.retriever import NHTSARetriever
from .rag.vectorstore import (
    _INTERACTION_TAG_PATTERNS,
    _METADATA_TAG_PATTERNS,
    build_control_vector,
    build_event_feature_vector,
    build_trajectory_vector,
)


def _extract_tags(text: str, patterns: dict) -> list:
    s = str(text or "").lower()
    tags = []
    for tag, keys in patterns.items():
        if any(k in s for k in keys):
            tags.append(tag)
    return list(dict.fromkeys(tags))


def _infer_environment(text: str) -> str:
    """
    Infer a soft environment hint string from scene text.
    Returns space-separated environment tags (can be multiple, or empty).
    """
    s = (text or "").lower()
    tags = []

    def _has_any(keywords):
        return any(k in s for k in keywords)

    # Location / road context
    if _has_any(("urban", "city", "downtown")):
        tags.append("urban")
    if _has_any(("rural", "country road")):
        tags.append("rural")
    if _has_any(("highway", "freeway", "expressway", "motorway", "merge")):
        tags.append("high_speed")
    if _has_any(("residential", "local road", "low speed", "school zone")):
        tags.append("low_speed")

    # Junction context
    has_non_signalized = _has_any(("non-signalized", "unsignalized", "stop sign"))
    has_signalized = _has_any(("signalized", "traffic light", "red light", "green light"))
    if has_non_signalized:
        tags.append("non_signalized")
    # Avoid tagging both due to substring overlap: "non-signalized" contains "signalized".
    if has_signalized and not has_non_signalized:
        tags.append("signalized")
    has_non_junction = _has_any(("non-junction", "midblock", "mid-block"))
    if has_non_junction:
        tags.append("non_junction")
    # "non-junction" should not be treated as "intersection".
    if _has_any(("intersection", "crossing", "cross traffic")) and not has_non_junction:
        tags.append("intersection")

    # Lighting / weather
    if _has_any(("daylight", "daytime", "day light")):
        tags.append("daylight")
    if _has_any(("dark", "night", "nighttime")):
        tags.append("night")
    if _has_any(("rain", "snow", "fog", "wet", "slippery", "adverse weather")):
        tags.append("adverse")
    elif _has_any(("clear weather", "clear sky", "dry road")):
        tags.append("clear")

    # Keep order while deduplicating.
    dedup = list(dict.fromkeys(tags))
    return " ".join(dedup)


class RetrievalAgent(BaseAgent):
    agent_name = "RetrievalAgent"
    _ALLOWED_FEATURE_MODES = ("trajectory", "control", "diff")

    def __init__(
        self,
        mode_db_path: str = "./out/nhtsa_rag/nhtsa_mode_vector_db.json",
        variant_db_path: str = "./out/nhtsa_rag/nhtsa_variant_vector_db.json",
        sample_db_path: str = "./out/nhtsa_rag/nhtsa_sample_vector_db.json",
        similarity_threshold: float = 0.5,
        vector_mode: str = "diff",
    ) -> None:
        self.retriever = NHTSARetriever(
            mode_db_path=mode_db_path,
            variant_db_path=variant_db_path,
            sample_db_path=sample_db_path,
        )
        self.similarity_threshold = similarity_threshold
        mode = str(vector_mode or "diff").strip().lower()
        if mode not in self._ALLOWED_FEATURE_MODES:
            raise ValueError(
                f"Unsupported vector_mode '{vector_mode}'. Allowed: {', '.join(self._ALLOWED_FEATURE_MODES)}"
            )
        self.vector_mode = mode

    def describe(self) -> str:
        return "Retrieve anchor trajectory from NHTSA RAG with similarity threshold."

    def _build_query(self, ctx: SceneContext, behavior: str = "", spatial_rel: str = "") -> str:
        def _clean(text: str) -> str:
            return str(text or "").strip()

        scene_understanding = _clean(ctx.scene_understanding)
        ego_driving_intention = _clean(ctx.ego_driving_intention)
        behavior_text = _clean(behavior)
        spatial_rel_text = _clean(spatial_rel)
        parts = []
        if scene_understanding:
            parts.append(f"scene_understanding: {scene_understanding}")
        if ego_driving_intention:
            parts.append(f"ego_driving_intention: {ego_driving_intention}")
        if spatial_rel_text:
            parts.append(f"spatial_relationships_with_ego: {spatial_rel_text}.")
        if behavior_text:
            parts.append(f"potential_adversarial_behaviors: {behavior_text}")

        return " ".join(parts).strip()

    def _read_scene_text(self, ctx: SceneContext) -> str:
        if isinstance(ctx.scene_text, str) and ctx.scene_text.strip():
            return ctx.scene_text
        scene_text_path = getattr(ctx, "scene_text_path", None)
        if not scene_text_path:
            return ""
        path = Path(str(scene_text_path))
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def _load_history_payload(self, ctx: SceneContext) -> Optional[Dict]:
        scene_text_path = getattr(ctx, "scene_text_path", None)
        if not scene_text_path:
            return None
        txt_path = Path(str(scene_text_path))
        if not txt_path.is_file():
            return None
        traj_path = txt_path.parent / f"{txt_path.stem}_trajectory.json"
        if not traj_path.is_file():
            return None
        try:
            return json.loads(traj_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _load_history_trajectories(self, ctx: SceneContext) -> Optional[List[List[Tuple[float, float]]]]:
        payload = self._load_history_payload(ctx)
        if not isinstance(payload, dict):
            return None
        history = payload.get("history_trajectory_local")
        if not isinstance(history, list):
            return None

        cleaned: List[List[Tuple[float, float]]] = []
        for seg in history:
            if not isinstance(seg, list):
                cleaned.append([])
                continue
            pts: List[Tuple[float, float]] = []
            for p in seg:
                if not isinstance(p, list) or len(p) < 2:
                    pts.append((0.0, 0.0))
                    continue
                x, y = p[0], p[1]
                try:
                    fx = 0.0 if x is None else float(x)
                except (TypeError, ValueError):
                    fx = 0.0
                try:
                    fy = 0.0 if y is None else float(y)
                except (TypeError, ValueError):
                    fy = 0.0
                pts.append((fx, fy))
            cleaned.append(pts)
        return cleaned

    def _load_history_controls(self, ctx: SceneContext) -> Optional[List[List[Dict]]]:
        payload = self._load_history_payload(ctx)
        if not isinstance(payload, dict):
            return None
        history = payload.get("history_control_local")
        if not isinstance(history, list):
            return None
        cleaned: List[List[Dict]] = []
        for seg in history:
            if not isinstance(seg, list):
                cleaned.append([])
                continue
            controls: List[Dict] = []
            for step in seg:
                if not isinstance(step, list) or len(step) < 2:
                    controls.append({"acceleration": 0.0, "steering_angle": 0.0})
                    continue
                acc, steer = step[0], step[1]
                try:
                    facc = 0.0 if acc is None else float(acc)
                except (TypeError, ValueError):
                    facc = 0.0
                try:
                    fsteer = 0.0 if steer is None else float(steer)
                except (TypeError, ValueError):
                    fsteer = 0.0
                controls.append({"acceleration": facc, "steering_angle": fsteer})
            cleaned.append(controls)
        return cleaned

    def _load_history_diffs(self, ctx: SceneContext) -> Optional[List[List[Dict]]]:
        payload = self._load_history_payload(ctx)
        if not isinstance(payload, dict):
            return None
        history = payload.get("history_diff")
        if not isinstance(history, list):
            return None

        cleaned: List[List[Dict]] = []
        for seg in history:
            if not isinstance(seg, list):
                cleaned.append([])
                continue
            diffs: List[Dict] = []
            for step in seg:
                if not isinstance(step, dict):
                    diffs.append({"delta_s": 0.0, "delta_heading": 0.0})
                    continue
                delta_s = step.get("delta_s")
                delta_heading = step.get("delta_heading")
                try:
                    f_delta_s = 0.0 if delta_s is None else float(delta_s)
                except (TypeError, ValueError):
                    f_delta_s = 0.0
                try:
                    f_delta_heading = 0.0 if delta_heading is None else float(delta_heading)
                except (TypeError, ValueError):
                    f_delta_heading = 0.0
                diffs.append({"delta_s": f_delta_s, "delta_heading": f_delta_heading})
            cleaned.append(diffs)
        return cleaned

    def _load_history_speed_mean(self, ctx: SceneContext) -> Optional[List[float]]:
        payload = self._load_history_payload(ctx)
        if not isinstance(payload, dict):
            return None
        raw = payload.get("history_speed_mean")
        if not isinstance(raw, list):
            return None
        out: List[float] = []
        for v in raw:
            try:
                out.append(0.0 if v is None else float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    @staticmethod
    def _parse_agent_length_width_entry(
        entry,
        default_length: float = 4.5,
        default_width: float = 1.8,
    ) -> Tuple[float, float]:
        """Parse one ``agent_length_width`` row: ``{length, width}`` in meters."""
        if not isinstance(entry, dict):
            return default_length, default_width
        length = entry.get("length")
        width = entry.get("width")
        try:
            length_f = float(length)
        except (TypeError, ValueError):
            length_f = float("nan")
        try:
            width_f = float(width)
        except (TypeError, ValueError):
            width_f = float("nan")
        if not math.isfinite(length_f) or length_f <= 0.0:
            length_f = default_length
        if not math.isfinite(width_f) or width_f <= 0.0:
            width_f = default_width
        return length_f, width_f

    def _load_query_vehicle_dims(
        self,
        ctx: SceneContext,
        target_row_id: Optional[int],
    ) -> Dict[str, float]:
        """Ego/target L×W from ``*_trajectory.json`` ``agent_length_width`` (row 0 = ego)."""
        defaults = {
            "ego_width": 1.8,
            "ego_length": 4.5,
            "target_width": 1.8,
            "target_length": 4.5,
        }
        payload = self._load_history_payload(ctx)
        if not isinstance(payload, dict):
            return dict(defaults)
        alw = payload.get("agent_length_width")
        if not isinstance(alw, list) or not alw:
            return dict(defaults)

        ego_length, ego_width = self._parse_agent_length_width_entry(
            alw[0], default_length=defaults["ego_length"], default_width=defaults["ego_width"]
        )
        if target_row_id is None or target_row_id < 0 or target_row_id >= len(alw):
            target_length, target_width = (
                defaults["target_length"],
                defaults["target_width"],
            )
        else:
            target_length, target_width = self._parse_agent_length_width_entry(
                alw[target_row_id],
                default_length=defaults["target_length"],
                default_width=defaults["target_width"],
            )
        return {
            "ego_width": ego_width,
            "ego_length": ego_length,
            "target_width": target_width,
            "target_length": target_length,
        }

    def _load_predicted_future(
        self, ctx: SceneContext, agent_id: Optional[int]
    ) -> Optional[List[Tuple[float, float]]]:
        """Load predicted future trajectory for ego (agent_id=None or 0) or a
        surrounding agent (agent_id>0).

        The JSON payload is produced by `process_nusc_scene_graph.py`:
        - ego: `predicted_ego_future_trajectory_local` -> List[[x,y], ...]
        - surrounding: `predicted_surrounding_future_trajectory_local` -> List[List[[x,y], ...], ...]

        Here `agent_id` is assumed to be the same row index used by
        `history_trajectory_local`: ego is row 0, surrounding agents are rows
        1..N-1, and they map to surrounding_future[row-1].
        """
        payload = self._load_history_payload(ctx)
        if not isinstance(payload, dict):
            return None

        future: Optional[list] = None
        if agent_id is None or agent_id == 0:
            raw = payload.get("predicted_ego_future_trajectory_local")
            if isinstance(raw, list):
                future = raw
        else:
            raw = payload.get("predicted_surrounding_future_trajectory_local")
            if isinstance(raw, list):
                idx = int(agent_id) - 1
                if 0 <= idx < len(raw) and isinstance(raw[idx], list):
                    future = raw[idx]
        if not isinstance(future, list):
            return None

        pts: List[Tuple[float, float]] = []
        for p in future:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                fx = 0.0 if p[0] is None else float(p[0])
                fy = 0.0 if p[1] is None else float(p[1])
            except (TypeError, ValueError):
                continue
            pts.append((fx, fy))
        return pts

    @staticmethod
    def _derive_future_controls(
        ego_history_traj: List[Tuple[float, float]],
        future_pts: List[Tuple[float, float]],
    ) -> List[Dict]:
        """Derive (acceleration, steering_angle) for each predicted future step
        by running ``compute_kinematics_from_trajectory`` over the combined
        [last history point, future points] sequence and dropping the anchor.
        """
        if not future_pts:
            return []
        anchor = tuple(ego_history_traj[-1]) if ego_history_traj else (0.0, 0.0)
        pts = [anchor] + [tuple(p) for p in future_pts]
        n = len(pts)
        t = np.arange(n, dtype=float) * DT_CTRL
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        acc, steer = compute_kinematics_from_trajectory(t, x, y)
        return [
            {"acceleration": float(acc[i]), "steering_angle": float(steer[i])}
            for i in range(1, n)
        ]

    @staticmethod
    def _derive_future_diffs(
        ego_history_traj: List[Tuple[float, float]],
        future_pts: List[Tuple[float, float]],
    ) -> List[Dict]:
        """Derive (delta_s, delta_heading) for each predicted future step.

        ``delta_s`` is the Euclidean displacement between consecutive points;
        ``delta_heading`` is the wrapped change in instantaneous velocity
        heading.  The first future step uses the last history point as its
        anchor so the step is continuous with the recorded history.
        """
        if not future_pts:
            return []
        anchor = tuple(ego_history_traj[-1]) if ego_history_traj else (0.0, 0.0)
        pts = [anchor] + [tuple(p) for p in future_pts]

        diffs: List[Dict] = []
        prev_heading = 0.0
        first = True
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            ds = math.hypot(dx, dy)
            heading = math.atan2(dy, dx) if ds > 1e-9 else prev_heading
            if first:
                dh = 0.0
                first = False
            else:
                dh = heading - prev_heading
                dh = (dh + math.pi) % (2 * math.pi) - math.pi
            prev_heading = heading
            diffs.append({"delta_s": float(ds), "delta_heading": float(dh)})
        return diffs

    def _extract_current_state(self, scene_text: str, agent_id: Optional[int]) -> Dict:
        if not scene_text:
            return {"heading": 0.0, "speed": 0.0}
        if agent_id is None:
            pattern = r"EGO:\s*.*?Current state:\s*\(x=[^,]+,\s*y=[^,]+,\s*heading=([-+]?\d+(?:\.\d+)?)\),\s*speed=([-+]?\d+(?:\.\d+)?)\s*m/s"
        else:
            pattern = (
                rf"Agent\s+{int(agent_id)}:\s*.*?Current state:\s*"
                rf"\(x=[^,]+,\s*y=[^,]+,\s*heading=([-+]?\d+(?:\.\d+)?)\),\s*"
                rf"speed=([-+]?\d+(?:\.\d+)?)\s*m/s"
            )
        match = re.search(pattern, scene_text, re.IGNORECASE | re.DOTALL)
        if not match:
            return {"heading": 0.0, "speed": 0.0}
        try:
            heading = float(match.group(1))
            speed = float(match.group(2))
        except (TypeError, ValueError):
            return {"heading": 0.0, "speed": 0.0}
        return {"heading": heading, "speed": speed}

    @staticmethod
    def _extract_current_xyh(scene_text: str, agent_id: Optional[int]) -> Dict[str, float]:
        """Extract current (x, y, heading) from scene_text for ego/agent blocks."""
        if not scene_text:
            return {"x": 0.0, "y": 0.0, "heading": 0.0}
        if agent_id is None:
            pattern = (
                r"EGO:\s*.*?Current state:\s*"
                r"\(x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m,\s*"
                r"heading=([-+]?\d+(?:\.\d+)?)\)"
            )
        else:
            pattern = (
                rf"Agent\s+{int(agent_id)}:\s*.*?Current state:\s*"
                rf"\(x=([-+]?\d+(?:\.\d+)?)\s*m,\s*y=([-+]?\d+(?:\.\d+)?)\s*m,\s*"
                rf"heading=([-+]?\d+(?:\.\d+)?)\)"
            )
        match = re.search(pattern, scene_text, re.IGNORECASE | re.DOTALL)
        if not match:
            return {"x": 0.0, "y": 0.0, "heading": 0.0}
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            heading = float(match.group(3))
        except (TypeError, ValueError):
            return {"x": 0.0, "y": 0.0, "heading": 0.0}
        return {"x": x, "y": y, "heading": heading}

    @staticmethod
    def _kinematics_at_trajectory_end(
        traj: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Longitudinal acc (m/s²) and signed yaw rate (rad/s) at last history sample."""
        if len(traj) < 2:
            return 0.0, 0.0
        n = len(traj)
        t = np.arange(n, dtype=float) * float(DT_CTRL)
        x = np.array([float(p[0]) for p in traj], dtype=float)
        y = np.array([float(p[1]) for p in traj], dtype=float)
        acc, _ = compute_kinematics_from_trajectory(t, x, y)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        heading = np.unwrap(np.arctan2(vy, vx))
        yaw_rate = np.gradient(heading, t)
        return float(acc[-1]), float(yaw_rate[-1])

    @staticmethod
    def _merge_current_state_kinematics(
        ego_state: Dict,
        target_state: Dict,
        ego_traj: List[Tuple[float, float]],
        target_traj: List[Tuple[float, float]],
    ) -> Tuple[Dict, Dict]:
        """Add ``acc`` and ``yaw_rate`` to match RAG ``*_current_state`` (target yaw is relative)."""
        ego_acc, ego_yr = RetrievalAgent._kinematics_at_trajectory_end(ego_traj)
        tgt_acc, tgt_yr_w = RetrievalAgent._kinematics_at_trajectory_end(target_traj)
        if len(ego_traj) >= 2 and len(target_traj) >= 2:
            tgt_yr = tgt_yr_w - ego_yr
        elif len(target_traj) >= 2:
            tgt_yr = tgt_yr_w
        else:
            tgt_yr = 0.0

        if ego_traj and target_traj:
            ex, ey = ego_traj[-1]
            tx, ty = target_traj[-1]
            delta_x = round(float(tx) - float(ex), 4)
            delta_y = round(float(ty) - float(ey), 4)
        else:
            delta_x = 0.0
            delta_y = 0.0

        try:
            ego_spd = float(ego_state.get("speed", 0.0))
        except (TypeError, ValueError):
            ego_spd = 0.0
        try:
            tgt_spd = float(target_state.get("speed", 0.0))
        except (TypeError, ValueError):
            tgt_spd = 0.0
        relative_speed = round(tgt_spd - ego_spd, 3)
        relative_acc = round(tgt_acc - ego_acc, 3)

        ego_out = {
            **ego_state,
            "acc": round(ego_acc, 3),
            "yaw_rate": round(ego_yr, 4),
        }
        tgt_out = {
            **target_state,
            "acc": round(tgt_acc, 3),
            "yaw_rate": round(tgt_yr, 4),
            "delta_x": delta_x,
            "delta_y": delta_y,
            "relative_speed": relative_speed,
            "relative_acc": relative_acc,
        }
        return ego_out, tgt_out

    def _compute_adv_current_state(
        self,
        ctx: SceneContext,
        scene_text: str,
        behavior_index: int,
    ) -> Dict[str, float]:
        """Return adversarial vehicle state in ego-local frame: (x, y, heading)."""
        adv_ids: List[int] = []
        for x in (ctx.adversarial_vehicle_ids or []):
            try:
                adv_ids.append(int(x))
            except (TypeError, ValueError):
                continue
        target_id = adv_ids[behavior_index] if behavior_index < len(adv_ids) else None

        # Prefer scene_text current-state parsing to avoid trajectory dependency.
        tgt_xyh = (
            self._extract_current_xyh(scene_text, agent_id=target_id)
            if target_id is not None
            else {"x": 0.0, "y": 0.0, "heading": 0.0}
        )
        ego_xyh = self._extract_current_xyh(scene_text, agent_id=None)

        x_local = float(tgt_xyh.get("x", 0.0))
        y_local = float(tgt_xyh.get("y", 0.0))
        tgt_h = float(tgt_xyh.get("heading", 0.0))
        ego_h = float(ego_xyh.get("heading", 0.0))
        heading_local = (tgt_h - ego_h + math.pi) % (2 * math.pi) - math.pi

        return {"x": x_local, "y": y_local, "heading": float(heading_local)}

    def _build_query_event_feature(
        self,
        ctx: SceneContext,
        ego_traj: List[Tuple[float, float]],
        target_traj: List[Tuple[float, float]],
        target_row_id: Optional[int] = None,
        dt: float = DT_CTRL,
    ) -> Dict:
        """Construct an ``event_feature`` dict from query history trajectories.

        Mirrors ``crash_data_processor.build_event_feature`` but sources its
        inputs from the recent history window of the current scene (ego +
        target), treating the latest sample as the analogue of ``impact_time``.
        Speed, heading and acceleration are derived from the (x, y) trajectory
        via finite differences. Vehicle L×W come from ``*_trajectory.json``
        ``agent_length_width`` (row 0 = ego, ``target_row_id`` = adversarial row).

        Returns an empty dict when either trajectory is too short.
        """
        if not ego_traj or not target_traj:
            return {}

        n = min(len(ego_traj), len(target_traj))
        if n < 2:
            return {}
        ego_traj = ego_traj[:n]
        target_traj = target_traj[:n]

        time = np.arange(n, dtype=float) * float(dt)
        x_ego = np.array([float(p[0]) for p in ego_traj], dtype=float)
        y_ego = np.array([float(p[1]) for p in ego_traj], dtype=float)
        x_sur = np.array([float(p[0]) for p in target_traj], dtype=float)
        y_sur = np.array([float(p[1]) for p in target_traj], dtype=float)

        def _derive_v_psi(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            vx = np.gradient(x, time)
            vy = np.gradient(y, time)
            v = np.hypot(vx, vy)
            psi = np.arctan2(vy, vx)
            return v, psi

        v_ego, psi_ego = _derive_v_psi(x_ego, y_ego)
        v_sur, psi_sur = _derive_v_psi(x_sur, y_sur)

        # Acceleration from trajectory (steer is discarded here).
        acc_ego, _ = compute_kinematics_from_trajectory(time, x_ego, y_ego)
        acc_sur, _ = compute_kinematics_from_trajectory(time, x_sur, y_sur)

        ts = EventTimeSeries(
            event_id=0,
            target_id=0,
            time=time,
            x_ego=x_ego,
            y_ego=y_ego,
            v_ego=v_ego,
            psi_ego=psi_ego,
            acc_ego=acc_ego,
            x_sur=x_sur,
            y_sur=y_sur,
            v_sur=v_sur,
            psi_sur=psi_sur,
        )
        # Treat the end of the recorded history as the "impact" anchor so the
        # whole history contributes to the pre-impact mask in build_event_feature.
        last_time = float(time[-1])
        vehicle_dims = self._load_query_vehicle_dims(ctx, target_row_id)
        return build_event_feature(
            ts,
            acc_ego,
            acc_sur,
            last_time,
            ego_width=vehicle_dims["ego_width"],
            ego_length=vehicle_dims["ego_length"],
            target_width=vehicle_dims["target_width"],
            target_length=vehicle_dims["target_length"],
        )

    def _build_query_features(
        self,
        ctx: SceneContext,
        scene_text: str,
        behavior_index: int,
        feature_mode: str,
    ) -> Tuple[List[float], Dict, List[float]]:
        adv_ids: List[int] = []
        for x in (ctx.adversarial_vehicle_ids or []):
            try:
                adv_ids.append(int(x))
            except (TypeError, ValueError):
                continue
        target_id = adv_ids[behavior_index] if behavior_index < len(adv_ids) else None

        query_sequence_vector: List[float] = []
        mode = str(feature_mode or "").strip().lower()

        # Predicted ego future (12 local (x, y) steps) shared across all modes.
        future_pts = self._load_predicted_future(ctx, agent_id=None) or []
        trajectories_cache: Optional[List[List[Tuple[float, float]]]] = None

        def _ego_history_traj() -> List[Tuple[float, float]]:
            nonlocal trajectories_cache
            if trajectories_cache is None:
                trajectories_cache = self._load_history_trajectories(ctx) or []
            return trajectories_cache[0] if trajectories_cache else []

        def _target_history_traj(agent_idx: Optional[int]) -> List[Tuple[float, float]]:
            nonlocal trajectories_cache
            if agent_idx is None:
                return []
            if trajectories_cache is None:
                trajectories_cache = self._load_history_trajectories(ctx) or []
            if not trajectories_cache:
                return []
            if 0 <= agent_idx < len(trajectories_cache):
                return trajectories_cache[agent_idx]
            return []

        if mode == "trajectory":
            trajectories_cache = self._load_history_trajectories(ctx) or []
            ego_traj = trajectories_cache[0] if trajectories_cache else []
            target_traj = (
                trajectories_cache[target_id]
                if target_id is not None and 0 <= target_id < len(trajectories_cache)
                else []
            )
            # Extend ego with predicted future so the query window is
            # ``4 history + 12 future = 16`` local (x, y) points.
            ego_traj_ext = list(ego_traj) + list(future_pts)
            if ego_traj_ext or target_traj:
                ego_dicts = [{"x": x, "y": y} for x, y in ego_traj_ext]
                tgt_dicts = [{"x": x, "y": y} for x, y in target_traj]
                query_sequence_vector = build_trajectory_vector(ego_dicts, tgt_dicts)
        elif mode == "control":
            controls = self._load_history_controls(ctx) or []
            ego_controls = controls[0] if controls else []
            target_controls = (
                controls[target_id]
                if target_id is not None and 0 <= target_id < len(controls)
                else []
            )
            future_ctrls = self._derive_future_controls(_ego_history_traj(), future_pts)
            ego_controls_ext = list(ego_controls) + future_ctrls
            target_future_pts = self._load_predicted_future(ctx, agent_id=target_id) or []
            target_future_ctrls = self._derive_future_controls(
                _target_history_traj(target_id), target_future_pts
            )
            target_controls_ext = list(target_controls) + target_future_ctrls
            if ego_controls_ext or target_controls_ext:
                query_sequence_vector = build_control_vector(
                    ego_controls_ext,
                    target_controls_ext,
                    use_control=True,
                )
        elif mode == "diff":
            diffs = self._load_history_diffs(ctx) or []
            ego_diffs = diffs[0] if diffs else []
            target_diffs = (
                diffs[target_id]
                if target_id is not None and 0 <= target_id < len(diffs)
                else []
            )
            future_diffs = self._derive_future_diffs(_ego_history_traj(), future_pts)
            ego_diffs_ext = list(ego_diffs) + future_diffs
            target_future_pts = self._load_predicted_future(ctx, agent_id=target_id) or []
            target_future_diffs = self._derive_future_diffs(
                _target_history_traj(target_id), target_future_pts
            )
            target_diffs_ext = list(target_diffs) + target_future_diffs
            if ego_diffs_ext or target_diffs_ext:
                query_sequence_vector = build_control_vector(
                    ego_diffs_ext,
                    target_diffs_ext,
                    use_control=False,
                )

        trajectories = self._load_history_trajectories(ctx) or []
        ego_traj = trajectories[0] if trajectories else []
        target_traj = (
            trajectories[target_id]
            if target_id is not None and 0 <= target_id < len(trajectories)
            else []
        )

        ego_state = self._extract_current_state(scene_text, agent_id=None)
        target_state = self._extract_current_state(scene_text, agent_id=target_id)
        ego_state, target_state = self._merge_current_state_kinematics(
            ego_state, target_state, ego_traj, target_traj
        )

        vehicle_dims = self._load_query_vehicle_dims(ctx, target_row_id=target_id)
        query_numeric_meta = {
            "ego_current_state": ego_state,
            "target_current_state": target_state,
            "ego_vehicle_length": vehicle_dims["ego_length"],
            "ego_vehicle_width": vehicle_dims["ego_width"],
            "adversary_vehicle_length": vehicle_dims["target_length"],
            "adversary_vehicle_width": vehicle_dims["target_width"],
        }

        # Build event_feature vector from history trajectories, mirroring the
        # pre-impact summary produced for RAG records.  If trajectory history
        # is unavailable the vector will be empty and the downstream cosine
        # comparison will be skipped.
        query_event_feature = self._build_query_event_feature(
            ctx, ego_traj, target_traj, target_row_id=target_id
        )
        query_event_feature_vector: List[float] = (
            build_event_feature_vector(query_event_feature) if query_event_feature else []
        )

        print("query_sequence_vector", query_sequence_vector)
        print("query_numeric_meta", query_numeric_meta)
        print("query_event_feature", query_event_feature)
        return query_sequence_vector, query_numeric_meta, query_event_feature_vector

    def _build_queries(self, ctx: SceneContext) -> list:
        behaviors = [str(b).strip() for b in (ctx.potential_adversarial_behaviors or []) if str(b).strip()]
        spatial_rels = [str(r).strip() for r in (getattr(ctx, "spatial_relationships_with_ego", []) or []) if str(r).strip()]
        if not behaviors:
            return [self._build_query(ctx, behavior="")]

        # Align spatial relations with behaviors by index, fill with "" if not enough
        queries = []
        for idx, behavior in enumerate(behaviors):
            spatial_rel = spatial_rels[idx] if idx < len(spatial_rels) else ""
            # _build_query should accept an additional spatial_rel parameter
            queries.append(self._build_query(ctx, behavior=behavior, spatial_rel=spatial_rel))
        return queries

    def run(self, message: AgentMessage):
        ctx: SceneContext = message.payload["context"]
        scene_understanding = str(ctx.scene_understanding or "")
        ego_intention = str(ctx.ego_driving_intention or "")
        scene_text_raw = self._read_scene_text(ctx)
        scene_blob = f"{scene_understanding} {ego_intention} {scene_text_raw}".strip()
        env = _infer_environment(scene_blob)
        base_metadata_tags = _extract_tags(f"{env} {scene_blob}", _METADATA_TAG_PATTERNS)

        queries = self._build_queries(ctx)
        spatial_rels = [str(r).strip() for r in (getattr(ctx, "spatial_relationships_with_ego", []) or []) if str(r).strip()]
        behaviors = [str(b).strip() for b in (ctx.potential_adversarial_behaviors or []) if str(b).strip()]
        messages = []
        for idx, query in enumerate(queries):
            behavior = behaviors[idx] if idx < len(behaviors) else ""
            spatial_rel = spatial_rels[idx] if idx < len(spatial_rels) else ""
            behavior_blob = f"{behavior} {spatial_rel}".strip()
            interaction_tags = _extract_tags(behavior_blob, _INTERACTION_TAG_PATTERNS)
            metadata_tags = list(dict.fromkeys(base_metadata_tags + _extract_tags(behavior_blob, _METADATA_TAG_PATTERNS)))
            feature_mode = self.vector_mode
            query_sequence_vector, query_numeric_meta, query_event_feature_vector = self._build_query_features(
                ctx=ctx,
                scene_text=scene_text_raw,
                behavior_index=idx,
                feature_mode=feature_mode,
            )
            future_pts = self._load_predicted_future(ctx, agent_id=None) or []
            print("future_pts:", future_pts)
            adv_current_state = self._compute_adv_current_state(ctx, scene_text_raw, idx)
            print("adv_current_state:", adv_current_state)
            if feature_mode == "trajectory":
                use_trajectory = True
            else:
                use_trajectory = False

            result = self.retriever.retrieve_anchor_with_score(
                query_text=query,
                env=env,
                scene_text=scene_blob,
                behavior_text=behavior_blob,
                metadata_tags=metadata_tags,
                interaction_tags=interaction_tags,
                query_sequence_vector=query_sequence_vector,
                query_numeric_meta=query_numeric_meta,
                query_event_feature_vector=query_event_feature_vector,
                threshold=self.similarity_threshold,
                top_k=1,
                use_trajectory=use_trajectory,
                predicted_ego_future=future_pts,
                adv_current_state=adv_current_state,
            ) or {
                "hit": False,
                "anchor_trajectory": [],
                "similarity_score": 0.0,
                "match": None,
                "top_mode": None,
                "reference_trajectory": {},
            }
            hit = bool(result.get("hit", False))
            top_mode = result.get("top_mode") or {}
            reference_trajectory = result.get("reference_trajectory") or {}
            if hit:
                retrieved_anchor = result.get("match") or {}
            else:
                retrieved_anchor = top_mode if top_mode else {}
            messages.append(
                AgentMessage(
                    sender=self.agent_name,
                    receiver=message.sender,
                    msg_type="response",
                    payload={
                        "behavior_index": idx,
                        "behavior": (
                            str(ctx.potential_adversarial_behaviors[idx]).strip()
                            if idx < len(ctx.potential_adversarial_behaviors)
                            else ""
                        ),
                        "query": query,
                        "retrieval_mode": feature_mode,
                        "retrieved_anchor": retrieved_anchor,
                        "retrieved_mode": top_mode,
                        "reference_trajectory": reference_trajectory,
                        "anchor_trajectory_6s": result.get("anchor_trajectory", []) if hit else [],
                        "retrieval_similarity": float(result.get("similarity_score", 0.0)),
                        "anchor_source": "rag" if hit else "llm",
                    },
                    trace_id=message.trace_id,
                )
            )
        return messages
