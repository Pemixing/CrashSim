import os
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentMessage, BaseAgent
from .memory import SceneContext
from .tools.api_tools import call_openai_chat, extract_json
from .tools.traj_tools import load_vehicle_dims_from_trajectory_json, trajectory_collision_check


def _build_retry_system_prompt(output_xy_trajectory: bool) -> str:
    if output_xy_trajectory:
        output_spec = (
            '  "anchor_trajectory_6s": [{"t":0.5,"x":float,"y":float}, ... 12 points],\n'
        )
    else:
        output_spec = (
            '  "anchor_trajectory_6s": [{"t":0.5,"acceleration":float,"steering_angle":float}, ... 12 control steps],\n'
        )

    return (
        "You are a trajectory refinement module for adversarial safety-critical scenario synthesis. "
        "You do NOT generate a scenario from scratch: you OPTIMIZE the adversarial vehicle's anchor trajectory "
        "given (1) the existing candidate adversarial trajectory, (2) that vehicle's inferred "
        "`potential_adversarial_behaviors`, (3) optional real-world collision reference (retrieved mode behavior "
        "and reference trajectory), and (4) the fixed predicted ego future trajectory.\n\n"

        "The user message will supply the last adversarial trajectory (collision check failed), ego future, "
        "history, behavior hints, and any retrieved reference example. Use ALL of them to guide the refinement.\n\n"

        "Goal:\n"
        "Adjust the adversarial vehicle so that it COLLIDES with the predicted ego future trajectory "
        "within the next 6 seconds. The refined trajectory must be human-aware: it should include "
        "strategic maneuvers, small hesitations, and emergency avoidance or braking actions, "
        "but ultimately still result in a collision, mimicking real-world accident behavior.\n\n"

        "Refinement Guidelines:\n"
        "1. 【IMPORTANT】Local and controlled modification:\n"
        "   - Adjust the trajectory with appropriate changes to improve interaction.\n"
        "   - Prioritize temporal adjustments (e.g., braking or acceleration) to align collision timing.\n"
        "   - Avoid unnecessary geometric alterations unless required by interaction constraints.\n\n"

        "2. Keep the ego future trajectory fixed; only change the adversarial plan.\n"

        "3. The refined trajectory must lead to collision with the ego within the 6s horizon, "
        "but not by naive straight-line acceleration; include realistic human driving behavior "
        "(strategic maneuvers, small hesitations, emergency braking or avoidance close to collision) "
        "while collision still occurs.\n"

        "4. Reflect `potential_adversarial_behaviors` and any retrieved reference example when provided.\n"

        "5. Unless the behavior implies reversing, prefer forward motion along the current heading.\n\n"

        "Output rules:\n"
        "- Always output valid JSON only.\n"
        "The output must follow this schema exactly:\n"
        "{\n"
        '  "adversarial_vehicle_id": int,\n'
        f"{output_spec}"
        '  "anchor_confidence": float,\n'
        '  "closest_t_index": int,\n'
        '  "behavior_tag": "string",\n'
        "}"
    )


def _build_step2_retry_prompt(
    ctx: SceneContext,
    adversarial_vehicle_history: List[Dict],
    adv_current_state: Dict[str, float],
    ego_future_trajectory: List[Dict],
    last_adv_trajectory: List[Dict],
    collision_info: Dict[str, object],
    retry_idx: int,
    output_xy_trajectory: bool,
    adversarial_behavior: str = "",
    reference_trajectory: str = "",
) -> str:
    if output_xy_trajectory:
        step2_requirement = (
            f"2. Regenerate Step 2 only: a new human-aware adversarial trajectory with exactly 12 points (t=0.5~6.0). "
            f"Current state at t=0.0: {adv_current_state}. The first future point (t=0.5) must be based on this current state. "
            "The trajectory must:\n"
            "- Lead to collision with the ego trajectory within 6 seconds, but not by naive straight-line acceleration\n"
            "- Include realistic human driving behavior: strategic maneuvers, small hesitations\n"
            "- Incorporate emergency braking or avoidance actions close to collision, but collision still occurs\n"
            "- Be smooth and physically plausible\n"
            "- Reflect the specified adversarial behavior\n"
        )
    else:
        step2_requirement = (
            f"2. Regenerate Step 2 only: a new human-aware adversarial control sequence with exactly 12 steps (t=0.5~6.0), "
            f"using format {{t, acceleration, steering_angle}}. Current state at t=0.0: {adv_current_state}. "
            "The first control step (t=0.5) must be based on this current state. The controls must:\n"
            "- Lead to collision with the ego trajectory within 6 seconds, but not via direct straight-line crash\n"
            "- Include realistic human driving behavior: strategic maneuvers, small hesitations\n"
            "- Incorporate emergency braking or avoidance actions close to collision, but collision still occurs\n"
            "- Be smooth and physically plausible\n"
            "- Reflect the specified adversarial behavior\n"
        )

    return (
        "Step 3 validation failed. Retry Step 2 only.\n\n"

        f"[Retry] {retry_idx}\n\n"

        "Use the following information to refine one human-aware adversarial anchor trajectory.\n\n"

        "Context:\n"
        f"- ego_intention: {ctx.ego_driving_intention}\n"
        f"- adversarial_vehicle_id: {ctx.adversarial_vehicle_ids}\n"
        f"- adversarial_history: {adversarial_vehicle_history}\n"
        f"- 【IMPORTANT】potential_adversarial_behavior: {ctx.potential_adversarial_behaviors}\n"
        f"- ego_future (fixed): {ego_future_trajectory}\n\n"

        "Failure:\n"
        f"- last_trajectory: {last_adv_trajectory}\n"
        f"- collision_info: {collision_info}\n"
        "- Adjust timing or interaction to fix the failure.\n\n"

        "Requirements:\n"
        f"- {step2_requirement}"
        "- If no collision occurs, revise until collision is achieved, ensuring the trajectory remains human-aware.\n"
        "- Avoid repeating the previous trajectory pattern.\n\n"
        + (
            "Reference example from a real-world collision event:\n"
            f"- adversarial_behavior (retrieved mode): {adversarial_behavior}\n"
            f"- reference_trajectory: {reference_trajectory}\n"
            if (adversarial_behavior or reference_trajectory)
            else ""
        )
        + "\n"

        "Output:\n"
        "- JSON only with key `anchor_trajectory_6s`.\n"
    )


class RetryAgent(BaseAgent):
    """Validate trajectory collision and, when it fails, call the LLM to refine the trajectory.

    The agent is intentionally stateless: each call takes a candidate adversarial
    trajectory (from RAG or the LLM branch of :class:`TrajectoryAgent`) along with
    the fixed ego future, and returns a refined trajectory plus the final
    collision-check info.
    """

    agent_name = "RetryAgent"

    def __init__(
        self,
        api_base: str = "https://your-api-endpoint/v1",
        api_key: str = "",
        model: str = "",
        llm_output_trajectory: bool = True,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model = model or "gpt-4o"
        self.llm_output_trajectory = llm_output_trajectory
        if not self.api_key:
            raise ValueError("LLM api_key is required.")

    def describe(self) -> str:
        return "Validate collision and refine adversarial trajectory via LLM retry."

    def run(self, message: AgentMessage) -> AgentMessage:
        # Local import avoids a circular dependency with trajectory_agent at import time.
        from .trajectory_agent import (
            _rag_reference_prompt_fields,
            _resolve_anchor_xy_trajectory,
        )

        payload = message.payload
        ctx: SceneContext = payload["context"]
        predicted_ego: List[Dict] = payload["predicted_ego"]
        adv_trajectory: List[Dict] = payload["adv_trajectory"]
        adv_history_points: List[Tuple[float, float]] = payload.get("adv_history_points", []) or []
        adv_current_state: Optional[Dict[str, float]] = payload.get("adv_current_state")
        parsed: Dict = payload.get("parsed") or {}
        adv_row_id = int(payload.get("adv_row_id", 0))
        dims = load_vehicle_dims_from_trajectory_json(ctx.scene_text_path, adv_row_id)

        def _check_collision(ego_traj: List[Dict], adv_traj: List[Dict]) -> Dict[str, Any]:
            return trajectory_collision_check(
                ego_traj,
                adv_traj,
                ego_length_m=dims["ego_length"],
                ego_width_m=dims["ego_width"],
                target_length_m=dims["target_length"],
                target_width_m=dims["target_width"],
            )

        collision_info: Optional[Dict[str, Any]] = payload.get("collision_info")
        if collision_info is None:
            collision_info = _check_collision(predicted_ego, adv_trajectory)

        rag_adv_behavior, rag_ref_trajectory = _rag_reference_prompt_fields(ctx)

        max_retry = self._resolve_max_retry()
        enable_retry = self._resolve_enable_retry()
        retry_count = 0
        norm_traj = adv_trajectory

        while enable_retry and (not collision_info.get("collision")) and retry_count < max_retry:
            retry_count += 1
            retry_raw = call_openai_chat(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model,
                system_prompt=_build_retry_system_prompt(self.llm_output_trajectory),
                user_content=_build_step2_retry_prompt(
                    ctx=ctx,
                    adversarial_vehicle_history=adv_history_points,
                    adv_current_state=adv_current_state,
                    ego_future_trajectory=predicted_ego,
                    last_adv_trajectory=norm_traj,
                    collision_info=collision_info,
                    retry_idx=retry_count,
                    output_xy_trajectory=self.llm_output_trajectory,
                    adversarial_behavior=rag_adv_behavior,
                    reference_trajectory=rag_ref_trajectory,
                ),
                temperature=0.5,
            )
            print(f"[RetryAgent] retry {retry_count} response:", retry_raw)
            retry_parsed = extract_json(retry_raw)
            retry_traj = _resolve_anchor_xy_trajectory(
                anchor_source="llm",
                anchor_trajectory_6s=retry_parsed.get("anchor_trajectory_6s"),
                history_points=adv_history_points,
                current_state=adv_current_state,
            )
            retry_collision = _check_collision(predicted_ego, retry_traj)
            print(f"[RetryAgent] collision check retry {retry_count}: {retry_collision}")

            parsed = retry_parsed if isinstance(retry_parsed, dict) else parsed
            norm_traj = retry_traj
            collision_info = retry_collision
            print("retry_traj", retry_traj)

        return AgentMessage(
            sender=self.agent_name,
            receiver=message.sender,
            msg_type="response",
            payload={
                "adv_trajectory": norm_traj,
                "collision_info": collision_info,
                "retry_count": retry_count,
                "max_retry": max_retry,
                "parsed": parsed,
            },
            trace_id=message.trace_id,
        )

    @staticmethod
    def _resolve_max_retry() -> int:
        try:
            return max(0, int(os.getenv("TRAJ_STEP3_MAX_RETRY", "3")))
        except (TypeError, ValueError):
            return 3

    @staticmethod
    def _resolve_enable_retry() -> bool:
        return os.getenv("TRAJ_STEP3_ENABLE_RETRY", "1").strip().lower() in {"1", "true", "yes", "y"}
