from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SceneContext:
    sample_id: str
    bev_image_path: Optional[str] = None
    scene_text_path: Optional[str] = None
    scene_text: Optional[str] = None

    scene_understanding: Optional[str] = None
    ego_driving_intention: Optional[str] = None
    adversarial_vehicle_ids: List[int] = field(default_factory=list)
    spatial_relationships_with_ego: List[str] = field(default_factory=list)
    potential_adversarial_behaviors: List[str] = field(default_factory=list)

    retrieved_anchor: Dict = field(default_factory=dict)
    retrieved_mode: Dict = field(default_factory=dict)
    reference_trajectory: Dict = field(default_factory=dict)
    retrieval_similarity: Optional[float] = None
    anchor_source: Optional[str] = None
    anchor_confidence: Optional[float] = None
    behavior_tag: Optional[str] = None

    adversarial_vehicle_id: Optional[int] = None
    anchor_trajectory_6s: Optional[List[Dict]] = None


class SharedMemory:
    def __init__(self) -> None:
        self._contexts: Dict[str, SceneContext] = {}

    def init_context(self, sample_id: str) -> SceneContext:
        ctx = SceneContext(sample_id=sample_id)
        self._contexts[sample_id] = ctx
        return ctx

    def get_context(self, sample_id: str) -> SceneContext:
        if sample_id not in self._contexts:
            raise KeyError(f"sample_id not initialized in memory: {sample_id}")
        return self._contexts[sample_id]

    def set_context(self, ctx: SceneContext) -> None:
        self._contexts[ctx.sample_id] = ctx
