from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .crash_data_processor import NHTSA_CODE_NAMES
from .vectorstore import JsonVectorStore, infer_precrash_code_hints


def _prior_weight_hi_lo(weight_max: float, weight_min: float) -> Tuple[float, float]:
    """Return (high, low) with high >= low for linear rank ramps."""
    hi = max(float(weight_max), float(weight_min))
    lo = min(float(weight_max), float(weight_min))
    return hi, lo


def build_nhtsa_type_prior_weights(
    weight_max: float = 1.0,
    weight_min: float = 1.0,
    code_keys_in_rank_order: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Dataset type priors for Layer-1 (``layer=="mode"``): precrash_code → weight.

    Defaults to **uniform** weights (1.0) so dict insertion order does not bias
    retrieval toward code ``"1"``. Pass a frequency-ranked ``code_keys_in_rank_order``
    to re-enable mild priors.
    """
    hi, lo = _prior_weight_hi_lo(weight_max, weight_min)
    keys = (
        list(code_keys_in_rank_order)
        if code_keys_in_rank_order is not None
        else list(NHTSA_CODE_NAMES.keys())
    )
    n = len(keys)
    if n <= 0:
        return {}
    if n == 1:
        return {str(keys[0]): hi}
    span = hi - lo
    return {
        str(k): hi - span * (float(i) / float(n - 1)) for i, k in enumerate(keys)
    }


# Default table (same defaults as ``build_nhtsa_type_prior_weights()``); kept for imports.
NHTSA_TYPE_PRIOR_WEIGHT: Dict[str, float] = build_nhtsa_type_prior_weights()


class NHTSARetriever:
    """Three-layer cascaded NHTSA RAG retriever.

    Layer 1 (Type):    21 NHTSA precrash modes  → coarse type matching
    Layer 2 (Variant): Per-variant text/embed   → mid-level matching
    Layer 3 (Sample):  8s-window control vector → fine kinematic matching
    """

    def __init__(
        self,
        mode_db_path: str = "./out/nhtsa_rag/nhtsa_mode_vector_db.json",
        variant_db_path: str = "./out/nhtsa_rag/nhtsa_variant_vector_db.json",
        sample_db_path: str = "./out/nhtsa_rag/nhtsa_sample_vector_db.json",
        prior_weight_max: float = 1.0,
        prior_weight_min: float = 0.8,
    ) -> None:
        self.mode_db_path = mode_db_path
        self.variant_db_path = variant_db_path
        self.sample_db_path = sample_db_path
        self.prior_weight_max = float(prior_weight_max)
        self.prior_weight_min = float(prior_weight_min)
        self._nhtsa_type_prior_weights = build_nhtsa_type_prior_weights(
            self.prior_weight_max,
            self.prior_weight_min,
        )

        self.mode_store = JsonVectorStore(mode_db_path)
        self.variant_store = JsonVectorStore(variant_db_path)
        self.sample_store = JsonVectorStore(sample_db_path)
        self._loaded = False

    def _build_rank_weights(self, keys_in_order: List[Tuple], weight_min: float, weight_max: float) -> Dict[Tuple, float]:
        """Build reverse-rank weights (top1 highest) in ``[prior_weight_min, prior_weight_max]``."""
        hi, lo = _prior_weight_hi_lo(weight_max, weight_min)
        deduped_keys: List[Tuple] = []
        seen: Set[Tuple] = set()
        for k in keys_in_order:
            if k in seen:
                continue
            seen.add(k)
            deduped_keys.append(k)
        n = len(deduped_keys)
        if n <= 0:
            return {}
        if n == 1:
            return {deduped_keys[0]: hi}
        span = hi - lo
        return {
            k: hi - span * (float(idx) / float(n - 1))
            for idx, k in enumerate(deduped_keys)
        }

    def ensure_built(self) -> None:
        if self._loaded:
            return
        for store in (self.mode_store, self.variant_store, self.sample_store):
            if Path(store.db_path).exists():
                store.load()
        self._loaded = True

    # ------------------------------------------------------------------
    # Three-layer cascaded query
    # ------------------------------------------------------------------

    def query_three_layer(
        self,
        query_text: str,
        top_k_type: int = 6,
        top_k_variant: int = 300,
        top_k_sample: int = 1,
        env: str = "",
        scene_text: str = "",
        behavior_text: str = "",
        metadata_tags: Optional[Iterable[str]] = None,
        interaction_tags: Optional[Iterable[str]] = None,
        query_sequence_vector: Optional[List[float]] = None,
        query_numeric_meta: Optional[Dict] = None,
        query_event_feature_vector: Optional[List[float]] = None,
        use_trajectory: bool = False,
        predicted_ego_future: Optional[List[Tuple[float, float]]] = None,
        adv_current_state: Optional[Dict] = None,
    ) -> List[Dict]:
        """Cascaded retrieval: Type → Variant → Sample.

        Args:
            query_numeric_meta: Dict with numeric fields for Layer 3 state matching,
                e.g. {"ego_v_mean": 12.5, "target_v_mean": 8.0,
                       "ego_current_state": {"heading": 0.1, "speed": 12.5},
                       "target_current_state": {"heading": -0.2, "speed": 8.0}}
            query_event_feature_vector: Fixed-length event-level feature vector
                built by ``vectorstore.build_event_feature_vector`` from an
                ``event_feature`` dict (see ``crash_data_processor.build_event_feature``).
                Used at the variant layer (dominant) and as a weak
                supplementary signal at the sample layer.
        """
        self.ensure_built()

        common_kwargs = dict(
            env=env,
            scene_text=scene_text,
            behavior_text=behavior_text,
            metadata_tags=metadata_tags,
            interaction_tags=interaction_tags,
            use_trajectory=use_trajectory,
        )

        # Layer 1: dataset prior on precrash_code (used only in vectorstore layer=="mode")
        type_prior_weights = self._nhtsa_type_prior_weights

        # Layer 1: Type matching → top precrash_codes
        type_hits = self.mode_store.query(
            query_text=query_text,
            top_k=top_k_type,
            layer="mode",
            type_prior_weights=type_prior_weights,
            **common_kwargs,
        )
        type_order: List[Tuple[str]] = []
        matched_codes: Set[str] = set()
        for h in type_hits:
            code = str(h.get("precrash_code", "")).strip()
            if code:
                matched_codes.add(code)
                type_order.append((code,))
        type_weights = self._build_rank_weights(type_order, 0.95, 1.0)
        for h in type_hits:
            code = str(h.get("precrash_code", "")).strip()
            h["type_weight"] = round(float(type_weights.get((code,), 0.0)), 4) if code else 0.0
        print(f"RAG Retriever: Layer 1 type hits={len(type_hits)}, matched_codes={[c for c, in type_order]}")

        if not matched_codes:
            return []

        # Layer 2: Variant matching, filtered by matched codes
        variant_hits = self.variant_store.query(
            query_text=query_text,
            top_k=top_k_variant,
            filter_codes=matched_codes,
            layer="variant",
            type_score_weights={k[0]: v for k, v in type_weights.items()},
            query_event_feature_vector=query_event_feature_vector,
            **common_kwargs,
        )

        if not variant_hits:
            return []

        # Collect (source_event_id, target_id) pairs for precise Layer 3 filter
        variant_codes: Set[str] = set()
        filter_variants: Set[Tuple[int, int]] = set()
        variant_order: List[Tuple[int, int]] = []
        for h in variant_hits:
            code = str(h.get("precrash_code", "")).strip()
            if code:
                variant_codes.add(code)
            eid = h.get("source_event_id")
            tid = h.get("target_id")
            if eid is not None and tid is not None:
                filter_variants.add((eid, tid))
                variant_order.append((eid, tid))
        variant_weights = self._build_rank_weights(variant_order, 0.95, 1.0)
        for h in variant_hits:
            eid = h.get("source_event_id")
            tid = h.get("target_id")
            if eid is not None and tid is not None:
                h["variant_weight"] = round(float(variant_weights.get((eid, tid), 0.0)), 4)
            else:
                h["variant_weight"] = 0.0
        print(f"RAG Retriever: Layer 2 variant hits={len(variant_hits)}, variant_codes={variant_codes}, filter_variants={filter_variants}")

        # Layer 3: Sample matching, filtered by variant (event_id, target_id)
        sample_hits = self.sample_store.query(
            query_text=query_text,
            top_k=top_k_sample,
            filter_codes=variant_codes,
            filter_variants=filter_variants if filter_variants else None,
            layer="sample",
            query_sequence_vector=query_sequence_vector,
            query_numeric_meta=query_numeric_meta,
            query_event_feature_vector=query_event_feature_vector,
            variant_score_weights=variant_weights,
            predicted_ego_future=predicted_ego_future,
            adv_current_state=adv_current_state,
            **common_kwargs,
        )
        sample_id = str(sample_hits[0].get("id", "")).strip() if sample_hits else "N/A"
        print(f"RAG Retriever: Layer 3 sample hits={len(sample_hits)}, top sample_id={sample_id}")

        # Enrich results with layer info
        for h in sample_hits:
            h["type_hits"] = type_hits
            h["variant_hits"] = variant_hits

        return sample_hits

    def _fetch_layer1_top(
        self,
        query_text: str,
        env: str = "",
        scene_text: str = "",
        behavior_text: str = "",
        metadata_tags: Optional[Iterable[str]] = None,
        interaction_tags: Optional[Iterable[str]] = None,
        use_trajectory: bool = False,
        best_hit: Optional[Dict] = None,
    ) -> Dict:
        """Return Layer-1 top mode and its reference_trajectory (fallback when no sample hit)."""
        type_hits: List[Dict] = []
        if isinstance(best_hit, dict):
            type_hits = list(best_hit.get("type_hits") or [])
        if not type_hits:
            self.ensure_built()
            type_hits = self.mode_store.query(
                query_text=query_text,
                top_k=1,
                layer="mode",
                type_prior_weights=self._nhtsa_type_prior_weights,
                env=env,
                scene_text=scene_text,
                behavior_text=behavior_text,
                metadata_tags=metadata_tags,
                interaction_tags=interaction_tags,
                use_trajectory=use_trajectory,
            )
        if not type_hits:
            return {"top_mode": None, "reference_trajectory": {}}
        top_mode = type_hits[0]
        ref = top_mode.get("reference_trajectory")
        if not ref and isinstance(top_mode.get("metadata"), dict):
            ref = top_mode["metadata"].get("reference_trajectory")
        if not isinstance(ref, dict):
            ref = {}
        return {"top_mode": top_mode, "reference_trajectory": ref}

    # ------------------------------------------------------------------
    # Backward-compatible API
    # ------------------------------------------------------------------

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
        query_numeric_meta: Optional[Dict] = None,
        query_event_feature_vector: Optional[List[float]] = None,
        use_trajectory: bool = False,
        predicted_ego_future: Optional[List[Tuple[float, float]]] = None,
        adv_current_state: Optional[Dict] = None,
    ) -> List[Dict]:
        return self.query_three_layer(
            query_text=query_text,
            top_k_sample=top_k,
            env=env,
            scene_text=scene_text,
            behavior_text=behavior_text,
            metadata_tags=metadata_tags,
            interaction_tags=interaction_tags,
            query_sequence_vector=query_sequence_vector,
            query_numeric_meta=query_numeric_meta,
            query_event_feature_vector=query_event_feature_vector,
            use_trajectory=use_trajectory,
            predicted_ego_future=predicted_ego_future,
            adv_current_state=adv_current_state,
        )

    def retrieve_anchor_with_score(
        self,
        query_text: str,
        env: str = "",
        scene_text: str = "",
        behavior_text: str = "",
        metadata_tags: Optional[Iterable[str]] = None,
        interaction_tags: Optional[Iterable[str]] = None,
        query_sequence_vector: Optional[List[float]] = None,
        query_numeric_meta: Optional[Dict] = None,
        query_event_feature_vector: Optional[List[float]] = None,
        threshold: float = 0.7,
        top_k: int = 1,
        use_trajectory: bool = False,
        predicted_ego_future: Optional[List[Tuple[float, float]]] = None,
        adv_current_state: Optional[Dict] = None,
    ) -> Dict:
        print(
            f"RAG Retriever: query='{query_text}', env='{env}', scene_text_len={len(scene_text)}, "
            f"behavior_text_len={len(behavior_text)}, metadata_tags={metadata_tags}, "
            f"interaction_tags={interaction_tags}, query_sequence_vector_len={len(query_sequence_vector) if query_sequence_vector else 0}, "
            f"query_numeric_meta_keys={list(query_numeric_meta.keys()) if isinstance(query_numeric_meta, dict) else []}, "
            f"query_event_feature_vector_len={len(query_event_feature_vector) if query_event_feature_vector else 0}, "
            f"threshold={threshold}, top_k={top_k}"
        )
        layer1_kwargs = dict(
            query_text=query_text,
            env=env,
            scene_text=scene_text,
            behavior_text=behavior_text,
            metadata_tags=metadata_tags,
            interaction_tags=interaction_tags,
            use_trajectory=use_trajectory,
        )
        hits = self.query(
            query_text=query_text,
            top_k=top_k,
            env=env,
            scene_text=scene_text,
            behavior_text=behavior_text,
            metadata_tags=metadata_tags,
            interaction_tags=interaction_tags,
            query_sequence_vector=query_sequence_vector,
            query_numeric_meta=query_numeric_meta,
            query_event_feature_vector=query_event_feature_vector,
            use_trajectory=use_trajectory,
            predicted_ego_future=predicted_ego_future,
            adv_current_state=adv_current_state,
        )
        if not hits:
            layer1 = self._fetch_layer1_top(**layer1_kwargs)
            return {
                "hit": False,
                "anchor_trajectory": [],
                "similarity_score": 0.0,
                "record_id": "",
                "match": None,
                "top_mode": layer1.get("top_mode"),
                "reference_trajectory": layer1.get("reference_trajectory") or {},
            }
        best = hits[0]
        layer1 = self._fetch_layer1_top(**layer1_kwargs, best_hit=best)
        score = float(best.get("similarity_score", 0.0))
        hit = score >= float(threshold)
        return {
            "hit": hit,
            "anchor_trajectory": best.get("anchor_control", []) if hit else [],
            "similarity_score": score,
            "record_id": str(best.get("record_id") or best.get("id") or ""),
            "match": best if hit else None,
            "top_mode": layer1.get("top_mode"),
            "reference_trajectory": layer1.get("reference_trajectory") or {},
        }
