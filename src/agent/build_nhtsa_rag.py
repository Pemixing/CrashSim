import argparse
import json
from pathlib import Path
from typing import Dict, List

from agent.rag.nhtsa_processor import NHTSAProcessor
from agent.rag.vectorstore import JsonVectorStore, build_control_vector, build_trajectory_vector, build_event_feature_vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_control_sequence(controls: List[Dict], use_control: bool) -> List[Dict]:
    """Normalize control dict keys based on *use_control*.

    - use_control=True  -> {"t","acceleration","steering_angle"}
    - use_control=False -> {"t","delta_s","delta_heading"}
    """
    out: List[Dict] = []
    for i, step in enumerate(controls or []):
        if not isinstance(step, dict):
            step = {}
        t = step.get("t", step.get("time", step.get("timestamp", i)))
        if use_control:
            out.append(
                {
                    "t": _as_float(t, float(i)),
                    "acceleration": _as_float(step.get("acceleration", step.get("acc", step.get("a", 0.0))), 0.0),
                    "steering_angle": _as_float(step.get("steering_angle", step.get("steer", step.get("steering", 0.0))), 0.0),
                }
            )
        else:
            out.append(
                {
                    "t": _as_float(t, float(i)),
                    "delta_s": _as_float(step.get("delta_s", step.get("ds", step.get("delta_speed", 0.0))), 0.0),
                    "delta_heading": _as_float(step.get("delta_heading", step.get("dtheta", step.get("delta_yaw", 0.0))), 0.0),
                }
            )
    return out


def _target_behavior_fragments_from_description(description: str) -> List[str]:
    """Lines from auto-generated descriptions like ``[behavior] target hard braking``."""
    if not description:
        return []
    prefix = "[behavior] target "
    out: List[str] = []
    for raw in str(description).splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            frag = line[len(prefix) :].strip()
            if frag:
                out.append(frag)
    return out


def _trajectory_sequence_xy_only(sequence: List) -> List[List[float]]:
    """Keep only x,y from trajectory steps (drop t and other keys)."""
    out: List[List[float]] = []
    for step in sequence or []:
        if not isinstance(step, dict):
            continue
        out.append([_as_float(step.get("x", 0.0)), _as_float(step.get("y", 0.0))])
    return out


def _reference_trajectory_from_profile(profile: Dict) -> Dict[str, List[List[float]]]:
    """Build reference_trajectory from profile ref_trajectory (xy only)."""
    ref = profile.get("ref_trajectory") or {}
    return {
        "ego_trajectory_sequence": _trajectory_sequence_xy_only(
            ref.get("ego_trajectory_sequence", [])
        ),
        "target_trajectory_sequence": _trajectory_sequence_xy_only(
            ref.get("target_trajectory_sequence", [])
        ),
    }


def _variant_adversarial_maneuver_text(builtin_adv: str, description: str) -> str:
    """Builtin NHTSA adversarial label plus any ``[behavior] target ...`` lines from *description*."""
    extra = _target_behavior_fragments_from_description(description)
    if not extra:
        return str(builtin_adv or "").strip() or "Unknown"
    base = str(builtin_adv or "").strip() or "Unknown"
    return f"{base}; {'; '.join(extra)}"


# ---------------------------------------------------------------------------
# Layer 1 – NHTSA type (16 precrash modes)
# ---------------------------------------------------------------------------

def _mode_vector_entries(processor: NHTSAProcessor, profiles: Dict[str, Dict]) -> list:
    """Layer 1: 16 NHTSA precrash mode entries (class-level text for coarse retrieval)."""
    packed: list = []
    for item in processor.BUILTIN_SCENARIOS[:16]:
        code = str(item.get("precrash_code", "")).strip()
        profile = profiles.get(code, {})
        doc = (
            f"[scenario_type] {item['type']}\n"
            f"[description] {item['description']}\n"
            f"[ego_maneuver] {item['ego_maneuver']}\n"
            f"[adversarial_maneuver] {item['adversarial_maneuver']}\n"
            f"[factors] {', '.join(item.get('factors', []))}"
            f"[environment] {item.get('env', '')}"
        )
        meta = {
            "record_id": item["id"],
            "precrash_code": code,
            "scenario_type": item["type"],
            "rag_layer": "mode",
            "description": item["description"],
            "ego_maneuver": item["ego_maneuver"],
            "adversarial_maneuver": item["adversarial_maneuver"],
            "factors": list(item.get("factors", [])),
            "severity": item.get("severity", "unknown"),
            "environment": item.get("env", ""),
            "anchor_trajectory": [],
            "anchor_control": [],
            "reference_trajectory": _reference_trajectory_from_profile(profile),
        }
        packed.append({"id": f"nhtsa_mode_{code.zfill(2)}", "document": doc, "metadata": meta})
    return packed


# ---------------------------------------------------------------------------
# Layer 2 – Variant (one entry per variant, text + embedding matching)
# ---------------------------------------------------------------------------

def _variant_vector_entries(
    profiles: Dict[str, Dict],
    scenario_map: Dict[str, Dict],
    use_control: bool = False,
) -> list:
    """Layer 2: one entry per crash variant for mid-level retrieval.

    Matches on description, event features.
    """
    packed: list = []
    for code, entry in profiles.items():
        builtin = scenario_map.get(code, {})
        nhtsa_type = entry.get("nhtsa_type", builtin.get("type", "Unknown"))
        ego_maneuver = builtin.get("ego_maneuver", "Unknown")
        adversarial_maneuver = builtin.get("adversarial_maneuver", "Unknown")
        environment = builtin.get("env", "urban daylight clear")
        severity = builtin.get("severity", "unknown")
        factors = list(builtin.get("factors", []))

        for idx, variant in enumerate(entry.get("variants", [])):
            eid = variant.get("source_event_id", idx)
            tid = variant.get("target_id", 0)
            desc = variant.get("description", "")
            adv_for_variant = _variant_adversarial_maneuver_text(adversarial_maneuver, desc)
            mdc = variant.get("impact_timestamp_mdc", None)
            is_crash = variant.get("is_crash", False)
            doc = [
                f"[nhtsa_type] {nhtsa_type}",
                f"[description] {desc}",
                f"[ego_maneuver] {ego_maneuver}",
                f"[adversarial_maneuver] {adv_for_variant}",
            ]

            event_feature = variant.get("event_feature", {})
            event_feature_vector = build_event_feature_vector(event_feature)

            meta = {
                "precrash_code": str(code).strip(),
                "scenario_type": nhtsa_type,
                "rag_layer": "variant",
                "description": desc,
                "ego_maneuver": ego_maneuver,
                "adversarial_maneuver": adv_for_variant,
                "factors": factors,
                "severity": severity,
                "environment": environment,
                "source_event_id": eid,
                "target_id": tid,
                "variant_index": idx,
                "anchor_trajectory": [],
                "anchor_control": [],
                "event_feature": event_feature,
                "impact_timestamp_mdc": mdc,
                "is_crash": is_crash,
            }

            vid = f"variant_{code}_{eid}_t{tid}_{idx}"
            entry_dict: Dict = {"id": vid, "document": doc, "metadata": meta}
            entry_dict["event_feature_vector"] = event_feature_vector
            packed.append(entry_dict)
    return packed


# ---------------------------------------------------------------------------
# Layer 3 – Sample (one entry per 10s sliding window, control-vector matching)
# ---------------------------------------------------------------------------

ANCHOR_STEPS = 12  # last 6s at 0.5s = 12 steps

def _sample_vector_entries(
    profiles: Dict[str, Dict],
    scenario_map: Dict[str, Dict],
    use_control: bool = False,
) -> list:
    """Layer 3: one entry per 10s sample window for fine kinematic matching.

    Each variant's sample_control_sequence contains sliding-window samples.
    For each sample, builds a control vector from the first 2s (4 steps)
    of ego+target controls, and stores the last 6s (12 steps) as anchor.
    """
    packed: list = []
    for code, entry in profiles.items():
        builtin = scenario_map.get(code, {})
        nhtsa_type = entry.get("nhtsa_type", builtin.get("type", "Unknown"))
        ego_maneuver = builtin.get("ego_maneuver", "Unknown")
        adversarial_maneuver = builtin.get("adversarial_maneuver", "Unknown")
        environment = builtin.get("env", "urban daylight clear")

        for v_idx, variant in enumerate(entry.get("variants", [])):
            eid = variant.get("source_event_id", v_idx)
            tid = variant.get("target_id", 0)
            samples = variant.get("sample_control_sequence", [])
            variant_event_feature = variant.get("event_feature", {})
            event_feature = variant.get("event_feature", {})
            event_feature_vector = build_event_feature_vector(event_feature)
            impact_state = variant.get("impact_timestamp_state", {})
            impact_timestamp_mdc = variant.get("impact_timestamp_mdc", None)

            for s_idx, sample in enumerate(samples):
                start_index = sample.get("start_index", s_idx)
                impact_time_index = sample.get("impact_time_index", None)
                stage_time_indices = sample.get("stage_time_indices")
                ego_state = sample.get("ego_current_state", {})
                target_state = sample.get("target_current_state", {})

                desc = sample.get("description", "")
                adv_for_variant = _variant_adversarial_maneuver_text(adversarial_maneuver, desc)

                # Use trajectory vector
                target_traj = sample.get("target_trajectory_sequence", [])
                ego_traj = sample.get("ego_trajectory_sequence", [])

                if not ego_traj and not target_traj:
                    continue

                # Build trajectory vector from first 2s (4 steps)
                tv = build_trajectory_vector(ego_traj, target_traj, n_steps=4)

                n_total = len(target_traj)
                anchor_start = max(0, n_total - ANCHOR_STEPS)
                target_trajectory_history = target_traj[:anchor_start] if anchor_start < n_total else []
                anchor_trajectory = target_traj[anchor_start:]
                
                ego_trajectory_history = ego_traj[:anchor_start] if anchor_start < n_total else []
                
                # Use control vector
                ego_controls = sample.get("ego_control_sequence", [])
                target_controls = sample.get("target_control_sequence", [])
                ego_controls = _normalize_control_sequence(ego_controls, use_control=use_control)
                target_controls = _normalize_control_sequence(target_controls, use_control=use_control)

                if not ego_controls and not target_controls:
                    continue

                # Build normalized control vector:
                #   ego   -> full 16-step window (captures intent + future motion)
                #   target -> first 2s (4 steps) only, matching the query-side window
                cv = build_control_vector(
                    ego_controls,
                    target_controls,
                    use_control=use_control,
                )
                n_total = len(target_controls)
                anchor_start = max(0, n_total - ANCHOR_STEPS)
                target_control_history = target_controls[:anchor_start] if anchor_start < n_total else []
                anchor_control = target_controls[anchor_start:]

                ego_control_history = ego_controls[:anchor_start] if anchor_start < n_total else []

                doc = (
                    f"[nhtsa_type] {nhtsa_type}\n"
                    f"[description] {desc}\n"
                    f"[adversarial_maneuver] {adv_for_variant}\n"
                    f"[ego_current_heading] {ego_state.get('heading', 0.0):.4f} rad\n"
                    f"[ego_current_speed] {ego_state.get('speed', 0.0):.2f} m/s\n"
                    f"[target_current_heading] {target_state.get('heading', 0.0):.4f} rad\n"
                    f"[target_current_speed] {target_state.get('speed', 0.0):.2f} m/s"
                )

                meta = {
                    "precrash_code": str(code).strip(),
                    "scenario_type": nhtsa_type,
                    "rag_layer": "sample",
                    "description": desc,
                    "ego_maneuver": ego_maneuver,
                    "adversarial_maneuver": adv_for_variant,
                    "environment": environment,
                    "source_event_id": eid,
                    "target_id": tid,
                    "variant_index": v_idx,
                    "start_index": start_index,
                    "ego_current_state": ego_state,
                    "target_current_state": target_state,
                    "ego_trajectory_history": ego_trajectory_history,
                    "target_trajectory_history": target_trajectory_history,
                    "ego_control_history": ego_control_history,
                    "target_control_history": target_control_history,
                    "anchor_trajectory": anchor_trajectory,
                    "anchor_control": anchor_control,
                    "event_feature": variant_event_feature,
                    "impact_state": impact_state,
                    "impact_timestamp_mdc": impact_timestamp_mdc,
                    "impact_time_index": impact_time_index,
                    "stage_time_indices": stage_time_indices,
                }

                sid = f"sample_{code}_{eid}_t{tid}_{v_idx}_s{start_index}"
                packed.append({
                    "id": sid,
                    "document": doc,
                    "metadata": meta,
                    "trajectory_vector": tv,
                    "control_vector": cv,
                    "event_feature_vector": event_feature_vector,
                })
    return packed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).parent.parent.parent
    default_out = root_dir / "out" / "nhtsa_rag"

    parser = argparse.ArgumentParser(description="Build NHTSA RAG database (three-layer).")
    parser.add_argument(
        "--output_mode_db", type=str,
        default=str(default_out / "nhtsa_mode_vector_db.json"),
        help="Output JSON vector DB for Layer 1 (21-class type matching).",
    )
    parser.add_argument(
        "--output_variant_db", type=str,
        default=str(default_out / "nhtsa_variant_vector_db.json"),
        help="Output JSON vector DB for Layer 2 (variant matching).",
    )
    parser.add_argument(
        "--output_sample_db", type=str,
        default=str(default_out / "nhtsa_sample_vector_db.json"),
        help="Output JSON vector DB for Layer 3 (sample control-sequence matching).",
    )
    parser.add_argument(
        "--crash_profiles_json", type=str,
        default=str(default_out / "crash_profiles.json"),
        help="Pre-built crash_profiles.json from crash_data_processor.",
    )
    parser.add_argument(
        "--crash_data_dir", type=str, default="",
        help="T7UUC1 crash root. If set and --crash_profiles_json missing, "
             "processes raw data first.",
    )
    parser.add_argument(
        "--export_records_json", type=str,
        default=str(default_out / "nhtsa_records.json"),
        help="Export normalized records for inspection.",
    )
    parser.add_argument(
        "--use_control", action="store_true", default=False,
        help="Use control (acc/steer) sequences instead of delta_s/delta_heading "
             "sequences in variant and sample metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = NHTSAProcessor()

    # --- Load / build crash profiles ---
    crash_profiles_path = (args.crash_profiles_json or "").strip()
    if not Path(crash_profiles_path).is_file() and args.crash_data_dir:
        from agent.rag.crash_data_processor import process_crash_data
        print(f"\nProcessing real crash data from {args.crash_data_dir} ...")
        process_crash_data(
            data_dir=args.crash_data_dir,
            output_dir=str(Path(crash_profiles_path).parent),
        )

    profiles: Dict[str, Dict] = {}
    if crash_profiles_path and Path(crash_profiles_path).is_file():
        with open(crash_profiles_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = data.get("profiles", {})
        total_variants = sum(len(p.get("variants", [])) for p in profiles.values())
        print(f"Loaded crash profiles: {len(profiles)} types, {total_variants} variants")

    if not profiles:
        print("No crash profiles found. Cannot build three-layer RAG.")
        return

    scenario_map = {
        item["precrash_code"]: item for item in processor.BUILTIN_SCENARIOS
    }

    # --- Build Layer 1: Type (Mode) ---
    mode_packed = _mode_vector_entries(processor, profiles)
    mode_db_path = Path(args.output_mode_db)
    mode_db_path.parent.mkdir(parents=True, exist_ok=True)
    mode_store = JsonVectorStore(str(mode_db_path))
    mode_store.build(mode_packed)

    # --- Build Layer 2: Variant ---
    variant_packed = _variant_vector_entries(profiles, scenario_map, use_control=args.use_control)
    variant_db_path = Path(args.output_variant_db)
    variant_store = JsonVectorStore(str(variant_db_path))
    variant_store.build(variant_packed)

    # --- Build Layer 3: Sample ---
    sample_packed = _sample_vector_entries(profiles, scenario_map, use_control=args.use_control)
    sample_db_path = Path(args.output_sample_db)
    sample_store = JsonVectorStore(str(sample_db_path))
    sample_store.build(sample_packed)

    # --- Export normalized records ---
    export_path = Path(args.export_records_json)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    records = processor.load_crash_samples(crash_profiles_path)
    with export_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "record_id": r.record_id,
                    "scenario_type": r.scenario_type,
                    "pre_crash_description": r.pre_crash_description,
                    "ego_maneuver": r.ego_maneuver,
                    "adversarial_maneuver": r.adversarial_maneuver,
                    "contributing_factors": r.contributing_factors,
                    "severity": r.severity,
                    "environment": r.environment,
                    "anchor_trajectory": r.anchor_trajectory,
                    "anchor_control": r.anchor_control,
                    "precrash_code": r.precrash_code,
                }
                for r in records
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --- Summary ---
    print(f"\nBuilt NHTSA RAG (three-layer):")
    print(f"  Layer 1 - Type DB (16 classes):    {mode_db_path}  ({len(mode_packed)} entries)")
    print(f"  Layer 2 - Variant DB:              {variant_db_path}  ({len(variant_packed)} entries)")
    print(f"  Layer 3 - Sample DB (control-vec):  {sample_db_path}  ({len(sample_packed)} entries)")
    print(f"  Exported records:                  {export_path}  ({len(records)} records)")
    print(
        "\nRetrieval: Layer 1 (type) → filter → Layer 2 (variant) → "
        "filter → Layer 3 (sample, first-2s control matching → last-6s anchor)"
    )


if __name__ == "__main__":
    main()
