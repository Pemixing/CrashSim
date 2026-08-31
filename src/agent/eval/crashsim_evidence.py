#!/usr/bin/env python3
"""Deterministic population evidence for post-rollout CrashSim interpretation.

This module consumes completed ``results.jsonl`` records only.  It never runs a
planner or an LLM.  Missing measurements remain explicit instead of being
imputed, and all selection/scoring rules are deterministic.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


FAILURE_TAXONOMY: Tuple[str, ...] = (
    "delayed_or_absent_response",
    "insufficient_longitudinal_mitigation",
    "inadequate_lateral_avoidance",
    "unstable_evasive_maneuver",
    "incorrect_interaction_handling",
    "overly_conservative_behavior",
)

CAPABILITY_DIMENSIONS: Tuple[str, ...] = (
    "risk_anticipation",
    "interaction_reasoning",
    "longitudinal_response",
    "lateral_response",
    "comfort_and_stability",
    "mobility_efficiency",
)

CAPABILITY_LEVELS: Tuple[str, ...] = (
    "critical_weakness",
    "major_weakness",
    "mixed",
    "generally_capable",
    "strong",
)

METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "minimum_clearance_m": ("min_box_separation_margin_m", "min_clearance_proxy_m"),
    "impact_speed_mps": ("ego_speed_at_collision_mps", "relative_speed_at_collision_mps"),
    "braking_onset_ttc_sec": ("brake_onset_ttc_sec",),
    "response_latency_sec": ("response_latency_sec",),
    "minimum_ttc_sec": ("min_ttc_sec",),
    "maximum_deceleration_mps2": ("ego_max_decel_mps2", "ego_max_decel"),
    "mean_absolute_acceleration_mps2": ("ego_mean_abs_accel_mps2",),
    "maximum_jerk_mps3": ("ego_max_abs_jerk_mps3", "ego_max_abs_jerk"),
    "jerk_rms_mps3": ("ego_jerk_rms_mps3",),
    "lateral_acceleration_rms_mps2": ("ego_lateral_accel_rms_mps2",),
    "maximum_lateral_acceleration_mps2": (
        "ego_lateral_accel_max_mps2",
        "ego_lat_accel_max",
    ),
    "route_progress_ratio": (
        "normalized_route_progress",
        "progress_metric_value",
        "ego_displacement_ratio",
        "ego_progress_ratio",
    ),
}

CASE_METRIC_KEYS: Tuple[str, ...] = (
    "collision",
    "near_miss",
    "min_ttc_sec",
    "response_latency_sec",
    "brake_onset_ttc_sec",
    "min_box_separation_margin_m",
    "min_clearance_proxy_m",
    "ego_speed_at_collision_mps",
    "relative_speed_at_collision_mps",
    "ego_max_decel_mps2",
    "ego_max_decel",
    "ego_max_abs_jerk_mps3",
    "ego_max_abs_jerk",
    "ego_lateral_accel_max_mps2",
    "ego_lat_accel_max",
    "progress_metric_value",
    "ego_progress_ratio",
)


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _first(record: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _metric(record: Mapping[str, Any], canonical: str) -> Optional[float]:
    return _finite(_first(record, METRIC_ALIASES[canonical]))


def _collision(record: Mapping[str, Any]) -> bool:
    value = record.get("collision")
    if value is None:
        value = record.get("did_coll", False)
    return bool(value)


def _near_miss(record: Mapping[str, Any]) -> bool:
    value = record.get("near_miss")
    if value is None:
        value = record.get("did_near_crash", False)
    return bool(value) and not _collision(record)


def _interactive(record: Mapping[str, Any]) -> bool:
    if "num_other_agents" in record:
        return int(record.get("num_other_agents") or 0) > 0
    return bool(record.get("has_other_agents", True))


def _distribution(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    unit: Optional[str],
) -> Dict[str, Any]:
    values = [_metric(record, metric) for record in records]
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    result: Dict[str, Any] = {
        "metric": metric,
        "unit": unit,
        "n_available": int(finite.size),
        "n_total": len(records),
        "missing_fraction": (
            float((len(records) - finite.size) / len(records)) if records else None
        ),
        "mean": None,
        "median": None,
        "p10": None,
        "p90": None,
        "min": None,
        "max": None,
    }
    if finite.size:
        result.update(
            mean=float(np.mean(finite)),
            median=float(np.median(finite)),
            p10=float(np.percentile(finite, 10)),
            p90=float(np.percentile(finite, 90)),
            min=float(np.min(finite)),
            max=float(np.max(finite)),
        )
    return result


def _rate(records: Sequence[Mapping[str, Any]], predicate) -> Dict[str, Any]:
    n = len(records)
    count = sum(bool(predicate(record)) for record in records)
    return {
        "count": count,
        "denominator": n,
        "rate": count / n if n else None,
    }


def _resolved_failure_label(record: Mapping[str, Any]) -> Optional[str]:
    raw = str(
        record.get("primary_rule_based_failure_label")
        or record.get("failure_mechanism")
        or ""
    )
    mapping = {
        "no_evident_response": "delayed_or_absent_response",
        "delayed_response": "delayed_or_absent_response",
        "late_or_absent_braking": "delayed_or_absent_response",
        "insufficient_longitudinal_response": "insufficient_longitudinal_mitigation",
        "insufficient_collision_mitigation": "insufficient_longitudinal_mitigation",
        "longitudinal_gap_management": "insufficient_longitudinal_mitigation",
        "inadequate_lateral_avoidance": "inadequate_lateral_avoidance",
        "lateral_conflict_handling": "inadequate_lateral_avoidance",
        "unstable_evasive_maneuver": "unstable_evasive_maneuver",
        "unstable_evasive_response": "unstable_evasive_maneuver",
        "maladaptive_response": "incorrect_interaction_handling",
        "incorrect_interaction_handling": "incorrect_interaction_handling",
        "crossing_conflict_handling": "incorrect_interaction_handling",
        "acceleration_into_conflict": "incorrect_interaction_handling",
        "overly_conservative_response": "overly_conservative_behavior",
        "overly_conservative_behavior": "overly_conservative_behavior",
    }
    return mapping.get(raw)


def _frequency(values: Iterable[Any], *, missing_label: str = "missing") -> Dict[str, Any]:
    normalized = [
        str(value).strip() if value not in (None, "") else missing_label for value in values
    ]
    counts = Counter(normalized)
    n = len(normalized)
    return {
        "n_total": n,
        "counts": dict(sorted(counts.items())),
        "fractions": {
            key: count / n if n else None for key, count in sorted(counts.items())
        },
    }


def build_population_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    all_records = list(records)
    interactive = [record for record in all_records if _interactive(record)]
    collisions = [record for record in interactive if _collision(record)]
    response_records = [
        record for record in interactive
        if record.get("risk_onset_time_sec") is not None
    ]

    progress = _distribution(interactive, "route_progress_ratio", unit=None)
    completion_available = [
        record for record in interactive if _metric(record, "route_progress_ratio") is not None
    ]
    completion = _rate(
        completion_available,
        lambda record: (_metric(record, "route_progress_ratio") or 0.0) >= 0.9,
    )
    completion["definition"] = "route_progress_ratio >= 0.9"

    return {
        "scope": {
            "n_records": len(all_records),
            "n_interactive": len(interactive),
            "n_collision": len(collisions),
            "n_response_evaluable": len(response_records),
        },
        "safety": {
            "collision_rate": _rate(interactive, _collision),
            "near_miss_rate": _rate(interactive, _near_miss),
            "minimum_clearance": _distribution(
                interactive, "minimum_clearance_m", unit="m"
            ),
            "impact_speed": _distribution(
                collisions, "impact_speed_mps", unit="m/s"
            ),
        },
        "response_behavior": {
            "braking_onset_timing": _distribution(
                collisions, "braking_onset_ttc_sec", unit="s"
            ),
            "response_latency": _distribution(
                response_records, "response_latency_sec", unit="s"
            ),
            "minimum_ttc": _distribution(
                interactive, "minimum_ttc_sec", unit="s"
            ),
            "maximum_deceleration": _distribution(
                interactive, "maximum_deceleration_mps2", unit="m/s^2"
            ),
            "mean_absolute_acceleration": _distribution(
                interactive, "mean_absolute_acceleration_mps2", unit="m/s^2"
            ),
        },
        "comfort": {
            "maximum_jerk": _distribution(
                interactive, "maximum_jerk_mps3", unit="m/s^3"
            ),
            "acceleration_variation": _distribution(
                interactive, "jerk_rms_mps3", unit="m/s^3"
            ),
            "trajectory_stability": _distribution(
                interactive, "lateral_acceleration_rms_mps2", unit="m/s^2"
            ),
            "maximum_lateral_acceleration": _distribution(
                interactive, "maximum_lateral_acceleration_mps2", unit="m/s^2"
            ),
        },
        "efficiency": {
            "route_progress": progress,
            "completion_rate": completion,
            "route_progress_note": (
                "Uses the available displacement/route-progress proxy; true map-route "
                "completion is unavailable when normalized_route_progress is missing."
            ),
        },
    }


def build_failure_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    all_records = list(records)
    labels = [_resolved_failure_label(record) for record in all_records]
    counts = Counter(label for label in labels if label in FAILURE_TAXONOMY)
    classified = sum(counts.values())
    adverse = [
        record for record in all_records
        if _collision(record) or _near_miss(record)
    ]
    n_adverse = len(adverse)
    adverse_labels = [_resolved_failure_label(record) for record in adverse]
    adverse_counts = Counter(label for label in adverse_labels if label in FAILURE_TAXONOMY)
    return {
        "taxonomy_version": "paper-six-pattern-v1",
        "taxonomy": list(FAILURE_TAXONOMY),
        "definitions": {
            "delayed_or_absent_response": "No appropriate response, or response delayed after detected risk.",
            "insufficient_longitudinal_mitigation": "Speed regulation did not sufficiently reduce collision severity.",
            "inadequate_lateral_avoidance": "Available lateral response did not provide effective clearance.",
            "unstable_evasive_maneuver": "Avoidance response exhibited excessive jerk or lateral instability.",
            "incorrect_interaction_handling": "Response was maladaptive for the surrounding-agent interaction.",
            "overly_conservative_behavior": "Progress was unnecessarily sacrificed without an adverse interaction.",
        },
        "n_records": len(all_records),
        "n_classified": classified,
        "n_adverse": n_adverse,
        "unclassified_or_non_failure": len(all_records) - classified,
        "patterns": {
            label: {
                "count": counts.get(label, 0),
                "population_rate": counts.get(label, 0) / len(all_records)
                if all_records else None,
                "classified_failure_share": counts.get(label, 0) / classified
                if classified else None,
                # Among collisions/near-misses only — avoids diluting sparse exclusive labels
                # across the full population of mostly non-failure scenes.
                "adverse_rate": adverse_counts.get(label, 0) / n_adverse
                if n_adverse else None,
            }
            for label in FAILURE_TAXONOMY
        },
    }


def build_scenario_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    all_records = list(records)
    density_values = []
    speed_values = []
    difficulty_values = []
    for record in all_records:
        density_values.append(
            record.get("odd_density_bin")
            or (
                "dense" if int(record.get("num_other_agents") or record.get("NA") or 1) >= 8
                else "medium" if int(record.get("num_other_agents") or record.get("NA") or 1) >= 4
                else "sparse"
            )
        )
        speed_values.append(record.get("odd_speed_bin") or "missing")
        difficulty_values.append(
            record.get("scenario_difficulty")
            or record.get("odd_difficulty")
            or (
                "high" if _collision(record)
                else "medium" if _near_miss(record)
                else "low"
            )
        )
    return {
        "crash_category_distribution": _frequency(
            record.get("crash_category_high")
            or record.get("crash_type_fine")
            or record.get("behavior_tag")
            for record in all_records
        ),
        "interaction_density_distribution": _frequency(density_values),
        "ego_speed_range_distribution": _frequency(speed_values),
        "scenario_difficulty_distribution": _frequency(difficulty_values),
        "difficulty_definition": (
            "Uses supplied scenario_difficulty/odd_difficulty; otherwise deterministic "
            "outcome proxy: collision=high, near_miss=medium, safe=low."
        ),
    }


def _stratum(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(
            record.get("crash_category_high")
            or record.get("behavior_tag")
            or "unknown_category"
        ),
        str(_resolved_failure_label(record) or "no_failure"),
        str(record.get("odd_density_bin") or "unknown_density"),
    )


def _severity(record: Mapping[str, Any]) -> Tuple[float, ...]:
    impact = _metric(record, "impact_speed_mps") or 0.0
    clearance = _metric(record, "minimum_clearance_m")
    jerk = _metric(record, "maximum_jerk_mps3") or 0.0
    return (
        2.0 if _collision(record) else 1.0 if _near_miss(record) else 0.0,
        impact,
        -(clearance if clearance is not None else 1e6),
        jerk,
    )


def _typical_distance(
    record: Mapping[str, Any], medians: Mapping[str, Optional[float]]
) -> float:
    total = 0.0
    used = 0
    for metric, median in medians.items():
        value = _metric(record, metric)
        if value is None or median is None:
            continue
        scale = max(abs(median), 1.0)
        total += abs(value - median) / scale
        used += 1
    return total / used if used else 0.0


def _round_robin(
    buckets: Mapping[Tuple[str, str, str], Sequence[Mapping[str, Any]]],
    limit: int,
) -> List[Mapping[str, Any]]:
    ordered = {key: list(value) for key, value in sorted(buckets.items())}
    selected: List[Mapping[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in ordered:
            if depth < len(ordered[key]):
                selected.append(ordered[key][depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def select_representative_cases(
    records: Sequence[Mapping[str, Any]], per_group: int = 8
) -> Dict[str, Any]:
    """Select diverse typical, severe and successful cases without random sampling."""
    all_records = list(records)
    failures = [
        record for record in all_records
        if _resolved_failure_label(record) in FAILURE_TAXONOMY
        and (_collision(record) or _near_miss(record)
             or _resolved_failure_label(record) == "overly_conservative_behavior")
    ]
    successful = [
        record for record in all_records
        if not _collision(record)
        and (
            _near_miss(record)
            or str(record.get("primary_rule_based_failure_label") or "")
            in ("safe_resolution", "near_miss_resolution")
            or (
                record.get("risk_onset_time_sec") is not None
                and record.get("response_onset_time_sec") is not None
            )
        )
    ]

    typical_buckets: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in failures:
        typical_buckets[_stratum(record)].append(record)
    typical_ranked: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    typical_metrics = (
        "minimum_clearance_m",
        "impact_speed_mps",
        "response_latency_sec",
        "maximum_jerk_mps3",
    )
    for key, group in typical_buckets.items():
        medians = {
            metric: (
                float(np.median(values)) if (
                    values := [
                        value for item in group
                        if (value := _metric(item, metric)) is not None
                    ]
                ) else None
            )
            for metric in typical_metrics
        }
        typical_ranked[key] = sorted(
            group,
            key=lambda item: (
                _typical_distance(item, medians),
                str(item.get("scene_token") or item.get("scene_name") or ""),
                int(item.get("sidx") or 0),
            ),
        )

    severe_buckets: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in failures:
        severe_buckets[_stratum(record)].append(record)
    for key in severe_buckets:
        severe_buckets[key].sort(key=_severity, reverse=True)

    success_buckets: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for record in successful:
        success_buckets[_stratum(record)].append(record)
    for key in success_buckets:
        success_buckets[key].sort(
            key=lambda item: (
                1 if _near_miss(item) else 0,
                _metric(item, "minimum_clearance_m") or 0.0,
                -(_metric(item, "response_latency_sec") or 0.0),
            ),
            reverse=True,
        )

    groups = {
        "typical_failure": _round_robin(typical_ranked, per_group),
        "severe_failure": _round_robin(severe_buckets, per_group),
        "successful_avoidance": _round_robin(success_buckets, per_group),
    }
    return {
        **groups,
        "selection_method": {
            "typical_failure": "closest-to-stratum-median, round-robin across category × mechanism × density",
            "severe_failure": "severity-ranked, round-robin across category × mechanism × density",
            "successful_avoidance": "collision-free risk/near-miss resolutions, round-robin across strata",
            "per_group_cap": per_group,
            "deterministic": True,
        },
    }


def compact_case(record: Mapping[str, Any], case_id: str, case_type: str) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "case_type": case_type,
        "scene_key": {
            "scene_token": record.get("scene_token"),
            "scene_name": record.get("scene_name"),
            "sidx": record.get("sidx"),
        },
        "crash_category": (
            record.get("crash_category_high")
            or record.get("crash_type_fine")
            or record.get("behavior_tag")
        ),
        "interaction_condition": {
            "density": record.get("odd_density_bin"),
            "speed": record.get("odd_speed_bin"),
            "difficulty": record.get("scenario_difficulty") or record.get("odd_difficulty"),
        },
        "outcome": (
            "collision" if _collision(record)
            else "near_miss" if _near_miss(record)
            else "successful_avoidance"
        ),
        "failure_mechanism": _resolved_failure_label(record),
        "metrics": {key: record.get(key) for key in CASE_METRIC_KEYS if key in record},
        "data_quality_flags": list(record.get("data_quality_flags") or []),
    }


def _clip_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _higher_better(value: Optional[float], bad: float, good: float) -> Optional[float]:
    if value is None or math.isclose(bad, good):
        return None
    return _clip_score(100.0 * (value - bad) / (good - bad))


def _lower_better(value: Optional[float], good: float, bad: float) -> Optional[float]:
    if value is None or math.isclose(good, bad):
        return None
    return _clip_score(100.0 * (bad - value) / (bad - good))


def _path_value(root: Mapping[str, Any], path: str) -> Optional[float]:
    value: Any = root
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _finite(value)


def _score_to_level(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score < 20:
        return "critical_weakness"
    if score < 40:
        return "major_weakness"
    if score < 60:
        return "mixed"
    if score < 80:
        return "generally_capable"
    return "strong"


def _collision_outcome_ceiling(collision_rate: Optional[float]) -> Optional[str]:
    """Cap optimistic interaction/lateral levels when overall collision risk is high.

    Exclusive primary-label pattern rates are often sparse even when many scenes collide.
    This ceiling keeps interaction/lateral from reading as ``strong`` solely because
    few scenes were labeled with that specific failure pattern.
    """
    if collision_rate is None:
        return None
    if collision_rate >= 0.50:
        return "major_weakness"
    if collision_rate >= 0.25:
        return "mixed"
    if collision_rate >= 0.12:
        return "generally_capable"
    return None


def _stricter_level(level_a: Optional[str], level_b: Optional[str]) -> Optional[str]:
    if level_a is None:
        return level_b
    if level_b is None:
        return level_a
    rank = {level: index for index, level in enumerate(CAPABILITY_LEVELS)}
    return level_a if rank[level_a] <= rank[level_b] else level_b


def _extract_risk_anticipation_signals(
    population: Mapping[str, Any],
    failures: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    safety = population.get("safety") if isinstance(population, Mapping) else None
    response = population.get("response_behavior") if isinstance(population, Mapping) else None
    patterns = failures.get("patterns") if isinstance(failures, Mapping) else None
    collision = None
    if isinstance(safety, Mapping):
        collision = _finite((safety.get("collision_rate") or {}).get("rate"))
    delayed = None
    if isinstance(patterns, Mapping):
        delayed = _finite(
            ((patterns.get("delayed_or_absent_response") or {}).get("adverse_rate"))
        )
    ttc = None
    if isinstance(response, Mapping):
        ttc = _finite((response.get("minimum_ttc") or {}).get("median"))
    return {
        "collision_rate": collision,
        "delayed_or_absent_response_adverse_rate": delayed,
        "minimum_ttc_median": ttc,
    }


def risk_anticipation_joint_magnitude_level(
    population: Mapping[str, Any],
    failures: Mapping[str, Any],
) -> Optional[str]:
    """Map joint anticipation magnitudes to a soft ordinal band.

    Absolute hard-set collision alone must not collapse every planner to
    ``critical_weakness``. Critical requires anticipation-mode collapse
    (elevated delayed/absent adverse share) jointly with high collision.
    """
    signals = _extract_risk_anticipation_signals(population, failures)
    collision = signals["collision_rate"]
    delayed = signals["delayed_or_absent_response_adverse_rate"]
    ttc = signals["minimum_ttc_median"]
    if collision is None:
        return None
    delayed_v = 0.0 if delayed is None else float(delayed)

    if (collision >= 0.55 and delayed_v >= 0.35) or (collision >= 0.70 and delayed_v >= 0.25):
        return "critical_weakness"
    if collision >= 0.50 or delayed_v >= 0.30:
        return "major_weakness"
    if collision >= 0.25 or delayed_v >= 0.20:
        if ttc is not None and ttc <= 0.9 and collision >= 0.30:
            return "major_weakness"
        return "mixed"
    if collision >= 0.12:
        return "generally_capable"
    return "strong"


def build_risk_anticipation_discrimination(
    population: Mapping[str, Any],
    failures: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prompt-time hard-set discrimination card (advisory; never overwrites LLM levels).

    On crash/hard sets, elevated collision is common. This card steers the LLM to
    discriminate risk anticipation via delayed/absent response and TTC rather than
    mapping every high collision_rate to ``critical_weakness``.
    """
    signals = _extract_risk_anticipation_signals(population, failures)
    collision = signals["collision_rate"]
    if collision is None:
        set_regime = "unknown"
    elif collision >= 0.25:
        set_regime = "hard_set_elevated_collision"
    elif collision >= 0.12:
        set_regime = "moderate_collision"
    else:
        set_regime = "low_collision"
    soft_band = risk_anticipation_joint_magnitude_level(population, failures)
    return {
        "role": (
            "prompt-time hard-set discrimination aid only — not an answer key; "
            "do not treat soft_advisory_band as a level to copy; never used to "
            "overwrite the LLM level post hoc"
        ),
        "set_regime": set_regime,
        "signals": signals,
        "soft_advisory_band": soft_band,
        "primary_discriminators_when_collision_elevated": [
            "failure_statistics.patterns.delayed_or_absent_response.adverse_rate",
            "population_statistics.response_behavior.minimum_ttc.median",
        ],
        "critical_weakness_requires": (
            "anticipation-specific collapse: jointly high collision AND elevated "
            "delayed_or_absent_response.adverse_rate "
            "(e.g. collision≥0.55 with delayed≥0.35, or collision≥0.70 with delayed≥0.25); "
            "elevated collision alone is insufficient"
        ),
        "discrimination_rules": [
            (
                "On hard/crash sets with elevated collision_rate, do NOT map collision "
                "alone to critical_weakness for risk_anticipation."
            ),
            (
                "Prefer delayed_or_absent_response.adverse_rate and minimum_ttc.median "
                "as primary discriminators when collision is elevated across the set."
            ),
            (
                "soft_advisory_band is a joint-magnitude check only; synthesize the full "
                "package and cite concrete evidence families."
            ),
        ],
    }


def build_capability_reference(
    population: Mapping[str, Any],
    failures: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create a transparent population heuristic prior for LLM audits.

    This is intentionally a *prior*, not an authoritative capability verdict.
    Outcome metrics dominate safety-related dimensions so that sparse exclusive
    failure-pattern labels cannot inflate interaction/lateral to ``strong``.
    For ``risk_anticipation``, the reference level uses a joint magnitude band
    (collision × delayed/absent × TTC) so hard-set absolute collision does not
    collapse every planner to ``critical_weakness``.
    """
    # Each signal: (path, direction, good, bad, weight)
    # For direction=="higher", good > bad; for "lower", good < bad.
    # risk_anticipation: collision weight lowered; delayed/TTC carry discrimination.
    specs: Dict[str, List[Tuple[str, str, float, float, float]]] = {
        "risk_anticipation": [
            ("population_statistics.safety.collision_rate.rate", "lower", 0.05, 0.60, 0.25),
            ("failure_statistics.patterns.delayed_or_absent_response.adverse_rate", "lower", 0.05, 0.55, 0.45),
            ("population_statistics.response_behavior.minimum_ttc.median", "higher", 3.0, 0.5, 0.30),
        ],
        "interaction_reasoning": [
            # Outcome-first: collision + clearance dominate; pattern is only a secondary signal.
            ("population_statistics.safety.collision_rate.rate", "lower", 0.05, 0.60, 0.45),
            ("population_statistics.safety.minimum_clearance.median", "higher", 2.0, 0.0, 0.30),
            ("failure_statistics.patterns.incorrect_interaction_handling.adverse_rate", "lower", 0.05, 0.45, 0.25),
        ],
        "longitudinal_response": [
            ("population_statistics.safety.impact_speed.median", "lower", 0.5, 6.0, 0.35),
            ("population_statistics.response_behavior.response_latency.median", "lower", 0.3, 2.0, 0.30),
            (
                "failure_statistics.patterns.insufficient_longitudinal_mitigation.adverse_rate",
                "lower", 0.05, 0.50, 0.35,
            ),
        ],
        "lateral_response": [
            ("population_statistics.safety.minimum_clearance.median", "higher", 2.0, 0.0, 0.40),
            ("population_statistics.safety.collision_rate.rate", "lower", 0.05, 0.60, 0.35),
            ("failure_statistics.patterns.inadequate_lateral_avoidance.adverse_rate", "lower", 0.05, 0.45, 0.25),
        ],
        "comfort_and_stability": [
            ("population_statistics.comfort.maximum_jerk.p90", "lower", 2.0, 8.0, 0.40),
            ("population_statistics.comfort.maximum_lateral_acceleration.p90", "lower", 1.5, 4.0, 0.35),
            ("failure_statistics.patterns.unstable_evasive_maneuver.population_rate", "lower", 0.01, 0.25, 0.25),
        ],
        "mobility_efficiency": [
            ("population_statistics.efficiency.route_progress.mean", "higher", 1.0, 0.4, 0.40),
            ("population_statistics.efficiency.completion_rate.rate", "higher", 0.9, 0.3, 0.35),
            ("failure_statistics.patterns.overly_conservative_behavior.population_rate", "lower", 0.01, 0.25, 0.25),
        ],
    }
    # Dimensions where exclusive-label sparsity previously inflated levels despite high collisions.
    outcome_ceiling_dims = {"interaction_reasoning", "lateral_response"}

    root = {
        "population_statistics": population,
        "failure_statistics": failures,
    }
    collision_rate = _path_value(root, "population_statistics.safety.collision_rate.rate")
    ceiling = _collision_outcome_ceiling(collision_rate)

    result: Dict[str, Any] = {}
    for dimension, signals in specs.items():
        components = []
        weighted_sum = 0.0
        available_weight = 0.0
        for path, direction, good, bad, weight in signals:
            value = _path_value(root, path)
            score = (
                _lower_better(value, good, bad)
                if direction == "lower"
                else _higher_better(value, bad, good)
            )
            components.append(
                {
                    "evidence_id": path,
                    "value": value,
                    "direction": f"{direction}_is_better",
                    "good_anchor": good,
                    "bad_anchor": bad,
                    "weight": weight,
                    "normalized_score": score,
                }
            )
            if score is not None:
                weighted_sum += score * weight
                available_weight += weight
        score = weighted_sum / available_weight if available_weight >= 0.5 else None
        level = _score_to_level(score)
        applied_ceiling = None
        level_source = "weighted_score_thresholds"
        if dimension == "risk_anticipation":
            joint_level = risk_anticipation_joint_magnitude_level(population, failures)
            if joint_level is not None:
                level = joint_level
                level_source = "joint_magnitude_band"
        if dimension in outcome_ceiling_dims and ceiling is not None:
            capped = _stricter_level(level, ceiling)
            if capped != level:
                applied_ceiling = ceiling
            level = capped
            if applied_ceiling is not None:
                level_source = "weighted_score_with_collision_ceiling"
        result[dimension] = {
            "population_reference_score": score,
            "reference_level": level,
            "role": "deterministic_heuristic_prior_only",
            "level_source": level_source,
            "evidence_coverage": available_weight,
            "components": components,
            "outcome_ceiling_applied": applied_ceiling,
            "level_thresholds": {
                "critical_weakness": "[0,20)",
                "major_weakness": "[20,40)",
                "mixed": "[40,60)",
                "generally_capable": "[60,80)",
                "strong": "[80,100]",
            },
        }
    return result


def build_structured_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    representative_per_group: int = 8,
) -> Dict[str, Any]:
    population = build_population_statistics(records)
    failures = build_failure_statistics(records)
    scenarios = build_scenario_statistics(records)
    representatives = select_representative_cases(records, representative_per_group)
    return {
        "population_statistics": population,
        "failure_statistics": failures,
        "scenario_statistics": scenarios,
        "representative_cases": representatives,
        "capability_population_reference": build_capability_reference(
            population, failures
        ),
        "missing_fields": {
            "population_metrics": sorted(
                {
                    item["metric"]
                    for section in (
                        population["safety"],
                        population["response_behavior"],
                        population["comfort"],
                        population["efficiency"],
                    )
                    for item in section.values()
                    if isinstance(item, dict)
                    and "n_available" in item
                    and item["n_available"] == 0
                }
            ),
            "successful_avoidance_cases": (
                [] if representatives["successful_avoidance"]
                else ["No collision-free risk/near-miss resolution matched selection rules."]
            ),
        },
    }
