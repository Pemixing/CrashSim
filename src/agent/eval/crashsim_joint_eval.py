#!/usr/bin/env python3
"""Pair nuScenes / nuCrash planner runs and compute a cross-dataset joint analysis.

This module does not generate figure captions, figure-facing prose, or figure
image assets. It discovers run directories, pairs them by (planner, seed),
compares deterministic evaluation reports, and writes JSON/markdown analysis.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agent.eval.crashsim_evidence import (
    CAPABILITY_DIMENSIONS,
    CAPABILITY_LEVELS,
    FAILURE_TAXONOMY,
)

DATASET_KINDS = ("nuScenes", "nuCrash")
_LEVEL_RANK = {level: index for index, level in enumerate(CAPABILITY_LEVELS)}

_RUN_DIR_PATTERNS = (
    re.compile(
        r"^(?P<kind>nuScenes|nuCrash)__(?P<planner>.+)__seed(?P<seed>-?\d+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<kind>nuScenes|nuCrash)__(?P<planner>.+)__(?P<seed>-?\d+)$",
        re.IGNORECASE,
    ),
)


def _norm_kind(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered in {"nuscenes", "nusc", "nu-scenes"}:
        return "nuScenes"
    if lowered in {"nucrash", "crash", "nu-crash", "nusc-crash"}:
        return "nuCrash"
    if "crash" in lowered:
        return "nuCrash"
    if "nusc" in lowered:
        return "nuScenes"
    return None


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def run_artifact_paths(run_dir: str) -> Dict[str, str]:
    run_dir = os.path.abspath(run_dir)
    return {
        "results": os.path.join(run_dir, "results.jsonl"),
        "deterministic_evaluation": os.path.join(run_dir, "interpretable_eval.json"),
        "llm_analysis": os.path.join(run_dir, "llm_interpretable_analysis.json"),
        "evidence_package": os.path.join(run_dir, "llm_evidence_package.json"),
    }


def valid_results_jsonl(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def deterministic_report_is_valid(
    report: Any,
    *,
    expected_evidence_version: Optional[str] = None,
) -> bool:
    if not isinstance(report, dict):
        return False
    if str(report.get("schema_version")) != "3.0":
        return False
    if expected_evidence_version:
        if str(report.get("deterministic_evidence_version") or "") != expected_evidence_version:
            return False
    required = (
        "population_statistics",
        "failure_statistics",
        "scenario_statistics",
        "capability_population_reference",
        "overall",
    )
    return all(report.get(key) for key in required)


def llm_analysis_is_valid(bundle: Any) -> bool:
    if not isinstance(bundle, dict):
        return False
    analysis = bundle.get("failure_analysis")
    guidance = bundle.get("improvement_guidance")
    if not isinstance(analysis, dict) or not analysis:
        return False
    if isinstance(guidance, list):
        return True
    if isinstance(guidance, dict) and (
        guidance.get("improvement_guidance") or guidance.get("prioritized_guidance")
    ):
        return True
    return False


def mark_stage(
    stages: Dict[str, Any],
    name: str,
    status: str,
    *,
    path: Optional[str] = None,
    reason: str = "",
) -> None:
    stages[name] = {"status": status, "path": path, "reason": reason}


def inspect_run_stages(
    run_record: Mapping[str, Any],
    *,
    expected_evidence_version: Optional[str] = None,
) -> Dict[str, Any]:
    paths = run_record.get("paths") or {}
    stages: Dict[str, Any] = {}
    results_path = str(paths.get("results") or "")
    if valid_results_jsonl(results_path):
        mark_stage(stages, "rollout", "present", path=results_path, reason="results.jsonl present")
    else:
        mark_stage(
            stages,
            "rollout",
            "missing",
            path=results_path or None,
            reason="results.jsonl missing/invalid",
        )

    det_path = str(paths.get("deterministic_evaluation") or "")
    report = None
    if os.path.isfile(det_path):
        try:
            report = _load_json(det_path)
        except (OSError, json.JSONDecodeError):
            report = None
    if deterministic_report_is_valid(report, expected_evidence_version=expected_evidence_version):
        mark_stage(
            stages,
            "deterministic_evidence",
            "present",
            path=det_path,
            reason="valid interpretable_eval.json present",
        )
    else:
        mark_stage(
            stages,
            "deterministic_evidence",
            "missing",
            path=det_path or None,
            reason="interpretable_eval.json missing/invalid",
        )

    llm_path = str(paths.get("llm_analysis") or "")
    bundle = None
    if os.path.isfile(llm_path):
        try:
            bundle = _load_json(llm_path)
        except (OSError, json.JSONDecodeError):
            bundle = None
    if llm_analysis_is_valid(bundle):
        mark_stage(
            stages,
            "single_dataset_llm",
            "present",
            path=llm_path,
            reason="valid llm_interpretable_analysis.json present",
        )
    else:
        mark_stage(
            stages,
            "single_dataset_llm",
            "missing",
            path=llm_path or None,
            reason="llm_interpretable_analysis.json missing/invalid",
        )

    ready = all(
        (stages.get(name) or {}).get("status") == "present"
        for name in ("rollout", "deterministic_evidence", "single_dataset_llm")
    )
    return {"stages": stages, "ready_for_joint": ready}


def _peek_results_meta(results_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if not os.path.isfile(results_path):
        return meta
    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return meta
                if row.get("planner"):
                    meta["planner"] = str(row.get("planner"))
                if row.get("seed") is not None:
                    meta["seed"] = int(row["seed"])
                kind = _norm_kind(row.get("dataset_kind") or row.get("dataset_version"))
                if kind:
                    meta["dataset_kind"] = kind
                version = str(row.get("dataset_version") or "")
                if kind is None and version:
                    meta["dataset_kind"] = "nuCrash" if "crash" in version.lower() else "nuScenes"
                return meta
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return meta
    return meta


def _parse_run_identity(run_dir: str, paths: Mapping[str, str]) -> Dict[str, Any]:
    name = os.path.basename(os.path.abspath(run_dir))
    identity: Dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "run_id": name,
        "planner": None,
        "seed": None,
        "dataset_kind": None,
        "paths": dict(paths),
    }
    for pattern in _RUN_DIR_PATTERNS:
        match = pattern.match(name)
        if match:
            identity["dataset_kind"] = _norm_kind(match.group("kind"))
            identity["planner"] = str(match.group("planner")).strip()
            identity["seed"] = int(match.group("seed"))
            break

    det_path = paths.get("deterministic_evaluation") or ""
    if os.path.isfile(det_path):
        try:
            report = _load_json(det_path)
            context = report.get("context") if isinstance(report, dict) else {}
            if isinstance(context, dict):
                if not identity["planner"] and context.get("planner"):
                    identity["planner"] = str(context.get("planner"))
                kind = _norm_kind(context.get("dataset_kind") or context.get("version"))
                if identity["dataset_kind"] is None and kind:
                    identity["dataset_kind"] = kind
                if identity["seed"] is None and context.get("seed") is not None:
                    identity["seed"] = int(context["seed"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    peeked = _peek_results_meta(paths.get("results") or "")
    for key in ("planner", "seed", "dataset_kind"):
        if identity.get(key) is None and peeked.get(key) is not None:
            identity[key] = peeked[key]
    if identity["seed"] is None:
        identity["seed"] = 0
    return identity


def _discover_run_dirs(runs_dir: str) -> List[str]:
    found: List[str] = []
    skip = {"comparison", "logs", "figures"}
    for root, dirnames, filenames in os.walk(runs_dir):
        dirnames[:] = [name for name in dirnames if name not in skip and not name.startswith(".")]
        rel = os.path.relpath(root, runs_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 3:
            dirnames[:] = []
            continue
        if "results.jsonl" in filenames:
            found.append(root)
            dirnames[:] = []
    return sorted(found)


def pair_runs_by_planner_seed(runs_dir: str) -> Dict[str, Any]:
    runs_dir = os.path.abspath(runs_dir)
    grouped: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    skipped: List[Dict[str, Any]] = []
    for run_dir in _discover_run_dirs(runs_dir):
        identity = _parse_run_identity(run_dir, run_artifact_paths(run_dir))
        planner = identity.get("planner")
        kind = identity.get("dataset_kind")
        seed = identity.get("seed")
        if not planner or kind not in DATASET_KINDS or seed is None:
            skipped.append(
                {
                    "run_dir": run_dir,
                    "reason": "could not resolve planner/dataset_kind/seed",
                    "parsed": identity,
                }
            )
            continue
        bucket = grouped.setdefault((str(planner), int(seed)), {})
        bucket[kind] = identity

    complete: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []
    for (planner, seed), by_kind in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        missing = [kind for kind in DATASET_KINDS if kind not in by_kind]
        if missing:
            incomplete.append(
                {
                    "planner": planner,
                    "seed": seed,
                    "missing_datasets": missing,
                    "reason": f"missing {', '.join(missing)}",
                    "present": {kind: rec.get("run_id") for kind, rec in by_kind.items()},
                }
            )
            continue
        complete.append(
            {
                "planner": planner,
                "seed": seed,
                "nuScenes": by_kind["nuScenes"],
                "nuCrash": by_kind["nuCrash"],
            }
        )
    return {
        "runs_dir": runs_dir,
        "complete_pairs": complete,
        "incomplete_pairs": incomplete,
        "skipped_runs": skipped,
    }


def joint_summary_output_dir(runs_dir: str, planner: str, seed: int) -> str:
    safe_planner = re.sub(r"[^A-Za-z0-9._-]+", "_", str(planner)).strip("_") or "planner"
    return os.path.join(
        os.path.abspath(runs_dir),
        "comparison",
        "joint",
        f"{safe_planner}__seed{int(seed)}",
    )


def _rate_payload(report: Mapping[str, Any], *keys: str) -> Dict[str, Any]:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, Mapping):
            return {"rate": None, "count": None, "denominator": None}
        cur = cur.get(key)
    if isinstance(cur, Mapping):
        return {
            "rate": _finite(cur.get("rate")),
            "count": cur.get("count"),
            "denominator": cur.get("denominator") or cur.get("n"),
            "ci95_low": _finite(cur.get("ci95_low")),
            "ci95_high": _finite(cur.get("ci95_high")),
        }
    return {"rate": _finite(cur), "count": None, "denominator": None}


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _capability_levels(bundle: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    analysis = bundle.get("failure_analysis") if isinstance(bundle, Mapping) else {}
    assessment = (analysis or {}).get("capability_assessment") or {}
    out: Dict[str, Optional[str]] = {}
    if not isinstance(assessment, Mapping):
        return {dim: None for dim in CAPABILITY_DIMENSIONS}
    for dim in CAPABILITY_DIMENSIONS:
        entry = assessment.get(dim)
        if isinstance(entry, Mapping):
            level = entry.get("level") or entry.get("capability_level")
            out[dim] = str(level) if level in _LEVEL_RANK else None
        else:
            out[dim] = None
    return out


def _compact_llm(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    analysis = bundle.get("failure_analysis") if isinstance(bundle, Mapping) else {}
    analysis = analysis if isinstance(analysis, Mapping) else {}
    guidance = bundle.get("improvement_guidance")
    if isinstance(guidance, dict):
        items = guidance.get("improvement_guidance") or guidance.get("prioritized_guidance") or []
    else:
        items = guidance if isinstance(guidance, list) else []
    compact_items = []
    for item in items[:5]:
        if not isinstance(item, Mapping):
            continue
        compact_items.append(
            {
                "priority": item.get("priority"),
                "affected_capability": item.get("affected_capability"),
                "observed_limitation": item.get("observed_limitation"),
                "recommended_improvement": item.get("recommended_improvement"),
            }
        )
    return {
        "experiment_result_summary": analysis.get("experiment_result_summary"),
        "capability_levels": _capability_levels(bundle),
        "improvement_guidance": compact_items,
    }


def _infer_limitation_scope(
    nusc_coll: Optional[float],
    crash_coll: Optional[float],
    min_support: int,
    nusc_n: Any,
    crash_n: Any,
) -> str:
    try:
        nusc_support = int(nusc_n or 0)
        crash_support = int(crash_n or 0)
    except (TypeError, ValueError):
        nusc_support, crash_support = 0, 0
    if nusc_coll is None or crash_coll is None:
        return "inconclusive"
    if nusc_support < min_support or crash_support < min_support:
        return "inconclusive"
    delta = crash_coll - nusc_coll
    if delta >= 0.10 and nusc_coll <= 0.05:
        return "long_tail_specific"
    if delta >= 0.10:
        return "amplified_on_long_tail"
    if abs(delta) <= 0.05 and max(nusc_coll, crash_coll) >= 0.10:
        return "general"
    if abs(delta) <= 0.05:
        return "general"
    if delta <= -0.10:
        return "inconclusive"
    return "amplified_on_long_tail" if crash_coll > nusc_coll else "inconclusive"


def build_deterministic_cross_dataset_comparison(
    *,
    nuscenes_report: Mapping[str, Any],
    nucrash_report: Mapping[str, Any],
    nuscenes_llm: Mapping[str, Any],
    nucrash_llm: Mapping[str, Any],
    min_support: int = 5,
) -> Dict[str, Any]:
    nusc_coll = _rate_payload(nuscenes_report, "overall", "collision")
    crash_coll = _rate_payload(nucrash_report, "overall", "collision")
    nusc_near = _rate_payload(nuscenes_report, "overall", "near_miss_excluding_collisions")
    crash_near = _rate_payload(nucrash_report, "overall", "near_miss_excluding_collisions")
    nusc_pop = _rate_payload(nuscenes_report, "population_statistics", "safety", "collision_rate")
    crash_pop = _rate_payload(nucrash_report, "population_statistics", "safety", "collision_rate")

    nusc_coll_rate = nusc_coll.get("rate") if nusc_coll.get("rate") is not None else nusc_pop.get("rate")
    crash_coll_rate = crash_coll.get("rate") if crash_coll.get("rate") is not None else crash_pop.get("rate")

    nusc_patterns = ((nuscenes_report.get("failure_statistics") or {}).get("patterns") or {})
    crash_patterns = ((nucrash_report.get("failure_statistics") or {}).get("patterns") or {})
    pattern_deltas = []
    for label in FAILURE_TAXONOMY:
        nusc_rate = _finite((nusc_patterns.get(label) or {}).get("adverse_rate"))
        crash_rate = _finite((crash_patterns.get(label) or {}).get("adverse_rate"))
        pattern_deltas.append(
            {
                "pattern": label,
                "nuScenes_adverse_rate": nusc_rate,
                "nuCrash_adverse_rate": crash_rate,
                "delta": _delta(nusc_rate, crash_rate),
            }
        )
    pattern_deltas.sort(key=lambda row: (row.get("delta") is None, -(row.get("delta") or 0.0)))

    nusc_levels = _capability_levels(nuscenes_llm)
    crash_levels = _capability_levels(nucrash_llm)
    capability_shift = []
    for dim in CAPABILITY_DIMENSIONS:
        nusc_level = nusc_levels.get(dim)
        crash_level = crash_levels.get(dim)
        nusc_rank = _LEVEL_RANK.get(nusc_level) if nusc_level else None
        crash_rank = _LEVEL_RANK.get(crash_level) if crash_level else None
        capability_shift.append(
            {
                "dimension": dim,
                "nuScenes_level": nusc_level,
                "nuCrash_level": crash_level,
                "rank_delta": (
                    None if nusc_rank is None or crash_rank is None else crash_rank - nusc_rank
                ),
            }
        )

    nusc_n = nusc_coll.get("denominator") or (nuscenes_report.get("overall") or {}).get("n_interactive")
    crash_n = crash_coll.get("denominator") or (nucrash_report.get("overall") or {}).get("n_interactive")
    limitation_scope = _infer_limitation_scope(
        nusc_coll_rate, crash_coll_rate, min_support, nusc_n, crash_n
    )
    return {
        "min_support": int(min_support),
        "safety": {
            "collision": {
                "nuScenes": nusc_coll,
                "nuCrash": crash_coll,
                "delta_rate": _delta(nusc_coll_rate, crash_coll_rate),
            },
            "near_miss_excluding_collisions": {
                "nuScenes": nusc_near,
                "nuCrash": crash_near,
                "delta_rate": _delta(nusc_near.get("rate"), crash_near.get("rate")),
            },
        },
        "failure_pattern_deltas": pattern_deltas,
        "capability_shift": capability_shift,
        "limitation_scope": limitation_scope,
        "limitation_scope_rule": (
            "long_tail_specific: nuScenes collision rate <= 5% and nuCrash-nuScenes >= 10pp; "
            "amplified_on_long_tail: both datasets show collisions and nuCrash-nuScenes >= 10pp; "
            "general: |delta| <= 5pp; otherwise inconclusive when support is low."
        ),
    }


def compact_single_dataset_for_joint(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    return _compact_llm(bundle)


def unavailable_joint_summary(
    *,
    planner_name: str,
    seed: int,
    reason: str,
    stage_status: Mapping[str, Any],
    nuscenes_run: Mapping[str, Any],
    nucrash_run: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "available": False,
        "unavailable_reason": reason,
        "planner": planner_name,
        "seed": int(seed),
        "source_runs": {
            "nuScenes": nuscenes_run.get("run_id"),
            "nuCrash": nucrash_run.get("run_id"),
        },
        "stage_status": dict(stage_status),
    }


def build_joint_analysis(
    *,
    planner_name: str,
    seed: int,
    nuscenes_run: Mapping[str, Any],
    nucrash_run: Mapping[str, Any],
    comparison: Mapping[str, Any],
    nuscenes_llm: Mapping[str, Any],
    nucrash_llm: Mapping[str, Any],
    stage_status: Mapping[str, Any],
    joint_llm: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "available": True,
        "analysis_scope": "paired nuScenes x nuCrash joint analysis",
        "planner": planner_name,
        "seed": int(seed),
        "source_runs": {
            "nuScenes": {
                "run_id": nuscenes_run.get("run_id"),
                "run_dir": nuscenes_run.get("run_dir"),
            },
            "nuCrash": {
                "run_id": nucrash_run.get("run_id"),
                "run_dir": nucrash_run.get("run_dir"),
            },
        },
        "limitation_scope": comparison.get("limitation_scope"),
        "deterministic_cross_dataset_comparison": comparison,
        "nuScenes_llm_summary": compact_single_dataset_for_joint(nuscenes_llm),
        "nuCrash_llm_summary": compact_single_dataset_for_joint(nucrash_llm),
        "joint_llm": dict(joint_llm) if isinstance(joint_llm, Mapping) else None,
        "stage_status": dict(stage_status),
    }


def _fmt_rate(payload: Mapping[str, Any]) -> str:
    rate = _finite(payload.get("rate"))
    if rate is None:
        return "n/a"
    count = payload.get("count")
    denom = payload.get("denominator")
    if count is not None and denom is not None:
        return f"{rate:.2%} ({count}/{denom})"
    return f"{rate:.2%}"


def render_joint_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# Joint analysis: {summary.get('planner')} seed={summary.get('seed')}",
        "",
    ]
    if not summary.get("available"):
        lines.append(f"Unavailable: {summary.get('unavailable_reason') or 'unknown reason'}")
        return "\n".join(lines)

    src = summary.get("source_runs") or {}
    lines.append(f"- nuScenes run: {(src.get('nuScenes') or {}).get('run_id')}")
    lines.append(f"- nuCrash run: {(src.get('nuCrash') or {}).get('run_id')}")
    lines.append(f"- Limitation scope: {summary.get('limitation_scope')}")
    comparison = summary.get("deterministic_cross_dataset_comparison") or {}
    safety = comparison.get("safety") or {}
    coll = safety.get("collision") or {}
    lines.extend(["", "## Safety rates", ""])
    lines.append(f"- nuScenes collision: {_fmt_rate(coll.get('nuScenes') or {})}")
    lines.append(f"- nuCrash collision: {_fmt_rate(coll.get('nuCrash') or {})}")
    delta = _finite(coll.get("delta_rate"))
    lines.append(f"- Collision-rate delta (nuCrash − nuScenes): {f'{delta:+.2%}' if delta is not None else 'n/a'}")

    lines.extend(["", "## Failure-pattern adverse-rate deltas", ""])
    for row in (comparison.get("failure_pattern_deltas") or [])[:8]:
        dlt = _finite(row.get("delta"))
        lines.append(
            f"- {row.get('pattern')}: nuScenes={row.get('nuScenes_adverse_rate')}, "
            f"nuCrash={row.get('nuCrash_adverse_rate')}, "
            f"delta={f'{dlt:+.3f}' if dlt is not None else 'n/a'}"
        )

    lines.extend(["", "## Capability shift", ""])
    for row in comparison.get("capability_shift") or []:
        lines.append(
            f"- {row.get('dimension')}: {row.get('nuScenes_level')} → {row.get('nuCrash_level')}"
        )

    joint_llm = summary.get("joint_llm") or {}
    if joint_llm:
        lines.extend(["", "## Joint LLM overlay", ""])
        for key in (
            "cross_dataset_diagnostic_synthesis",
            "naturalistic_performance_summary",
            "long_tail_performance_summary",
            "observed_limitation",
            "possible_mechanism",
            "improvement_guidance",
        ):
            value = joint_llm.get(key)
            if value:
                lines.append(f"- **{key.replace('_', ' ')}:** {value}")
    return "\n".join(lines) + "\n"


def save_joint_analysis(summary: Mapping[str, Any], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "joint_analysis.json")
    md_path = os.path.join(out_dir, "joint_analysis.md")
    comparison_path = os.path.join(out_dir, "cross_dataset_comparison.json")
    _write_json(json_path, summary)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_joint_markdown(summary))
    comparison = summary.get("deterministic_cross_dataset_comparison")
    if isinstance(comparison, Mapping):
        _write_json(comparison_path, comparison)
    else:
        comparison_path = ""
    paths = {"json": json_path, "markdown": md_path}
    if comparison_path:
        paths["comparison"] = comparison_path
    return paths
