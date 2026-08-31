#!/usr/bin/env python3
"""
crashsim_eval_stats.py

Theory-grounded scientific attribution layer for the Interpretable AV Evaluation.

This module augments the deterministic evidence produced by ``crashsim_planner.py``
with methods that address the main methodological gaps of a correlational,
heuristic-labelled evaluation. It is intentionally decoupled from the rollout
code: it consumes the per-scene records (the same dicts written to
``results.jsonl``) and returns a JSON-serializable analysis dict.

Implemented methods (each degrades gracefully if a backend or the data is
insufficient, and never raises into the caller):

1. Cluster-robust / mixed-effects logistic regression for collision risk
   (``statsmodels`` GEE with scene-clustered robust covariance, plus a
   Bayesian random-intercept GLMM). This corrects for the non-independence of
   multiple sub-sequences sampled from the same driving scene and reports odds
   ratios with confidence/credible intervals.
2. Game-theoretic feature attribution via SHAP on a cross-validated surrogate
   collision classifier (Lundberg & Lee, 2017). Falls back to permutation
   importance when SHAP/numba are unavailable.
3. Multiple-comparison control (Benjamini-Hochberg FDR) over per-feature group
   tests, with bias-corrected-and-accelerated (BCa) bootstrap intervals for the
   effect sizes (Efron, 1987).
4. Extreme-value analysis (peaks-over-threshold + Generalized Pareto
   distribution) to extrapolate a rare collision probability from the tail of
   the near-miss clearance distribution (Pickands, 1975; Coles, 2001).
5. Survival analysis of time-to-collision with right-censoring for safe
   sequences: Kaplan-Meier curves + multivariate log-rank test and a Cox
   proportional-hazards model (Cox, 1972) reporting hazard ratios.

All heavy third-party imports are performed lazily inside each function so that
importing this module (and running the rollout) never requires these packages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Features excluded from any predictive/associational model because they are
# (near-)deterministic functions of the collision outcome itself. Using them
# would create target leakage and inflate apparent attribution.
LEAKAGE_EXCLUDED_FIELDS: Tuple[str, ...] = (
    "did_coll",
    "did_near_crash",
    "coll_step",
    "ttc_sec",
    "min_dist_m",
    "min_clearance_proxy_m",
    "min_box_separation_margin_m",
    "relative_speed_at_collision_mps",
    "relative_heading_at_collision_rad",
    "relative_long_gap_at_collision_m",
    "relative_lat_gap_at_collision_m",
    "ego_speed_at_collision",
    "rel_speed_at_min_dist",
    "rel_heading_at_min_dist",
    "rel_long_gap_at_min_dist",
    "rel_lat_gap_at_min_dist",
    "outcome_class",
    "diagnostic_scorecard",
    "failure_mechanism",
    "conflict_geometry",
    "evidence",
)

DEFAULT_NUMERIC_FEATURES: Tuple[str, ...] = (
    "ego_avg_speed_mps",
    "ego_max_decel",
    "ego_max_accel",
    "ego_mean_abs_accel",
    "ego_max_abs_jerk",
    "ego_lat_accel_max",
    "ego_progress_ratio",
    "ego_path_efficiency",
    "ego_ade_m",
    "NA",
)

DEFAULT_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "behavior_tag",
    "odd_region",
    "odd_speed_bin",
    "odd_density_bin",
)


@dataclass
class ScientificConfig:
    """Knobs for the scientific attribution layer (all CLI-exposed upstream)."""

    enable: bool = True
    numeric_features: Tuple[str, ...] = DEFAULT_NUMERIC_FEATURES
    categorical_features: Tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES
    min_events: int = 8               # min collisions required for modeling
    min_samples: int = 30             # min interactive sequences required
    bootstrap_n: int = 2000
    random_state: int = 0
    fdr_alpha: float = 0.05
    evt_quantile: float = 0.90        # POT threshold as a quantile of severity
    evt_min_exceedances: int = 25
    shap_max_background: int = 200
    cv_folds: int = 5

    @classmethod
    def from_cfg(cls, cfg: Any) -> "ScientificConfig":
        def _get(name: str, default: Any) -> Any:
            return getattr(cfg, name, default)

        return cls(
            enable=bool(_get("eval_enable_advanced_stats", True)),
            min_events=int(_get("eval_stats_min_events", 8)),
            min_samples=int(_get("eval_stats_min_samples", 30)),
            bootstrap_n=int(_get("eval_bootstrap_n", 2000)),
            random_state=int(_get("eval_stats_seed", 0)),
            fdr_alpha=float(_get("eval_fdr_alpha", 0.05)),
            evt_quantile=float(_get("eval_evt_quantile", 0.90)),
            evt_min_exceedances=int(_get("eval_evt_min_exceedances", 25)),
            shap_max_background=int(_get("eval_shap_max_background", 200)),
            cv_folds=int(_get("eval_stats_cv_folds", 5)),
        )


# ---------------------------------------------------------------------------
# Small numeric helpers (numpy-only; safe against NaN/inf)
# ---------------------------------------------------------------------------
def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 2 or b.size < 2:
        return None
    pooled_num = (a.size - 1) * float(np.var(a, ddof=1)) + (b.size - 1) * float(np.var(b, ddof=1))
    pooled_den = a.size + b.size - 2
    if pooled_den <= 0:
        return None
    pooled = math.sqrt(max(0.0, pooled_num / pooled_den))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Benjamini-Hochberg FDR control. Returns q-values and reject flags."""
    p = np.asarray([x if (x is not None and math.isfinite(x)) else 1.0 for x in pvalues], dtype=np.float64)
    n = p.size
    if n == 0:
        return {"qvalues": [], "reject": [], "alpha": alpha}
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    # enforce monotonicity of q-values from the largest p downward
    q_min = np.minimum.accumulate(q[::-1])[::-1]
    q_min = np.clip(q_min, 0.0, 1.0)
    qvalues = np.empty(n, dtype=np.float64)
    qvalues[order] = q_min
    reject = qvalues <= alpha
    return {
        "qvalues": [float(x) for x in qvalues],
        "reject": [bool(x) for x in reject],
        "alpha": float(alpha),
        "n_tests": int(n),
        "n_significant": int(np.sum(reject)),
    }


def _bca_interval(
    boot_stats: np.ndarray,
    jackknife_stats: np.ndarray,
    point_stat: float,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """Bias-corrected and accelerated bootstrap CI (Efron, 1987)."""
    try:
        from scipy.stats import norm
    except Exception:
        return None, None
    boot = boot_stats[np.isfinite(boot_stats)]
    if boot.size < 10:
        return None, None
    prop = float(np.mean(boot < point_stat))
    prop = min(max(prop, 1.0 / (boot.size + 1)), 1.0 - 1.0 / (boot.size + 1))
    z0 = norm.ppf(prop)
    jack = jackknife_stats[np.isfinite(jackknife_stats)]
    a = 0.0
    if jack.size >= 3:
        jbar = float(np.mean(jack))
        diffs = jbar - jack
        denom = 6.0 * (float(np.sum(diffs ** 2)) ** 1.5)
        if denom > 1e-12:
            a = float(np.sum(diffs ** 3)) / denom
    zl, zu = norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)

    def _adjust(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-12:
            return norm.cdf(z0 + z)
        return float(norm.cdf(z0 + (z0 + z) / denom))

    lo_p, hi_p = _adjust(zl), _adjust(zu)
    lo = float(np.quantile(boot, min(max(lo_p, 0.0), 1.0)))
    hi = float(np.quantile(boot, min(max(hi_p, 0.0), 1.0)))
    return lo, hi


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _build_dataframe(records: Sequence[Dict[str, Any]], config: ScientificConfig):
    """Assemble a modeling DataFrame from interactive scene records.

    Returns (df, used_numeric, used_categorical) or (None, [], []) when pandas
    is unavailable or there is not enough usable data.
    """
    try:
        import pandas as pd
    except Exception:
        return None, [], []

    rows: List[Dict[str, Any]] = []
    for r in records:
        if not bool(r.get("has_other_agents", True)):
            continue
        row: Dict[str, Any] = {
            "did_coll": int(bool(r.get("did_coll", False))),
            "scene": str(r.get("scene_token") or r.get("scene_name") or "unknown"),
            "FT": _finite(r.get("FT")) or 0.0,
            "dt": _finite(r.get("dt")) or 0.5,
            "coll_step": _finite(r.get("coll_step")),
            "ttc_sec": _finite(r.get("ttc_sec")),
            "min_clearance_proxy_m": _finite(r.get("min_clearance_proxy_m")),
            "min_box_separation_margin_m": _finite(r.get("min_box_separation_margin_m")),
        }
        for key in config.numeric_features:
            row[key] = _finite(r.get(key))
        for key in config.categorical_features:
            val = r.get(key)
            row[key] = str(val) if val not in (None, "") else "unknown"
        rows.append(row)

    if len(rows) < config.min_samples:
        return None, [], []
    df = pd.DataFrame(rows)

    # keep numeric features with enough coverage and non-zero variance
    used_numeric: List[str] = []
    for key in config.numeric_features:
        col = df[key].astype(float)
        if col.notna().mean() >= 0.6 and float(np.nanstd(col.values)) > 1e-9:
            df[key] = col.fillna(col.median())
            used_numeric.append(key)

    # keep categoricals with 2..12 levels each having adequate support
    used_categorical: List[str] = []
    for key in config.categorical_features:
        counts = df[key].value_counts()
        levels = counts[counts >= 3]
        if 2 <= len(levels) <= 12:
            df[key] = np.where(df[key].isin(levels.index), df[key], "other")
            used_categorical.append(key)

    return df, used_numeric, used_categorical


# ---------------------------------------------------------------------------
# 1) Cluster-robust logistic regression + Bayesian random-intercept GLMM
# ---------------------------------------------------------------------------
def compute_cluster_robust_logistic(records, config):
    result: Dict[str, Any] = {"available": False, "method": "GEE (scene-clustered) + Bayesian GLMM"}
    df, num, cat = _build_dataframe(records, config)
    if df is None:
        result["reason"] = "insufficient or unavailable data (need pandas and enough interactive samples)"
        return result
    n_coll = int(df["did_coll"].sum())
    if n_coll < config.min_events or (len(df) - n_coll) < config.min_events:
        result["reason"] = f"insufficient class balance (collisions={n_coll}, n={len(df)})"
        return result
    if not num and not cat:
        result["reason"] = "no usable predictors after coverage/variance filtering"
        return result

    try:
        import pandas as pd  # noqa: F401
        import patsy
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable
    except Exception as e:
        result["reason"] = f"statsmodels/patsy unavailable: {e}"
        return result

    terms = list(num) + [f"C({c})" for c in cat]
    formula = "did_coll ~ " + " + ".join(terms)
    result["formula"] = formula
    result["n"] = int(len(df))
    result["n_coll"] = n_coll
    result["predictors"] = {"numeric": num, "categorical": cat}

    # GEE with exchangeable working correlation, clustered by scene → robust SEs
    try:
        y, X = patsy.dmatrices(formula, df, return_type="dataframe")
        model = GEE(y, X, groups=df["scene"].values, family=sm.families.Binomial(),
                    cov_struct=Exchangeable())
        fit = model.fit(maxiter=100)
        conf = fit.conf_int()
        gee_rows: List[Dict[str, Any]] = []
        for name in fit.params.index:
            if name == "Intercept":
                continue
            coef = float(fit.params[name])
            lo, hi = float(conf.loc[name][0]), float(conf.loc[name][1])
            pval = float(fit.pvalues[name])
            gee_rows.append({
                "term": name,
                "coef": coef,
                "odds_ratio": float(math.exp(coef)),
                "or_ci95_low": float(math.exp(lo)),
                "or_ci95_high": float(math.exp(hi)),
                "p_value": pval,
            })
        # FDR across the reported coefficients
        fdr = benjamini_hochberg([r["p_value"] for r in gee_rows], config.fdr_alpha)
        for r, q, rej in zip(gee_rows, fdr["qvalues"], fdr["reject"]):
            r["q_value_bh"] = q
            r["significant_fdr"] = rej
        gee_rows.sort(key=lambda r: abs(math.log(max(r["odds_ratio"], 1e-9))), reverse=True)
        result["gee"] = {
            "n_clusters": int(df["scene"].nunique()),
            "cov_struct": "exchangeable",
            "coefficients": gee_rows,
            "interpretation": "Population-averaged odds ratios with scene-cluster-robust SEs; OR>1 raises collision odds.",
        }
        result["available"] = True
    except Exception as e:
        result["gee_error"] = str(e)

    # Bayesian random-intercept GLMM (subject-specific), best-effort
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        vc = {"scene": "0 + C(scene)"}
        glmm = BinomialBayesMixedGLM.from_formula(formula, vc, df)
        gres = glmm.fit_vb(verbose=False)
        fe_names = list(gres.model.exog_names)
        means = np.asarray(gres.fe_mean, dtype=np.float64)
        sds = np.asarray(gres.fe_sd, dtype=np.float64)
        glmm_rows: List[Dict[str, Any]] = []
        for name, m, s in zip(fe_names, means, sds):
            if name == "Intercept":
                continue
            glmm_rows.append({
                "term": name,
                "posterior_mean_coef": float(m),
                "odds_ratio": float(math.exp(m)),
                "or_cri95_low": float(math.exp(m - 1.96 * s)),
                "or_cri95_high": float(math.exp(m + 1.96 * s)),
                "posterior_sd": float(s),
            })
        glmm_rows.sort(key=lambda r: abs(math.log(max(r["odds_ratio"], 1e-9))), reverse=True)
        result["random_effects_glmm"] = {
            "estimator": "BinomialBayesMixedGLM (variational Bayes)",
            "random_effect": "scene intercept",
            "fixed_effects": glmm_rows,
            "interpretation": "Subject-specific ORs with a per-scene random intercept; 95% credible intervals.",
        }
        result["available"] = True
    except Exception as e:
        result["glmm_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# 2) SHAP feature attribution on a surrogate collision classifier
# ---------------------------------------------------------------------------
def compute_shap_feature_importance(records, config):
    result: Dict[str, Any] = {"available": False, "method": "SHAP on cross-validated surrogate classifier"}
    df, num, cat = _build_dataframe(records, config)
    if df is None:
        result["reason"] = "insufficient or unavailable data"
        return result
    n_coll = int(df["did_coll"].sum())
    if n_coll < config.min_events or (len(df) - n_coll) < config.min_events:
        result["reason"] = f"insufficient class balance (collisions={n_coll}, n={len(df)})"
        return result
    if not num and not cat:
        result["reason"] = "no usable predictors"
        return result

    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        result["reason"] = f"scikit-learn unavailable: {e}"
        return result

    X = df[list(num)].copy()
    for c in cat:
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=False)
        X = pd.concat([X, dummies], axis=1)
    X = X.astype(float)
    y = df["did_coll"].astype(int).values
    feature_names = list(X.columns)
    result["n"] = int(len(df))
    result["n_coll"] = n_coll
    result["n_features"] = len(feature_names)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )

    # surrogate fidelity: cross-validated AUC
    try:
        folds = max(2, min(config.cv_folds, n_coll, len(df) - n_coll))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
        proba = cross_val_predict(clf, X.values, y, cv=skf, method="predict_proba")[:, 1]
        result["surrogate_cv_auc"] = float(roc_auc_score(y, proba))
    except Exception as e:
        result["surrogate_cv_auc"] = None
        result["cv_auc_error"] = str(e)

    clf.fit(X.values, y)

    # Preferred: game-theoretic SHAP values
    shap_ok = False
    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        bg = X
        if len(X) > config.shap_max_background:
            bg = X.sample(config.shap_max_background, random_state=config.random_state)
        sv = explainer.shap_values(bg.values, check_additivity=False)
        # binary RF → list [class0, class1] or array (n, f, 2) depending on version
        if isinstance(sv, list):
            sv_pos = np.asarray(sv[1])
        else:
            arr = np.asarray(sv)
            sv_pos = arr[..., 1] if arr.ndim == 3 else arr
        mean_abs = np.mean(np.abs(sv_pos), axis=0)
        mean_signed = np.mean(sv_pos, axis=0)
        rows = [
            {
                "feature": feature_names[i],
                "mean_abs_shap": float(mean_abs[i]),
                "mean_signed_shap": float(mean_signed[i]),
                "direction": ("increases_risk" if mean_signed[i] > 0 else "decreases_risk"),
            }
            for i in range(len(feature_names))
        ]
        rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
        result["attribution_backend"] = "shap.TreeExplainer"
        result["feature_attribution"] = rows[:20]
        result["available"] = True
        shap_ok = True
    except Exception as e:
        result["shap_error"] = str(e)

    # Fallback: permutation importance (still model-agnostic, non-leaky)
    if not shap_ok:
        try:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(
                clf, X.values, y, n_repeats=10, random_state=config.random_state, n_jobs=-1
            )
            rows = [
                {"feature": feature_names[i],
                 "permutation_importance_mean": float(pi.importances_mean[i]),
                 "permutation_importance_std": float(pi.importances_std[i])}
                for i in range(len(feature_names))
            ]
            rows.sort(key=lambda r: r["permutation_importance_mean"], reverse=True)
            result["attribution_backend"] = "permutation_importance (shap unavailable)"
            result["feature_attribution"] = rows[:20]
            result["available"] = True
        except Exception as e:
            result["permutation_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# 3) FDR-controlled group tests + BCa bootstrap effect sizes
# ---------------------------------------------------------------------------
def compute_feature_significance(records, config):
    result: Dict[str, Any] = {"available": False, "method": "Mann-Whitney U + BH-FDR + BCa bootstrap Cohen's d"}
    interactive = [r for r in records if bool(r.get("has_other_agents", True))]
    collided = [r for r in interactive if bool(r.get("did_coll", False))]
    safe = [r for r in interactive if not bool(r.get("did_coll", False))]
    if len(collided) < config.min_events or len(safe) < config.min_events:
        result["reason"] = f"insufficient class balance (collisions={len(collided)}, safe={len(safe)})"
        return result

    try:
        from scipy.stats import mannwhitneyu
    except Exception as e:
        result["reason"] = f"scipy unavailable: {e}"
        return result

    rng = np.random.default_rng(config.random_state)
    rows: List[Dict[str, Any]] = []
    for key in config.numeric_features:
        a = np.asarray([v for v in (_finite(r.get(key)) for r in collided) if v is not None], dtype=np.float64)
        b = np.asarray([v for v in (_finite(r.get(key)) for r in safe) if v is not None], dtype=np.float64)
        if a.size < 3 or b.size < 3:
            continue
        try:
            _, pval = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            continue
        d = _cohens_d(a, b)
        # BCa bootstrap for Cohen's d
        boot = np.empty(config.bootstrap_n, dtype=np.float64)
        for i in range(config.bootstrap_n):
            ra = a[rng.integers(0, a.size, a.size)]
            rb = b[rng.integers(0, b.size, b.size)]
            bd = _cohens_d(ra, rb)
            boot[i] = bd if bd is not None else np.nan
        # jackknife over the union for acceleration
        union = np.concatenate([a, b])
        labels = np.concatenate([np.ones(a.size), np.zeros(b.size)])
        jack = np.empty(union.size, dtype=np.float64)
        for i in range(union.size):
            mask = np.ones(union.size, dtype=bool)
            mask[i] = False
            ja = union[mask][labels[mask] == 1]
            jb = union[mask][labels[mask] == 0]
            jd = _cohens_d(ja, jb)
            jack[i] = jd if jd is not None else np.nan
        lo, hi = _bca_interval(boot, jack, d if d is not None else float("nan"), config.fdr_alpha)
        rows.append({
            "feature": key,
            "n_coll": int(a.size),
            "n_safe": int(b.size),
            "mean_coll": float(np.mean(a)),
            "mean_safe": float(np.mean(b)),
            "cohens_d": d,
            "cohens_d_bca_low": lo,
            "cohens_d_bca_high": hi,
            "p_value": float(pval),
        })

    if not rows:
        result["reason"] = "no numeric feature had enough finite values in both groups"
        return result

    fdr = benjamini_hochberg([r["p_value"] for r in rows], config.fdr_alpha)
    for r, q, rej in zip(rows, fdr["qvalues"], fdr["reject"]):
        r["q_value_bh"] = q
        r["significant_fdr"] = rej
    rows.sort(key=lambda r: abs(r.get("cohens_d") or 0.0), reverse=True)
    result["available"] = True
    result["n_tests"] = fdr["n_tests"]
    result["n_significant_fdr"] = fdr["n_significant"]
    result["fdr_alpha"] = config.fdr_alpha
    result["bootstrap_n"] = config.bootstrap_n
    result["features"] = rows
    return result


# ---------------------------------------------------------------------------
# 4) Extreme value analysis (POT + GPD) for rare-collision extrapolation
# ---------------------------------------------------------------------------
def compute_extreme_value_analysis(records, config):
    result: Dict[str, Any] = {"available": False, "method": "Peaks-over-threshold + Generalized Pareto (EVT)"}
    interactive = [r for r in records if bool(r.get("has_other_agents", True))]
    # Severity: smaller clearance is more dangerous → severity = -clearance.
    # Collision boundary is clearance <= 0, i.e. severity >= 0.
    sev = np.asarray(
        [(-v) for v in (_finite(r.get("min_clearance_proxy_m")) for r in interactive) if v is not None],
        dtype=np.float64,
    )
    if sev.size < config.min_samples:
        result["reason"] = f"insufficient clearance samples (n={sev.size})"
        return result

    try:
        from scipy.stats import genpareto
    except Exception as e:
        result["reason"] = f"scipy unavailable: {e}"
        return result

    n = sev.size
    empirical_collision_rate = float(np.mean(sev >= 0.0))
    u = float(np.quantile(sev, config.evt_quantile))
    exceed = sev[sev > u] - u
    if exceed.size < config.evt_min_exceedances:
        result["reason"] = f"too few exceedances above threshold (got {exceed.size}, need {config.evt_min_exceedances})"
        result["threshold"] = u
        result["empirical_collision_rate"] = empirical_collision_rate
        return result

    try:
        shape, loc, scale = genpareto.fit(exceed, floc=0.0)
    except Exception as e:
        result["reason"] = f"GPD fit failed: {e}"
        return result

    zeta_u = float(exceed.size) / float(n)  # P(severity > u)
    # P(severity >= 0) extrapolated: only meaningful when u < 0 (clearance>0 at threshold)
    if u < 0.0:
        tail = float(genpareto.sf(0.0 - u, shape, loc=0.0, scale=scale))
        evt_collision_prob = zeta_u * tail
    else:
        evt_collision_prob = empirical_collision_rate  # threshold already in collision region

    # bootstrap CI for the extrapolated probability
    rng = np.random.default_rng(config.random_state)
    boot_probs: List[float] = []
    for _ in range(min(config.bootstrap_n, 1000)):
        rs = sev[rng.integers(0, n, n)]
        ex = rs[rs > u] - u
        if ex.size < 10:
            continue
        try:
            sh, _, sc = genpareto.fit(ex, floc=0.0)
        except Exception:
            continue
        zu = float(ex.size) / float(n)
        if u < 0.0:
            boot_probs.append(zu * float(genpareto.sf(0.0 - u, sh, loc=0.0, scale=sc)))
        else:
            boot_probs.append(float(np.mean(rs >= 0.0)))
    ci_low = float(np.quantile(boot_probs, 0.025)) if len(boot_probs) >= 20 else None
    ci_high = float(np.quantile(boot_probs, 0.975)) if len(boot_probs) >= 20 else None

    result.update({
        "available": True,
        "severity_definition": "severity = -min_clearance_proxy_m (SAT margin); collision boundary severity >= 0",
        "n": n,
        "threshold_quantile": config.evt_quantile,
        "threshold_severity": u,
        "n_exceedances": int(exceed.size),
        "gpd_shape_xi": float(shape),
        "gpd_scale_sigma": float(scale),
        "prob_exceed_threshold": zeta_u,
        "empirical_collision_rate": empirical_collision_rate,
        "evt_extrapolated_collision_prob": float(evt_collision_prob),
        "evt_collision_prob_ci95_low": ci_low,
        "evt_collision_prob_ci95_high": ci_high,
        "interpretation": (
            "GPD tail model extrapolates collision probability from the near-miss clearance tail. "
            "xi>0 indicates a heavy tail (more rare-but-severe risk). Compare EVT vs empirical rate: "
            "large divergence flags under-sampled tail risk."
        ),
    })
    return result


# ---------------------------------------------------------------------------
# 5) Survival analysis of time-to-collision (KM + log-rank + Cox PH)
# ---------------------------------------------------------------------------
def _survival_frame(records, config):
    try:
        import pandas as pd
    except Exception:
        return None
    rows: List[Dict[str, Any]] = []
    for r in records:
        if not bool(r.get("has_other_agents", True)):
            continue
        dt = _finite(r.get("dt")) or 0.5
        ft = _finite(r.get("FT")) or 0.0
        horizon = max(dt, ft * dt)
        event = int(bool(r.get("did_coll", False)))
        if event:
            step = _finite(r.get("coll_step"))
            ttc = _finite(r.get("ttc_sec"))
            if step is not None:
                duration = max(dt, step * dt)
            elif ttc is not None:
                duration = max(dt, ttc)
            else:
                duration = horizon
        else:
            duration = horizon if horizon > 0 else 1.0
        row = {
            "duration": float(duration),
            "event": event,
            "behavior_tag": str(r.get("behavior_tag") or "unknown"),
            "odd_region": str(r.get("odd_region") or "unknown"),
        }
        for key in config.numeric_features:
            row[key] = _finite(r.get(key))
        rows.append(row)
    if len(rows) < config.min_samples:
        return None
    return pd.DataFrame(rows)


def compute_survival_analysis(records, config):
    result: Dict[str, Any] = {"available": False, "method": "Kaplan-Meier + log-rank + Cox proportional hazards"}
    df = _survival_frame(records, config)
    if df is None:
        result["reason"] = "insufficient or unavailable data for survival modeling"
        return result
    n_events = int(df["event"].sum())
    if n_events < config.min_events:
        result["reason"] = f"insufficient collision events for survival modeling (events={n_events})"
        return result

    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
    except Exception as e:
        result["reason"] = f"lifelines unavailable: {e}"
        return result

    result["n"] = int(len(df))
    result["n_events"] = n_events

    # Kaplan-Meier overall
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(df["duration"], df["event"])
        result["km_overall"] = {
            "median_survival_time_sec": (
                float(kmf.median_survival_time_) if math.isfinite(float(kmf.median_survival_time_)) else None
            ),
        }
    except Exception as e:
        result["km_error"] = str(e)

    # grouped log-rank by ODD/behavior when groups have enough events
    for group_key in ("behavior_tag", "odd_region"):
        try:
            counts = df.groupby(group_key)["event"].sum()
            valid_groups = counts[counts >= 2].index.tolist()
            sub = df[df[group_key].isin(valid_groups)]
            if sub[group_key].nunique() >= 2:
                lr = multivariate_logrank_test(sub["duration"], sub[group_key], sub["event"])
                result[f"logrank_{group_key}"] = {
                    "test_statistic": float(lr.test_statistic),
                    "p_value": float(lr.p_value),
                    "n_groups": int(sub[group_key].nunique()),
                    "interpretation": "Log-rank tests whether time-to-collision hazard differs across groups.",
                }
        except Exception as e:
            result[f"logrank_{group_key}_error"] = str(e)

    # Cox proportional hazards on non-leaky numeric covariates
    try:
        num_cols = [c for c in config.numeric_features
                    if c in df.columns and df[c].notna().mean() >= 0.6 and float(np.nanstd(df[c].values)) > 1e-9]
        if num_cols:
            cox_df = df[["duration", "event"] + num_cols].copy()
            for c in num_cols:
                cox_df[c] = cox_df[c].fillna(cox_df[c].median())
                std = float(cox_df[c].std())
                if std > 1e-9:
                    cox_df[c] = (cox_df[c] - float(cox_df[c].mean())) / std  # standardize → HR per 1 SD
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_df, duration_col="duration", event_col="event")
            summ = cph.summary
            hr_rows = []
            for term in summ.index:
                hr_rows.append({
                    "covariate": term,
                    "hazard_ratio_per_sd": float(summ.loc[term, "exp(coef)"]),
                    "hr_ci95_low": float(summ.loc[term, "exp(coef) lower 95%"]),
                    "hr_ci95_high": float(summ.loc[term, "exp(coef) upper 95%"]),
                    "p_value": float(summ.loc[term, "p"]),
                })
            fdr = benjamini_hochberg([r["p_value"] for r in hr_rows], config.fdr_alpha)
            for r, q, rej in zip(hr_rows, fdr["qvalues"], fdr["reject"]):
                r["q_value_bh"] = q
                r["significant_fdr"] = rej
            hr_rows.sort(key=lambda r: abs(math.log(max(r["hazard_ratio_per_sd"], 1e-9))), reverse=True)
            result["cox_ph"] = {
                "covariates_standardized": True,
                "concordance_index": float(cph.concordance_index_),
                "hazard_ratios": hr_rows,
                "interpretation": "HR per +1 SD of the covariate; HR>1 shortens time-to-collision (higher hazard).",
            }
            result["available"] = True
    except Exception as e:
        result["cox_error"] = str(e)

    # KM alone is still a usable deliverable
    if "km_overall" in result:
        result["available"] = True
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def compute_advanced_scientific_attribution(records, config):
    """Run every scientific method; each block degrades independently."""
    interactive = [r for r in records if bool(r.get("has_other_agents", True))]
    n_coll = sum(bool(r.get("did_coll", False)) for r in interactive)
    out: Dict[str, Any] = {
        "schema_version": "1.0",
        "enabled": bool(config.enable),
        "n_interactive": len(interactive),
        "n_collisions": int(n_coll),
        "config": {
            "min_events": config.min_events,
            "min_samples": config.min_samples,
            "bootstrap_n": config.bootstrap_n,
            "fdr_alpha": config.fdr_alpha,
            "evt_quantile": config.evt_quantile,
            "random_state": config.random_state,
        },
        "methods_overview": {
            "cluster_robust_logistic": "GEE/GLMM odds ratios adjusting for scene clustering",
            "shap_feature_importance": "game-theoretic feature attribution (surrogate model)",
            "feature_significance": "BH-FDR controlled group tests + BCa bootstrap effect sizes",
            "extreme_value_analysis": "GPD tail extrapolation of rare collision probability",
            "survival_analysis": "time-to-collision hazard modeling with right-censoring",
        },
    }
    if not config.enable:
        out["reason"] = "advanced statistics disabled via --no_eval_advanced_stats"
        return out
    if len(interactive) < config.min_samples:
        out["reason"] = f"too few interactive sequences (n={len(interactive)}, need {config.min_samples})"
        return out

    out["cluster_robust_logistic"] = _safe(compute_cluster_robust_logistic, records, config)
    out["shap_feature_importance"] = _safe(compute_shap_feature_importance, records, config)
    out["feature_significance"] = _safe(compute_feature_significance, records, config)
    out["extreme_value_analysis"] = _safe(compute_extreme_value_analysis, records, config)
    out["survival_analysis"] = _safe(compute_survival_analysis, records, config)
    return out


def _safe(fn, records, config) -> Dict[str, Any]:
    try:
        return fn(records, config)
    except Exception as e:  # never let the scientific layer break the pipeline
        return {"available": False, "reason": f"unexpected error in {fn.__name__}: {e}"}


# ---------------------------------------------------------------------------
# Rendering + compaction for the LLM evidence package
# ---------------------------------------------------------------------------
def _fmt(value: Any, spec: str = ".3f") -> str:
    v = _finite(value)
    return format(v, spec) if v is not None else "n/a"


def render_advanced_stats_section(adv: Dict[str, Any]) -> str:
    if not adv or not adv.get("enabled", False):
        return ""
    lines = ["## Theory-grounded scientific attribution", ""]
    lines.append(
        "These analyses move beyond correlational effect sizes: they adjust for scene-level "
        "clustering, control the false-discovery rate, quantify game-theoretic feature "
        "attribution, extrapolate rare-collision risk via extreme value theory, and model "
        "time-to-collision with censoring. All results remain auditable and hypothesis-level, "
        "not causal proof."
    )
    if adv.get("reason"):
        lines.extend(["", f"_Not computed: {adv['reason']}_"])
        return "\n".join(lines)

    crl = adv.get("cluster_robust_logistic") or {}
    if crl.get("available"):
        lines.extend(["", "### Scene-clustered logistic regression (odds ratios)", ""])
        gee = crl.get("gee") or {}
        rows = gee.get("coefficients") or []
        if rows:
            lines.append(f"GEE, exchangeable working correlation, {gee.get('n_clusters', '?')} scene clusters.")
            lines.append("")
            lines.append("| term | odds ratio | 95% CI | p | q (BH) | sig |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for r in rows[:12]:
                lines.append(
                    f"| {r['term']} | {_fmt(r['odds_ratio'], '.3f')} | "
                    f"{_fmt(r['or_ci95_low'], '.3f')}–{_fmt(r['or_ci95_high'], '.3f')} | "
                    f"{_fmt(r['p_value'], '.3g')} | {_fmt(r.get('q_value_bh'), '.3g')} | "
                    f"{'yes' if r.get('significant_fdr') else 'no'} |"
                )
        glmm = crl.get("random_effects_glmm") or {}
        if glmm.get("fixed_effects"):
            lines.extend(["", f"Random-intercept GLMM ({glmm.get('estimator')}), top effects:", ""])
            lines.append("| term | odds ratio | 95% CrI |")
            lines.append("|---|---:|---:|")
            for r in glmm["fixed_effects"][:8]:
                lines.append(
                    f"| {r['term']} | {_fmt(r['odds_ratio'], '.3f')} | "
                    f"{_fmt(r['or_cri95_low'], '.3f')}–{_fmt(r['or_cri95_high'], '.3f')} |"
                )
    elif crl:
        lines.extend(["", f"_Logistic regression not computed: {crl.get('reason', 'n/a')}_"])

    shp = adv.get("shap_feature_importance") or {}
    if shp.get("available"):
        lines.extend(["", "### Feature attribution (SHAP surrogate model)", ""])
        auc = shp.get("surrogate_cv_auc")
        lines.append(
            f"Backend: {shp.get('attribution_backend')}; surrogate cross-validated AUC = {_fmt(auc, '.3f')} "
            "(≈0.5 means the model is uninformative and rankings below are unreliable)."
        )
        lines.append("")
        lines.append("| feature | mean |SHAP| | direction |")
        lines.append("|---|---:|---|")
        for r in (shp.get("feature_attribution") or [])[:12]:
            val = r.get("mean_abs_shap", r.get("permutation_importance_mean"))
            lines.append(f"| {r['feature']} | {_fmt(val, '.4f')} | {r.get('direction', 'n/a')} |")
    elif shp:
        lines.extend(["", f"_SHAP attribution not computed: {shp.get('reason', 'n/a')}_"])

    fsig = adv.get("feature_significance") or {}
    if fsig.get("available"):
        lines.extend(["", "### Group differences with FDR control + BCa bootstrap", ""])
        lines.append(
            f"{fsig.get('n_significant_fdr', 0)}/{fsig.get('n_tests', 0)} features significant at "
            f"BH-FDR α={fsig.get('fdr_alpha')}."
        )
        lines.append("")
        lines.append("| feature | Cohen's d | d 95% BCa | p | q (BH) | sig |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in (fsig.get("features") or [])[:12]:
            lines.append(
                f"| {r['feature']} | {_fmt(r.get('cohens_d'))} | "
                f"{_fmt(r.get('cohens_d_bca_low'))}–{_fmt(r.get('cohens_d_bca_high'))} | "
                f"{_fmt(r.get('p_value'), '.3g')} | {_fmt(r.get('q_value_bh'), '.3g')} | "
                f"{'yes' if r.get('significant_fdr') else 'no'} |"
            )
    elif fsig:
        lines.extend(["", f"_FDR group tests not computed: {fsig.get('reason', 'n/a')}_"])

    evt = adv.get("extreme_value_analysis") or {}
    if evt.get("available"):
        lines.extend(["", "### Extreme value analysis (rare-collision extrapolation)", ""])
        lines.append(
            f"- GPD tail fit above severity threshold {_fmt(evt.get('threshold_severity'))} "
            f"(q={evt.get('threshold_quantile')}, {evt.get('n_exceedances')} exceedances)."
        )
        lines.append(
            f"- Shape ξ = {_fmt(evt.get('gpd_shape_xi'))}, scale σ = {_fmt(evt.get('gpd_scale_sigma'))}."
        )
        lines.append(
            f"- EVT-extrapolated collision probability = {_fmt(evt.get('evt_extrapolated_collision_prob'), '.4f')} "
            f"(95% CI {_fmt(evt.get('evt_collision_prob_ci95_low'), '.4f')}–{_fmt(evt.get('evt_collision_prob_ci95_high'), '.4f')}); "
            f"empirical rate = {_fmt(evt.get('empirical_collision_rate'), '.4f')}."
        )
    elif evt:
        lines.extend(["", f"_EVT not computed: {evt.get('reason', 'n/a')}_"])

    surv = adv.get("survival_analysis") or {}
    if surv.get("available"):
        lines.extend(["", "### Survival analysis of time-to-collision", ""])
        km = surv.get("km_overall") or {}
        lines.append(f"- Kaplan-Meier median survival time: {_fmt(km.get('median_survival_time_sec'), '.2f')} s.")
        for gk in ("behavior_tag", "odd_region"):
            lr = surv.get(f"logrank_{gk}")
            if lr:
                lines.append(
                    f"- Log-rank across {gk}: χ²={_fmt(lr.get('test_statistic'), '.2f')}, "
                    f"p={_fmt(lr.get('p_value'), '.3g')} ({lr.get('n_groups')} groups)."
                )
        cox = surv.get("cox_ph") or {}
        if cox.get("hazard_ratios"):
            lines.append(f"- Cox PH concordance index = {_fmt(cox.get('concordance_index'))}.")
            lines.append("")
            lines.append("| covariate | HR / +1 SD | 95% CI | p | q (BH) |")
            lines.append("|---|---:|---:|---:|---:|")
            for r in cox["hazard_ratios"][:10]:
                lines.append(
                    f"| {r['covariate']} | {_fmt(r['hazard_ratio_per_sd'])} | "
                    f"{_fmt(r['hr_ci95_low'])}–{_fmt(r['hr_ci95_high'])} | "
                    f"{_fmt(r['p_value'], '.3g')} | {_fmt(r.get('q_value_bh'), '.3g')} |"
                )
    elif surv:
        lines.extend(["", f"_Survival analysis not computed: {surv.get('reason', 'n/a')}_"])

    return "\n".join(lines)


def compact_for_llm(adv: Dict[str, Any]) -> Dict[str, Any]:
    """A compact, token-bounded view of the scientific evidence for the LLM."""
    if not adv or not adv.get("enabled", False):
        return {}
    crl = adv.get("cluster_robust_logistic") or {}
    shp = adv.get("shap_feature_importance") or {}
    fsig = adv.get("feature_significance") or {}
    evt = adv.get("extreme_value_analysis") or {}
    surv = adv.get("survival_analysis") or {}
    out: Dict[str, Any] = {
        "methods_note": (
            "Scene-clustered odds ratios, SHAP attribution, FDR-controlled effect sizes, "
            "EVT rare-risk extrapolation, and survival hazard ratios. Prefer effects that are "
            "FDR-significant with confidence intervals excluding the null; treat these as "
            "strong associations, still not causal proof."
        ),
    }
    if crl.get("available"):
        out["scene_clustered_odds_ratios"] = ((crl.get("gee") or {}).get("coefficients") or [])[:8]
    if shp.get("available"):
        out["shap_surrogate_cv_auc"] = shp.get("surrogate_cv_auc")
        out["shap_top_features"] = (shp.get("feature_attribution") or [])[:8]
    if fsig.get("available"):
        out["fdr_significant_features"] = [
            r for r in (fsig.get("features") or []) if r.get("significant_fdr")
        ][:8]
    if evt.get("available"):
        out["extreme_value"] = {
            "gpd_shape_xi": evt.get("gpd_shape_xi"),
            "evt_extrapolated_collision_prob": evt.get("evt_extrapolated_collision_prob"),
            "empirical_collision_rate": evt.get("empirical_collision_rate"),
        }
    if surv.get("available"):
        out["survival"] = {
            "cox_concordance": (surv.get("cox_ph") or {}).get("concordance_index"),
            "cox_top_hazard_ratios": ((surv.get("cox_ph") or {}).get("hazard_ratios") or [])[:6],
            "logrank_behavior_p": (surv.get("logrank_behavior_tag") or {}).get("p_value"),
            "logrank_region_p": (surv.get("logrank_odd_region") or {}).get("p_value"),
        }
    return out
