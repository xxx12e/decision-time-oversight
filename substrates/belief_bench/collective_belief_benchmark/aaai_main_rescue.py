from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .config import load_config
from .defenses import (
    COMBINED_V1,
    MESSAGE_SUPPORT_CHECKING,
    NO_DEFENSE,
    TOPOLOGY_AWARE_PRUNING,
    TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP,
    TOPOLOGY_AWARE_PRUNING_SUPPORT_CHECKING,
)
from .graphs import build_graph
from .reviewer_stress import BASE_SEED, StressCondition, _base_config, run_stress_trial
from .simulation import _select_committed_agents, stable_seed
from .v4 import _pagerank_power


OUT_DIR = Path("outputs/aaai_main_rescue")
BOOTSTRAP_SEED = 20260602


PREDICTOR_SPECS: dict[str, list[str]] = {
    "raw_count_only": ["actual_committed_count"],
    "degree_share": ["degree_share"],
    "sum_degree_mass": ["sum_degree_mass"],
    "count_plus_degree_share": ["actual_committed_count", "degree_share"],
    "max_degree": ["max_degree"],
    "count_plus_max_degree": ["actual_committed_count", "max_degree"],
    "pagerank_mass": ["pagerank_mass"],
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False).astype(bool)


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        avg = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg
        start = end
    return ranks


def _spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2 or np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return float("nan")
    rx = _rank_average(x_arr)
    ry = _rank_average(y_arr)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / denom) if denom else float("nan")


def _clip_prob(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _design_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    levels: dict[str, list[Any]] | None = None,
    scale: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[Any]], dict[str, tuple[float, float]]]:
    categorical_cols = categorical_cols or []
    levels = copy.deepcopy(levels) if levels is not None else {}
    scale = copy.deepcopy(scale) if scale is not None else {}
    columns: list[np.ndarray] = [np.ones(len(df), dtype=float)]
    names = ["intercept"]

    for col in numeric_cols:
        raw = df[col].to_numpy(dtype=float)
        if col not in scale:
            mean = float(np.nanmean(raw)) if len(raw) else 0.0
            std = float(np.nanstd(raw)) if len(raw) else 1.0
            if not np.isfinite(std) or std < 1e-9:
                std = 1.0
            scale[col] = (mean, std)
        mean, std = scale[col]
        columns.append((raw - mean) / std)
        names.append(col)

    for col in categorical_cols:
        if col not in levels:
            vals = sorted(str(v) for v in df[col].dropna().unique())
            levels[col] = vals
        vals = levels[col]
        for level in vals[1:]:
            columns.append((df[col].astype(str).to_numpy() == level).astype(float))
            names.append(f"{col}={level}")

    return np.column_stack(columns), names, levels, scale


def _fit_binomial_design(
    x: np.ndarray,
    y_count: np.ndarray,
    n_count: np.ndarray,
    names: list[str],
    ridge: float = 1e-6,
    max_iter: int = 200,
) -> dict[str, Any]:
    y = np.asarray(y_count, dtype=float)
    n = np.asarray(n_count, dtype=float)
    rate = np.divide(y, n, out=np.zeros_like(y), where=n > 0)
    beta = np.zeros(x.shape[1], dtype=float)
    converged = False
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        eta = x @ beta
        mu = _sigmoid(eta)
        var = np.maximum(mu * (1.0 - mu), 1e-8)
        weights = np.maximum(n * var, 1e-8)
        z = eta + (rate - mu) / var
        xtw = x.T * weights
        lhs = xtw @ x + penalty
        rhs = xtw @ z
        try:
            new_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.pinv(lhs) @ rhs
        if float(np.max(np.abs(new_beta - beta))) < 1e-8:
            beta = new_beta
            converged = True
            break
        beta = new_beta

    pred = _clip_prob(_sigmoid(x @ beta))
    loglik = float(np.sum(y * np.log(pred) + (n - y) * np.log(1.0 - pred)))
    null_rate = _clip_prob(float(np.sum(y) / max(1.0, np.sum(n))))
    null_loglik = float(np.sum(y * math.log(null_rate) + (n - y) * math.log(1.0 - null_rate)))
    deviance = _binomial_deviance(y, n, pred)
    null_deviance = _binomial_deviance(y, n, np.full_like(y, null_rate, dtype=float))
    pseudo_r2 = 1.0 - (loglik / null_loglik) if null_loglik != 0 else float("nan")
    deviance_improvement = null_deviance - deviance
    k = x.shape[1]
    sample_n = max(1.0, float(np.sum(n)))
    aic = 2 * k - 2 * loglik
    bic = math.log(sample_n) * k - 2 * loglik

    cov = np.full((k, k), np.nan)
    se = np.full(k, np.nan)
    try:
        final_mu = _clip_prob(_sigmoid(x @ beta))
        final_w = np.maximum(n * final_mu * (1.0 - final_mu), 1e-8)
        fisher = (x.T * final_w) @ x + penalty
        cov = np.linalg.pinv(fisher)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        pass
    coefficients = []
    for idx, name in enumerate(names):
        coef = float(beta[idx])
        err = float(se[idx]) if np.isfinite(se[idx]) else float("nan")
        coefficients.append(
            {
                "name": name,
                "coef": coef,
                "ci_low": coef - 1.96 * err if np.isfinite(err) else float("nan"),
                "ci_high": coef + 1.96 * err if np.isfinite(err) else float("nan"),
                "sign": "positive" if coef > 0 else "negative" if coef < 0 else "zero",
            }
        )
    return {
        "beta": beta,
        "pred": pred,
        "loglik": loglik,
        "null_loglik": null_loglik,
        "deviance": deviance,
        "null_deviance": null_deviance,
        "pseudo_r2": pseudo_r2,
        "deviance_improvement": deviance_improvement,
        "aic": float(aic),
        "bic": float(bic),
        "converged": converged,
        "coefficients": coefficients,
    }


def _fit_binomial(
    train: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> dict[str, Any]:
    x, names, levels, scale = _design_matrix(train, numeric_cols, categorical_cols)
    fit = _fit_binomial_design(
        x,
        train["false_count"].to_numpy(dtype=float),
        train["total_trials"].to_numpy(dtype=float),
        names,
    )
    fit["names"] = names
    fit["levels"] = levels
    fit["scale"] = scale
    fit["numeric_cols"] = numeric_cols
    fit["categorical_cols"] = categorical_cols or []
    return fit


def _predict_binomial(fit: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x, _, _, _ = _design_matrix(
        df,
        fit["numeric_cols"],
        fit["categorical_cols"],
        levels=fit["levels"],
        scale=fit["scale"],
    )
    return _clip_prob(_sigmoid(x @ fit["beta"]))


def _binomial_deviance(y_count: np.ndarray, n_count: np.ndarray, pred_prob: np.ndarray) -> float:
    y = np.asarray(y_count, dtype=float)
    n = np.asarray(n_count, dtype=float)
    pred = np.asarray(_clip_prob(pred_prob), dtype=float)
    saturated = np.divide(y, n, out=np.zeros_like(y), where=n > 0)
    terms = np.zeros_like(y, dtype=float)
    pos = y > 0
    neg = (n - y) > 0
    terms[pos] += y[pos] * np.log(saturated[pos] / pred[pos])
    terms[neg] += (n[neg] - y[neg]) * np.log((1.0 - saturated[neg]) / (1.0 - pred[neg]))
    return float(2.0 * np.sum(terms))


def _weighted_auc(scores: np.ndarray, y_count: np.ndarray, n_count: np.ndarray) -> float:
    score = np.asarray(scores, dtype=float)
    pos = np.asarray(y_count, dtype=float)
    neg = np.asarray(n_count, dtype=float) - pos
    total_pos = float(np.sum(pos))
    total_neg = float(np.sum(neg))
    if total_pos <= 0 or total_neg <= 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    score = score[order]
    pos = pos[order]
    neg = neg[order]
    auc_num = 0.0
    cum_neg = 0.0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        group_pos = float(np.sum(pos[start:end]))
        group_neg = float(np.sum(neg[start:end]))
        auc_num += group_pos * (cum_neg + 0.5 * group_neg)
        cum_neg += group_neg
        start = end
    return float(auc_num / (total_pos * total_neg))


def _prediction_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    y = test["false_count"].to_numpy(dtype=float)
    n = test["total_trials"].to_numpy(dtype=float)
    rate = np.divide(y, n, out=np.zeros_like(y), where=n > 0)
    p = np.asarray(_clip_prob(pred), dtype=float)
    total = max(1.0, float(np.sum(n)))
    log_loss = float(-np.sum(y * np.log(p) + (n - y) * np.log(1.0 - p)) / total)
    brier = float(np.sum(n * (rate - p) ** 2) / total)
    return {
        "heldout_log_loss": log_loss,
        "brier_score": brier,
        "auc": _weighted_auc(p, y, n),
        "spearman": _spearman(p, rate),
        "binomial_deviance": _binomial_deviance(y, n, p),
    }


def _centrality_features(graph: nx.Graph, committed: set[int]) -> dict[str, float]:
    degree = nx.degree_centrality(graph)
    pagerank = _pagerank_power(graph)
    total_degree = float(sum(degree.values()))
    if not committed:
        return {
            "sum_degree_mass": 0.0,
            "degree_share": 0.0,
            "max_degree": 0.0,
            "pagerank_mass": 0.0,
            "max_raw_degree": 0.0,
            "average_committed_degree": 0.0,
        }
    degree_values = [float(degree[node]) for node in committed]
    raw_degrees = [float(graph.degree[node]) for node in committed]
    sum_degree = float(sum(degree_values))
    return {
        "sum_degree_mass": sum_degree,
        "degree_share": sum_degree / total_degree if total_degree > 0 else 0.0,
        "max_degree": float(max(degree_values)),
        "pagerank_mass": float(sum(float(pagerank[node]) for node in committed)),
        "max_raw_degree": float(max(raw_degrees)),
        "average_committed_degree": float(np.mean(raw_degrees)),
    }


def _feature_for_v2_row(row: pd.Series, config: dict[str, Any], cache: dict[tuple[Any, ...], dict[str, float]]) -> dict[str, float]:
    key = (
        int(row["trial"]),
        float(row["minority_ratio"]),
        str(row["graph_topology"]),
        str(row["committed_placement"]),
        str(row["experiment"]),
        int(row["graph_seed"]),
    )
    if key not in cache:
        graph, source_groups = build_graph(
            int(config["n_agents"]),
            str(row["graph_topology"]),
            config["graph"],
            int(row["graph_seed"]),
        )
        placement_rng = np.random.default_rng(
            stable_seed(
                int(config["seed"]),
                "placement",
                int(row["trial"]),
                float(row["minority_ratio"]),
                str(row["graph_topology"]),
                str(row["committed_placement"]),
                str(row["experiment"]),
            )
        )
        committed = _select_committed_agents(
            graph,
            float(row["minority_ratio"]),
            str(row["committed_placement"]),
            placement_rng,
            source_groups=source_groups,
        )
        cache[key] = _centrality_features(graph, committed)
    return cache[key]


def build_condition_dataset() -> pd.DataFrame:
    v2_config = load_config("outputs/v2/config_snapshot.yaml")
    v2_trials = pd.read_csv("outputs/v2/trial_results.csv")
    trial = v2_trials[
        (v2_trials["model_type"] == "scripted")
        & (v2_trials["defense"] == NO_DEFENSE)
        & (v2_trials["experiment"].isin(["core_sweep", "hub_placement"]))
    ].copy()
    for col in ["false_consensus", "true_consensus", "no_consensus"]:
        trial[col] = _to_bool(trial[col])
    cache: dict[tuple[Any, ...], dict[str, float]] = {}
    feature_rows = [_feature_for_v2_row(row, v2_config, cache) for _, row in trial.iterrows()]
    features = pd.DataFrame(feature_rows, index=trial.index)
    trial = pd.concat([trial.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    trial["actual_committed_count"] = trial["committed_agents"].astype(int)
    trial["actual_minority_ratio"] = trial["actual_committed_count"] / int(v2_config["n_agents"])

    group_cols = [
        "experiment",
        "graph_topology",
        "committed_placement",
        "evidence_strength",
        "update_rule",
        "minority_ratio",
        "actual_committed_count",
        "actual_minority_ratio",
    ]
    agg = trial.groupby(group_cols, dropna=False).agg(
        false_count=("false_consensus", "sum"),
        true_count=("true_consensus", "sum"),
        no_count=("no_consensus", "sum"),
        total_trials=("trial", "count"),
        false_consensus_rate=("false_consensus", "mean"),
        true_consensus_rate=("true_consensus", "mean"),
        no_consensus_rate=("no_consensus", "mean"),
        average_individual_accuracy=("average_individual_accuracy", "mean"),
        time_to_consensus=("time_to_consensus", "mean"),
        intervention_cost=("intervention_cost", "mean"),
        degree_share=("degree_share", "mean"),
        sum_degree_mass=("sum_degree_mass", "mean"),
        max_degree=("max_degree", "mean"),
        pagerank_mass=("pagerank_mass", "mean"),
        max_raw_degree=("max_raw_degree", "mean"),
        average_committed_degree=("average_committed_degree", "mean"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "condition_level_dataset.csv", index=False)
    return agg


def run_heldout_prediction(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_specs = [
        ("leave_one_topology_out", "graph_topology"),
        ("leave_one_placement_rule_out", "committed_placement"),
        ("leave_one_evidence_strength_out", "evidence_strength"),
        ("leave_one_update_rule_out", "update_rule"),
    ]
    rows: list[dict[str, Any]] = []
    for split_name, split_col in split_specs:
        values = sorted(dataset[split_col].dropna().unique())
        if len(values) < 2:
            continue
        for value in values:
            train = dataset[dataset[split_col] != value].copy()
            test = dataset[dataset[split_col] == value].copy()
            if len(train) < 4 or len(test) == 0:
                continue
            for predictor_name, predictors in PREDICTOR_SPECS.items():
                fit = _fit_binomial(train, predictors)
                pred = _predict_binomial(fit, test)
                metrics = _prediction_metrics(test, pred)
                rows.append(
                    {
                        "split_type": split_name,
                        "heldout_variable": split_col,
                        "heldout_value": str(value),
                        "predictor": predictor_name,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_trials": int(train["total_trials"].sum()),
                        "test_trials": int(test["total_trials"].sum()),
                        **metrics,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "heldout_prediction.csv", index=False)
    summary_table = (
        result.groupby("predictor", as_index=False)
        .agg(
            mean_log_loss=("heldout_log_loss", "mean"),
            mean_brier=("brier_score", "mean"),
            mean_auc=("auc", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_deviance=("binomial_deviance", "mean"),
            evaluations=("heldout_log_loss", "count"),
        )
        .sort_values(["mean_log_loss", "mean_brier"], ascending=[True, True])
    )
    best_log_loss = str(summary_table.iloc[0]["predictor"]) if len(summary_table) else None
    raw = summary_table[summary_table["predictor"] == "raw_count_only"]
    best_position = summary_table[summary_table["predictor"] != "raw_count_only"].head(1)
    summary = {
        "condition_rows": int(len(dataset)),
        "total_trials": int(dataset["total_trials"].sum()),
        "best_mean_log_loss_predictor": best_log_loss,
        "raw_count_mean_log_loss": float(raw["mean_log_loss"].iloc[0]) if len(raw) else None,
        "best_position_aware_predictor": str(best_position["predictor"].iloc[0]) if len(best_position) else None,
        "best_position_aware_mean_log_loss": float(best_position["mean_log_loss"].iloc[0]) if len(best_position) else None,
        "position_aware_beats_raw_by_log_loss": bool(
            len(raw) and len(best_position) and float(best_position["mean_log_loss"].iloc[0]) < float(raw["mean_log_loss"].iloc[0])
        ),
        "summary_by_predictor": summary_table.to_dict(orient="records"),
    }
    (OUT_DIR / "heldout_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    _write_heldout_table(summary_table)
    return result, summary


def run_binomial_models(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_specs = [
        ("count_only", ["actual_committed_count"], []),
        ("count_plus_degree_share", ["actual_committed_count", "degree_share"], []),
        ("count_plus_max_degree", ["actual_committed_count", "max_degree"], []),
        (
            "count_plus_degree_share_evidence_topology",
            ["actual_committed_count", "degree_share", "evidence_strength"],
            ["graph_topology"],
        ),
        (
            "count_plus_degree_share_evidence_topology_placement",
            ["actual_committed_count", "degree_share", "evidence_strength"],
            ["graph_topology", "committed_placement"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    count_fit: dict[str, Any] | None = None
    coefficient_rows: list[dict[str, Any]] = []
    for model_name, numeric, categorical in model_specs:
        fit = _fit_binomial(dataset, numeric, categorical)
        if model_name == "count_only":
            count_fit = fit
        improvement = float(fit["pseudo_r2"] - count_fit["pseudo_r2"]) if count_fit is not None else 0.0
        key_terms = [
            coef
            for coef in fit["coefficients"]
            if coef["name"] in {"actual_committed_count", "degree_share", "max_degree", "evidence_strength"}
        ]
        for coef in fit["coefficients"]:
            coefficient_rows.append({"model": model_name, **coef})
        rows.append(
            {
                "model": model_name,
                "numeric_terms": "+".join(numeric),
                "categorical_terms": "+".join(categorical),
                "condition_rows": len(dataset),
                "total_trials": int(dataset["total_trials"].sum()),
                "pseudo_r2": fit["pseudo_r2"],
                "deviance": fit["deviance"],
                "null_deviance": fit["null_deviance"],
                "deviance_improvement": fit["deviance_improvement"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "pseudo_r2_improvement_over_count": improvement,
                "converged": fit["converged"],
                "key_coefficients_json": json.dumps(_json_safe(key_terms), sort_keys=True),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "binomial_models.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(OUT_DIR / "binomial_coefficients.csv", index=False)
    count = result[result["model"] == "count_only"].iloc[0]
    best = result.sort_values("pseudo_r2", ascending=False).iloc[0]
    summary = {
        "condition_rows": int(len(dataset)),
        "total_trials": int(dataset["total_trials"].sum()),
        "count_only_pseudo_r2": float(count["pseudo_r2"]),
        "best_model": str(best["model"]),
        "best_model_pseudo_r2": float(best["pseudo_r2"]),
        "best_model_improvement_over_count": float(best["pseudo_r2_improvement_over_count"]),
        "position_terms_positive": _position_terms_positive(pd.DataFrame(coefficient_rows)),
        "models": result.to_dict(orient="records"),
    }
    (OUT_DIR / "binomial_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    _write_binomial_table(result)
    return result, summary


def _position_terms_positive(coefficients: pd.DataFrame) -> dict[str, Any]:
    subset = coefficients[coefficients["name"].isin(["degree_share", "max_degree"])]
    return {
        "terms": subset.to_dict(orient="records"),
        "all_positive": bool(len(subset) and (subset["coef"] > 0).all()),
    }


def _aggregate_trials(trials: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    working = trials.copy()
    for col in ["false_consensus", "true_consensus", "no_consensus"]:
        working[col] = _to_bool(working[col])
    return working.groupby(group_cols, dropna=False).agg(
        n_trials=("trial", "count"),
        false_count=("false_consensus", "sum"),
        true_count=("true_consensus", "sum"),
        no_count=("no_consensus", "sum"),
        false_consensus_rate=("false_consensus", "mean"),
        true_consensus_rate=("true_consensus", "mean"),
        no_consensus_rate=("no_consensus", "mean"),
        average_individual_accuracy=("average_individual_accuracy", "mean"),
        time_to_consensus=("time_to_consensus", "mean"),
        intervention_cost=("intervention_cost", "mean"),
        committed_agents=("committed_agents", "mean"),
        committed_mean_degree=("committed_mean_degree", "mean"),
        degree_influence_mass=("degree_influence_mass", "mean"),
        pagerank_influence_mass=("pagerank_influence_mass", "mean"),
        messages_suppressed=("messages_suppressed", "mean"),
        messages_seen=("messages_seen", "mean"),
    ).reset_index()


def run_adaptive_topology_attack(trials_per_condition: int = 100) -> tuple[pd.DataFrame, dict[str, Any]]:
    placements = [
        "high_degree",
        "medium_degree_below_threshold",
        "low_degree",
        "high_betweenness",
        "dispersed",
        "mixed_high_and_below_threshold",
    ]
    defenses = [
        ("no_defense", NO_DEFENSE),
        ("support_checking", MESSAGE_SUPPORT_CHECKING),
        ("combined", COMBINED_V1),
        ("topology_aware_pruning", TOPOLOGY_AWARE_PRUNING),
        ("topology_aware_pruning_plus_confidence_cap", TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP),
        ("topology_aware_pruning_plus_support_checking", TOPOLOGY_AWARE_PRUNING_SUPPORT_CHECKING),
    ]
    config = _base_config()
    rows: list[dict[str, Any]] = []
    total = len(placements) * len(defenses) * trials_per_condition
    completed = 0
    for placement in placements:
        for defense_label, defense in defenses:
            condition = StressCondition(
                experiment="aaai_main_rescue_adaptive_topology",
                minority_ratio=0.10,
                evidence_strength=0.66,
                graph_topology="hub",
                committed_placement=placement,
                defense=defense,
                defense_strength=0.0 if defense == NO_DEFENSE else 1.0,
                adversary_mode="fabricated_self_support",
                fabrication_probability=1.0,
                memory_length=3,
                update_rule="bayesian_like_agent",
                social_weight=2.1,
            )
            for trial in range(trials_per_condition):
                result = run_stress_trial(condition, trial, config)
                result["defense_label"] = defense_label
                rows.append(result)
                completed += 1
                if completed % 600 == 0 or completed == total:
                    print(f"adaptive topology attack {completed}/{total}")
    trials = pd.DataFrame(rows)
    trials.to_csv(OUT_DIR / "adaptive_topology_attack_trials.csv", index=False)
    aggregate = _aggregate_trials(trials, ["committed_placement", "defense_label"])
    aggregate.to_csv(OUT_DIR / "adaptive_topology_attack.csv", index=False)
    topology_rows = aggregate[aggregate["defense_label"].str.startswith("topology_aware_pruning")]
    worst = topology_rows.sort_values("false_consensus_rate", ascending=False).head(1)
    best_by_placement = {}
    for placement, subset in aggregate.groupby("committed_placement"):
        best = subset.sort_values(["false_consensus_rate", "true_consensus_rate"], ascending=[True, False]).iloc[0]
        best_by_placement[placement] = best.to_dict()
    summary = {
        "scope": {
            "graph": "hub",
            "n": 32,
            "rounds": 8,
            "evidence": 0.66,
            "protocol": "fabricated_self_support",
            "minority_ratio": 0.10,
            "trials_per_condition": trials_per_condition,
        },
        "topology_pruning_worst_case": worst.iloc[0].to_dict() if len(worst) else None,
        "topology_pruning_survives_all_adaptive_placements": bool(
            len(topology_rows) and float(topology_rows["false_consensus_rate"].max()) <= 0.05
        ),
        "best_by_placement": best_by_placement,
        "interpretation": _adaptive_interpretation(aggregate),
    }
    (OUT_DIR / "adaptive_topology_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    _write_adaptive_table(aggregate)
    return aggregate, summary


def _adaptive_interpretation(aggregate: pd.DataFrame) -> str:
    topology = aggregate[aggregate["defense_label"].str.startswith("topology_aware_pruning")]
    if not len(topology):
        return "No topology-aware pruning rows were produced."
    worst = float(topology["false_consensus_rate"].max())
    if worst <= 0.05:
        return "Topology-aware pruning remained low-false-consensus across the targeted adaptive placements."
    return (
        "Topology-aware pruning degraded under at least one adaptive placement; report it as a targeted baseline, "
        "not a globally reliable defense."
    )


def maybe_local_llm_sanity() -> dict[str, Any]:
    cfg = load_config("configs/v2.yaml").get("local_llm", {})
    model_path = Path(str(cfg.get("model_path", ""))) if cfg.get("model_path") else None
    missing = []
    if not cfg.get("enabled", False):
        missing.append("local_llm.enabled is false")
    if model_path is None or not model_path.exists():
        missing.append("configured local model path is unavailable")
    if importlib.util.find_spec("transformers") is None:
        missing.append("transformers is unavailable")
    if importlib.util.find_spec("torch") is None:
        missing.append("torch is unavailable")
    if missing:
        status = {
            "status": "skipped",
            "reason": "; ".join(missing),
            "no_external_apis": True,
            "paper_wording": "Local LLM validation is left to future work; the present study prioritizes controlled leakage-audited mechanism analysis.",
        }
        (OUT_DIR / "local_llm_sanity_summary.json").write_text(json.dumps(_json_safe(status), indent=2), encoding="utf-8")
        return status
    status = {
        "status": "not_run",
        "reason": "Local dependencies and path appear available, but this rescue script does not invoke the model without an explicit local-model run command.",
        "no_external_apis": True,
    }
    (OUT_DIR / "local_llm_sanity_summary.json").write_text(json.dumps(_json_safe(status), indent=2), encoding="utf-8")
    return status


def maybe_scale_sensitivity(run_scale: bool, trials_per_condition: int = 50) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not run_scale:
        summary = {
            "status": "not_run",
            "reason": "Optional scale sensitivity was not requested for this run; primary rescue scope prioritized held-out, binomial, adaptive attack, and LLM feasibility checks.",
        }
        (OUT_DIR / "scale_sensitivity_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        return None, summary
    config = _base_config()
    rows: list[dict[str, Any]] = []
    ratios = [0.0, 0.05, 0.10, 0.20]
    sizes = [16, 32, 64]
    topologies = ["hub", "random"]
    placements = ["high_degree", "low_degree"]
    total = len(sizes) * len(topologies) * len(placements) * len(ratios) * trials_per_condition
    completed = 0
    for n_agents in sizes:
        config_n = copy.deepcopy(config)
        config_n["n_agents"] = n_agents
        for topology in topologies:
            for placement in placements:
                for ratio in ratios:
                    condition = StressCondition(
                        experiment="aaai_main_rescue_scale_sensitivity",
                        minority_ratio=ratio,
                        evidence_strength=0.66,
                        graph_topology=topology,
                        committed_placement=placement,
                        defense=NO_DEFENSE,
                        defense_strength=0.0,
                        adversary_mode="honest_current",
                        fabrication_probability=0.0,
                        memory_length=3,
                        update_rule="bayesian_like_agent",
                        social_weight=2.1,
                    )
                    for trial in range(trials_per_condition):
                        result = run_stress_trial(condition, trial, config_n)
                        result["n_agents"] = n_agents
                        rows.append(result)
                        completed += 1
                        if completed % 600 == 0 or completed == total:
                            print(f"scale sensitivity {completed}/{total}")
    trials = pd.DataFrame(rows)
    trials.to_csv(OUT_DIR / "scale_sensitivity_trials.csv", index=False)
    aggregate = _aggregate_trials(trials, ["n_agents", "graph_topology", "committed_placement", "minority_ratio"])
    aggregate.to_csv(OUT_DIR / "scale_sensitivity.csv", index=False)
    summary_rows = []
    for n_agents, subset in aggregate.groupby("n_agents"):
        count_r2 = _linear_r2(subset, ["minority_ratio"], "false_consensus_rate")
        mass_r2 = _linear_r2(subset, ["degree_influence_mass"], "false_consensus_rate")
        summary_rows.append(
            {
                "n_agents": int(n_agents),
                "minority_ratio_r2": count_r2,
                "degree_influence_mass_r2": mass_r2,
                "position_aware_better": bool(mass_r2 > count_r2),
            }
        )
    summary = {
        "status": "run",
        "trials_per_condition": trials_per_condition,
        "summary_by_size": summary_rows,
        "position_aware_better_all_sizes": bool(all(row["position_aware_better"] for row in summary_rows)),
    }
    (OUT_DIR / "scale_sensitivity_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return aggregate, summary


def _linear_r2(df: pd.DataFrame, predictors: list[str], target: str) -> float:
    if len(df) < len(predictors) + 2:
        return float("nan")
    x = df[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = df[target].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def _write_heldout_table(summary_table: pd.DataFrame) -> None:
    path = OUT_DIR / "table_heldout_prediction.tex"
    top = summary_table.head(7).copy()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Predictor & Log loss & Brier & AUC & $\\rho$ & Dev. \\\\",
        "\\midrule",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"{_tex_label(row['predictor'])} & {row['mean_log_loss']:.3f} & {row['mean_brier']:.3f} & "
            f"{row['mean_auc']:.3f} & {row['mean_spearman']:.3f} & {row['mean_deviance']:.1f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Held-out prediction averaged across leave-one-topology, placement, evidence-strength, and update-rule splits. Lower log loss, Brier score, and deviance are better.}",
            "\\label{tab:heldout-prediction}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_binomial_table(result: pd.DataFrame) -> None:
    path = OUT_DIR / "table_binomial_models.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & McFadden $R^2$ & Deviance & AIC & $\\Delta R^2$ \\\\",
        "\\midrule",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"{_tex_label(row['model'])} & {row['pseudo_r2']:.3f} & {row['deviance']:.1f} & "
            f"{row['aic']:.1f} & {row['pseudo_r2_improvement_over_count']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Condition-level binomial models for false consensus. Improvements are relative to count-only.}",
            "\\label{tab:binomial-models}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_adaptive_table(aggregate: pd.DataFrame) -> None:
    path = OUT_DIR / "table_adaptive_topology_attack.tex"
    display = aggregate.copy()
    display["sort_defense"] = display["defense_label"].map(
        {
            "no_defense": 0,
            "support_checking": 1,
            "combined": 2,
            "topology_aware_pruning": 3,
            "topology_aware_pruning_plus_confidence_cap": 4,
            "topology_aware_pruning_plus_support_checking": 5,
        }
    )
    display = display.sort_values(["committed_placement", "sort_defense"])
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Placement & Defense & False & True & No & Cost \\\\",
        "\\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"{_tex_label(row['committed_placement'])} & {_tex_label(row['defense_label'])} & "
            f"{row['false_consensus_rate']:.3f} & {row['true_consensus_rate']:.3f} & "
            f"{row['no_consensus_rate']:.3f} & {row['intervention_cost']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Targeted adaptive fabricated-citation attack on topology-aware pruning. Scope: hub graph, $n=32$, evidence 0.66, 10\\% nominal committed minority, 100 seeds per condition.}",
            "\\label{tab:adaptive-topology-attack}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tex_label(value: Any) -> str:
    text = str(value).replace("_", " ")
    return text[:1].upper() + text[1:]


def write_decision_memo(
    heldout_summary: dict[str, Any],
    binomial_summary: dict[str, Any],
    adaptive_summary: dict[str, Any],
    llm_summary: dict[str, Any],
    scale_summary: dict[str, Any],
) -> None:
    heldout_support = bool(heldout_summary.get("position_aware_beats_raw_by_log_loss"))
    binomial_support = float(binomial_summary.get("best_model_improvement_over_count") or 0.0) > 0
    adaptive_survives = bool(adaptive_summary.get("topology_pruning_survives_all_adaptive_placements"))
    llm_status = llm_summary.get("status")
    scale_status = scale_summary.get("status")
    if heldout_support and binomial_support and adaptive_survives:
        recommendation = "AAAI main is more defensible, with topology-aware pruning still framed as targeted."
    elif heldout_support and binomial_support:
        recommendation = "AAAI main is plausible if the defense story is downgraded to a threat-model-dependent baseline comparison."
    else:
        recommendation = "Prefer workshop or backup unless the main paper foregrounds limitations and treats the measurement result as preliminary."

    lines = [
        "# AAAI Main Rescue Memo",
        "",
        "No broad benchmark was run. This package adds held-out condition-level prediction, controlled binomial models, and a targeted adaptive fabricated-citation attack.",
        "",
        "## 1. Held-out Prediction",
        (
            f"Position-aware support by held-out log loss: `{heldout_support}`. "
            f"Best position-aware predictor: `{heldout_summary.get('best_position_aware_predictor')}` "
            f"with mean log loss `{_fmt(heldout_summary.get('best_position_aware_mean_log_loss'))}`; "
            f"raw count mean log loss `{_fmt(heldout_summary.get('raw_count_mean_log_loss'))}`."
        ),
        "",
        "## 2. Controlled Binomial Models",
        (
            f"Best model: `{binomial_summary.get('best_model')}` with pseudo-R2 "
            f"`{_fmt(binomial_summary.get('best_model_pseudo_r2'))}`. "
            f"Improvement over count-only: `{_fmt(binomial_summary.get('best_model_improvement_over_count'))}`."
        ),
        "",
        "## 3. Adaptive Topology Attack",
        (
            f"Topology-aware pruning survives all targeted adaptive placements at false-consensus <= 0.05: "
            f"`{adaptive_survives}`. Interpretation: {adaptive_summary.get('interpretation')}"
        ),
        "",
        "## 4. Local LLM Sanity",
        f"Status: `{llm_status}`. No external APIs were used.",
        "",
        "## 5. Scale Sensitivity",
        f"Status: `{scale_status}`. Summary: `{scale_summary}`.",
        "",
        "## 6. Exact Paper Text To Update",
        "- Add a held-out prediction paragraph reporting log loss/Brier/deviance comparisons from `heldout_prediction.csv`.",
        "- Add a controlled binomial-model table using `table_binomial_models.tex`; state that aggregate OLS is descriptive and binomial models are condition-level robustness checks.",
        "- Add the topology-aware pruning rule definition if not already present: prune messages from senders with degree centrality >= 0.35 using graph metadata only.",
        "- If the adaptive attack degrades pruning, state that topology-aware pruning is a targeted baseline and not a general defense ranking.",
        "- Keep local LLM validation as future work unless `local_llm_sanity.csv` exists with completed rows.",
        "",
        "## 7. Venue Recommendation",
        recommendation,
    ]
    (OUT_DIR / "AAAI_MAIN_RESCUE_MEMO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def run_all(run_scale: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    dataset = build_condition_dataset()
    heldout, heldout_summary = run_heldout_prediction(dataset)
    binomial, binomial_summary = run_binomial_models(dataset)
    adaptive, adaptive_summary = run_adaptive_topology_attack()
    llm_summary = maybe_local_llm_sanity()
    _, scale_summary = maybe_scale_sensitivity(run_scale)
    write_decision_memo(heldout_summary, binomial_summary, adaptive_summary, llm_summary, scale_summary)
    manifest = {
        "condition_rows": int(len(dataset)),
        "heldout_rows": int(len(heldout)),
        "binomial_models": int(len(binomial)),
        "adaptive_rows": int(len(adaptive)),
        "local_llm_status": llm_summary.get("status"),
        "scale_status": scale_summary.get("status"),
        "outputs": [
            "condition_level_dataset.csv",
            "heldout_prediction.csv",
            "table_heldout_prediction.tex",
            "heldout_summary.json",
            "binomial_models.csv",
            "binomial_coefficients.csv",
            "table_binomial_models.tex",
            "binomial_summary.json",
            "adaptive_topology_attack.csv",
            "adaptive_topology_attack_trials.csv",
            "table_adaptive_topology_attack.tex",
            "adaptive_topology_summary.json",
            "local_llm_sanity_summary.json",
            "scale_sensitivity_summary.json",
            "AAAI_MAIN_RESCUE_MEMO.md",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AAAI main-track rescue analyses.")
    parser.add_argument("--run-scale", action="store_true", help="Run optional n=16/32/64 scale sensitivity.")
    args = parser.parse_args()
    manifest = run_all(run_scale=args.run_scale)
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
