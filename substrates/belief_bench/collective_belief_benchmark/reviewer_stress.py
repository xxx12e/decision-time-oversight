from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .config import load_config
from .defenses import (
    COMBINED_V1,
    EVIDENCE_ANCHORING,
    MESSAGE_SUPPORT_CHECKING,
    NO_DEFENSE,
    UNSUPPORTED_QUARANTINE,
    social_signal,
)
from .graphs import build_graph
from .schema import AgentState, Message, PublicMessage, UpdateStats, probability_from_logodds, sign_to_state
from .simulation import (
    Condition,
    _apply_update_rule,
    _make_agents,
    _population_metrics,
    _private_true_signal_fraction,
    _select_committed_agents,
    _update_rule_social_scale,
    stable_seed,
)
from .v4 import _centrality_masses, _ols_r2, _spearman


OUT_DIR = Path("outputs/reviewer_stress")
PLOT_DIR = OUT_DIR / "plots"
BASE_SEED = 20260601


DEFENSES = [
    NO_DEFENSE,
    MESSAGE_SUPPORT_CHECKING,
    EVIDENCE_ANCHORING,
    UNSUPPORTED_QUARANTINE,
    COMBINED_V1,
]

ADVERSARY_SPECS = [
    ("honest_current", 0.0),
    ("no_citation", 0.0),
    ("fabricated_self_support", 1.0),
    ("coordinated_fabrication", 1.0),
    ("mixed_fabrication", 0.0),
    ("mixed_fabrication", 0.25),
    ("mixed_fabrication", 0.50),
    ("mixed_fabrication", 0.75),
    ("mixed_fabrication", 1.0),
]


@dataclass(frozen=True)
class StressCondition:
    experiment: str
    minority_ratio: float
    evidence_strength: float
    graph_topology: str
    committed_placement: str
    defense: str
    adversary_mode: str = "honest_current"
    fabrication_probability: float = 0.0
    memory_length: int = 3
    update_rule: str = "bayesian_like_agent"
    defense_strength: float = 1.0
    social_weight: float = 2.1
    committed_logodds: float = 4.0
    committed_confidence: float = 0.99
    pair_id: int | None = None
    sensitivity_parameter: str = "default"
    sensitivity_value: float = 0.0


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
        return float(value)
    return value


def _bootstrap_ci(values: pd.Series | np.ndarray, seed: int, iterations: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(iterations, len(arr)))
    means = arr[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _deterministic_unit(*parts: object) -> float:
    return stable_seed(BASE_SEED, "unit", *parts) / float(2**32 - 1)


def _maybe_fabricate_committed_message(
    message: Message,
    condition: StressCondition,
    round_index: int,
) -> Message:
    if condition.adversary_mode in {"honest_current", "no_citation"}:
        return replace(message, evidence_signal=None, supported=False)
    if condition.adversary_mode in {"fabricated_self_support", "coordinated_fabrication"}:
        return replace(message, evidence_signal=message.claim_sign, supported=True)
    if condition.adversary_mode == "mixed_fabrication":
        fabricate = _deterministic_unit(
            condition.experiment,
            condition.minority_ratio,
            condition.evidence_strength,
            condition.committed_placement,
            condition.defense,
            condition.pair_id,
            condition.fabrication_probability,
            round_index,
            message.sender_id,
        ) < condition.fabrication_probability
        if fabricate:
            return replace(message, evidence_signal=message.claim_sign, supported=True)
        return replace(message, evidence_signal=None, supported=False)
    raise ValueError(f"Unknown adversary mode: {condition.adversary_mode}")


def _citation_status(message: Message | PublicMessage) -> str:
    if message.evidence_signal is None:
        return "no_citation"
    if message.evidence_signal == message.claim_sign:
        return "supported_citation"
    return "unsupported_citation"


def _base_config() -> dict[str, Any]:
    config = load_config("configs/v2.yaml")
    config = copy.deepcopy(config)
    config["n_agents"] = 32
    config["rounds"] = 8
    config["consensus_threshold"] = 0.8
    config["balanced_private_evidence"] = True
    return config


def _shared_trial_id(condition: StressCondition, trial_index: int) -> int:
    return int(condition.pair_id if condition.pair_id is not None else trial_index)


def _make_public_messages(
    agents: list[AgentState],
    graph: nx.Graph,
    condition: StressCondition,
    round_index: int,
    n_agents: int,
) -> tuple[dict[int, PublicMessage], dict[str, int]]:
    outbound: dict[int, PublicMessage] = {}
    citation_counts = {
        "committed_no_citation": 0,
        "committed_unsupported_citation": 0,
        "committed_supported_citation": 0,
        "normal_no_citation": 0,
        "normal_unsupported_citation": 0,
        "normal_supported_citation": 0,
    }
    for agent in agents:
        message = agent.make_message(round_index, condition.committed_confidence)
        if agent.is_committed:
            message = _maybe_fabricate_committed_message(message, condition, round_index)
        status = _citation_status(message)
        prefix = "committed" if agent.is_committed else "normal"
        citation_counts[f"{prefix}_{status}"] += 1
        outbound[agent.agent_id] = message.to_public(
            sender_degree_centrality=graph.degree[agent.agent_id] / max(1, n_agents - 1)
        )
    return outbound, citation_counts


def run_stress_trial(condition: StressCondition, trial_index: int, config: dict[str, Any]) -> dict[str, Any]:
    n_agents = int(config["n_agents"])
    trial_id = _shared_trial_id(condition, trial_index)
    paired = condition.experiment == "cross_size_power"

    state_seed = stable_seed(BASE_SEED, condition.experiment, "state", trial_id, condition.evidence_strength)
    true_sign = int(np.random.default_rng(state_seed).choice([1, -1]))
    false_sign = -true_sign

    graph_seed = stable_seed(BASE_SEED, condition.experiment, "graph", trial_id, n_agents, condition.graph_topology)
    graph, source_groups = build_graph(n_agents, condition.graph_topology, config["graph"], graph_seed)

    placement_rng = np.random.default_rng(
        stable_seed(
            BASE_SEED,
            condition.experiment,
            "placement",
            trial_id,
            condition.minority_ratio,
            condition.graph_topology,
            condition.committed_placement,
        )
    )
    committed_agents = _select_committed_agents(
        graph,
        condition.minority_ratio,
        condition.committed_placement,
        placement_rng,
        source_groups=source_groups,
    )

    evidence_seed = stable_seed(
        BASE_SEED,
        condition.experiment,
        "paired_evidence" if paired else "evidence",
        trial_id,
        condition.evidence_strength,
        condition.graph_topology,
        condition.update_rule,
    )
    evidence_rng = np.random.default_rng(evidence_seed)
    agents = _make_agents(
        n_agents,
        committed_agents,
        source_groups,
        true_sign,
        condition.evidence_strength,
        condition.committed_logodds,
        evidence_rng,
        balanced_private_evidence=True,
    )

    degree_values = [degree for _, degree in graph.degree()]
    committed_degrees = [graph.degree[node] for node in committed_agents]
    centrality = _centrality_masses(graph, committed_agents)

    update_cfg = config["update"]
    threshold = float(config["consensus_threshold"])
    max_logodds = float(update_cfg.get("max_logodds", 6.0))
    base_anchor = float(update_cfg.get("evidence_anchor", 0.12))
    base_confidence_cap = float(update_cfg.get("confidence_cap", 0.95))
    decay_base = float(update_cfg.get("memory_decay", 0.55))
    defense_strength = 0.0 if condition.defense == NO_DEFENSE else condition.defense_strength

    stats = UpdateStats()
    consensus_time: int | None = None
    round_accuracies = [float(_population_metrics(agents, true_sign, threshold)["average_belief_accuracy"])]
    citation_counts = {
        "committed_no_citation": 0,
        "committed_unsupported_citation": 0,
        "committed_supported_citation": 0,
        "normal_no_citation": 0,
        "normal_unsupported_citation": 0,
        "normal_supported_citation": 0,
    }

    initial_metrics = _population_metrics(agents, true_sign, threshold)
    if initial_metrics["consensus"] != "none":
        consensus_time = 0

    for round_index in range(1, int(config["rounds"]) + 1):
        outbound, round_citations = _make_public_messages(agents, graph, condition, round_index, n_agents)
        for key, value in round_citations.items():
            citation_counts[key] += value

        for agent in agents:
            messages = [outbound[int(neighbor)] for neighbor in graph.neighbors(agent.agent_id)]
            if agent.is_committed:
                continue
            agent.memory.extend(messages)
            while agent.memory and agent.memory[0].round_index <= round_index - condition.memory_length:
                agent.memory.popleft()

        new_logodds: dict[int, float] = {}
        for agent in agents:
            if agent.is_committed:
                continue
            signal, profile = social_signal(
                agent=agent,
                messages=agent.memory,
                current_round=round_index,
                defense=condition.defense,
                strength=defense_strength,
                base_confidence_cap=base_confidence_cap,
                decay_base=decay_base,
                stats=stats,
                update_rule=condition.update_rule,
            )
            signal = _update_rule_social_scale(condition.update_rule, signal)
            anchor = min(0.90, base_anchor + profile.evidence_anchor_boost)
            updated = (1.0 - anchor) * agent.belief_logodds + anchor * agent.private_llr
            updated += condition.social_weight * signal
            updated = _apply_update_rule(agent.belief_logodds, updated, condition.update_rule)
            new_logodds[agent.agent_id] = max(-max_logodds, min(max_logodds, updated))

        for agent in agents:
            if agent.agent_id in new_logodds:
                agent.belief_logodds = new_logodds[agent.agent_id]

        metrics = _population_metrics(agents, true_sign, threshold)
        round_accuracies.append(float(metrics["average_belief_accuracy"]))
        if consensus_time is None and metrics["consensus"] != "none":
            consensus_time = round_index

    final_metrics = _population_metrics(agents, true_sign, threshold)
    time_to_consensus = consensus_time if consensus_time is not None else int(config["rounds"]) + 1
    messages_seen = max(0, stats.messages_seen)
    unsupported_seen = max(0, stats.unsupported_seen)
    support_pass_rate = (messages_seen - unsupported_seen) / messages_seen if messages_seen else 0.0
    unsupported_suppression_rate = stats.unsupported_suppressed / unsupported_seen if unsupported_seen else 0.0

    result = {
        **asdict(condition),
        "trial": trial_index,
        "pair_id": trial_id,
        "seed": stable_seed(BASE_SEED, "stress", condition.experiment, trial_index, condition.defense, condition.adversary_mode),
        "graph_seed": graph_seed,
        "state_seed": state_seed,
        "evidence_seed": evidence_seed,
        "true_state": sign_to_state(true_sign),
        "false_state": sign_to_state(false_sign),
        "committed_agents": len(committed_agents),
        "normal_agents": int(final_metrics["normal_agents"]),
        "private_true_signal_fraction": float(_private_true_signal_fraction(agents, true_sign)),
        "graph_edges": graph.number_of_edges(),
        "graph_mean_degree": float(np.mean(degree_values)),
        "graph_max_degree": int(max(degree_values) if degree_values else 0),
        "committed_mean_degree": float(np.mean(committed_degrees)) if committed_degrees else 0.0,
        "committed_degree_ratio": (
            float(np.mean(committed_degrees) / np.mean(degree_values))
            if committed_degrees and np.mean(degree_values) > 0
            else 0.0
        ),
        "degree_influence_mass": float(centrality["degree_influence_mass"]),
        "betweenness_influence_mass": float(centrality["betweenness_influence_mass"]),
        "pagerank_influence_mass": float(centrality["pagerank_influence_mass"]),
        "average_committed_degree": float(centrality["average_committed_degree"]),
        "consensus_state": final_metrics["consensus"],
        "true_consensus": bool(final_metrics["true_consensus"]),
        "false_consensus": bool(final_metrics["false_consensus"]),
        "no_consensus": bool(final_metrics["consensus"] == "none"),
        "time_to_consensus": int(time_to_consensus),
        "final_true_fraction": float(final_metrics["true_fraction"]),
        "final_false_fraction": float(final_metrics["false_fraction"]),
        "final_average_belief_accuracy": float(final_metrics["average_belief_accuracy"]),
        "average_individual_accuracy": float(final_metrics["average_belief_accuracy"]),
        "trajectory_average_belief_accuracy": float(np.mean(round_accuracies)),
        "intervention_cost": float(stats.normalized_cost()),
        "messages_seen": int(stats.messages_seen),
        "messages_used": int(stats.messages_used),
        "messages_suppressed": int(stats.messages_suppressed),
        "unsupported_seen": int(stats.unsupported_seen),
        "unsupported_suppressed": int(stats.unsupported_suppressed),
        "support_check_pass_rate": float(support_pass_rate),
        "unsupported_suppression_rate": float(unsupported_suppression_rate),
        **citation_counts,
    }
    return result


def _run_conditions(conditions: list[StressCondition], trials: int, config: dict[str, Any], progress_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(conditions) * trials
    completed = 0
    for condition in conditions:
        for trial in range(trials):
            rows.append(run_stress_trial(condition, trial, config))
            completed += 1
            if completed % 2000 == 0 or completed == total:
                print(f"{progress_prefix} {completed}/{total}")
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, group_cols: list[str], seed_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, subset in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_trials": int(len(subset)),
                "false_consensus_rate": float(subset["false_consensus"].mean()),
                "true_consensus_rate": float(subset["true_consensus"].mean()),
                "no_consensus_rate": float(subset["no_consensus"].mean()),
                "average_individual_accuracy": float(subset["average_individual_accuracy"].mean()),
                "time_to_consensus": float(subset["time_to_consensus"].mean()),
                "intervention_cost": float(subset["intervention_cost"].mean()),
                "support_check_pass_rate": float(subset["support_check_pass_rate"].mean()),
                "unsupported_suppression_rate": float(subset["unsupported_suppression_rate"].mean()),
                "degree_influence_mass": float(subset["degree_influence_mass"].mean()),
                "betweenness_influence_mass": float(subset["betweenness_influence_mass"].mean()),
                "pagerank_influence_mass": float(subset["pagerank_influence_mass"].mean()),
                "average_committed_degree": float(subset["average_committed_degree"].mean()),
                "committed_degree_ratio": float(subset["committed_degree_ratio"].mean()),
            }
        )
        for source, output in [
            ("false_consensus", "false_consensus_rate"),
            ("true_consensus", "true_consensus_rate"),
            ("no_consensus", "no_consensus_rate"),
            ("average_individual_accuracy", "average_individual_accuracy"),
            ("time_to_consensus", "time_to_consensus"),
            ("intervention_cost", "intervention_cost"),
            ("support_check_pass_rate", "support_check_pass_rate"),
            ("unsupported_suppression_rate", "unsupported_suppression_rate"),
        ]:
            low, high = _bootstrap_ci(
                subset[source],
                seed=stable_seed(BASE_SEED, seed_prefix, output, *key),
                iterations=1000,
            )
            row[f"{output}_ci_low"] = low
            row[f"{output}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose_citation_behavior(fabricated_trials: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    honest = fabricated_trials[fabricated_trials["adversary_mode"] == "honest_current"].copy()
    records: list[dict[str, Any]] = []
    for agent_type in ["committed", "normal"]:
        no_count = int(honest[f"{agent_type}_no_citation"].sum())
        unsupported_count = int(honest[f"{agent_type}_unsupported_citation"].sum())
        supported_count = int(honest[f"{agent_type}_supported_citation"].sum())
        total = no_count + unsupported_count + supported_count
        records.append(
            {
                "agent_type": agent_type,
                "messages": total,
                "no_citation_count": no_count,
                "unsupported_citation_count": unsupported_count,
                "supported_citation_count": supported_count,
                "no_citation_fraction": no_count / total if total else 0.0,
                "unsupported_citation_fraction": unsupported_count / total if total else 0.0,
                "supported_citation_fraction": supported_count / total if total else 0.0,
            }
        )
    by_type = pd.DataFrame(records)
    summary = {
        "scope": "honest_current adversary rows from targeted fabricated-citation stress test",
        "rows": by_type.to_dict(orient="records"),
    }
    by_type.to_csv(OUT_DIR / "citation_behavior_by_agent_type.csv", index=False)
    (OUT_DIR / "citation_behavior_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return by_type, summary


def run_fabricated_citation(config: dict[str, Any], trials: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conditions: list[StressCondition] = []
    for adversary_mode, fabrication_probability in ADVERSARY_SPECS:
        for ratio in [0.05, 0.10, 0.20]:
            for defense in DEFENSES:
                conditions.append(
                    StressCondition(
                        experiment="fabricated_citation",
                        minority_ratio=ratio,
                        evidence_strength=0.66,
                        graph_topology="hub",
                        committed_placement="high_degree",
                        defense=defense,
                        defense_strength=0.0 if defense == NO_DEFENSE else 1.0,
                        adversary_mode=adversary_mode,
                        fabrication_probability=fabrication_probability,
                    )
                )
    trial_df = _run_conditions(conditions, trials, config, "fabricated citation")
    aggregate = _aggregate(
        trial_df,
        ["adversary_mode", "fabrication_probability", "minority_ratio", "defense"],
        "fabricated_citation",
    )
    aggregate.to_csv(OUT_DIR / "fabricated_citation_results.csv", index=False)

    key = aggregate[
        (aggregate["minority_ratio"] == 0.10)
        & (aggregate["defense"].isin([MESSAGE_SUPPORT_CHECKING, COMBINED_V1]))
        & (aggregate["adversary_mode"].isin(["honest_current", "fabricated_self_support", "coordinated_fabrication"]))
    ]
    degradation_rows = []
    for defense in [MESSAGE_SUPPORT_CHECKING, COMBINED_V1]:
        honest = key[(key["defense"] == defense) & (key["adversary_mode"] == "honest_current")]
        fabricated = key[(key["defense"] == defense) & (key["adversary_mode"] == "fabricated_self_support")]
        if not honest.empty and not fabricated.empty:
            degradation_rows.append(
                {
                    "defense": defense,
                    "minority_ratio": 0.10,
                    "honest_false_consensus_rate": float(honest["false_consensus_rate"].iloc[0]),
                    "fabricated_self_support_false_consensus_rate": float(fabricated["false_consensus_rate"].iloc[0]),
                    "degradation": float(fabricated["false_consensus_rate"].iloc[0] - honest["false_consensus_rate"].iloc[0]),
                }
            )
    summary = {
        "scope": "hub graph, high-degree committed placement, n=32, T=8, evidence_strength=0.66",
        "trials_per_condition": trials,
        "adversary_modes": [{"mode": mode, "fabrication_probability": prob} for mode, prob in ADVERSARY_SPECS],
        "degradation_examples": degradation_rows,
        "note": "coordinated_fabrication is operationalized as protocol-visible mutually consistent fabricated evidence_signal fields; the current schema has no citation-authentication channel.",
    }
    (OUT_DIR / "fabricated_citation_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return trial_df, aggregate, summary


def run_cross_size_power(config: dict[str, Any], trials: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conditions: list[StressCondition] = []
    for evidence in [0.58, 0.66, 0.74]:
        for label, ratio, placement in [
            ("five_percent_high_degree", 0.05, "high_degree"),
            ("twenty_percent_low_degree", 0.20, "low_degree"),
        ]:
            conditions.append(
                StressCondition(
                    experiment="cross_size_power",
                    minority_ratio=ratio,
                    evidence_strength=evidence,
                    graph_topology="hub",
                    committed_placement=placement,
                    defense=NO_DEFENSE,
                    defense_strength=0.0,
                    adversary_mode="honest_current",
                    pair_id=None,
                    sensitivity_parameter=label,
                )
            )
    trial_df = _run_conditions(conditions, trials, config, "cross-size power")
    trial_df.to_csv(OUT_DIR / "cross_size_power_trial_results.csv", index=False)
    summary_df = _aggregate(
        trial_df,
        ["evidence_strength", "sensitivity_parameter", "minority_ratio", "committed_placement"],
        "cross_size_power",
    )
    summary_df.to_csv(OUT_DIR / "cross_size_power_results.csv", index=False)

    tests = []
    for evidence, subset in trial_df.groupby("evidence_strength"):
        a = subset[subset["sensitivity_parameter"] == "five_percent_high_degree"].sort_values("pair_id")
        b = subset[subset["sensitivity_parameter"] == "twenty_percent_low_degree"].sort_values("pair_id")
        merged = a[["pair_id", "false_consensus"]].merge(
            b[["pair_id", "false_consensus"]],
            on="pair_id",
            suffixes=("_five_high", "_twenty_low"),
        )
        diffs = merged["false_consensus_five_high"].astype(float) - merged["false_consensus_twenty_low"].astype(float)
        low, high = _bootstrap_ci(diffs, stable_seed(BASE_SEED, "cross_size_diff", evidence), iterations=5000)
        observed = float(diffs.mean())
        rng = np.random.default_rng(stable_seed(BASE_SEED, "cross_size_permutation", evidence))
        signs = rng.choice([-1.0, 1.0], size=(10000, len(diffs)))
        null_means = (signs * diffs.to_numpy(dtype=float)).mean(axis=1)
        p_two_sided = float(np.mean(np.abs(null_means) >= abs(observed)))
        tests.append(
            {
                "evidence_strength": float(evidence),
                "paired_trials": int(len(diffs)),
                "paired_difference_five_high_minus_twenty_low": observed,
                "paired_difference_ci_low": low,
                "paired_difference_ci_high": high,
                "paired_sign_flip_p_two_sided": p_two_sided,
                "five_high_greater": bool(observed > 0 and low > 0),
            }
        )
    summary = {
        "scope": "paired-seed hub graph no-defense cross-size comparison",
        "trials_per_condition": trials,
        "condition_summary": summary_df.to_dict(orient="records"),
        "paired_tests": tests,
        "all_evidence_strengths_support_greater": bool(all(item["five_high_greater"] for item in tests)),
    }
    (OUT_DIR / "cross_size_power_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return trial_df, summary_df, summary


def _predictor_summary(df: pd.DataFrame) -> dict[str, Any]:
    predictors = ["minority_ratio", "degree_influence_mass", "betweenness_influence_mass", "pagerank_influence_mass"]
    result: dict[str, Any] = {}
    for predictor in predictors:
        result[predictor] = {
            "spearman": _spearman(df[predictor], df["false_consensus_rate"]),
            "ols_r2": _ols_r2(df, [predictor], "false_consensus_rate")["r2"],
        }
    if len(df) and df["committed_placement"].nunique() >= 3:
        loo_rows = []
        for placement in sorted(df["committed_placement"].dropna().unique()):
            subset = df[df["committed_placement"] != placement]
            if len(subset) < 3:
                continue
            loo_rows.append(
                {
                    "left_out_placement": placement,
                    "minority_ratio_r2": _ols_r2(subset, ["minority_ratio"], "false_consensus_rate")["r2"],
                    "degree_influence_mass_r2": _ols_r2(subset, ["degree_influence_mass"], "false_consensus_rate")["r2"],
                }
            )
        result["leave_one_placement_out"] = loo_rows
    return result


def run_influence_reframe(cross_summary: pd.DataFrame) -> dict[str, Any]:
    existing_path = Path("outputs/v4/influence_mass_results.csv")
    existing = pd.read_csv(existing_path)
    existing_no_defense = existing[existing["defense"] == NO_DEFENSE].copy()
    existing_no_defense["source"] = "existing_hub_placement"

    cross = cross_summary.copy()
    cross["defense"] = NO_DEFENSE
    cross["source"] = "high_power_cross_size"
    cross["false_consensus_rate"] = cross["false_consensus_rate"].astype(float)
    cross = cross[
        [
            "source",
            "evidence_strength",
            "committed_placement",
            "minority_ratio",
            "defense",
            "degree_influence_mass",
            "betweenness_influence_mass",
            "pagerank_influence_mass",
            "average_committed_degree",
            "false_consensus_rate",
            "true_consensus_rate",
            "no_consensus_rate",
            "n_trials",
        ]
    ]
    existing_cols = [
        "committed_placement",
        "minority_ratio",
        "defense",
        "degree_influence_mass",
        "betweenness_influence_mass",
        "pagerank_influence_mass",
        "average_committed_degree",
        "false_consensus_rate",
        "true_consensus_rate",
        "no_consensus_rate",
        "n_trials",
    ]
    existing_reduced = existing_no_defense[existing_cols].copy()
    existing_reduced["source"] = "existing_hub_placement"
    existing_reduced["evidence_strength"] = np.nan
    combined = pd.concat([existing_reduced[cross.columns], cross], ignore_index=True)

    summary = {
        "existing_hub_placement": _predictor_summary(existing_reduced),
        "combined_with_high_power_cross_size": _predictor_summary(combined),
        "existing_rows": int(len(existing_reduced)),
        "combined_rows": int(len(combined)),
        "interpretation": "Predictive comparison only; adding high-power cross-size rows can overweight the two headline placements and should be read as sensitivity, not a new full benchmark.",
    }
    (OUT_DIR / "influence_mass_reframe_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    table_rows = []
    for scope, stats in [
        ("Existing hub placement", summary["existing_hub_placement"]),
        ("With high-power cross-size", summary["combined_with_high_power_cross_size"]),
    ]:
        for predictor in ["minority_ratio", "degree_influence_mass", "betweenness_influence_mass", "pagerank_influence_mass"]:
            table_rows.append(
                {
                    "Scope": scope,
                    "Predictor": predictor.replace("_", " "),
                    "Spearman rho": stats[predictor]["spearman"],
                    "OLS R2": stats[predictor]["ols_r2"],
                }
            )
    table = pd.DataFrame(table_rows)
    _write_tex_table(OUT_DIR / "table_influence_mass_reframe.tex", table, "Influence-mass reframe robustness.", "tab:influence-mass-reframe")
    return summary


def run_magic_constants(config: dict[str, Any], trials: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = []
    for value in [1.4, 2.1, 2.8]:
        specs.append(("social_scale", value))
    for value in [2.0, 4.0, 6.0]:
        specs.append(("committed_log_odds", value))
    for value in [0.8, 0.9, 0.99]:
        specs.append(("committed_confidence", value))

    conditions: list[StressCondition] = []
    for parameter, value in specs:
        for ratio in [0.10, 0.20]:
            for placement in ["high_degree", "low_degree"]:
                social_weight = value if parameter == "social_scale" else 2.1
                committed_logodds = value if parameter == "committed_log_odds" else 4.0
                committed_confidence = value if parameter == "committed_confidence" else 0.99
                conditions.append(
                    StressCondition(
                        experiment="magic_constants_sensitivity",
                        minority_ratio=ratio,
                        evidence_strength=0.66,
                        graph_topology="hub",
                        committed_placement=placement,
                        defense=NO_DEFENSE,
                        defense_strength=0.0,
                        social_weight=social_weight,
                        committed_logodds=committed_logodds,
                        committed_confidence=committed_confidence,
                        adversary_mode="honest_current",
                        sensitivity_parameter=parameter,
                        sensitivity_value=float(value),
                    )
                )
    trial_df = _run_conditions(conditions, trials, config, "magic constants")
    summary = _aggregate(
        trial_df,
        ["sensitivity_parameter", "sensitivity_value", "minority_ratio", "committed_placement"],
        "magic_constants",
    )
    gaps = []
    for (parameter, value, ratio), subset in summary.groupby(["sensitivity_parameter", "sensitivity_value", "minority_ratio"]):
        high = subset[subset["committed_placement"] == "high_degree"]
        low = subset[subset["committed_placement"] == "low_degree"]
        if high.empty or low.empty:
            continue
        gaps.append(
            {
                "sensitivity_parameter": parameter,
                "sensitivity_value": float(value),
                "minority_ratio": float(ratio),
                "high_degree_false_consensus_rate": float(high["false_consensus_rate"].iloc[0]),
                "low_degree_false_consensus_rate": float(low["false_consensus_rate"].iloc[0]),
                "high_minus_low_gap": float(high["false_consensus_rate"].iloc[0] - low["false_consensus_rate"].iloc[0]),
            }
        )
    gap_df = pd.DataFrame(gaps)
    summary.to_csv(OUT_DIR / "magic_constants_sensitivity.csv", index=False)
    gap_df.to_csv(OUT_DIR / "magic_constants_sensitivity_gaps.csv", index=False)

    no_def = summary.copy()
    influence_stats = _predictor_summary(no_def)
    result = {
        "scope": "one-parameter-at-a-time sensitivity strip; no_defense only; hub graph; evidence_strength=0.66",
        "trials_per_condition": trials,
        "gap_summary": gap_df.to_dict(orient="records"),
        "influence_predictor_summary": influence_stats,
    }
    (OUT_DIR / "magic_constants_sensitivity_summary.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    return summary, result


def _write_tex_table(path: Path, df: pd.DataFrame, caption: str, label: str) -> None:
    lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{%s}" % ("l" * len(df.columns)), "\\toprule"]
    lines.append(" & ".join(str(col).replace("_", "\\_") for col in df.columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_fig(fig: plt.Figure, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(PLOT_DIR / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_fabricated_citation(aggregate: pd.DataFrame) -> None:
    mixed = aggregate[aggregate["adversary_mode"] == "mixed_fabrication"].copy()
    honest_fab = aggregate[aggregate["adversary_mode"].isin(["honest_current", "fabricated_self_support"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for defense, subset in mixed.groupby("defense"):
        curve = subset.groupby("fabrication_probability", as_index=False)["false_consensus_rate"].mean()
        axes[0].plot(curve["fabrication_probability"], curve["false_consensus_rate"], marker="o", label=defense)
    axes[0].set_title("False consensus vs fabrication")
    axes[0].set_xlabel("Fabrication probability")
    axes[0].set_ylabel("False consensus rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.25)

    trade = honest_fab.groupby(["adversary_mode", "defense"], as_index=False)[
        ["false_consensus_rate", "true_consensus_rate", "no_consensus_rate"]
    ].mean()
    x = np.arange(len(trade))
    axes[1].bar(x, trade["true_consensus_rate"], label="true", color="#4b8f67")
    axes[1].bar(x, trade["false_consensus_rate"], bottom=trade["true_consensus_rate"], label="false", color="#b5544c")
    axes[1].bar(
        x,
        trade["no_consensus_rate"],
        bottom=trade["true_consensus_rate"] + trade["false_consensus_rate"],
        label="none",
        color="#8c8c8c",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r.adversary_mode}\n{r.defense}" for r in trade.itertuples()], rotation=80, ha="right", fontsize=6)
    axes[1].set_title("Outcome accounting")
    axes[1].set_ylim(0, 1.02)

    degradation = []
    for defense in DEFENSES:
        honest = honest_fab[(honest_fab["defense"] == defense) & (honest_fab["adversary_mode"] == "honest_current")]
        fabricated = honest_fab[(honest_fab["defense"] == defense) & (honest_fab["adversary_mode"] == "fabricated_self_support")]
        if honest.empty or fabricated.empty:
            continue
        degradation.append(
            {
                "defense": defense,
                "degradation": float(fabricated["false_consensus_rate"].mean() - honest["false_consensus_rate"].mean()),
            }
        )
    deg = pd.DataFrame(degradation)
    axes[2].bar(deg["defense"], deg["degradation"], color="#3a7ca5")
    axes[2].set_title("Defense degradation")
    axes[2].set_ylabel("Delta false consensus")
    axes[2].tick_params(axis="x", rotation=80, labelsize=7)
    axes[2].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Fabricated-citation stress test")
    _save_fig(fig, "figure_fabricated_citation_decay")


def plot_cross_size_power(summary: pd.DataFrame, tests: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, subset in summary.groupby("sensitivity_parameter"):
        subset = subset.sort_values("evidence_strength")
        ax.errorbar(
            subset["evidence_strength"],
            subset["false_consensus_rate"],
            yerr=[
                subset["false_consensus_rate"] - subset["false_consensus_rate_ci_low"],
                subset["false_consensus_rate_ci_high"] - subset["false_consensus_rate"],
            ],
            marker="o",
            capsize=3,
            label=label.replace("_", " "),
        )
    ax.set_xlabel("Evidence strength")
    ax.set_ylabel("False consensus rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("High-power cross-size comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_fig(fig, "figure_cross_size_power")


def write_recommendation(
    fabricated_summary: dict[str, Any],
    cross_summary: dict[str, Any],
    influence_summary: dict[str, Any],
    magic_summary: dict[str, Any],
) -> None:
    support_degrades = any(row["degradation"] > 0.05 for row in fabricated_summary.get("degradation_examples", []))
    all_cross_supported = bool(cross_summary.get("all_evidence_strengths_support_greater", False))
    existing = influence_summary["existing_hub_placement"]
    combined = influence_summary["combined_with_high_power_cross_size"]
    degree_r2 = combined["degree_influence_mass"]["ols_r2"]
    minority_r2 = combined["minority_ratio"]["ols_r2"]
    title_reco = "Influence Mass, Not Minority Count" if degree_r2 > minority_r2 else "Network Position Can Matter More Than Minority Size"

    lines = [
        "# Paper Reframe Recommendation",
        "",
        "No broad benchmark was run. These are targeted robustness stress tests in the neutral synthetic A/B hidden-state setting.",
        "",
        "## 1. Should the current title remain?",
        "",
        f"Recommendation: `{title_reco}`.",
        "",
        f"After adding high-power cross-size rows, degree influence-mass R2 is `{degree_r2:.3f}` versus minority-ratio R2 `{minority_r2:.3f}`. The safer title foregrounds influence mass if the paper wants to emphasize the most statistically stable mechanism rather than a single cross-size contrast.",
        "",
        "## 2. Or should it become \"Influence Mass, Not Minority Count\"?",
        "",
        "Use that framing if the final abstract emphasizes predictive mechanism. Keep the original title only if the cross-size claim remains statistically supported across evidence strengths and the paper wants a more intuitive statement.",
        "",
        "## 3. Is the 5% high-degree > 20% low-degree claim statistically supported?",
        "",
        f"All evidence strengths support a positive paired difference: `{all_cross_supported}`.",
        "",
        "Paired tests:",
    ]
    for row in cross_summary.get("paired_tests", []):
        lines.append(
            f"- evidence={row['evidence_strength']:.2f}: diff={row['paired_difference_five_high_minus_twenty_low']:.3f}, "
            f"95% CI [{row['paired_difference_ci_low']:.3f}, {row['paired_difference_ci_high']:.3f}], "
            f"p={row['paired_sign_flip_p_two_sided']:.4f}, greater={row['five_high_greater']}"
        )
    lines.extend(
        [
            "",
            "## 4. Does support checking survive fabricated citations?",
            "",
            f"Support-checking degrades under fabricated citations: `{support_degrades}`.",
            "",
            "The stress test assumes fabricated cited-evidence fields are protocol-visible but unauthenticated. If fabricated citations pass the support check, support checking should be described as effective only under verifiable or authenticated citation protocols.",
            "",
            "## 5. If not, how should the defense contribution be reframed?",
            "",
            "Reframe defense claims around leakage-free structural checks under a verifiable-message protocol. State that message-support checking is not a general solution to fabricated citations without citation authentication or external verification.",
            "",
            "## 6. Which results should go in the abstract?",
            "",
            "- Main no-defense false consensus and combined-defense true/false consensus from the verified benchmark.",
            "- Influence-mass R2 versus minority-ratio R2.",
            "- The cross-size comparison only if the high-power paired test remains positive with confidence intervals excluding zero.",
            "- A concise caveat that support-checking assumes verifiable cited evidence.",
            "",
            "## 7. Which results should be demoted to appendix or limitations?",
            "",
            "- Fabricated-citation failure modes should go to limitations or an appendix stress-test section.",
            "- Magic-constant sensitivity should be appendix unless it materially changes the headline.",
            "- Cross-size evidence-strength-specific rows can be appendix, with only the aggregate paired conclusion in the main text.",
            "- Any unsupported local LLM generalization should remain out of the main claims.",
            "",
            "## Supporting diagnostics",
            "",
            f"- Existing degree influence-mass R2: `{existing['degree_influence_mass']['ols_r2']:.3f}`.",
            f"- Existing minority-ratio R2: `{existing['minority_ratio']['ols_r2']:.3f}`.",
            f"- Magic-constant sensitivity scope: {magic_summary['scope']}",
        ]
    )
    (OUT_DIR / "PAPER_REFRAME_RECOMMENDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(
    fabricated_trials: int = 100,
    cross_size_trials: int = 200,
    magic_trials: int = 100,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    config = _base_config()

    fabricated_trials_df, fabricated_aggregate, fabricated_summary = run_fabricated_citation(config, fabricated_trials)
    diagnose_citation_behavior(fabricated_trials_df)
    plot_fabricated_citation(fabricated_aggregate)

    cross_trials, cross_results, cross_summary = run_cross_size_power(config, cross_size_trials)
    plot_cross_size_power(cross_results, cross_summary)

    influence_summary = run_influence_reframe(cross_results)
    magic_results, magic_summary = run_magic_constants(config, magic_trials)

    write_recommendation(fabricated_summary, cross_summary, influence_summary, magic_summary)

    manifest = {
        "fabricated_citation_rows": int(len(fabricated_aggregate)),
        "fabricated_citation_trial_rows": int(len(fabricated_trials_df)),
        "cross_size_rows": int(len(cross_results)),
        "cross_size_trial_rows": int(len(cross_trials)),
        "magic_constants_rows": int(len(magic_results)),
        "outputs": [
            "citation_behavior_summary.json",
            "citation_behavior_by_agent_type.csv",
            "fabricated_citation_results.csv",
            "fabricated_citation_summary.json",
            "plots/figure_fabricated_citation_decay.png",
            "plots/figure_fabricated_citation_decay.pdf",
            "cross_size_power_results.csv",
            "cross_size_power_summary.json",
            "plots/figure_cross_size_power.png",
            "plots/figure_cross_size_power.pdf",
            "influence_mass_reframe_summary.json",
            "table_influence_mass_reframe.tex",
            "magic_constants_sensitivity.csv",
            "magic_constants_sensitivity_summary.json",
            "PAPER_REFRAME_RECOMMENDATION.md",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted robustness stress tests.")
    parser.add_argument("--fabricated-trials", type=int, default=100)
    parser.add_argument("--cross-size-trials", type=int, default=200)
    parser.add_argument("--magic-trials", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_all(
        fabricated_trials=args.fabricated_trials,
        cross_size_trials=args.cross_size_trials,
        magic_trials=args.magic_trials,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
