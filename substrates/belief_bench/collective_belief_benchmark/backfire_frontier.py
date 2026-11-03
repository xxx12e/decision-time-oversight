"""Backfire frontier for identity-blind topology-aware pruning.

This is a targeted scripted-agent stress strip. It does not call LLMs or
external APIs, and it does not modify the paper.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .aaai_main_rescue import _aggregate_trials
from .config import load_config
from .defenses import (
    COMBINED_V1,
    NO_DEFENSE,
    TOPOLOGY_AWARE_PRUNING,
    TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP,
    _confidence_weight,
    _diversity_filter,
    _memory_weight,
    profile_for,
)
from .graphs import build_graph
from .reviewer_stress import (
    BASE_SEED,
    StressCondition,
    _make_public_messages,
)
from .schema import AgentState, PublicMessage, UpdateStats, sign_to_state
from .simulation import (
    _apply_update_rule,
    _make_agents,
    _population_metrics,
    _private_true_signal_fraction,
    _select_committed_agents,
    _update_rule_social_scale,
    stable_seed,
)
from .v4 import _centrality_masses


OUT_DIR = Path("outputs/backfire_frontier")
PLOT_DIR = OUT_DIR

THRESHOLDS = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60]
THRESHOLD_PLACEMENTS = [
    "high_degree",
    "medium_below_threshold",
    "low_degree",
    "high_betweenness",
    "dispersed",
    "mixed_high_and_below_threshold",
]
THRESHOLD_DEFENSES = [
    "no_defense",
    "topology_pruning",
    "topology_pruning_plus_confidence_cap",
    "combined",
]
CROSS_TOPOLOGIES = ["hub", "random", "small_world", "clustered"]
CROSS_N = [16, 32, 64]
CROSS_EVIDENCE = [0.58, 0.66, 0.74]
CROSS_PLACEMENTS = ["high_degree", "medium_below_threshold", "mixed"]
CROSS_DEFENSES = ["no_defense", "topology_pruning", "topology_pruning_plus_confidence_cap"]


@dataclass(frozen=True)
class FrontierCondition:
    experiment: str
    n_agents: int
    rounds: int
    minority_ratio: float
    evidence_strength: float
    graph_topology: str
    committed_placement: str
    defense_label: str
    pruning_threshold: float
    adversary_mode: str = "fabricated_self_support"
    fabrication_probability: float = 1.0
    memory_length: int = 3
    update_rule: str = "bayesian_like_agent"
    defense_strength: float = 1.0
    social_weight: float = 2.1
    committed_logodds: float = 4.0
    committed_confidence: float = 0.99


@dataclass
class MechanismStats:
    topology_pruned_normal_messages: int = 0
    topology_pruned_committed_messages: int = 0
    topology_retained_normal_messages: int = 0
    topology_retained_committed_messages: int = 0
    topology_pruned_normal_degree_sum: float = 0.0
    topology_pruned_committed_degree_sum: float = 0.0
    topology_retained_normal_degree_sum: float = 0.0
    topology_retained_committed_degree_sum: float = 0.0


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
    return value


def _base_config(n_agents: int, rounds: int = 8) -> dict[str, Any]:
    config = load_config("configs/v2.yaml")
    config = copy.deepcopy(config)
    config["n_agents"] = int(n_agents)
    config["rounds"] = int(rounds)
    config["consensus_threshold"] = 0.8
    config["balanced_private_evidence"] = True
    return config


def _defense_name(defense_label: str) -> str:
    if defense_label == "no_defense":
        return NO_DEFENSE
    if defense_label == "topology_pruning":
        return TOPOLOGY_AWARE_PRUNING
    if defense_label == "topology_pruning_plus_confidence_cap":
        return TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP
    if defense_label == "combined":
        return COMBINED_V1
    raise ValueError(f"Unknown defense label: {defense_label}")


def _placement_name(label: str) -> str:
    return "mixed_high_and_below_threshold" if label == "mixed" else label


def _select_committed_with_threshold(
    graph: nx.Graph,
    ratio: float,
    placement: str,
    rng: np.random.Generator,
    threshold: float,
    source_groups: dict[int, int] | None,
) -> set[int]:
    placement = _placement_name(placement)
    n_agents = graph.number_of_nodes()
    committed_count = int(round(ratio * n_agents))
    committed_count = max(0, min(n_agents - 1, committed_count))
    if committed_count == 0:
        return set()
    nodes = list(graph.nodes)
    degree_centrality = {node: graph.degree[node] / max(1, n_agents - 1) for node in nodes}
    jitter = {node: rng.random() for node in nodes}
    if placement == "medium_below_threshold":
        below = [node for node in nodes if degree_centrality[node] < threshold]
        ordered = sorted(below, key=lambda node: (degree_centrality[node], jitter[node]), reverse=True)
        if len(ordered) < committed_count:
            remaining = [node for node in nodes if node not in ordered]
            ordered.extend(sorted(remaining, key=lambda node: (degree_centrality[node], jitter[node]), reverse=True))
        return {int(node) for node in ordered[:committed_count]}
    if placement == "mixed_high_and_below_threshold":
        high = sorted(
            [node for node in nodes if degree_centrality[node] >= threshold],
            key=lambda node: (degree_centrality[node], jitter[node]),
            reverse=True,
        )
        below = sorted(
            [node for node in nodes if degree_centrality[node] < threshold],
            key=lambda node: (degree_centrality[node], jitter[node]),
            reverse=True,
        )
        high_count = min(len(high), max(1, committed_count // 2))
        selected = high[:high_count] + below[: max(0, committed_count - high_count)]
        if len(selected) < committed_count:
            remaining = [node for node in nodes if node not in selected]
            selected.extend(sorted(remaining, key=lambda node: (degree_centrality[node], jitter[node]), reverse=True))
        return {int(node) for node in selected[:committed_count]}
    return _select_committed_agents(graph, ratio, placement, rng, source_groups=source_groups)


def _is_topology_defense(defense: str) -> bool:
    return defense in {TOPOLOGY_AWARE_PRUNING, TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP}


def _social_signal_with_tau(
    agent: AgentState,
    messages: list[PublicMessage],
    current_round: int,
    defense: str,
    strength: float,
    base_confidence_cap: float,
    decay_base: float,
    stats: UpdateStats,
    mechanism: MechanismStats,
    committed_agents: set[int],
    raw_degrees: dict[int, int],
    pruning_threshold: float,
    update_rule: str,
) -> tuple[float, Any]:
    profile = profile_for(defense, strength, base_confidence_cap)
    message_list = list(messages)
    stats.messages_seen += len(message_list)
    stats.cost_units += profile.overhead_cost
    stats.cost_opportunities += 1 + len(message_list)
    if not message_list:
        return 0.0, profile

    message_list = _diversity_filter(message_list, profile, stats)
    weighted_sum = 0.0
    weight_total = 0.0
    used = 0

    for message in message_list:
        if not message.supported:
            stats.unsupported_seen += 1
        support_multiplier = profile.supported_multiplier if message.supported else profile.unsupported_multiplier
        if support_multiplier <= 0:
            stats.messages_suppressed += 1
            if not message.supported:
                stats.unsupported_suppressed += 1
            stats.cost_units += 0.35
            continue

        sender = int(message.sender_id)
        sender_is_committed = sender in committed_agents
        sender_degree = float(raw_degrees[sender])
        if _is_topology_defense(defense) and message.sender_degree_centrality >= pruning_threshold:
            stats.messages_suppressed += 1
            stats.cost_units += 0.12
            if sender_is_committed:
                mechanism.topology_pruned_committed_messages += 1
                mechanism.topology_pruned_committed_degree_sum += sender_degree
            else:
                mechanism.topology_pruned_normal_messages += 1
                mechanism.topology_pruned_normal_degree_sum += sender_degree
            continue

        if _is_topology_defense(defense):
            if sender_is_committed:
                mechanism.topology_retained_committed_messages += 1
                mechanism.topology_retained_committed_degree_sum += sender_degree
            else:
                mechanism.topology_retained_normal_messages += 1
                mechanism.topology_retained_normal_degree_sum += sender_degree

        age = current_round - message.round_index
        weight = (
            _confidence_weight(message.confidence, profile)
            * support_multiplier
            * _memory_weight(age, profile, decay_base)
        )
        if update_rule == "bounded_confidence_agent":
            current_direction = 1 if agent.belief_logodds >= 0 else -1
            if message.claim_sign != current_direction:
                distance = min(1.0, abs(agent.belief_logodds) / 6.0)
                weight *= 0.25 + 0.75 * (1.0 - distance)
        elif update_rule == "recency_biased_agent":
            weight *= decay_base ** max(0, age)
        if weight <= 0:
            stats.messages_suppressed += 1
            stats.cost_units += 0.20
            continue
        weighted_sum += message.claim_sign * weight
        weight_total += abs(weight)
        used += 1

    stats.messages_used += used
    if weight_total == 0:
        return 0.0, profile
    signal = (weighted_sum / weight_total) * profile.social_scale
    if update_rule == "social_proof_agent" and message_list:
        vote = sum(message.claim_sign for message in message_list)
        if vote != 0:
            majority_sign = 1 if vote > 0 else -1
            majority_margin = abs(vote) / len(message_list)
            signal = 0.72 * signal + 0.28 * majority_sign * majority_margin
    return signal, profile


def run_frontier_trial(condition: FrontierCondition, trial_index: int) -> dict[str, Any]:
    config = _base_config(condition.n_agents, condition.rounds)
    n_agents = int(config["n_agents"])
    trial_id = trial_index
    defense = _defense_name(condition.defense_label)

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
            condition.pruning_threshold,
        )
    )
    committed_agents = _select_committed_with_threshold(
        graph,
        condition.minority_ratio,
        condition.committed_placement,
        placement_rng,
        condition.pruning_threshold,
        source_groups,
    )

    evidence_seed = stable_seed(
        BASE_SEED,
        condition.experiment,
        "evidence",
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

    update_cfg = config["update"]
    threshold = float(config["consensus_threshold"])
    max_logodds = float(update_cfg.get("max_logodds", 6.0))
    base_anchor = float(update_cfg.get("evidence_anchor", 0.12))
    base_confidence_cap = float(update_cfg.get("confidence_cap", 0.95))
    decay_base = float(update_cfg.get("memory_decay", 0.55))
    defense_strength = 0.0 if defense == NO_DEFENSE else condition.defense_strength
    raw_degrees = {int(node): int(graph.degree[node]) for node in graph.nodes}

    stats = UpdateStats()
    mechanism = MechanismStats()
    consensus_time: int | None = None
    round_accuracies = [float(_population_metrics(agents, true_sign, threshold)["average_belief_accuracy"])]
    initial_metrics = _population_metrics(agents, true_sign, threshold)
    if initial_metrics["consensus"] != "none":
        consensus_time = 0

    stress_condition = StressCondition(
        experiment=condition.experiment,
        minority_ratio=condition.minority_ratio,
        evidence_strength=condition.evidence_strength,
        graph_topology=condition.graph_topology,
        committed_placement=condition.committed_placement,
        defense=defense,
        defense_strength=defense_strength,
        adversary_mode=condition.adversary_mode,
        fabrication_probability=condition.fabrication_probability,
        memory_length=condition.memory_length,
        update_rule=condition.update_rule,
        social_weight=condition.social_weight,
        committed_logodds=condition.committed_logodds,
        committed_confidence=condition.committed_confidence,
        pair_id=trial_id,
        sensitivity_parameter="pruning_threshold",
        sensitivity_value=condition.pruning_threshold,
    )

    for round_index in range(1, int(config["rounds"]) + 1):
        outbound, _ = _make_public_messages(agents, graph, stress_condition, round_index, n_agents)
        for agent in agents:
            if agent.is_committed:
                continue
            messages = [outbound[int(neighbor)] for neighbor in graph.neighbors(agent.agent_id)]
            agent.memory.extend(messages)
            while agent.memory and agent.memory[0].round_index <= round_index - condition.memory_length:
                agent.memory.popleft()

        new_logodds: dict[int, float] = {}
        for agent in agents:
            if agent.is_committed:
                continue
            signal, profile = _social_signal_with_tau(
                agent=agent,
                messages=list(agent.memory),
                current_round=round_index,
                defense=defense,
                strength=defense_strength,
                base_confidence_cap=base_confidence_cap,
                decay_base=decay_base,
                stats=stats,
                mechanism=mechanism,
                committed_agents=committed_agents,
                raw_degrees=raw_degrees,
                pruning_threshold=condition.pruning_threshold,
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
    degree_values = [degree for _, degree in graph.degree()]
    committed_degrees = [graph.degree[node] for node in committed_agents]
    centrality = _centrality_masses(graph, committed_agents)

    pruned_total = mechanism.topology_pruned_normal_messages + mechanism.topology_pruned_committed_messages
    retained_total = mechanism.topology_retained_normal_messages + mechanism.topology_retained_committed_messages
    result = {
        **asdict(condition),
        "defense": defense,
        "trial": trial_index,
        "seed": stable_seed(BASE_SEED, "backfire_frontier", condition.experiment, trial_index, condition.defense_label),
        "true_state": sign_to_state(true_sign),
        "false_state": sign_to_state(false_sign),
        "committed_agents": len(committed_agents),
        "normal_agents": int(final_metrics["normal_agents"]),
        "graph_edges": graph.number_of_edges(),
        "graph_mean_degree": float(np.mean(degree_values)),
        "graph_max_degree": int(max(degree_values) if degree_values else 0),
        "committed_mean_degree": float(np.mean(committed_degrees)) if committed_degrees else 0.0,
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
        "average_individual_accuracy": float(final_metrics["average_belief_accuracy"]),
        "trajectory_average_belief_accuracy": float(np.mean(round_accuracies)),
        "intervention_cost": float(stats.normalized_cost()),
        "messages_seen": int(stats.messages_seen),
        "messages_used": int(stats.messages_used),
        "messages_suppressed": int(stats.messages_suppressed),
        "topology_pruned_normal_messages": mechanism.topology_pruned_normal_messages,
        "topology_pruned_committed_messages": mechanism.topology_pruned_committed_messages,
        "topology_retained_normal_messages": mechanism.topology_retained_normal_messages,
        "topology_retained_committed_messages": mechanism.topology_retained_committed_messages,
        "pruned_message_share_from_normal_agents": (
            mechanism.topology_pruned_normal_messages / pruned_total if pruned_total else 0.0
        ),
        "pruned_message_share_from_committed_agents": (
            mechanism.topology_pruned_committed_messages / pruned_total if pruned_total else 0.0
        ),
        "retained_committed_message_share": (
            mechanism.topology_retained_committed_messages / retained_total if retained_total else 0.0
        ),
        "average_degree_pruned_normal_senders": (
            mechanism.topology_pruned_normal_degree_sum / mechanism.topology_pruned_normal_messages
            if mechanism.topology_pruned_normal_messages
            else 0.0
        ),
        "average_degree_retained_committed_senders": (
            mechanism.topology_retained_committed_degree_sum / mechanism.topology_retained_committed_messages
            if mechanism.topology_retained_committed_messages
            else 0.0
        ),
    }
    return result


def _run_conditions(conditions: list[FrontierCondition], trials_per_condition: int, progress: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(conditions) * trials_per_condition
    completed = 0
    for condition in conditions:
        for trial in range(trials_per_condition):
            rows.append(run_frontier_trial(condition, trial))
            completed += 1
            if completed % 2500 == 0 or completed == total:
                print(f"{progress} {completed}/{total}", flush=True)
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
                "cost": float(subset["intervention_cost"].mean()),
                "average_individual_accuracy": float(subset["average_individual_accuracy"].mean()),
                "time_to_consensus": float(subset["time_to_consensus"].mean()),
                "pruned_message_share_from_normal_agents": float(subset["pruned_message_share_from_normal_agents"].mean()),
                "pruned_message_share_from_committed_agents": float(subset["pruned_message_share_from_committed_agents"].mean()),
                "retained_committed_message_share": float(subset["retained_committed_message_share"].mean()),
                "average_degree_pruned_normal_senders": float(subset["average_degree_pruned_normal_senders"].mean()),
                "average_degree_retained_committed_senders": float(subset["average_degree_retained_committed_senders"].mean()),
                "topology_pruned_normal_messages": float(subset["topology_pruned_normal_messages"].mean()),
                "topology_pruned_committed_messages": float(subset["topology_pruned_committed_messages"].mean()),
                "topology_retained_committed_messages": float(subset["topology_retained_committed_messages"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_conditions() -> tuple[list[FrontierCondition], list[FrontierCondition]]:
    topology_conditions = []
    tau_independent = []
    for placement in THRESHOLD_PLACEMENTS:
        for tau in THRESHOLDS:
            for defense_label in ["topology_pruning", "topology_pruning_plus_confidence_cap"]:
                topology_conditions.append(
                    FrontierCondition(
                        experiment="threshold_frontier",
                        n_agents=32,
                        rounds=8,
                        minority_ratio=0.10,
                        evidence_strength=0.66,
                        graph_topology="hub",
                        committed_placement=placement,
                        defense_label=defense_label,
                        pruning_threshold=tau,
                    )
                )
        for defense_label in ["no_defense", "combined"]:
            tau_independent.append(
                FrontierCondition(
                    experiment="threshold_frontier",
                    n_agents=32,
                    rounds=8,
                    minority_ratio=0.10,
                    evidence_strength=0.66,
                    graph_topology="hub",
                    committed_placement=placement,
                    defense_label=defense_label,
                    pruning_threshold=0.35,
                )
            )
    return topology_conditions, tau_independent


def _cross_conditions() -> list[FrontierCondition]:
    conditions = []
    for topology in CROSS_TOPOLOGIES:
        for n_agents in CROSS_N:
            for evidence in CROSS_EVIDENCE:
                for placement in CROSS_PLACEMENTS:
                    for defense_label in CROSS_DEFENSES:
                        conditions.append(
                            FrontierCondition(
                                experiment="cross_setting_backfire",
                                n_agents=n_agents,
                                rounds=8,
                                minority_ratio=0.10,
                                evidence_strength=evidence,
                                graph_topology=topology,
                                committed_placement=placement,
                                defense_label=defense_label,
                                pruning_threshold=0.35,
                            )
                        )
    return conditions


def run_all() -> tuple[pd.DataFrame, dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topology_conditions, tau_independent = _threshold_conditions()
    threshold_topology_trials = _run_conditions(topology_conditions, 100, "threshold topology")
    threshold_base_trials = _run_conditions(tau_independent, 100, "threshold baselines")
    # Expand tau-independent baselines to each tau at aggregate time by duplicating trials.
    expanded = []
    for tau in THRESHOLDS:
        temp = threshold_base_trials.copy()
        temp["pruning_threshold"] = tau
        expanded.append(temp)
    threshold_trials = pd.concat([threshold_topology_trials, *expanded], ignore_index=True)

    cross_trials = _run_conditions(_cross_conditions(), 50, "cross-setting")
    all_trials = pd.concat([threshold_trials, cross_trials], ignore_index=True)

    group_cols = [
        "experiment",
        "n_agents",
        "evidence_strength",
        "graph_topology",
        "committed_placement",
        "minority_ratio",
        "adversary_mode",
        "pruning_threshold",
        "defense_label",
    ]
    aggregate = _aggregate(all_trials, group_cols)
    aggregate.to_csv(OUT_DIR / "backfire_frontier_results.csv", index=False)

    threshold = aggregate[aggregate["experiment"] == "threshold_frontier"].copy()
    threshold_worst = (
        threshold.groupby(["pruning_threshold", "defense_label"], as_index=False)
        .agg(
            worst_case_false_consensus=("false_consensus_rate", "max"),
            mean_false_consensus=("false_consensus_rate", "mean"),
            mean_cost=("cost", "mean"),
        )
        .sort_values(["pruning_threshold", "defense_label"])
    )
    threshold_worst.to_csv(OUT_DIR / "threshold_frontier_worst_case.csv", index=False)

    cross = aggregate[aggregate["experiment"] == "cross_setting_backfire"].copy()
    mechanism = _mechanism_audit(aggregate)
    mechanism.to_csv(OUT_DIR / "backfire_mechanism.csv", index=False)
    _write_tables(threshold_worst, cross, mechanism)
    _write_plot(threshold_worst)
    summary = _write_memo(aggregate, threshold_worst, cross, mechanism)
    return aggregate, summary


def _mechanism_audit(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_cols = [
        "experiment",
        "n_agents",
        "evidence_strength",
        "graph_topology",
        "committed_placement",
        "minority_ratio",
        "adversary_mode",
        "pruning_threshold",
    ]
    no_def = aggregate[aggregate["defense_label"] == "no_defense"]
    topo = aggregate[aggregate["defense_label"] == "topology_pruning"]
    base_map = {tuple(row[col] for col in key_cols): row for _, row in no_def.iterrows()}
    for _, row in topo.iterrows():
        key = tuple(row[col] for col in key_cols)
        base = base_map.get(key)
        if base is None:
            continue
        increase = float(row["false_consensus_rate"] - base["false_consensus_rate"])
        if increase > 1e-12:
            item = {col: row[col] for col in key_cols}
            item.update(
                {
                    "topology_pruning_false_consensus_rate": float(row["false_consensus_rate"]),
                    "no_defense_false_consensus_rate": float(base["false_consensus_rate"]),
                    "false_consensus_increase": increase,
                    "pruned_message_share_from_normal_agents": float(row["pruned_message_share_from_normal_agents"]),
                    "pruned_message_share_from_committed_agents": float(row["pruned_message_share_from_committed_agents"]),
                    "retained_committed_message_share": float(row["retained_committed_message_share"]),
                    "average_degree_pruned_normal_senders": float(row["average_degree_pruned_normal_senders"]),
                    "average_degree_retained_committed_senders": float(row["average_degree_retained_committed_senders"]),
                    "topology_pruned_normal_messages": float(row["topology_pruned_normal_messages"]),
                    "topology_pruned_committed_messages": float(row["topology_pruned_committed_messages"]),
                    "topology_retained_committed_messages": float(row["topology_retained_committed_messages"]),
                }
            )
            rows.append(item)
    return pd.DataFrame(rows).sort_values("false_consensus_increase", ascending=False) if rows else pd.DataFrame()


def _write_tables(threshold_worst: pd.DataFrame, cross: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    # Threshold frontier table.
    pivot = threshold_worst.pivot(index="pruning_threshold", columns="defense_label", values="worst_case_false_consensus")
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "$\\tau$ & No defense & Top. prune & Top. prune + cap & Combined \\\\",
        "\\midrule",
    ]
    for tau, row in pivot.iterrows():
        lines.append(
            f"{tau:.2f} & {row.get('no_defense', float('nan')):.3f} & "
            f"{row.get('topology_pruning', float('nan')):.3f} & "
            f"{row.get('topology_pruning_plus_confidence_cap', float('nan')):.3f} & "
            f"{row.get('combined', float('nan')):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUT_DIR / "table_threshold_frontier.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Cross-setting backfire table.
    rows = []
    key_cols = ["graph_topology", "n_agents", "evidence_strength"]
    for key, subset in cross.groupby(key_cols, dropna=False):
        no_def = subset[subset["defense_label"] == "no_defense"]
        topo = subset[subset["defense_label"] == "topology_pruning"]
        merged = topo.merge(
            no_def[
                [
                    "committed_placement",
                    "false_consensus_rate",
                ]
            ].rename(columns={"false_consensus_rate": "no_defense_fc"}),
            on="committed_placement",
            how="left",
        )
        merged["increase"] = merged["false_consensus_rate"] - merged["no_defense_fc"]
        rows.append(
            {
                "graph_topology": key[0],
                "n_agents": int(key[1]),
                "evidence_strength": float(key[2]),
                "backfire_conditions": int((merged["increase"] > 0).sum()),
                "max_false_consensus_increase": float(merged["increase"].max()),
                "max_topology_pruning_false_consensus": float(merged["false_consensus_rate"].max()),
            }
        )
    cross_table = pd.DataFrame(rows).sort_values("max_false_consensus_increase", ascending=False)
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Setting & Backfire conds. & Max $\\Delta$ FC & Max top. FC \\\\",
        "\\midrule",
    ]
    for _, row in cross_table.head(12).iterrows():
        setting = f"{row['graph_topology']}, n={int(row['n_agents'])}, e={row['evidence_strength']:.2f}"
        lines.append(
            f"{setting} & {int(row['backfire_conditions'])} & "
            f"{row['max_false_consensus_increase']:.3f} & {row['max_topology_pruning_false_consensus']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUT_DIR / "table_cross_setting_backfire.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Condition & $\\Delta$ FC & Pruned normal & Pruned committed & Retained committed & Deg. pruned normal \\\\",
        "\\midrule",
    ]
    for _, row in mechanism.head(12).iterrows():
        cond = (
            f"{row['experiment']}, {row['graph_topology']}, n={int(row['n_agents'])}, "
            f"e={row['evidence_strength']:.2f}, {row['committed_placement']}, $\\tau$={row['pruning_threshold']:.2f}"
        )
        lines.append(
            f"{cond} & {row['false_consensus_increase']:.3f} & "
            f"{row['pruned_message_share_from_normal_agents']:.3f} & "
            f"{row['pruned_message_share_from_committed_agents']:.3f} & "
            f"{row['retained_committed_message_share']:.3f} & "
            f"{row['average_degree_pruned_normal_senders']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUT_DIR / "table_backfire_mechanism.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(threshold_worst: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    labels = {
        "no_defense": "No defense",
        "topology_pruning": "Topology pruning",
        "topology_pruning_plus_confidence_cap": "Topology pruning + cap",
        "combined": "Combined",
    }
    for defense, subset in threshold_worst.groupby("defense_label"):
        subset = subset.sort_values("pruning_threshold")
        ax.plot(
            subset["pruning_threshold"],
            subset["worst_case_false_consensus"],
            marker="o",
            linewidth=2,
            label=labels.get(defense, defense),
        )
    ax.set_xlabel("Pruning threshold $\\tau$", fontsize=12)
    ax.set_ylabel("Worst-case\nfalse consensus", fontsize=11)
    ax.set_ylim(-0.02, 1.18)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=11)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure_threshold_frontier.png", dpi=220)
    fig.savefig(PLOT_DIR / "figure_threshold_frontier.pdf")
    plt.close(fig)


def _write_memo(
    aggregate: pd.DataFrame,
    threshold_worst: pd.DataFrame,
    cross: pd.DataFrame,
    mechanism: pd.DataFrame,
) -> dict[str, Any]:
    topo = aggregate[aggregate["defense_label"] == "topology_pruning"]
    no_def = aggregate[aggregate["defense_label"] == "no_defense"]
    key_cols = [
        "experiment",
        "n_agents",
        "evidence_strength",
        "graph_topology",
        "committed_placement",
        "minority_ratio",
        "adversary_mode",
        "pruning_threshold",
    ]
    no_def_map = {tuple(row[col] for col in key_cols): row for _, row in no_def.iterrows()}
    backfire_count = 0
    for _, row in topo.iterrows():
        base = no_def_map.get(tuple(row[col] for col in key_cols))
        if base is not None and float(row["false_consensus_rate"]) > float(base["false_consensus_rate"]):
            backfire_count += 1

    safe_thresholds = []
    for tau, subset in threshold_worst[threshold_worst["defense_label"] == "topology_pruning"].groupby("pruning_threshold"):
        threshold_rows = aggregate[
            (aggregate["experiment"] == "threshold_frontier")
            & (aggregate["pruning_threshold"] == tau)
            & (aggregate["defense_label"].isin(["topology_pruning", "no_defense"]))
        ]
        no_def_tau = threshold_rows[threshold_rows["defense_label"] == "no_defense"]
        topo_tau = threshold_rows[threshold_rows["defense_label"] == "topology_pruning"]
        base = {row["committed_placement"]: row["false_consensus_rate"] for _, row in no_def_tau.iterrows()}
        safe = True
        for _, row in topo_tau.iterrows():
            if float(row["false_consensus_rate"]) > float(base.get(row["committed_placement"], -1.0)):
                safe = False
                break
        if safe:
            safe_thresholds.append(float(tau))

    mechanism_support = False
    if len(mechanism):
        mechanism_support = bool(
            (mechanism["pruned_message_share_from_normal_agents"].mean() > 0.75)
            and (mechanism["average_degree_pruned_normal_senders"].mean() > mechanism["average_degree_retained_committed_senders"].mean())
        )
    summary = {
        "total_aggregate_rows": int(len(aggregate)),
        "topology_pruning_backfire_condition_count": int(backfire_count),
        "topology_pruning_backfires_beyond_original_single_cell": bool(backfire_count > 1),
        "safe_thresholds_without_backfire_in_threshold_frontier": safe_thresholds,
        "any_fixed_threshold_safe_across_adaptive_placements": bool(len(safe_thresholds) > 0),
        "mechanism_rows": int(len(mechanism)),
        "mechanism_supports_silencing_honest_hubs": mechanism_support,
        "max_false_consensus_increase": float(mechanism["false_consensus_increase"].max()) if len(mechanism) else 0.0,
        "worst_threshold_frontier": threshold_worst.sort_values("worst_case_false_consensus", ascending=False).head(1).to_dict(orient="records"),
    }
    lines = [
        "# Backfire Frontier Memo",
        "",
        "No external APIs or LLM experiments were used. This is a targeted scripted-agent stress strip.",
        "",
        "## 1. Does topology-aware pruning backfire beyond the original single cell?",
        "",
        (
            f"Answer: {'yes' if summary['topology_pruning_backfires_beyond_original_single_cell'] else 'not clearly'}. "
            f"Topology-pruning false consensus exceeded no-defense false consensus in "
            f"{summary['topology_pruning_backfire_condition_count']} aggregate condition(s). "
            f"The maximum observed increase was {summary['max_false_consensus_increase']:.3f}."
        ),
        "",
        "## 2. Is there any fixed threshold that is safe across adaptive placements?",
        "",
        (
            f"Answer: {'yes' if summary['any_fixed_threshold_safe_across_adaptive_placements'] else 'no'}. "
            f"Thresholds with no threshold-frontier backfire relative to no defense: "
            f"{summary['safe_thresholds_without_backfire_in_threshold_frontier']}."
        ),
        "",
        "## 3. Does the mechanism audit support silencing honest hubs while retaining below-threshold attackers?",
        "",
        (
            f"Answer: {'yes' if mechanism_support else 'mixed or not enough evidence'}. "
            "Mechanism rows report the normal/committed composition of pruned and retained messages and sender degrees."
        ),
        "",
        "## 4. Should the paper be reframed around topology-defense backfire?",
        "",
        "Answer: no. Treat topology-aware pruning backfire as a limitation and targeted robustness result. The main paper should remain framed around counting-is-not-enough / position-aware risk summaries unless a separate defense paper is written.",
        "",
        "## 5. Which claims should remain from the counting-is-not-enough framing?",
        "",
        "- Raw committed-minority count is an insufficient risk unit in the controlled scripted benchmark.",
        "- Position-aware summaries improve held-out prediction and controlled binomial models over raw count.",
        "- Defense claims are threat-model dependent; topology-aware pruning is not a globally robust defense.",
        "- Support-checking and topology-guided safeguards require explicit protocol boundaries.",
        "",
        "## Files",
        "",
        "- `backfire_frontier_results.csv`",
        "- `table_threshold_frontier.tex`",
        "- `table_cross_setting_backfire.tex`",
        "- `table_backfire_mechanism.tex`",
        "- `figure_threshold_frontier.pdf/png`",
    ]
    (OUT_DIR / "BACKFIRE_FRONTIER_MEMO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "backfire_frontier_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def main() -> None:
    aggregate, summary = run_all()
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
