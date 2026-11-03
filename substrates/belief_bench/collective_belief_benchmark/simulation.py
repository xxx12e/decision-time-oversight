from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from .defenses import NO_DEFENSE, prepare_messages_for_llm, social_signal
from .graphs import build_graph
from .local_llm import LocalLLMAdapter
from .schema import AgentState, PublicMessage, UpdateStats, probability_from_logodds, sign_to_state, state_to_sign


@dataclass(frozen=True)
class Condition:
    model_type: str
    minority_ratio: float
    evidence_strength: float
    memory_length: int
    graph_topology: str
    defense: str
    defense_strength: float
    update_rule: str = "bayesian_like_agent"
    committed_placement: str = "high_degree"
    experiment: str = "core_sweep"


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (base_seed, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _select_committed_agents(
    graph,
    ratio: float,
    selection: str,
    rng: np.random.Generator,
    source_groups: dict[int, int] | None = None,
) -> set[int]:
    n_agents = graph.number_of_nodes()
    committed_count = int(round(ratio * n_agents))
    committed_count = max(0, min(n_agents - 1, committed_count))
    if committed_count == 0:
        return set()

    nodes = list(graph.nodes)
    if selection in {"high_degree", "hub"}:
        jitter = {node: rng.random() for node in nodes}
        ordered = sorted(nodes, key=lambda node: (graph.degree[node], jitter[node]), reverse=True)
        return {int(node) for node in ordered[:committed_count]}
    if selection == "low_degree":
        jitter = {node: rng.random() for node in nodes}
        ordered = sorted(nodes, key=lambda node: (graph.degree[node], jitter[node]))
        return {int(node) for node in ordered[:committed_count]}
    if selection == "random":
        return {int(node) for node in rng.choice(nodes, size=committed_count, replace=False)}
    if selection == "medium_degree_below_threshold":
        threshold = 0.35
        jitter = {node: rng.random() for node in nodes}
        degree_centrality = {node: graph.degree[node] / max(1, n_agents - 1) for node in nodes}
        below = [node for node in nodes if degree_centrality[node] < threshold]
        ordered = sorted(below, key=lambda node: (degree_centrality[node], jitter[node]), reverse=True)
        if len(ordered) < committed_count:
            remaining = [node for node in nodes if node not in ordered]
            ordered.extend(sorted(remaining, key=lambda node: (degree_centrality[node], jitter[node]), reverse=True))
        return {int(node) for node in ordered[:committed_count]}
    if selection == "high_betweenness":
        jitter = {node: rng.random() for node in nodes}
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        ordered = sorted(nodes, key=lambda node: (betweenness[node], graph.degree[node], jitter[node]), reverse=True)
        return {int(node) for node in ordered[:committed_count]}
    if selection == "dispersed":
        jitter = {node: rng.random() for node in nodes}
        candidates = sorted(nodes, key=lambda node: (graph.degree[node], jitter[node]), reverse=True)
        selected: list[int] = []
        for node in candidates:
            if all(not graph.has_edge(node, chosen) for chosen in selected):
                selected.append(node)
            if len(selected) >= committed_count:
                break
        if len(selected) < committed_count:
            for node in candidates:
                if node not in selected:
                    selected.append(node)
                if len(selected) >= committed_count:
                    break
        return {int(node) for node in selected[:committed_count]}
    if selection == "mixed_high_and_below_threshold":
        threshold = 0.35
        jitter = {node: rng.random() for node in nodes}
        degree_centrality = {node: graph.degree[node] / max(1, n_agents - 1) for node in nodes}
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
    if selection == "clustered":
        if not source_groups:
            raise ValueError("clustered committed placement requires source group assignments.")
        groups: dict[int, list[int]] = {}
        for node, group in source_groups.items():
            groups.setdefault(group, []).append(node)
        eligible = [group for group, group_nodes in groups.items() if len(group_nodes) >= committed_count]
        if not eligible:
            eligible = list(groups)
        group_scores = {
            group: (sum(graph.degree[node] for node in groups[group]) / max(1, len(groups[group])), rng.random())
            for group in eligible
        }
        selected_group = max(eligible, key=lambda group: group_scores[group])
        cluster_nodes = groups[selected_group]
        ordered = sorted(cluster_nodes, key=lambda node: (graph.degree[node], rng.random()), reverse=True)
        selected = ordered[:committed_count]
        if len(selected) < committed_count:
            remaining = [node for node in nodes if node not in selected]
            remaining = sorted(remaining, key=lambda node: (graph.degree[node], rng.random()), reverse=True)
            selected.extend(remaining[: committed_count - len(selected)])
        return {int(node) for node in selected}
    raise ValueError(f"Unknown committed selection strategy: {selection}")


def _make_agents(
    n_agents: int,
    committed_agents: set[int],
    source_groups: dict[int, int],
    true_sign: int,
    evidence_strength: float,
    committed_logodds: float,
    rng: np.random.Generator,
    balanced_private_evidence: bool = False,
) -> list[AgentState]:
    false_sign = -true_sign
    llr_magnitude = log(evidence_strength / (1.0 - evidence_strength))
    agents: list[AgentState] = []
    normal_ids = [agent_id for agent_id in range(n_agents) if agent_id not in committed_agents]
    balanced_signals: dict[int, int] = {}
    if balanced_private_evidence and normal_ids:
        true_signal_count = int(round(evidence_strength * len(normal_ids)))
        true_signal_count = max(0, min(len(normal_ids), true_signal_count))
        true_ids = set(rng.choice(normal_ids, size=true_signal_count, replace=False)) if true_signal_count else set()
        balanced_signals = {
            agent_id: true_sign if agent_id in true_ids else false_sign
            for agent_id in normal_ids
        }
    for agent_id in range(n_agents):
        if agent_id in committed_agents:
            agents.append(
                AgentState(
                    agent_id=agent_id,
                    is_committed=True,
                    private_signal=None,
                    private_llr=0.0,
                    source_group=source_groups[agent_id],
                    belief_logodds=false_sign * committed_logodds,
                )
            )
            continue
        if balanced_private_evidence:
            signal = balanced_signals[agent_id]
        else:
            signal = true_sign if rng.random() < evidence_strength else false_sign
        private_llr = signal * llr_magnitude
        agents.append(
            AgentState(
                agent_id=agent_id,
                is_committed=False,
                private_signal=signal,
                private_llr=private_llr,
                source_group=source_groups[agent_id],
                belief_logodds=private_llr,
            )
        )
    return agents


def _private_true_signal_fraction(agents: list[AgentState], true_sign: int) -> float:
    normal_signals = [agent.private_signal for agent in agents if not agent.is_committed and agent.private_signal is not None]
    if not normal_signals:
        return 0.0
    return sum(1 for signal in normal_signals if signal == true_sign) / len(normal_signals)


def _population_metrics(
    agents: list[AgentState],
    true_sign: int,
    threshold: float,
) -> dict[str, float | int | str | bool]:
    normal_agents = [agent for agent in agents if not agent.is_committed]
    if not normal_agents:
        normal_agents = agents
    true_claims = sum(1 for agent in normal_agents if agent.claim_sign() == true_sign)
    false_claims = len(normal_agents) - true_claims
    true_fraction = true_claims / len(normal_agents)
    false_fraction = false_claims / len(normal_agents)
    avg_prob_true = float(np.mean([probability_from_logodds(agent.belief_logodds, true_sign) for agent in normal_agents]))
    if true_fraction >= threshold:
        consensus = "true"
    elif false_fraction >= threshold:
        consensus = "false"
    else:
        consensus = "none"
    return {
        "normal_agents": len(normal_agents),
        "true_fraction": true_fraction,
        "false_fraction": false_fraction,
        "average_belief_accuracy": avg_prob_true,
        "consensus": consensus,
        "true_consensus": consensus == "true",
        "false_consensus": consensus == "false",
    }


def _apply_local_llm_updates(
    agents: list[AgentState],
    llm: LocalLLMAdapter,
    condition: Condition,
    stats: UpdateStats,
    evidence_strength: float,
    base_anchor: float,
    base_confidence_cap: float,
    decay_base: float,
    current_round: int,
    max_logodds: float,
) -> None:
    for agent in agents:
        if agent.is_committed or agent.private_signal is None:
            continue
        defended_messages, profile, instruction = prepare_messages_for_llm(
            messages=agent.memory,
            current_round=current_round,
            defense=condition.defense,
            strength=condition.defense_strength,
            base_confidence_cap=base_confidence_cap,
            decay_base=decay_base,
            stats=stats,
        )
        current = sign_to_state(agent.claim_sign())
        decision = llm.decide(
            agent.private_signal,
            evidence_strength,
            current,
            defended_messages,
            defense_instruction=instruction,
        )
        sign = state_to_sign(decision.belief_state)
        confidence = max(0.51, min(0.99, decision.confidence))
        magnitude = log(confidence / (1.0 - confidence))
        llm_logodds = sign * magnitude
        anchor = min(0.90, base_anchor + profile.evidence_anchor_boost)
        updated = (1.0 - anchor) * llm_logodds + anchor * agent.private_llr
        agent.belief_logodds = max(-max_logodds, min(max_logodds, updated))


def _apply_update_rule(
    current: float,
    candidate: float,
    update_rule: str,
) -> float:
    if update_rule == "stubborn_but_not_committed_agent":
        return current + 0.38 * (candidate - current)
    return candidate


def _update_rule_social_scale(update_rule: str, signal: float) -> float:
    if update_rule == "social_proof_agent":
        return signal * (1.0 + 0.45 * min(1.0, abs(signal)))
    if update_rule == "stubborn_but_not_committed_agent":
        return signal * 0.55
    return signal


def run_trial(condition: Condition, trial_index: int, config: dict[str, Any], llm: LocalLLMAdapter | None = None) -> dict[str, Any]:
    seed = stable_seed(
        int(config["seed"]),
        "condition",
        trial_index,
        condition.model_type,
        condition.minority_ratio,
        condition.evidence_strength,
        condition.memory_length,
        condition.graph_topology,
        condition.defense,
        condition.defense_strength,
        condition.update_rule,
        condition.committed_placement,
        condition.experiment,
    )
    state_seed = stable_seed(
        int(config["seed"]),
        "state",
        trial_index,
        condition.model_type,
        condition.minority_ratio,
        condition.evidence_strength,
        condition.memory_length,
        condition.graph_topology,
        condition.update_rule,
        condition.experiment,
    )
    true_sign = int(np.random.default_rng(state_seed).choice([1, -1]))
    false_sign = -true_sign

    n_agents = int(config["n_agents"])
    graph_seed = stable_seed(
        int(config["seed"]),
        "graph",
        trial_index,
        n_agents,
        condition.graph_topology,
        condition.experiment,
    )
    graph, source_groups = build_graph(n_agents, condition.graph_topology, config["graph"], graph_seed)
    selection_rng = np.random.default_rng(
        stable_seed(
            int(config["seed"]),
            "placement",
            trial_index,
            condition.minority_ratio,
            condition.graph_topology,
            condition.committed_placement,
            condition.experiment,
        )
    )
    committed_agents = _select_committed_agents(
        graph,
        condition.minority_ratio,
        condition.committed_placement,
        selection_rng,
        source_groups=source_groups,
    )
    evidence_rng = np.random.default_rng(
        stable_seed(
            int(config["seed"]),
            "evidence",
            trial_index,
            condition.model_type,
            condition.minority_ratio,
            condition.evidence_strength,
            condition.memory_length,
            condition.graph_topology,
            condition.update_rule,
            condition.experiment,
        )
    )
    agents = _make_agents(
        n_agents,
        committed_agents,
        source_groups,
        true_sign,
        condition.evidence_strength,
        float(config["committed"].get("logodds", 4.0)),
        evidence_rng,
        balanced_private_evidence=bool(config.get("balanced_private_evidence", False)),
    )
    degree_values = [degree for _, degree in graph.degree()]
    committed_degrees = [graph.degree[node] for node in committed_agents]
    evidence_labels = config.get("evidence_strength_labels", {})
    evidence_label = evidence_labels.get(str(condition.evidence_strength), evidence_labels.get(condition.evidence_strength, str(condition.evidence_strength)))

    threshold = float(config["consensus_threshold"])
    update_cfg = config["update"]
    max_logodds = float(update_cfg.get("max_logodds", 6.0))
    social_weight = float(update_cfg.get("social_weight", 2.1))
    base_anchor = float(update_cfg.get("evidence_anchor", 0.12))
    base_confidence_cap = float(update_cfg.get("confidence_cap", 0.95))
    decay_base = float(update_cfg.get("memory_decay", 0.55))
    committed_confidence = float(config["committed"].get("confidence", 0.99))

    stats = UpdateStats()
    consensus_time: int | None = None
    round_accuracies: list[float] = []

    initial_metrics = _population_metrics(agents, true_sign, threshold)
    round_accuracies.append(float(initial_metrics["average_belief_accuracy"]))
    if initial_metrics["consensus"] != "none":
        consensus_time = 0

    for round_index in range(1, int(config["rounds"]) + 1):
        outbound = {
            agent.agent_id: agent.make_message(round_index, committed_confidence)
            for agent in agents
        }
        for agent in agents:
            messages: list[PublicMessage] = [
                outbound[int(neighbor)].to_public(
                    sender_degree_centrality=(
                        graph.degree[int(neighbor)] / max(1, n_agents - 1)
                    )
                )
                for neighbor in graph.neighbors(agent.agent_id)
            ]
            if agent.is_committed:
                continue
            agent.memory.extend(messages)
            while agent.memory and agent.memory[0].round_index <= round_index - condition.memory_length:
                agent.memory.popleft()

        if condition.model_type == "local_llm":
            if llm is None:
                raise ValueError("Local LLM model_type requested but no LocalLLMAdapter was initialized.")
            _apply_local_llm_updates(
                agents=agents,
                llm=llm,
                condition=condition,
                stats=stats,
                evidence_strength=condition.evidence_strength,
                base_anchor=base_anchor,
                base_confidence_cap=base_confidence_cap,
                decay_base=decay_base,
                current_round=round_index,
                max_logodds=max_logodds,
            )
        else:
            new_logodds: dict[int, float] = {}
            for agent in agents:
                if agent.is_committed:
                    continue
                signal, profile = social_signal(
                    agent=agent,
                    messages=agent.memory,
                    current_round=round_index,
                    defense=condition.defense,
                    strength=condition.defense_strength,
                    base_confidence_cap=base_confidence_cap,
                    decay_base=decay_base,
                    stats=stats,
                    update_rule=condition.update_rule,
                )
                signal = _update_rule_social_scale(condition.update_rule, signal)
                anchor = min(0.90, base_anchor + profile.evidence_anchor_boost)
                updated = (1.0 - anchor) * agent.belief_logodds + anchor * agent.private_llr
                updated += social_weight * signal
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
    no_consensus = final_metrics["consensus"] == "none"

    result = {
        **asdict(condition),
        "trial": trial_index,
        "seed": seed,
        "graph_seed": graph_seed,
        "state_seed": state_seed,
        "true_state": sign_to_state(true_sign),
        "false_state": sign_to_state(false_sign),
        "evidence_label": evidence_label,
        "committed_agents": len(committed_agents),
        "normal_agents": final_metrics["normal_agents"],
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
        "consensus_state": final_metrics["consensus"],
        "true_consensus": bool(final_metrics["true_consensus"]),
        "false_consensus": bool(final_metrics["false_consensus"]),
        "no_consensus": bool(no_consensus),
        "time_to_consensus": time_to_consensus,
        "final_true_fraction": float(final_metrics["true_fraction"]),
        "final_false_fraction": float(final_metrics["false_fraction"]),
        "final_average_belief_accuracy": float(final_metrics["average_belief_accuracy"]),
        "average_individual_accuracy": float(final_metrics["average_belief_accuracy"]),
        "trajectory_average_belief_accuracy": float(np.mean(round_accuracies)),
        "intervention_cost": float(stats.normalized_cost()),
        "messages_seen": stats.messages_seen,
        "messages_used": stats.messages_used,
        "messages_suppressed": stats.messages_suppressed,
        "messages_diversity_filtered": stats.messages_diversity_filtered,
        "unsupported_seen": stats.unsupported_seen,
        "unsupported_suppressed": stats.unsupported_suppressed,
    }
    return result


def iter_conditions(config: dict[str, Any]) -> list[Condition]:
    conditions: list[Condition] = []
    model_type = str(config["model_type"])
    update_rules = list(config.get("update_rules", ["bayesian_like_agent"]))
    placements = list(config.get("committed_placements", [str(config.get("committed", {}).get("selection", "high_degree"))]))
    experiment = str(config.get("experiment", "core_sweep"))
    for ratio, evidence, memory, topology, defense, update_rule, placement in product(
        config["minority_ratios"],
        config["evidence_strengths"],
        config["memory_lengths"],
        config["graph_topologies"],
        config["defenses"],
        update_rules,
        placements,
    ):
        strengths = [0.0] if defense == NO_DEFENSE else list(config["defense_strengths"])
        for strength in strengths:
            conditions.append(
                Condition(
                    model_type=model_type,
                    minority_ratio=float(ratio),
                    evidence_strength=float(evidence),
                    memory_length=int(memory),
                    graph_topology=str(topology),
                    defense=str(defense),
                    defense_strength=float(strength),
                    update_rule=str(update_rule),
                    committed_placement=str(placement),
                    experiment=experiment,
                )
            )
    return conditions


def run_conditions(
    config: dict[str, Any],
    conditions: list[Condition],
    llm: LocalLLMAdapter | None = None,
    progress_prefix: str = "completed",
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    total = len(conditions) * int(config["trials"])
    completed = 0
    for condition in conditions:
        for trial_index in range(int(config["trials"])):
            results.append(run_trial(condition, trial_index, config, llm=llm))
            completed += 1
            if completed % 1000 == 0 or completed == total:
                print(f"{progress_prefix} {completed}/{total} simulations")
    return pd.DataFrame(results)


def _bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    clean = values.astype(float)
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0:
        return float("nan"), float("nan")
    if len(clean) == 1 or iterations <= 0:
        value = float(np.mean(clean))
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(iterations, len(clean)))
    means = clean[indices].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def aggregate_results(
    trials: pd.DataFrame,
    bootstrap_iterations: int = 500,
    ci_level: float = 0.95,
    ci_seed: int = 20260601,
) -> pd.DataFrame:
    group_cols = [
        "model_type",
        "experiment",
        "update_rule",
        "committed_placement",
        "minority_ratio",
        "evidence_strength",
        "evidence_label",
        "memory_length",
        "graph_topology",
        "defense",
        "defense_strength",
    ]
    aggregations = {
        "false_consensus": "mean",
        "true_consensus": "mean",
        "no_consensus": "mean",
        "time_to_consensus": "mean",
        "final_true_fraction": "mean",
        "final_false_fraction": "mean",
        "final_average_belief_accuracy": "mean",
        "trajectory_average_belief_accuracy": "mean",
        "intervention_cost": "mean",
        "messages_seen": "mean",
        "messages_suppressed": "mean",
        "messages_diversity_filtered": "mean",
        "unsupported_suppressed": "mean",
        "committed_degree_ratio": "mean",
        "committed_mean_degree": "mean",
        "graph_mean_degree": "mean",
        "private_true_signal_fraction": "mean",
        "trial": "count",
    }
    aggregate = trials.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    aggregate = aggregate.rename(
        columns={
            "false_consensus": "false_consensus_rate",
            "true_consensus": "true_consensus_rate",
            "no_consensus": "no_consensus_rate",
            "time_to_consensus": "mean_time_to_consensus",
            "trial": "n_trials",
        }
    )
    aggregate["average_individual_accuracy"] = aggregate["final_average_belief_accuracy"]
    aggregate["time_to_consensus"] = aggregate["mean_time_to_consensus"]
    aggregate["outcome_rate_sum"] = (
        aggregate["false_consensus_rate"]
        + aggregate["true_consensus_rate"]
        + aggregate["no_consensus_rate"]
    )

    ci_specs = {
        "false_consensus": "false_consensus_rate",
        "true_consensus": "true_consensus_rate",
        "no_consensus": "no_consensus_rate",
        "average_individual_accuracy": "average_individual_accuracy",
        "time_to_consensus": "time_to_consensus",
        "intervention_cost": "intervention_cost",
    }
    ci_rows: list[dict[str, Any]] = []
    for key, subset in trials.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for source_col, output_col in ci_specs.items():
            group_seed = stable_seed(ci_seed, output_col, *key)
            low, high = _bootstrap_mean_ci(
                subset[source_col].to_numpy(dtype=float),
                iterations=bootstrap_iterations,
                ci_level=ci_level,
                seed=group_seed,
            )
            row[f"{output_col}_ci_low"] = low
            row[f"{output_col}_ci_high"] = high
        ci_rows.append(row)
    ci_df = pd.DataFrame(ci_rows)
    aggregate = aggregate.merge(ci_df, on=group_cols, how="left")
    return aggregate


def run_sweep(config: dict[str, Any], out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    llm = None
    if str(config["model_type"]) == "local_llm":
        local_cfg = config["local_llm"]
        llm = LocalLLMAdapter(
            model_path=str(local_cfg.get("model_path")),
            max_new_tokens=int(local_cfg.get("max_new_tokens", 96)),
            temperature=float(local_cfg.get("temperature", 0.0)),
        )

    trial_df = run_conditions(config, iter_conditions(config), llm=llm)
    aggregate_df = aggregate_results(
        trial_df,
        bootstrap_iterations=int(config.get("bootstrap_iterations", 500)),
        ci_level=float(config.get("ci_level", 0.95)),
        ci_seed=int(config.get("seed", 20260601)),
    )

    trial_df.to_csv(out_path / "trial_results.csv", index=False)
    aggregate_df.to_csv(out_path / "aggregate_results.csv", index=False)
    with (out_path / "trial_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (out_path / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_df.to_dict(orient="records"), handle, indent=2)
    return trial_df, aggregate_df
