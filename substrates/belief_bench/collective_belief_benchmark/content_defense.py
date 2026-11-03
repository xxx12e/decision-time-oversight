"""Block A: content/behaviour-aware pruning defence, as a single-variable
contrast against identity-blind degree pruning.

Iron rules honoured here:
* Scripted only; no LLM, no external API.
* identity-blind: the defence never reads is_committed / committed_sender.
* leakage-audited: it never reads hidden state / true sign / true label.
* single-variable contrast: the content arm reuses the *exact* topology-pruning
  defence chassis (same DefenseProfile, same social-signal arithmetic); the ONLY
  difference from degree pruning is which senders are pruned -- degree pruning
  uses sender_degree_centrality, content pruning uses an anomaly score computed
  from public message content + public behavioural history.
* fixed design: the three signal weights and the threshold theta are written
  here, reported, and held constant across every condition. They are NOT tuned
  to make reversal appear or disappear.

The no-defence and degree-pruning arms are produced by the *unmodified*
backfire_frontier.run_frontier_trial, so they reproduce the published numbers
exactly. This module only adds the content arm.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, replace
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backfire_frontier import (
    FrontierCondition,
    MechanismStats,
    _base_config,
    _cross_conditions,
    _select_committed_with_threshold,
    _threshold_conditions,
)
from .defenses import (
    TOPOLOGY_AWARE_PRUNING,
    _confidence_weight,
    _diversity_filter,
    _memory_weight,
    profile_for,
)
from .graphs import build_graph
from .reviewer_stress import BASE_SEED, StressCondition, _make_public_messages
from .schema import AgentState, PublicMessage, UpdateStats, sign_to_state
from .simulation import (
    _apply_update_rule,
    _make_agents,
    _population_metrics,
    _update_rule_social_scale,
    stable_seed,
)
from .v4 import _centrality_masses

OUT_DIR = Path("outputs/content_defense")
FRONTIER_CSV = Path("outputs/backfire_frontier/backfire_frontier_results.csv")

CONTENT_DEFENSE = "content_aware_pruning"

# ---------------------------------------------------------------------------
# FIXED content-aware pruning design (reported; NOT tuned to results)
# ---------------------------------------------------------------------------
THETA = 0.5            # prune sender if anomaly_score >= THETA (constant everywhere)
W_PROVENANCE = 0.4     # signal 1 weight
W_BEHAVIOUR = 0.4      # signal 2 weight
W_DIVERGENCE = 0.2     # signal 3 weight
REUSE_NORM = 3.0       # provenance: bloc size that saturates s1
CONF_STD_SCALE = 0.05  # behaviour: confidence-std that saturates the constancy term
CONF_ROUND = 2         # behaviour/provenance: confidence rounding for signatures

KEY = ["experiment", "n_agents", "evidence_strength", "graph_topology",
       "committed_placement", "minority_ratio", "adversary_mode", "pruning_threshold"]


# ---------------------------------------------------------------------------
# Anomaly score from PUBLIC information only (PublicMessage has no identity /
# hidden-state field; degree is deliberately NOT consulted).
# ---------------------------------------------------------------------------
def anomaly_scores(public_log: dict[int, list[PublicMessage]], current_round: int) -> dict[int, float]:
    # Signal 1 -- provenance: self-supporting senders that share an identical
    # public message signature this round (the coordinated-fabrication bloc).
    sig_count: dict[tuple, int] = {}
    sender_sig: dict[int, tuple] = {}
    for sender, hist in public_log.items():
        cur = [m for m in hist if m.round_index == current_round]
        if not cur:
            continue
        m = cur[-1]
        self_support = m.evidence_signal is not None and m.evidence_signal == m.claim_sign
        if self_support:
            sig = (m.claim_sign, m.evidence_signal, round(float(m.confidence), CONF_ROUND))
            sig_count[sig] = sig_count.get(sig, 0) + 1
            sender_sig[sender] = sig

    scores: dict[int, float] = {}
    for sender, hist in public_log.items():
        # s1: provenance / citation reuse
        s1 = 0.0
        if sender in sender_sig:
            reuse = sig_count[sender_sig[sender]]
            s1 = min(1.0, max(0.0, (reuse - 1) / REUSE_NORM))

        # s2: behavioural fingerprint (needs >= 2 public messages)
        confs = [float(m.confidence) for m in hist]
        claims = [int(m.claim_sign) for m in hist]
        s2 = 0.0
        if len(hist) >= 2:
            conf_constancy = max(0.0, 1.0 - statistics.pstdev(confs) / CONF_STD_SCALE)
            modal = max(set(claims), key=claims.count)
            claim_const = claims.count(modal) / len(claims)
            mean_conf = sum(confs) / len(confs)
            high_conf = min(1.0, max(0.0, (mean_conf - 0.9) / 0.09))
            s2 = conf_constancy * claim_const * high_conf

        # s3: private-evidence divergence (public unsupported-citation rate)
        cited = [m for m in hist if m.evidence_signal is not None]
        s3 = (sum(1 for m in cited if m.evidence_signal != m.claim_sign) / len(cited)) if cited else 0.0

        scores[sender] = W_PROVENANCE * s1 + W_BEHAVIOUR * s2 + W_DIVERGENCE * s3
    return scores


# ---------------------------------------------------------------------------
# Social signal -- byte-for-byte the topology-pruning arithmetic, except the
# prune predicate is `sender in pruned_senders` instead of degree >= tau.
# ---------------------------------------------------------------------------
def _social_signal_content(
    agent: AgentState,
    messages: list[PublicMessage],
    current_round: int,
    strength: float,
    base_confidence_cap: float,
    decay_base: float,
    stats: UpdateStats,
    mechanism: MechanismStats,
    committed_agents: set[int],
    raw_degrees: dict[int, int],
    pruned_senders: set[int],
    update_rule: str,
) -> tuple[float, Any]:
    profile = profile_for(TOPOLOGY_AWARE_PRUNING, strength, base_confidence_cap)
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
        if sender in pruned_senders:  # <-- ONLY difference vs degree pruning
            stats.messages_suppressed += 1
            stats.cost_units += 0.12
            if sender_is_committed:
                mechanism.topology_pruned_committed_messages += 1
                mechanism.topology_pruned_committed_degree_sum += sender_degree
            else:
                mechanism.topology_pruned_normal_messages += 1
                mechanism.topology_pruned_normal_degree_sum += sender_degree
            continue

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


def run_content_trial(condition: FrontierCondition, trial_index: int) -> dict[str, Any]:
    config = _base_config(condition.n_agents, condition.rounds)
    n_agents = int(config["n_agents"])
    trial_id = trial_index
    defense = CONTENT_DEFENSE

    state_seed = stable_seed(BASE_SEED, condition.experiment, "state", trial_id, condition.evidence_strength)
    true_sign = int(np.random.default_rng(state_seed).choice([1, -1]))
    false_sign = -true_sign

    graph_seed = stable_seed(BASE_SEED, condition.experiment, "graph", trial_id, n_agents, condition.graph_topology)
    graph, source_groups = build_graph(n_agents, condition.graph_topology, config["graph"], graph_seed)
    placement_rng = np.random.default_rng(
        stable_seed(
            BASE_SEED, condition.experiment, "placement", trial_id, condition.minority_ratio,
            condition.graph_topology, condition.committed_placement, condition.pruning_threshold,
        )
    )
    committed_agents = _select_committed_with_threshold(
        graph, condition.minority_ratio, condition.committed_placement,
        placement_rng, condition.pruning_threshold, source_groups,
    )

    evidence_seed = stable_seed(
        BASE_SEED, condition.experiment, "evidence", trial_id, condition.evidence_strength,
        condition.graph_topology, condition.update_rule,
    )
    agents = _make_agents(
        n_agents, committed_agents, source_groups, true_sign, condition.evidence_strength,
        condition.committed_logodds, np.random.default_rng(evidence_seed), balanced_private_evidence=True,
    )

    update_cfg = config["update"]
    threshold = float(config["consensus_threshold"])
    max_logodds = float(update_cfg.get("max_logodds", 6.0))
    base_anchor = float(update_cfg.get("evidence_anchor", 0.12))
    base_confidence_cap = float(update_cfg.get("confidence_cap", 0.95))
    decay_base = float(update_cfg.get("memory_decay", 0.55))
    defense_strength = condition.defense_strength
    raw_degrees = {int(node): int(graph.degree[node]) for node in graph.nodes}

    stats = UpdateStats()
    mechanism = MechanismStats()
    consensus_time: int | None = None
    round_accuracies = [float(_population_metrics(agents, true_sign, threshold)["average_belief_accuracy"])]
    if _population_metrics(agents, true_sign, threshold)["consensus"] != "none":
        consensus_time = 0

    stress_condition = StressCondition(
        experiment=condition.experiment, minority_ratio=condition.minority_ratio,
        evidence_strength=condition.evidence_strength, graph_topology=condition.graph_topology,
        committed_placement=condition.committed_placement, defense=defense,
        defense_strength=defense_strength, adversary_mode=condition.adversary_mode,
        fabrication_probability=condition.fabrication_probability, memory_length=condition.memory_length,
        update_rule=condition.update_rule, social_weight=condition.social_weight,
        committed_logodds=condition.committed_logodds, committed_confidence=condition.committed_confidence,
        pair_id=trial_id, sensitivity_parameter="pruning_threshold",
        sensitivity_value=condition.pruning_threshold,
    )

    public_log: dict[int, list[PublicMessage]] = {int(a.agent_id): [] for a in agents}

    for round_index in range(1, int(config["rounds"]) + 1):
        outbound, _ = _make_public_messages(agents, graph, stress_condition, round_index, n_agents)
        for sid, msg in outbound.items():
            public_log[int(sid)].append(msg)
        anomaly = anomaly_scores(public_log, round_index)
        pruned_senders = {sender for sender, score in anomaly.items() if score >= THETA}

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
            signal, profile = _social_signal_content(
                agent=agent, messages=list(agent.memory), current_round=round_index,
                strength=defense_strength, base_confidence_cap=base_confidence_cap, decay_base=decay_base,
                stats=stats, mechanism=mechanism, committed_agents=committed_agents,
                raw_degrees=raw_degrees, pruned_senders=pruned_senders, update_rule=condition.update_rule,
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
    pruned_total = mechanism.topology_pruned_normal_messages + mechanism.topology_pruned_committed_messages
    centrality = _centrality_masses(graph, committed_agents)

    return {
        **asdict(condition),
        "defense": defense,
        "trial": trial_index,
        "true_state": sign_to_state(true_sign),
        "false_state": sign_to_state(false_sign),
        "committed_agents": len(committed_agents),
        "normal_agents": int(final_metrics["normal_agents"]),
        "degree_influence_mass": float(centrality["degree_influence_mass"]),
        "consensus_state": final_metrics["consensus"],
        "true_consensus": bool(final_metrics["true_consensus"]),
        "false_consensus": bool(final_metrics["false_consensus"]),
        "no_consensus": bool(final_metrics["consensus"] == "none"),
        "time_to_consensus": int(time_to_consensus),
        "average_individual_accuracy": float(final_metrics["average_belief_accuracy"]),
        "intervention_cost": float(stats.normalized_cost()),
        "messages_seen": int(stats.messages_seen),
        "messages_suppressed": int(stats.messages_suppressed),
        "pruned_normal_messages": mechanism.topology_pruned_normal_messages,
        "pruned_committed_messages": mechanism.topology_pruned_committed_messages,
        "pruned_message_share_from_normal_agents": (
            mechanism.topology_pruned_normal_messages / pruned_total if pruned_total else 0.0
        ),
    }


# ---------------------------------------------------------------------------
def _content_conditions() -> list[FrontierCondition]:
    topo = [c for c in _threshold_conditions()[0] if c.defense_label == "topology_pruning"]
    topo += [c for c in _cross_conditions() if c.defense_label == "topology_pruning"]
    return [replace(c, defense_label="content_pruning") for c in topo]


def _seeds_for(condition: FrontierCondition) -> int:
    return 100 if condition.experiment == "threshold_frontier" else 50


def _aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df.groupby(KEY, as_index=False).agg(
        n_trials=("false_consensus", "size"),
        false_consensus_rate=("false_consensus", "mean"),
        true_consensus_rate=("true_consensus", "mean"),
        no_consensus_rate=("no_consensus", "mean"),
        cost=("intervention_cost", "mean"),
        pruned_share_normal=("pruned_message_share_from_normal_agents", "mean"),
    )
    return agg


def run_all() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = _content_conditions()
    rows: list[dict[str, Any]] = []
    total = sum(_seeds_for(c) for c in conditions)
    done = 0
    for c in conditions:
        for t in range(_seeds_for(c)):
            rows.append(run_content_trial(c, t))
            done += 1
            if done % 2000 == 0 or done == total:
                print(f"content trials {done}/{total}", flush=True)
    agg = _aggregate(rows)
    agg.to_csv(OUT_DIR / "content_frontier_results.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "content_frontier_trials.csv", index=False)
    return agg


if __name__ == "__main__":
    run_all()
