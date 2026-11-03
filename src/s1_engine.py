"""S1 engine — check-placement sweep on the multi-agent belief-capture benchmark.

A faithful schedule-parameterized variant of content_defense.run_content_trial:
identical seeds/agents/graphs/update rules/adversary; the ONLY change is that the
content-anomaly check runs at SCHEDULED rounds only (attention budget B), and the
pruned-sender set persists across rounds once a check fires.

Originals are never modified; everything imports from the benchmark package.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from _paths import belief_bench, add_to_syspath          # noqa: E402

BENCH = add_to_syspath(belief_bench())

import numpy as np                                                    # noqa: E402
from collective_belief_benchmark.backfire_frontier import (           # noqa: E402
    FrontierCondition, StressCondition, _base_config, _make_public_messages, BASE_SEED,
)
from collective_belief_benchmark.content_defense import (             # noqa: E402
    anomaly_scores, _social_signal_content, THETA, CONTENT_DEFENSE,
)
from collective_belief_benchmark.backfire_frontier import MechanismStats  # noqa: E402
from collective_belief_benchmark.simulation import (                  # noqa: E402
    _population_metrics, _make_agents, _select_committed_agents,
    _update_rule_social_scale, _apply_update_rule, stable_seed, sign_to_state,
)
from collective_belief_benchmark.graphs import build_graph            # noqa: E402
from collective_belief_benchmark.schema import PublicMessage          # noqa: E402
from collective_belief_benchmark.defenses import UpdateStats          # noqa: E402

T_ROUNDS = 15
N_AGENTS = 32
EXPERIMENT = "p1_placement"

# _base_config resolves "configs/v2.yaml" relative to CWD -> fetch once from the
# benchmark root and deep-copy per trial.
import copy  # noqa: E402
import os    # noqa: E402

_cwd = os.getcwd()
os.chdir(BENCH)
try:
    _CONFIG_TEMPLATE = _base_config(N_AGENTS, T_ROUNDS)
finally:
    os.chdir(_cwd)


def make_condition(committed_count: int, evidence: float, topology: str) -> FrontierCondition:
    return FrontierCondition(
        experiment=EXPERIMENT, n_agents=N_AGENTS, rounds=T_ROUNDS,
        minority_ratio=committed_count / N_AGENTS, evidence_strength=evidence,
        graph_topology=topology, committed_placement="high_degree",
        defense_label=CONTENT_DEFENSE, pruning_threshold=0.35,
    )


class SuspicionPolicy:
    """Online: spend a check when the causally observable round-mean outbound
    confidence rises >= delta above the round-1 baseline. delta is calibrated on
    clean (0-committed) arms per (topology, evidence) cell: 95th percentile of
    max-round shifts, i.e. ~5% clean-arm trigger rate by construction."""

    def __init__(self, budget: int, delta: float):
        self.budget = budget
        self.delta = delta
        self.baseline: float | None = None

    def decide(self, round_index: int, mean_conf: float) -> bool:
        if round_index == 1:
            self.baseline = mean_conf
            return False
        if self.budget <= 0:
            return False
        if mean_conf - (self.baseline or mean_conf) >= self.delta:
            self.budget -= 1
            return True
        return False


class Instrument:
    """A detector a check may run. `kind` is one of:

      'theta'   -- the deployed content-anomaly score at an arbitrary threshold.
                   theta=THETA reproduces the paper's instrument exactly.
      'perfect' -- an upper-envelope detector that identifies every committed
                   sender from the visible prefix with no false positives. It is
                   not deployable; it exists to separate 'this ceiling is the
                   instrument's fault' from 'this ceiling is the deadline's'.
      'blind'   -- a detector that never flags, i.e. the no-check floor, used to
                   confirm an arm's plumbing rather than to make a claim.
    """

    def __init__(self, kind: str, theta: float = THETA):
        assert kind in ("theta", "perfect", "blind")
        self.kind, self.theta = kind, float(theta)

    @property
    def label(self) -> str:
        return f"theta={self.theta:.2f}" if self.kind == "theta" else self.kind

    def flag(self, public_log, round_index, committed_agents) -> set:
        if self.kind == "blind":
            return set()
        if self.kind == "perfect":
            # a check reveals the causal prefix; a sender with no message in it
            # cannot be flagged, which is what keeps this an upper envelope on
            # DETECTION and not a bypass of the causal-prefix rule
            return {int(s) for s in committed_agents if public_log[int(s)]}
        anomaly = anomaly_scores(public_log, round_index)
        return {s for s, score in anomaly.items() if score >= self.theta}


def s1_trial(cond: FrontierCondition, trial_id: int, schedule: frozenset[int],
             policy: SuspicionPolicy | None = None,
             instrument: "Instrument | None" = None) -> dict:
    """One trial with checks only at `schedule` rounds (or online `policy`).
    Deterministic in (cond, trial_id) — identical to a run_content_trial
    trajectory when schedule == all rounds, and to no-defense when empty.

    `instrument` swaps the DETECTOR a check runs, holding placement fixed; it
    exists so the instrument-sensitivity sweep can ask whether the regime map is
    a property of the fault dynamics or of one anomaly score at one threshold.
    None reproduces the frozen deployed instrument exactly (see
    src/instrument_sensitivity.py, which asserts that reproduction cell by cell
    before quoting any number from a swapped arm)."""
    assert cond.n_agents == N_AGENTS and cond.rounds == T_ROUNDS
    config = copy.deepcopy(_CONFIG_TEMPLATE)
    n_agents = int(config["n_agents"])

    state_seed = stable_seed(BASE_SEED, cond.experiment, "state", trial_id, cond.evidence_strength)
    true_sign = int(np.random.default_rng(state_seed).choice([1, -1]))
    false_sign = -true_sign

    graph_seed = stable_seed(BASE_SEED, cond.experiment, "graph", trial_id, n_agents, cond.graph_topology)
    graph, source_groups = build_graph(n_agents, cond.graph_topology, config["graph"], graph_seed)
    placement_rng = np.random.default_rng(stable_seed(
        BASE_SEED, cond.experiment, "placement", trial_id, cond.minority_ratio,
        cond.graph_topology, cond.committed_placement, cond.pruning_threshold))
    committed_agents = _select_committed_agents(
        graph, cond.minority_ratio, cond.committed_placement, placement_rng,
        source_groups=source_groups)

    evidence_seed = stable_seed(BASE_SEED, cond.experiment, "evidence", trial_id,
                                cond.evidence_strength, cond.graph_topology, cond.update_rule)
    agents = _make_agents(n_agents, committed_agents, source_groups, true_sign,
                          cond.evidence_strength, cond.committed_logodds,
                          np.random.default_rng(evidence_seed), balanced_private_evidence=True)

    update_cfg = config["update"]
    threshold = float(config["consensus_threshold"])
    max_logodds = float(update_cfg.get("max_logodds", 6.0))
    base_anchor = float(update_cfg.get("evidence_anchor", 0.12))
    base_confidence_cap = float(update_cfg.get("confidence_cap", 0.95))
    decay_base = float(update_cfg.get("memory_decay", 0.55))
    raw_degrees = {int(node): int(graph.degree[node]) for node in graph.nodes}

    stats, mechanism = UpdateStats(), MechanismStats()
    stress = StressCondition(
        experiment=cond.experiment, minority_ratio=cond.minority_ratio,
        evidence_strength=cond.evidence_strength, graph_topology=cond.graph_topology,
        committed_placement=cond.committed_placement, defense=CONTENT_DEFENSE,
        defense_strength=cond.defense_strength, adversary_mode=cond.adversary_mode,
        fabrication_probability=cond.fabrication_probability, memory_length=cond.memory_length,
        update_rule=cond.update_rule, social_weight=cond.social_weight,
        committed_logodds=cond.committed_logodds, committed_confidence=cond.committed_confidence,
        pair_id=trial_id, sensitivity_parameter="pruning_threshold",
        sensitivity_value=cond.pruning_threshold)

    public_log: dict[int, list[PublicMessage]] = {int(a.agent_id): [] for a in agents}
    checked_rounds: list[int] = []
    pruned_senders: set[int] = set()
    prune_events: list[tuple[int, int, bool]] = []
    false_frac_by_round: list[float] = []
    mean_conf_by_round: list[float] = []
    capture_round: int | None = None

    for round_index in range(1, int(config["rounds"]) + 1):
        outbound, _ = _make_public_messages(agents, graph, stress, round_index, n_agents)
        for sid, msg in outbound.items():
            public_log[int(sid)].append(msg)
        mean_conf_by_round.append(
            float(np.mean([m.confidence for m in outbound.values()])))

        check_now = (round_index in schedule) if policy is None else \
            policy.decide(round_index, mean_conf_by_round[-1])
        if check_now:
            checked_rounds.append(round_index)
            if instrument is None:
                anomaly = anomaly_scores(public_log, round_index)
                flagged = {s for s, score in anomaly.items() if score >= THETA}
            else:
                flagged = instrument.flag(public_log, round_index, committed_agents)
            newly = flagged - pruned_senders
            for s in sorted(newly):
                prune_events.append((round_index, int(s), int(s) in committed_agents))
            pruned_senders |= newly

        for agent in agents:
            if agent.is_committed:
                continue
            messages = [outbound[int(nb)] for nb in graph.neighbors(agent.agent_id)]
            agent.memory.extend(messages)
            while agent.memory and agent.memory[0].round_index <= round_index - cond.memory_length:
                agent.memory.popleft()

        new_logodds: dict[int, float] = {}
        for agent in agents:
            if agent.is_committed:
                continue
            signal, profile = _social_signal_content(
                agent=agent, messages=list(agent.memory), current_round=round_index,
                strength=cond.defense_strength, base_confidence_cap=base_confidence_cap,
                decay_base=decay_base, stats=stats, mechanism=mechanism,
                committed_agents=committed_agents, raw_degrees=raw_degrees,
                pruned_senders=pruned_senders, update_rule=cond.update_rule)
            signal = _update_rule_social_scale(cond.update_rule, signal)
            anchor = min(0.90, base_anchor + profile.evidence_anchor_boost)
            updated = (1.0 - anchor) * agent.belief_logodds + anchor * agent.private_llr
            updated += cond.social_weight * signal
            updated = _apply_update_rule(agent.belief_logodds, updated, cond.update_rule)
            new_logodds[agent.agent_id] = max(-max_logodds, min(max_logodds, updated))
        for agent in agents:
            if agent.agent_id in new_logodds:
                agent.belief_logodds = new_logodds[agent.agent_id]

        normals = [a for a in agents if not a.is_committed]
        ff = sum(1 for a in normals if a.claim_sign() == false_sign) / max(1, len(normals))
        false_frac_by_round.append(ff)
        if capture_round is None and ff >= threshold:
            capture_round = round_index

    final_ff = false_frac_by_round[-1]
    committed_pruned = sum(1 for _, _, c in prune_events if c)
    normal_pruned = sum(1 for _, _, c in prune_events if not c)
    return {
        **asdict(cond), "trial": trial_id, "schedule": sorted(schedule),
        "checked_rounds": checked_rounds,
        "true_state": sign_to_state(true_sign), "false_state": sign_to_state(false_sign),
        "n_committed": len(committed_agents),
        "capture_round": capture_round, "final_false_frac": final_ff,
        "false_consensus": final_ff >= threshold,
        "false_frac_by_round": false_frac_by_round,
        "mean_conf_by_round": mean_conf_by_round,
        "prune_events": prune_events,
        "committed_pruned": committed_pruned, "normal_pruned": normal_pruned,
        "timely_committed_prune": any(
            c and (capture_round is None or r <= capture_round) for r, _, c in prune_events),
    }


# ---------------- placement policies (schedules) ----------------

def sched_uniform(B: int, T: int = T_ROUNDS) -> frozenset[int]:
    return frozenset(max(2, round(i * T / (B + 1))) for i in range(1, B + 1))


def sched_random(B: int, seed: int, T: int = T_ROUNDS) -> frozenset[int]:
    import random
    return frozenset(random.Random(seed).sample(range(2, T + 1), B))


def sched_front(B: int, T: int = T_ROUNDS) -> frozenset[int]:
    return frozenset(range(2, 2 + B))


def sched_back(B: int, T: int = T_ROUNDS) -> frozenset[int]:
    return frozenset(range(T - B + 1, T + 1))


def sched_geometric(B: int, T: int = T_ROUNDS) -> frozenset[int]:
    # AI-control-style audit-early: rounds 2, 4, 8, ... capped at T
    return frozenset(min(T, 2 ** (i + 1)) for i in range(B))


def sched_oracle(B: int, capture_round: int | None, T: int = T_ROUNDS) -> frozenset[int]:
    c = capture_round if capture_round is not None else T
    lo = max(2, c - B + 1)
    return frozenset(range(lo, min(T, lo + B - 1) + 1))


POLICY_IDS = ("uniform", "random", "front", "back", "geometric", "suspicion", "oracle")
