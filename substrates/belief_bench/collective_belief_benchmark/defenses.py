from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from math import ceil
from statistics import median
from typing import Iterable

from .schema import AgentState, PublicMessage, UpdateStats


NO_DEFENSE = "no_defense"
MAJORITY_VOTE = "majority_vote"
CONFIDENCE_WEIGHTED = "confidence_weighted_update"
CONFIDENCE_WEIGHTED_AGGREGATION = "confidence_weighted_aggregation"
PROMPT_CAUTION = "prompt_only_caution"
EVIDENCE_ANCHORING = "evidence_anchoring"
SOURCE_DIVERSITY = "source_diversity_constraint"
MEMORY_DECAY = "memory_decay"
UNSUPPORTED_QUARANTINE = "unsupported_claim_quarantine"
COMBINED = "combined_inoculation_policy"
EVIDENCE_ANCHORING_ONLY = "evidence_anchoring_only"
SOURCE_DIVERSITY_ONLY = "source_diversity_only"
MEMORY_DECAY_ONLY = "memory_decay_only"
UNSUPPORTED_QUARANTINE_ONLY = "unsupported_claim_quarantine_only"
COMBINED_V1 = "combined"
COMBINED_MINUS_EVIDENCE_ANCHORING = "combined_minus_evidence_anchoring"
COMBINED_MINUS_SOURCE_DIVERSITY = "combined_minus_source_diversity"
COMBINED_MINUS_MEMORY_DECAY = "combined_minus_memory_decay"
COMBINED_MINUS_CONFIDENCE_CAP = "combined_minus_confidence_cap"
COMBINED_MINUS_QUARANTINE = "combined_minus_quarantine"
RANDOM_SOURCE_DROPOUT = "random_source_dropout"
HUB_DOWNWEIGHTING = "hub_downweighting"
MESSAGE_SUPPORT_CHECKING = "message_support_checking"
CONFIDENCE_CAP_ONLY = "confidence_cap_only"
TRIMMED_MEAN_AGGREGATION = "trimmed_mean_or_median_aggregation"
SIMPLE_REPUTATION = "simple_reputation_truth_discovery"
TOPOLOGY_AWARE_PRUNING = "topology_aware_pruning"
TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP = "topology_aware_pruning_plus_confidence_cap"
TOPOLOGY_AWARE_PRUNING_SUPPORT_CHECKING = "topology_aware_pruning_plus_support_checking"


DEFENSE_LABELS = {
    NO_DEFENSE: "No defense",
    MAJORITY_VOTE: "Majority vote",
    CONFIDENCE_WEIGHTED: "Confidence-weighted",
    CONFIDENCE_WEIGHTED_AGGREGATION: "Confidence-weighted aggregation",
    PROMPT_CAUTION: "Prompt caution",
    EVIDENCE_ANCHORING: "Evidence anchoring",
    SOURCE_DIVERSITY: "Source diversity",
    MEMORY_DECAY: "Memory decay",
    UNSUPPORTED_QUARANTINE: "Unsupported quarantine",
    COMBINED: "Combined inoculation",
    EVIDENCE_ANCHORING_ONLY: "Evidence anchoring only",
    SOURCE_DIVERSITY_ONLY: "Source diversity only",
    MEMORY_DECAY_ONLY: "Memory decay only",
    UNSUPPORTED_QUARANTINE_ONLY: "Unsupported quarantine only",
    COMBINED_V1: "Combined",
    COMBINED_MINUS_EVIDENCE_ANCHORING: "Combined minus evidence anchoring",
    COMBINED_MINUS_SOURCE_DIVERSITY: "Combined minus source diversity",
    COMBINED_MINUS_MEMORY_DECAY: "Combined minus memory decay",
    COMBINED_MINUS_CONFIDENCE_CAP: "Combined minus confidence cap",
    COMBINED_MINUS_QUARANTINE: "Combined minus quarantine",
    RANDOM_SOURCE_DROPOUT: "Random source dropout",
    HUB_DOWNWEIGHTING: "Hub downweighting",
    MESSAGE_SUPPORT_CHECKING: "Message support checking",
    CONFIDENCE_CAP_ONLY: "Confidence cap only",
    TRIMMED_MEAN_AGGREGATION: "Trimmed/median aggregation",
    SIMPLE_REPUTATION: "Simple reputation truth discovery",
    TOPOLOGY_AWARE_PRUNING: "Topology-aware pruning",
    TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP: "Topology-aware pruning + confidence cap",
    TOPOLOGY_AWARE_PRUNING_SUPPORT_CHECKING: "Topology-aware pruning + support checking",
}


@dataclass(frozen=True)
class DefenseProfile:
    majority_vote: bool = False
    confidence_weighted: bool = False
    evidence_anchor_boost: float = 0.0
    social_scale: float = 1.0
    confidence_cap: float = 0.99
    unsupported_multiplier: float = 1.0
    supported_multiplier: float = 1.0
    diversity_strength: float = 0.0
    memory_decay_strength: float = 0.0
    overhead_cost: float = 0.0


def profile_for(defense: str, strength: float, base_confidence_cap: float) -> DefenseProfile:
    if defense == NO_DEFENSE or strength <= 0:
        return DefenseProfile(confidence_cap=0.99)

    s = max(0.0, min(1.0, strength))

    if defense == MAJORITY_VOTE:
        return DefenseProfile(
            majority_vote=True,
            social_scale=1.0 - 0.10 * s,
            evidence_anchor_boost=0.05 * s,
            overhead_cost=0.03 * s,
        )
    if defense in {CONFIDENCE_WEIGHTED, CONFIDENCE_WEIGHTED_AGGREGATION}:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.08 * s,
            confidence_cap=max(0.60, base_confidence_cap - 0.15 * s),
            evidence_anchor_boost=0.08 * s,
            overhead_cost=0.04 * s,
        )
    if defense == CONFIDENCE_CAP_ONLY:
        return DefenseProfile(
            confidence_weighted=True,
            confidence_cap=max(0.58, base_confidence_cap - 0.22 * s),
            overhead_cost=0.04 * s,
        )
    if defense == TRIMMED_MEAN_AGGREGATION:
        return DefenseProfile(
            confidence_weighted=True,
            confidence_cap=max(0.62, base_confidence_cap - 0.12 * s),
            social_scale=1.0 - 0.05 * s,
            overhead_cost=0.06 * s,
        )
    if defense == SIMPLE_REPUTATION:
        return DefenseProfile(
            confidence_weighted=True,
            confidence_cap=max(0.62, base_confidence_cap - 0.10 * s),
            unsupported_multiplier=1.0 - 0.35 * s,
            overhead_cost=0.10 * s,
        )
    if defense == TOPOLOGY_AWARE_PRUNING:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.08 * s,
            confidence_cap=max(0.62, base_confidence_cap - 0.08 * s),
            overhead_cost=0.08 * s,
        )
    if defense == TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.08 * s,
            confidence_cap=max(0.58, base_confidence_cap - 0.22 * s),
            overhead_cost=0.12 * s,
        )
    if defense == TOPOLOGY_AWARE_PRUNING_SUPPORT_CHECKING:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.08 * s,
            confidence_cap=max(0.62, base_confidence_cap - 0.08 * s),
            unsupported_multiplier=1.0 - s,
            overhead_cost=0.20 * s,
        )
    if defense == RANDOM_SOURCE_DROPOUT:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.05 * s,
            overhead_cost=0.06 * s,
        )
    if defense == HUB_DOWNWEIGHTING:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.05 * s,
            evidence_anchor_boost=0.08 * s,
            overhead_cost=0.09 * s,
        )
    if defense == MESSAGE_SUPPORT_CHECKING:
        return DefenseProfile(
            confidence_weighted=True,
            unsupported_multiplier=1.0 - s,
            overhead_cost=0.12 * s,
        )
    if defense == PROMPT_CAUTION:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.35 * s,
            confidence_cap=max(0.60, base_confidence_cap - 0.10 * s),
            evidence_anchor_boost=0.22 * s,
            overhead_cost=0.08 * s,
        )
    if defense == EVIDENCE_ANCHORING:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.18 * s,
            evidence_anchor_boost=0.50 * s,
            unsupported_multiplier=1.0 - 0.65 * s,
            supported_multiplier=1.0 + 0.20 * s,
            overhead_cost=0.12 * s,
        )
    if defense == EVIDENCE_ANCHORING_ONLY:
        return DefenseProfile(
            confidence_weighted=True,
            evidence_anchor_boost=0.55 * s,
            unsupported_multiplier=1.0 - 0.70 * s,
            supported_multiplier=1.0 + 0.20 * s,
            overhead_cost=0.12 * s,
        )
    if defense == SOURCE_DIVERSITY:
        return DefenseProfile(
            confidence_weighted=True,
            diversity_strength=s,
            social_scale=1.0 - 0.08 * s,
            evidence_anchor_boost=0.10 * s,
            overhead_cost=0.14 * s,
        )
    if defense == SOURCE_DIVERSITY_ONLY:
        return DefenseProfile(
            confidence_weighted=True,
            diversity_strength=s,
            overhead_cost=0.14 * s,
        )
    if defense == MEMORY_DECAY:
        return DefenseProfile(
            confidence_weighted=True,
            memory_decay_strength=s,
            social_scale=1.0 - 0.05 * s,
            evidence_anchor_boost=0.10 * s,
            overhead_cost=0.08 * s,
        )
    if defense == MEMORY_DECAY_ONLY:
        return DefenseProfile(
            confidence_weighted=True,
            memory_decay_strength=s,
            overhead_cost=0.08 * s,
        )
    if defense == UNSUPPORTED_QUARANTINE:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.10 * s,
            evidence_anchor_boost=0.18 * s,
            unsupported_multiplier=1.0 - s,
            overhead_cost=0.18 * s,
        )
    if defense == UNSUPPORTED_QUARANTINE_ONLY:
        return DefenseProfile(
            confidence_weighted=True,
            unsupported_multiplier=1.0 - s,
            overhead_cost=0.18 * s,
        )
    if defense == COMBINED:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.40 * s,
            confidence_cap=max(0.58, base_confidence_cap - 0.20 * s),
            evidence_anchor_boost=0.58 * s,
            unsupported_multiplier=1.0 - s,
            supported_multiplier=1.0 + 0.15 * s,
            diversity_strength=s,
            memory_decay_strength=s,
            overhead_cost=0.28 * s,
        )
    if defense == COMBINED_V1:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.40 * s,
            confidence_cap=max(0.58, base_confidence_cap - 0.20 * s),
            evidence_anchor_boost=0.58 * s,
            unsupported_multiplier=1.0 - s,
            supported_multiplier=1.0 + 0.15 * s,
            diversity_strength=s,
            memory_decay_strength=s,
            overhead_cost=0.28 * s,
        )
    if defense == COMBINED_MINUS_EVIDENCE_ANCHORING:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.40 * s,
            confidence_cap=max(0.58, base_confidence_cap - 0.20 * s),
            unsupported_multiplier=1.0 - s,
            supported_multiplier=1.0 + 0.15 * s,
            diversity_strength=s,
            memory_decay_strength=s,
            overhead_cost=0.24 * s,
        )
    if defense == COMBINED_MINUS_SOURCE_DIVERSITY:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.40 * s,
            confidence_cap=max(0.58, base_confidence_cap - 0.20 * s),
            evidence_anchor_boost=0.58 * s,
            unsupported_multiplier=1.0 - s,
            supported_multiplier=1.0 + 0.15 * s,
            memory_decay_strength=s,
            overhead_cost=0.24 * s,
        )
    if defense == COMBINED_MINUS_MEMORY_DECAY:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.40 * s,
            confidence_cap=max(0.58, base_confidence_cap - 0.20 * s),
            evidence_anchor_boost=0.58 * s,
            unsupported_multiplier=1.0 - s,
            supported_multiplier=1.0 + 0.15 * s,
            diversity_strength=s,
            overhead_cost=0.24 * s,
        )
    if defense == COMBINED_MINUS_CONFIDENCE_CAP:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.40 * s,
            confidence_cap=0.99,
            evidence_anchor_boost=0.58 * s,
            unsupported_multiplier=1.0 - s,
            supported_multiplier=1.0 + 0.15 * s,
            diversity_strength=s,
            memory_decay_strength=s,
            overhead_cost=0.24 * s,
        )
    if defense == COMBINED_MINUS_QUARANTINE:
        return DefenseProfile(
            confidence_weighted=True,
            social_scale=1.0 - 0.40 * s,
            confidence_cap=max(0.58, base_confidence_cap - 0.20 * s),
            evidence_anchor_boost=0.58 * s,
            supported_multiplier=1.0 + 0.15 * s,
            diversity_strength=s,
            memory_decay_strength=s,
            overhead_cost=0.24 * s,
        )

    raise ValueError(f"Unknown defense: {defense}")


def _dropout_keep(agent_id: int, message: PublicMessage, current_round: int, strength: float) -> bool:
    drop_rate = 0.30 * max(0.0, min(1.0, strength))
    if drop_rate <= 0:
        return True
    payload = f"{agent_id}|{message.sender_id}|{message.round_index}|{current_round}".encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value >= drop_rate


def _confidence_weight(confidence: float, profile: DefenseProfile) -> float:
    capped = min(confidence, profile.confidence_cap)
    if profile.confidence_weighted:
        return max(0.05, 2.0 * capped - 1.0)
    return 0.45 + 0.55 * max(0.0, 2.0 * capped - 1.0)


def _memory_weight(age: int, profile: DefenseProfile, decay_base: float) -> float:
    if profile.memory_decay_strength <= 0:
        return 1.0
    decayed = decay_base ** max(0, age)
    return (1.0 - profile.memory_decay_strength) + profile.memory_decay_strength * decayed


def _diversity_filter(
    messages: list[PublicMessage],
    profile: DefenseProfile,
    stats: UpdateStats,
) -> list[PublicMessage]:
    if profile.diversity_strength <= 0 or len(messages) <= 2:
        return messages

    groups = sorted({message.source_group for message in messages})
    if not groups:
        return messages

    fair_cap = max(1, ceil(len(messages) / len(groups)))
    permissive_cap = len(messages)
    cap = max(1, ceil((1.0 - profile.diversity_strength) * permissive_cap + profile.diversity_strength * fair_cap))

    selected: list[PublicMessage] = []
    by_group: dict[int, list[PublicMessage]] = {group: [] for group in groups}
    for message in messages:
        by_group[message.source_group].append(message)
    for group_messages in by_group.values():
        group_messages.sort(key=lambda message: (message.round_index, -message.sender_id), reverse=True)
        selected.extend(group_messages[:cap])

    filtered = len(messages) - len(selected)
    stats.messages_diversity_filtered += filtered
    stats.cost_units += 0.25 * filtered
    return selected


def social_signal(
    agent: AgentState,
    messages: Iterable[PublicMessage],
    current_round: int,
    defense: str,
    strength: float,
    base_confidence_cap: float,
    decay_base: float,
    stats: UpdateStats,
    update_rule: str = "bayesian_like_agent",
) -> tuple[float, DefenseProfile]:
    profile = profile_for(defense, strength, base_confidence_cap)
    message_list = list(messages)
    stats.messages_seen += len(message_list)
    stats.cost_units += profile.overhead_cost
    stats.cost_opportunities += 1 + len(message_list)

    if not message_list:
        return 0.0, profile

    message_list = _diversity_filter(message_list, profile, stats)

    if profile.majority_vote:
        vote = sum(message.claim_sign for message in message_list)
        if vote == 0:
            return 0.0, profile
        stats.messages_used += len(message_list)
        margin = abs(vote) / max(1, len(message_list))
        return (1 if vote > 0 else -1) * margin * profile.social_scale, profile

    if defense == TRIMMED_MEAN_AGGREGATION:
        values: list[float] = []
        for message in message_list:
            if not message.supported:
                stats.unsupported_seen += 1
            values.append(message.claim_sign * _confidence_weight(message.confidence, profile))
        if not values:
            return 0.0, profile
        values.sort()
        trim = int(0.20 * len(values))
        retained = values[trim : len(values) - trim] if len(values) - 2 * trim >= 1 else values
        stats.messages_used += len(retained)
        stats.messages_suppressed += len(values) - len(retained)
        stats.cost_units += 0.08 * (len(values) - len(retained))
        return float(median(retained)) * profile.social_scale, profile

    sender_reliability: dict[int, float] = {}
    if defense == SIMPLE_REPUTATION:
        by_sender: dict[int, list[bool]] = {}
        for message in message_list:
            by_sender.setdefault(message.sender_id, []).append(message.supported)
        sender_reliability = {
            sender: (sum(1 for supported in values if supported) + 1.0) / (len(values) + 2.0)
            for sender, values in by_sender.items()
        }

    weighted_sum = 0.0
    weight_total = 0.0
    used = 0

    for message in message_list:
        if defense == RANDOM_SOURCE_DROPOUT and not _dropout_keep(agent.agent_id, message, current_round, strength):
            stats.messages_suppressed += 1
            stats.cost_units += 0.08
            continue
        if not message.supported:
            stats.unsupported_seen += 1

        support_multiplier = profile.supported_multiplier if message.supported else profile.unsupported_multiplier
        if support_multiplier <= 0:
            stats.messages_suppressed += 1
            if not message.supported:
                stats.unsupported_suppressed += 1
            stats.cost_units += 0.35
            continue

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
        if defense == HUB_DOWNWEIGHTING:
            hub_penalty = 1.0 - 0.70 * max(0.0, min(1.0, message.sender_degree_centrality)) * max(0.0, min(1.0, strength))
            weight *= max(0.10, hub_penalty)
        if defense == SIMPLE_REPUTATION:
            reliability = sender_reliability.get(message.sender_id, 0.5)
            weight *= 0.20 + 0.80 * reliability
        if defense in {
            TOPOLOGY_AWARE_PRUNING,
            TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP,
            TOPOLOGY_AWARE_PRUNING_SUPPORT_CHECKING,
        } and message.sender_degree_centrality >= 0.35:
            stats.messages_suppressed += 1
            stats.cost_units += 0.12
            continue
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


def prompt_instruction(defense: str) -> str:
    if defense == NO_DEFENSE:
        return "No additional defense policy is active."
    if defense == MAJORITY_VOTE:
        return "Compare neighbor claims as votes, but keep your private evidence visible."
    if defense == CONFIDENCE_WEIGHTED:
        return "Weight neighbor claims by confidence, capped to avoid over-trusting extreme certainty."
    if defense == CONFIDENCE_WEIGHTED_AGGREGATION:
        return "Aggregate neighbor claims with confidence weighting."
    if defense == PROMPT_CAUTION:
        return "Be cautious about social agreement and avoid changing belief without evidence support."
    if defense == EVIDENCE_ANCHORING:
        return "Anchor on your private evidence and discount claims that lack disclosed supporting evidence."
    if defense == EVIDENCE_ANCHORING_ONLY:
        return "Anchor on your private evidence and message-cited evidence."
    if defense == SOURCE_DIVERSITY:
        return "Prefer diverse source groups over repeated claims from one neighborhood."
    if defense == SOURCE_DIVERSITY_ONLY:
        return "Prefer diverse source groups over repeated claims from one neighborhood."
    if defense == MEMORY_DECAY:
        return "Give recent messages more weight than older messages."
    if defense == MEMORY_DECAY_ONLY:
        return "Give recent messages more weight than older messages."
    if defense == UNSUPPORTED_QUARANTINE:
        return "Ignore claims that do not disclose supporting private evidence."
    if defense == UNSUPPORTED_QUARANTINE_ONLY:
        return "Ignore claims that do not disclose supporting private evidence."
    if defense == RANDOM_SOURCE_DROPOUT:
        return "Randomly sample a subset of sources before updating."
    if defense == HUB_DOWNWEIGHTING:
        return "Downweight messages from highly connected senders using graph structure only."
    if defense == MESSAGE_SUPPORT_CHECKING:
        return "Use only claims whose cited evidence supports the claim."
    if defense == CONFIDENCE_CAP_ONLY:
        return "Cap message confidence before weighting neighbor claims."
    if defense == TRIMMED_MEAN_AGGREGATION:
        return "Aggregate signed neighbor confidence with a trimmed median rule."
    if defense == SIMPLE_REPUTATION:
        return "Weight sources by protocol-visible support consistency in recent memory."
    if defense == TOPOLOGY_AWARE_PRUNING:
        return "Prune high-centrality senders using graph structure only."
    if defense == TOPOLOGY_AWARE_PRUNING_CONFIDENCE_CAP:
        return "Prune high-centrality senders using graph structure only and cap message confidence."
    if defense == TOPOLOGY_AWARE_PRUNING_SUPPORT_CHECKING:
        return "Prune high-centrality senders using graph structure only and require cited evidence support."
    if defense in {
        COMBINED,
        COMBINED_V1,
        COMBINED_MINUS_EVIDENCE_ANCHORING,
        COMBINED_MINUS_SOURCE_DIVERSITY,
        COMBINED_MINUS_MEMORY_DECAY,
        COMBINED_MINUS_CONFIDENCE_CAP,
        COMBINED_MINUS_QUARANTINE,
    }:
        return "Anchor on private evidence, require support for claims, prefer diverse sources, and discount stale messages."
    return "Apply the configured synthetic defense policy."


def prepare_messages_for_llm(
    messages: Iterable[PublicMessage],
    current_round: int,
    defense: str,
    strength: float,
    base_confidence_cap: float,
    decay_base: float,
    stats: UpdateStats,
) -> tuple[list[PublicMessage], DefenseProfile, str]:
    profile = profile_for(defense, strength, base_confidence_cap)
    message_list = list(messages)
    stats.messages_seen += len(message_list)
    stats.cost_units += profile.overhead_cost
    stats.cost_opportunities += 1 + len(message_list)
    message_list = _diversity_filter(message_list, profile, stats)

    prepared: list[PublicMessage] = []
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
        age = current_round - message.round_index
        adjusted_confidence = min(message.confidence, profile.confidence_cap)
        adjusted_confidence *= support_multiplier
        adjusted_confidence *= _memory_weight(age, profile, decay_base)
        adjusted_confidence = max(0.51, min(0.99, adjusted_confidence))
        prepared.append(replace(message, confidence=adjusted_confidence))

    stats.messages_used += len(prepared)
    return prepared, profile, prompt_instruction(defense)
