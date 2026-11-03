from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import tanh


STATE_TO_SIGN = {"A": 1, "B": -1}
SIGN_TO_STATE = {1: "A", -1: "B"}


def sign_to_state(sign: int) -> str:
    return SIGN_TO_STATE[1 if sign >= 0 else -1]


def state_to_sign(state: str) -> int:
    return STATE_TO_SIGN[state.upper()]


def probability_from_logodds(logodds: float, target_sign: int) -> float:
    x = max(-60.0, min(60.0, target_sign * logodds))
    return 1.0 / (1.0 + pow(2.718281828459045, -x))


@dataclass(frozen=True)
class PublicMessage:
    round_index: int
    sender_id: int
    source_group: int
    claim_sign: int
    confidence: float
    evidence_signal: int | None
    sender_degree_centrality: float = 0.0

    @property
    def claim_state(self) -> str:
        return sign_to_state(self.claim_sign)

    @property
    def supported(self) -> bool:
        return self.evidence_signal is not None and self.evidence_signal == self.claim_sign


@dataclass(frozen=True)
class Message:
    round_index: int
    sender_id: int
    source_group: int
    claim_sign: int
    confidence: float
    supported: bool
    evidence_signal: int | None
    committed_sender: bool

    @property
    def claim_state(self) -> str:
        return sign_to_state(self.claim_sign)

    def to_public(self, sender_degree_centrality: float = 0.0) -> PublicMessage:
        return PublicMessage(
            round_index=self.round_index,
            sender_id=self.sender_id,
            source_group=self.source_group,
            claim_sign=self.claim_sign,
            confidence=self.confidence,
            evidence_signal=self.evidence_signal,
            sender_degree_centrality=sender_degree_centrality,
        )


@dataclass
class AgentState:
    agent_id: int
    is_committed: bool
    private_signal: int | None
    private_llr: float
    source_group: int
    belief_logodds: float
    memory: deque[PublicMessage] = field(default_factory=deque)

    def claim_sign(self) -> int:
        if self.belief_logodds == 0 and self.private_signal is not None:
            return self.private_signal
        return 1 if self.belief_logodds >= 0 else -1

    def confidence(self) -> float:
        return max(0.51, min(0.99, 0.5 + 0.5 * tanh(abs(self.belief_logodds) / 2.5)))

    def make_message(self, round_index: int, committed_confidence: float) -> Message:
        claim = self.claim_sign()
        if self.is_committed:
            confidence = committed_confidence
            supported = False
            evidence_signal = None
        else:
            confidence = self.confidence()
            supported = self.private_signal == claim
            evidence_signal = self.private_signal
        return Message(
            round_index=round_index,
            sender_id=self.agent_id,
            source_group=self.source_group,
            claim_sign=claim,
            confidence=confidence,
            supported=supported,
            evidence_signal=evidence_signal,
            committed_sender=self.is_committed,
        )


@dataclass
class UpdateStats:
    messages_seen: int = 0
    messages_used: int = 0
    messages_suppressed: int = 0
    messages_diversity_filtered: int = 0
    unsupported_seen: int = 0
    unsupported_suppressed: int = 0
    cost_units: float = 0.0
    cost_opportunities: int = 0

    def normalized_cost(self) -> float:
        if self.cost_opportunities == 0:
            return 0.0
        return self.cost_units / self.cost_opportunities
