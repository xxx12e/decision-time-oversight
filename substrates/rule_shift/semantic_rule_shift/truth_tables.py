"""Three-valued truth tables for small formal semantic systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

TruthValue = str

VALUES: tuple[TruthValue, ...] = ("T", "F", "U")
UNARY_OPS: tuple[str, ...] = ("NOT",)
BINARY_OPS: tuple[str, ...] = ("AND", "OR")
TOKENS: tuple[str, ...] = VALUES + UNARY_OPS + BINARY_OPS
LABEL_TO_ID: dict[str, int] = {value: idx for idx, value in enumerate(VALUES)}
ID_TO_LABEL: dict[int, str] = {idx: value for value, idx in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class SemanticSystem:
    """A named collection of truth tables over T/F/U."""

    name: str
    not_table: Mapping[TruthValue, TruthValue]
    and_table: Mapping[tuple[TruthValue, TruthValue], TruthValue]
    or_table: Mapping[tuple[TruthValue, TruthValue], TruthValue]
    description: str

    def apply(self, op: str, args: Sequence[TruthValue]) -> TruthValue:
        if op == "NOT":
            if len(args) != 1:
                raise ValueError("NOT expects one argument")
            return self.not_table[args[0]]
        if op == "AND":
            if len(args) != 2:
                raise ValueError("AND expects two arguments")
            return self.and_table[(args[0], args[1])]
        if op == "OR":
            if len(args) != 2:
                raise ValueError("OR expects two arguments")
            return self.or_table[(args[0], args[1])]
        raise KeyError(f"Unknown operator: {op}")

    def evaluate_postfix(self, tokens: Sequence[str]) -> TruthValue:
        stack: list[TruthValue] = []
        for token in tokens:
            if token in VALUES:
                stack.append(token)
            elif token == "NOT":
                if not stack:
                    raise ValueError(f"Malformed formula: {tokens!r}")
                stack.append(self.apply(token, [stack.pop()]))
            elif token in BINARY_OPS:
                if len(stack) < 2:
                    raise ValueError(f"Malformed formula: {tokens!r}")
                right = stack.pop()
                left = stack.pop()
                stack.append(self.apply(token, [left, right]))
            else:
                raise KeyError(f"Unknown token: {token}")
        if len(stack) != 1:
            raise ValueError(f"Malformed formula left {len(stack)} values: {tokens!r}")
        return stack[0]


def _commutative_table(
    fn: Callable[[TruthValue, TruthValue], TruthValue],
) -> dict[tuple[TruthValue, TruthValue], TruthValue]:
    return {(a, b): fn(a, b) for a in VALUES for b in VALUES}


def _k3_and(a: TruthValue, b: TruthValue) -> TruthValue:
    if a == "F" or b == "F":
        return "F"
    if a == "T" and b == "T":
        return "T"
    return "U"


def _k3_or(a: TruthValue, b: TruthValue) -> TruthValue:
    if a == "T" or b == "T":
        return "T"
    if a == "F" and b == "F":
        return "F"
    return "U"


def _bochvar_and(a: TruthValue, b: TruthValue) -> TruthValue:
    if a == "U" or b == "U":
        return "U"
    return "T" if a == "T" and b == "T" else "F"


def _bochvar_or(a: TruthValue, b: TruthValue) -> TruthValue:
    if a == "U" or b == "U":
        return "U"
    return "T" if a == "T" or b == "T" else "F"


def _custom_boundary_1_and(a: TruthValue, b: TruthValue) -> TruthValue:
    if a == "U" and b == "U":
        return "F"
    return _k3_and(a, b)


def _custom_boundary_1_or(a: TruthValue, b: TruthValue) -> TruthValue:
    if a == "U" and b == "U":
        return "T"
    return _k3_or(a, b)


_NOT = {"T": "F", "F": "T", "U": "U"}
_BOUNDARY_2_NOT = {"T": "F", "F": "T", "U": "T"}
_BOUNDARY_1_AND = _commutative_table(_custom_boundary_1_and)
_BOUNDARY_1_OR = _commutative_table(_custom_boundary_1_or)

SEMANTICS: dict[str, SemanticSystem] = {
    "k3": SemanticSystem(
        name="k3",
        not_table=_NOT,
        and_table=_commutative_table(_k3_and),
        or_table=_commutative_table(_k3_or),
        description="Strong Kleene K3 semantics.",
    ),
    "lp": SemanticSystem(
        name="lp",
        not_table=_NOT,
        and_table=_commutative_table(_k3_and),
        or_table=_commutative_table(_k3_or),
        description=(
            "LP-style paraconsistent variant. With only NOT/AND/OR and three "
            "observable labels, these truth functions are intentionally the same "
            "as K3, exposing an equivalence class that probes cannot separate."
        ),
    ),
    "bochvar": SemanticSystem(
        name="bochvar",
        not_table=_NOT,
        and_table=_commutative_table(_bochvar_and),
        or_table=_commutative_table(_bochvar_or),
        description="Bochvar/internal infectious semantics: U contaminates compounds.",
    ),
    "custom_boundary_1": SemanticSystem(
        name="custom_boundary_1",
        not_table=_NOT,
        and_table=_BOUNDARY_1_AND,
        or_table=_BOUNDARY_1_OR,
        description="Custom boundary semantics differing from K3 only on U AND U and U OR U.",
    ),
    "custom_boundary_2": SemanticSystem(
        name="custom_boundary_2",
        not_table=_BOUNDARY_2_NOT,
        and_table=_BOUNDARY_1_AND,
        or_table=_BOUNDARY_1_OR,
        description=(
            "Small-shift custom variant of custom_boundary_1: exactly one "
            "truth-table entry differs, NOT U = T instead of U."
        ),
    ),
    "boundary": SemanticSystem(
        name="boundary",
        not_table=_NOT,
        and_table=_BOUNDARY_1_AND,
        or_table=_BOUNDARY_1_OR,
        description="Backward-compatible alias of custom_boundary_1.",
    ),
}


def get_semantic(name: str) -> SemanticSystem:
    try:
        return SEMANTICS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SEMANTICS))
        raise KeyError(f"Unknown semantic system {name!r}. Known systems: {known}") from exc


def truth_table_difference_counts(
    left: SemanticSystem,
    right: SemanticSystem,
) -> dict[str, int]:
    """Count ordered truth-table entries that differ by operator."""

    counts = {
        "NOT": sum(left.not_table[value] != right.not_table[value] for value in VALUES),
        "AND": sum(
            left.and_table[(a, b)] != right.and_table[(a, b)]
            for a in VALUES
            for b in VALUES
        ),
        "OR": sum(
            left.or_table[(a, b)] != right.or_table[(a, b)]
            for a in VALUES
            for b in VALUES
        ),
    }
    counts["total"] = sum(counts.values())
    return counts
