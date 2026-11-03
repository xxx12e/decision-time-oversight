"""V3 diagnosability scaffold for semantic rule shifts.

The V3 framing is symbolic first: before asking whether a detector wins, it
asks whether two semantics are observable under a particular probe language.
Online experiments then use the symbolic diagnosability layer to compare
selected, adaptive, passive, and random probes under equal budgets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import itertools
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

import torch


THREE_VALUES = ("T", "F", "U")
FOUR_VALUES = ("T", "F", "B", "N")
VARIABLES = ("p", "q")


@dataclass(frozen=True)
class V3Formula:
    tokens: tuple[str, ...]
    depth: int

    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass(frozen=True)
class V3Language:
    name: str
    operators: tuple[str, ...]
    constants: bool = False
    diagnostic: bool = False


@dataclass(frozen=True)
class V3SemanticSystem:
    name: str
    values: tuple[str, ...]
    unary: Mapping[str, Mapping[str, str]]
    binary: Mapping[str, Mapping[tuple[str, str], str]]
    description: str

    @property
    def value_set(self) -> str:
        return "four" if self.values == FOUR_VALUES else "three"

    def evaluate(self, formula: V3Formula, assignment: Mapping[str, str] | None = None) -> str:
        assignment = assignment or {}
        stack: list[str] = []
        for token in formula.tokens:
            if token in VARIABLES:
                stack.append(assignment[token])
            elif token in self.values:
                stack.append(token)
            elif token in self.unary:
                if not stack:
                    raise ValueError(f"Malformed formula: {formula.tokens!r}")
                stack.append(self.unary[token][stack.pop()])
            elif token in self.binary:
                if len(stack) < 2:
                    raise ValueError(f"Malformed formula: {formula.tokens!r}")
                right = stack.pop()
                left = stack.pop()
                stack.append(self.binary[token][(left, right)])
            else:
                raise KeyError(f"Unknown token {token!r} for {self.name}")
        if len(stack) != 1:
            raise ValueError(f"Malformed formula left {len(stack)} values: {formula.tokens!r}")
        return stack[0]

    def truth_vector(self, formula: V3Formula, variables: Sequence[str] = VARIABLES) -> tuple[str, ...]:
        if not any(token in variables for token in formula.tokens):
            return (self.evaluate(formula, {}),)
        outputs = []
        for values in itertools.product(self.values, repeat=len(variables)):
            assignment = dict(zip(variables, values))
            outputs.append(self.evaluate(formula, assignment))
        return tuple(outputs)


LANGUAGES: dict[str, V3Language] = {
    "L0": V3Language("L0", ("NOT", "AND", "OR"), constants=False),
    "L1": V3Language("L1", ("NOT", "AND", "OR"), constants=True),
    "L2": V3Language("L2", ("NOT", "AND", "OR", "IMPLIES"), constants=True),
    "L3": V3Language("L3", ("NOT", "AND", "OR", "IMPLIES", "EQUIV", "XOR"), constants=True),
    "L4": V3Language(
        "L4",
        ("NOT", "AND", "OR", "IMPLIES", "EQUIV", "XOR", "DIAG_EQ"),
        constants=True,
        diagnostic=True,
    ),
}


def build_v3_semantics() -> dict[str, V3SemanticSystem]:
    """Build documented three-valued and four-valued candidate semantics."""

    semantics: dict[str, V3SemanticSystem] = {}

    k3_three = _three_system(
        name="k3_3",
        not_table={"T": "F", "F": "T", "U": "U"},
        and_fn=_k3_and,
        or_fn=_k3_or,
        description="Strong Kleene K3 over T/F/U.",
    )
    semantics[k3_three.name] = k3_three
    semantics["lp_3"] = _three_system(
        name="lp_3",
        not_table={"T": "F", "F": "T", "U": "U"},
        and_fn=_k3_and,
        or_fn=_k3_or,
        description=(
            "LP-style three-valued semantics with the same observable NOT/AND/OR "
            "truth functions as K3 in this restricted interface."
        ),
    )
    semantics["bochvar_3"] = _three_system(
        name="bochvar_3",
        not_table={"T": "F", "F": "T", "U": "U"},
        and_fn=_bochvar_and,
        or_fn=_bochvar_or,
        description="Bochvar/internal three-valued semantics where U is infectious.",
    )
    semantics["custom_boundary_1_3"] = _three_system(
        name="custom_boundary_1_3",
        not_table={"T": "F", "F": "T", "U": "U"},
        and_fn=lambda a, b: "F" if (a, b) == ("U", "U") else _k3_and(a, b),
        or_fn=lambda a, b: "T" if (a, b) == ("U", "U") else _k3_or(a, b),
        description="Three-valued two-entry perturbation of K3 on U AND U and U OR U.",
    )
    semantics["custom_boundary_2_3"] = _three_system(
        name="custom_boundary_2_3",
        not_table={"T": "F", "F": "T", "U": "T"},
        and_fn=lambda a, b: "F" if (a, b) == ("U", "U") else _k3_and(a, b),
        or_fn=lambda a, b: "T" if (a, b) == ("U", "U") else _k3_or(a, b),
        description="One-entry perturbation of custom_boundary_1_3: NOT U = T.",
    )
    semantics["k3_imp_flip_3"] = _with_binary_override(
        k3_three,
        name="k3_imp_flip_3",
        op="IMPLIES",
        entry=("T", "U"),
        value="T",
        description=(
            "K3-identical under L0/L1, but one IMPLIES entry differs under L2: "
            "T IMPLIES U = T."
        ),
    )
    semantics["k3_xor_flip_3"] = _with_binary_override(
        k3_three,
        name="k3_xor_flip_3",
        op="XOR",
        entry=("U", "U"),
        value="T",
        description="Sparse L3-only perturbation: U XOR U = T.",
    )

    fde = _four_system(
        name="fde4",
        normalize=lambda value: value,
        description="Belnap/FDE bilattice-style semantics over T/F/B/N.",
    )
    semantics[fde.name] = fde
    semantics["lp4"] = _four_system(
        name="lp4",
        normalize=lambda value: "F" if value == "N" else value,
        description="LP-style total paraconsistent semantics: N collapses to F in connectives.",
    )
    semantics["k3_lift4"] = _four_system(
        name="k3_lift4",
        normalize=lambda value: "N" if value == "B" else value,
        description="K3-style four-valued lift: B is treated as unknown/neither.",
    )
    semantics["projection_k3_n4"] = _projected_k3_four_system(
        name="projection_k3_n4",
        unknown_value="N",
        description="Projection-based K3 lift that represents observable U internally as N.",
    )
    semantics["projection_k3_b4"] = _projected_k3_four_system(
        name="projection_k3_b4",
        unknown_value="B",
        description="Projection-based K3 lift that represents observable U internally as B.",
    )
    semantics["bochvar4"] = _bochvar_four_system()
    semantics["custom_boundary_1_4"] = _with_binary_override(
        fde,
        name="custom_boundary_1_4",
        op="AND",
        entry=("B", "N"),
        value="F",
        description="One-entry FDE perturbation: B AND N = F.",
    )
    semantics["custom_boundary_2_4"] = _with_binary_override(
        semantics["custom_boundary_1_4"],
        name="custom_boundary_2_4",
        op="OR",
        entry=("B", "N"),
        value="T",
        description="Two-entry FDE perturbation: B AND N = F, B OR N = T.",
    )
    semantics["rare_implies_flip4"] = _with_binary_override(
        fde,
        name="rare_implies_flip4",
        op="IMPLIES",
        entry=("N", "B"),
        value="B",
        description="Sparse L2 perturbation: N IMPLIES B = B.",
    )
    semantics["rare_xor_flip4"] = _with_binary_override(
        fde,
        name="rare_xor_flip4",
        op="XOR",
        entry=("B", "N"),
        value="T",
        description="Sparse L3 perturbation: B XOR N = T.",
    )
    semantics["projection_flip4"] = _with_binary_override(
        fde,
        name="projection_flip4",
        op="AND",
        entry=("T", "N"),
        value="B",
        description=(
            "Projection-equivalent perturbation under the bilattice-to-three "
            "observable map: T AND N changes from N to B, both observed as U."
        ),
    )
    return semantics


def enumerate_v3_formulas(
    values: Sequence[str],
    language: V3Language,
    max_depth: int = 2,
    max_formulas: int = 2500,
) -> list[V3Formula]:
    """Enumerate postfix formulas for a language, deduplicating exact tokens."""

    leaves = [V3Formula((var,), 0) for var in VARIABLES]
    if language.constants:
        leaves.extend(V3Formula((value,), 0) for value in values)
    by_depth: list[list[V3Formula]] = [leaves]
    seen = {formula.tokens for formula in leaves}
    all_formulas = list(leaves)

    unary_ops = [op for op in language.operators if op == "NOT"]
    binary_ops = [op for op in language.operators if op != "NOT"]
    commutative_ops = {"AND", "OR", "EQUIV", "XOR", "DIAG_EQ"}

    for depth in range(1, max_depth + 1):
        current: list[V3Formula] = []
        previous = [formula for bucket in by_depth for formula in bucket]
        for child in previous:
            for op in unary_ops:
                candidate = V3Formula(child.tokens + (op,), depth)
                if candidate.tokens not in seen:
                    seen.add(candidate.tokens)
                    current.append(candidate)
        for left in previous:
            for right in previous:
                for op in binary_ops:
                    if op in commutative_ops and right.tokens < left.tokens:
                        continue
                    candidate = V3Formula(left.tokens + right.tokens + (op,), depth)
                    if candidate.tokens not in seen:
                        seen.add(candidate.tokens)
                        current.append(candidate)
                    if len(all_formulas) + len(current) >= max_formulas:
                        break
                if len(all_formulas) + len(current) >= max_formulas:
                    break
            if len(all_formulas) + len(current) >= max_formulas:
                break
        by_depth.append(current)
        all_formulas.extend(current)
        if len(all_formulas) >= max_formulas:
            break
    return all_formulas[:max_formulas]


def compute_diagnosability(
    semantics: Mapping[str, V3SemanticSystem],
    languages: Mapping[str, V3Language],
    max_depth: int = 2,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, list[V3Formula]]]:
    rows: list[dict[str, object]] = []
    distinguishing: dict[str, object] = {}
    formula_pools: dict[str, list[V3Formula]] = {}

    for value_set, values in (("three", THREE_VALUES), ("four", FOUR_VALUES)):
        names = [name for name, sem in semantics.items() if sem.value_set == value_set]
        for language_name, language in languages.items():
            max_formulas = 1600 if language_name in {"L0", "L1"} else 2400
            formulas = enumerate_v3_formulas(values, language, max_depth=max_depth, max_formulas=max_formulas)
            formula_pools[f"{value_set}:{language_name}"] = formulas
            vectors = {
                name: [semantics[name].truth_vector(formula) for formula in formulas]
                for name in names
            }
            min_probe_set = greedy_set_cover_probes(names, formulas, vectors)
            for left, right in itertools.combinations(names, 2):
                differing_indices = [
                    idx for idx, _ in enumerate(formulas) if vectors[left][idx] != vectors[right][idx]
                ]
                pair_key = f"{value_set}:{language_name}:{left}->{right}"
                if differing_indices:
                    shortest = min(formulas[idx].depth for idx in differing_indices)
                    distance = mean(
                        _truth_vector_distance(vectors[left][idx], vectors[right][idx])
                        for idx in range(len(formulas))
                    )
                    distinguishing[pair_key] = [
                        {
                            "formula": formulas[idx].text(),
                            "depth": formulas[idx].depth,
                            left: vectors[left][idx],
                            right: vectors[right][idx],
                        }
                        for idx in differing_indices[:20]
                    ]
                else:
                    shortest = None
                    distance = 0.0
                    distinguishing[pair_key] = []
                rows.append(
                    {
                        "value_set": value_set,
                        "language": language_name,
                        "source": left,
                        "target": right,
                        "distinguishable": bool(differing_indices),
                        "num_distinguishing_formulas": len(differing_indices),
                        "shortest_distinguishing_depth": shortest,
                        "min_probe_set_size_all_candidates": len(min_probe_set),
                        "pairwise_semantic_distance": distance,
                        "formula_pool_size": len(formulas),
                    }
                )
    return rows, distinguishing, formula_pools


def greedy_set_cover_probes(
    candidate_names: Sequence[str],
    formulas: Sequence[V3Formula],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    max_probes: int | None = None,
) -> list[V3Formula]:
    """Select formulas that separate as many remaining candidate pairs as possible."""

    pairs = {tuple(sorted(pair)) for pair in itertools.combinations(candidate_names, 2)}
    covered: set[tuple[str, str]] = set()
    selected: list[V3Formula] = []
    remaining = list(range(len(formulas)))
    max_probes = len(formulas) if max_probes is None else max_probes
    while remaining and covered != pairs and len(selected) < max_probes:
        best_idx = remaining[0]
        best_key: tuple[int, int, int] | None = None
        for idx in remaining:
            formula_pairs = {
                pair
                for pair in pairs
                if vectors[pair[0]][idx] != vectors[pair[1]][idx]
            }
            new_pairs = formula_pairs - covered
            key = (len(new_pairs), len(formula_pairs), -formulas[idx].depth)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        selected.append(formulas[best_idx])
        for pair in pairs:
            if vectors[pair[0]][best_idx] != vectors[pair[1]][best_idx]:
                covered.add(pair)
        remaining.remove(best_idx)
    return selected


def source_contrastive_probes(
    source_name: str,
    candidate_names: Sequence[str],
    formulas: Sequence[V3Formula],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    max_probes: int = 8,
) -> list[int]:
    """Select probes that separate the known source from possible alternatives.

    This uses the source semantic identity and predefined candidate library, but
    not the realized target identity or any post-shift labels.
    """

    alternatives = {name for name in candidate_names if name != source_name}
    covered: set[str] = set()
    selected: list[int] = []
    remaining = list(range(len(formulas)))
    while remaining and len(selected) < max_probes and covered != alternatives:
        best_idx = remaining[0]
        best_key: tuple[int, int, int] | None = None
        for idx in remaining:
            separated = {
                name
                for name in alternatives
                if vectors[source_name][idx] != vectors[name][idx]
            }
            new = separated - covered
            key = (len(new), len(separated), -formulas[idx].depth)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        selected.append(best_idx)
        covered.update(
            name for name in alternatives if vectors[source_name][best_idx] != vectors[name][best_idx]
        )
        remaining.remove(best_idx)

    if len(selected) < max_probes:
        ranked = sorted(
            remaining,
            key=lambda idx: (
                sum(vectors[source_name][idx] != vectors[name][idx] for name in alternatives),
                -formulas[idx].depth,
            ),
            reverse=True,
        )
        selected.extend(ranked[: max_probes - len(selected)])
    return selected


def information_gain_probe(
    candidate_names: Sequence[str],
    formulas: Sequence[V3Formula],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    posterior: Mapping[str, float],
    used_indices: set[int] | None = None,
) -> int:
    used_indices = used_indices or set()
    base_entropy = _entropy([posterior[name] for name in candidate_names])
    best_idx = 0
    best_gain = -1.0
    for idx, _ in enumerate(formulas):
        if idx in used_indices:
            continue
        groups: dict[tuple[str, ...], float] = defaultdict(float)
        for name in candidate_names:
            groups[vectors[name][idx]] += posterior[name]
        expected_entropy = 0.0
        for observation, prob in groups.items():
            if prob <= 0:
                continue
            conditional = [
                posterior[name] / prob
                for name in candidate_names
                if vectors[name][idx] == observation
            ]
            expected_entropy += prob * _entropy(conditional)
        gain = base_entropy - expected_entropy
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    return best_idx


def worst_case_ambiguity_probe(
    candidate_names: Sequence[str],
    formulas: Sequence[V3Formula],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    posterior: Mapping[str, float],
    used_indices: set[int] | None = None,
) -> int:
    used_indices = used_indices or set()
    active = [name for name in candidate_names if posterior[name] > 1e-6]
    best_idx = 0
    best_key: tuple[float, int] | None = None
    for idx, _ in enumerate(formulas):
        if idx in used_indices:
            continue
        buckets: Counter[tuple[str, ...]] = Counter(vectors[name][idx] for name in active)
        largest_bucket = max(buckets.values()) if buckets else 0
        key = (-largest_bucket, len(buckets))
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    return best_idx


def update_bayesian_posterior(
    candidate_names: Sequence[str],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    posterior: Mapping[str, float],
    formula_index: int,
    observation: tuple[str, ...],
    observation_accuracy: float = 0.995,
) -> dict[str, float]:
    scores = {}
    miss = (1.0 - observation_accuracy) / max(len(candidate_names) - 1, 1)
    for name in candidate_names:
        likelihood = observation_accuracy if vectors[name][formula_index] == observation else miss
        scores[name] = posterior[name] * likelihood
    total = sum(scores.values())
    if total <= 0:
        return {name: 1.0 / len(candidate_names) for name in candidate_names}
    return {name: score / total for name, score in scores.items()}


def run_v3_experiment(
    requested_device: str,
    out_dir: Path,
    mode: str = "v3",
    base_seed: int = 7,
) -> dict[str, object]:
    start = time.perf_counter()
    device = torch.device(requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if mode in {"v3", "v3_smoke"} and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch reports it unavailable.")
    if mode == "v3" and device.type != "cuda":
        raise RuntimeError("Full V3 stops rather than silently running on CPU.")

    out_dir.mkdir(parents=True, exist_ok=True)
    Path("paper_assets").mkdir(exist_ok=True)

    max_depth = 2 if mode != "v3_smoke" else 1
    diag_rows, distinguishing, formula_pools = compute_diagnosability(
        SEMANTICS_V3, LANGUAGES, max_depth=max_depth
    )
    regimes = _v3_regimes()
    seeds = [base_seed, base_seed + 17, base_seed + 31] if mode == "v3" else [base_seed]
    budgets = [1, 2, 4, 8, 16, 32] if mode == "v3" else [2, 8]

    metrics: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    timelines: dict[str, object] = {}
    for regime in regimes:
        value_set = SEMANTICS_V3[regime["source"]].value_set
        language_name = regime["language"]
        formulas = formula_pools[f"{value_set}:{language_name}"]
        candidate_names = [
            name for name, sem in SEMANTICS_V3.items() if sem.value_set == value_set
        ]
        vectors = _vectors_for_candidates(candidate_names, formulas)
        selected_indices = source_contrastive_probes(
            source_name=regime["source"],
            candidate_names=candidate_names,
            formulas=formulas,
            vectors=vectors,
            max_probes=8,
        )
        diagnosable = _pair_diagnosable(diag_rows, value_set, language_name, regime["source"], regime["target"])
        source_vector_list = vectors[regime["source"]]
        target_vector_list = vectors[regime["target"]]
        distinguishing_indices = [
            idx for idx in range(len(formulas)) if source_vector_list[idx] != target_vector_list[idx]
        ]

        for seed in seeds:
            for strategy in ("confidence", "input_histogram", "oracle_error_rate", "passive_natural", "random", "selected", "eig_adaptive", "worst_case"):
                result = simulate_v3_online_monitor(
                    source=regime["source"],
                    target=regime["target"],
                    candidate_names=candidate_names,
                    formulas=formulas,
                    vectors=vectors,
                    selected_indices=selected_indices,
                    strategy=strategy,
                    budget=8,
                    seed=seed,
                    diagnosable=diagnosable,
                )
                result.update(regime)
                result.update(
                    {
                        "seed": seed,
                        "value_set": value_set,
                        "diagnosable": diagnosable,
                        "num_distinguishing_formulas": len(distinguishing_indices),
                        "formula_pool_size": len(formulas),
                        "distinguishing_density": len(distinguishing_indices) / max(len(formulas), 1),
                    }
                )
                metrics.append(result)
                if regime["name"] in {"sparse_l2_hard", "sparse_l3_hard"} and strategy in {"random", "selected", "eig_adaptive"}:
                    current = timelines.get(strategy)
                    current_max = (
                        -1
                        if current is None
                        else max(row["confident_wrong_cumulative"] for row in current)
                    )
                    new_max = max(row["confident_wrong_cumulative"] for row in result["timeline"])
                    if new_max > current_max:
                        timelines[strategy] = result["timeline"]

            for budget in budgets:
                for strategy in ("random", "selected", "eig_adaptive"):
                    result = simulate_v3_online_monitor(
                        source=regime["source"],
                        target=regime["target"],
                        candidate_names=candidate_names,
                        formulas=formulas,
                        vectors=vectors,
                        selected_indices=selected_indices,
                        strategy=strategy,
                        budget=budget,
                        seed=seed,
                        diagnosable=diagnosable,
                    )
                    budget_rows.append(
                        {
                            "regime": regime["name"],
                            "difficulty": regime["difficulty"],
                            "language": language_name,
                            "strategy": strategy,
                            "budget": budget,
                            "seed": seed,
                            "diagnosable": diagnosable,
                            "detection_delay": result["detection_delay"],
                            "false_alarm_rate": result["false_alarm_rate"],
                            "confident_wrong_before_alarm": result["confident_wrong_before_alarm"],
                            "probes_used": result["probes_used"],
                            "semantic_identification_accuracy": result["semantic_identification_accuracy"],
                        }
                    )

    aggregate_rows = _aggregate_v3_metrics(metrics)
    budget_aggregates = _aggregate_v3_budget(budget_rows)
    runtime_s = time.perf_counter() - start
    cuda = _cuda_status(device)
    summary = {
        "mode": mode,
        "cuda": cuda,
        "runtime_seconds": runtime_s,
        "diagnosability_rows": diag_rows,
        "metrics": [{k: v for k, v in row.items() if k != "timeline"} for row in metrics],
        "metric_aggregates": aggregate_rows,
        "budget_rows": budget_rows,
        "budget_aggregates": budget_aggregates,
        "headline": _v3_headline(mode, aggregate_rows, diag_rows, runtime_s),
        "interpreter_note": {
            "actual": sys.executable,
        },
    }

    _write_dict_csv(out_dir / "diagnosability_matrix.csv", diag_rows)
    _write_json(out_dir / "distinguishing_formulas.json", distinguishing)
    _write_dict_csv(out_dir / "metrics.csv", summary["metrics"])
    _write_dict_csv(out_dir / "metric_aggregates.csv", aggregate_rows)
    _write_dict_csv(out_dir / "probe_budget.csv", budget_rows)
    _write_dict_csv(out_dir / "probe_budget_aggregates.csv", budget_aggregates)
    _write_json(out_dir / "metrics.json", summary)
    _write_v3_plots(out_dir, diag_rows, aggregate_rows, budget_aggregates, timelines)
    _write_v3_report(Path("paper_assets") / "SEMANTIC_RULE_SHIFT_V3_DIAGNOSABILITY.md", summary)
    return summary


def simulate_v3_online_monitor(
    source: str,
    target: str,
    candidate_names: Sequence[str],
    formulas: Sequence[V3Formula],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    selected_indices: Sequence[int],
    strategy: str,
    budget: int,
    seed: int,
    diagnosable: bool,
    n_pre: int = 80,
    n_post: int = 80,
    probe_interval: int = 8,
    alarm_posterior: float = 0.25,
) -> dict[str, object]:
    rng = random.Random(seed + _stable_offset(source, target, strategy, str(budget)))
    n_total = n_pre + n_post
    posterior = {name: 1.0 / len(candidate_names) for name in candidate_names}
    used_indices: set[int] = set()
    alarm_index = None
    probes_used = 0
    semantic_correct = 0
    confident_wrong_before_alarm = 0
    timeline = []

    # Passive samples and task stream reuse the same formula multiset before and
    # after the changepoint, so input histograms are exactly matched.
    stream_indices = [rng.randrange(len(formulas)) for _ in range(n_pre)]
    stream_indices = stream_indices + list(stream_indices)
    random_order = list(range(len(formulas)))
    rng.shuffle(random_order)
    random_cursor = 0
    selected_cursor = 0

    for step in range(n_total):
        true_name = source if step < n_pre else target
        source_output = vectors[source][stream_indices[step]]
        true_output = vectors[true_name][stream_indices[step]]
        confident_wrong = step >= n_pre and source_output != true_output
        if confident_wrong and alarm_index is None:
            confident_wrong_before_alarm += 1

        if strategy in {"random", "selected", "eig_adaptive", "worst_case"} and step % probe_interval == 0:
            posterior = _soften_posterior(posterior, candidate_names, mixing=0.12)
            for _ in range(budget):
                if strategy == "random":
                    formula_index = random_order[random_cursor % len(random_order)]
                    random_cursor += 1
                elif strategy == "selected":
                    formula_index = selected_indices[selected_cursor % len(selected_indices)]
                    selected_cursor += 1
                elif strategy == "eig_adaptive":
                    formula_index = information_gain_probe(candidate_names, formulas, vectors, posterior, used_indices)
                else:
                    formula_index = worst_case_ambiguity_probe(candidate_names, formulas, vectors, posterior, used_indices)
                used_indices.add(formula_index)
                probes_used += 1
                observation = vectors[true_name][formula_index]
                posterior = update_bayesian_posterior(
                    candidate_names, vectors, posterior, formula_index, observation
                )
        elif strategy == "passive_natural":
            posterior = _soften_posterior(posterior, candidate_names, mixing=0.03)
            formula_index = stream_indices[step]
            observation = true_output
            posterior = update_bayesian_posterior(
                candidate_names, vectors, posterior, formula_index, observation, observation_accuracy=0.99
            )

        best_name = max(posterior, key=posterior.get)
        semantic_correct += int(best_name == true_name)
        score = 1.0 - posterior[source]

        if strategy == "oracle_error_rate":
            recent = [
                vectors[source][stream_indices[j]] != vectors[source if j < n_pre else target][stream_indices[j]]
                for j in range(max(0, step - 15), step + 1)
            ]
            decision = step >= n_pre and mean(recent) > 0.20
        elif strategy in {"confidence", "input_histogram"}:
            decision = False
        else:
            decision = step >= 10 and best_name != source and posterior[source] < alarm_posterior

        if alarm_index is None and decision:
            alarm_index = step
        timeline.append(
            {
                "step": step,
                "score": score,
                "best": best_name,
                "true": true_name,
                "probes": probes_used,
                "confident_wrong_cumulative": confident_wrong_before_alarm,
                "alarm": alarm_index == step,
            }
        )

    post_alarm = alarm_index is not None and alarm_index >= n_pre
    false_alarm_rate = 1.0 if alarm_index is not None and alarm_index < n_pre else 0.0
    input_js = 0.0
    output_js = _label_js(
        [vectors[source][idx] for idx in stream_indices[:n_pre]],
        [vectors[target][idx] for idx in stream_indices[n_pre:]],
    )
    return {
        "strategy": strategy,
        "budget": budget if strategy in {"random", "selected", "eig_adaptive", "worst_case"} else 0,
        "alarm_index": alarm_index,
        "detection_delay": alarm_index - n_pre if post_alarm else None,
        "false_alarm_rate": false_alarm_rate,
        "confident_wrong_before_alarm": confident_wrong_before_alarm,
        "probes_used": probes_used,
        "semantic_identification_accuracy": semantic_correct / n_total,
        "input_js_divergence": input_js,
        "output_label_js_divergence": output_js,
        "theoretically_diagnosable": diagnosable,
        "timeline": timeline,
    }


def _soften_posterior(
    posterior: Mapping[str, float],
    candidate_names: Sequence[str],
    mixing: float,
) -> dict[str, float]:
    uniform = 1.0 / len(candidate_names)
    softened = {
        name: (1.0 - mixing) * posterior[name] + mixing * uniform
        for name in candidate_names
    }
    total = sum(softened.values())
    return {name: value / total for name, value in softened.items()}


def _stable_offset(*parts: str) -> int:
    text = "|".join(parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 100000


def _three_system(
    name: str,
    not_table: Mapping[str, str],
    and_fn: Callable[[str, str], str],
    or_fn: Callable[[str, str], str],
    description: str,
) -> V3SemanticSystem:
    unary = {"NOT": dict(not_table)}
    binary = {
        "AND": {(a, b): and_fn(a, b) for a in THREE_VALUES for b in THREE_VALUES},
        "OR": {(a, b): or_fn(a, b) for a in THREE_VALUES for b in THREE_VALUES},
    }
    _add_derived_binary(binary, THREE_VALUES, unary)
    return V3SemanticSystem(name, THREE_VALUES, unary, binary, description)


def _four_system(
    name: str,
    normalize: Callable[[str], str],
    description: str,
) -> V3SemanticSystem:
    def pair(value: str) -> tuple[int, int]:
        value = normalize(value)
        return {
            "T": (1, 0),
            "F": (0, 1),
            "B": (1, 1),
            "N": (0, 0),
        }[value]

    def from_pair(bits: tuple[int, int]) -> str:
        return {
            (1, 0): "T",
            (0, 1): "F",
            (1, 1): "B",
            (0, 0): "N",
        }[bits]

    unary = {
        "NOT": {
            value: from_pair((pair(value)[1], pair(value)[0]))
            for value in FOUR_VALUES
        }
    }
    binary = {
        "AND": {
            (a, b): from_pair((pair(a)[0] & pair(b)[0], pair(a)[1] | pair(b)[1]))
            for a in FOUR_VALUES
            for b in FOUR_VALUES
        },
        "OR": {
            (a, b): from_pair((pair(a)[0] | pair(b)[0], pair(a)[1] & pair(b)[1]))
            for a in FOUR_VALUES
            for b in FOUR_VALUES
        },
    }
    _add_derived_binary(binary, FOUR_VALUES, unary)
    return V3SemanticSystem(name, FOUR_VALUES, unary, binary, description)


def _bochvar_four_system() -> V3SemanticSystem:
    fde = _four_system("fde_temp", lambda value: value, "")
    unary = {
        "NOT": {
            value: "N" if value in {"B", "N"} else fde.unary["NOT"][value]
            for value in FOUR_VALUES
        }
    }
    binary = {}
    for op in ("AND", "OR"):
        binary[op] = {}
        for a in FOUR_VALUES:
            for b in FOUR_VALUES:
                binary[op][(a, b)] = "N" if "N" in (a, b) or "B" in (a, b) else fde.binary[op][(a, b)]
    _add_derived_binary(binary, FOUR_VALUES, unary)
    return V3SemanticSystem(
        "bochvar4",
        FOUR_VALUES,
        unary,
        binary,
        "Four-valued Bochvar/internal variant where B/N contaminate compounds.",
    )


def _projected_k3_four_system(
    name: str,
    unknown_value: str,
    description: str,
) -> V3SemanticSystem:
    def project(value: str) -> str:
        return value if value in {"T", "F"} else "U"

    def lift(value: str) -> str:
        return {"T": "T", "F": "F", "U": unknown_value}[value]

    unary = {
        "NOT": {
            value: lift({"T": "F", "F": "T", "U": "U"}[project(value)])
            for value in FOUR_VALUES
        }
    }
    binary = {
        "AND": {
            (a, b): lift(_k3_and(project(a), project(b)))
            for a in FOUR_VALUES
            for b in FOUR_VALUES
        },
        "OR": {
            (a, b): lift(_k3_or(project(a), project(b)))
            for a in FOUR_VALUES
            for b in FOUR_VALUES
        },
    }
    _add_derived_binary(binary, FOUR_VALUES, unary)
    return V3SemanticSystem(name, FOUR_VALUES, unary, binary, description)


def _with_binary_override(
    base: V3SemanticSystem,
    name: str,
    op: str,
    entry: tuple[str, str],
    value: str,
    description: str,
) -> V3SemanticSystem:
    unary = {key: dict(table) for key, table in base.unary.items()}
    binary = {key: dict(table) for key, table in base.binary.items()}
    binary[op][entry] = value
    return V3SemanticSystem(name, base.values, unary, binary, description)


def _add_derived_binary(
    binary: dict[str, dict[tuple[str, str], str]],
    values: Sequence[str],
    unary: Mapping[str, Mapping[str, str]],
) -> None:
    binary["IMPLIES"] = {
        (a, b): binary["OR"][(unary["NOT"][a], b)] for a in values for b in values
    }
    binary["EQUIV"] = {
        (a, b): binary["AND"][(binary["IMPLIES"][(a, b)], binary["IMPLIES"][(b, a)])]
        for a in values
        for b in values
    }
    binary["XOR"] = {
        (a, b): unary["NOT"][binary["EQUIV"][(a, b)]] for a in values for b in values
    }
    binary["DIAG_EQ"] = {
        (a, b): "T" if a == b else "F" for a in values for b in values
    }


def _k3_and(a: str, b: str) -> str:
    if a == "F" or b == "F":
        return "F"
    if a == "T" and b == "T":
        return "T"
    return "U"


def _k3_or(a: str, b: str) -> str:
    if a == "T" or b == "T":
        return "T"
    if a == "F" and b == "F":
        return "F"
    return "U"


def _bochvar_and(a: str, b: str) -> str:
    if a == "U" or b == "U":
        return "U"
    return "T" if a == "T" and b == "T" else "F"


def _bochvar_or(a: str, b: str) -> str:
    if a == "U" or b == "U":
        return "U"
    return "T" if a == "T" or b == "T" else "F"


SEMANTICS_V3 = build_v3_semantics()


def _vectors_for_candidates(
    names: Sequence[str], formulas: Sequence[V3Formula]
) -> dict[str, list[tuple[str, ...]]]:
    return {
        name: [SEMANTICS_V3[name].truth_vector(formula) for formula in formulas]
        for name in names
    }


def _pair_diagnosable(
    rows: Sequence[dict[str, object]],
    value_set: str,
    language: str,
    source: str,
    target: str,
) -> bool:
    for row in rows:
        if row["value_set"] != value_set or row["language"] != language:
            continue
        if {row["source"], row["target"]} == {source, target}:
            return bool(row["distinguishable"])
    return False


def _truth_vector_distance(left: Sequence[str], right: Sequence[str]) -> float:
    width = max(len(left), len(right))
    return sum(a != b for a, b in itertools.zip_longest(left, right)) / width


def _entropy(probs: Sequence[float]) -> float:
    return -sum(prob * math.log2(prob) for prob in probs if prob > 0)


def _label_js(left_vectors: Sequence[tuple[str, ...]], right_vectors: Sequence[tuple[str, ...]]) -> float:
    left = Counter(item for vector in left_vectors for item in vector)
    right = Counter(item for vector in right_vectors for item in vector)
    values = sorted(set(left) | set(right))
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 or right_total == 0:
        return 0.0
    p = [left[value] / left_total for value in values]
    q = [right[value] / right_total for value in values]
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)


def _v3_regimes() -> list[dict[str, str]]:
    return [
        {
            "name": "equivalent_l0",
            "difficulty": "equivalent",
            "source": "k3_3",
            "target": "lp_3",
            "language": "L0",
        },
        {
            "name": "l2_unlocked",
            "difficulty": "equiv_under_l0_distinguishable_l2",
            "source": "k3_3",
            "target": "k3_imp_flip_3",
            "language": "L2",
        },
        {
            "name": "easy_large",
            "difficulty": "easy_dense",
            "source": "fde4",
            "target": "bochvar4",
            "language": "L1",
        },
        {
            "name": "medium_boundary",
            "difficulty": "medium",
            "source": "fde4",
            "target": "custom_boundary_2_4",
            "language": "L1",
        },
        {
            "name": "sparse_l2_hard",
            "difficulty": "sparse_hard",
            "source": "fde4",
            "target": "rare_implies_flip4",
            "language": "L2",
        },
        {
            "name": "sparse_l3_hard",
            "difficulty": "one_entry_sparse",
            "source": "fde4",
            "target": "rare_xor_flip4",
            "language": "L3",
        },
    ]


def _aggregate_v3_metrics(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["name"]), str(row["strategy"]))].append(row)
    output = []
    for (regime, strategy), items in sorted(grouped.items()):
        output.append(
            {
                "regime": regime,
                "difficulty": items[0]["difficulty"],
                "language": items[0]["language"],
                "source": items[0]["source"],
                "target": items[0]["target"],
                "strategy": strategy,
                "diagnosable": items[0]["diagnosable"],
                "num_distinguishing_formulas": items[0]["num_distinguishing_formulas"],
                "formula_pool_size": items[0]["formula_pool_size"],
                "distinguishing_density": items[0]["distinguishing_density"],
                "mean_detection_delay": _mean_optional(row["detection_delay"] for row in items),
                "std_detection_delay": _std_optional(row["detection_delay"] for row in items),
                "miss_rate": sum(row["detection_delay"] is None for row in items) / len(items),
                "mean_false_alarm_rate": mean(float(row["false_alarm_rate"]) for row in items),
                "mean_confident_wrong_before_alarm": mean(float(row["confident_wrong_before_alarm"]) for row in items),
                "mean_probes_used": mean(float(row["probes_used"]) for row in items),
                "mean_semantic_identification_accuracy": mean(float(row["semantic_identification_accuracy"]) for row in items),
                "mean_input_js_divergence": mean(float(row["input_js_divergence"]) for row in items),
                "mean_output_label_js_divergence": mean(float(row["output_label_js_divergence"]) for row in items),
            }
        )
    return output


def _aggregate_v3_budget(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["strategy"]), int(row["budget"]))].append(row)
    output = []
    for (regime, strategy, budget), items in sorted(grouped.items()):
        output.append(
            {
                "regime": regime,
                "difficulty": items[0]["difficulty"],
                "language": items[0]["language"],
                "strategy": strategy,
                "budget": budget,
                "diagnosable": items[0]["diagnosable"],
                "mean_detection_delay": _mean_optional(row["detection_delay"] for row in items),
                "miss_rate": sum(row["detection_delay"] is None for row in items) / len(items),
                "mean_confident_wrong_before_alarm": mean(float(row["confident_wrong_before_alarm"]) for row in items),
                "mean_probes_used": mean(float(row["probes_used"]) for row in items),
                "mean_semantic_identification_accuracy": mean(float(row["semantic_identification_accuracy"]) for row in items),
            }
        )
    return output


def _v3_headline(
    mode: str,
    aggregate_rows: Sequence[dict[str, object]],
    diag_rows: Sequence[dict[str, object]],
    runtime_s: float,
) -> dict[str, object]:
    sparse_selected = [
        row for row in aggregate_rows if row["strategy"] == "selected" and "sparse" in str(row["difficulty"])
    ]
    sparse_random = [
        row for row in aggregate_rows if row["strategy"] == "random" and "sparse" in str(row["difficulty"])
    ]
    equiv = [
        row for row in aggregate_rows if row["strategy"] == "selected" and not bool(row["diagnosable"])
    ]
    return {
        "mode": mode,
        "diagnosability_rows": len(diag_rows),
        "sparse_selected_mean_delay": _mean_optional(row["mean_detection_delay"] for row in sparse_selected),
        "sparse_random_mean_delay": _mean_optional(row["mean_detection_delay"] for row in sparse_random),
        "equivalent_selected_miss_rate": mean(float(row["miss_rate"]) for row in equiv) if equiv else None,
        "runtime_seconds": runtime_s,
    }


def _cuda_status(device: torch.device) -> dict[str, object]:
    return {
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "selected_device": str(device),
    }


def _write_dict_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_v3_plots(
    out_dir: Path,
    diag_rows: Sequence[dict[str, object]],
    aggregate_rows: Sequence[dict[str, object]],
    budget_rows: Sequence[dict[str, object]],
    timelines: Mapping[str, object],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    languages = sorted({str(row["language"]) for row in diag_rows})
    rates = []
    for language in languages:
        language_rows = [row for row in diag_rows if row["language"] == language]
        rates.append(sum(bool(row["distinguishable"]) for row in language_rows) / len(language_rows))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(languages, rates, color="#34699a")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("distinguishable pair fraction")
    ax.set_title("Diagnosability by language level")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnosability_by_language.png", dpi=160)
    plt.close(fig)

    regimes = sorted({str(row["regime"]) for row in aggregate_rows})
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.25
    for offset, strategy in enumerate(("random", "selected", "eig_adaptive")):
        values = []
        for regime in regimes:
            row = next(
                (item for item in aggregate_rows if item["regime"] == regime and item["strategy"] == strategy),
                None,
            )
            delay = None if row is None else row["mean_detection_delay"]
            values.append(160 if delay is None else float(delay))
        ax.bar([i + (offset - 1) * width for i in range(len(regimes))], values, width=width, label=strategy)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, rotation=35, ha="right")
    ax.set_ylabel("mean delay (misses as 160)")
    ax.set_title("Selected vs random probe delay by regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "selected_vs_random_delay.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for strategy in ("random", "selected", "eig_adaptive"):
        grouped = [
            row for row in budget_rows if row["strategy"] == strategy and row["regime"] == "sparse_l3_hard"
        ]
        grouped = sorted(grouped, key=lambda row: int(row["budget"]))
        if not grouped:
            continue
        ax.plot(
            [int(row["budget"]) for row in grouped],
            [160 if row["mean_detection_delay"] is None else float(row["mean_detection_delay"]) for row in grouped],
            marker="o",
            label=strategy,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("probe budget")
    ax.set_ylabel("mean delay")
    ax.set_title("Probe budget tradeoff on sparse hard shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "probe_budget_tradeoff.png", dpi=160)
    plt.close(fig)

    if timelines:
        fig, ax = plt.subplots(figsize=(10, 4))
        for strategy, timeline in timelines.items():
            ax.plot(
                [row["step"] for row in timeline],
                [row["confident_wrong_cumulative"] for row in timeline],
                label=f"{strategy} cumulative confident wrong",
            )
            alarms = [row["step"] for row in timeline if row["alarm"]]
            if alarms:
                ax.axvline(alarms[0], linestyle=":", label=f"{strategy} alarm")
        ax.axvline(80, color="black", linestyle="--", label="changepoint")
        ax.set_xlabel("step")
        ax.set_ylabel("count")
        ax.set_title("Representative sparse hard-shift timeline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "timeline_sparse_hard.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = []
    ys = []
    labels = []
    for row in aggregate_rows:
        if row["strategy"] != "selected":
            continue
        random_row = next(
            (
                item
                for item in aggregate_rows
                if item["regime"] == row["regime"] and item["strategy"] == "random"
            ),
            None,
        )
        if random_row is None:
            continue
        selected_delay = 160 if row["mean_detection_delay"] is None else float(row["mean_detection_delay"])
        random_delay = 160 if random_row["mean_detection_delay"] is None else float(random_row["mean_detection_delay"])
        xs.append(float(row["num_distinguishing_formulas"]))
        ys.append(random_delay - selected_delay)
        labels.append(str(row["regime"]))
    ax.scatter(xs, ys, color="#7a3e9d")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=7)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("number of distinguishing formulas")
    ax.set_ylabel("random delay - selected delay")
    ax.set_title("Distinguishing formulas vs selected-probe advantage")
    fig.tight_layout()
    fig.savefig(out_dir / "distinguishing_vs_advantage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    rows = [row for row in aggregate_rows if row["strategy"] == "confidence"]
    labels = [str(row["regime"]) for row in rows]
    ax.bar(range(len(rows)), [float(row["mean_confident_wrong_before_alarm"]) for row in rows])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("confident wrong before alarm")
    ax.set_title("Confidence detector misses while source-rule confidence stays high")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_high_accuracy_drops.png", dpi=160)
    plt.close(fig)


def _write_v3_report(path: Path, summary: Mapping[str, object]) -> None:
    cuda = summary["cuda"]
    note = summary["interpreter_note"]
    headline = summary["headline"]
    aggregates = summary["metric_aggregates"]
    diag_rows = summary["diagnosability_rows"]
    budget_rows = summary["budget_aggregates"]
    lines = [
        "# Semantic Rule-Shift V3 Diagnosability",
        "",
        "## Commands Run",
        "",
        f"- Requested interpreter existed: `{note['requested_exists']}`",
        f"- Requested interpreter: `{note['requested']}`",
        f"- Actual interpreter: `{note['actual']}`",
        f"- `{cuda['python_executable']} -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\"`",
        f"- `{cuda['python_executable']} -m pytest`",
        f"- `{cuda['python_executable']} -m semantic_rule_shift.experiments --mode v3 --device cuda --out results/semantic_rule_shift_v3`",
        "",
        "## CUDA Status",
        "",
        f"- Torch: `{cuda['torch']}`",
        f"- CUDA available: `{cuda['cuda_available']}`",
        f"- GPU: `{cuda['device']}`",
        f"- Selected device: `{cuda['selected_device']}`",
        f"- Runtime: `{summary['runtime_seconds']:.2f}` seconds",
        "",
        "## Plain-Language Proposition",
        "",
        "If two semantic systems induce identical truth vectors for every formula in the allowed probe language, then any detector that only observes outputs from that language receives the same evidence under both worlds. Such a detector cannot distinguish the two semantics with probability better than chance without extra assumptions or another observation channel.",
        "",
        "This explains the V2 K3 <-> LP result: under the current NOT/AND/OR, T/F/U interface, `k3_3` and `lp_3` have identical truth functions, so the correct conclusion is observational equivalence rather than detector failure.",
        "",
        "## Diagnosability Summary",
        "",
        "| Language | Value set | Pair count | Distinguishable fraction |",
        "| --- | --- | ---: | ---: |",
    ]
    for language in sorted({row["language"] for row in diag_rows}):
        for value_set in ("three", "four"):
            rows = [row for row in diag_rows if row["language"] == language and row["value_set"] == value_set]
            if not rows:
                continue
            lines.append(
                f"| {language} | {value_set} | {len(rows)} | {sum(bool(row['distinguishable']) for row in rows) / len(rows):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Online Results",
            "",
            "| Regime | Difficulty | Language | Strategy | Diagnosable | Delay | Miss | Confident wrong | Probes | Semantic ID |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    display = {"confidence", "input_histogram", "oracle_error_rate", "passive_natural", "random", "selected", "eig_adaptive", "worst_case"}
    for row in aggregates:
        if row["strategy"] not in display:
            continue
        lines.append(
            "| {regime} | {difficulty} | {language} | {strategy} | {diag} | {delay} | {miss:.3f} | {cw:.2f} | {probes:.1f} | {sid:.3f} |".format(
                regime=row["regime"],
                difficulty=row["difficulty"],
                language=row["language"],
                strategy=row["strategy"],
                diag=row["diagnosable"],
                delay=_fmt_float(row["mean_detection_delay"]),
                miss=float(row["miss_rate"]),
                cw=float(row["mean_confident_wrong_before_alarm"]),
                probes=float(row["mean_probes_used"]),
                sid=float(row["mean_semantic_identification_accuracy"]),
            )
        )

    lines.extend(
        [
            "",
            "## Probe Budget Sweep",
            "",
            "| Regime | Strategy | Budget | Delay | Miss | Confident wrong | Semantic ID |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in budget_rows:
        if row["regime"] not in {"sparse_l2_hard", "sparse_l3_hard"}:
            continue
        lines.append(
            "| {regime} | {strategy} | {budget} | {delay} | {miss:.3f} | {cw:.2f} | {sid:.3f} |".format(
                regime=row["regime"],
                strategy=row["strategy"],
                budget=row["budget"],
                delay=_fmt_float(row["mean_detection_delay"]),
                miss=float(row["miss_rate"]),
                cw=float(row["mean_confident_wrong_before_alarm"]),
                sid=float(row["mean_semantic_identification_accuracy"]),
            )
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            _v3_verdict(headline),
            "",
            "## Failure Cases And Caveats",
            "",
            "- The V3 online experiment is symbolic. It uses a high-confidence source-rule reasoner to count confident wrong decisions; this isolates diagnosability but is not a full neural architecture benchmark.",
            "- Active truth-vector probes are stronger than passive deployment. The report therefore includes passive natural samples and random probes with equal budgets.",
            "- Wins on sparse regimes must be interpreted together with language expressivity: a shift can be impossible under L0 and easy under L2/L3.",
            "- Artificial `L4` diagnostic operators are included only as an upper bound and are not needed for the main V3 claims.",
            "",
            "## Next Experiments",
            "",
            "- Train a neural reasoner over the V3 language levels, including four-valued outputs.",
            "- Replace truth-vector probes with single-assignment probes and measure the extra sample complexity.",
            "- Add canonical formula minimization so formula-pool sizes are not inflated by syntactic variants.",
            "- Calibrate detector thresholds on held-out streams and run 20-50 seeds.",
            "- Decide whether the paper is a standalone diagnosability paper or a theory section in a broader semantic reasoning work.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _v3_verdict(headline: Mapping[str, object]) -> str:
    selected = headline["sparse_selected_mean_delay"]
    random_delay = headline["sparse_random_mean_delay"]
    equiv_miss = headline["equivalent_selected_miss_rate"]
    if selected is not None and random_delay is not None and float(selected) < float(random_delay) and equiv_miss == 1.0:
        return (
            "The V3 reframe is stronger than the V2 detector-only story: it explains "
            "observational equivalence, shows that selected/adaptive probes can win on "
            "sparse diagnosable regimes, and preserves the high-confidence wrong-answer "
            "motivation under matched inputs."
        )
    return (
        "V3 is promising but not conclusive. It provides the right diagnosability scaffold, "
        "but selected probes need stronger evidence across more realistic neural runs."
    )


def _mean_optional(values: Iterable[object]) -> float | None:
    vals = [float(value) for value in values if value is not None and value != ""]
    return None if not vals else mean(vals)


def _std_optional(values: Iterable[object]) -> float | None:
    vals = [float(value) for value in values if value is not None and value != ""]
    if not vals:
        return None
    return 0.0 if len(vals) == 1 else pstdev(vals)


def _fmt_float(value: object) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.3f}"
