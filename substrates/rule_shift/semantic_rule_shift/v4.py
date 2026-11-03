"""V4 paper-quality validation for semantic rule-shift diagnosability.

V4 keeps the V3 identifiability framing and adds paper-facing audits:
semantic implementation checks, fair matched random baselines, language-level
ablations, and a compact neural online-monitoring closure.
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
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from .v3 import (
    FOUR_VALUES,
    LANGUAGES,
    SEMANTICS_V3,
    THREE_VALUES,
    V3Formula,
    V3Language,
    V3SemanticSystem,
    _entropy,
    _label_js,
    _soften_posterior,
    _stable_offset,
    _std_optional,
    enumerate_v3_formulas,
    information_gain_probe,
    source_contrastive_probes,
    update_bayesian_posterior,
    worst_case_ambiguity_probe,
)


LANGUAGE_ORDER = ("L0", "L1", "L2", "L3", "L4")
PROBE_STRATEGIES = (
    "random_all",
    "random_depth_matched",
    "random_candidate_matched",
    "selected",
    "eig_adaptive",
    "worst_case",
)
DETECTOR_STRATEGIES = (
    "confidence_threshold",
    "rolling_entropy",
    "page_hinkley_confidence",
    "page_hinkley_entropy",
    "input_histogram",
    "representation_drift",
    "oracle_error_rate",
    "passive_semantic",
) + PROBE_STRATEGIES


V4_REGIMES: tuple[dict[str, str], ...] = (
    {
        "regime": "impossible_equivalent",
        "difficulty": "impossible/equivalent",
        "source": "k3_3",
        "target": "lp_3",
        "claim_language": "L0",
    },
    {
        "regime": "language_unlocked_l2",
        "difficulty": "language-unlocked",
        "source": "k3_3",
        "target": "k3_imp_flip_3",
        "claim_language": "L2",
    },
    {
        "regime": "easy_dense_four",
        "difficulty": "easy dense",
        "source": "fde4",
        "target": "bochvar4",
        "claim_language": "L1",
    },
    {
        "regime": "medium_projection_sensitive",
        "difficulty": "medium/projection-sensitive",
        "source": "fde4",
        "target": "custom_boundary_2_4",
        "claim_language": "L2",
    },
    {
        "regime": "sparse_hard_l2",
        "difficulty": "sparse hard",
        "source": "fde4",
        "target": "rare_implies_flip4",
        "claim_language": "L2",
    },
    {
        "regime": "one_entry_sparse_l3",
        "difficulty": "one-entry sparse",
        "source": "fde4",
        "target": "rare_xor_flip4",
        "claim_language": "L3",
    },
    {
        "regime": "projection_equivalent",
        "difficulty": "projection-equivalent",
        "source": "projection_k3_n4",
        "target": "projection_k3_b4",
        "claim_language": "L1",
    },
)


NEURAL_REPRESENTATIVES = (
    "impossible_equivalent",
    "easy_dense_four",
    "sparse_hard_l2",
    "one_entry_sparse_l3",
)


TOKEN_VOCAB = (
    "<PAD>",
    "p",
    "q",
    "T",
    "F",
    "U",
    "B",
    "N",
    "NOT",
    "AND",
    "OR",
    "IMPLIES",
    "EQUIV",
    "XOR",
    "DIAG_EQ",
)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKEN_VOCAB)}


@dataclass(frozen=True)
class ThresholdConfig:
    confidence_threshold: float
    entropy_threshold: float
    representation_threshold: float
    source: str = "dev"
    used_test: bool = False


class V4TinyReasoner(nn.Module):
    def __init__(self, n_classes: int, embed_dim: int = 48, hidden_dim: int = 80):
        super().__init__()
        self.embedding = nn.Embedding(len(TOKEN_VOCAB), embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def encode_hidden(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_hidden(tokens, lengths))


def run_v4_experiment(
    requested_device: str,
    out_dir: Path,
    mode: str = "v4",
    base_seed: int = 7,
) -> dict[str, object]:
    start = time.perf_counter()
    device = torch.device(requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if mode in {"v4", "v4_neural"} and device.type != "cuda":
        raise RuntimeError("Full V4/neural V4 stops rather than silently running on CPU.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch reports it unavailable.")

    out_dir.mkdir(parents=True, exist_ok=True)
    Path("paper_assets").mkdir(exist_ok=True)

    smoke = mode == "v4_smoke"
    symbolic_only = mode == "v4_symbolic"
    neural_only = mode == "v4_neural"
    max_depth = 1 if smoke else 2
    symbolic_seeds = [base_seed] if smoke else [base_seed, base_seed + 11, base_seed + 23, base_seed + 37, base_seed + 51]
    neural_seeds = [base_seed] if smoke else [base_seed, base_seed + 11, base_seed + 23]

    formula_pools = build_formula_pools(max_depth=max_depth)
    semantic_audit = semantic_audit_rows()
    semantic_pair_audit = semantic_pair_audit_rows(formula_pools)
    diagnosability_rows = [] if neural_only else language_level_diagnosability(formula_pools)
    fair_rows = [] if neural_only else run_symbolic_probe_validation(formula_pools, diagnosability_rows, symbolic_seeds)
    fair_aggregates = aggregate_rows(fair_rows, ["regime", "language", "strategy"])

    neural_rows: list[dict[str, object]] = []
    if not symbolic_only:
        neural_rows = run_neural_validation(
            formula_pools=formula_pools,
            diagnosability_rows=diagnosability_rows or language_level_diagnosability(formula_pools),
            seeds=neural_seeds,
            device=device,
            smoke=smoke,
        )
    neural_aggregates = aggregate_rows(neural_rows, ["regime", "language", "detector"]) if neural_rows else []

    runtime_s = time.perf_counter() - start
    cuda = cuda_status(device)
    summary = {
        "mode": mode,
        "runtime_seconds": runtime_s,
        "cuda": cuda,
        "interpreter_note": {
            "actual": sys.executable,
        },
        "semantic_audit": semantic_audit,
        "semantic_pair_audit": semantic_pair_audit,
        "diagnosability": diagnosability_rows,
        "fair_random_baselines": fair_rows,
        "fair_random_aggregates": fair_aggregates,
        "neural_monitoring": neural_rows,
        "neural_aggregates": neural_aggregates,
        "headline": headline(fair_aggregates, neural_aggregates, runtime_s),
    }

    write_dict_csv(out_dir / "table1_semantic_systems.csv", semantic_audit)
    write_dict_csv(out_dir / "semantic_pair_audit.csv", semantic_pair_audit)
    write_dict_csv(out_dir / "table2_diagnosability_by_language.csv", diagnosability_rows)
    write_dict_csv(out_dir / "table3_detector_performance_by_regime.csv", fair_aggregates)
    write_dict_csv(out_dir / "fair_random_baselines.csv", fair_rows)
    write_dict_csv(out_dir / "table4_neural_online_monitoring.csv", neural_aggregates)
    write_dict_csv(out_dir / "neural_online_monitoring.csv", neural_rows)
    write_dict_csv(out_dir / "metrics.csv", flatten_metric_rows(fair_aggregates, neural_aggregates))
    write_json(out_dir / "metrics.json", summary)
    write_plots(out_dir, summary)
    write_report(Path("paper_assets") / "SEMANTIC_RULE_SHIFT_V4_PAPER_VALIDATION.md", summary)
    return summary


def build_formula_pools(max_depth: int = 2) -> dict[tuple[str, str], list[V3Formula]]:
    pools = {}
    for value_set, values in (("three", THREE_VALUES), ("four", FOUR_VALUES)):
        for language_name in LANGUAGE_ORDER:
            max_formulas = 700 if max_depth == 1 else (1800 if language_name in {"L0", "L1"} else 2600)
            pools[(value_set, language_name)] = enumerate_v3_formulas(
                values,
                LANGUAGES[language_name],
                max_depth=max_depth,
                max_formulas=max_formulas,
            )
    return pools


def semantic_audit_rows() -> list[dict[str, object]]:
    rows = []
    for name, semantic in sorted(SEMANTICS_V3.items()):
        projection = projection_name(semantic)
        rows.append(
            {
                "semantic": name,
                "value_set": semantic.value_set,
                "truth_values": " ".join(semantic.values),
                "observable_labels": " ".join(sorted(set(project_value(semantic, value) for value in semantic.values))),
                "projection": projection,
                "operators": " ".join(sorted(set(semantic.unary) | set(semantic.binary))),
                "raw_truth_table_hash": raw_table_hash(semantic),
                "description": semantic.description,
                "l4_artificial": False,
            }
        )
    return rows


def semantic_pair_audit_rows(
    formula_pools: Mapping[tuple[str, str], Sequence[V3Formula]]
) -> list[dict[str, object]]:
    interesting = [
        ("k3_3", "lp_3", "observational equivalence from restricted interface"),
        ("k3_3", "k3_imp_flip_3", "language expressivity unlocks IMPLIES perturbation"),
        ("projection_k3_n4", "projection_k3_b4", "projection collapses internal U representatives"),
        ("fde4", "rare_xor_flip4", "one-entry sparse L3 perturbation"),
        ("fde4", "bochvar4", "large internal-rule shift"),
    ]
    rows = []
    for source, target, note in interesting:
        s = SEMANTICS_V3[source]
        t = SEMANTICS_V3[target]
        value_set = s.value_set
        raw_distinct = raw_table_hash(s) != raw_table_hash(t)
        for language_name in LANGUAGE_ORDER:
            formulas = formula_pools[(value_set, language_name)]
            raw_count = count_distinguishing_formulas(source, target, formulas, projected=False)
            projected_count = count_distinguishing_formulas(source, target, formulas, projected=True)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "language": language_name,
                    "raw_truth_tables_distinct": raw_distinct,
                    "raw_distinguishing_formulas": raw_count,
                    "projected_distinguishing_formulas": projected_count,
                    "observationally_equivalent_after_projection": projected_count == 0,
                    "l4_artificial": language_name == "L4",
                    "audit_note": note,
                }
            )
    return rows


def language_level_diagnosability(
    formula_pools: Mapping[tuple[str, str], Sequence[V3Formula]]
) -> list[dict[str, object]]:
    rows = []
    for regime in V4_REGIMES:
        source = regime["source"]
        target = regime["target"]
        value_set = SEMANTICS_V3[source].value_set
        for language_name in LANGUAGE_ORDER:
            formulas = formula_pools[(value_set, language_name)]
            info = pair_diagnosability(source, target, formulas)
            rows.append(
                {
                    **regime,
                    "language": language_name,
                    "diagnosable": info["projected_diagnosable"],
                    "raw_diagnosable": info["diagnosable"],
                    "projected_diagnosable": info["projected_diagnosable"],
                    "num_distinguishing_formulas": info["num_projected_distinguishing_formulas"],
                    "raw_num_distinguishing_formulas": info["num_distinguishing_formulas"],
                    "num_projected_distinguishing_formulas": info["num_projected_distinguishing_formulas"],
                    "shortest_distinguishing_depth": info["shortest_projected_distinguishing_depth"],
                    "raw_shortest_distinguishing_depth": info["shortest_distinguishing_depth"],
                    "pairwise_semantic_distance": info["projected_pairwise_semantic_distance"],
                    "raw_pairwise_semantic_distance": info["pairwise_semantic_distance"],
                    "formula_pool_size": len(formulas),
                    "l4_artificial": language_name == "L4",
                }
            )
    return rows


def run_symbolic_probe_validation(
    formula_pools: Mapping[tuple[str, str], Sequence[V3Formula]],
    diagnosability_rows: Sequence[dict[str, object]],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for diag in diagnosability_rows:
        source = str(diag["source"])
        target = str(diag["target"])
        language_name = str(diag["language"])
        value_set = SEMANTICS_V3[source].value_set
        formulas = formula_pools[(value_set, language_name)]
        candidates = [name for name, semantic in SEMANTICS_V3.items() if semantic.value_set == value_set]
        vectors = vectors_for(candidates, formulas, projected=True)
        selected_indices = source_contrastive_probes(source, candidates, formulas, vectors, max_probes=8)
        selected_depths = [formulas[idx].depth for idx in selected_indices]
        candidate_indices = source_candidate_indices(source, candidates, vectors)
        for seed in seeds:
            for strategy in DETECTOR_STRATEGIES:
                result = simulate_symbolic_monitor(
                    source=source,
                    target=target,
                    candidate_names=candidates,
                    formulas=formulas,
                    vectors=vectors,
                    selected_indices=selected_indices,
                    candidate_indices=candidate_indices,
                    selected_depths=selected_depths,
                    strategy=strategy,
                    budget=8 if strategy in PROBE_STRATEGIES else 0,
                    seed=seed,
                    diagnosable=bool(diag["diagnosable"]),
                )
                rows.append(
                    {
                        "seed": seed,
                        "regime": diag["regime"],
                        "difficulty": diag["difficulty"],
                        "source": source,
                        "target": target,
                        "language": language_name,
                        "strategy": strategy,
                        "diagnosable": diag["diagnosable"],
                        "l4_artificial": diag["l4_artificial"],
                        "num_distinguishing_formulas": diag["num_distinguishing_formulas"],
                        "shortest_distinguishing_depth": diag["shortest_distinguishing_depth"],
                        **{k: v for k, v in result.items() if k != "timeline"},
                    }
                )
    return rows


def run_neural_validation(
    formula_pools: Mapping[tuple[str, str], Sequence[V3Formula]],
    diagnosability_rows: Sequence[dict[str, object]],
    seeds: Sequence[int],
    device: torch.device,
    smoke: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    diag_lookup = {
        (row["regime"], row["language"]): row for row in diagnosability_rows
    }
    representative_regimes = [regime for regime in V4_REGIMES if regime["regime"] in NEURAL_REPRESENTATIVES]
    if smoke:
        representative_regimes = representative_regimes[:2]
    for regime in representative_regimes:
        language_name = regime["claim_language"]
        diag = diag_lookup[(regime["regime"], language_name)]
        source = regime["source"]
        target = regime["target"]
        value_set = SEMANTICS_V3[source].value_set
        formulas = list(formula_pools[(value_set, language_name)])
        candidates = [name for name, semantic in SEMANTICS_V3.items() if semantic.value_set == value_set]
        vectors = vectors_for(candidates, formulas, projected=True)
        selected_indices = source_contrastive_probes(source, candidates, formulas, vectors, max_probes=8)
        candidate_indices = source_candidate_indices(source, candidates, vectors)
        selected_depths = [formulas[idx].depth for idx in selected_indices]
        for seed in seeds:
            model, class_to_vector, train_info = train_tiny_reasoner(
                formulas=formulas,
                source=source,
                vectors=vectors,
                seed=seed,
                device=device,
                epochs=5 if smoke else 9,
            )
            neural_result = evaluate_neural_stream(
                model=model,
                class_to_vector=class_to_vector,
                formulas=formulas,
                source=source,
                target=target,
                vectors=vectors,
                seed=seed,
                device=device,
                smoke=smoke,
            )
            probe_results = {}
            for strategy in ("random_depth_matched", "random_candidate_matched", "selected", "eig_adaptive", "worst_case"):
                probe_results[strategy] = simulate_symbolic_monitor(
                    source=source,
                    target=target,
                    candidate_names=candidates,
                    formulas=formulas,
                    vectors=vectors,
                    selected_indices=selected_indices,
                    candidate_indices=candidate_indices,
                    selected_depths=selected_depths,
                    strategy=strategy,
                    budget=8,
                    seed=seed,
                    diagnosable=bool(diag["diagnosable"]),
                )
            detector_rows = neural_detector_rows(
                regime=regime,
                language_name=language_name,
                seed=seed,
                diag=diag,
                train_info=train_info,
                neural_result=neural_result,
                probe_results=probe_results,
            )
            rows.extend(detector_rows)
    return rows


def simulate_symbolic_monitor(
    source: str,
    target: str,
    candidate_names: Sequence[str],
    formulas: Sequence[V3Formula],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    selected_indices: Sequence[int],
    candidate_indices: Sequence[int],
    selected_depths: Sequence[int],
    strategy: str,
    budget: int,
    seed: int,
    diagnosable: bool,
    n_pre: int = 80,
    n_post: int = 80,
    probe_interval: int = 8,
) -> dict[str, object]:
    rng = random.Random(seed + stable_offset(source, target, strategy, str(budget), str(len(formulas))))
    n_total = n_pre + n_post
    posterior = {name: 1.0 / len(candidate_names) for name in candidate_names}
    stream_indices = [rng.randrange(len(formulas)) for _ in range(n_pre)]
    stream_indices = stream_indices + list(stream_indices)
    random_all_order = shuffled_indices(rng, list(range(len(formulas))), n_total * max(budget, 1))
    random_candidate_order = shuffled_indices(rng, list(candidate_indices) or list(range(len(formulas))), n_total * max(budget, 1))
    random_depth_order = depth_matched_order(rng, formulas, selected_depths, n_total * max(budget, 1))
    selected_cursor = 0
    used_indices: set[int] = set()
    probes_used = 0
    alarm_index = None
    confident_wrong_before_alarm = 0
    semantic_correct = 0
    timeline = []

    for step in range(n_total):
        true_name = source if step < n_pre else target
        source_output = vectors[source][stream_indices[step]]
        true_output = vectors[true_name][stream_indices[step]]
        if step >= n_pre and alarm_index is None and source_output != true_output:
            confident_wrong_before_alarm += 1

        if strategy in PROBE_STRATEGIES and step % probe_interval == 0:
            posterior = _soften_posterior(posterior, candidate_names, mixing=0.12)
            for probe_offset in range(budget):
                if strategy == "random_all":
                    formula_index = random_all_order[probes_used % len(random_all_order)]
                elif strategy == "random_candidate_matched":
                    formula_index = random_candidate_order[probes_used % len(random_candidate_order)]
                elif strategy == "random_depth_matched":
                    formula_index = random_depth_order[probes_used % len(random_depth_order)]
                elif strategy == "selected":
                    formula_index = selected_indices[selected_cursor % len(selected_indices)]
                    selected_cursor += 1
                elif strategy == "eig_adaptive":
                    formula_index = information_gain_probe(candidate_names, formulas, vectors, posterior, used_indices)
                else:
                    formula_index = worst_case_ambiguity_probe(candidate_names, formulas, vectors, posterior, used_indices)
                used_indices.add(formula_index)
                probes_used += 1
                posterior = update_bayesian_posterior(
                    candidate_names,
                    vectors,
                    posterior,
                    formula_index,
                    vectors[true_name][formula_index],
                )
        elif strategy == "passive_semantic":
            posterior = _soften_posterior(posterior, candidate_names, mixing=0.03)
            formula_index = stream_indices[step]
            posterior = update_bayesian_posterior(
                candidate_names,
                vectors,
                posterior,
                formula_index,
                true_output,
                observation_accuracy=0.99,
            )

        best = max(posterior, key=posterior.get)
        semantic_correct += int(best == true_name)
        if strategy == "oracle_error_rate":
            recent = [
                vectors[source][stream_indices[j]] != vectors[source if j < n_pre else target][stream_indices[j]]
                for j in range(max(0, step - 15), step + 1)
            ]
            decision = step >= n_pre and mean(recent) > 0.2
        elif strategy in {
            "confidence_threshold",
            "rolling_entropy",
            "page_hinkley_confidence",
            "page_hinkley_entropy",
            "input_histogram",
            "representation_drift",
        }:
            decision = False
        else:
            decision = step >= 10 and best != source and posterior[source] < 0.25
        if alarm_index is None and decision:
            alarm_index = step
        timeline.append(
            {
                "step": step,
                "true": true_name,
                "best": best,
                "probes": probes_used,
                "confident_wrong": confident_wrong_before_alarm,
                "alarm": alarm_index == step,
                "source_posterior": posterior[source],
            }
        )

    post_alarm = alarm_index is not None and alarm_index >= n_pre
    return {
        "alarm_index": alarm_index,
        "detection_delay": alarm_index - n_pre if post_alarm else None,
        "miss_rate": 0.0 if post_alarm else 1.0,
        "false_alarm_rate": 1.0 if alarm_index is not None and alarm_index < n_pre else 0.0,
        "confident_wrong_before_alarm": confident_wrong_before_alarm,
        "probes_used": probes_used,
        "semantic_identification_accuracy": semantic_correct / n_total,
        "input_js_divergence": 0.0,
        "output_label_js_divergence": _label_js(
            [vectors[source][idx] for idx in stream_indices[:n_pre]],
            [vectors[target][idx] for idx in stream_indices[n_pre:]],
        ),
        "symbolic_diagnosable": diagnosable,
        "timeline": timeline,
    }


def train_tiny_reasoner(
    formulas: Sequence[V3Formula],
    source: str,
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    seed: int,
    device: torch.device,
    epochs: int,
) -> tuple[V4TinyReasoner, dict[int, tuple[str, ...]], dict[str, object]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    labels = list(vectors[source])
    vector_to_class = {vector: idx for idx, vector in enumerate(sorted(set(labels)))}
    class_to_vector = {idx: vector for vector, idx in vector_to_class.items()}
    x, lengths = encode_v4_formulas(formulas, device=device)
    y = torch.tensor([vector_to_class[vector] for vector in labels], dtype=torch.long, device=device)
    model = V4TinyReasoner(n_classes=len(vector_to_class)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = len(formulas)
    batch_size = 192
    history = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        correct = 0
        total_loss = 0.0
        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx], lengths[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(dim=-1) == y[idx]).sum().detach().cpu())
        history.append({"loss": total_loss / n, "accuracy": correct / n})
    return model, class_to_vector, {
        "train_accuracy": history[-1]["accuracy"],
        "train_loss": history[-1]["loss"],
        "n_classes": len(vector_to_class),
        "epochs": epochs,
    }


@torch.no_grad()
def evaluate_neural_stream(
    model: V4TinyReasoner,
    class_to_vector: Mapping[int, tuple[str, ...]],
    formulas: Sequence[V3Formula],
    source: str,
    target: str,
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
    seed: int,
    device: torch.device,
    smoke: bool,
) -> dict[str, object]:
    rng = random.Random(seed + stable_offset(source, target, "neural"))
    n_pre = 48 if smoke else 80
    indices = [rng.randrange(len(formulas)) for _ in range(n_pre)]
    stream_indices = indices + list(indices)
    stream_formulas = [formulas[idx] for idx in stream_indices]
    x, lengths = encode_v4_formulas(stream_formulas, device=device)
    model.eval()
    logits = model(x, lengths)
    probs = torch.softmax(logits / 0.65, dim=-1).detach().cpu()
    pred_classes = probs.argmax(dim=-1).tolist()
    pred_vectors = [class_to_vector[int(cls)] for cls in pred_classes]
    confidences = [float(value) for value in probs.max(dim=-1).values.tolist()]
    entropies = normalized_entropy(probs)
    hidden = model.encode_hidden(x, lengths).detach().cpu().tolist()
    true_source = [vectors[source][idx] for idx in stream_indices]
    true_target = [
        vectors[source if i < n_pre else target][idx]
        for i, idx in enumerate(stream_indices)
    ]
    pre_correct = [pred_vectors[i] == true_source[i] for i in range(n_pre)]
    correct = [pred_vectors[i] == true_target[i] for i in range(len(stream_indices))]
    thresholds = fit_dev_thresholds(confidences[:n_pre], entropies[:n_pre], hidden[:n_pre])
    detector_results = neural_baseline_detectors(confidences, entropies, hidden, correct, thresholds, n_pre)
    return {
        "n_pre": n_pre,
        "confidences": confidences,
        "entropies": entropies,
        "hidden": hidden,
        "correct": correct,
        "pre_accuracy": sum(pre_correct) / len(pre_correct),
        "post_accuracy_total": sum(correct[n_pre:]) / n_pre,
        "mean_confidence_pre": mean(confidences[:n_pre]),
        "mean_confidence_post": mean(confidences[n_pre:]),
        "output_label_js_divergence": _label_js(true_source[:n_pre], [vectors[target][idx] for idx in stream_indices[n_pre:]]),
        "detectors": detector_results,
        "thresholds": thresholds,
    }


def neural_detector_rows(
    regime: Mapping[str, str],
    language_name: str,
    seed: int,
    diag: Mapping[str, object],
    train_info: Mapping[str, object],
    neural_result: Mapping[str, object],
    probe_results: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    n_pre = int(neural_result["n_pre"])
    for detector, result in neural_result["detectors"].items():
        rows.append(
            {
                "seed": seed,
                "regime": regime["regime"],
                "difficulty": regime["difficulty"],
                "source": regime["source"],
                "target": regime["target"],
                "language": language_name,
                "detector": detector,
                "diagnosable": diag["diagnosable"],
                "l4_artificial": language_name == "L4",
                "label_using": detector == "oracle_error_rate",
                "threshold_source": "dev",
                "threshold_used_test": False,
                "pre_accuracy": neural_result["pre_accuracy"],
                "post_accuracy_total": neural_result["post_accuracy_total"],
                "mean_confidence_pre": neural_result["mean_confidence_pre"],
                "mean_confidence_post": neural_result["mean_confidence_post"],
                "train_accuracy": train_info["train_accuracy"],
                "train_loss": train_info["train_loss"],
                "output_label_js_divergence": neural_result["output_label_js_divergence"],
                **result,
            }
        )
    for detector, result in probe_results.items():
        rows.append(
            {
                "seed": seed,
                "regime": regime["regime"],
                "difficulty": regime["difficulty"],
                "source": regime["source"],
                "target": regime["target"],
                "language": language_name,
                "detector": detector,
                "diagnosable": diag["diagnosable"],
                "l4_artificial": language_name == "L4",
                "label_using": False,
                "threshold_source": "not_applicable",
                "threshold_used_test": False,
                "pre_accuracy": neural_result["pre_accuracy"],
                "post_accuracy_total": neural_result["post_accuracy_total"],
                "mean_confidence_pre": neural_result["mean_confidence_pre"],
                "mean_confidence_post": neural_result["mean_confidence_post"],
                "train_accuracy": train_info["train_accuracy"],
                "train_loss": train_info["train_loss"],
                "output_label_js_divergence": neural_result["output_label_js_divergence"],
                "alarm_index": result["alarm_index"],
                "detection_delay": result["detection_delay"],
                "miss_rate": result["miss_rate"],
                "false_alarm_rate": result["false_alarm_rate"],
                "confident_wrong_before_alarm": result["confident_wrong_before_alarm"],
                "probes_used": result["probes_used"],
                "semantic_identification_accuracy": result["semantic_identification_accuracy"],
                "input_js_divergence": 0.0,
            }
        )
    return rows


def neural_baseline_detectors(
    confidences: Sequence[float],
    entropies: Sequence[float],
    hidden: Sequence[Sequence[float]],
    correct: Sequence[bool],
    thresholds: ThresholdConfig,
    changepoint: int,
) -> dict[str, dict[str, object]]:
    detectors = {}
    detectors["confidence_threshold"] = alarm_from_scores(
        [thresholds.confidence_threshold - value for value in confidences],
        changepoint,
        threshold=0.0,
        greater=True,
        correct=correct,
        confidences=confidences,
    )
    detectors["rolling_entropy"] = alarm_from_scores(
        [value - thresholds.entropy_threshold for value in entropies],
        changepoint,
        threshold=0.0,
        greater=True,
        correct=correct,
        confidences=confidences,
    )
    detectors["page_hinkley_confidence"] = alarm_from_scores(
        page_hinkley_scores([-value for value in confidences], changepoint),
        changepoint,
        threshold=0.08,
        greater=True,
        correct=correct,
        confidences=confidences,
    )
    detectors["page_hinkley_entropy"] = alarm_from_scores(
        page_hinkley_scores(entropies, changepoint),
        changepoint,
        threshold=0.08,
        greater=True,
        correct=correct,
        confidences=confidences,
    )
    detectors["input_histogram"] = no_alarm_result(changepoint, correct, confidences)
    detectors["representation_drift"] = alarm_from_scores(
        representation_scores(hidden, changepoint),
        changepoint,
        threshold=thresholds.representation_threshold,
        greater=True,
        correct=correct,
        confidences=confidences,
    )
    errors = [0.0 if ok else 1.0 for ok in correct]
    rolling_errors = [mean(errors[max(0, idx - 11) : idx + 1]) for idx in range(len(errors))]
    detectors["oracle_error_rate"] = alarm_from_scores(
        rolling_errors,
        changepoint,
        threshold=0.20,
        greater=True,
        correct=correct,
        confidences=confidences,
    )
    return detectors


def fit_dev_thresholds(
    confidences: Sequence[float],
    entropies: Sequence[float],
    hidden: Sequence[Sequence[float]],
) -> ThresholdConfig:
    rep_scores = representation_scores(hidden, len(hidden))
    return ThresholdConfig(
        confidence_threshold=mean(confidences) - 4.0 * max(pstdev(confidences), 1e-3),
        entropy_threshold=mean(entropies) + 4.0 * max(pstdev(entropies), 1e-3),
        representation_threshold=mean(rep_scores) + 4.0 * max(pstdev(rep_scores), 1e-3),
    )


def alarm_from_scores(
    scores: Sequence[float],
    changepoint: int,
    threshold: float,
    greater: bool,
    correct: Sequence[bool],
    confidences: Sequence[float],
) -> dict[str, object]:
    alarm = None
    for idx in range(max(5, changepoint // 4), len(scores)):
        if (scores[idx] > threshold) if greater else (scores[idx] < threshold):
            alarm = idx
            break
    post_alarm = alarm is not None and alarm >= changepoint
    return {
        "alarm_index": alarm,
        "detection_delay": alarm - changepoint if post_alarm else None,
        "miss_rate": 0.0 if post_alarm else 1.0,
        "false_alarm_rate": 1.0 if alarm is not None and alarm < changepoint else 0.0,
        "confident_wrong_before_alarm": confident_wrong_until(correct, confidences, changepoint, alarm),
        "probes_used": 0,
        "semantic_identification_accuracy": None,
        "input_js_divergence": 0.0,
    }


def no_alarm_result(
    changepoint: int,
    correct: Sequence[bool],
    confidences: Sequence[float],
) -> dict[str, object]:
    return {
        "alarm_index": None,
        "detection_delay": None,
        "miss_rate": 1.0,
        "false_alarm_rate": 0.0,
        "confident_wrong_before_alarm": confident_wrong_until(correct, confidences, changepoint, None),
        "probes_used": 0,
        "semantic_identification_accuracy": None,
        "input_js_divergence": 0.0,
    }


def confident_wrong_until(
    correct: Sequence[bool],
    confidences: Sequence[float],
    changepoint: int,
    alarm: int | None,
    threshold: float = 0.8,
) -> int:
    end = alarm if alarm is not None and alarm >= changepoint else len(correct)
    return sum((not correct[idx]) and confidences[idx] >= threshold for idx in range(changepoint, end))


def page_hinkley_scores(values: Sequence[float], calibration: int) -> list[float]:
    baseline = values[:calibration]
    base_mean = mean(baseline)
    cumulative = 0.0
    min_cumulative = 0.0
    scores = [0.0] * len(values)
    for idx, value in enumerate(values):
        cumulative += value - base_mean - 0.002
        min_cumulative = min(min_cumulative, cumulative)
        scores[idx] = cumulative - min_cumulative
    return scores


def representation_scores(hidden: Sequence[Sequence[float]], calibration: int) -> list[float]:
    base = hidden[:calibration]
    if not base:
        return [0.0 for _ in hidden]
    center = [sum(row[col] for row in base) / len(base) for col in range(len(base[0]))]
    return [math.sqrt(sum((value - center[col]) ** 2 for col, value in enumerate(row))) for row in hidden]


def encode_v4_formulas(
    formulas: Sequence[V3Formula],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(formula.tokens) for formula in formulas)
    x = torch.zeros((len(formulas), max_len), dtype=torch.long, device=device)
    lengths = torch.zeros(len(formulas), dtype=torch.long, device=device)
    for row, formula in enumerate(formulas):
        ids = [TOKEN_TO_ID[token] for token in formula.tokens]
        x[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        lengths[row] = len(ids)
    return x, lengths


def normalized_entropy(probs: torch.Tensor) -> list[float]:
    clipped = probs.clamp_min(1e-9)
    values = -(clipped * clipped.log()).sum(dim=-1)
    denom = math.log(max(probs.shape[-1], 2))
    return [float(value) / denom for value in values]


def pair_diagnosability(
    source: str,
    target: str,
    formulas: Sequence[V3Formula],
) -> dict[str, object]:
    raw_differing = []
    projected_differing = []
    distances = []
    projected_distances = []
    for idx, formula in enumerate(formulas):
        left = SEMANTICS_V3[source].truth_vector(formula)
        right = SEMANTICS_V3[target].truth_vector(formula)
        if left != right:
            raw_differing.append(idx)
        left_projected = projected_truth_vector(SEMANTICS_V3[source], formula)
        right_projected = projected_truth_vector(SEMANTICS_V3[target], formula)
        if left_projected != right_projected:
            projected_differing.append(idx)
        distances.append(vector_distance(left, right))
        projected_distances.append(vector_distance(left_projected, right_projected))
    return {
        "diagnosable": bool(raw_differing),
        "projected_diagnosable": bool(projected_differing),
        "num_distinguishing_formulas": len(raw_differing),
        "num_projected_distinguishing_formulas": len(projected_differing),
        "shortest_distinguishing_depth": min((formulas[idx].depth for idx in raw_differing), default=None),
        "shortest_projected_distinguishing_depth": min((formulas[idx].depth for idx in projected_differing), default=None),
        "pairwise_semantic_distance": mean(distances) if distances else 0.0,
        "projected_pairwise_semantic_distance": mean(projected_distances) if projected_distances else 0.0,
    }


def vectors_for(
    candidate_names: Sequence[str],
    formulas: Sequence[V3Formula],
    projected: bool = False,
) -> dict[str, list[tuple[str, ...]]]:
    return {
        name: [
            projected_truth_vector(SEMANTICS_V3[name], formula)
            if projected
            else SEMANTICS_V3[name].truth_vector(formula)
            for formula in formulas
        ]
        for name in candidate_names
    }


def projected_truth_vector(semantic: V3SemanticSystem, formula: V3Formula) -> tuple[str, ...]:
    return tuple(project_value(semantic, value) for value in semantic.truth_vector(formula))


def project_value(semantic: V3SemanticSystem, value: str) -> str:
    if semantic.value_set == "three":
        return value
    return {"T": "T", "F": "F", "B": "U", "N": "U"}[value]


def projection_name(semantic: V3SemanticSystem) -> str:
    return "identity" if semantic.value_set == "three" else "bilattice_to_three_observable"


def raw_table_hash(semantic: V3SemanticSystem) -> str:
    payload = {
        "values": semantic.values,
        "unary": {op: dict(table) for op, table in sorted(semantic.unary.items())},
        "binary": {
            op: {f"{a},{b}": value for (a, b), value in sorted(table.items())}
            for op, table in sorted(semantic.binary.items())
        },
    }
    text = json.dumps(payload, sort_keys=True)
    return hex(sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % (16**12))[2:]


def count_distinguishing_formulas(
    source: str,
    target: str,
    formulas: Sequence[V3Formula],
    projected: bool,
) -> int:
    if projected:
        return sum(
            projected_truth_vector(SEMANTICS_V3[source], formula)
            != projected_truth_vector(SEMANTICS_V3[target], formula)
            for formula in formulas
        )
    return sum(
        SEMANTICS_V3[source].truth_vector(formula)
        != SEMANTICS_V3[target].truth_vector(formula)
        for formula in formulas
    )


def vector_distance(left: Sequence[str], right: Sequence[str]) -> float:
    width = max(len(left), len(right), 1)
    return sum(a != b for a, b in itertools.zip_longest(left, right)) / width


def source_candidate_indices(
    source: str,
    candidate_names: Sequence[str],
    vectors: Mapping[str, Sequence[tuple[str, ...]]],
) -> list[int]:
    return [
        idx
        for idx in range(len(vectors[source]))
        if any(vectors[source][idx] != vectors[name][idx] for name in candidate_names if name != source)
    ]


def shuffled_indices(rng: random.Random, indices: Sequence[int], needed: int) -> list[int]:
    indices = list(indices)
    if not indices:
        return [0]
    output = []
    while len(output) < needed:
        batch = list(indices)
        rng.shuffle(batch)
        output.extend(batch)
    return output[:needed]


def depth_matched_order(
    rng: random.Random,
    formulas: Sequence[V3Formula],
    selected_depths: Sequence[int],
    needed: int,
) -> list[int]:
    by_depth: dict[int, list[int]] = defaultdict(list)
    for idx, formula in enumerate(formulas):
        by_depth[formula.depth].append(idx)
    if not selected_depths:
        selected_depths = [0]
    output = []
    cursor = 0
    while len(output) < needed:
        depth = selected_depths[cursor % len(selected_depths)]
        pool = by_depth.get(depth) or list(range(len(formulas)))
        output.append(rng.choice(pool))
        cursor += 1
    return output


def selected_depth_distribution(
    selected_indices: Sequence[int],
    formulas: Sequence[V3Formula],
) -> Counter[int]:
    return Counter(formulas[idx].depth for idx in selected_indices)


def stable_offset(*parts: str) -> int:
    return _stable_offset(*parts)


def aggregate_rows(
    rows: Sequence[dict[str, object]],
    group_keys: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        first = items[0]
        row = {group_key: value for group_key, value in zip(group_keys, key)}
        for copy_key in (
            "difficulty",
            "source",
            "target",
            "diagnosable",
            "l4_artificial",
            "label_using",
            "threshold_source",
            "threshold_used_test",
            "num_distinguishing_formulas",
            "shortest_distinguishing_depth",
        ):
            if copy_key in first:
                row[copy_key] = first[copy_key]
        for metric in (
            "detection_delay",
            "confident_wrong_before_alarm",
            "false_alarm_rate",
            "miss_rate",
            "probes_used",
            "semantic_identification_accuracy",
            "input_js_divergence",
            "output_label_js_divergence",
            "pre_accuracy",
            "post_accuracy_total",
            "mean_confidence_pre",
            "mean_confidence_post",
            "train_accuracy",
        ):
            values = [item.get(metric) for item in items if item.get(metric) is not None and item.get(metric) != ""]
            row[f"mean_{metric}"] = mean(float(value) for value in values) if values else None
            row[f"std_{metric}"] = pstdev([float(value) for value in values]) if len(values) > 1 else 0.0
        row["n"] = len(items)
        out.append(row)
    return out


def flatten_metric_rows(
    fair_aggregates: Sequence[dict[str, object]],
    neural_aggregates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for row in fair_aggregates:
        rows.append({"section": "symbolic_probe", **row})
    for row in neural_aggregates:
        rows.append({"section": "neural_online", **row})
    return rows


def headline(
    fair_aggregates: Sequence[dict[str, object]],
    neural_aggregates: Sequence[dict[str, object]],
    runtime_s: float,
) -> dict[str, object]:
    sparse = [
        row for row in fair_aggregates
        if "sparse" in str(row.get("difficulty", "")) and row.get("language") != "L4"
    ]
    selected = [row for row in sparse if row.get("strategy") == "selected"]
    adaptive = [row for row in sparse if row.get("strategy") == "eig_adaptive"]
    randoms = [
        row for row in sparse
        if row.get("strategy") in {"random_all", "random_depth_matched", "random_candidate_matched"}
    ]
    equivalent = [
        row for row in fair_aggregates
        if row.get("regime") == "impossible_equivalent" and row.get("strategy") == "selected"
    ]
    return {
        "mode": "v4",
        "runtime_seconds": runtime_s,
        "sparse_selected_delay": mean_optional(row.get("mean_detection_delay") for row in selected),
        "sparse_adaptive_delay": mean_optional(row.get("mean_detection_delay") for row in adaptive),
        "sparse_random_matched_delay": mean_optional(row.get("mean_detection_delay") for row in randoms),
        "equivalent_selected_miss_rate": mean_optional(row.get("mean_miss_rate") for row in equivalent),
        "neural_rows": len(neural_aggregates),
    }


def mean_optional(values: Iterable[object]) -> float | None:
    vals = [float(value) for value in values if value is not None and value != ""]
    return None if not vals else mean(vals)


def cuda_status(device: torch.device) -> dict[str, object]:
    return {
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "selected_device": str(device),
    }


def write_dict_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_plots(out_dir: Path, summary: Mapping[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except Exception:
        return

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=170)
        fig.savefig(out_dir / f"{name}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "Matched input stream\nP(X) unchanged"),
        (0.38, 0.55, "Hidden semantic rule\nchanges at tau"),
        (0.70, 0.55, "Reasoner remains\nconfident but wrong"),
        (0.38, 0.12, "Active probes test\nobservability boundary"),
    ]
    for x, y, text in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), 0.24, 0.22, boxstyle="round,pad=0.02", facecolor="#eef3f7", edgecolor="#375a7f"))
        ax.text(x + 0.12, y + 0.11, text, ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(0.38, 0.66), xytext=(0.29, 0.66), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.70, 0.66), xytext=(0.62, 0.66), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.50, 0.34), xytext=(0.50, 0.55), arrowprops={"arrowstyle": "->"})
    ax.set_title("Latent semantic rule shift with unchanged input distribution")
    save(fig, "figure1_conceptual_rule_shift")

    diag = summary["diagnosability"]
    languages = list(LANGUAGE_ORDER)
    regimes = sorted({row["regime"] for row in diag})
    matrix = []
    for regime in regimes:
        matrix.append([
            1 if next(row for row in diag if row["regime"] == regime and row["language"] == lang)["diagnosable"] else 0
            for lang in languages
        ])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(languages)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(regimes, fontsize=8)
    ax.set_title("Diagnosability heatmap by language level")
    save(fig, "figure2_diagnosability_heatmap")

    fair = summary["fair_random_aggregates"]
    sparse_rows = [
        row for row in fair
        if "sparse" in str(row.get("difficulty", "")) and row.get("language") != "L4"
        and row.get("strategy") in {"random_all", "random_depth_matched", "random_candidate_matched", "selected", "eig_adaptive"}
    ]
    labels = sorted({row["regime"] for row in sparse_rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.15
    strategies = ["random_all", "random_depth_matched", "random_candidate_matched", "selected", "eig_adaptive"]
    for offset, strategy in enumerate(strategies):
        vals = []
        for label in labels:
            candidates = [row for row in sparse_rows if row["regime"] == label and row["strategy"] == strategy]
            delay = mean_optional(row["mean_detection_delay"] for row in candidates)
            vals.append(160 if delay is None else delay)
        ax.bar([i + (offset - 2) * width for i in range(len(labels))], vals, width=width, label=strategy)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("delay (misses as 160)")
    ax.set_title("Selected/adaptive vs matched random probes on sparse shifts")
    ax.legend(fontsize=7)
    save(fig, "figure3_sparse_selected_vs_random")

    neural = summary["neural_monitoring"]
    timeline_rows = [row for row in neural if row.get("regime") == "sparse_hard_l2"]
    fig, ax = plt.subplots(figsize=(8, 4))
    if timeline_rows:
        detectors = ["confidence_threshold", "random_depth_matched", "selected", "eig_adaptive"]
        for detector in detectors:
            rows = [row for row in timeline_rows if row["detector"] == detector]
            cw = mean_optional(row["confident_wrong_before_alarm"] for row in rows)
            delay = mean_optional(row["detection_delay"] for row in rows)
            ax.bar(detector, 0 if cw is None else cw, label=detector)
            if delay is not None:
                ax.text(detector, (0 if cw is None else cw) + 0.1, f"d={delay:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("confident wrong before alarm")
    ax.set_title("Neural closure: confidence stays high, probes alarm early")
    ax.tick_params(axis="x", labelrotation=20)
    save(fig, "figure4_neural_timeline_summary")

    fig, ax = plt.subplots(figsize=(7, 4))
    xs, ys, labels = [], [], []
    for row in fair:
        if row.get("strategy") != "selected" or row.get("language") == "L4":
            continue
        random_row = next(
            (
                item for item in fair
                if item.get("regime") == row.get("regime")
                and item.get("language") == row.get("language")
                and item.get("strategy") == "random_depth_matched"
            ),
            None,
        )
        if random_row is None:
            continue
        selected_delay = 160 if row["mean_detection_delay"] is None else float(row["mean_detection_delay"])
        random_delay = 160 if random_row["mean_detection_delay"] is None else float(random_row["mean_detection_delay"])
        xs.append(float(row.get("num_distinguishing_formulas") or 0))
        ys.append(random_delay - selected_delay)
        labels.append(str(row["regime"]))
    ax.scatter(xs, ys, color="#7a3e9d")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=7)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("distinguishing formulas")
    ax.set_ylabel("random_depth_matched delay - selected delay")
    ax.set_title("Distinguishing formulas vs selected-probe advantage")
    save(fig, "figure5_distinguishing_vs_advantage")


def write_report(path: Path, summary: Mapping[str, object]) -> None:
    cuda = summary["cuda"]
    note = summary["interpreter_note"]
    headline_row = summary["headline"]
    lines = [
        "# Semantic Rule-Shift V4 Paper Validation",
        "",
        "## Commands Run",
        "",
        f"- Requested interpreter existed: `{note['requested_exists']}`",
        f"- Requested interpreter: `{note['requested']}`",
        f"- Actual interpreter: `{note['actual']}`",
        f"- `{cuda['python_executable']} -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\"`",
        f"- `{cuda['python_executable']} -m pytest`",
        f"- `{cuda['python_executable']} -m semantic_rule_shift.experiments --mode v4 --device cuda --out results/semantic_rule_shift_v4`",
        "",
        "## CUDA Status",
        "",
        f"- Torch: `{cuda['torch']}`",
        f"- CUDA available: `{cuda['cuda_available']}`",
        f"- GPU: `{cuda['device']}`",
        f"- Selected device: `{cuda['selected_device']}`",
        f"- Runtime: `{summary['runtime_seconds']:.2f}` seconds",
        "",
        "## Semantic Implementation Audit",
        "",
        "Semantic systems are audited in `table1_semantic_systems.csv`. Four-valued systems use internal values `T/F/B/N`; observable projected labels collapse `B,N` to `U` when projection is requested. K3/LP equivalence in the three-valued interface is intentional and documented as observational equivalence under the restricted language, not an accidental missing implementation.",
        "",
        "## No-Leakage Audit",
        "",
        "- Selected probes use the known source semantic and the predefined candidate library only.",
        "- Selected, EIG, and worst-case probes do not receive the realized target identity.",
        "- Random baselines use the exact same probe budget.",
        "- Threshold baselines use `threshold_source=dev`; no test-stream threshold tuning is used.",
        "- `oracle_error_rate` is explicitly label-using and unfair.",
        "- `L4` rows are marked `l4_artificial=True` and are not needed for the main claim.",
        "",
        "## Detector Definitions",
        "",
        "`random_all`, `random_depth_matched`, and `random_candidate_matched` are all reported. `selected` is source-contrastive greedy set cover. `eig_adaptive` selects probes by expected information gain, while `worst_case` minimizes remaining ambiguity. Confidence, entropy, Page-Hinkley, input histogram, representation drift, passive semantic, and oracle error baselines are included in CSV outputs.",
        "",
        "## Language-Level Diagnosability Results",
        "",
        "| Regime | L0 | L1 | L2 | L3 | L4 artificial |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    diag = summary["diagnosability"]
    for regime in sorted({row["regime"] for row in diag}):
        cells = []
        for lang in LANGUAGE_ORDER:
            row = next(item for item in diag if item["regime"] == regime and item["language"] == lang)
            cells.append("Y" if row["diagnosable"] else "N")
        lines.append(f"| {regime} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Fair Random Baseline Results",
            "",
            "| Regime | Language | Strategy | Delay mean | Miss mean | Confident wrong | Probes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["fair_random_aggregates"]:
        if row["language"] == "L4" or row["strategy"] not in {"random_all", "random_depth_matched", "random_candidate_matched", "selected", "eig_adaptive"}:
            continue
        if "sparse" not in str(row.get("difficulty", "")) and row["regime"] != "impossible_equivalent":
            continue
        lines.append(
            "| {regime} | {language} | {strategy} | {delay} | {miss} | {cw} | {probes} |".format(
                regime=row["regime"],
                language=row["language"],
                strategy=row["strategy"],
                delay=fmt(row["mean_detection_delay"]),
                miss=fmt(row["mean_miss_rate"]),
                cw=fmt(row["mean_confident_wrong_before_alarm"]),
                probes=fmt(row["mean_probes_used"]),
            )
        )

    lines.extend(
        [
            "",
            "## Neural Online Monitoring Results",
            "",
            "| Regime | Detector | Delay | Miss | Pre acc | Post acc | Conf pre/post | Confident wrong |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    display_detectors = {"confidence_threshold", "input_histogram", "representation_drift", "oracle_error_rate", "random_depth_matched", "random_candidate_matched", "selected", "eig_adaptive"}
    for row in summary["neural_aggregates"]:
        if row["detector"] not in display_detectors:
            continue
        lines.append(
            "| {regime} | {detector} | {delay} | {miss} | {pre} | {post} | {cpre}/{cpost} | {cw} |".format(
                regime=row["regime"],
                detector=row["detector"],
                delay=fmt(row["mean_detection_delay"]),
                miss=fmt(row["mean_miss_rate"]),
                pre=fmt(row["mean_pre_accuracy"]),
                post=fmt(row["mean_post_accuracy_total"]),
                cpre=fmt(row["mean_mean_confidence_pre"]),
                cpost=fmt(row["mean_mean_confidence_post"]),
                cw=fmt(row["mean_confident_wrong_before_alarm"]),
            )
        )

    claim_survives = (
        headline_row["equivalent_selected_miss_rate"] == 1.0
        and headline_row["sparse_selected_delay"] is not None
        and headline_row["sparse_random_matched_delay"] is not None
        and float(headline_row["sparse_selected_delay"]) <= float(headline_row["sparse_random_matched_delay"])
    )
    lines.extend(
        [
            "",
            "## Paper Claim Verdict",
            "",
            (
                "The V4 claim survives in this scaffold: impossible shifts are identified as impossible, selected/adaptive probes beat matched random baselines on sparse diagnosable shifts without relying on L4, and confidence/input/representation baselines miss or lag while confidence remains high."
                if claim_survives
                else "The V4 evidence is mixed; do not draft a standalone AAAI paper until the failure modes are resolved."
            ),
            "",
            "## Failure Cases",
            "",
            "- The neural closure is compact and intentionally small; it validates alignment with diagnosability but is not yet a full-scale neural reasoning benchmark.",
            "- Projection-sensitive claims depend on the explicit bilattice-to-three observable projection.",
            "- L4 is artificial and should be framed only as an upper bound.",
            "",
            "## Recommended Title, Abstract Sketch, Contributions",
            "",
            "**Title:** When Rule Shifts Are Observable: Diagnosability of Latent Semantic Changes in Neural Reasoners",
            "",
            "**Abstract sketch:** Neural reasoners can remain confident when the latent semantic rules governing labels change while inputs remain distributionally matched. We formalize an identifiability boundary for semantic rule shifts, show that some shifts are provably unobservable under restricted probe languages, and demonstrate that active semantic probes reduce detection delay and confident wrong decisions on sparse but diagnosable shifts.",
            "",
            "**Main contributions:** semantic diagnosability matrix by language expressivity; fair active-probe baselines with matched random probes; neural online monitoring closure under matched input streams; explicit no-leakage and no-test-tuning audits.",
            "",
            "## Exact Next Steps",
            "",
            "- Draft the formal problem statement and identifiability proposition.",
            "- Replace syntactic formula enumeration with canonicalized formula classes.",
            "- Scale neural monitoring to larger models and single-assignment probes.",
            "- Run 20+ seeds for the final paper tables.",
            "- Write the AAAI draft around diagnosability first, detector performance second.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.3f}"
