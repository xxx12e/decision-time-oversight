from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .config import load_config, save_config_snapshot
from .defenses import (
    COMBINED_V1,
    CONFIDENCE_WEIGHTED_AGGREGATION,
    DEFENSE_LABELS,
    EVIDENCE_ANCHORING_ONLY,
    HUB_DOWNWEIGHTING,
    MAJORITY_VOTE,
    MESSAGE_SUPPORT_CHECKING,
    NO_DEFENSE,
    RANDOM_SOURCE_DROPOUT,
)
from .graphs import build_graph
from .report import _markdown_table
from .simulation import (
    Condition,
    _select_committed_agents,
    aggregate_results,
    run_conditions,
    run_sweep,
    stable_seed,
)
from .v3 import _compute_paper_numbers, _load_v2, _save_fig, write_tex_table


PLACEMENT_LABELS = {
    "high_degree": "High-degree / hub",
    "clustered": "Clustered",
    "random": "Random",
    "low_degree": "Low-degree",
}


def _label_defense(defense: str) -> str:
    return DEFENSE_LABELS.get(defense, defense.replace("_", " ").title())


def _label_placement(placement: str) -> str:
    return PLACEMENT_LABELS.get(placement, placement.replace("_", " ").title())


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


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    return float(x.astype(float).corr(y.astype(float)))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    return float(x.rank(method="average").corr(y.rank(method="average")))


def _ols_r2(df: pd.DataFrame, predictors: list[str], target: str) -> dict[str, Any]:
    x = df[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = df[target].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "predictors": predictors,
        "coefficients": {"intercept": float(coef[0]), **{name: float(value) for name, value in zip(predictors, coef[1:])}},
        "r2": float(r2),
    }


def _centrality_masses(graph: nx.Graph, committed: set[int]) -> dict[str, float]:
    if not committed:
        return {
            "degree_influence_mass": 0.0,
            "betweenness_influence_mass": 0.0,
            "pagerank_influence_mass": 0.0,
            "average_committed_degree": 0.0,
        }
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    pagerank = _pagerank_power(graph)
    return {
        "degree_influence_mass": float(sum(degree[node] for node in committed)),
        "betweenness_influence_mass": float(sum(betweenness[node] for node in committed)),
        "pagerank_influence_mass": float(sum(pagerank[node] for node in committed)),
        "average_committed_degree": float(np.mean([graph.degree[node] for node in committed])),
    }


def _pagerank_power(graph: nx.Graph, damping: float = 0.85, iterations: int = 80, tolerance: float = 1e-10) -> dict[int, float]:
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: idx for idx, node in enumerate(nodes)}
    rank = np.ones(n, dtype=float) / n
    teleport = (1.0 - damping) / n
    for _ in range(iterations):
        new_rank = np.full(n, teleport, dtype=float)
        dangling = 0.0
        for node in nodes:
            i = index[node]
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                dangling += rank[i]
                continue
            share = damping * rank[i] / len(neighbors)
            for neighbor in neighbors:
                new_rank[index[neighbor]] += share
        if dangling:
            new_rank += damping * dangling / n
        if np.abs(new_rank - rank).sum() < tolerance:
            rank = new_rank
            break
        rank = new_rank
    total = rank.sum()
    if total > 0:
        rank = rank / total
    return {node: float(rank[index[node]]) for node in nodes}


def compute_influence_mass(v2: dict[str, pd.DataFrame], v2_config: dict[str, Any], out_dir: Path, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    trial = v2["trial"]
    hub_results = v2["hub"]
    hub_trial = trial[trial["experiment"] == "hub_placement"].copy()
    base_trial = hub_trial[hub_trial["defense"] == NO_DEFENSE].copy()
    records: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], dict[str, float]] = {}
    for _, row in base_trial.iterrows():
        key = (
            int(row["trial"]),
            float(row["minority_ratio"]),
            str(row["graph_topology"]),
            str(row["committed_placement"]),
            str(row["experiment"]),
        )
        if key not in cache:
            graph, source_groups = build_graph(
                int(v2_config["n_agents"]),
                str(row["graph_topology"]),
                v2_config["graph"],
                int(row["graph_seed"]),
            )
            placement_rng = np.random.default_rng(
                stable_seed(
                    int(v2_config["seed"]),
                    "placement",
                    int(row["trial"]),
                    float(row["minority_ratio"]),
                    str(row["graph_topology"]),
                    str(row["committed_placement"]),
                    str(row["experiment"]),
                )
            )
            committed = _select_committed_agents(
                graph,
                float(row["minority_ratio"]),
                str(row["committed_placement"]),
                placement_rng,
                source_groups=source_groups,
            )
            cache[key] = _centrality_masses(graph, committed)
        records.append({**{name: key[idx] for idx, name in enumerate(["trial", "minority_ratio", "graph_topology", "committed_placement", "experiment"])}, **cache[key]})

    influence_trial = pd.DataFrame(records)
    influence_by_condition = influence_trial.groupby(["minority_ratio", "committed_placement"], as_index=False).agg(
        degree_influence_mass=("degree_influence_mass", "mean"),
        betweenness_influence_mass=("betweenness_influence_mass", "mean"),
        pagerank_influence_mass=("pagerank_influence_mass", "mean"),
        average_committed_degree=("average_committed_degree", "mean"),
    )
    influence_by_condition["influence_mass"] = influence_by_condition["degree_influence_mass"]

    result = hub_results.merge(influence_by_condition, on=["minority_ratio", "committed_placement"], how="left")
    result = result[
        [
            "committed_placement",
            "placement_label",
            "minority_ratio",
            "defense",
            "defense_label",
            "influence_mass",
            "degree_influence_mass",
            "betweenness_influence_mass",
            "pagerank_influence_mass",
            "average_committed_degree",
            "false_consensus_rate",
            "true_consensus_rate",
            "no_consensus_rate",
            "average_individual_accuracy",
            "time_to_consensus",
            "intervention_cost",
            "n_trials",
        ]
    ]
    result.to_csv(out_dir / "influence_mass_results.csv", index=False)

    no_defense = result[result["defense"] == NO_DEFENSE].copy()
    predictor_summaries = {}
    for predictor in ["minority_ratio", "degree_influence_mass", "betweenness_influence_mass", "pagerank_influence_mass"]:
        predictor_summaries[predictor] = {
            "pearson": _pearson(no_defense[predictor], no_defense["false_consensus_rate"]),
            "spearman": _spearman(no_defense[predictor], no_defense["false_consensus_rate"]),
            "simple_regression": _ols_r2(no_defense, [predictor], "false_consensus_rate"),
        }
    combined = _ols_r2(no_defense, ["minority_ratio", "degree_influence_mass"], "false_consensus_rate")
    best_single = max(predictor_summaries.items(), key=lambda item: item[1]["simple_regression"]["r2"])
    summary = {
        "analysis_scope": "V2 hub-placement conditions, no_defense rows for predictive-power comparisons",
        "default_influence_mass": "degree_influence_mass",
        "condition_rows": int(len(result)),
        "no_defense_condition_rows": int(len(no_defense)),
        "predictors": predictor_summaries,
        "combined_regression_minority_plus_degree_influence": combined,
        "best_single_predictor_by_r2": {
            "name": best_single[0],
            "r2": best_single[1]["simple_regression"]["r2"],
        },
        "influence_mass_better_than_minority_ratio": bool(
            predictor_summaries["degree_influence_mass"]["simple_regression"]["r2"]
            > predictor_summaries["minority_ratio"]["simple_regression"]["r2"]
        ),
    }
    (out_dir / "influence_mass_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return result, summary


def plot_influence_mass(influence: pd.DataFrame, out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = influence[influence["defense"] == NO_DEFENSE].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for placement, subset in data.groupby("committed_placement"):
        ax.scatter(
            subset["degree_influence_mass"],
            subset["false_consensus_rate"],
            s=70 + 260 * subset["minority_ratio"],
            label=_label_placement(placement),
            alpha=0.82,
            edgecolor="#222222",
            linewidth=0.5,
        )
    x = data["degree_influence_mass"].to_numpy(dtype=float)
    y = data["false_consensus_rate"].to_numpy(dtype=float)
    if len(np.unique(x)) > 1:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#111111", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Influence mass: sum of degree centrality over committed agents")
    ax.set_ylabel("False consensus rate")
    ax.set_title("Influence mass vs belief capture")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return _save_fig(fig, plot_dir, "figure_5_influence_mass_vs_capture")


def write_influence_table(influence: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> Path:
    no_defense = influence[influence["defense"] == NO_DEFENSE].copy()
    table = no_defense.groupby("placement_label", as_index=False).agg(
        influence_mass=("degree_influence_mass", "mean"),
        betweenness_mass=("betweenness_influence_mass", "mean"),
        pagerank_mass=("pagerank_influence_mass", "mean"),
        false_consensus_rate=("false_consensus_rate", "mean"),
    )
    table.loc[len(table)] = {
        "placement_label": "R2: minority ratio",
        "influence_mass": summary["predictors"]["minority_ratio"]["simple_regression"]["r2"],
        "betweenness_mass": np.nan,
        "pagerank_mass": np.nan,
        "false_consensus_rate": np.nan,
    }
    table.loc[len(table)] = {
        "placement_label": "R2: influence mass",
        "influence_mass": summary["predictors"]["degree_influence_mass"]["simple_regression"]["r2"],
        "betweenness_mass": np.nan,
        "pagerank_mass": np.nan,
        "false_consensus_rate": np.nan,
    }
    path = out_dir / "table_influence_mass.tex"
    write_tex_table(path, table, "Influence-mass analysis for hub-placement no-defense conditions.", "tab:influence-mass")
    return path


def maybe_run_local_llm(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    local_cfg = config.get("local_llm", {})
    json_path = out_dir / "local_llm_sanity.json"
    csv_path = out_dir / "local_llm_sanity.csv"
    model_path = local_cfg.get("model_path") or os.environ.get("LOCAL_LLM_MODEL_PATH")
    missing: list[str] = []
    if not local_cfg.get("enabled", True):
        missing.append("local_llm.enabled is false")
    if importlib.util.find_spec("transformers") is None:
        missing.append("transformers")
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if not model_path:
        missing.append("local model path")
    elif not Path(str(model_path)).exists():
        missing.append(f"existing local model path: {model_path}")
    if missing:
        reason = "Missing " + ", ".join(missing)
        status = {"status": "skipped", "reason": reason, "no_external_apis": True}
        pd.DataFrame([status]).to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    subset = load_config("configs/v2.yaml")
    subset["model_type"] = "local_llm"
    subset["experiment"] = "v4_local_llm_sanity"
    subset["n_agents"] = int(local_cfg.get("n_agents", 10))
    subset["rounds"] = int(local_cfg.get("rounds", 4))
    subset["trials"] = int(local_cfg.get("trials", 3))
    subset["minority_ratios"] = list(local_cfg.get("minority_ratios", [0.0, 0.05, 0.10, 0.20]))
    subset["evidence_strengths"] = list(local_cfg.get("evidence_strengths", [0.66]))
    subset["memory_lengths"] = list(local_cfg.get("memory_lengths", [1]))
    subset["graph_topologies"] = list(local_cfg.get("graph_topologies", ["random", "hub"]))
    subset["defenses"] = list(local_cfg.get("defenses", [NO_DEFENSE, EVIDENCE_ANCHORING_ONLY, COMBINED_V1]))
    subset["defense_strengths"] = list(local_cfg.get("defense_strengths", [1.0]))
    subset["update_rules"] = ["local_llm_agent"]
    subset["committed_placements"] = list(local_cfg.get("committed_placements", ["random", "high_degree"]))
    subset["local_llm"]["model_path"] = str(model_path)
    subset["local_llm"]["max_new_tokens"] = int(local_cfg.get("max_new_tokens", 48))
    subset["local_llm"]["temperature"] = float(local_cfg.get("temperature", 0.0))
    local_out = out_dir / "local_llm_run"
    _, aggregate = run_sweep(subset, local_out)
    aggregate.to_csv(csv_path, index=False)
    status = {"status": "completed", "aggregate_rows": len(aggregate), "csv": csv_path.as_posix(), "no_external_apis": True}
    json_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def run_baseline_comparison(config: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    sweep = config["baseline_sweep"]
    base = load_config("configs/v2.yaml")
    base["experiment"] = "v4_baseline_comparison"
    base["n_agents"] = int(sweep["n_agents"])
    base["rounds"] = int(sweep["rounds"])
    base["trials"] = int(sweep["trials"])
    base["minority_ratios"] = list(sweep["minority_ratios"])
    base["evidence_strengths"] = list(sweep["evidence_strengths"])
    base["memory_lengths"] = list(sweep["memory_lengths"])
    base["graph_topologies"] = list(sweep["graph_topologies"])
    base["update_rules"] = list(sweep["update_rules"])
    base["committed_placements"] = list(sweep["committed_placements"])
    base["defenses"] = list(sweep["defenses"])
    base["defense_strengths"] = list(sweep["defense_strengths"])
    conditions = []
    for condition in __import__("collective_belief_benchmark.simulation", fromlist=["iter_conditions"]).iter_conditions(base):
        conditions.append(condition)
    trials = run_conditions(base, conditions, progress_prefix="completed v4 baselines")
    aggregate = aggregate_results(trials, bootstrap_iterations=500, ci_level=0.95, ci_seed=int(config["seed"]))
    summary = aggregate.groupby("defense", as_index=False).agg(
        false_consensus_rate=("false_consensus_rate", "mean"),
        true_consensus_rate=("true_consensus_rate", "mean"),
        no_consensus_rate=("no_consensus_rate", "mean"),
        average_individual_accuracy=("average_individual_accuracy", "mean"),
        time_to_consensus=("time_to_consensus", "mean"),
        intervention_cost=("intervention_cost", "mean"),
    )
    baseline_false = float(summary.loc[summary["defense"] == NO_DEFENSE, "false_consensus_rate"].iloc[0])
    summary["false_consensus_reduction_vs_no_defense"] = baseline_false - summary["false_consensus_rate"]
    descriptions = {
        NO_DEFENSE: "No intervention.",
        MAJORITY_VOTE: "Aggregates neighbor claims as unweighted votes.",
        CONFIDENCE_WEIGHTED_AGGREGATION: "Weights neighbor claims by reported confidence.",
        RANDOM_SOURCE_DROPOUT: "Drops a deterministic random subset of message sources.",
        HUB_DOWNWEIGHTING: "Downweights highly connected senders using graph structure only.",
        MESSAGE_SUPPORT_CHECKING: "Uses only claims whose cited evidence supports the claim.",
        COMBINED_V1: "Evidence anchoring plus support checking, diversity, and memory decay.",
    }
    summary["defense_label"] = summary["defense"].map(_label_defense)
    summary["description"] = summary["defense"].map(descriptions)
    summary.to_csv(out_dir / "baseline_comparison.csv", index=False)
    write_tex_table(
        out_dir / "table_baseline_comparison.tex",
        summary[["defense_label", "false_consensus_rate", "true_consensus_rate", "no_consensus_rate", "intervention_cost", "false_consensus_reduction_vs_no_defense"]],
        "Targeted comparison of conceptually comparable aggregation and inoculation baselines.",
        "tab:baseline-comparison",
    )
    return summary


def plot_llm_if_available(out_dir: Path, status: dict[str, Any]) -> list[Path]:
    if status.get("status") != "completed":
        return []
    csv_path = out_dir / "local_llm_sanity.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "minority_ratio" not in df.columns:
        return []
    plot_dir = out_dir / "plots"
    data = df.groupby(["minority_ratio", "graph_topology", "committed_placement", "defense"], as_index=False)["false_consensus_rate"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, subset in data.groupby(["graph_topology", "committed_placement", "defense"]):
        subset = subset.sort_values("minority_ratio")
        ax.plot(subset["minority_ratio"] * 100, subset["false_consensus_rate"], marker="o", label="/".join(map(str, key)))
    ax.set_xlabel("Committed minority (%)")
    ax.set_ylabel("False consensus rate")
    ax.set_title("V4 local LLM sanity subset")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6)
    return _save_fig(fig, plot_dir, "figure_6_llm_sanity")


def create_paper_skeleton(numbers: dict[str, Any], influence_summary: dict[str, Any], local_status: dict[str, Any], baseline: pd.DataFrame) -> None:
    paper = Path("paper")
    sections = paper / "sections"
    sections.mkdir(parents=True, exist_ok=True)
    main = r"""\documentclass[letterpaper]{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\title{Collective Belief Capture in Artificial Agent Populations}
\author{Anonymous Authors}
\date{}
\begin{document}
\maketitle
\begin{abstract}
Errors in artificial agent populations can become collective rather than individual. We study a synthetic hidden-state benchmark in which agents exchange messages about a binary state. Small committed false-belief minorities can induce false consensus, and network position can matter more than minority size. In verified V2/V4 results, 5\% high-degree committed placement produced higher false consensus than 20\% low-degree placement. Leakage-free structural inoculation reduced false consensus while changing consensus formation. This paper studies artificial agent populations only; it is not a human social simulation.
\end{abstract}
\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_problem_setup}
\input{sections/04_benchmark_and_inoculation}
\input{sections/05_experiments}
\input{sections/06_limitations}
\input{sections/07_conclusion}
\end{document}
"""
    (paper / "main.tex").write_text(main, encoding="utf-8")
    files = {
        "01_introduction.tex": rf"""\section{{Introduction}}
Artificial agents increasingly communicate with other agents. This creates the possibility that errors become collective rather than individual. We study this question in a synthetic A/B hidden-state benchmark.

Main verified numbers used in this draft: no-defense false consensus is {numbers['no_defense_false_consensus']:.3f}, combined false consensus is {numbers['combined_false_consensus']:.3f}, and combined true consensus is {numbers['combined_true_consensus']:.3f}.

% Related literature spans multi-agent systems, belief propagation, social learning, and AI safety.
""",
        "02_related_work.tex": r"""\section{Related Work}
% Relevant areas: social learning on networks, opinion dynamics, multi-agent communication, misinformation robustness, and LLM-agent evaluation.

No fabricated citations are included in this skeleton.
""",
        "03_problem_setup.tex": r"""\section{Problem Setup}
We study artificial agent populations on a neutral binary hidden-state task. The hidden state is either $A$ or $B$. Normal agents receive noisy private evidence. A committed false-belief minority asserts the false state and does not update. Agents communicate for a fixed number of rounds over a configurable graph.

The benchmark avoids political, medical, and real-world persuasion content.
""",
        "04_benchmark_and_inoculation.tex": r"""\section{Benchmark and Inoculation}
We compare no defense, majority vote, confidence-weighted aggregation, random source dropout, hub downweighting, message support checking, and a combined inoculation policy. Defenses are leakage-free: they do not access the hidden state or committed-agent identity.

% Implementation details and validity tests are described in the accompanying sections.
""",
        "05_experiments.tex": rf"""\section{{Experiments}}
\paragraph{{Network position.}} High-degree/hub placement false consensus is {numbers['hub_placement_false_consensus']:.3f}; low-degree placement is {numbers['low_degree_placement_false_consensus']:.3f}; random placement is {numbers['random_placement_false_consensus']:.3f}; clustered placement is {numbers['clustered_placement_false_consensus']:.3f}.

\paragraph{{Minority size versus influence mass.}} The V4 influence-mass analysis compares regressions using minority ratio and centrality mass. Degree influence mass better than minority ratio: {influence_summary['influence_mass_better_than_minority_ratio']}.

\paragraph{{Local LLM sanity.}} Local LLM sanity status: {local_status['status']}. This result should not be used to claim real LLM-agent society behavior unless the sanity subset is completed.
""",
        "06_limitations.tex": r"""\section{Limitations}
This work studies artificial agent populations, not humans. Scripted update rules are controlled mechanisms, not full LLM cognition. Structured-message defenses assume support/citation fields can be inspected. Local LLM sanity evidence is optional and should be reported only if it runs.
""",
        "07_conclusion.tex": r"""\section{Conclusion}
The benchmark supports a narrow claim: in synthetic artificial-agent populations, false beliefs can become collective, network position can amplify minority influence, and structural inoculation can reduce false consensus while affecting consensus formation.
""",
    }
    for name, text in files.items():
        (sections / name).write_text(text, encoding="utf-8")


def write_v4_report(
    numbers: dict[str, Any],
    influence_summary: dict[str, Any],
    local_status: dict[str, Any],
    baseline: pd.DataFrame,
    out_dir: Path,
    report_path: Path,
) -> Path:
    baseline_view = baseline[["defense_label", "false_consensus_rate", "true_consensus_rate", "no_consensus_rate", "intervention_cost", "false_consensus_reduction_vs_no_defense"]]
    supported = pd.DataFrame(
        [
            {"claim": "Small committed minorities can induce false consensus in artificial agent populations.", "support": f"No-defense false consensus = {numbers['no_defense_false_consensus']:.3f}."},
            {"claim": "Network position can matter more than raw minority size.", "support": f"5% high-degree = {numbers['five_percent_high_degree_false_consensus']:.3f}; 20% low-degree = {numbers['twenty_percent_low_degree_false_consensus']:.3f}."},
            {"claim": "Structural inoculation can reduce false consensus.", "support": f"Combined false consensus = {numbers['combined_false_consensus']:.3f} in V2 core sweep."},
        ]
    )
    unsupported = pd.DataFrame(
        [
            {"claim": "This is a human social simulation.", "status": "Not supported."},
            {"claim": "This proves behavior of real LLM societies.", "status": "Not supported unless local LLM sanity completes and remains limited."},
            {"claim": "Influence mass is universally causal.", "status": "Not supported; V4 reports predictive association in this synthetic benchmark."},
        ]
    )
    lines = [
        "# Collective Belief Capture V4 Final Readiness",
        "",
        "## What V4 Added",
        "",
        "- Influence-mass analysis using degree centrality, betweenness centrality, and PageRank mass over committed false agents.",
        "- Predictive comparison of minority ratio versus influence mass for false consensus.",
        "- Targeted comparable-baseline sweep: majority vote, confidence weighting, random source dropout, hub downweighting, message support checking, no defense, and combined inoculation.",
        "- Local LLM sanity run/skip files with exact missing-dependency documentation.",
        "- Paper skeleton under `paper/` with no fabricated citations.",
        "",
        "## Influence-Mass Finding",
        "",
        f"Degree influence-mass R2: `{influence_summary['predictors']['degree_influence_mass']['simple_regression']['r2']:.3f}`.",
        f"Minority-ratio R2: `{influence_summary['predictors']['minority_ratio']['simple_regression']['r2']:.3f}`.",
        f"Combined minority + degree influence R2: `{influence_summary['combined_regression_minority_plus_degree_influence']['r2']:.3f}`.",
        f"Influence mass explains false consensus better than raw minority ratio in this analysis: `{influence_summary['influence_mass_better_than_minority_ratio']}`.",
        "",
        "## Local LLM Sanity",
        "",
        f"Status: `{local_status.get('status')}`.",
        f"Reason/output: `{local_status.get('reason', local_status.get('csv', 'n/a'))}`.",
        "",
        "## Comparable Baselines",
        "",
        _markdown_table(baseline_view),
        "",
        "These targeted baselines do not change the core conclusion if combined/message-support-style defenses still reduce false consensus relative to no defense. They are not a full benchmark rerun.",
        "",
        "## Supported Claims",
        "",
        _markdown_table(supported),
        "",
        "## Claims Not Supported",
        "",
        _markdown_table(unsupported),
        "",
        "## Final Paper-Readiness Judgment",
        "",
        "Ready for manuscript drafting as a synthetic artificial-agent benchmark paper, with careful limitations. The strongest safe framing is structural: network position and source support mechanisms affect collective belief capture in controlled artificial agent populations.",
        "",
        "## Remaining Limitations",
        "",
        "- Local LLM sanity is skipped unless local dependencies/model path are configured.",
        "- Influence mass is predictive evidence here, not proof of universal causality.",
        "- Support-checking defenses are strong under structured-message assumptions.",
        "- No human social simulation claims should be made.",
        "",
        "## Outputs",
        "",
        f"- Influence mass CSV: `{(out_dir / 'influence_mass_results.csv').as_posix()}`",
        f"- Influence summary: `{(out_dir / 'influence_mass_summary.json').as_posix()}`",
        f"- Baseline comparison: `{(out_dir / 'baseline_comparison.csv').as_posix()}`",
        f"- Local LLM status: `{(out_dir / 'local_llm_sanity.json').as_posix()}`",
        f"- Paper skeleton: `paper/main.tex`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_v4(config: dict[str, Any], out_dir: str | Path, report_path: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    save_config_snapshot(config, out / "config_snapshot.yaml")
    v2 = _load_v2(config["v2_dir"])
    v2_config = load_config(Path(config["v2_dir"]) / "config_snapshot.yaml")
    numbers = _compute_paper_numbers(v2)
    influence, influence_summary = compute_influence_mass(v2, v2_config, out, int(config["seed"]))
    influence_figs = plot_influence_mass(influence, out)
    table_path = write_influence_table(influence, influence_summary, out)
    local_status = maybe_run_local_llm(config, out)
    llm_figs = plot_llm_if_available(out, local_status)
    baseline = run_baseline_comparison(config, out)
    create_paper_skeleton(numbers, influence_summary, local_status, baseline)
    report = write_v4_report(numbers, influence_summary, local_status, baseline, out, Path(report_path))
    return {
        "influence_rows": len(influence),
        "baseline_rows": len(baseline),
        "influence_figures": len(influence_figs),
        "llm_figures": len(llm_figs),
        "local_llm_status": local_status.get("status"),
        "table": table_path.as_posix(),
        "report": report.as_posix(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 final paper-readiness analysis.")
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default="COLLECTIVE_BELIEF_CAPTURE_V4_FINAL_READINESS.md")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--skip-local-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.out is not None:
        config["out_dir"] = args.out
    if args.local_model_path is not None:
        config["local_llm"]["model_path"] = args.local_model_path
    if args.skip_local_llm:
        config["local_llm"]["enabled"] = False
    result = run_v4(config, config.get("out_dir", "outputs/v4"), args.report)
    print(f"influence rows: {result['influence_rows']}")
    print(f"baseline rows: {result['baseline_rows']}")
    print(f"influence figures: {result['influence_figures']}")
    print(f"llm figures: {result['llm_figures']}")
    print(f"local llm: {result['local_llm_status']}")
    print(f"report: {result['report']}")
