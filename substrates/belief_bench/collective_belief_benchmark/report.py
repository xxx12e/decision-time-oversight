from __future__ import annotations

from pathlib import Path

import pandas as pd

from .defenses import DEFENSE_LABELS, NO_DEFENSE
from .plotting import _max_strength_slice


def _label(defense: str) -> str:
    return DEFENSE_LABELS.get(defense, defense.replace("_", " ").title())


def _tipping_points(aggregate: pd.DataFrame) -> pd.DataFrame:
    strong = _max_strength_slice(aggregate)
    rows = []
    for defense, subset in strong.groupby("defense"):
        curve = subset.groupby("minority_ratio", as_index=False)["false_consensus_rate"].mean().sort_values("minority_ratio")
        tipping = None
        for _, row in curve.iterrows():
            if row["false_consensus_rate"] >= 0.50:
                tipping = row["minority_ratio"]
                break
        rows.append(
            {
                "defense": _label(defense),
                "tipping_point_minority_ratio": "not reached" if tipping is None else f"{tipping:.0%}",
                "mean_false_consensus_rate": curve["false_consensus_rate"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_false_consensus_rate")


def _markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return ""
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{value:.{float_digits}f}")
    columns = [str(col) for col in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.to_numpy()]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    header = "| " + " | ".join(columns[index].ljust(widths[index]) for index in range(len(columns))) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_report(
    aggregate: pd.DataFrame,
    out_dir: str | Path,
    report_path: str | Path,
    config: dict,
) -> Path:
    out_dir = Path(out_dir)
    report_path = Path(report_path)
    strong = _max_strength_slice(aggregate)

    headline = (
        strong.groupby("defense", as_index=False)
        .agg(
            false_consensus_rate=("false_consensus_rate", "mean"),
            true_consensus_rate=("true_consensus_rate", "mean"),
            average_belief_accuracy=("final_average_belief_accuracy", "mean"),
            intervention_cost=("intervention_cost", "mean"),
        )
        .sort_values("false_consensus_rate")
    )
    headline["defense"] = headline["defense"].map(_label)
    tipping = _tipping_points(aggregate)

    no_defense = headline[headline["defense"] == _label(NO_DEFENSE)]
    best_defense = headline.iloc[0] if not headline.empty else None
    baseline_false = float(no_defense["false_consensus_rate"].iloc[0]) if not no_defense.empty else float("nan")

    plot_root = out_dir / "plots"
    rel_plot_root = Path("outputs") / out_dir.name / "plots" if out_dir.name else plot_root
    if out_dir.is_absolute():
        # Keep generated report portable within this workspace when using outputs/latest.
        try:
            rel_plot_root = plot_root.relative_to(report_path.parent)
        except ValueError:
            rel_plot_root = plot_root

    lines = [
        "# Collective Belief Capture Report",
        "",
        "## Scope",
        "",
        "This benchmark studies artificial agent populations on a neutral synthetic hidden-state task. The hidden state is binary (`A` or `B`), and normal agents receive noisy private evidence before communicating with neighboring agents. Some agents are committed to the false state and do not update.",
        "",
        "The benchmark intentionally avoids political, medical, and real-world persuasion topics. It uses local computation only and does not call external APIs.",
        "",
        "## Paper Framing",
        "",
        "As AI agents increasingly communicate with other agents, errors may become collective rather than individual. This benchmark tests how false beliefs can spread through agent populations and evaluates minimal inoculation mechanisms that preserve useful consensus while reducing belief capture.",
        "",
        "This is not a human social simulation paper; it is a controlled study of artificial LLM-agent population dynamics under synthetic evidence.",
        "",
        "## Benchmark Design",
        "",
        f"- Agents: `{config['n_agents']}`",
        f"- Rounds: `{config['rounds']}`",
        f"- Trials per condition: `{config['trials']}`",
        f"- Consensus threshold: `{config['consensus_threshold']:.0%}` of normal agents",
        f"- Model type: `{config['model_type']}`",
        f"- Committed minority ratios: `{', '.join(str(x) for x in config['minority_ratios'])}`",
        f"- Evidence strengths: `{', '.join(str(x) for x in config['evidence_strengths'])}`",
        f"- Memory lengths: `{', '.join(str(x) for x in config['memory_lengths'])}`",
        f"- Graph topologies: `{', '.join(config['graph_topologies'])}`",
        "",
        "Committed agents are assigned to high-degree nodes by default to stress-test capture under adversarial placement. Change `committed.selection` to `random` for random placement.",
        "",
        "## Metrics",
        "",
        "- False consensus rate: share of trials where normal agents converge to the false state.",
        "- True consensus rate: share of trials where normal agents converge to the true state.",
        "- Time to consensus: first round where either consensus threshold is crossed; non-consensus uses `rounds + 1`.",
        "- Average belief accuracy: final mean probability normal agents assign to the true state.",
        "- Intervention cost: normalized local overhead from filtering, quarantine, and caution operations.",
        "- Robustness by graph structure: false consensus rate stratified by topology.",
        "- Tipping-point minority ratio: lowest committed minority ratio where mean false consensus reaches at least 50%.",
        "",
        "## Headline Results",
        "",
        f"Baseline false consensus rate with no defense: `{baseline_false:.3f}`.",
    ]
    if best_defense is not None:
        lines.append(
            f"Lowest observed false consensus in this run: `{best_defense['defense']}` at `{best_defense['false_consensus_rate']:.3f}` false consensus with `{best_defense['true_consensus_rate']:.3f}` true consensus."
        )
    lines.extend(
        [
            "",
            _markdown_table(headline),
            "",
            "## Tipping Points",
            "",
            _markdown_table(tipping),
            "",
            "## Plots",
            "",
            f"![False consensus vs minority ratio]({(rel_plot_root / 'false_consensus_vs_minority_ratio.png').as_posix()})",
            "",
            f"![False consensus vs network topology]({(rel_plot_root / 'false_consensus_vs_network_topology.png').as_posix()})",
            "",
            f"![Defense strength vs false consensus]({(rel_plot_root / 'defense_strength_vs_false_consensus.png').as_posix()})",
            "",
            f"![True/false consensus tradeoff]({(rel_plot_root / 'true_false_consensus_tradeoff.png').as_posix()})",
            "",
            "## Output Files",
            "",
            f"- Trial CSV: `{(out_dir / 'trial_results.csv').as_posix()}`",
            f"- Trial JSON: `{(out_dir / 'trial_results.json').as_posix()}`",
            f"- Aggregate CSV: `{(out_dir / 'aggregate_results.csv').as_posix()}`",
            f"- Aggregate JSON: `{(out_dir / 'aggregate_results.json').as_posix()}`",
            f"- Config snapshot: `{(out_dir / 'config_snapshot.yaml').as_posix()}`",
            "",
            "## Local LLM Support",
            "",
            "The scripted Bayesian-like agents are the default baseline. To run local open-weight LLM agents, install local `transformers` dependencies and pass `--model-type local_llm --local-model-path <local model directory>`. The loader uses local files only.",
            "",
            "For paper-scale experiments, use scripted agents for broad sweeps and local LLM agents for smaller targeted comparisons because local generation is much slower.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
