from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, save_config_snapshot
from .defenses import (
    COMBINED_V1,
    DEFENSE_LABELS,
    EVIDENCE_ANCHORING_ONLY,
    NO_DEFENSE,
    SOURCE_DIVERSITY_ONLY,
    UNSUPPORTED_QUARANTINE_ONLY,
)
from .report import _markdown_table
from .simulation import _bootstrap_mean_ci, run_sweep, stable_seed


PLACEMENT_ORDER = ["high_degree", "clustered", "random", "low_degree"]
PLACEMENT_LABELS = {
    "high_degree": "High-degree / hub",
    "clustered": "Clustered",
    "random": "Random",
    "low_degree": "Low-degree",
}
DEFENSE_ORDER = [
    NO_DEFENSE,
    EVIDENCE_ANCHORING_ONLY,
    SOURCE_DIVERSITY_ONLY,
    UNSUPPORTED_QUARANTINE_ONLY,
    COMBINED_V1,
]
FIG_COLORS = {
    "high_degree": "#005f73",
    "clustered": "#9b2226",
    "random": "#ca6702",
    "low_degree": "#577590",
    NO_DEFENSE: "#7a7a7a",
    EVIDENCE_ANCHORING_ONLY: "#3a7ca5",
    SOURCE_DIVERSITY_ONLY: "#b07d30",
    UNSUPPORTED_QUARANTINE_ONLY: "#2a9d8f",
    COMBINED_V1: "#264653",
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


def _save_fig(fig: plt.Figure, plot_dir: Path, stem: str) -> list[Path]:
    png = plot_dir / f"{stem}.png"
    pdf = plot_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _load_v2(v2_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(v2_dir)
    required = {
        "trial": root / "trial_results.csv",
        "aggregate": root / "aggregate_results.csv",
        "update_rule": root / "update_rule_results.csv",
        "hub": root / "hub_placement_results.csv",
        "cost": root / "defense_cost_results.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required V2 files: {missing}")
    return {key: pd.read_csv(path) for key, path in required.items()}


def _compute_paper_numbers(v2: dict[str, pd.DataFrame]) -> dict[str, Any]:
    cost = v2["cost"]
    hub = v2["hub"]
    aggregate = v2["aggregate"]

    def cost_value(defense: str, column: str) -> float:
        return float(cost.loc[cost["defense"] == defense, column].iloc[0])

    def placement_value(placement: str, minority_ratio: float | None = None) -> float:
        rows = hub[(hub["defense"] == NO_DEFENSE) & (hub["committed_placement"] == placement)]
        if minority_ratio is not None:
            rows = rows[rows["minority_ratio"] == minority_ratio]
        return float(rows["false_consensus_rate"].mean())

    outcome_error = float((aggregate["outcome_rate_sum"] - 1.0).abs().max())
    return {
        "no_defense_false_consensus": cost_value(NO_DEFENSE, "false_consensus_rate"),
        "combined_false_consensus": cost_value(COMBINED_V1, "false_consensus_rate"),
        "combined_true_consensus": cost_value(COMBINED_V1, "true_consensus_rate"),
        "hub_placement_false_consensus": placement_value("high_degree"),
        "low_degree_placement_false_consensus": placement_value("low_degree"),
        "random_placement_false_consensus": placement_value("random"),
        "clustered_placement_false_consensus": placement_value("clustered"),
        "five_percent_high_degree_false_consensus": placement_value("high_degree", 0.05),
        "twenty_percent_low_degree_false_consensus": placement_value("low_degree", 0.20),
        "outcome_accounting_max_error": outcome_error,
    }


def _bootstrap_group_ci(
    trial_df: pd.DataFrame,
    group_col: str,
    value_col: str,
    seed: int,
    bootstrap_iterations: int,
    ci_level: float,
) -> pd.DataFrame:
    rows = []
    for group, subset in trial_df.groupby(group_col):
        low, high = _bootstrap_mean_ci(
            subset[value_col].to_numpy(dtype=float),
            iterations=bootstrap_iterations,
            ci_level=ci_level,
            seed=stable_seed(seed, "v3-ci", group_col, group, value_col),
        )
        rows.append({group_col: group, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def figure_capture_by_placement(v2: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    hub = v2["hub"]
    data = hub[hub["defense"] == NO_DEFENSE].copy()
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    for placement in PLACEMENT_ORDER:
        subset = data[data["committed_placement"] == placement].sort_values("minority_ratio")
        if subset.empty:
            continue
        x = subset["minority_ratio"].to_numpy(dtype=float) * 100
        y = subset["false_consensus_rate"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=FIG_COLORS[placement],
            label=_label_placement(placement),
        )
        if {"false_consensus_rate_ci_low", "false_consensus_rate_ci_high"}.issubset(subset.columns):
            ax.fill_between(
                x,
                subset["false_consensus_rate_ci_low"].to_numpy(dtype=float),
                subset["false_consensus_rate_ci_high"].to_numpy(dtype=float),
                color=FIG_COLORS[placement],
                alpha=0.10,
            )

    high_5 = data[(data["committed_placement"] == "high_degree") & (data["minority_ratio"] == 0.05)]
    low_20 = data[(data["committed_placement"] == "low_degree") & (data["minority_ratio"] == 0.20)]
    if not high_5.empty and not low_20.empty:
        hx = 5
        hy = float(high_5["false_consensus_rate"].iloc[0])
        lx = 20
        ly = float(low_20["false_consensus_rate"].iloc[0])
        ax.scatter([hx, lx], [hy, ly], s=90, facecolors="none", edgecolors="#111111", linewidths=1.7, zorder=5)
        ax.annotate(
            "5% high-degree\n> 20% low-degree",
            xy=(hx, hy),
            xytext=(10, min(0.96, hy + 0.22)),
            arrowprops={"arrowstyle": "->", "linewidth": 1.1, "color": "#222222"},
            fontsize=9,
        )
        ax.annotate(
            "",
            xy=(lx, ly),
            xytext=(11.5, min(0.94, hy + 0.17)),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#222222"},
        )

    ax.set_xlabel("Committed false-belief minority (%)")
    ax.set_ylabel("False consensus rate")
    ax.set_title("Network position amplifies belief capture")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-1, 31)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    return _save_fig(fig, plot_dir, "figure_1_capture_by_placement")


def figure_hub_amplification(v2: dict[str, pd.DataFrame], out_dir: Path, seed: int, bootstrap_iterations: int, ci_level: float) -> list[Path]:
    plot_dir = out_dir / "plots"
    trial = v2["trial"]
    placement_trial = trial[(trial["experiment"] == "hub_placement") & (trial["defense"] == NO_DEFENSE)].copy()
    keep = ["high_degree", "random", "low_degree"]
    placement_trial = placement_trial[placement_trial["committed_placement"].isin(keep)]
    means = placement_trial.groupby("committed_placement", as_index=False).agg(
        false_consensus_rate=("false_consensus", "mean"),
        committed_degree_ratio=("committed_degree_ratio", "mean"),
    )
    ci = _bootstrap_group_ci(
        placement_trial,
        "committed_placement",
        "false_consensus",
        seed,
        bootstrap_iterations,
        ci_level,
    )
    means = means.merge(ci, on="committed_placement", how="left")
    means["sort_order"] = means["committed_placement"].map({name: idx for idx, name in enumerate(["low_degree", "random", "high_degree"])})
    means = means.sort_values("sort_order")
    x = np.arange(len(means))
    y = means["false_consensus_rate"].to_numpy(dtype=float)
    yerr = np.vstack([y - means["ci_low"].to_numpy(dtype=float), means["ci_high"].to_numpy(dtype=float) - y])

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    bars = ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        color=[FIG_COLORS[name] for name in means["committed_placement"]],
        alpha=0.90,
    )
    for bar, degree_ratio in zip(bars, means["committed_degree_ratio"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"deg x{degree_ratio:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_label_placement(name) for name in means["committed_placement"]])
    ax.set_ylabel("False consensus rate")
    ax.set_title("Hub-position amplification")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    return _save_fig(fig, plot_dir, "figure_2_hub_amplification")


def figure_inoculation_pareto(v2: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    cost = v2["cost"].copy()
    cost = cost[cost["defense"].isin(DEFENSE_ORDER)]
    cost["defense_label"] = cost["defense"].map(_label_defense)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    retention = cost["true_consensus_retention"].clip(lower=0, upper=3.5)
    sizes = 70 + 130 * retention
    for _, row in cost.iterrows():
        ax.scatter(
            row["intervention_cost"],
            row["false_consensus_rate"],
            s=float(sizes.loc[row.name]),
            color=FIG_COLORS.get(row["defense"], "#555555"),
            alpha=0.82,
            edgecolor="#222222",
            linewidth=0.5,
        )
        ax.annotate(
            row["defense_label"],
            (row["intervention_cost"], row["false_consensus_rate"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Intervention cost")
    ax.set_ylabel("False consensus rate")
    ax.set_title("Inoculation cost vs false consensus")
    ax.set_ylim(-0.03, max(0.62, float(cost["false_consensus_rate"].max()) + 0.08))
    ax.set_xlim(-0.005, float(cost["intervention_cost"].max()) + 0.04)
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.96, "Marker size: true-consensus retention", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    return _save_fig(fig, plot_dir, "figure_3_inoculation_pareto")


def figure_consensus_tradeoff(v2: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    cost = v2["cost"].copy()
    cost = cost[cost["defense"].isin(DEFENSE_ORDER)]
    order = [NO_DEFENSE, EVIDENCE_ANCHORING_ONLY, SOURCE_DIVERSITY_ONLY, UNSUPPORTED_QUARANTINE_ONLY, COMBINED_V1]
    cost["sort_order"] = cost["defense"].map({name: idx for idx, name in enumerate(order)})
    cost = cost.sort_values("sort_order")
    y = np.arange(len(cost))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(y, cost["true_consensus_rate"], label="True consensus", color="#4b8f67")
    ax.barh(y, cost["false_consensus_rate"], left=cost["true_consensus_rate"], label="False consensus", color="#b5544c")
    left = cost["true_consensus_rate"] + cost["false_consensus_rate"]
    ax.barh(y, cost["no_consensus_rate"], left=left, label="No consensus", color="#8c8c8c")
    ax.set_yticks(y)
    ax.set_yticklabels(cost["defense"].map(_label_defense), fontsize=8)
    ax.set_xlabel("Outcome share")
    ax.set_title("Consensus tradeoff by defense")
    ax.set_xlim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save_fig(fig, plot_dir, "figure_4_consensus_tradeoff")


def figure_llm_sanity(local_csv: Path, out_dir: Path) -> list[Path]:
    plot_dir = out_dir / "plots"
    df = pd.read_csv(local_csv)
    if "status" in df.columns and set(df["status"].dropna()) == {"skipped"}:
        return []
    if "minority_ratio" not in df.columns:
        return []
    data = df.groupby(["minority_ratio", "graph_topology", "committed_placement", "defense"], as_index=False)["false_consensus_rate"].mean()
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for (topology, placement, defense), subset in data.groupby(["graph_topology", "committed_placement", "defense"]):
        subset = subset.sort_values("minority_ratio")
        ax.plot(
            subset["minority_ratio"] * 100,
            subset["false_consensus_rate"],
            marker="o",
            linewidth=1.5,
            label=f"{topology}/{placement}/{_label_defense(defense)}",
        )
    ax.set_xlabel("Committed false-belief minority (%)")
    ax.set_ylabel("False consensus rate")
    ax.set_title("Local LLM sanity subset")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6)
    return _save_fig(fig, plot_dir, "figure_5_llm_sanity")


def write_tex_table(path: Path, df: pd.DataFrame, caption: str, label: str, float_digits: int = 3) -> None:
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda value: f"{value:.{float_digits}f}")
    escaped_cols = [str(col).replace("_", "\\_") for col in table.columns]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" * len(table.columns) + "}",
        "\\toprule",
        " & ".join(escaped_cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        cells = [str(value).replace("_", "\\_").replace("%", "\\%") for value in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables(v2: dict[str, pd.DataFrame], sanity: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    numbers = _compute_paper_numbers(v2)
    cost = v2["cost"].copy()
    cost["defense"] = cost["defense"].map(_label_defense)
    hub = v2["hub"].copy()
    hub = hub[hub["defense"] == NO_DEFENSE]
    hub["placement"] = hub["committed_placement"].map(_label_placement)

    main = pd.DataFrame(
        [
            {"metric": "No-defense false consensus", "value": numbers["no_defense_false_consensus"]},
            {"metric": "Combined false consensus", "value": numbers["combined_false_consensus"]},
            {"metric": "Combined true consensus", "value": numbers["combined_true_consensus"]},
            {"metric": "Hub minus random capture gap", "value": sanity["hub_vs_random_capture"]["hub_minus_random"]},
            {"metric": "5% high-degree false consensus", "value": numbers["five_percent_high_degree_false_consensus"]},
            {"metric": "20% low-degree false consensus", "value": numbers["twenty_percent_low_degree_false_consensus"]},
        ]
    )
    hub_table = hub.groupby("placement", as_index=False).agg(
        false_consensus_rate=("false_consensus_rate", "mean"),
        committed_degree_ratio=("committed_degree_ratio", "mean"),
    )
    defense_table = cost[
        [
            "defense",
            "intervention_cost",
            "false_consensus_rate",
            "false_consensus_reduction",
            "true_consensus_retention",
            "no_consensus_increase",
        ]
    ]
    validity = pd.DataFrame(
        [
            {"check": "Outcome accounting max error", "value": sanity["schema"]["outcome_rate_sum_max_abs_error"]},
            {"check": "Update-rule coverage", "value": sanity["update_rule_coverage"]["status"]},
            {"check": "0% committed false consensus", "value": sanity["zero_percent_committed"]["no_defense_false_consensus_rate"]},
            {"check": "Minority-ratio Spearman", "value": sanity["minority_ratio_modulation"]["spearman"]},
            {"check": "Evidence-strength Spearman", "value": sanity["evidence_strength_modulation"]["spearman"]},
            {"check": "Placement evidence max range", "value": sanity["hub_placement_validity"]["private_evidence_fraction_max_range_across_placements"]},
        ]
    )

    paths = {
        "main": table_dir / "table_main_results.tex",
        "hub": table_dir / "table_hub_placement.tex",
        "defense": table_dir / "table_defense_ablation.tex",
        "validity": table_dir / "table_validity_checks.tex",
    }
    write_tex_table(paths["main"], main, "Main synthetic-agent benchmark results used in the paper narrative.", "tab:main-results")
    write_tex_table(paths["hub"], hub_table, "Committed-agent placement results on hub graphs.", "tab:hub-placement")
    write_tex_table(paths["defense"], defense_table, "Defense cost and outcome tradeoffs.", "tab:defense-cost")
    write_tex_table(paths["validity"], validity, "Validity and sanity checks for V3 paper-readiness artifacts.", "tab:validity-checks")
    for table_path in paths.values():
        (out_dir / table_path.name).write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")
    return paths


def maybe_run_local_llm_sanity(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    local_cfg = config.get("local_llm", {})
    json_path = out_dir / "local_llm_sanity.json"
    csv_path = out_dir / "local_llm_sanity.csv"
    if not local_cfg.get("enabled", True):
        status = {"status": "skipped", "reason": "local_llm.enabled is false", "no_external_apis": True}
        csv_path.write_text("status,reason,no_external_apis\nskipped,local_llm.enabled is false,True\n", encoding="utf-8")
        json_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    model_path = local_cfg.get("model_path") or os.environ.get("LOCAL_LLM_MODEL_PATH")
    missing: list[str] = []
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

    subset = {
        "seed": int(config.get("seed", 20260601)),
        "model_type": "local_llm",
        "n_agents": int(local_cfg.get("n_agents", 10)),
        "rounds": int(local_cfg.get("rounds", 4)),
        "trials": int(local_cfg.get("trials", 3)),
        "consensus_threshold": 0.8,
        "ci_level": float(config.get("ci_level", 0.95)),
        "bootstrap_iterations": min(int(config.get("bootstrap_iterations", 1000)), 200),
        "balanced_private_evidence": True,
        "experiment": "local_llm_sanity",
        "minority_ratios": list(local_cfg.get("minority_ratios", [0.0, 0.05, 0.10, 0.20])),
        "evidence_strengths": list(local_cfg.get("evidence_strengths", [0.66])),
        "evidence_strength_labels": {"0.66": "medium"},
        "memory_lengths": list(local_cfg.get("memory_lengths", [1])),
        "graph_topologies": list(local_cfg.get("graph_topologies", ["random", "hub"])),
        "defenses": list(local_cfg.get("defenses", [NO_DEFENSE, EVIDENCE_ANCHORING_ONLY, COMBINED_V1])),
        "defense_strengths": list(local_cfg.get("defense_strengths", [1.0])),
        "update_rules": ["local_llm_agent"],
        "committed_placements": list(local_cfg.get("committed_placements", ["random", "high_degree"])),
        "graph": {
            "target_degree": 4,
            "random_edge_prob": 0.20,
            "small_world_rewire_prob": 0.18,
            "hub_attachment": 2,
            "clusters": 3,
            "clustered_intra_prob": 0.42,
            "clustered_inter_prob": 0.04,
            "source_groups": 3,
        },
        "committed": {"selection": "high_degree", "confidence": 0.99, "logodds": 4.0},
        "update": {
            "evidence_anchor": 0.12,
            "social_weight": 2.1,
            "max_logodds": 6.0,
            "memory_decay": 0.55,
            "confidence_cap": 0.95,
        },
        "local_llm": {
            "model_path": str(model_path),
            "max_new_tokens": int(local_cfg.get("max_new_tokens", 48)),
            "temperature": float(local_cfg.get("temperature", 0.0)),
        },
    }
    local_out = out_dir / "local_llm_run"
    trial_df, aggregate_df = run_sweep(subset, local_out)
    aggregate_df.to_csv(csv_path, index=False)
    status = {
        "status": "completed",
        "trial_rows": len(trial_df),
        "aggregate_rows": len(aggregate_df),
        "output_dir": local_out.as_posix(),
        "csv": csv_path.as_posix(),
        "no_external_apis": True,
    }
    json_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def run_tests_record(out_dir: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    passed = None
    for line in combined_output.splitlines():
        line = line.strip()
        if line.startswith("Ran ") and " tests" in line:
            try:
                passed = int(line.split()[1])
            except (IndexError, ValueError):
                passed = None
    status = {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "tests_passed": passed if result.returncode == 0 else None,
        "output": combined_output.strip(),
    }
    (out_dir / "test_results.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def write_verify_script_copy(out_dir: Path) -> None:
    source = Path("verify_paper_numbers.py")
    target = out_dir / "verify_paper_numbers.py"
    if source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def format_verify_numbers(numbers: dict[str, Any], test_results: dict[str, Any] | None = None) -> str:
    lines = [
        f"no_defense_false_consensus={numbers['no_defense_false_consensus']:.12g}",
        f"combined_false_consensus={numbers['combined_false_consensus']:.12g}",
        f"combined_true_consensus={numbers['combined_true_consensus']:.12g}",
        f"hub_placement_false_consensus={numbers['hub_placement_false_consensus']:.12g}",
        f"low_degree_placement_false_consensus={numbers['low_degree_placement_false_consensus']:.12g}",
        f"random_placement_false_consensus={numbers['random_placement_false_consensus']:.12g}",
        f"clustered_placement_false_consensus={numbers['clustered_placement_false_consensus']:.12g}",
        f"five_percent_high_degree_false_consensus={numbers['five_percent_high_degree_false_consensus']:.12g}",
        f"twenty_percent_low_degree_false_consensus={numbers['twenty_percent_low_degree_false_consensus']:.12g}",
        f"outcome_accounting_max_error={numbers['outcome_accounting_max_error']:.12g}",
    ]
    if test_results and test_results.get("tests_passed") is not None:
        lines.append(f"tests_passed={test_results['tests_passed']}")
    else:
        lines.append("tests_passed=unavailable")
    return "\n".join(lines)


def verify_paper_numbers_main() -> None:
    parser = argparse.ArgumentParser(description="Print exact V2/V3 paper numbers.")
    parser.add_argument("--v2-dir", default="outputs/v2")
    parser.add_argument("--v3-dir", default="outputs/v3")
    args = parser.parse_args()
    v2 = _load_v2(args.v2_dir)
    numbers = _compute_paper_numbers(v2)
    test_path = Path(args.v3_dir) / "test_results.json"
    test_results = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None
    print(format_verify_numbers(numbers, test_results))


def write_report(
    numbers: dict[str, Any],
    sanity: dict[str, Any],
    local_status: dict[str, Any],
    test_results: dict[str, Any],
    figure_paths: list[Path],
    table_paths: dict[str, Path],
    out_dir: Path,
    report_path: Path,
) -> Path:
    supported = pd.DataFrame(
        [
            {"claim": "Committed false-belief minorities can induce false consensus in synthetic agent populations.", "support": f"No defense false consensus = {numbers['no_defense_false_consensus']:.3f}."},
            {"claim": "Network position can matter more than minority size.", "support": f"5% high-degree = {numbers['five_percent_high_degree_false_consensus']:.3f}; 20% low-degree = {numbers['twenty_percent_low_degree_false_consensus']:.3f}."},
            {"claim": "Inoculation can reduce false consensus while changing consensus formation.", "support": f"Combined false consensus = {numbers['combined_false_consensus']:.3f}; combined true consensus = {numbers['combined_true_consensus']:.3f}."},
        ]
    )
    unsupported = pd.DataFrame(
        [
            {"claim": "This models human social behavior.", "status": "Not supported; the benchmark studies artificial agents only."},
            {"claim": "The result generalizes to real deployed LLM societies.", "status": "Not supported without local/real LLM evidence; current local LLM status is skipped."},
            {"claim": "The combined defense is universally optimal.", "status": "Not supported; V2 reports benchmark-specific cost and no-consensus tradeoffs."},
        ]
    )
    checklist = pd.DataFrame(
        [
            {"item": "Figure 1 capture by placement", "status": "ready"},
            {"item": "Figure 2 hub amplification", "status": "ready"},
            {"item": "Figure 3 inoculation Pareto", "status": "ready"},
            {"item": "Figure 4 consensus tradeoff", "status": "ready"},
            {"item": "Figure 5 local LLM sanity", "status": "ready" if local_status.get("status") == "completed" else "skipped"},
            {"item": "LaTeX tables", "status": "ready"},
            {"item": "Paper skeleton", "status": "not created; no paper folder/template existed"},
        ]
    )
    lines = [
        "# Collective Belief Capture V3 Paper Readiness",
        "",
        "## Paper Thesis",
        "",
        "Errors in artificial agent populations can become collective rather than individual. Small committed false-belief minorities can induce false consensus, and network position can matter more than minority size. Inoculation mechanisms reduce false consensus while trading off against consensus formation.",
        "",
        "This is not a human social simulation paper. The evidence below is from synthetic hidden-state artificial-agent benchmarks unless explicitly marked as local LLM sanity evidence.",
        "",
        "## Main Claims Supported By V2/V3 Data",
        "",
        _markdown_table(supported),
        "",
        "## Claims Not Yet Supported",
        "",
        _markdown_table(unsupported),
        "",
        "## Paper Numbers",
        "",
        "```text",
        format_verify_numbers(numbers, test_results),
        "```",
        "",
        "## Figure Checklist",
        "",
        _markdown_table(checklist),
        "",
        "## Local LLM Sanity Status",
        "",
        f"Status: `{local_status.get('status')}`.",
        f"Reason/output: `{local_status.get('reason', local_status.get('output_dir', 'n/a'))}`.",
        "",
        "The local LLM sanity subset is not the main result. It should only be discussed if it runs with a configured local open-weight model. No external APIs are used.",
        "",
        "## Limitations",
        "",
        "- Scripted update rules are controlled stress-test agents, not real deployed agents.",
        "- The defense mechanisms depend on structured messages and cited evidence fields.",
        "- The strongest quarantine/combined results assume unsupported claims can be identified from message structure.",
        "- V3 currently does not provide local LLM evidence because the local dependency/model path check failed.",
        "- The benchmark uses neutral synthetic A/B hidden-state tasks only.",
        "",
        "## Recommended AAAI Title",
        "",
        "Collective Belief Capture in Artificial Agent Populations",
        "",
        "## Abstract Draft",
        "",
        "As AI systems increasingly communicate with other AI systems, errors may become collective rather than individual. We introduce a synthetic hidden-state benchmark for studying collective belief capture in artificial agent populations. Normal agents receive noisy private evidence about a binary state and exchange messages over configurable communication graphs, while a small committed minority asserts the false state. Across multiple scripted update rules, we find that false beliefs can become population-level consensus and that network position can amplify minority influence: in our hub-placement experiment, a 5% high-degree committed minority produced higher false consensus than a 20% low-degree minority. We evaluate lightweight inoculation mechanisms based on evidence anchoring, source diversity, and unsupported-claim quarantine, showing that they can reduce false consensus while changing true-consensus and no-consensus rates. The benchmark is synthetic, avoids real-world persuasion domains, and is intended as a controlled testbed for collective error dynamics in artificial agent societies.",
        "",
        "## Remaining Risks",
        "",
        "- Need a local open-weight LLM sanity run before making even weak LLM-specific claims.",
        "- Need careful wording around quarantine because it may be strong under the structured-message assumptions.",
        "- Need related work and citations added by the paper author; V3 intentionally fabricates none.",
        "- Need final figure styling to match the target AAAI template after manuscript integration.",
        "",
        "## Artifact Index",
        "",
        f"- Figures: `{(out_dir / 'plots').as_posix()}`",
        f"- Tables: `{(out_dir / 'tables').as_posix()}`",
        f"- Local LLM status: `{(out_dir / 'local_llm_sanity.json').as_posix()}`",
        f"- Verifier: `{(out_dir / 'verify_paper_numbers.py').as_posix()}`",
        f"- Test result record: `{(out_dir / 'test_results.json').as_posix()}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_v3(config: dict[str, Any], out_dir: str | Path, report_path: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, out / "config_snapshot.yaml")
    v2 = _load_v2(config.get("v2_dir", "outputs/v2"))
    sanity_path = Path(config.get("v2_dir", "outputs/v2")) / "sanity_checks.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8")) if sanity_path.exists() else {}
    numbers = _compute_paper_numbers(v2)
    local_status = maybe_run_local_llm_sanity(config, out)
    figure_paths = []
    figure_paths.extend(figure_capture_by_placement(v2, out))
    figure_paths.extend(
        figure_hub_amplification(
            v2,
            out,
            int(config.get("seed", 20260601)),
            int(config.get("bootstrap_iterations", 1000)),
            float(config.get("ci_level", 0.95)),
        )
    )
    figure_paths.extend(figure_inoculation_pareto(v2, out))
    figure_paths.extend(figure_consensus_tradeoff(v2, out))
    figure_paths.extend(figure_llm_sanity(out / "local_llm_sanity.csv", out))
    table_paths = write_tables(v2, sanity, out)
    test_results = run_tests_record(out)
    write_verify_script_copy(out)
    report = write_report(numbers, sanity, local_status, test_results, figure_paths, table_paths, out, Path(report_path))
    (out / "paper_numbers.json").write_text(json.dumps(_json_safe(numbers), indent=2), encoding="utf-8")
    (out / "figure_manifest.json").write_text(json.dumps([path.as_posix() for path in figure_paths], indent=2), encoding="utf-8")
    return {
        "figures": len(figure_paths),
        "tables": len(table_paths),
        "local_llm_status": local_status.get("status"),
        "report": report.as_posix(),
        "tests": test_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V3 paper-readiness artifacts from V2 data.")
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default="COLLECTIVE_BELIEF_CAPTURE_V3_PAPER_READINESS.md")
    parser.add_argument("--v2-dir", default=None)
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--skip-local-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.v2_dir is not None:
        config["v2_dir"] = args.v2_dir
    if args.out is not None:
        config["out_dir"] = args.out
    if args.local_model_path is not None:
        config["local_llm"]["model_path"] = args.local_model_path
    if args.skip_local_llm:
        config["local_llm"]["enabled"] = False
    out_dir = args.out or config.get("out_dir", "outputs/v3")
    result = run_v3(config, out_dir, args.report)
    print(f"figures: {result['figures']}")
    print(f"tables: {result['tables']}")
    print(f"local llm: {result['local_llm_status']}")
    print(f"tests: {result['tests']['status']} ({result['tests'].get('tests_passed')} passed)")
    print(f"report: {result['report']}")
