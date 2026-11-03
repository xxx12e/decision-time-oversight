from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .defenses import DEFENSE_LABELS, NO_DEFENSE


def _max_strength_slice(aggregate: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for defense, subset in aggregate.groupby("defense"):
        if defense == NO_DEFENSE:
            frames.append(subset[subset["defense_strength"] == subset["defense_strength"].min()])
        else:
            frames.append(subset[subset["defense_strength"] == subset["defense_strength"].max()])
    return pd.concat(frames, ignore_index=True)


def _label(defense: str) -> str:
    return DEFENSE_LABELS.get(defense, defense.replace("_", " ").title())


def make_plots(aggregate: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    strong = _max_strength_slice(aggregate)

    plt.figure(figsize=(10, 6))
    plot_data = (
        strong.groupby(["minority_ratio", "defense"], as_index=False)["false_consensus_rate"]
        .mean()
        .sort_values("minority_ratio")
    )
    for defense, subset in plot_data.groupby("defense"):
        subset = subset.sort_values("minority_ratio")
        plt.plot(subset["minority_ratio"] * 100, subset["false_consensus_rate"], marker="o", linewidth=1.8, label=_label(defense))
    plt.xlabel("Committed false-belief minority (%)")
    plt.ylabel("False consensus rate")
    plt.title("False consensus vs committed minority ratio")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    path = plot_dir / "false_consensus_vs_minority_ratio.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    plt.figure(figsize=(10, 6))
    topology_data = strong.groupby(["graph_topology", "defense"], as_index=False)["false_consensus_rate"].mean()
    pivot = topology_data.pivot(index="graph_topology", columns="defense", values="false_consensus_rate").fillna(0)
    ordered_cols = [col for col in DEFENSE_LABELS if col in pivot.columns]
    pivot = pivot[ordered_cols]
    x = range(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for offset, defense in enumerate(pivot.columns):
        positions = [value - 0.4 + width / 2 + offset * width for value in x]
        plt.bar(positions, pivot[defense], width=width, label=_label(defense))
    plt.xticks(list(x), [item.replace("_", " ") for item in pivot.index], rotation=20, ha="right")
    plt.ylabel("False consensus rate")
    plt.title("False consensus vs network topology")
    plt.ylim(0, 1.02)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    path = plot_dir / "false_consensus_vs_network_topology.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    plt.figure(figsize=(10, 6))
    strength_data = aggregate[aggregate["defense"] != NO_DEFENSE]
    strength_data = strength_data.groupby(["defense_strength", "defense"], as_index=False)["false_consensus_rate"].mean()
    for defense, subset in strength_data.groupby("defense"):
        subset = subset.sort_values("defense_strength")
        plt.plot(subset["defense_strength"], subset["false_consensus_rate"], marker="o", linewidth=1.8, label=_label(defense))
    plt.xlabel("Defense strength")
    plt.ylabel("False consensus rate")
    plt.title("Defense strength vs false consensus")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    path = plot_dir / "defense_strength_vs_false_consensus.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    plt.figure(figsize=(8, 7))
    tradeoff = (
        strong.groupby("defense", as_index=False)
        .agg(
            true_consensus_rate=("true_consensus_rate", "mean"),
            false_consensus_rate=("false_consensus_rate", "mean"),
            intervention_cost=("intervention_cost", "mean"),
        )
        .sort_values("false_consensus_rate")
    )
    sizes = 80 + 900 * tradeoff["intervention_cost"].clip(lower=0, upper=1)
    plt.scatter(tradeoff["false_consensus_rate"], tradeoff["true_consensus_rate"], s=sizes, alpha=0.65)
    for _, row in tradeoff.iterrows():
        plt.annotate(_label(row["defense"]), (row["false_consensus_rate"], row["true_consensus_rate"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    plt.xlabel("False consensus rate")
    plt.ylabel("True consensus rate")
    plt.title("True consensus / false consensus tradeoff")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    path = plot_dir / "true_false_consensus_tradeoff.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    return paths
