r"""P1 figures, drawn at their printed size (R11 redraw, 2026-11-03).

Every figure is drawn at the exact width it is printed at in the JAAMAS manuscript
(sn-jnl text block 130.7 mm; WIDTH below), with 8 pt Arial lettering, so LaTeX never
rescales the JAAMAS figures; the acmart build prints them at \linewidth (a few percent larger).
Colours are the Okabe-Ito set (validated for colour-vision deficiency with the dataviz
skill's validate_palette.js); every series also carries a marker or line-style difference.

  fig_windows        <- results/s1/review_s2/BENCH_L{0,1}_STD.jsonl, analysis.json
  fig_regime_map     <- results/s1/stats.json
  fig_latency        <- results/s1/strict_regime_map.json, results/s1/review_s2/analysis.json
  fig_s2_missrate    <- results/s2/stats.json
  fig_dissociation   <- results/s3/metrics_{qwen35,gemma4}.json
  fig_zoo            <- results/s4/zoo.json

Usage (from paper1_oversight/):  python -X utf8 figures/make_figs.py
Output: figures/final/*.pdf (+ .png previews), synced to paper/figures/ and jaamas/figures/.
A mechanical gate refuses to sync if any two text boxes overlap or a label covers a marker.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
P = HERE.parent
RES = P / "results"
OUT = HERE / "final"
sys.path.insert(0, str(P / "src"))

WIDTH = 372.0 / 72.27          # 130.7 mm, the measured sn-jnl text block
BLUE, VERM, SKY, ORANGE = "#0072B2", "#D55E00", "#56B4E9", "#E69F00"
INK, MUTED, RULE, LGRAY = "#222222", "#5f6368", "#bdbdbd", "#e3e3e0"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.titlesize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.handlelength": 1.6,
})

REGIME = [("placement_critical", "placement-critical", BLUE, "P"),
          ("budget_saturated", "budget-saturated", SKY, "S"),
          ("ceiling_bound", "ceiling-bound", VERM, "C"),
          ("intermediate", "intermediate", MUTED, "I"),
          ("benign", "benign", LGRAY, "")]
RCOLOR = {k: c for k, _, c, _ in REGIME}
RSHORT = {k: s for k, _, _, s in REGIME}
FAILS: list[str] = []


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def gate(fig, name):
    """No two text boxes may intersect, and no text may cover a data marker."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    items = []
    for t in list(fig.texts) + [t for ax in fig.axes for t in ax.texts]:
        if t.get_text().strip() and t.get_visible():
            items.append((t.get_text()[:24], t.get_window_extent(ren)))
    for leg in [ax.get_legend() for ax in fig.axes] + list(fig.legends):
        if leg is not None:
            items += [(t.get_text()[:24], t.get_window_extent(ren)) for t in leg.get_texts()]
    bad = [(a[0], b[0]) for i, a in enumerate(items) for b in items[i + 1:]
           if a[1].x0 < b[1].x1 - 1 and b[1].x0 < a[1].x1 - 1
           and a[1].y0 < b[1].y1 - 1 and b[1].y0 < a[1].y1 - 1]
    for ax in fig.axes:
        pts = []
        for ln in ax.lines:
            if ln.get_marker() in (None, "None", "", " ", "|", "_"):
                continue
            pts += [ax.transData.transform((float(x), float(y)))
                    for x, y in zip(ln.get_xdata(), ln.get_ydata())]
        for t in ax.texts:
            bb = t.get_window_extent(ren)
            bad += [(t.get_text()[:24], "marker") for (px, py) in pts
                    if bb.x0 - 1 < px < bb.x1 + 1 and bb.y0 - 1 < py < bb.y1 + 1]
    print(f"[{name}] collisions: {bad if bad else 'NONE'}")
    if bad:
        FAILS.append(name)


def save(fig, name):
    gate(fig, name)
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    w, h = fig.get_size_inches()
    print(f"[{name}] {w * 25.4:.1f} x {h * 25.4:.1f} mm")
    plt.close(fig)


def _jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# --------------------------------------------------------------------------------------
def fig_windows():
    """Prevention windows of every captured run in one windowed cell, at both latencies.

    The example cell is the windowed cell whose windows end in the most distinct rounds, chosen
    by src/review_s2_window_details.py (4 committed agents, evidence 0.74, hub); pooled numbers
    are in the text and in results/s1/review_s2/window_details.json.
    """
    from review_s2_runs import ALL_SCHEDS, SCHED_INDEX
    D = RES / "s1" / "review_s2"
    CELL = json.loads((D / "window_details.json").read_text(encoding="utf-8"))["example_cell"]["BENCH_L0_STD"]["cell"]
    runs = {}
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD"):
        for r in _jsonl(D / f"{arm}.jsonl"):
            if r["cell"] != CELL or not r["base_fc"]:
                continue
            W = sorted(min(s) for s in ALL_SCHEDS if len(s) == 1 and r["fc"][SCHED_INDEX[s]] == "0")
            runs.setdefault(r["trial"], {})[arm] = ((W[0], W[-1]) if W else None, r["capture_round"])
    trials = sorted(runs, key=lambda t: (-(runs[t]["BENCH_L0_STD"][0] or (0, 0))[1],
                                         runs[t]["BENCH_L0_STD"][1], t))
    n = len(trials)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, arm, lab in ((axes[0], "BENCH_L0_STD", "zero latency"),
                         (axes[1], "BENCH_L1_STD", "one round of latency")):
        hit = {2: 0, "uni": 0, 3: 0}
        for i, t in enumerate(trials):
            y = n - 1 - i
            w, cap = runs[t][arm]
            if w:
                ax.plot([w[0] - 0.42, w[1] + 0.42], [y, y], color=SKY, lw=2.2, solid_capstyle="butt", zorder=1)
                hit[2] += w[0] <= 2 <= w[1]
                hit[3] += w[0] <= 3 <= w[1]
                hit["uni"] += any(w[0] <= c <= w[1] for c in (5, 10))
            ax.plot(cap, y, marker="|", ms=4.5, mew=1.0, color=INK, ls="", zorder=4)
        ax.plot([2, 2], [-0.8, n - 0.2], color=BLUE, lw=1.3, zorder=3)
        for c in (5, 10):
            ax.plot([c, c], [-0.8, n - 0.2], color=VERM, lw=1.1, ls=(0, (3, 2)), zorder=3)
        top = n + 0.3
        ax.text(2, top, f"{hit[2]}", color=BLUE, ha="center", va="bottom", fontweight="bold")
        ax.text(7.5, top, f"{hit['uni']}", color=VERM, ha="center", va="bottom", fontweight="bold")
        ax.set_xlim(0.5, 15.5)
        ax.set_ylim(-1, n + 2.6)
        ax.set_xticks(range(1, 16))
        ax.set_xticklabels([str(x) if x in (1, 2, 3, 5, 10, 15) else "" for x in range(1, 16)])
        ax.set_xlabel("round")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(lab, loc="left", pad=12, color=INK)
        nw = sum(1 for t in trials if runs[t][arm][0])
        print(f"  {CELL} {arm}: {n} captured, {nw} with a window, prevented by round 2: {hit[2]}, "
              f"round 3: {hit[3]}, rounds 5+10: {hit['uni']}")
    panel_label(axes[0], "a", x=-0.03, y=1.06)
    panel_label(axes[1], "b", x=-0.03, y=1.06)
    axes[0].set_ylabel(f"captured runs ({n} of 50)")
    handles = [Line2D([], [], color=SKY, lw=3.0, label="prevention window"),
               Line2D([], [], marker="|", ls="", ms=6, mew=1.0, color=INK, label="capture round"),
               Line2D([], [], color=BLUE, lw=1.3, label="audit-early check (round 2)"),
               Line2D([], [], color=VERM, lw=1.1, ls=(0, (3, 2)), label="two uniform checks")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.0, handletextpad=0.4, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    save(fig, "fig_windows")


# --------------------------------------------------------------------------------------
def fig_regime_map():
    stats = json.loads((RES / "s1" / "stats.json").read_text(encoding="utf-8"))
    cells = stats["cells"]
    topos = ["hub", "random", "small_world", "clustered"]
    counts = [1, 2, 3, 4, 6]
    windowed = {w["cell"] for w in stats["h2"]["cells"]}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.05), sharey=True)
    for pi, (ax, ev) in enumerate(zip(axes, [0.58, 0.66, 0.74])):
        for yi, topo in enumerate(topos):
            for xi, cc in enumerate(counts):
                key = f"{cc}|{ev}|{topo}"
                reg = cells[key]["regime"] if key in cells else "benign"
                ax.add_patch(plt.Rectangle((xi + 0.04, yi + 0.04), 0.92, 0.92, facecolor=RCOLOR[reg],
                                           edgecolor="none"))
                if RSHORT[reg]:
                    ax.text(xi + 0.5, yi + 0.5, RSHORT[reg], ha="center", va="center",
                            color="white", fontweight="bold")
                if key in windowed:
                    ax.plot(xi + 0.82, yi + 0.2, marker="o", ms=3.2, mfc="white", mec="white", mew=0)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 4)
        ax.set_xticks([i + 0.5 for i in range(5)])
        ax.set_xticklabels(counts)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"evidence {ev}", pad=2)
        panel_label(ax, "abc"[pi], x=0.0)
    axes[0].set_yticks([i + 0.5 for i in range(4)])
    axes[0].set_yticklabels(["hub", "random", "small-world", "clustered"])
    axes[1].set_xlabel("committed agents (of 32)")
    handles = [Patch(fc=RCOLOR[k], label=f"{n} ({RSHORT[k]})" if RSHORT[k] else n)
               for k, n, _, _ in REGIME if k != "intermediate"]
    handles.append(Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=MUTED, mew=0.6,
                          label="windowed cell"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               handlelength=0.9, columnspacing=0.8, handletextpad=0.35, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
    save(fig, "fig_regime_map")


# --------------------------------------------------------------------------------------
def fig_latency():
    d = json.loads((RES / "s1" / "strict_regime_map.json").read_text(encoding="utf-8"))
    A = json.loads((RES / "s1" / "review_s2" / "analysis.json").read_text(encoding="utf-8"))
    one_msg = [A["regimes"][f"{det}_L{l}_STD"]["counts"] for det in ("SINGLE", "PERFECT") for l in (0, 1)]
    assert all(c == one_msg[0] for c in one_msg), "one-message detectors differ across latency"
    rows = [("benchmark, zero latency", d["h3_regimes"]["intercept"]["counts"]),
            ("benchmark, one round", d["h3_regimes"]["strict"]["counts"]),
            ("benchmark, one round,\ncheck may move a round", d["h3_latency_aware_library"]["strict"]["counts"]),
            ("perfect or single-message,\neither latency", one_msg[0])]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 2.75), gridspec_kw={"width_ratios": [1.0, 0.85]})
    for i, (lab, cnt) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0
        for k, _, col, short in REGIME:
            v = cnt.get(k, 0)
            if not v:
                continue
            ax.barh(y, v - 0.35, left=left, height=0.62, color=col, edgecolor="none")
            if v >= 5:
                ax.text(left + (v - 0.35) / 2, y, str(v), ha="center", va="center",
                        color=INK if k == "benign" else "white", fontsize=7.5)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("cells (of 60)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    panel_label(ax, "a", x=0.0)
    handles = [Patch(fc=c, label=n) for k, n, c, _ in REGIME]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               handlelength=0.9, columnspacing=0.8, handletextpad=0.35, fontsize=7.5)

    ci = d["h2_placement_vs_budget"]["intercept"]["cells"]
    cs = {c["cell"]: c for c in d["h2_placement_vs_budget"]["strict"]["cells"]}
    ci = sorted(ci, key=lambda c: c["delta_timely"])
    for y, c in enumerate(ci):
        z, o = c["delta_timely"], cs[c["cell"]]["delta_timely"]
        bx.plot([o, z], [y, y], color=RULE, lw=0.9, zorder=1)
        bx.plot(z, y, "o", ms=4.2, color=BLUE, mec="white", mew=0.6, zorder=3)
        bx.plot(o, y, "s", ms=3.9, color=VERM, mec="white", mew=0.6, zorder=3)
    bx.axvline(0, color=MUTED, lw=0.6, zorder=0)
    bx.set_yticks(range(len(ci)))
    bx.set_yticklabels([c["cell"].replace("|", " / ").replace("_", "-") for c in ci], fontsize=7.5)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(-0.4, 1.0)
    bx.set_xticks([-0.4, 0, 0.5, 1.0])
    bx.set_xticklabels(["−0.4", "0", "0.5", "1"])
    bx.set_xlabel("timely-catch difference")
    panel_label(bx, "b", x=0.0)
    top = ci[-1]
    bx.text(top["delta_timely"], len(ci) - 0.35, "zero latency", ha="right", va="bottom", color=INK)
    bx.text(cs[top["cell"]]["delta_timely"] - 0.03, len(ci) - 0.35, "one round", ha="right", va="bottom",
            color=INK)
    bx.set_ylim(-0.6, len(ci) + 0.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    save(fig, "fig_latency")


# --------------------------------------------------------------------------------------
def fig_s2_missrate():
    stats = json.loads((RES / "s2" / "stats.json").read_text(encoding="utf-8"))
    budgets = [1, 2, 4, 8, 16]
    series = [("front", VERM, (0, (3, 2)), "front-loaded", "s", 4.2),
              ("geometric", ORANGE, "-", "audit-early", "^", 4.6),
              ("random", MUTED, "-", "random", "D", 3.6),
              ("uniform", BLUE, "-", "uniform", "o", 4.0),
              ("back", SKY, "-", "back-loaded", "v", 4.6),
              ("oracle", INK, (0, (1, 1.5)), "onset-informed reference", "P", 4.2)]

    def pooled(pid, B, field):
        vals = [v[f"{pid}@B{B}"][field] for k, v in stats["cells"].items()
                if k.endswith("|well_specified") and f"{pid}@B{B}" in v
                and not k.startswith("impossible_equivalent")]
        return sum(vals) / len(vals)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    for pid, col, ls, lab, mk, ms in series:
        kw = dict(color=col, ls=ls, lw=1.3, marker=mk, ms=ms, mfc="white", mew=1.0, mec=col, label=lab,
                  zorder=4 if pid in ("back", "oracle") else 3)
        a1.plot(budgets, [pooled(pid, B, "miss") for B in budgets], **kw)
        a2.plot(budgets, [pooled(pid, B, "mean_pen") for B in budgets], **kw)
    for ax, ylab, lim, ticks in ((a1, "miss rate", (-0.04, 1.06), [0, 0.25, 0.5, 0.75, 1.0]),
                                 (a2, "penalized delay (steps)", (-3, 100), [0, 25, 50, 75, 100])):
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels(budgets)
        ax.minorticks_off()
        ax.set_xlabel("checks per run (budget $B$)")
        ax.set_ylabel(ylab)
        ax.set_ylim(*lim)
        ax.set_yticks(ticks)
        ax.grid(axis="y", color=LGRAY, lw=0.5)
        ax.set_axisbelow(True)
    panel_label(a1, "a")
    panel_label(a2, "b")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=2.2,
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=2.0)
    save(fig, "fig_s2_missrate")


# --------------------------------------------------------------------------------------
def fig_dissociation():
    rows = []
    for fam, flab in (("qwen35", "qwen3.5"), ("gemma4", "gemma-4")):
        m = json.loads((RES / "s3" / f"metrics_{fam}.json").read_text(encoding="utf-8"))
        for adv in ("plain", "jitter"):
            b = m["by_adv"][adv]
            rows.append((f"{flab}, {adv}", b["posthoc_AUC"], b["posthoc_AUC_ci95"],
                         b["timely_recall_at_C"], b["timely_recall_ci95"]))
    fig, ax = plt.subplots(figsize=(0.72 * WIDTH, 2.25))
    for i, (lab, auc, auc_ci, rec, rec_ci) in enumerate(rows):
        y = len(rows) - 1 - i
        ya, yr = y + 0.14, y - 0.14
        ax.plot(auc_ci, [ya, ya], color=BLUE, lw=1.0, zorder=2)
        ax.plot(auc, ya, "o", ms=4.6, mfc="white", mec=BLUE, mew=1.2, zorder=3)
        ax.plot(rec_ci, [yr, yr], color=VERM, lw=1.0, zorder=2)
        ax.plot(rec, yr, "s", ms=4.2, color=VERM, mec="white", mew=0.5, zorder=3)
        ax.text(1.20, y, f"{auc:.2f}", ha="right", va="center", color=INK, clip_on=False)
        ax.text(1.42, y, f"{rec:.2f}", ha="right", va="center", color=INK, clip_on=False)
    ax.text(1.20, len(rows) - 0.45, "AUC", ha="right", va="bottom", color=MUTED, clip_on=False)
    ax.text(1.42, len(rows) - 0.45, "recall", ha="right", va="bottom", color=MUTED, clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score (95% CI)")
    ax.grid(axis="x", color=LGRAY, lw=0.5)
    ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", lw=1.0, ms=4.6, mfc="white", mec=BLUE, mew=1.2,
                              color=BLUE, label="transcript AUC"),
                       Line2D([], [], marker="s", ls="-", lw=1.0, ms=4.2, color=VERM, mec="white",
                              label="timely recall")],
              loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=1.2)
    fig.subplots_adjust(left=0.25, right=0.70, top=0.92, bottom=0.33)
    save(fig, "fig_dissociation")


# --------------------------------------------------------------------------------------
MONITOR_NAMES = {  # identical to the row names of Table tab:zoo
    "deployed_442": "benchmark detector ($0.4\,s_1 + 0.4\,s_2$)",
    "s1_only": "provenance term $s_1$", "s2_only": "behavioral term $s_2$",
    "s1s2_equal": "$0.5\,s_1 + 0.5\,s_2$", "w622": "$0.6\,s_1 + 0.2\,s_2$",
    "w262": "$0.2\,s_1 + 0.6\,s_2$", "conf_mean": "mean confidence",
    "conf_constancy": "confidence constancy", "claim_constancy": "claim constancy",
    "high_conf_share": "high-confidence share", "round1_conf": "round-1 confidence",
    "evidence_rate": "evidence rate", "s1_raw_reuse": "raw signature reuse",
}


def fig_zoo():
    """Transcript AUC and deadline AUC of every monitor, one row each, in the order of Table tab:zoo."""
    zoo = json.loads((RES / "s4" / "zoo.json").read_text(encoding="utf-8"))
    mons = zoo["monitors"]
    order = zoo["rank_posthoc"]
    fig, ax = plt.subplots(figsize=(0.8 * WIDTH, 3.4))
    n = len(order)
    h = 0.36
    for i, k in enumerate(order):
        y = n - 1 - i
        ax.barh(y + h / 2 + 0.02, mons[k]["posthoc_auc"], height=h, color=BLUE, edgecolor="none")
        ax.barh(y - h / 2 - 0.02, mons[k]["timely_auc"], height=h, color=VERM, edgecolor="none")
    ax.axvline(0.5, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([MONITOR_NAMES[k] for k in order][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("AUC")
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="x", color=LGRAY, lw=0.5)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(fc=BLUE, label="completed transcript"),
                        Patch(fc=VERM, label="deadline (one round of latency)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=1.2,
               handlelength=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_zoo")


if __name__ == "__main__":
    for fn in (fig_windows, fig_regime_map, fig_latency, fig_s2_missrate, fig_dissociation, fig_zoo):
        fn()
    if FAILS:
        raise SystemExit(f"GATE FAILED, figures not synced: {FAILS}")
    for dest in (P / "paper" / "figures", P / "jaamas" / "figures"):
        if not dest.parent.exists():  # the public repository ships jaamas/ only
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(OUT.glob("fig_*.pdf")):
            shutil.copy2(f, dest / f.name)
    print("synced", sorted(f.name for f in OUT.glob("fig_*.pdf")))
