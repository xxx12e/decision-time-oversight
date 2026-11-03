r"""Variance components for the H2 contrast.

Statistical dependence across cells and seeds is complex, and a cell-level sign
test on twelve windowed cells does not by itself show how the variance divides
between cells and seeds; this hierarchical decomposition does.

The registered test is an exact sign test over cells, and the reported interval
is a paired bootstrap over the same twelve cells -- both already treat the CELL
as the unit, which is the conservative choice. What neither shows is how the
variance divides. This fits a one-way random-effects decomposition of the
per-seed paired delta d_{cs} = 1{best-B1 timely} - 1{uniform-B2 timely}:

    d_{cs} = mu + a_c + e_{cs},    a_c ~ (0, sigma_a^2),  e_{cs} ~ (0, sigma_e^2)

by the standard ANOVA moment estimator, and reports sigma_a (between cells),
sigma_e (between seeds within a cell), the intraclass correlation, the design
effect, and the effective number of independent observations. It also gives the
random-effects standard error of mu for comparison with the cell bootstrap the
paper quotes, so a reader can see the two agree rather than take it on faith.

Output: ../results/s1/variance_components.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "s1"


def decompose(groups: dict[str, list[float]]) -> dict:
    """One-way random effects by ANOVA moments; balanced or not."""
    k = len(groups)
    ns = [len(v) for v in groups.values()]
    n_tot = sum(ns)
    means = {g: sum(v) / len(v) for g, v in groups.items()}
    grand = sum(sum(v) for v in groups.values()) / n_tot
    ss_b = sum(len(v) * (means[g] - grand) ** 2 for g, v in groups.items())
    ss_w = sum((x - means[g]) ** 2 for g, v in groups.items() for x in v)
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (n_tot - k)
    # n0: the balanced-equivalent group size
    n0 = (n_tot - sum(n * n for n in ns) / n_tot) / (k - 1)
    var_a = max(0.0, (ms_b - ms_w) / n0)
    var_e = ms_w
    icc = var_a / (var_a + var_e) if (var_a + var_e) > 0 else 0.0
    nbar = n_tot / k
    deff = 1 + (nbar - 1) * icc
    # SE of the grand mean under the random-effects model = SE of the cell means
    cell_means = list(means.values())
    mu = sum(cell_means) / k
    se_re = sqrt(sum((m - mu) ** 2 for m in cell_means) / (k * (k - 1)))
    return {"n_cells": k, "n_obs": n_tot, "mean": mu,
            "sigma_between_cells": sqrt(var_a), "sigma_within_cell": sqrt(var_e),
            "icc": icc, "design_effect": deff,
            "effective_n": n_tot / deff if deff else None,
            "se_random_effects": se_re,
            "se_if_seeds_were_independent": sqrt(var_a + var_e) / sqrt(n_tot),
            "naive_se_understates_by_factor": (
                se_re / (sqrt(var_a + var_e) / sqrt(n_tot))
                if var_a + var_e > 0 else None)}


def main():
    rows = [json.loads(x) for x in (RES / "trials.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    stats = json.loads((RES / "stats.json").read_text(encoding="utf-8"))
    windowed = {w["cell"]: w["best_B1_policy"] for w in stats["h2"]["cells"]}

    by = defaultdict(dict)
    for r in rows:
        cell = f"{r['cell_committed']}|{r['cell_evidence']}|{r['cell_topology']}"
        if cell in windowed:
            by[cell][(r["policy"], r["budget"], r["trial"])] = r

    out = {"note": "one-way random effects on the per-seed paired delta; the cell "
                   "is the grouping factor. The registered sign test and the "
                   "paired bootstrap the paper reports both use the cell as the "
                   "unit already -- this quantifies what that choice costs and "
                   "why the seed-level n is not the effective n.",
           "contrasts": {}}

    for opp_B in (2, 3):
        for endpoint, field, sign in (("timely", "timely_committed_prune", 1),
                                      ("prevention", "false_consensus", -1)):
            groups = {}
            for cell, best in windowed.items():
                ds = []
                for t in range(50):
                    b = by[cell].get((best, 1, t))
                    u = by[cell].get(("uniform", opp_B, t))
                    if b is None or u is None:
                        continue
                    ds.append(sign * (int(b[field]) - int(u[field])))
                if ds:
                    groups[cell] = ds
            out["contrasts"][f"vs_uniform@B{opp_B}|{endpoint}"] = decompose(groups)

    (RES / "variance_components.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["contrasts"].items():
        print(f"{k:28s} mean={v['mean']:+.4f} sigma_cell={v['sigma_between_cells']:.3f} "
              f"sigma_seed={v['sigma_within_cell']:.3f} ICC={v['icc']:.3f} "
              f"deff={v['design_effect']:.1f} n_eff={v['effective_n']:.1f} "
              f"SE_re={v['se_random_effects']:.4f} "
              f"(naive SE understates x{v['naive_se_understates_by_factor']:.1f})")


if __name__ == "__main__":
    main()
