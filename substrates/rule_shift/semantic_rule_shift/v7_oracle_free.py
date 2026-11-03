"""V7: oracle-free probe selection vs matched-random under the REALISTIC monitor.

Every strategy runs through the paper's own online monitor ``simulate_symbolic_monitor``:
``eig_adaptive`` / ``worst_case`` are ORACLE-FREE adaptive selectors (they pick the next
probe from a posterior over CANDIDATE systems, never the realized target or the
distinguishing set), observe the true stream output, update Bayes, and the monitor alarms
only when the source posterior collapses. Detection delay is the real number of post-shift
steps to that alarm (a miss is penalized as the full post-shift window of 80).

Sub-experiments (all CPU):
  * spectrum   : advantage vs density at budget 8, well/misspecified, over the regimes plus
                 auto-discovered low-density cells. Reports per-strategy mean+std and, for
                 EIG vs matched-random, the PAIRED per-seed advantage (mean, std, k/N>0).
  * budget     : advantage vs probe budget b in {1,2,4,8,16} on representative cells.
  * phase      : on-grid vs off-grid shift (probe-grid floor check).
  * rho_est    : true rho vs the rho a monitor can ESTIMATE from its candidate set alone
                 (no ground truth), and the fraction of the true distinguishing set that
                 the candidate-estimated region covers -- which explains why the
                 information-gain selector fails under misspecification.

Run from project root: python -m semantic_rule_shift.v7_oracle_free [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .v3 import (
    FOUR_VALUES,
    SEMANTICS_V3,
    _with_binary_override,
    source_contrastive_probes,
)
from .v4 import (
    V4_REGIMES,
    build_formula_pools,
    projected_truth_vector,
    simulate_symbolic_monitor,
)

N_POST = 80
PROBE_INTERVAL = 8

REGIME_CELLS = [
    ("impossible_equivalent", "L2"),
    ("one_entry_sparse_l3", "L3"),
    ("sparse_hard_l2", "L3"),
    ("sparse_hard_l2", "L2"),
    ("language_unlocked_l2", "L3"),
    ("language_unlocked_l2", "L2"),
    ("easy_dense_four", "L0"),
    ("easy_dense_four", "L2"),
]

FLIP_CANDIDATES = [
    ("XOR", ("B", "B"), "T", "L3"),
    ("XOR", ("N", "N"), "T", "L3"),
    ("EQUIV", ("B", "N"), "F", "L3"),
    ("EQUIV", ("N", "B"), "F", "L3"),
    ("IMPLIES", ("B", "N"), "N", "L2"),
    ("IMPLIES", ("N", "N"), "B", "L2"),
    ("IMPLIES", ("B", "B"), "N", "L2"),
    ("AND", ("N", "N"), "B", "L2"),
    ("OR", ("B", "B"), "N", "L2"),
]

OPTIMIZED = ("eig_adaptive", "worst_case", "selected")
RANDOM = ("random_depth_matched", "random_candidate_matched")
ALL_STRATS = OPTIMIZED + RANDOM


def density_region(rho: float) -> str:
    if rho <= 0.0:
        return "zero"
    if rho < 0.05:
        return "low_nonzero"
    if rho < 0.25:
        return "medium"
    return "high"


def build_vectors(systems, formulas):
    return {name: [projected_truth_vector(sys, f) for f in formulas] for name, sys in systems.items()}


def rho_of(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) / max(len(a), 1)


def penalized(delay):
    return float(delay) if delay is not None else float(N_POST)


def stdev(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def regime_by_name(name):
    for r in V4_REGIMES:
        if r["regime"] == name:
            return r
    raise KeyError(name)


def make_cell(label, source_sys, target_sys, candidate_systems, formulas, lang):
    systems = dict(candidate_systems)
    systems[source_sys.name] = source_sys
    systems[target_sys.name] = target_sys
    vectors = build_vectors(systems, formulas)
    rho = rho_of(vectors[source_sys.name], vectors[target_sys.name])
    return {
        "label": label, "source": source_sys.name, "target": target_sys.name, "lang": lang,
        "candidate_names_full": list(systems.keys()), "vectors": vectors, "formulas": formulas,
        "rho": rho, "region": density_region(rho), "diagnosable": rho > 0.0,
    }


def regime_cells(pools):
    cells = []
    for regime_name, lang in REGIME_CELLS:
        reg = regime_by_name(regime_name)
        source, target = reg["source"], reg["target"]
        vs = SEMANTICS_V3[source].value_set
        formulas = pools[(vs, lang)]
        cand = {n: SEMANTICS_V3[n] for n, s in SEMANTICS_V3.items() if s.value_set == vs}
        cells.append(make_cell(regime_name, SEMANTICS_V3[source], SEMANTICS_V3[target], cand, formulas, lang))
    return cells


def discover_low_density_cells(pools, want=4):
    fde = SEMANTICS_V3["fde4"]
    four = {n: s for n, s in SEMANTICS_V3.items() if s.value_set == "four"}
    found = []
    for op, entry, value, lang in FLIP_CANDIDATES:
        name = f"flip_{op}_{entry[0]}{entry[1]}_{value}".lower()
        try:
            tgt = _with_binary_override(fde, name=name, op=op, entry=entry, value=value,
                                        description=f"sparse fde4 flip {op}{entry}={value}")
        except Exception:
            continue
        formulas = pools[("four", lang)]
        cell = make_cell(f"flip_{op.lower()}_{entry[0]}{entry[1]}", fde, tgt, dict(four), formulas, lang)
        if 0.0 < cell["rho"] < 0.05:
            found.append(cell)
        if len(found) >= want:
            break
    return found


def restrict(cell, condition):
    c = dict(cell)
    if condition == "misspecified":
        c["candidate_names_full"] = [n for n in cell["candidate_names_full"] if n != cell["target"]]
    return c


def run_cell(cell, strategy, budget, seed, n_pre):
    cand = cell["candidate_names_full"]
    formulas, vectors = cell["formulas"], cell["vectors"]
    selected_indices = source_contrastive_probes(cell["source"], cand, formulas, vectors)
    selected_depths = [formulas[i].depth for i in selected_indices]
    candidate_indices = [
        i for i in range(len(formulas))
        if any(vectors[cell["source"]][i] != vectors[n][i] for n in cand if n != cell["source"])
    ]
    res = simulate_symbolic_monitor(
        source=cell["source"], target=cell["target"], candidate_names=cand, formulas=formulas,
        vectors=vectors, selected_indices=selected_indices, candidate_indices=candidate_indices,
        selected_depths=selected_depths, strategy=strategy, budget=budget, seed=seed,
        diagnosable=cell["diagnosable"], n_pre=n_pre, n_post=N_POST, probe_interval=PROBE_INTERVAL,
    )
    return res["detection_delay"]


def aggregate(cell, condition, strategy, budget, seeds, n_pre):
    c = restrict(cell, condition)
    delays = [run_cell(c, strategy, budget, seed, n_pre) for seed in seeds]
    pens = [penalized(d) for d in delays]
    miss = statistics.mean(1.0 if d is None else 0.0 for d in delays)
    return statistics.mean(pens), miss


def run_spectrum(cells, seeds, budget=8, n_pre=80):
    """Per-strategy mean+std rows AND paired EIG-vs-random advantage stats (single seed pass)."""
    rows, adv_rows = [], []
    for cell in cells:
        for condition in ("well_specified", "misspecified"):
            c = restrict(cell, condition)
            per = {s: [] for s in ALL_STRATS}
            miss = {s: [] for s in ALL_STRATS}
            for seed in seeds:
                for s in ALL_STRATS:
                    d = run_cell(c, s, budget, seed, n_pre)
                    per[s].append(penalized(d))
                    miss[s].append(1.0 if d is None else 0.0)
            for s in ALL_STRATS:
                rows.append({
                    "label": cell["label"], "source": cell["source"], "target": cell["target"],
                    "language": cell["lang"], "rho": round(cell["rho"], 4), "region": cell["region"],
                    "condition": condition, "budget": budget, "n_pre": n_pre, "strategy": s,
                    "n_seeds": len(seeds), "mean_penalized_delay": round(statistics.mean(per[s]), 3),
                    "std_penalized_delay": round(stdev(per[s]), 3), "miss_rate": round(statistics.mean(miss[s]), 3),
                })
            # paired per-seed advantage: random_depth_matched - eig_adaptive
            paired = [r - e for r, e in zip(per["random_depth_matched"], per["eig_adaptive"])]
            adv_rows.append({
                "label": cell["label"], "language": cell["lang"], "rho": round(cell["rho"], 4),
                "region": cell["region"], "condition": condition, "n_seeds": len(seeds),
                "adv_mean": round(statistics.mean(paired), 3), "adv_std": round(stdev(paired), 3),
                "k_positive": sum(1 for a in paired if a > 0), "k_negative": sum(1 for a in paired if a < 0),
            })
    return rows, adv_rows


def budget_rows(cells, seeds, budgets):
    rows = []
    for cell in cells:
        for condition in ("well_specified", "misspecified"):
            for budget in budgets:
                for strat in ("eig_adaptive", "random_depth_matched"):
                    mean, miss = aggregate(cell, condition, strat, budget, seeds, 80)
                    rows.append({
                        "label": cell["label"], "rho": round(cell["rho"], 4), "region": cell["region"],
                        "condition": condition, "budget": budget, "strategy": strat,
                        "mean_penalized_delay": round(mean, 3), "miss_rate": round(miss, 3),
                    })
    return rows


def phase_rows(cells, seeds):
    rows = []
    for cell in cells:
        for n_pre, tag in ((80, "on_grid"), (84, "off_grid")):
            for strat in ("eig_adaptive", "random_depth_matched"):
                mean, miss = aggregate(cell, "well_specified", strat, 8, seeds, n_pre)
                rows.append({
                    "label": cell["label"], "rho": round(cell["rho"], 4), "phase": tag, "n_pre": n_pre,
                    "strategy": strat, "mean_penalized_delay": round(mean, 3), "miss_rate": round(miss, 3),
                })
    return rows


def rho_estimation_rows(cells):
    """True rho vs candidate-estimable rho (no ground truth); coverage of the true distinguishing set."""
    rows = []
    for cell in cells:
        formulas, vectors, src, tgt = cell["formulas"], cell["vectors"], cell["source"], cell["target"]
        true_idx = {i for i in range(len(formulas)) if vectors[src][i] != vectors[tgt][i]}
        for condition in ("well_specified", "misspecified"):
            cand = restrict(cell, condition)["candidate_names_full"]
            est_idx = {i for i in range(len(formulas))
                       if any(vectors[src][i] != vectors[n][i] for n in cand if n != src)}
            est_rho = len(est_idx) / max(len(formulas), 1)
            true_in_est = len(true_idx & est_idx) / max(len(true_idx), 1)
            rows.append({
                "label": cell["label"], "language": cell["lang"], "condition": condition,
                "true_rho": round(cell["rho"], 4), "candidate_est_rho": round(est_rho, 4),
                "true_distinguishing_covered_by_candidates": round(true_in_est, 4),
            })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/oracle_free")
    args = ap.parse_args()

    pools = build_formula_pools(max_depth=2)
    out = Path(args.out)
    low = discover_low_density_cells(pools, want=(1 if args.smoke else 4))
    print("low-density cells:", [(c["label"], c["lang"], round(c["rho"], 4)) for c in low])

    if args.smoke:
        seeds = [7, 18]
        spec, adv = run_spectrum(regime_cells(pools)[:2] + low, seeds)
        write_csv(out / "oracle_free_results.csv", spec)
        write_csv(out / "oracle_free_advstats.csv", adv)
        write_csv(out / "oracle_free_rho_estimation.csv", rho_estimation_rows(regime_cells(pools)[:2] + low))
        print(f"smoke: spectrum={len(spec)} adv={len(adv)}")
        for a in adv:
            print(" ADV", a["label"], a["condition"], a["adv_mean"], "+/-", a["adv_std"], f"k+={a['k_positive']}/{a['n_seeds']}")
        return

    seeds = [7, 18, 30, 41, 52, 63, 74, 85]
    spec_cells = regime_cells(pools) + low
    spec, adv = run_spectrum(spec_cells, seeds)
    write_csv(out / "oracle_free_results.csv", spec)
    write_csv(out / "oracle_free_advstats.csv", adv)
    write_csv(out / "oracle_free_rho_estimation.csv", rho_estimation_rows(spec_cells))

    reps = []
    for want_label, want_lang in (("one_entry_sparse_l3", "L3"), ("sparse_hard_l2", "L2"), ("easy_dense_four", "L2")):
        for c in spec_cells:
            if c["label"] == want_label and c["lang"] == want_lang:
                reps.append(c)
                break
    write_csv(out / "oracle_free_budget.csv", budget_rows(reps, seeds, [1, 2, 4, 8, 16]))
    low_l3 = [c for c in spec_cells if c["label"] == "one_entry_sparse_l3" and c["lang"] == "L3"]
    write_csv(out / "oracle_free_phase.csv", phase_rows(low + low_l3, seeds))

    print(f"\nwrote spectrum={len(spec)} adv={len(adv)} -> {out}")
    print("\nlow-density advantage (well-specified, b=8): mean +/- std (k+/N)")
    for a in adv:
        if a["region"] == "low_nonzero" and a["condition"] == "well_specified":
            print(f"  {a['label']:<26}{a['language']:<4} adv={a['adv_mean']:>5} +/- {a['adv_std']:<5} (k+={a['k_positive']}/{a['n_seeds']})")
    print("\nrho estimability (true vs candidate-estimated; coverage of true distinguishing set):")
    for r in rho_estimation_rows(spec_cells):
        if r["true_rho"] > 0:
            print(f"  {r['label']:<26}{r['condition']:<14} true={r['true_rho']:.3f} est={r['candidate_est_rho']:.3f} cover={r['true_distinguishing_covered_by_candidates']:.2f}")


if __name__ == "__main__":
    main()
