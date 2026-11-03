r"""Study 1 regime map recomputed under the STRICT timing convention
(answers: "why is there no strict-convention regime map?").

The paper's Study-1 grid is computed under the INTERCEPT convention: a check at
round t scores everything produced through round t and prunes before round-t
beliefs update (zero overseer latency). Studies 3-4 already report both
conventions and show that one round of reading latency changes the monitor
ranking; the reviewer's point is that the regime map -- the paper's first
headline -- was never re-derived under the stricter one.

This script recomputes the ENTIRE Study-1 sweep under both conventions with one
harness, so the only thing that differs between the two arms is the information
available to the overseer:

  INTERCEPT  check at round t -> anomaly_scores(log[1..t],   t)    then prune
  STRICT     check at round t -> anomaly_scores(log[1..t-1], t-1)  then prune

The ACTION timing is held fixed in both arms (the prune still takes effect
before round-t beliefs update). Only the overseer's reading window moves, which
is exactly "one round of overseer reading latency" and exactly the offset
src/timing_conventions.py applies for S3/S4. Implemented by rebinding
``s1_engine.anomaly_scores`` to a truncating wrapper, so the substrate itself
(s1_engine.s1_trial) is reused verbatim and never edited.

The online suspicion policy is lagged in the same way -- at round t it acts on
the round-(t-1) mean outbound confidence -- and its trigger delta is
re-calibrated on the clean (0-committed) arms only, over the window a strict
policy could actually act on. No policy sees a label, a capture round it could
not observe, or a future round; the only oracle arm is the paper's own
onset-heuristic reference, kept identical in both arms.

Everything downstream (regime classification, H2 placement-vs-budget contrast,
sign tests, bootstrap CIs) replicates src/stats_s1.py and
scripts/verify_p1_numbers.py line for line, including their seeds, so the
intercept arm is a byte-level reproduction check on the harness.

Usage:
  python -X utf8 strict_regime_map.py --probe 40      # timing probe
  python -X utf8 strict_regime_map.py --workers 20    # full run

Output: ../results/s1/strict_regime_map.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s1_engine  # noqa: E402
from s1_engine import (  # noqa: E402
    T_ROUNDS, make_condition, s1_trial,
    sched_uniform, sched_random, sched_front, sched_back, sched_geometric, sched_oracle,
)

RES = Path(__file__).resolve().parent.parent / "results" / "s1"
OUT_F = RES / "strict_regime_map.json"

# ---- grid: identical to sweep_s1.py (PREREG_P1 S1) ----
COUNTS = (1, 2, 3, 4, 6)
EVIDENCE = (0.58, 0.66, 0.74)
TOPOS = ("hub", "random", "small_world", "clustered")
BUDGETS = (1, 2, 3)
N_SEEDS = 50
RAND_SEED_BASE = 20260810          # sweep_s1.RAND_SEED_BASE (random-policy schedules)
POLICIES = ("uniform", "random", "front", "back", "geometric", "suspicion", "oracle")
NONORACLE = ("uniform", "random", "front", "back", "geometric", "suspicion")
CLEAN_POLICIES = ("uniform", "geometric", "suspicion")
CONVENTIONS = ("intercept", "strict")
ROUNDS = tuple(range(2, T_ROUNDS + 1))   # candidate check rounds (true_oracle_s1.ROUNDS)

# ---- statistics seeds: reused verbatim from the repo so the intercept arm
# reproduces the published numbers exactly ----
BOOT_SEED_CELL = 20260810          # stats_s1.BOOT_SEED  (per-cell CIs, 10k)
BOOT_N_CELL = 10_000
BOOT_SEED_AGG = 20260808           # verify_p1_numbers aggregate CI seed, 20k
BOOT_N_AGG = 20_000

# =====================================================================
# timing convention: the ONLY thing that differs between the two arms
# =====================================================================
_ORIG_ANOMALY = s1_engine.anomaly_scores


def _strict_anomaly(public_log, current_round):
    """STRICT: a check at round t reads only rounds 1..t-1."""
    r = current_round - 1
    if r < 1:
        return {int(s): 0.0 for s in public_log}
    trunc = {s: [m for m in hist if m.round_index <= r] for s, hist in public_log.items()}
    return _ORIG_ANOMALY(trunc, r)


def _set_convention(conv: str) -> None:
    s1_engine.anomaly_scores = _strict_anomaly if conv == "strict" else _ORIG_ANOMALY


class StrictSuspicionPolicy:
    """s1_engine.SuspicionPolicy with one round of reading latency: the decision
    at round t is taken on the round-(t-1) mean outbound confidence, and the
    round-1 reading is what fixes the baseline (first available at round 2)."""

    def __init__(self, budget: int, delta: float):
        self.budget = budget
        self.delta = delta
        self.baseline: float | None = None
        self.prev: float | None = None

    def decide(self, round_index: int, mean_conf: float) -> bool:
        visible, self.prev = self.prev, mean_conf     # round t sees round t-1
        if visible is None:                           # t == 1: nothing readable
            return False
        if self.baseline is None:                     # t == 2: fix baseline
            self.baseline = visible
            return False
        if self.budget <= 0:
            return False
        if visible - self.baseline >= self.delta:
            self.budget -= 1
            return True
        return False


def _make_policy(conv: str, budget: int, delta: float):
    if conv == "strict":
        return StrictSuspicionPolicy(budget, delta)
    return s1_engine.SuspicionPolicy(budget, delta)


# =====================================================================
# workers
# =====================================================================
def _run_baseline(job):
    """Baseline arms never fire a check, so they are convention-invariant."""
    cc, ev, topo, t = job
    _set_convention("intercept")
    r = s1_trial(make_condition(cc, ev, topo), t, frozenset())
    return {"trial": t, "capture_round": r["capture_round"],
            "false_consensus": bool(r["false_consensus"]),
            "mean_conf_by_round": r["mean_conf_by_round"]}


def _run_single_round(job):
    """Exhaustive B=1 placement: every candidate check round 2..T, so the
    placement-beats-budget contrast can be re-run against the MOST generous
    single check that exists, not just the six heuristic schedules. Answers the
    rebuttal 'the reversal is only because your policy library was tuned for
    zero latency'."""
    conv, cc, ev, topo, t, r = job
    _set_convention(conv)
    res = s1_trial(make_condition(cc, ev, topo), t, frozenset({r}))
    return {"conv": conv, "cell": f"{cc}|{ev}|{topo}", "trial": t, "round": r,
            "false_consensus": bool(res["false_consensus"]),
            "timely": bool(res["timely_committed_prune"])}


def _run_one(job):
    conv, cc, ev, topo, t, pid, B, delta, cap = job
    _set_convention(conv)
    cond = make_condition(cc, ev, topo)
    if pid == "suspicion":
        r = s1_trial(cond, t, frozenset(), policy=_make_policy(conv, B, delta))
    else:
        sched = {
            "uniform": lambda: sched_uniform(B),
            "random": lambda: sched_random(B, RAND_SEED_BASE + t * 100 + B),
            "front": lambda: sched_front(B),
            "back": lambda: sched_back(B),
            "geometric": lambda: sched_geometric(B),
            "oracle": lambda: sched_oracle(B, cap),
        }[pid]()
        r = s1_trial(cond, t, sched)
    return {"conv": conv, "cell": f"{cc}|{ev}|{topo}", "trial": t, "policy": pid,
            "budget": B, "false_consensus": bool(r["false_consensus"]),
            "timely": bool(r["timely_committed_prune"]),
            "intervened": bool(r["prune_events"]),
            "n_checks": len(r["checked_rounds"])}


# =====================================================================
# statistics helpers (verbatim recipes from stats_s1.py / verify_p1_numbers.py)
# =====================================================================
def sign_test_p(k_pos: int, n: int) -> float:
    return sum(math.comb(n, k) for k in range(k_pos, n + 1)) / 2 ** n if n else 1.0


def boot_ci_cell(values, n_boot=BOOT_N_CELL, seed=BOOT_SEED_CELL):
    """stats_s1.boot_ci"""
    rng = random.Random(seed)
    pt = sum(values) / len(values)
    bs = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return [pt, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]]


def boot_ci_agg(rng, ds, n_boot=BOOT_N_AGG):
    """verify_p1_numbers aggregate CI: shared rng, drawn in a fixed endpoint order."""
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(n_boot))
    return [sum(ds) / len(ds), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]]


def classify(fc_base, pol):
    """PREREG H3 regime rule, verbatim from stats_s1.main()."""
    oracle_any = max(pol[f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS)
    uni1 = pol["uniform@B1"]["fc_prevention"]
    best1_val, best1_pid = max((pol[f"{p}@B1"]["fc_prevention"], p) for p in NONORACLE)
    uni23 = max(pol[f"uniform@B{B}"]["fc_prevention"] for B in (2, 3))
    if fc_base < 0.10:
        regime = "benign"
    elif oracle_any <= 0.05:
        regime = "ceiling_bound"
    elif best1_val >= 2 * max(uni1, 0.02) and best1_val >= 0.10:
        regime = "placement_critical"
    elif uni23 >= oracle_any - 0.05:
        regime = "budget_saturated"
    else:
        regime = "intermediate"
    return regime, {"oracle_any": oracle_any, "uniform_B1": uni1, "uniform_B23": uni23,
                    "best_B1_policy": best1_pid, "best_B1_fc_prevention": best1_val}


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 4))
    ap.add_argument("--probe", type=int, default=0, help="run N policy jobs inline and stop")
    a = ap.parse_args()
    t0 = time.time()

    # ---------------- pass 1: baselines (convention-invariant) ----------------
    base_jobs = [(cc, ev, topo, t) for cc in COUNTS for ev in EVIDENCE
                 for topo in TOPOS for t in range(N_SEEDS)]
    base_jobs += [(0, ev, topo, t) for ev in EVIDENCE for topo in TOPOS for t in range(N_SEEDS)]
    base = defaultdict(list)
    print(f"baselines: {len(base_jobs)} on {a.workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for job, r in zip(base_jobs, ex.map(_run_baseline, base_jobs, chunksize=8)):
            base[(job[0], job[1], job[2])].append(r)
    for v in base.values():
        v.sort(key=lambda r: r["trial"])
    print(f"  baselines done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- suspicion-delta calibration (clean arms only) ----------------
    # intercept: policy acts at round t on mean_conf[t]  -> shift indices 0..T-1
    # strict   : policy acts at round t on mean_conf[t-1] -> shift indices 0..T-2
    shifts = {c: defaultdict(list) for c in CONVENTIONS}
    for (cc, ev, topo), rows in base.items():
        if cc != 0:
            continue
        for r in rows:
            mc = r["mean_conf_by_round"]
            shifts["intercept"][f"{topo}|{ev}"].append(max(m - mc[0] for m in mc))
            shifts["strict"][f"{topo}|{ev}"].append(max(m - mc[0] for m in mc[:T_ROUNDS - 1]))
    deltas = {}
    for conv in CONVENTIONS:
        d = {}
        for k, v in shifts[conv].items():
            vs = sorted(v)
            d[k] = vs[min(len(vs) - 1, int(0.95 * len(vs)))]
        deltas[conv] = d
    print(f"  delta[intercept] ev-wise: "
          f"{sorted({round(v, 6) for v in deltas['intercept'].values()})}", flush=True)
    print(f"  delta[strict]    ev-wise: "
          f"{sorted({round(v, 6) for v in deltas['strict'].values()})}", flush=True)

    # ---------------- pass 2: policy arms, both conventions ----------------
    jobs = []
    for conv in CONVENTIONS:
        for cc in COUNTS:
            for ev in EVIDENCE:
                for topo in TOPOS:
                    for t in range(N_SEEDS):
                        cap = base[(cc, ev, topo)][t]["capture_round"]
                        for pid in POLICIES:
                            for B in BUDGETS:
                                jobs.append((conv, cc, ev, topo, t, pid, B,
                                             deltas[conv][f"{topo}|{ev}"], cap))
        for ev in EVIDENCE:                       # clean arms: false-intervention only
            for topo in TOPOS:
                for t in range(N_SEEDS):
                    cap = base[(0, ev, topo)][t]["capture_round"]
                    for pid in CLEAN_POLICIES:
                        for B in BUDGETS:
                            jobs.append((conv, 0, ev, topo, t, pid, B,
                                         deltas[conv][f"{topo}|{ev}"], cap))
    print(f"policy jobs: {len(jobs)}", flush=True)

    if a.probe:
        tp = time.time()
        for j in jobs[:a.probe]:
            _run_one(j)
        per = (time.time() - tp) / a.probe
        print(f"per-sim {per * 1000:.1f}ms -> full {len(jobs)} on {a.workers} workers "
              f"~ {len(jobs) * per / a.workers / 60:.1f} min")
        return

    arms = defaultdict(lambda: defaultdict(dict))     # [conv][cell][(pid,B)][trial] = row
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_run_one, jobs, chunksize=32)):
            arms[r["conv"]][r["cell"]].setdefault((r["policy"], r["budget"]), {})[r["trial"]] = r
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  policy arms done ({time.time() - t0:.0f}s)", flush=True)

    # =====================================================================
    # per-cell aggregation + regime classification, per convention
    # =====================================================================
    cell_keys = [f"{cc}|{ev}|{topo}" for cc in COUNTS for ev in EVIDENCE for topo in TOPOS]
    cells_out, regimes, per_topo, windowed_meta = {}, {}, {}, {}
    for conv in CONVENTIONS:
        regimes[conv] = {}
        per_topo[conv] = defaultdict(lambda: defaultdict(int))

    for ck in sorted(cell_keys):
        cc, ev, topo = int(ck.split("|")[0]), float(ck.split("|")[1]), ck.split("|")[2]
        brows = base[(cc, ev, topo)]
        n = len(brows)
        fc_base = sum(r["false_consensus"] for r in brows) / n
        captured = [r["capture_round"] for r in brows if r["capture_round"] is not None]
        med_c = median(captured) if len(captured) >= n / 2 else None
        entry = {"fc_baseline": fc_base, "capture_rate": len(captured) / n,
                 "median_capture": med_c,
                 "windowed": bool(med_c is not None and 3 <= med_c <= 10),
                 "by_convention": {}}
        windowed_meta[ck] = entry["windowed"]
        for conv in CONVENTIONS:
            A = arms[conv][ck]
            pol = {}
            for p in POLICIES:
                for B in BUDGETS:
                    ts = A[(p, B)]
                    pol[f"{p}@B{B}"] = {
                        "fc_prevention": fc_base - sum(v["false_consensus"] for v in ts.values()) / len(ts),
                        "timely": sum(v["timely"] for v in ts.values()) / len(ts),
                        "mean_checks": sum(v["n_checks"] for v in ts.values()) / len(ts)}
            regime, diag = classify(fc_base, pol)
            regimes[conv][ck] = regime
            per_topo[conv][topo][regime] += 1
            entry["by_convention"][conv] = {"regime": regime, **diag, "policies": pol}
        entry["regime_changed"] = (entry["by_convention"]["intercept"]["regime"]
                                   != entry["by_convention"]["strict"]["regime"])
        cells_out[ck] = entry

    # ---------------- regime shares + transitions ----------------
    h3 = {}
    named = ("ceiling_bound", "placement_critical", "budget_saturated")
    for conv in CONVENTIONS:
        counts = defaultdict(int)
        for ck in cell_keys:
            counts[regimes[conv][ck]] += 1
        ncell = sum(counts.values())
        shares = {k: v / ncell for k, v in counts.items()}
        stability = {r: sum(1 for tp in per_topo[conv] if per_topo[conv][tp].get(r, 0) > 0)
                     for r in named}
        h3[conv] = {"n_cells": ncell, "counts": dict(counts), "shares": shares,
                    "per_topology": {t: dict(d) for t, d in per_topo[conv].items()},
                    "ge10pct": {r: shares.get(r, 0) >= 0.10 for r in named},
                    "present_in_topologies": stability,
                    "registered_pass": all(shares.get(r, 0) >= 0.10 for r in named)
                                       and all(stability[r] >= 3 for r in named)}

    trans = defaultdict(int)
    changed = []
    for ck in sorted(cell_keys):
        a_, b_ = regimes["intercept"][ck], regimes["strict"][ck]
        trans[f"{a_}->{b_}"] += 1
        if a_ != b_:
            changed.append({"cell": ck, "intercept": a_, "strict": b_,
                            "median_capture": cells_out[ck]["median_capture"],
                            "fc_baseline": cells_out[ck]["fc_baseline"],
                            "oracle_any_intercept": cells_out[ck]["by_convention"]["intercept"]["oracle_any"],
                            "oracle_any_strict": cells_out[ck]["by_convention"]["strict"]["oracle_any"],
                            "best_B1_intercept": cells_out[ck]["by_convention"]["intercept"]["best_B1_fc_prevention"],
                            "best_B1_strict": cells_out[ck]["by_convention"]["strict"]["best_B1_fc_prevention"]})
    ORDER = {"placement_critical": 3, "budget_saturated": 2, "intermediate": 1,
             "ceiling_bound": 0, "benign": 0}
    downgrades = sum(1 for c in changed if ORDER[c["strict"]] < ORDER[c["intercept"]])
    upgrades = sum(1 for c in changed if ORDER[c["strict"]] > ORDER[c["intercept"]])
    transitions = {"matrix": dict(sorted(trans.items())), "n_changed": len(changed),
                   "n_unchanged": len(cell_keys) - len(changed),
                   "n_downgraded": downgrades, "n_upgraded": upgrades,
                   "n_lateral": len(changed) - downgrades - upgrades,
                   "changed_cells": changed}

    # =====================================================================
    # H2: placement beats budget -- both endpoints, both conventions
    # =====================================================================
    wcells = [ck for ck in sorted(cell_keys) if windowed_meta[ck]]
    h2 = {}
    for conv in CONVENTIONS:
        # --- recipe A (stats_s1.py): per-seed paired deltas inside each cell,
        #     best-B1 policy selected by fc_prevention, as registered ---
        perseed = {"timely": [], "prevention": []}
        percell = []
        for ck in wcells:
            A = arms[conv][ck]
            bp = cells_out[ck]["by_convention"][conv]["best_B1_policy"]
            u2 = A[("uniform", 2)]
            b1 = A[(bp, 1)]
            g1 = A[("geometric", 1)]
            seeds = sorted(u2)
            d_tc = [int(b1[s]["timely"]) - int(u2[s]["timely"]) for s in seeds]
            d_pv = [int(not b1[s]["false_consensus"]) - int(not u2[s]["false_consensus"])
                    for s in seeds]
            d_tc_geo = [int(g1[s]["timely"]) - int(u2[s]["timely"]) for s in seeds]
            d_pv_geo = [int(not g1[s]["false_consensus"]) - int(not u2[s]["false_consensus"])
                        for s in seeds]
            perseed["timely"].append(sum(d_tc) / len(d_tc))
            perseed["prevention"].append(sum(d_pv) / len(d_pv))
            percell.append({"cell": ck, "best_B1_policy": bp,
                            "delta_timely": sum(d_tc) / len(d_tc),
                            "delta_prevention": sum(d_pv) / len(d_pv),
                            "delta_timely_geometric": sum(d_tc_geo) / len(d_tc_geo),
                            "delta_prevention_geometric": sum(d_pv_geo) / len(d_pv_geo),
                            "ci_timely": boot_ci_cell(d_tc),
                            "ci_prevention": boot_ci_cell(d_pv)})
        conv_out = {"n_windowed": len(wcells), "cells": percell, "registered_recipe": {}}
        for ep in ("timely", "prevention"):
            ds = perseed[ep]
            pos = sum(1 for d in ds if d > 0)
            neg = sum(1 for d in ds if d < 0)
            conv_out["registered_recipe"][ep] = {
                "k_positive": pos, "k_negative": neg, "n_ties": len(ds) - pos - neg,
                "sign_p_one_sided": sign_test_p(pos, pos + neg),
                "mean_delta": sum(ds) / len(ds),
                "max_delta": max(ds), "min_delta": min(ds),
                "min_positive_delta": min([d for d in ds if d > 0], default=None)}
        # geometric-B1 robustness line (fixed a priori policy)
        for ep in ("timely", "prevention"):
            ds = [c[f"delta_{ep}_geometric"] for c in percell]
            pos = sum(1 for d in ds if d > 0)
            neg = sum(1 for d in ds if d < 0)
            conv_out["registered_recipe"][f"{ep}_geometric_B1"] = {
                "k_positive": pos, "k_negative": neg, "n_ties": len(ds) - pos - neg,
                "sign_p_one_sided": sign_test_p(pos, pos + neg),
                "mean_delta": sum(ds) / len(ds)}

        # --- recipe B (verify_p1_numbers.py): cell-level deltas, best-B1 chosen
        #     per endpoint, shared bootstrap rng drawn timely-then-prevention ---
        cellmax = {"timely": [], "prevention": []}
        for ck in wcells:
            p = cells_out[ck]["by_convention"][conv]["policies"]
            b_tc = max(v["timely"] for k, v in p.items()
                       if k.endswith("@B1") and not k.startswith("oracle"))
            b_pv = max(v["fc_prevention"] for k, v in p.items()
                       if k.endswith("@B1") and not k.startswith("oracle"))
            cellmax["timely"].append(b_tc - p["uniform@B2"]["timely"])
            cellmax["prevention"].append(b_pv - p["uniform@B2"]["fc_prevention"])
        rng = random.Random(BOOT_SEED_AGG)
        conv_out["verify_recipe"] = {}
        for ep in ("timely", "prevention"):
            ds = cellmax[ep]
            pos = sum(1 for d in ds if d > 0)
            nnz = sum(1 for d in ds if d != 0)
            pt, lo, hi = boot_ci_agg(rng, ds)
            conv_out["verify_recipe"][ep] = {
                "k_positive": pos, "n_nonzero": nnz,
                "sign_p_one_sided": sign_test_p(pos, nnz),
                "mean_delta": pt, "paired_boot_ci": [lo, hi],
                "max_delta": max(ds), "min_delta": min(ds),
                "per_cell_delta": dict(zip(wcells, ds))}
        h2[conv] = conv_out

    h2["windowed_cells"] = wcells
    h2["endpoint_note"] = ("timely = timely_committed_prune rate; prevention = "
                           "fc_baseline - false-consensus rate (paired per seed in "
                           "the registered recipe, cell-level in the verify recipe)")

    # =====================================================================
    # exhaustive single-round placement: the most generous one-check policy
    # =====================================================================
    sr_jobs = [(conv, cc, ev, topo, t, r)
               for conv in CONVENTIONS for cc in COUNTS for ev in EVIDENCE
               for topo in TOPOS for t in range(N_SEEDS) for r in ROUNDS]
    print(f"single-round enumeration jobs: {len(sr_jobs)}", flush=True)
    sr = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_run_single_round, sr_jobs, chunksize=32)):
            sr[r["conv"]][r["cell"]][r["round"]][r["trial"]] = r
            if (i + 1) % 30000 == 0:
                print(f"  {i + 1}/{len(sr_jobs)} ({time.time() - t0:.0f}s)", flush=True)

    exhaustive = {}
    for conv in CONVENTIONS:
        per_cell, profile = {}, {r: {"timely": [], "prevention": []} for r in ROUNDS}
        for ck in sorted(cell_keys):
            fc_base = cells_out[ck]["fc_baseline"]
            rounds = {}
            for r in ROUNDS:
                ts = sr[conv][ck][r]
                rounds[r] = {
                    "timely": sum(v["timely"] for v in ts.values()) / len(ts),
                    "fc_prevention": fc_base - sum(v["false_consensus"] for v in ts.values()) / len(ts)}
                if windowed_meta[ck]:
                    profile[r]["timely"].append(rounds[r]["timely"])
                    profile[r]["prevention"].append(rounds[r]["fc_prevention"])
            bt = max(ROUNDS, key=lambda r: rounds[r]["timely"])
            bp = max(ROUNDS, key=lambda r: rounds[r]["fc_prevention"])
            per_cell[ck] = {
                "best_round_timely": bt, "best_timely": rounds[bt]["timely"],
                "best_round_prevention": bp, "best_prevention": rounds[bp]["fc_prevention"],
                "by_round": {str(r): rounds[r] for r in ROUNDS}}
        # contrast against the doubled uniform budget on the windowed cells
        rng = random.Random(BOOT_SEED_AGG)
        contrast = {}
        for ep, key in (("timely", "best_timely"), ("prevention", "best_prevention")):
            ds = []
            for ck in wcells:
                u2 = cells_out[ck]["by_convention"][conv]["policies"]["uniform@B2"]
                ds.append(per_cell[ck][key] - (u2["timely"] if ep == "timely"
                                               else u2["fc_prevention"]))
            pos = sum(1 for d in ds if d > 0)
            nnz = sum(1 for d in ds if d != 0)
            pt, lo, hi = boot_ci_agg(rng, ds)
            contrast[ep] = {"k_positive": pos, "n_nonzero": nnz, "n_cells": len(ds),
                            "sign_p_one_sided": sign_test_p(pos, nnz),
                            "mean_delta": pt, "paired_boot_ci": [lo, hi],
                            "per_cell_delta": dict(zip(wcells, ds))}
        exhaustive[conv] = {
            "note": "best fixed single check round chosen PER CELL on the endpoint "
                    "itself (an upper bound on any deployable one-check policy, and a "
                    "strictly more generous fit than the paper's best-of-six-heuristics "
                    "recipe); contrasted against uniform@B2 on the 12 windowed cells",
            "windowed_round_profile": {
                str(r): {"mean_timely": sum(profile[r]["timely"]) / len(profile[r]["timely"]),
                         "mean_prevention": sum(profile[r]["prevention"]) / len(profile[r]["prevention"])}
                for r in ROUNDS},
            "contrast_vs_uniform_B2": contrast,
            "per_cell": per_cell}

    # ---- where exactly do the two conventions diverge? (mechanism) ----
    div = {}
    for r in ROUNDS:
        nd = nt = 0
        for ck in cell_keys:
            for t in range(N_SEEDS):
                i_ = sr["intercept"][ck][r][t]
                s_ = sr["strict"][ck][r][t]
                nt += 1
                if (i_["timely"] != s_["timely"]
                        or i_["false_consensus"] != s_["false_consensus"]):
                    nd += 1
        div[str(r)] = {"n_pairs": nt, "n_outcome_differs": nd}
    # is a strict round-2 check identical to spending no check at all?
    n_r2_null = sum(
        1 for ck in cell_keys for t in range(N_SEEDS)
        if sr["strict"][ck][2][t]["false_consensus"]
        == base[(int(ck.split("|")[0]), float(ck.split("|")[1]), ck.split("|")[2])][t]["false_consensus"]
        and not sr["strict"][ck][2][t]["timely"])
    exhaustive["convention_divergence_by_check_round"] = {
        "per_round": div,
        "rounds_with_any_difference": [r for r in ROUNDS if div[str(r)]["n_outcome_differs"] > 0],
        "strict_round2_identical_to_no_check": {
            "n_pairs": len(cell_keys) * N_SEEDS, "n_identical": n_r2_null},
        "interpretation": "one round of reading latency changes the outcome only for "
                          "checks placed at round 2, because the dominant anomaly signal "
                          "(confidence/claim constancy) needs >=2 public messages to exist "
                          "-- the same mechanism the paper documents in Study 4",
    }

    # =====================================================================
    # regime map re-instrumented with a latency-aware placement library:
    # the single-check class is "any fixed round 2..15" instead of the six
    # heuristics, which cannot express a round-3 check at B=1.
    # =====================================================================
    h3_aug = {}
    for conv in CONVENTIONS:
        counts = defaultdict(int)
        per_topo_a = defaultdict(lambda: defaultdict(int))
        cellwise = {}
        for ck in sorted(cell_keys):
            topo = ck.split("|")[2]
            fc_base = cells_out[ck]["fc_baseline"]
            pol = dict(cells_out[ck]["by_convention"][conv]["policies"])
            br = exhaustive[conv]["per_cell"][ck]
            # add the best fixed single round as an extra B=1 candidate
            pol["bestround@B1"] = {"fc_prevention": br["best_prevention"],
                                   "timely": br["best_timely"]}
            oracle_any = max(pol[f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS)
            uni1 = pol["uniform@B1"]["fc_prevention"]
            best1_val, best1_pid = max(
                [(pol[f"{p}@B1"]["fc_prevention"], p) for p in NONORACLE]
                + [(pol["bestround@B1"]["fc_prevention"], f"round{br['best_round_prevention']}")])
            uni23 = max(pol[f"uniform@B{B}"]["fc_prevention"] for B in (2, 3))
            if fc_base < 0.10:
                regime = "benign"
            elif oracle_any <= 0.05:
                regime = "ceiling_bound"
            elif best1_val >= 2 * max(uni1, 0.02) and best1_val >= 0.10:
                regime = "placement_critical"
            elif uni23 >= oracle_any - 0.05:
                regime = "budget_saturated"
            else:
                regime = "intermediate"
            counts[regime] += 1
            per_topo_a[topo][regime] += 1
            cellwise[ck] = {"regime": regime, "best_B1_policy": best1_pid,
                            "best_B1_fc_prevention": best1_val}
        ncell = sum(counts.values())
        stability = {r: sum(1 for tp in per_topo_a if per_topo_a[tp].get(r, 0) > 0)
                     for r in named}
        h3_aug[conv] = {
            "counts": dict(counts), "shares": {k: v / ncell for k, v in counts.items()},
            "per_topology": {t: dict(d) for t, d in per_topo_a.items()},
            "present_in_topologies": stability,
            "registered_pass": all((counts.get(r, 0) / ncell) >= 0.10 for r in named)
                               and all(stability[r] >= 3 for r in named),
            "cells": cellwise}
    h3_aug["note"] = ("same PREREG H3 rule, but the B=1 candidate set is the six "
                      "heuristics PLUS the best fixed check round 2..15 chosen per cell. "
                      "Selection is fit on the cell's own outcomes -- the same fitting "
                      "level as the paper's registered best-of-six recipe, not a "
                      "deployable policy. Isolates how much of the strict-convention "
                      "regime collapse is a property of the fault dynamics versus a "
                      "property of a policy library that cannot express a round-3 check.")

    # =====================================================================
    # held-out placement selection (calibration seeds pick, held-out seeds score)
    # =====================================================================
    CAL = list(range(0, N_SEEDS // 2))
    EVA = list(range(N_SEEDS // 2, N_SEEDS))

    def _rate(d, seeds, field):
        return sum(d[s][field] for s in seeds) / len(seeds)

    split = {"calibration_seeds": [CAL[0], CAL[-1]], "evaluation_seeds": [EVA[0], EVA[-1]],
             "note": "the check round / policy is chosen on the calibration seeds ONLY and "
                     "scored on the disjoint evaluation seeds, so the placement advantage "
                     "is not fit on the outcome it is measured against",
             "by_convention": {}}
    for conv in CONVENTIONS:
        res = {}
        for family in ("six_heuristics_B1", "any_single_round"):
            rng = random.Random(BOOT_SEED_AGG)
            fam = {}
            for ep in ("timely", "prevention"):
                ds, detail = [], []
                for ck in wcells:
                    cc, ev, topo = int(ck.split("|")[0]), float(ck.split("|")[1]), ck.split("|")[2]
                    brows = base[(cc, ev, topo)]
                    fcb_e = sum(brows[s]["false_consensus"] for s in EVA) / len(EVA)

                    def score(d, seeds, fcb):
                        return (_rate(d, seeds, "timely") if ep == "timely"
                                else fcb - _rate(d, seeds, "false_consensus"))

                    if family == "six_heuristics_B1":
                        cands = {p: arms[conv][ck][(p, 1)] for p in NONORACLE}
                        fcb_c = sum(brows[s]["false_consensus"] for s in CAL) / len(CAL)
                        pick = max(cands, key=lambda p: score(cands[p], CAL, fcb_c))
                        chosen = cands[pick]
                    else:
                        cands = {r: sr[conv][ck][r] for r in ROUNDS}
                        fcb_c = sum(brows[s]["false_consensus"] for s in CAL) / len(CAL)
                        pick = max(cands, key=lambda r: score(cands[r], CAL, fcb_c))
                        chosen = cands[pick]
                    u2 = arms[conv][ck][("uniform", 2)]
                    d = score(chosen, EVA, fcb_e) - score(u2, EVA, fcb_e)
                    ds.append(d)
                    detail.append({"cell": ck, "picked": pick, "delta_heldout": d})
                pos = sum(1 for x in ds if x > 0)
                nnz = sum(1 for x in ds if x != 0)
                pt, lo, hi = boot_ci_agg(rng, ds)
                fam[ep] = {"k_positive": pos, "n_nonzero": nnz, "n_cells": len(ds),
                           "sign_p_one_sided": sign_test_p(pos, nnz),
                           "mean_delta": pt, "paired_boot_ci": [lo, hi],
                           "per_cell": detail}
            res[family] = fam
        split["by_convention"][conv] = res

    # ---------------- clean arms: false-intervention rates ----------------
    clean = {}
    for conv in CONVENTIONS:
        c = {}
        for ev in EVIDENCE:
            for topo in TOPOS:
                ck = f"0|{ev}|{topo}"
                A = arms[conv][ck]
                c[f"{ev}|{topo}"] = {
                    f"{p}@B{B}": sum(v["intervened"] for v in A[(p, B)].values()) / len(A[(p, B)])
                    for p in CLEAN_POLICIES for B in BUDGETS}
        clean[conv] = {"per_cell": c,
                       "max_false_intervention": max(v for d in c.values() for v in d.values())}

    # ---------------- harness validation against the published intercept run ----
    published = json.loads((RES / "stats.json").read_text(encoding="utf-8"))
    pub_counts = published["h3"]["counts"]
    pub_h2 = published["h2"]["best"]
    ic = h3["intercept"]["counts"]
    irec = h2["intercept"]["registered_recipe"]["timely"]
    validation = {
        "regime_counts_published": pub_counts,
        "regime_counts_reproduced": ic,
        "regime_counts_match": all(ic.get(k, 0) == v for k, v in pub_counts.items())
                               and sum(ic.values()) == sum(pub_counts.values()),
        "h2_timely_published": {k: pub_h2[k] for k in
                                ("k_positive", "k_negative", "n_windowed",
                                 "sign_p_one_sided", "mean_delta")},
        "h2_timely_reproduced": {k: irec[k] for k in
                                 ("k_positive", "k_negative", "sign_p_one_sided", "mean_delta")},
        "h2_timely_match": (irec["k_positive"] == pub_h2["k_positive"]
                            and irec["k_negative"] == pub_h2["k_negative"]
                            and abs(irec["mean_delta"] - pub_h2["mean_delta"]) < 1e-12),
        "per_cell_regime_match": sum(
            1 for ck in cell_keys
            if published["cells"].get(ck, {}).get("regime") == regimes["intercept"][ck]),
        "n_cells": len(cell_keys),
    }

    out = {
        "meta": {
            "what": "Study 1 regime map + placement-vs-budget contrast under BOTH "
                    "timing conventions (strict-convention regime map)",
            "conventions": {
                "intercept": "check at round t reads rounds 1..t, prunes before round-t "
                             "beliefs update (zero overseer latency; the paper's Study-1 arm)",
                "strict": "check at round t reads only rounds 1..t-1, prunes before round-t "
                          "beliefs update (one round of overseer reading latency)"},
            "implementation": "s1_engine.s1_trial reused verbatim; strict arm rebinds "
                              "s1_engine.anomaly_scores to a wrapper that truncates the "
                              "public log to rounds <= t-1 and scores at round t-1. The "
                              "online suspicion policy is lagged one round and its trigger "
                              "delta recalibrated on the clean 0-committed arms only.",
            "grid": {"committed": list(COUNTS), "evidence": list(EVIDENCE),
                     "topologies": list(TOPOS), "budgets": list(BUDGETS),
                     "policies": list(POLICIES), "n_seeds": N_SEEDS,
                     "rounds": T_ROUNDS, "n_committed_cells": len(cell_keys),
                     "trials_per_convention": len(jobs) // len(CONVENTIONS)},
            "seeds": {"trial_ids": "range(50) through stable_seed(BASE_SEED, ...) as in "
                                   "sweep_s1.py", "random_policy_base": RAND_SEED_BASE,
                      "boot_cell": BOOT_SEED_CELL, "boot_agg": BOOT_SEED_AGG},
            "regime_rule": "PREREG H3, verbatim from stats_s1.py: benign if fc_base<0.10; "
                           "ceiling_bound if oracle_any<=0.05; placement_critical if "
                           "best_B1>=2*max(uniform_B1,0.02) and best_B1>=0.10; "
                           "budget_saturated if max(uniform_B2,uniform_B3)>=oracle_any-0.05; "
                           "else intermediate",
        },
        "validation": validation,
        "suspicion_deltas": deltas,
        "h3_regimes": h3,
        "h3_latency_aware_library": h3_aug,
        "regime_transitions": transitions,
        "h2_placement_vs_budget": h2,
        "exhaustive_single_round": exhaustive,
        "heldout_placement_selection": split,
        "clean_arms": clean,
        "cells": cells_out,
    }
    OUT_F.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---------------- console ----------------
    print("\n=== HARNESS VALIDATION (intercept arm vs published stats.json) ===")
    print(f"  regime counts match: {validation['regime_counts_match']}  "
          f"{validation['regime_counts_reproduced']}")
    print(f"  per-cell regime agreement: {validation['per_cell_regime_match']}/60")
    print(f"  H2 timely match: {validation['h2_timely_match']}  {validation['h2_timely_reproduced']}")
    print("\n=== REGIME SHARES (60 committed cells) ===")
    keys = sorted(set(h3["intercept"]["counts"]) | set(h3["strict"]["counts"]))
    for k in keys:
        i_c = h3["intercept"]["counts"].get(k, 0)
        s_c = h3["strict"]["counts"].get(k, 0)
        print(f"  {k:<19} intercept {i_c:>2}/60 ({i_c/60:5.1%})   "
              f"strict {s_c:>2}/60 ({s_c/60:5.1%})   delta {s_c-i_c:+d}")
    print(f"  registered_pass: intercept={h3['intercept']['registered_pass']} "
          f"strict={h3['strict']['registered_pass']}")
    print(f"\n=== TRANSITIONS === changed {transitions['n_changed']}/60 "
          f"(down {downgrades}, up {upgrades}, lateral {transitions['n_lateral']})")
    for k, v in transitions["matrix"].items():
        if not k.startswith(k.split('->')[1]) or k.split('->')[0] != k.split('->')[1]:
            print(f"  {k}: {v}")
    print("\n=== H2 PLACEMENT BEATS BUDGET (12 windowed cells) ===")
    for conv in CONVENTIONS:
        for ep in ("timely", "prevention"):
            r = h2[conv]["registered_recipe"][ep]
            v = h2[conv]["verify_recipe"][ep]
            print(f"  {conv:<9} {ep:<10} registered {r['k_positive']}+/{r['k_negative']}- "
                  f"p={r['sign_p_one_sided']:.2e} mean={r['mean_delta']:+.3f} | "
                  f"verify {v['k_positive']}/{v['n_nonzero']} p={v['sign_p_one_sided']:.2e} "
                  f"mean={v['mean_delta']:+.3f} CI[{v['paired_boot_ci'][0]:+.3f},"
                  f"{v['paired_boot_ci'][1]:+.3f}]")
    print("\n=== BEST FIXED SINGLE ROUND (per cell) vs UNIFORM B=2, 12 windowed cells ===")
    for conv in CONVENTIONS:
        for ep in ("timely", "prevention"):
            c = exhaustive[conv]["contrast_vs_uniform_B2"][ep]
            print(f"  {conv:<9} {ep:<10} wins {c['k_positive']}/{c['n_cells']} "
                  f"p={c['sign_p_one_sided']:.2e} mean={c['mean_delta']:+.3f} "
                  f"CI[{c['paired_boot_ci'][0]:+.3f},{c['paired_boot_ci'][1]:+.3f}]")
    print("  windowed-cell mean timely by check round (round: intercept / strict)")
    for r in ROUNDS:
        pi = exhaustive["intercept"]["windowed_round_profile"][str(r)]["mean_timely"]
        ps = exhaustive["strict"]["windowed_round_profile"][str(r)]["mean_timely"]
        print(f"    r={r:<3} {pi:.3f} / {ps:.3f}")
    dv = exhaustive["convention_divergence_by_check_round"]
    print(f"  check rounds where the two conventions differ at all: "
          f"{dv['rounds_with_any_difference']}")
    print(f"  strict round-2 check identical to spending no check: "
          f"{dv['strict_round2_identical_to_no_check']['n_identical']}/"
          f"{dv['strict_round2_identical_to_no_check']['n_pairs']} cell-seed pairs")
    print("\n=== REGIME SHARES WITH A LATENCY-AWARE PLACEMENT LIBRARY (any fixed round) ===")
    for k in keys:
        i_c = h3_aug["intercept"]["counts"].get(k, 0)
        s_c = h3_aug["strict"]["counts"].get(k, 0)
        print(f"  {k:<19} intercept {i_c:>2}/60 ({i_c/60:5.1%})   "
              f"strict {s_c:>2}/60 ({s_c/60:5.1%})")
    print("\n=== HELD-OUT PLACEMENT SELECTION (pick on seeds 0-24, score on 25-49) ===")
    for conv in CONVENTIONS:
        for fam in ("six_heuristics_B1", "any_single_round"):
            for ep in ("timely", "prevention"):
                c = split["by_convention"][conv][fam][ep]
                print(f"  {conv:<9} {fam:<18} {ep:<10} wins {c['k_positive']}/{c['n_cells']} "
                      f"p={c['sign_p_one_sided']:.2e} mean={c['mean_delta']:+.3f} "
                      f"CI[{c['paired_boot_ci'][0]:+.3f},{c['paired_boot_ci'][1]:+.3f}]")
    print("\n=== CLEAN ARMS ===")
    for conv in CONVENTIONS:
        print(f"  {conv:<9} max false-intervention {clean[conv]['max_false_intervention']:.3f}")
    print(f"\nwrote {OUT_F}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
