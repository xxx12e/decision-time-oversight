r"""S2 extension: how accurate must an onset estimate be before placement-aware
scheduling beats uniform coverage?

Reviewer objection this answers: Study 2's reversal is partly predetermined.
Audit-early (geometric) spacing puts every check in steps 2--16 while the shift
step tau is drawn uniform[40,120], so "100% miss at B<=4" follows from the two
windows being disjoint, not from data.

Two extensions, both on the paper's own rule-shift substrate, its endpoints
(miss, penalized delay with horizon = 160 - tau, false alarm) and its budgets
B in {1,2,4,8,16}:

  (i)  Onset distributions that OVERLAP the audit-early window: a broad
       uniform[5,155], a bimodal 0.5*U[6,20] + 0.5*U[95,135], and an
       early-concentrated uniform[2,30], alongside the paper's uniform[40,120].

  (ii) A placement-aware policy driven by a CORRUPTED onset estimate
            tauhat = clip(round(tau + b + sigma * z)),  z ~ N(0,1),
       with corruption magnitude E = RMSE = sqrt(b^2 + sigma^2) swept along
       three rays: pure noise (b=0), late bias (b=+E/sqrt2), early bias
       (b=-E/sqrt2).  Two placement-aware policies:
         aware_point : checks at tauhat, tauhat+1, ... (the paper's
                       onset-informed reference applied to the estimate)
         aware_bayes : the schedule minimising EXPECTED penalized delay under
                       the policy's OWN belief N(tauhat, s^2), s = max(E, 1),
                       solved exactly by dynamic programming.
       Neither policy ever sees tau: only tauhat and its own stated scale s.

Ground-truth discipline: nothing here is fit on labels.  The policies see only
(tauhat, s); the crossover is READ OFF the sweep, not fitted.  tau, the ray and
the corruption level are experiment design, exactly as tau is in sweep_s2.py.

Exactness: the S2 check loop contains no RNG (the `base` list drawn in
s2_engine.s2_trial is never read), and information_gain_probe /
update_bayesian_posterior / _soften_posterior are deterministic.  Because the
schedule is sorted and tau is a single threshold, the sequence of true
semantics seen by the checks is monotone, so the alarm CONDITION after i checks
depends only on (cell, condition, k) with k = number of checks that fire before
tau; the schedule enters the outcome only through the absolute step at which
the alarm may fire.  We precompute that state table with the substrate's own
instruments and then evaluate millions of schedules by lookup.  The equivalence
is VERIFIED against s2_engine.s2_trial on a sample of the paper's own
configurations; the pass count is written into the output JSON.

Usage:  python -X utf8 onset_accuracy_sweep.py [--workers N]
Output: ..\results\s2\onset_accuracy_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from array import array
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2_engine import (  # noqa: E402
    L_TOTAL, PROBES_PER_CHECK, get_cells, make_schedule, s2_trial, tau_for,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "s2"

SEED = 20260808
SEEDS = list(range(20260808, 20260858))          # the 50 fixed prevalence seeds
BUDGETS = (1, 2, 4, 8, 16)                       # the paper's budgets
CONDITIONS = ("well_specified", "misspecified")
MAX_B = 16
MIN_TAU, MAX_TAU = 1, L_TOTAL - 1                # horizon >= 1

BLIND = ("uniform", "random", "front", "back", "geometric")   # onset-blind
AWARE = ("aware_point", "aware_bayes")

# corruption magnitude grid: RMSE of the onset estimate, in stream steps
E_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
          12.0, 14.0, 16.0, 18.0, 20.0, 23.0, 26.0, 30.0, 35.0, 40.0,
          50.0, 60.0, 80.0)
RAYS = {"noise": 0.0,
        "late_bias": 1.0 / math.sqrt(2.0),
        "early_bias": -1.0 / math.sqrt(2.0)}

ONSET_DISTS = {
    "paper_u40_120": {"kind": "uniform", "lo": 40, "hi": 120,
                      "note": "the paper's onset law; disjoint from the audit-early window"},
    "broad_u5_155": {"kind": "uniform", "lo": 5, "hi": 155,
                     "note": "broad; fully overlaps the audit-early window"},
    "bimodal_6_20__95_135": {"kind": "bimodal", "lo1": 6, "hi1": 20,
                             "lo2": 95, "hi2": 135, "p1": 0.5,
                             "note": "bimodal; early mode sits inside the audit-early window"},
    "early_u2_30": {"kind": "uniform", "lo": 2, "hi": 30,
                    "note": "early-concentrated; audit-early should now win"},
}

N_BOOT = 1000
M_MISS, M_PEN, M_FA, M_NCK = 0, 1, 2, 3


# --------------------------------------------------------------------------
# onset draws
# --------------------------------------------------------------------------

def draw_tau(dist: str, cell_key: str, seed: int) -> int:
    d = ONSET_DISTS[dist]
    rng = random.Random(f"onset|{dist}|{cell_key}|{seed}")
    if d["kind"] == "uniform":
        t = rng.randint(d["lo"], d["hi"])
    else:
        t = (rng.randint(d["lo1"], d["hi1"]) if rng.random() < d["p1"]
             else rng.randint(d["lo2"], d["hi2"]))
    return max(MIN_TAU, min(MAX_TAU, t))


def draw_z(dist: str, cell_key: str, seed: int, ray: str) -> float:
    """One standard normal per unit, REUSED across corruption levels (common
    random numbers), so each unit's curve in E is smooth and the crossover is
    not blurred by resampling noise."""
    return random.Random(f"onsetz|{dist}|{cell_key}|{seed}|{ray}").gauss(0.0, 1.0)


# --------------------------------------------------------------------------
# alarm-condition state table (built with the substrate's own instruments)
# --------------------------------------------------------------------------

def build_state_table(cell_key: str, condition: str) -> list[list[bool]]:
    """states[k][i] = does the alarm CONDITION hold after i checks, when the
    first k of them fired before the shift and the rest after it?

    Replays exactly the check body of s2_engine.s2_trial (soften 0.12, then
    PROBES_PER_CHECK EIG probes with Bayesian updates), branching off a shared
    pre-shift spine so the table costs 16 + sum_k (16-k) = 152 checks.
    """
    from semantic_rule_shift import v3, v4                     # noqa: E402
    from semantic_rule_shift import v7_oracle_free as v7       # noqa: E402

    cell = v7.restrict(get_cells()[cell_key], condition)
    cand = cell["candidate_names_full"]
    formulas, vectors = cell["formulas"], cell["vectors"]
    source, target = cell["source"], cell["target"]

    def cond_of(post):
        best = max(post, key=post.get)
        return bool(best != source and post[source] < 0.25)

    def do_check(post, used, true_name):
        post = v4._soften_posterior(post, cand, mixing=0.12)
        for _ in range(PROBES_PER_CHECK):
            fi = v3.information_gain_probe(cand, formulas, vectors, post, used)
            used.add(fi)
            post = v3.update_bayesian_posterior(
                cand, vectors, post, fi, vectors[true_name][fi])
        return post

    post = {n: 1.0 / len(cand) for n in cand}
    used: set[int] = set()
    spine = [(dict(post), set(used), cond_of(post))]
    for _ in range(MAX_B):                                     # all pre-shift
        post = do_check(post, used, source)
        spine.append((dict(post), set(used), cond_of(post)))

    states = [[False] * (MAX_B + 1) for _ in range(MAX_B + 1)]
    for k in range(MAX_B + 1):
        row = states[k]
        for i in range(k + 1):                                 # pre-shift prefix
            row[i] = spine[i][2]
        post, used = dict(spine[k][0]), set(spine[k][1])
        for i in range(k + 1, MAX_B + 1):                      # post-shift tail
            post = do_check(post, used, target)
            row[i] = cond_of(post)
    return states


def evaluate(states, sched, tau):
    """-> (alarm, miss, penalized_delay, false_alarm, n_checks) for a sorted,
    distinct schedule against shift step tau.  Mirrors s2_trial's bookkeeping."""
    row = states[bisect_left(sched, tau)]                      # k pre-shift checks
    alarm = None
    if row[bisect_right(sched, 10)]:
        alarm = 10
    else:
        for i, t in enumerate(sched):
            if t > 10 and row[i + 1]:
                alarm = t
                break
    if alarm is not None and alarm >= tau:
        return alarm, 0.0, float(alarm - tau), 0.0, float(len(sched))
    fa = 1.0 if alarm is not None else 0.0
    return alarm, 1.0, float(L_TOTAL - tau), fa, float(len(sched))


# --------------------------------------------------------------------------
# placement-aware schedules (see only tauhat and the stated error scale s)
# --------------------------------------------------------------------------

_POS = np.arange(L_TOTAL, dtype=np.float64)
_AMASK = np.triu(np.ones((L_TOTAL, L_TOTAL), dtype=bool), 1)   # [a, b] -> a < b
_COL = np.arange(L_TOTAL)
_bayes_cache: dict = {}
_point_cache: dict = {}
_blind_cache: dict = {}


def _belief(tauhat: int, s: float) -> np.ndarray:
    w = np.zeros(L_TOTAL)
    idx = np.arange(MIN_TAU, MAX_TAU + 1)
    w[idx] = np.exp(-0.5 * ((idx - tauhat) / s) ** 2)
    tot = w.sum()
    if tot <= 0:                                               # belief off-grid
        w[:] = 0.0
        w[min(max(int(tauhat), MIN_TAU), MAX_TAU)] = 1.0
        tot = 1.0
    return w / tot


def bayes_schedules(tauhat: int, s: float) -> dict:
    """For every budget, the B distinct steps minimising EXPECTED penalized
    delay under the policy's own belief N(tauhat, s^2) truncated to [1,159].
    Planning model: the first check at or after tau detects with delay t - tau;
    if no check lands at or after tau the cost is the horizon 160 - tau.  Exact
    DP; uses no ground truth."""
    key = (tauhat, round(s, 6))
    hit = _bayes_cache.get(key)
    if hit is not None:
        return hit
    w = _belief(tauhat, s)
    W = np.cumsum(w)
    S = np.cumsum(_POS * w)
    Stot = float(S[-1])
    D = np.empty((MAX_B + 1, L_TOTAL))
    ARG = np.zeros((MAX_B + 1, L_TOTAL), dtype=np.int64)
    D[0] = np.inf
    D[1] = _POS * W - S                                        # one check at b
    for j in range(2, MAX_B + 1):
        A = np.where(_AMASK, (D[j - 1] + S)[:, None] - np.outer(W, _POS), np.inf)
        a_star = A.argmin(axis=0)
        D[j] = _POS * W - S + A[a_star, _COL]
        ARG[j] = a_star
    tail = L_TOTAL * (1.0 - W) - (Stot - S)
    out = {}
    for B in BUDGETS:
        b = int(np.argmin(D[B] + tail))
        sched = [b]
        for j in range(B, 1, -1):
            b = int(ARG[j][b])
            sched.append(b)
        out[B] = tuple(sorted(set(sched)))
    _bayes_cache[key] = out
    return out


def point_schedule(tauhat: int, B: int) -> tuple:
    """The paper's onset-informed reference applied to the ESTIMATE."""
    key = (tauhat, B)
    hit = _point_cache.get(key)
    if hit is None:
        hit = tuple(sorted({min(L_TOTAL - 1, max(0, tauhat + i)) for i in range(B)}))
        _point_cache[key] = hit
    return hit


def blind_schedule(policy: str, B: int, cell_key: str, seed: int) -> tuple:
    key = (policy, B, seed if policy == "random" else 0)
    hit = _blind_cache.get(key)
    if hit is None:
        hit = tuple(sorted(make_schedule(policy, B, cell_key, seed)))
        _blind_cache[key] = hit
    return hit


# --------------------------------------------------------------------------
# worker jobs
# --------------------------------------------------------------------------

_LOCAL_TABLES: dict = {}


def _local_table(cell_key: str, cond: str):
    key = (cell_key, cond)
    if key not in _LOCAL_TABLES:
        _LOCAL_TABLES[key] = build_state_table(cell_key, cond)
    return _LOCAL_TABLES[key]


def _job(spec):
    if spec[0] == "table":
        _, cell_key, cond = spec
        return ("table", cell_key, cond, build_state_table(cell_key, cond))
    _, chunk = spec
    ok = bad = 0
    diffs = []
    for cell_key, cond, seed, policy, B in chunk:
        tau = tau_for(cell_key, seed)
        sched = tuple(sorted(make_schedule(policy, B, cell_key, seed)))
        ref = s2_trial(cell_key, cond, seed, frozenset(sched))
        alarm, miss, pen, fa, _ = evaluate(_local_table(cell_key, cond), sched, tau)
        same = (alarm == ref["alarm"] and bool(miss) == ref["miss"]
                and bool(fa) == ref["false_alarm"]
                and abs(pen - ref["penalized_delay"]) < 1e-9)
        ok += same
        bad += (not same)
        if not same and len(diffs) < 5:
            diffs.append({"cell": cell_key, "condition": cond, "seed": seed,
                          "policy": policy, "budget": B,
                          "ref": [ref["alarm"], ref["miss"], ref["penalized_delay"]],
                          "got": [alarm, bool(miss), pen]})
    return ("verify", ok, bad, diffs)


# --------------------------------------------------------------------------
# crossover estimation
# --------------------------------------------------------------------------

def _first_cross(curve, lv):
    if curve[0] <= 0:
        return float(lv[0])
    for i in range(1, len(curve)):
        if curve[i] <= 0:
            a, b = curve[i - 1], curve[i]
            f = a / (a - b) if (a - b) != 0 else 0.0
            return float(lv[i - 1] + f * (lv[i] - lv[i - 1]))
    return None


def crossover(levels, delta_units, rng):
    """delta_units: (n_clusters, n_levels) of paired (uniform - aware) values;
    positive means the placement-aware policy is better.  Returns the smallest
    corruption magnitude at which that advantage disappears."""
    lv = np.asarray(levels, dtype=float)
    m = delta_units.mean(axis=0)
    point = _first_cross(m, lv)
    n = delta_units.shape[0]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    reps = delta_units[idx].mean(axis=1)
    xs = [c for c in (_first_cross(r, lv) for r in reps) if c is not None]
    frac = len(xs) / N_BOOT
    ci = ([round(float(np.percentile(xs, 2.5)), 2), round(float(np.percentile(xs, 97.5)), 2)]
          if len(xs) >= 0.5 * N_BOOT else None)
    lo95 = np.percentile(reps, 5.0, axis=0)                    # one-sided 95 %
    ns = next((float(lv[i]) for i in range(len(lv)) if lo95[i] <= 0), None)
    return {
        "crossover_E": None if point is None else round(point, 2),
        "crossover_E_ci95": ci,
        "boot_frac_with_crossover": frac,
        "first_E_not_significantly_better": ns,
        "mean_delta_by_E": [round(float(x), 3) for x in m],
        "lower95_delta_by_E": [round(float(x), 3) for x in lo95],
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=min(12, max(4, (os.cpu_count() or 8) - 4)))
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    cells = get_cells()
    cell_keys = sorted(cells)
    rho = {k: cells[k]["rho"] for k in cell_keys}
    diagnosable = [k for k in cell_keys if rho[k] > 0.0]
    cidx = {k: i for i, k in enumerate(cell_keys)}
    nC, nS = len(cell_keys), len(SEEDS)

    # ---- verification sample drawn from the PAPER's own configuration -------
    vrng = random.Random(SEED)
    paper_seeds = [7, 18, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129]
    vsample = [(vrng.choice(cell_keys), vrng.choice(CONDITIONS),
                vrng.choice(paper_seeds),
                vrng.choice(("uniform", "random", "front", "back",
                             "geometric", "oracle")),
                vrng.choice(BUDGETS)) for _ in range(96)]

    jobs = [("table", c, cond) for c in cell_keys for cond in CONDITIONS]
    jobs += [("verify", vsample[i::8]) for i in range(8)]

    tables: dict = {}
    v_ok = v_bad = 0
    v_diffs: list = []
    print(f"dispatching {len(jobs)} jobs on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_job, jobs):
            if res[0] == "table":
                tables[(res[1], res[2])] = res[3]
            else:
                v_ok += res[1]
                v_bad += res[2]
                v_diffs += res[3]
    print(f"tables built; verification vs s2_engine.s2_trial: {v_ok} match, "
          f"{v_bad} mismatch  ({time.time() - t0:.0f}s)", flush=True)
    if v_bad:
        print("MISMATCH EXAMPLES:", json.dumps(v_diffs, indent=1), flush=True)

    # ---- sweep --------------------------------------------------------------
    # store[(dist, cond)][(policy_id, B)] = 4 arrays over (cell, seed) in order
    policy_ids = list(BLIND) + ["oracle"]
    ray_levels = []
    for ray, bcoef in RAYS.items():
        for E in E_GRID:
            b = bcoef * E
            sg = math.sqrt(max(E * E - b * b, 0.0))
            ray_levels.append((ray, E, b, sg, max(E, 1.0),
                               f"aware_point|{ray}|{E:g}", f"aware_bayes|{ray}|{E:g}"))
            policy_ids += [f"aware_point|{ray}|{E:g}", f"aware_bayes|{ray}|{E:g}"]

    store: dict = {}
    for dist in ONSET_DISTS:
        for cond in CONDITIONS:
            store[(dist, cond)] = {(p, B): tuple(array("f") for _ in range(4))
                                   for p in policy_ids for B in BUDGETS}

    for dist in ONSET_DISTS:
        slots = {cond: store[(dist, cond)] for cond in CONDITIONS}
        for cell_key in cell_keys:
            for seed in SEEDS:
                tau = draw_tau(dist, cell_key, seed)
                zs = {ray: draw_z(dist, cell_key, seed, ray) for ray in RAYS}
                blind_sc = {(p, B): blind_schedule(p, B, cell_key, seed)
                            for p in BLIND for B in BUDGETS}
                oracle_sc = {B: point_schedule(tau, B) for B in BUDGETS}
                aware_sc = {}
                for ray, E, b, sg, s_eff, pid_pt, pid_by in ray_levels:
                    th = int(min(L_TOTAL - 1, max(0, round(tau + b + sg * zs[ray]))))
                    bs = bayes_schedules(th, s_eff)
                    for B in BUDGETS:
                        aware_sc[(pid_pt, B)] = point_schedule(th, B)
                        aware_sc[(pid_by, B)] = bs[B]
                combined = list(blind_sc.items()) + list(aware_sc.items())
                for cond in CONDITIONS:
                    st = tables[(cell_key, cond)]
                    slot = slots[cond]
                    for (pid, B), sc in combined:
                        _, miss, pen, fa, nck = evaluate(st, sc, tau)
                        a = slot[(pid, B)]
                        a[M_MISS].append(miss)
                        a[M_PEN].append(pen)
                        a[M_FA].append(fa)
                        a[M_NCK].append(nck)
                    for B in BUDGETS:
                        _, miss, pen, fa, nck = evaluate(st, oracle_sc[B], tau)
                        a = slot[("oracle", B)]
                        a[M_MISS].append(miss)
                        a[M_PEN].append(pen)
                        a[M_FA].append(fa)
                        a[M_NCK].append(nck)
        print(f"  swept {dist}  ({time.time() - t0:.0f}s)", flush=True)

    mats: dict = {}
    for k, slot in store.items():
        for pk, arrs in slot.items():
            mats[(k[0], k[1], pk[0], pk[1])] = np.stack(
                [np.frombuffer(a, dtype=np.float32).reshape(nC, nS) for a in arrs])

    groups = {"well_diagnosable": ("well_specified", diagnosable),
              "well_all": ("well_specified", cell_keys),
              "misspecified_all": ("misspecified", cell_keys)}

    def clusters(dist, gname, pol, B, metric):
        cond, cs = groups[gname]
        m = mats[(dist, cond, pol, B)][metric]
        return m[[cidx[c] for c in cs], :].astype(np.float64).mean(axis=0)

    def summarise(dist, gname, pol, B):
        return {"miss": round(float(clusters(dist, gname, pol, B, M_MISS).mean()), 4),
                "mean_pen": round(float(clusters(dist, gname, pol, B, M_PEN).mean()), 3),
                "fa": round(float(clusters(dist, gname, pol, B, M_FA).mean()), 4),
                "mean_checks": round(float(clusters(dist, gname, pol, B, M_NCK).mean()), 3)}

    out = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "project_seed": SEED,
            "seeds": {"first": SEEDS[0], "last": SEEDS[-1], "n": len(SEEDS)},
            "budgets": list(BUDGETS),
            "L_TOTAL": L_TOTAL,
            "probes_per_check": PROBES_PER_CHECK,
            "cells_rho": {k: rho[k] for k in cell_keys},
            "diagnosable_cells": diagnosable,
            "onset_distributions": ONSET_DISTS,
            "corruption_grid_rmse_steps": list(E_GRID),
            "rays": {k: f"bias = {v:.4f}*E, sigma = sqrt(E^2 - bias^2)"
                     for k, v in RAYS.items()},
            "aware_policies": {
                "aware_point": "checks at tauhat..tauhat+B-1 (onset-informed reference on the estimate)",
                "aware_bayes": "DP-optimal schedule under the policy's own belief N(tauhat, max(E,1)^2)"},
            "endpoints": "paper's miss / penalized delay (horizon 160-tau) / false alarm",
            "bootstrap": {"n": N_BOOT,
                          "cluster": "seed (50 clusters); cells averaged within a cluster"},
            "verification_vs_s2_engine": {"n": v_ok + v_bad, "match": v_ok,
                                          "mismatch": v_bad, "examples": v_diffs},
            "n_trial_equivalents": len(mats) * nC * nS,
        },
        "predetermination_check": {},
        "crossover": {},
        "aware_curves": {},
        "headline": {},
    }

    # ---- (i) is the 100 % miss an artifact of disjoint windows? -------------
    for dist in ONSET_DISTS:
        for gname in groups:
            out["predetermination_check"][f"{dist}|{gname}"] = {
                f"B{B}": {p: summarise(dist, gname, p, B)
                          for p in list(BLIND) + ["oracle"]}
                for B in BUDGETS}

    # ---- (ii) crossover ----------------------------------------------------
    rng = np.random.default_rng(SEED)
    gname = "well_diagnosable"
    for dist in ONSET_DISTS:
        for B in BUDGETS:
            up = clusters(dist, gname, "uniform", B, M_PEN)
            um = clusters(dist, gname, "uniform", B, M_MISS)
            for ray in RAYS:
                for pol in AWARE:
                    ids = [f"{pol}|{ray}|{E:g}" for E in E_GRID]
                    dp = np.stack([up - clusters(dist, gname, i, B, M_PEN) for i in ids], axis=1)
                    dm = np.stack([um - clusters(dist, gname, i, B, M_MISS) for i in ids], axis=1)
                    key = f"{dist}|B{B}|{ray}|{pol}"
                    out["crossover"][key] = {
                        "uniform_mean_pen": round(float(up.mean()), 3),
                        "uniform_miss": round(float(um.mean()), 4),
                        "penalized_delay": crossover(E_GRID, dp, rng),
                        "miss": crossover(E_GRID, dm, rng)}
                    out["aware_curves"][key] = {
                        "E": list(E_GRID),
                        "mean_pen": [round(float(clusters(dist, gname, i, B, M_PEN).mean()), 3) for i in ids],
                        "miss": [round(float(clusters(dist, gname, i, B, M_MISS).mean()), 4) for i in ids],
                        "fa": [round(float(clusters(dist, gname, i, B, M_FA).mean()), 4) for i in ids],
                        "mean_checks": [round(float(clusters(dist, gname, i, B, M_NCK).mean()), 3) for i in ids]}

    # ---- headline ----------------------------------------------------------
    for dist in ONSET_DISTS:
        blk = out["predetermination_check"][f"{dist}|well_diagnosable"]
        out["headline"][f"audit_early_miss|{dist}"] = {
            f"B{B}": blk[f"B{B}"]["geometric"]["miss"] for B in BUDGETS}
        out["headline"][f"uniform_miss|{dist}"] = {
            f"B{B}": blk[f"B{B}"]["uniform"]["miss"] for B in BUDGETS}
    for dist in ONSET_DISTS:
        for B in BUDGETS:
            for pol in AWARE:
                c = out["crossover"][f"{dist}|B{B}|noise|{pol}"]["penalized_delay"]
                out["headline"][f"crossover_pen_noise|{dist}|B{B}|{pol}"] = {
                    "E_star": c["crossover_E"], "ci95": c["crossover_E_ci95"],
                    "first_E_not_sig_better": c["first_E_not_significantly_better"]}

    out["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "onset_accuracy_sweep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---- console -----------------------------------------------------------
    print("\n=== miss rate, well-specified diagnosable cells ===")
    for dist in ONSET_DISTS:
        blk = out["predetermination_check"][f"{dist}|well_diagnosable"]
        print(f"  {dist}")
        for B in BUDGETS:
            r = blk[f"B{B}"]
            print(f"     B{B:<3} audit-early={r['geometric']['miss']:.3f} "
                  f"front={r['front']['miss']:.3f} uniform={r['uniform']['miss']:.3f} "
                  f"back={r['back']['miss']:.3f} oracle={r['oracle']['miss']:.3f}")
    print("\n=== crossover in onset RMSE (penalized delay, pure-noise ray) ===")
    for dist in ONSET_DISTS:
        for pol in AWARE:
            for B in BUDGETS:
                c = out["crossover"][f"{dist}|B{B}|noise|{pol}"]["penalized_delay"]
                print(f"  {dist:<22} {pol:<12} B{B:<3} E*={c['crossover_E']} "
                      f"CI={c['crossover_E_ci95']} first-ns={c['first_E_not_significantly_better']}")
    print(f"\nwrote {OUT / 'onset_accuracy_sweep.json'} ({out['meta']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
