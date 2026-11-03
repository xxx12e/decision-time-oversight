r"""Non-unit check cost + interface-level alarm burden (supplementary analysis).

Reviewer objection, verbatim in spirit: "the attention budget does not model human
attention. A check costs one unit whether it faces 5 agents or 32, whether the
prefix is 1 round or 14. And agent-level FPR is not interface-level alarm burden."

PART A -- non-unit cost model.  A check fired at round t in trial tau is priced

    cost(t, tau) = a + b * prefix_len(t) + c * n_flagged(t, tau)

with, under the INTERCEPT convention that the Study-1 engine implements (a check
at round t sees rounds 1..t and acts before beliefs absorb round t),

    prefix_len(t) = t                      rounds of transcript the operator reads
    n_flagged(t, tau) = |pruned set after the check|   the standing alarm list the
                                                       operator must adjudicate

Both quantities are read exactly off the archived trial records (`checked_rounds`
and `prune_events`); no re-simulation is needed to price the paper's own 72,000
trials.  Study 1 holds the panel fixed at N=32 agents, so the reviewer's
"5 agents or 32" axis is *not* sweepable inside S1 -- it is absorbed into a --
and is answered instead in Part B, which does span panel sizes 5..32.

The total budget is held fixed in COST units, not in check counts.  Because the
answer depends on what "the same budget" means once checks stop being fungible,
TWO anchors are reported, and they bracket the honest range:

  uniform-anchored  K_B = a*B + b*sum(uniform_B rounds) + c*B*F_bar
                    "the operator has as much attention as the paper's own
                    uniform-B schedule consumes at the average alarm load"
  early-anchored    K_B = B * (a + b*2 + c*F_bar)
                    "the operator has as much attention as B cheapest-possible
                    (round-2) checks" -- the stress test, under which any later
                    look costs strictly more than one nominal unit

Both give K_B == B exactly under the unit model (a,b,c)=(1,0,0), so the unit arm
reproduces the published numbers as a self-check.  Every policy is then allowed
the largest nominal budget whose cost fits K_B; budgets 4..6 are simulated fresh
so a cheap early-looking policy can actually buy the extra looks that a
length-sensitive price makes affordable.  Two affordability rules are reported:
`plan` (flags priced at F_bar -- no per-trial clairvoyance, the deployable rule)
and `realized` (flags priced at that trial's actual alarm list -- a reference).
The preregistered regime rules and the registered H2 contrast are then re-derived
verbatim with cost envelopes substituted for check counts.

PART B -- interface-level alarm burden.  The paper's monitor operating point is a
matched *per-agent* FPR of 6.25%.  We convert that to operator-facing units:
expected flags per scored round and per trial, and the probability an operator
sees at least one false alarm, for every panel size the benchmark actually uses
(5, 7, 10, 15, 21 agents on the replay panels; 32 in Study 1), analytically and
empirically by replaying the monitor zoo at its matched threshold.  The Study-1
content check's own clean-arm alarm load is measured for contrast.

Reuses: s1_engine / sweep_s1 (simulation), s4_zoo (replay + matched threshold),
stats_s1 (sign test, bootstrap).  Modifies nothing.

Usage: python -X utf8 cost_model_sensitivity.py [--workers N] [--no-extend]
Output: ..\results\s1\cost_model_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_engine import T_ROUNDS, N_AGENTS, sched_uniform          # noqa: E402
from sweep_s1 import _run_one, _clean_key                        # noqa: E402
from stats_s1 import sign_test_p                                 # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results" / "s1"
OUT_F = RES / "cost_model_sensitivity.json"

COUNTS = (1, 2, 3, 4, 6)
EVIDENCE = (0.58, 0.66, 0.74)
TOPOS = ("hub", "random", "small_world", "clustered")
N_SEEDS = 50
POLICIES = ("uniform", "random", "front", "back", "geometric", "suspicion", "oracle")
NONORACLE = ("uniform", "random", "front", "back", "geometric", "suspicion")
CLEAN_POLICIES = ("uniform", "geometric", "suspicion")
BASE_BUDGETS = (1, 2, 3)          # already in results/s1/trials.jsonl
EXT_BUDGETS = (4, 5, 6)           # simulated here so cost budgets can buy more looks
ALL_BUDGETS = BASE_BUDGETS + EXT_BUDGETS
ENVELOPES = (1, 2, 3)             # budget indices B whose COST is held fixed
ANCHORS = ("uniform", "early")    # two ways to denominate "the same budget" in cost

SEED = 20260808                   # project seed convention
BOOT_N = 20_000

# (a, b, c): mildly -> strongly length-sensitive, plus an alarm-triage-dominated arm.
COST_SETTINGS = {
    # name          a     b      c      description
    "unit":        (1.0, 0.000, 0.00),   # paper's model; reproduces published numbers
    "mild":        (1.0, 0.025, 0.01),   # prefix ~20% of a mid-horizon check
    "moderate":    (1.0, 0.125, 0.05),   # prefix ~= fixed cost at mid-horizon
    "strong":      (1.0, 0.375, 0.15),   # prefix 3x the fixed cost at mid-horizon
    "length_prop": (0.2, 1.000, 0.30),   # cost ~ proportional to rounds read
    "alarm_heavy": (1.0, 0.050, 0.50),   # triage of the alarm list dominates
}
SETTING_NOTES = {
    "unit": "paper's model: cost=1 per check regardless of prefix or alarm list",
    "mild": "prefix term is ~20% of a check's cost at the mid-horizon (t=8)",
    "moderate": "prefix term equals the fixed cost at the mid-horizon",
    "strong": "prefix term is 3x the fixed cost at the mid-horizon",
    "length_prop": "cost nearly proportional to rounds read (operator reads the transcript)",
    "alarm_heavy": "adjudicating the alarm list dominates; prefix nearly free",
}

# matched per-agent FPR operating points used in the paper
FPR_GRID = (0.01, 0.02, 0.0625, 0.10, 0.20)
DEPLOYED_FPR = 0.0625
# (n_agents, n_committed, n_scored_rounds) for every panel the benchmark uses
PANELS = [
    ("replay_panel_5", 5, 2, 4),
    ("replay_panel_7", 7, 2, 4),
    ("replay_panel_10", 10, 2, 4),
    ("replay_panel_15", 15, 2, 4),
    ("replay_panel_21", 21, 2, 4),
    ("s1_benchmark_32", N_AGENTS, 3, T_ROUNDS - 1),
]


# ---------------------------------------------------------------- load / extend

def compact(r: dict) -> dict:
    return {"fc": bool(r["false_consensus"]),
            "tc": bool(r["timely_committed_prune"]),
            "cap": r["capture_round"],
            "checks": [int(x) for x in r["checked_rounds"]],
            "pe": sorted(int(e[0]) for e in r["prune_events"]),
            "normal_pruned": int(r["normal_pruned"])}


def load_archive() -> tuple[dict, dict]:
    """(cell, seed, policy, budget) -> compact record;  (cell, seed) -> baseline."""
    arms, base = {}, {}
    with open(RES / "trials.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            cell = f"{r['cell_committed']}|{r['cell_evidence']}|{r['cell_topology']}"
            k = compact(r)
            if r["policy"] == "baseline":
                base[(cell, r["trial"])] = k
            else:
                arms[(cell, r["trial"], r["policy"], r["budget"])] = k
    return arms, base


def extend_budgets(arms: dict, base: dict, workers: int) -> int:
    """Simulate budgets 4..6 (absent from the archive) for every policy."""
    deltas = json.loads((RES / "calibration.json").read_text(encoding="utf-8"))
    jobs = []
    for cc in COUNTS:
        for ev in EVIDENCE:
            for topo in TOPOS:
                for t in range(N_SEEDS):
                    cap = base[(f"{cc}|{ev}|{topo}", t)]["cap"]
                    for pid in POLICIES:
                        for B in EXT_BUDGETS:
                            jobs.append((cc, ev, topo, t, pid, B,
                                         deltas[_clean_key(ev, topo)], cap))
    for ev in EVIDENCE:
        for topo in TOPOS:
            for t in range(N_SEEDS):
                cap = base[(f"0|{ev}|{topo}", t)]["cap"]
                for pid in CLEAN_POLICIES:
                    for B in EXT_BUDGETS:
                        jobs.append((0, ev, topo, t, pid, B,
                                     deltas[_clean_key(ev, topo)], cap))
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for job, r in zip(jobs, ex.map(_run_one, jobs, chunksize=64)):
            cell = f"{job[0]}|{job[1]}|{job[2]}"
            arms[(cell, job[3], job[4], job[5])] = compact(r)
    print(f"  extended {len(jobs)} trials in {time.time() - t0:.0f}s", flush=True)
    return len(jobs)


# ------------------------------------------------------------------ cost model

def alarm_cum(pe: list[int], t: int) -> int:
    """Size of the standing alarm list right after the check at round t."""
    return sum(1 for r in pe if r <= t)


def cost_of(rec: dict, a: float, b: float, c: float, fbar: float, rule: str) -> float:
    if rule == "realized":
        return sum(a + b * t + c * alarm_cum(rec["pe"], t) for t in rec["checks"])
    return sum(a + b * t + c * fbar for t in rec["checks"])       # "plan"


def envelope(B: int, a: float, b: float, c: float, fbar: float, anchor: str) -> float:
    if anchor == "early":
        return B * (a + 2 * b + c * fbar)
    return a * B + b * sum(sched_uniform(B)) + c * B * fbar       # "uniform"


def cost_table(arms: dict, a: float, b: float, c: float, fbar: float, rule: str) -> dict:
    return {k: cost_of(rec, a, b, c, fbar, rule) for k, rec in arms.items()}


def affordable_budget(ct: dict, cell, seed, pid, K) -> int:
    """Largest nominal budget whose cost fits the envelope; 0 = cannot afford a check."""
    best = 0
    for B in ALL_BUDGETS:
        v = ct.get((cell, seed, pid, B))
        if v is not None and v <= K + 1e-9:
            best = B
    return best


# --------------------------------------------------------------- regime re-derivation

def cell_rates(arms, base, ct, cell, Ks):
    """Per policy, per cost envelope: (fc rate, timely rate, mean checks bought)."""
    fc_base = sum(base[(cell, s)]["fc"] for s in range(N_SEEDS)) / N_SEEDS
    out = {}
    for pid in POLICIES:
        row = {}
        for m, K in Ks.items():
            fc = tc = nb = nz = 0
            for s in range(N_SEEDS):
                B = affordable_budget(ct, cell, s, pid, K)
                rec = base[(cell, s)] if B == 0 else arms[(cell, s, pid, B)]
                fc += rec["fc"]
                tc += rec["tc"]
                nb += len(rec["checks"])
                nz += (B == 0)
            row[m] = {"fc": fc / N_SEEDS, "timely": tc / N_SEEDS,
                      "fc_prevention": fc_base - fc / N_SEEDS,
                      "mean_checks": nb / N_SEEDS,
                      "share_priced_out": nz / N_SEEDS}
        out[pid] = row
    return fc_base, out


def classify(fc_base: float, rates: dict) -> tuple[str, str, float]:
    """Preregistered H3 rule, verbatim, with cost envelopes in place of budgets."""
    oracle_any = max(rates["oracle"][m]["fc_prevention"] for m in ENVELOPES)
    uni1 = rates["uniform"][1]["fc_prevention"]
    best1_val, best1_pid = max((rates[p][1]["fc_prevention"], p) for p in NONORACLE)
    uni23 = max(rates["uniform"][m]["fc_prevention"] for m in (2, 3))
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
    return regime, best1_pid, best1_val


def boot_mean_ci(ds: list[float], seed: int = SEED, n: int = BOOT_N):
    rng = random.Random(seed)
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(n))
    return [bs[int(0.025 * n)], bs[int(0.975 * n)]]


# ------------------------------------------------------------------ alarm burden

def analytic_alarm_burden() -> dict:
    out = {}
    for name, n, ncomm, T in PANELS:
        nh = n - ncomm
        row = {"n_agents": n, "n_committed": ncomm, "n_honest": nh, "n_scored_rounds": T}
        for q in FPR_GRID:
            row[f"fpr_{q}"] = {
                "expected_flags_per_scored_round": q * nh,
                "p_any_flag_in_a_round": 1 - (1 - q) ** nh,
                "expected_flags_per_trial": q * nh * T,
                "p_any_flag_in_a_trial": 1 - (1 - q) ** (nh * T),
                "clean_rounds_between_false_alarms": (1 / (1 - (1 - q) ** nh)
                                                      if q > 0 else None)}
        out[name] = row
    return out


def empirical_alarm_burden() -> dict:
    """Replay the zoo at its matched 6.25% per-agent FPR and count what an operator
    would actually see, per panel size.  Strict convention (prefix rounds 1..t-1)."""
    from s4_zoo import MONITORS, load_trials, prefix_pairs, ROUNDS
    trials = load_trials(["qwen35", "gemma4"])
    for t in trials:
        ids = set()
        for r, d in t["mz"].items():
            ids |= set(d.keys())
        t["n_agents"] = len(ids)

    out = {"n_trials": len(trials), "convention": "strict (check at t sees rounds 1..t-1)",
           "monitors": {}}
    for mon in ("deployed_442", "conf_constancy", "high_conf_share"):
        fn = MONITORS[mon]
        post = []
        for t in trials:
            post += [(fn(h, s), c) for h, s, c in
                     prefix_pairs(t["mz"], ROUNDS + 1, t["committed"])]
        neg = sorted((s for s, c in post if not c), reverse=True)
        thr = neg[max(0, int(DEPLOYED_FPR * len(neg)) - 1)] + 1e-9 if neg else 0.5

        by_n = defaultdict(lambda: {"trials": 0, "rounds": 0, "flags": 0,
                                    "rounds_with_flag": 0, "trials_with_flag": 0,
                                    "honest_agent_rounds": 0})
        for t in trials:
            d = by_n[t["n_agents"]]
            d["trials"] += 1
            tflag = 0
            for td in range(2, ROUNDS + 1):      # actionable decision rounds 2..T
                pairs = prefix_pairs(t["mz"], td, t["committed"])
                if not pairs:
                    continue
                hon = [(fn(h, s)) for h, s, c in pairs if not c]
                if not hon:
                    continue
                d["rounds"] += 1
                d["honest_agent_rounds"] += len(hon)
                f = sum(1 for x in hon if x >= thr)
                d["flags"] += f
                d["rounds_with_flag"] += (f > 0)
                tflag += f
            d["trials_with_flag"] += (tflag > 0)
        rows, tot = {}, {"trials": 0, "rounds": 0, "flags": 0, "rounds_with_flag": 0,
                         "trials_with_flag": 0, "honest_agent_rounds": 0}
        for n in sorted(by_n):
            d = by_n[n]
            for k in tot:
                tot[k] += d[k]
            rows[f"panel_{n}"] = {
                "n_agents": n, "n_trials": d["trials"],
                "flags_per_trial": d["flags"] / d["trials"],
                "flags_per_scored_round": d["flags"] / d["rounds"] if d["rounds"] else None,
                "share_rounds_with_ge1_flag": d["rounds_with_flag"] / d["rounds"] if d["rounds"] else None,
                "share_trials_with_ge1_flag": d["trials_with_flag"] / d["trials"],
                "realized_per_agent_fpr": d["flags"] / d["honest_agent_rounds"]
                                          if d["honest_agent_rounds"] else None}
        rows["ALL"] = {
            "n_trials": tot["trials"],
            "flags_per_trial": tot["flags"] / tot["trials"],
            "flags_per_scored_round": tot["flags"] / tot["rounds"],
            "share_rounds_with_ge1_flag": tot["rounds_with_flag"] / tot["rounds"],
            "share_trials_with_ge1_flag": tot["trials_with_flag"] / tot["trials"],
            "realized_per_agent_fpr": tot["flags"] / tot["honest_agent_rounds"]}
        out["monitors"][mon] = {"matched_threshold": thr, "target_fpr": DEPLOYED_FPR,
                                "by_panel": rows}
    return out


def s1_clean_alarm_burden(arms: dict) -> dict:
    """What the Study-1 content check actually costs an operator on clean arms."""
    out = {}
    agg = defaultdict(lambda: {"n": 0, "flags": 0, "checks": 0, "any": 0})
    for (cell, seed, pid, B), rec in arms.items():
        if not cell.startswith("0|"):
            continue
        d = agg[f"{pid}@B{B}"]
        d["n"] += 1
        d["flags"] += rec["normal_pruned"]
        d["checks"] += len(rec["checks"])
        d["any"] += bool(rec["pe"])
    for k in sorted(agg):
        d = agg[k]
        out[k] = {"n_trials": d["n"],
                  "false_flags_per_trial": d["flags"] / d["n"],
                  "false_flags_per_check": d["flags"] / d["checks"] if d["checks"] else None,
                  "share_trials_with_ge1_false_flag": d["any"] / d["n"],
                  "checks_per_trial": d["checks"] / d["n"],
                  "implied_per_agent_per_check_fpr":
                      (d["flags"] / (d["checks"] * N_AGENTS)) if d["checks"] else None}
    return out


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--no-extend", action="store_true")
    a_ = ap.parse_args()

    t0 = time.time()
    print("loading archived S1 trials...", flush=True)
    arms, base = load_archive()
    print(f"  {len(arms)} policy arms, {len(base)} baselines ({time.time() - t0:.0f}s)",
          flush=True)

    n_ext = 0
    if not a_.no_extend:
        print(f"simulating budgets {EXT_BUDGETS} ({a_.workers} workers)...", flush=True)
        n_ext = extend_budgets(arms, base, a_.workers)

    budgets_present = sorted({k[3] for k in arms})
    # ---- F_bar: mean standing alarm list per check, committed cells, archived budgets
    tot = cnt = 0
    for (cell, seed, pid, B), rec in arms.items():
        if cell.startswith("0|") or B > 3:
            continue
        for t in rec["checks"]:
            tot += alarm_cum(rec["pe"], t)
            cnt += 1
    fbar = tot / cnt
    print(f"F_bar (mean alarm-list size per check, risk cells, B<=3) = {fbar:.3f} "
          f"over {cnt} checks", flush=True)

    cells = [f"{cc}|{ev}|{topo}" for cc in COUNTS for ev in EVIDENCE for topo in TOPOS]

    # ---- windowed cells (median capture in [3,10]) -- identical to stats_s1
    windowed, base_fc = [], {}
    for cell in cells:
        caps = [base[(cell, s)]["cap"] for s in range(N_SEEDS)]
        got = [c for c in caps if c is not None]
        med = median(got) if len(got) >= N_SEEDS / 2 else None
        base_fc[cell] = sum(base[(cell, s)]["fc"] for s in range(N_SEEDS)) / N_SEEDS
        if med is not None and 3 <= med <= 10:
            windowed.append(cell)

    results = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": SEED, "boot_n": BOOT_N,
            "cost_model": "cost(check at round t) = a + b*prefix_len(t) + c*n_flagged(t)",
            "prefix_len": "t (INTERCEPT convention: a check at round t sees rounds 1..t "
                          "and acts before beliefs absorb round t -- what the S1 engine does)",
            "n_flagged": "size of the standing alarm list after the check "
                         "(cumulative pruned senders), read off prune_events",
            "panel_size_note": f"Study 1 fixes the panel at N={N_AGENTS} agents, so the "
                               "reviewer's 5-vs-32 axis is absorbed into a and is not "
                               "sweepable inside S1; it is answered in alarm_burden, "
                               "which spans panels of 5,7,10,15,21 and 32 agents",
            "budget_envelopes": {
                "uniform": "K_B = a*B + b*sum(uniform_B check rounds) + c*B*F_bar -- the "
                           "cost the paper's own uniform-B schedule consumes at the "
                           "average alarm load",
                "early": "K_B = B*(a + 2b + c*F_bar) -- B cheapest-possible (round-2) "
                         "checks; the stress test, under which every later look costs "
                         "strictly more than one nominal unit"},
            "both_anchors_reduce_to": "K_B == B under the unit model (a,b,c)=(1,0,0)",
            "affordability_rules": {
                "plan": "flags priced at F_bar -- no per-trial clairvoyance (primary)",
                "realized": "flags priced at that trial's actual alarm list (reference)"},
            "policy_under_cost": "each policy is given the largest nominal budget in "
                                 f"{ALL_BUDGETS} whose cost fits the envelope; 0 means it "
                                 "cannot afford a single check and the run is unchecked",
            "budgets_available": budgets_present,
            "n_new_simulations": n_ext,
            "F_bar": fbar, "n_checks_for_F_bar": cnt,
            "n_risk_cells": len(cells), "n_seeds": N_SEEDS,
            "windowed_cells": windowed,
        },
        "cost_settings": {}, "regimes": {}, "h2_cost": {}, "checks_afforded": {},
    }

    # ---------------------------------------------------------------- sweep
    for name, (a, b, c) in COST_SETTINGS.items():
        results["cost_settings"][name] = {
            "a": a, "b": b, "c": c, "note": SETTING_NOTES[name],
            "price_of_one_check": {
                "at_round_2": a + 2 * b + c * fbar,
                "at_round_8": a + 8 * b + c * fbar,
                "at_round_15": a + 15 * b + c * fbar},
            "envelopes": {anc: {str(m): envelope(m, a, b, c, fbar, anc) for m in ENVELOPES}
                          for anc in ANCHORS},
        }
        for rule in ("plan", "realized"):
            ct = cost_table(arms, a, b, c, fbar, rule)
            for anchor in ANCHORS:
                Ks = {m: envelope(m, a, b, c, fbar, anchor) for m in ENVELOPES}
                tag = f"{anchor}|{rule}"
                counts = defaultdict(int)
                per_cell, rates_all = {}, {}
                for cell in cells:
                    fc_base, rates = cell_rates(arms, base, ct, cell, Ks)
                    regime, bp, bv = classify(fc_base, rates)
                    counts[regime] += 1
                    per_cell[cell] = {
                        "regime": regime, "best_K1_policy": bp, "best_K1_prevention": bv,
                        "fc_baseline": fc_base,
                        "uniform_K1_prevention": rates["uniform"][1]["fc_prevention"],
                        "oracle_any_prevention":
                            max(rates["oracle"][m]["fc_prevention"] for m in ENVELOPES)}
                    rates_all[cell] = rates
                shares = {k: v / len(cells) for k, v in counts.items()}
                node = {"counts": dict(counts), "shares": shares}
                if rule == "plan":
                    node["cells"] = per_cell
                results["regimes"].setdefault(name, {})[tag] = node

                # mean checks each policy can afford under each envelope, and how
                # often it is priced out of a check entirely (the mechanism behind
                # any regime movement)
                results["checks_afforded"].setdefault(name, {})[tag] = {
                    str(m): {p: sum(rates_all[cl][p][m]["mean_checks"] for cl in cells) / len(cells)
                             for p in POLICIES} for m in ENVELOPES}
                results.setdefault("share_priced_out", {}).setdefault(name, {})[tag] = {
                    str(m): {p: sum(rates_all[cl][p][m]["share_priced_out"] for cl in cells) / len(cells)
                             for p in POLICIES} for m in ENVELOPES}

                # ------------------------------------------------ H2 in cost units
                # placement arm: best non-oracle policy at cost envelope K1
                # comparator:    uniform at a DOUBLED cost envelope 2*K1
                K2x = 2 * Ks[1]
                d_tc, d_pv, rows = [], [], []
                for cell in windowed:
                    r = rates_all[cell]
                    b1tc = max(r[p][1]["timely"] for p in NONORACLE)
                    b1pv = max(r[p][1]["fc_prevention"] for p in NONORACLE)
                    b1chk = max(((r[p][1]["fc_prevention"], r[p][1]["mean_checks"])
                                 for p in NONORACLE))[1]
                    tc = fc = nb = 0
                    for s in range(N_SEEDS):
                        B = affordable_budget(ct, cell, s, "uniform", K2x)
                        rec = base[(cell, s)] if B == 0 else arms[(cell, s, "uniform", B)]
                        tc += rec["tc"]
                        fc += rec["fc"]
                        nb += len(rec["checks"])
                    u_tc = tc / N_SEEDS
                    u_pv = base_fc[cell] - fc / N_SEEDS
                    d_tc.append(b1tc - u_tc)
                    d_pv.append(b1pv - u_pv)
                    rows.append({"cell": cell,
                                 "best_K1_timely": b1tc, "uniform_2K1_timely": u_tc,
                                 "delta_timely": b1tc - u_tc,
                                 "best_K1_prevention": b1pv, "uniform_2K1_prevention": u_pv,
                                 "delta_prevention": b1pv - u_pv,
                                 "best_K1_prev_arm_mean_checks": b1chk,
                                 "uniform_2K1_mean_checks": nb / N_SEEDS})
                h2 = {"n_windowed": len(windowed), "envelope_K1": Ks[1],
                      "doubled_envelope": K2x}
                if rule == "plan":
                    h2["cells"] = rows
                for tg, ds in (("timely", d_tc), ("prevention", d_pv)):
                    pos = sum(1 for d in ds if d > 0)
                    neg = sum(1 for d in ds if d < 0)
                    h2[tg] = {"k_positive": pos, "k_negative": neg,
                              "n_ties": len(ds) - pos - neg,
                              "sign_p_one_sided": sign_test_p(pos, pos + neg) if pos + neg else 1.0,
                              "mean_delta": sum(ds) / len(ds),
                              "ci95_mean_delta": boot_mean_ci(ds),
                              "min_delta": min(ds), "max_delta": max(ds)}
                h2["best_K1_prev_arm_mean_checks"] = sum(r["best_K1_prev_arm_mean_checks"] for r in rows) / len(rows)
                h2["uniform_2K1_mean_checks"] = sum(r["uniform_2K1_mean_checks"]
                                                    for r in rows) / len(rows)
                results["h2_cost"].setdefault(name, {})[tag] = h2
        print(f"  [{name}] uniform-anchored K={[round(envelope(m, a, b, c, fbar, 'uniform'), 2) for m in ENVELOPES]}"
              f" early-anchored K={[round(envelope(m, a, b, c, fbar, 'early'), 2) for m in ENVELOPES]}",
              flush=True)

        # ------------------------------------------- anchor-free efficiency frontier
        # What does each fixed arm cost, and what does it buy per unit of cost?
        # No budget envelope is involved, so this cannot be an artifact of the anchor.
        ct = cost_table(arms, a, b, c, fbar, "realized")
        eff = {}
        for setname, cellset in (("windowed12", windowed), ("all60", cells)):
            row = {}
            for pid in POLICIES:
                for B in ALL_BUDGETS:
                    cst = pv = tc = 0.0
                    for cl in cellset:
                        fcb = base_fc[cl]
                        f = sum(arms[(cl, s, pid, B)]["fc"] for s in range(N_SEEDS)) / N_SEEDS
                        tc += sum(arms[(cl, s, pid, B)]["tc"] for s in range(N_SEEDS)) / N_SEEDS
                        pv += fcb - f
                        cst += sum(ct[(cl, s, pid, B)] for s in range(N_SEEDS)) / N_SEEDS
                    n = len(cellset)
                    cst, pv, tc = cst / n, pv / n, tc / n
                    row[f"{pid}@B{B}"] = {
                        "mean_cost_per_trial": cst, "mean_prevention": pv,
                        "mean_timely": tc,
                        "prevention_per_unit_cost": pv / cst if cst > 0 else None,
                        "timely_per_unit_cost": tc / cst if cst > 0 else None}
            eff[setname] = row
        results.setdefault("efficiency_frontier", {})[name] = eff
        # the paper's own headline pair, re-priced
        w = eff["windowed12"]
        results["cost_settings"][name]["headline_arm_costs"] = {
            "geometric@B1_cost": w["geometric@B1"]["mean_cost_per_trial"],
            "uniform@B2_cost": w["uniform@B2"]["mean_cost_per_trial"],
            "cost_ratio_uniformB2_over_geometricB1":
                w["uniform@B2"]["mean_cost_per_trial"] / w["geometric@B1"]["mean_cost_per_trial"],
            "geometric@B1_prevention": w["geometric@B1"]["mean_prevention"],
            "uniform@B2_prevention": w["uniform@B2"]["mean_prevention"],
            "geometric@B1_timely": w["geometric@B1"]["mean_timely"],
            "uniform@B2_timely": w["uniform@B2"]["mean_timely"],
            "note": "mean over the 12 windowed cells; realized-flag pricing"}

    # -------------------------------------------------- label stability vs paper
    ref = results["regimes"]["unit"]["uniform|plan"]["cells"]
    stab = {}
    for name in COST_SETTINGS:
        for anchor in ANCHORS:
            tag = f"{anchor}|plan"
            cur = results["regimes"][name][tag]["cells"]
            same = sum(1 for cl in cells if cur[cl]["regime"] == ref[cl]["regime"])
            conf = defaultdict(int)
            for cl in cells:
                if cur[cl]["regime"] != ref[cl]["regime"]:
                    conf[f"{ref[cl]['regime']} -> {cur[cl]['regime']}"] += 1
            stab.setdefault(name, {})[tag] = {
                "n_cells": len(cells), "same_label_as_unit_cost": same,
                "agreement": same / len(cells), "changes": dict(conf)}
    results["regime_label_stability_vs_unit_cost"] = stab

    # ------------------------------------------------------------ alarm burden
    print("computing alarm burden...", flush=True)
    results["alarm_burden"] = {
        "note": "agent-level FPR q converted to interface-level load. A 'scored round' "
                "is one round at which the interface scores every honest agent.",
        "analytic": analytic_alarm_burden(),
        "empirical_replay_zoo": empirical_alarm_burden(),
        "s1_clean_arm_content_check": s1_clean_alarm_burden(arms),
    }

    OUT_F.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT_F}  ({time.time() - t0:.0f}s total)")

    # ------------------------------------------------------------- console read-back
    for anchor in ANCHORS:
        tag = f"{anchor}|plan"
        print(f"\n=== REGIME SHARES ({anchor}-anchored budget, plan rule, 60 risk cells) ===")
        for name in COST_SETTINGS:
            sh = results["regimes"][name][tag]["shares"]
            st = results["regime_label_stability_vs_unit_cost"][name][tag]
            print(f"  {name:<12} " + "  ".join(
                f"{k[:9]}={sh.get(k, 0.0):.2f}" for k in
                ("placement_critical", "ceiling_bound", "budget_saturated",
                 "intermediate", "benign"))
                + f"   agree={st['agreement']:.2f} {st['changes'] or ''}")
        print(f"\n=== H2 IN COST UNITS ({anchor}-anchored): best@K1 vs uniform@2*K1, "
              f"12 windowed cells ===")
        for name in COST_SETTINGS:
            h = results["h2_cost"][name][tag]
            print(f"  {name:<12} timely {h['timely']['k_positive']}+/{h['timely']['k_negative']}- "
                  f"mean {h['timely']['mean_delta']:+.3f} p={h['timely']['sign_p_one_sided']:.1e} | "
                  f"prev {h['prevention']['k_positive']}+/{h['prevention']['k_negative']}- "
                  f"mean {h['prevention']['mean_delta']:+.3f} "
                  f"p={h['prevention']['sign_p_one_sided']:.1e} | checks: "
                  f"best@K1 {h['best_K1_prev_arm_mean_checks']:.2f} vs "
                  f"uniform@2K1 {h['uniform_2K1_mean_checks']:.2f}")
    print("\n=== MEAN CHECKS AFFORDED at K3 (uniform-anchored, plan) ===")
    for name in COST_SETTINGS:
        ca = results["checks_afforded"][name]["uniform|plan"]["3"]
        print(f"  {name:<12} " + "  ".join(f"{p[:4]}={ca[p]:.2f}" for p in POLICIES))
    print("\n=== ANCHOR-FREE: the paper's headline pair, re-priced (12 windowed cells) ===")
    for name in COST_SETTINGS:
        h = results["cost_settings"][name]["headline_arm_costs"]
        print(f"  {name:<12} geometric@B1 cost {h['geometric@B1_cost']:6.2f} "
              f"(prev {h['geometric@B1_prevention']:.3f}) vs uniform@B2 cost "
              f"{h['uniform@B2_cost']:6.2f} (prev {h['uniform@B2_prevention']:.3f}) "
              f"-> the winner is {h['cost_ratio_uniformB2_over_geometricB1']:.2f}x cheaper")
    print("\n=== PREVENTION PER UNIT COST (windowed12, realized pricing) ===")
    for name in COST_SETTINGS:
        e = results["efficiency_frontier"][name]["windowed12"]
        best = max((v["prevention_per_unit_cost"] or 0, k) for k, v in e.items())
        print(f"  {name:<12} geometric@B1={e['geometric@B1']['prevention_per_unit_cost']:.4f}  "
              f"uniform@B2={e['uniform@B2']['prevention_per_unit_cost']:.4f}  "
              f"uniform@B3={e['uniform@B3']['prevention_per_unit_cost']:.4f}  "
              f"best={best[1]} ({best[0]:.4f})")
    print("\n=== REALIZED (ex-post) PRICING: regime counts + H2, uniform anchor ===")
    for name in COST_SETTINGS:
        r = results["regimes"][name]["uniform|realized"]["counts"]
        h = results["h2_cost"][name]["uniform|realized"]
        po = results["share_priced_out"][name]["uniform|realized"]["1"]["geometric"]
        print(f"  {name:<12} {dict(sorted(r.items()))}  geometric priced out of K1 in "
              f"{po:.2f} of trials | H2 prev {h['prevention']['k_positive']}+/"
              f"{h['prevention']['k_negative']}- mean {h['prevention']['mean_delta']:+.3f}")

    print("\n=== ALARM BURDEN at matched per-agent FPR 6.25% (analytic) ===")
    for k, v in results["alarm_burden"]["analytic"].items():
        d = v["fpr_0.0625"]
        print(f"  {k:<18} honest={v['n_honest']:<3} flags/round={d['expected_flags_per_scored_round']:.2f} "
              f"P(any|round)={d['p_any_flag_in_a_round']:.3f} "
              f"flags/trial={d['expected_flags_per_trial']:.2f} "
              f"P(any|trial)={d['p_any_flag_in_a_trial']:.3f}")
    e = results["alarm_burden"]["empirical_replay_zoo"]["monitors"]["deployed_442"]["by_panel"]["ALL"]
    print(f"  EMPIRICAL deployed monitor: {e['flags_per_trial']:.2f} false flags/trial, "
          f"{e['flags_per_scored_round']:.2f}/round, "
          f"{e['share_rounds_with_ge1_flag']:.3f} of rounds carry >=1 alarm")


if __name__ == "__main__":
    main()
