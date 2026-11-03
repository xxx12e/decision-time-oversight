r"""Is the regime map a property of the fault dynamics, or of one anomaly score
at one threshold?  (Instrument-sensitivity analysis.)

Every bound in Study 1 is an oracle-placement bound on a FIXED instrument, so a
weak instrument would make ceilings look intrinsic when they are merely the
detector's. This sweep holds placement machinery, seeds and cells fixed and
swaps only the detector:

  theta=0.35 .. 0.65   the deployed content-anomaly score, re-thresholded
  perfect              an upper-envelope detector: every committed sender that
                       has spoken in the visible prefix is flagged, no false
                       positives. Not deployable -- it exists to answer whether
                       a ceiling survives an instrument that cannot be improved.

The theta=0.50 arm IS the paper's instrument, so it must reproduce results/s1/
stats.json exactly; the script asserts that cell by cell before any swapped arm
is written out, and refuses to emit a results file if it does not.

Output: ../results/s1/instrument_sensitivity.json
Usage:  python -X utf8 instrument_sensitivity.py [--workers N] [--seeds N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_engine import (  # noqa: E402
    Instrument, THETA, make_condition, s1_trial,
    sched_uniform, sched_random, sched_front, sched_back, sched_geometric, sched_oracle,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "s1"
COUNTS = (1, 2, 3, 4, 6)
EVIDENCE = (0.58, 0.66, 0.74)
TOPOS = ("hub", "random", "small_world", "clustered")
BUDGETS = (1, 2, 3)
NONORACLE = ("uniform", "random", "front", "back", "geometric")
N_SEEDS = 50
RAND_SEED_BASE = 20260810

ARMS = [("theta", 0.35), ("theta", 0.40), ("theta", 0.45), ("theta", 0.50),
        ("theta", 0.55), ("theta", 0.60), ("theta", 0.65), ("perfect", 0.0)]


def sign_test_p(k_pos: int, n: int) -> float:
    return sum(comb(n, k) for k in range(k_pos, n + 1)) / 2 ** n if n else 1.0


def _sched(policy_id, B, t, cap_round):
    return {
        "uniform": lambda: sched_uniform(B),
        "random": lambda: sched_random(B, RAND_SEED_BASE + t * 100 + B),
        "front": lambda: sched_front(B),
        "back": lambda: sched_back(B),
        "geometric": lambda: sched_geometric(B),
        "oracle": lambda: sched_oracle(B, cap_round),
    }[policy_id]()


def _run(job):
    kind, theta, cc, ev, topo, t, policy_id, B, cap_round = job
    inst = Instrument(kind, theta)
    cond = make_condition(cc, ev, topo)
    if policy_id == "baseline":
        r = s1_trial(cond, t, frozenset(), instrument=inst)
    else:
        r = s1_trial(cond, t, _sched(policy_id, B, t, cap_round), instrument=inst)
    return {"cell": f"{cc}|{ev}|{topo}", "policy": policy_id, "budget": B,
            "trial": t, "false_consensus": bool(r["false_consensus"]),
            "timely": bool(r["timely_committed_prune"]),
            "capture_round": r["capture_round"]}


def sweep_arm(kind, theta, workers, seeds):
    """Baselines first (they define each cell's capture round for the oracle
    schedule), then every policy x budget, exactly as sweep_s1.py orders it."""
    cells = [(cc, ev, topo) for cc in COUNTS for ev in EVIDENCE for topo in TOPOS]
    base_jobs = [(kind, theta, cc, ev, topo, t, "baseline", 0, None)
                 for (cc, ev, topo) in cells for t in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        base = list(ex.map(_run, base_jobs, chunksize=16))
    cap = {}
    for r in base:
        cap[(r["cell"], r["trial"])] = r["capture_round"]

    jobs = []
    for (cc, ev, topo) in cells:
        key = f"{cc}|{ev}|{topo}"
        for t in range(seeds):
            for p in NONORACLE + ("oracle",):
                for B in BUDGETS:
                    jobs.append((kind, theta, cc, ev, topo, t, p, B, cap[(key, t)]))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        runs = list(ex.map(_run, jobs, chunksize=16))
    return base, runs


def summarize(base, runs, seeds):
    """Regime labels + the H2 windowed contrast, by the same registered rules
    stats_s1.py applies -- reimplemented here only because this sweep never
    writes trials.jsonl."""
    fc_base, caps = defaultdict(list), defaultdict(list)
    for r in base:
        fc_base[r["cell"]].append(r["false_consensus"])
        if r["capture_round"] is not None:
            caps[r["cell"]].append(r["capture_round"])
    agg = defaultdict(lambda: defaultdict(list))
    for r in runs:
        agg[r["cell"]][(r["policy"], r["budget"])].append(r)

    cells_out, regimes = {}, defaultdict(int)
    for cell, arms in agg.items():
        b = sum(fc_base[cell]) / len(fc_base[cell])
        med = median(caps[cell]) if len(caps[cell]) >= len(fc_base[cell]) / 2 else None
        pol = {}
        for (p, B), rs in arms.items():
            pol[f"{p}@B{B}"] = {
                "fc_prevention": b - sum(x["false_consensus"] for x in rs) / len(rs),
                "timely": sum(x["timely"] for x in rs) / len(rs)}
        orc = max(pol[f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS)
        uni1 = pol["uniform@B1"]["fc_prevention"]
        best1, best1_pid = max((pol[f"{p}@B1"]["fc_prevention"], p) for p in NONORACLE)
        uni23 = max(pol[f"uniform@B{B}"]["fc_prevention"] for B in (2, 3))
        if b < 0.10:
            reg = "benign"
        elif orc <= 0.05:
            reg = "ceiling_bound"
        elif best1 >= 2 * max(uni1, 0.02) and best1 >= 0.10:
            reg = "placement_critical"
        elif uni23 >= orc - 0.05:
            reg = "budget_saturated"
        else:
            reg = "intermediate"
        regimes[reg] += 1
        cells_out[cell] = {"fc_baseline": b, "median_capture": med, "regime": reg,
                           "oracle_any": orc, "best_B1": best1_pid, "policies": pol}

    win = sorted(c for c, v in cells_out.items()
                 if v["median_capture"] is not None and 3 <= v["median_capture"] <= 10)
    h2 = {}
    for endpoint in ("timely", "fc_prevention"):
        ds = []
        for c in win:
            v = cells_out[c]
            ds.append(v["policies"][f"{v['best_B1']}@B1"][endpoint]
                      - v["policies"]["uniform@B2"][endpoint])
        pos = sum(1 for d in ds if d > 1e-12)
        neg = sum(1 for d in ds if d < -1e-12)
        h2[endpoint] = {"n_windowed": len(ds), "k_positive": pos, "k_negative": neg,
                        "n_ties": len(ds) - pos - neg,
                        "sign_p_one_sided": sign_test_p(pos, pos + neg),
                        "mean_delta": sum(ds) / len(ds) if ds else None}

    cb = [c for c, v in cells_out.items() if v["regime"] == "ceiling_bound"]
    cb_capt = sum(sum(fc_base[c]) for c in cb)
    cb_prev = 0.0
    for c in cb:
        v = cells_out[c]
        cb_prev += max(v["policies"][f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS) \
            * len(fc_base[c])
    risk = [c for c, v in cells_out.items() if v["regime"] != "benign"]
    return {
        "regime_counts": dict(regimes),
        "regime_shares": {k: v / len(cells_out) for k, v in regimes.items()},
        "h2": h2,
        "ceiling_bound_cells": sorted(cb),
        "ceiling_bound_baseline_captures": cb_capt,
        "ceiling_bound_oracle_prevention_share": (cb_prev / cb_capt) if cb_capt else None,
        "median_oracle_prevention_over_risk_cells":
            median([cells_out[c]["oracle_any"] for c in risk]) if risk else None,
        "n_risk_cells": len(risk),
        "cells": {c: {k: v[k] for k in ("fc_baseline", "median_capture", "regime",
                                        "oracle_any", "best_B1")}
                  for c, v in cells_out.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    published = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    out = {"note": "placement machinery, cells and seeds held fixed; only the "
                   "detector a check runs is swapped. theta=0.50 is the paper's "
                   "own instrument and must reproduce results/s1/stats.json.",
           "deployed_theta": THETA, "n_seeds": a.seeds, "arms": {}}

    for kind, theta in ARMS:
        label = Instrument(kind, theta).label
        print(f"[{label}] sweeping ...", flush=True)
        base, runs = sweep_arm(kind, theta, a.workers, a.seeds)
        out["arms"][label] = summarize(base, runs, a.seeds)
        s = out["arms"][label]
        print(f"[{label}] regimes={s['regime_counts']} "
              f"H2 timely {s['h2']['timely']['k_positive']}+/"
              f"{s['h2']['timely']['k_negative']}- "
              f"mean={s['h2']['timely']['mean_delta']:+.3f} "
              f"ceiling prevention={s['ceiling_bound_oracle_prevention_share']}", flush=True)

    # ---- the reproduction gate: the deployed arm must be the published map ----
    rep, mism = out["arms"][f"theta={THETA:.2f}"], []
    for cell, v in published["cells"].items():
        got = rep["cells"].get(cell)
        if got is None or got["regime"] != v["regime"]:
            mism.append((cell, v["regime"], got and got["regime"]))
    pub_h2 = published["h2"]["best"]
    h2_ok = (rep["h2"]["timely"]["k_positive"] == pub_h2["k_positive"]
             and rep["h2"]["timely"]["n_windowed"] == pub_h2["n_windowed"]
             and abs(rep["h2"]["timely"]["mean_delta"] - pub_h2["mean_delta"]) < 1e-9)
    # BOTH registered sign tests, not just timely: the paper says "both sign
    # tests to the digit", so the gate has to assert prevention too.
    pub_pv = published["h2"]["registered_family"]["tests"]["vs_uniform@B2|prevention"]
    pv = rep["h2"]["fc_prevention"]
    pv_ok = (pv["k_positive"] == pub_pv["k_positive"]
             and pv["n_ties"] == pub_pv["n_ties"]
             and abs(pv["mean_delta"] - pub_pv["mean_delta"]) < 1e-9
             and pv["sign_p_one_sided"] == pub_pv["sign_p_one_sided"])
    out["reproduction_check"] = {
        "regime_labels_match": not mism, "mismatches": mism,
        "h2_matches": bool(h2_ok), "h2_prevention_matches": bool(pv_ok),
        "passes": bool(not mism and h2_ok and pv_ok)}
    if not out["reproduction_check"]["passes"]:
        print("REPRODUCTION FAILED -- swapped arms are not comparable:", mism, h2_ok)
        (OUT / "instrument_sensitivity_FAILED.json").write_text(
            json.dumps(out, indent=1), encoding="utf-8")
        raise SystemExit(1)

    (OUT / "instrument_sensitivity.json").write_text(json.dumps(out, indent=1),
                                                     encoding="utf-8")
    print("\nreproduction check PASSED; wrote instrument_sensitivity.json")
    print("\narm            placement ceiling saturated benign  ceilingPrev  H2mean")
    for label, s in out["arms"].items():
        c = s["regime_counts"]
        cp = s["ceiling_bound_oracle_prevention_share"]
        print(f"{label:14s} {c.get('placement_critical',0):9d} {c.get('ceiling_bound',0):7d} "
              f"{c.get('budget_saturated',0):9d} {c.get('benign',0):6d} "
              f"{('%.3f' % cp) if cp is not None else '  n/a':>12s} "
              f"{s['h2']['timely']['mean_delta']:+.3f}")


if __name__ == "__main__":
    main()
