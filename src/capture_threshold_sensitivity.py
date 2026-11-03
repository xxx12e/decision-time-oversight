r"""Capture-threshold sensitivity of the Study 1 regime map.

Protocol: prereg/PROTOCOL_SUPPLEMENT_S1_capture_threshold.md (hashed before running).
Capture is registered as >= 80% of honest agents adopting the false claim. This
re-runs the sweep at 0.60 / 0.70 / 0.80 / 0.90 with the deployed detector, the
intercept convention, and the registered regime rules unchanged. The 0.80 arm must
reproduce results/s1/stats.json before any other arm is written.

Output: ../results/s1/capture_threshold_sensitivity.json
Usage:  python -X utf8 capture_threshold_sensitivity.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s1_engine  # noqa: E402
from s1_engine import make_condition, s1_trial  # noqa: E402
from instrument_sensitivity import (  # noqa: E402
    COUNTS, EVIDENCE, TOPOS, BUDGETS, NONORACLE, N_SEEDS, _sched, summarize)

OUT = Path(__file__).resolve().parent.parent / "results" / "s1"
THRESHOLDS = (0.60, 0.70, 0.80, 0.90)


def _run(job):
    thr, cc, ev, topo, t, policy_id, B, cap_round = job
    s1_engine._CONFIG_TEMPLATE["consensus_threshold"] = thr
    cond = make_condition(cc, ev, topo)
    sched = frozenset() if policy_id == "baseline" else _sched(policy_id, B, t, cap_round)
    r = s1_trial(cond, t, sched)
    return {"cell": f"{cc}|{ev}|{topo}", "policy": policy_id, "budget": B,
            "trial": t, "false_consensus": bool(r["false_consensus"]),
            "timely": bool(r["timely_committed_prune"]),
            "capture_round": r["capture_round"]}


def sweep(thr, workers, seeds):
    cells = [(cc, ev, topo) for cc in COUNTS for ev in EVIDENCE for topo in TOPOS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        base = list(ex.map(_run, [(thr, cc, ev, topo, t, "baseline", 0, None)
                                  for (cc, ev, topo) in cells for t in range(seeds)],
                           chunksize=16))
    cap = {(r["cell"], r["trial"]): r["capture_round"] for r in base}
    jobs = [(thr, cc, ev, topo, t, p, B, cap[(f"{cc}|{ev}|{topo}", t)])
            for (cc, ev, topo) in cells for t in range(seeds)
            for p in NONORACLE + ("oracle",) for B in BUDGETS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        runs = list(ex.map(_run, jobs, chunksize=16))
    return base, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    published = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    out = {"protocol": "prereg/PROTOCOL_SUPPLEMENT_S1_capture_threshold.md",
           "registered_threshold": 0.80, "arms": {}}
    for thr in THRESHOLDS:
        base, runs = sweep(thr, a.workers, N_SEEDS)
        s = summarize(base, runs, N_SEEDS)
        out["arms"][f"{thr:.2f}"] = s
        print(f"[capture>={thr:.2f}] regimes={s['regime_counts']} "
              f"windowed={s['h2']['timely']['n_windowed']} "
              f"timely {s['h2']['timely']['k_positive']}+/{s['h2']['timely']['k_negative']}- "
              f"mean={s['h2']['timely']['mean_delta']:+.3f} | prevention "
              f"{s['h2']['fc_prevention']['k_positive']}+/{s['h2']['fc_prevention']['k_negative']}- "
              f"mean={s['h2']['fc_prevention']['mean_delta']:+.3f}", flush=True)

    rep = out["arms"]["0.80"]
    mism = [c for c, v in published["cells"].items()
            if rep["cells"].get(c, {}).get("regime") != v["regime"]]
    pb = published["h2"]["best"]
    pv = published["h2"]["registered_family"]["tests"]["vs_uniform@B2|prevention"]
    ok = (not mism
          and rep["h2"]["timely"]["k_positive"] == pb["k_positive"]
          and abs(rep["h2"]["timely"]["mean_delta"] - pb["mean_delta"]) < 1e-9
          and rep["h2"]["fc_prevention"]["k_positive"] == pv["k_positive"]
          and abs(rep["h2"]["fc_prevention"]["mean_delta"] - pv["mean_delta"]) < 1e-9)
    out["reproduction_check"] = {"passes": bool(ok), "label_mismatches": mism}
    if not ok:
        print("REPRODUCTION FAILED at 0.80:", mism)
        raise SystemExit(1)
    (OUT / "capture_threshold_sensitivity.json").write_text(json.dumps(out, indent=1),
                                                            encoding="utf-8")
    print("reproduction check PASSED; wrote capture_threshold_sensitivity.json")


if __name__ == "__main__":
    main()
