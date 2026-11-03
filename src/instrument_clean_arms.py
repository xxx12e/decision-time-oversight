r"""False-intervention rate per instrument arm, on the committed-free clean cells.

The instrument sweep shows that lowering the anomaly threshold shrinks the
ceiling-bound region. That is only good news if the extra prunes land on
committed senders; a threshold low enough to flag honest agents buys prevention
by breaking the panel. This measures the price on the clean arms, so the
sensitivity story can say which arms are honest improvements and which are not.

Merges `clean_false_intervention` into results/s1/instrument_sensitivity.json.
Usage: python -X utf8 instrument_clean_arms.py [--workers N] [--seeds N]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_engine import (  # noqa: E402
    Instrument, make_condition, s1_trial, sched_uniform, sched_geometric, sched_front,
)
from instrument_sensitivity import ARMS, EVIDENCE, TOPOS, BUDGETS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "s1"
POLICIES = ("uniform", "geometric", "front")


def _run(job):
    kind, theta, ev, topo, t, pid, B = job
    sched = {"uniform": sched_uniform, "geometric": sched_geometric,
             "front": sched_front}[pid](B)
    r = s1_trial(make_condition(0, ev, topo), t, sched,
                 instrument=Instrument(kind, theta))
    return pid, B, bool(r["prune_events"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=50)
    a = ap.parse_args()

    path = OUT / "instrument_sensitivity.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for kind, theta in ARMS:
        label = Instrument(kind, theta).label
        jobs = [(kind, theta, ev, topo, t, pid, B)
                for ev in EVIDENCE for topo in TOPOS for t in range(a.seeds)
                for pid in POLICIES for B in BUDGETS]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_run, jobs, chunksize=32))
        per = {}
        for pid, B, fired in rows:
            k = f"{pid}@B{B}"
            per.setdefault(k, [0, 0])
            per[k][0] += fired
            per[k][1] += 1
        rates = {k: v[0] / v[1] for k, v in sorted(per.items())}
        out[label] = {"by_policy": {k: round(v, 4) for k, v in rates.items()},
                      "max_false_intervention_rate": round(max(rates.values()), 4)}
        print(f"{label:14s} max clean false-intervention rate "
              f"{out[label]['max_false_intervention_rate']:.4f}", flush=True)

    doc["clean_false_intervention"] = {
        "note": "committed-free cells only; a 'false intervention' is any prune on "
                "a clean run. The perfect arm is 0 by construction, which is what "
                "makes it the honest upper envelope: it improves detection without "
                "buying it from the honest population.",
        "n_seeds": a.seeds, "policies": list(POLICIES), "arms": out}
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("merged into instrument_sensitivity.json")


if __name__ == "__main__":
    main()
