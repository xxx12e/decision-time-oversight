r"""S1 full placement x budget sweep (CPU, multiprocessing).

Grid (PREREG_P1 S1): committed {1,2,3,4,6} x evidence {0.58,0.66,0.74} x
topology {hub,random,small_world,clustered}; policies uniform/random/front/back/
geometric/suspicion/oracle x budgets {1,2,3}; 50 seeds; clean arms (0 committed)
for false-intervention rates + suspicion delta calibration (95th pct clean shift).

Usage: python -X utf8 sweep_s1.py [--tiny] [--workers N]
Output: ..\results\s1\trials.jsonl (+ calibration.json), resumable per (cell,seed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_engine import (  # noqa: E402
    T_ROUNDS, make_condition, s1_trial, SuspicionPolicy,
    sched_uniform, sched_random, sched_front, sched_back, sched_geometric, sched_oracle,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "s1"
COUNTS = (1, 2, 3, 4, 6)
EVIDENCE = (0.58, 0.66, 0.74)
TOPOS = ("hub", "random", "small_world", "clustered")
BUDGETS = (1, 2, 3)
N_SEEDS = 50
RAND_SEED_BASE = 20260810


def _clean_key(ev, topo):
    return f"{topo}|{ev}"


def calibrate_deltas(workers: int, seeds: int) -> dict[str, float]:
    """95th percentile of max round-mean-confidence shift on clean no-check arms."""
    jobs = [(0, ev, topo, t) for ev in EVIDENCE for topo in TOPOS for t in range(seeds)]
    shifts: dict[str, list[float]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (cc, ev, topo, t), res in zip(jobs, ex.map(_run_baseline, jobs, chunksize=8)):
            mc = res["mean_conf_by_round"]
            shifts.setdefault(_clean_key(ev, topo), []).append(max(m - mc[0] for m in mc))
    deltas = {}
    for k, v in shifts.items():
        vs = sorted(v)
        deltas[k] = vs[min(len(vs) - 1, int(0.95 * len(vs)))]
    return deltas


def _run_baseline(job):
    cc, ev, topo, t = job
    return s1_trial(make_condition(cc, ev, topo), t, frozenset())


def _run_one(job):
    cc, ev, topo, t, policy_id, B, delta, cap_round = job
    cond = make_condition(cc, ev, topo)
    if policy_id == "baseline":
        r = s1_trial(cond, t, frozenset())
    elif policy_id == "suspicion":
        r = s1_trial(cond, t, frozenset(), policy=SuspicionPolicy(B, delta))
    else:
        sched = {
            "uniform": lambda: sched_uniform(B),
            "random": lambda: sched_random(B, RAND_SEED_BASE + t * 100 + B),
            "front": lambda: sched_front(B),
            "back": lambda: sched_back(B),
            "geometric": lambda: sched_geometric(B),
            "oracle": lambda: sched_oracle(B, cap_round),
        }[policy_id]()
        r = s1_trial(cond, t, sched)
    r.update({"policy": policy_id, "budget": B, "cell_committed": cc,
              "cell_evidence": ev, "cell_topology": topo, "suspicion_delta": delta})
    # drop bulky per-round arrays for policy runs; keep for baselines
    if policy_id != "baseline":
        r.pop("mean_conf_by_round", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 4))
    a = ap.parse_args()
    counts, evidence, topos, seeds = COUNTS, EVIDENCE, TOPOS, N_SEEDS
    if a.tiny:
        counts, evidence, topos, seeds = (2, 6), (0.66,), ("hub", "random"), 4

    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / ("trials_tiny.jsonl" if a.tiny else "trials.jsonl")
    done = set()
    if out_f.exists():
        for line in out_f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["cell_committed"], r["cell_evidence"], r["cell_topology"],
                          r["trial"], r["policy"], r["budget"]))

    print(f"calibrating suspicion deltas ({a.workers} workers)...", flush=True)
    deltas = calibrate_deltas(a.workers, seeds)
    (OUT / "calibration.json").write_text(json.dumps(deltas, indent=1), encoding="utf-8")
    print(f"deltas: {deltas}", flush=True)

    # pass 1: baselines (committed cells + clean arms) -> capture rounds for oracle
    base_jobs = [(cc, ev, topo, t) for cc in counts for ev in evidence
                 for topo in topos for t in range(seeds)]
    base_jobs += [(0, ev, topo, t) for ev in evidence for topo in topos for t in range(seeds)]
    cap: dict[tuple, int | None] = {}
    with open(out_f, "a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        for job, res in zip(base_jobs, ex.map(_run_baseline, base_jobs, chunksize=8)):
            cc, ev, topo, t = job
            cap[(cc, ev, topo, t)] = res["capture_round"]
            key = (cc, ev, topo, t, "baseline", 0)
            if key not in done:
                res.update({"policy": "baseline", "budget": 0, "cell_committed": cc,
                            "cell_evidence": ev, "cell_topology": topo, "suspicion_delta": None})
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"baselines done: {len(base_jobs)}", flush=True)

        # pass 2: policy runs
        POLICIES = ("uniform", "random", "front", "back", "geometric", "suspicion", "oracle")
        CLEAN_POLICIES = ("uniform", "geometric", "suspicion")
        jobs = []
        for cc in counts:
            for ev in evidence:
                for topo in topos:
                    for t in range(seeds):
                        for pid in POLICIES:
                            for B in BUDGETS:
                                if (cc, ev, topo, t, pid, B) in done:
                                    continue
                                jobs.append((cc, ev, topo, t, pid, B,
                                             deltas[_clean_key(ev, topo)],
                                             cap[(cc, ev, topo, t)]))
        for ev in evidence:
            for topo in topos:
                for t in range(seeds):
                    for pid in CLEAN_POLICIES:
                        for B in BUDGETS:
                            if (0, ev, topo, t, pid, B) in done:
                                continue
                            jobs.append((0, ev, topo, t, pid, B,
                                         deltas[_clean_key(ev, topo)],
                                         cap[(0, ev, topo, t)]))
        print(f"policy jobs: {len(jobs)}", flush=True)
        for i, res in enumerate(ex.map(_run_one, jobs, chunksize=16)):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            if (i + 1) % 2000 == 0:
                fh.flush()
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
