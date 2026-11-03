r"""Exhaustive placement oracle for Study 1 (supersedes the
onset-heuristic "oracle" as the bound).

For every committed cell x seed, enumerate EVERY fixed schedule at B in {1,2,3}
over check rounds 2..15 (C(14,1)+C(14,2)+C(14,3) = 469 simulations per pair) and
record, per (cell, seed):

  tc_any        any schedule achieves a timely committed prune
  prev_any      any schedule averts false consensus (baseline captured)
  onset_tc      the onset-heuristic schedule's timely catch (recomputed)
  onset_prev    the onset-heuristic schedule's prevention (recomputed)

The exhaustive maximum is a true optimum over fixed placements by construction;
the comparison quantifies exactly how far the onset heuristic sits below it.

Usage:
  python -X utf8 true_oracle_s1.py --sample 4      # timing probe
  python -X utf8 true_oracle_s1.py --workers 20    # full 3,000-pair run (resumable)
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from s1_engine import (T_ROUNDS, make_condition, s1_trial, sched_oracle)

OUT = Path(__file__).resolve().parent.parent / "results" / "s1"
PAIRS_F = OUT / "true_oracle_pairs.jsonl"
AGG_F = OUT / "true_oracle.json"

COUNTS = (1, 2, 3, 4, 6)
EVIDENCE = (0.58, 0.66, 0.74)
TOPOS = ("hub", "random", "small_world", "clustered")
N_SEEDS = 50
ROUNDS = tuple(range(2, T_ROUNDS + 1))          # 14 candidate check rounds
BUDGETS = (1, 2, 3)


def enumerate_pair(job):
    cc, ev, topo, t = job
    cond = make_condition(cc, ev, topo)
    base = s1_trial(cond, t, frozenset())
    cap = base["capture_round"]
    row = {"cell": f"{cc}|{ev}|{topo}", "trial": t,
           "baseline_captured": bool(base["false_consensus"]),
           "capture_round": cap, "per_B": {}}
    for B in BUDGETS:
        tc_any = prev_any = False
        tc_scheds = 0
        n_scheds = 0
        for comb in itertools.combinations(ROUNDS, B):
            n_scheds += 1
            r = s1_trial(cond, t, frozenset(comb))
            tc = bool(r["timely_committed_prune"])
            pv = row["baseline_captured"] and not r["false_consensus"]
            tc_any |= tc
            prev_any |= pv
            tc_scheds += tc
            if tc_any and prev_any and B == BUDGETS[-1]:
                pass                      # keep counting tc_scheds; cheap anyway
        onset = s1_trial(cond, t, sched_oracle(B, cap))
        row["per_B"][str(B)] = {
            "tc_any": tc_any, "prev_any": prev_any,
            "tc_sched_frac": tc_scheds / n_scheds,
            "onset_tc": bool(onset["timely_committed_prune"]),
            "onset_prev": row["baseline_captured"] and not onset["false_consensus"],
        }
    return row


def aggregate() -> dict:
    rows = [json.loads(l) for l in PAIRS_F.read_text(encoding="utf-8").splitlines() if l.strip()]
    cells: dict[str, list] = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    agg = {"n_pairs": len(rows), "cells": {}, "totals": {}}
    tot = {f"B{B}": {"exh_tc": 0, "onset_tc": 0, "exh_prev": 0, "onset_prev": 0}
           for B in BUDGETS}
    n_capt = 0
    for cell, rs in cells.items():
        c = {"n": len(rs), "baseline_captured": sum(r["baseline_captured"] for r in rs)}
        for B in BUDGETS:
            k = str(B)
            c[f"B{B}"] = {
                "exhaustive_timely": sum(r["per_B"][k]["tc_any"] for r in rs) / len(rs),
                "onset_timely": sum(r["per_B"][k]["onset_tc"] for r in rs) / len(rs),
                "exhaustive_prevented": sum(r["per_B"][k]["prev_any"] for r in rs) / len(rs),
                "onset_prevented": sum(r["per_B"][k]["onset_prev"] for r in rs) / len(rs),
            }
            tot[f"B{B}"]["exh_tc"] += sum(r["per_B"][k]["tc_any"] for r in rs)
            tot[f"B{B}"]["onset_tc"] += sum(r["per_B"][k]["onset_tc"] for r in rs)
            tot[f"B{B}"]["exh_prev"] += sum(r["per_B"][k]["prev_any"] for r in rs)
            tot[f"B{B}"]["onset_prev"] += sum(r["per_B"][k]["onset_prev"] for r in rs)
        agg["cells"][cell] = c
        n_capt += c["baseline_captured"]
    agg["totals"] = {"baseline_captured": n_capt, **tot}
    # headline: how much headroom did the heuristic leave on the table?
    for B in BUDGETS:
        d = tot[f"B{B}"]
        agg["totals"][f"B{B}"]["onset_share_of_exhaustive_tc"] = (
            d["onset_tc"] / d["exh_tc"] if d["exh_tc"] else None)
        agg["totals"][f"B{B}"]["onset_share_of_exhaustive_prev"] = (
            d["onset_prev"] / d["exh_prev"] if d["exh_prev"] else None)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0, help="timing probe: run N pairs inline")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    if a.aggregate_only:
        AGG_F.write_text(json.dumps(aggregate(), indent=1), encoding="utf-8")
        print(f"aggregated -> {AGG_F}")
        return

    jobs = [(cc, ev, topo, t) for cc in COUNTS for ev in EVIDENCE
            for topo in TOPOS for t in range(N_SEEDS)]
    if a.sample:
        t0 = time.time()
        for j in jobs[:a.sample]:
            enumerate_pair(j)
        dt = (time.time() - t0) / a.sample
        print(f"per-pair {dt:.1f}s -> full {len(jobs)} pairs on {a.workers} workers "
              f"~ {len(jobs) * dt / a.workers / 60:.0f} min")
        return

    done = set()
    if PAIRS_F.exists():
        for line in PAIRS_F.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["cell"], r["trial"]))
    todo = [j for j in jobs if (f"{j[0]}|{j[1]}|{j[2]}", j[3]) not in done]
    print(f"{len(todo)} pairs to run ({len(done)} cached)", flush=True)
    t0 = time.time()
    with open(PAIRS_F, "a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, row in enumerate(ex.map(enumerate_pair, todo, chunksize=4)):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"{i+1}/{len(todo)} ({el/60:.0f} min, "
                      f"ETA {el / (i+1) * (len(todo)-i-1) / 60:.0f} min)", flush=True)
    AGG_F.write_text(json.dumps(aggregate(), indent=1), encoding="utf-8")
    print(f"done -> {AGG_F}", flush=True)


if __name__ == "__main__":
    main()
