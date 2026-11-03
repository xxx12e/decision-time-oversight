r"""S2 full sweep: 10 cells x {well,mis} x 24 seeds x 6 policies x 5 budgets.

Usage: python -X utf8 sweep_s2.py [--workers N]
Output: ..\results\s2\trials.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2_engine import s2_trial, make_schedule, get_cells, S2_POLICIES  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "s2"
SEEDS = [7, 18, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 140, 151, 162,
         173, 184, 195, 206, 217, 228, 239, 250, 261]
BUDGETS = (1, 2, 4, 8, 16)
CONDITIONS = ("well_specified", "misspecified")


def _run(job):
    cell_key, cond, seed, policy, B = job
    r = s2_trial(cell_key, cond, seed, make_schedule(policy, B, cell_key, seed))
    r.update({"policy": policy, "budget": B})
    r.pop("schedule", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 4))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / "trials.jsonl"
    done = set()
    if out_f.exists():
        for line in out_f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["cell"], r["condition"], r["seed"], r["policy"], r["budget"]))

    cells = list(get_cells())
    jobs = [(c, cond, s, p, B)
            for c in cells for cond in CONDITIONS for s in SEEDS
            for p in S2_POLICIES for B in BUDGETS
            if (c, cond, s, p, B) not in done]
    print(f"jobs: {len(jobs)}", flush=True)
    with open(out_f, "a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(_run, jobs, chunksize=10)):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            if (i + 1) % 1000 == 0:
                fh.flush()
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    print("S2 SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
