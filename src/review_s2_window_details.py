r"""Supplement S2: window-rule details quoted in the manuscript (reads results/s1/review_s2/*.jsonl).

Counts that analysis.json reports only as rates, plus the per-cell comparison of the best fixed
schedule with the clairvoyant bound:
  - union-rule disagreements out of all multi-check outcomes of captured runs;
  - the round at which windows open in the twelve windowed cells;
  - in every cell with a preventable capture, how many runs the best single round and the best
    fixed schedule of up to three checks fall short of the clairvoyant bound;
  - the no-check capture rate on the windowed cells under the standard and delayed-onset faults.

Usage: python -X utf8 review_s2_window_details.py
Output: ../results/s1/review_s2/window_details.json
"""
from __future__ import annotations

import json
from collections import Counter

from review_s2_analyze import D, load
from review_s2_runs import ALL_SCHEDS, SCHED_INDEX


def main():
    A = json.loads((D / "analysis.json").read_text(encoding="utf-8"))
    windowed = A["windowed_cells"]
    out = {}
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD"):
        by = load(arm)
        disagree = total = 0
        opens = Counter()
        cells = {}
        for cell, trials in by.items():
            rows = list(trials.values())
            bound = 0
            prevented_by = Counter()
            for r in rows:
                if not r["base_fc"]:
                    continue
                bound += "0" in r["fc"]
                for s in ALL_SCHEDS:
                    prevented_by[s] += r["fc"][SCHED_INDEX[s]] == "0"
                W = {min(s) for s in ALL_SCHEDS if len(s) == 1 and r["fc"][SCHED_INDEX[s]] == "0"}
                if W and cell in windowed:
                    opens[min(W)] += 1
                for s in ALL_SCHEDS:
                    if len(s) > 1:
                        total += 1
                        disagree += (r["fc"][SCHED_INDEX[s]] == "0") != bool(W & s)
            if bound:
                best1 = max(v for s, v in prevented_by.items() if len(s) == 1)
                best3 = max(prevented_by.values())
                cells[cell] = {"bound_runs": bound, "best_single_runs": best1, "best_upto3_runs": best3}
        short1 = [c["bound_runs"] - c["best_single_runs"] for c in cells.values()]
        short3 = [c["bound_runs"] - c["best_upto3_runs"] for c in cells.values()]
        out[arm] = {
            "union_rule_disagreements": disagree, "multi_check_outcomes": total,
            "window_open_round_windowed": {str(k): v for k, v in sorted(opens.items())},
            "cells_with_preventable_capture": len(cells),
            "best_single_equals_bound": sum(x == 0 for x in short1),
            "best_single_max_shortfall_runs": max(short1),
            "best_upto3_equals_bound": sum(x == 0 for x in short3),
            "per_cell": cells,
        }
    # the example cell drawn in fig_windows (widest range of window ends among windowed cells)
    ex = {}
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD"):
        by = load(arm)
        def window_end(r):
            W = [min(s) for s in ALL_SCHEDS if len(s) == 1 and r["fc"][SCHED_INDEX[s]] == "0"]
            return max(W) if W else None
        ends = {c: {window_end(r) for r in by[c].values() if r["base_fc"]} - {None} for c in windowed}
        if arm == "BENCH_L0_STD":
            cell = max(windowed, key=lambda c: (len(ends[c]), c))
        rows = [r for r in by[cell].values() if r["base_fc"]]

        def prevented(sched):
            return sum(r["fc"][SCHED_INDEX[frozenset(sched)]] == "0" for r in rows)
        ex[arm] = {"cell": cell, "captured": len(rows),
                   "with_window": sum(any(r["fc"][SCHED_INDEX[s]] == "0" for s in ALL_SCHEDS if len(s) == 1) for r in rows),
                   "prevented_round2": prevented({2}), "prevented_round3": prevented({3}),
                   "prevented_rounds_5_10": prevented({5, 10})}
    out["example_cell"] = ex
    for arm in ("BENCH_L0_STD", "BENCH_L0_DELAY"):
        by = load(arm)
        rows = [r for c in windowed for r in by[c].values()]
        out.setdefault("base_capture_windowed", {})[arm] = sum(r["base_fc"] for r in rows) / len(rows)
    (D / "window_details.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD"):
        print(arm, {k: v for k, v in out[arm].items() if k != "per_cell"})
    print(out["base_capture_windowed"])


if __name__ == "__main__":
    main()
