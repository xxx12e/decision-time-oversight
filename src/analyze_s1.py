r"""S1 first-pass analysis: per-cell outcomes, regime classification (PREREG H3),
H2 placement-vs-budget contrast, clean-arm false-intervention rates.

Output: ..\results\s1\summary.json + console regime table.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

RES = Path(__file__).resolve().parent.parent / "results" / "s1"
BUDGETS = (1, 2, 3)
POLICIES = ("uniform", "random", "front", "back", "geometric", "suspicion", "oracle")


def load():
    rows = []
    for line in (RES / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    rows = load()
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["cell_committed"], r["cell_evidence"], r["cell_topology"])
        cells[key][(r["policy"], r["budget"])].append(r)

    summary = {}
    regime_counts = defaultdict(int)
    h2_cells = []

    for key in sorted(cells):
        cc, ev, topo = key
        arms = cells[key]
        base = arms[("baseline", 0)]
        cap_rounds = [t["capture_round"] for t in base]
        n = len(base)
        fc_base = sum(1 for t in base if t["false_consensus"]) / n
        captured = [c for c in cap_rounds if c is not None]
        med_c = median(captured) if len(captured) >= n / 2 else None

        cell = {"n_seeds": n, "fc_baseline": fc_base,
                "capture_rate": len(captured) / n, "median_capture_round": med_c,
                "policies": {}}
        for pid in POLICIES:
            for B in BUDGETS:
                trials = arms.get((pid, B), [])
                if not trials:
                    continue
                fc = sum(1 for t in trials if t["false_consensus"]) / len(trials)
                cell["policies"][f"{pid}@B{B}"] = {
                    "fc": fc,
                    "fc_prevention": fc_base - fc,
                    "timely_catch": sum(1 for t in trials if t["timely_committed_prune"]) / len(trials),
                    "normal_pruned_mean": sum(t["normal_pruned"] for t in trials) / len(trials),
                }

        if cc == 0:
            # clean arm: false-intervention accounting only
            cell["false_intervention"] = {
                k: sum(1 for t in v if t["prune_events"]) / len(v)
                for k, v in ((f"{p}@B{B}", arms.get((p, B), []))
                             for p in ("uniform", "geometric", "suspicion") for B in BUDGETS)
                if v}
            summary[f"{cc}|{ev}|{topo}"] = cell
            continue

        # regime classification (PREREG H3)
        pol = cell["policies"]
        oracle3 = pol.get("oracle@B3", {}).get("fc_prevention", 0.0)
        oracle_any = max(pol.get(f"oracle@B{B}", {}).get("fc_prevention", 0.0) for B in BUDGETS)
        uni1 = pol.get("uniform@B1", {}).get("fc_prevention", 0.0)
        best1 = max((pol.get(f"{p}@B1", {}).get("fc_prevention", -1.0), p)
                    for p in POLICIES if p != "oracle")
        uni23 = max(pol.get(f"uniform@B{B}", {}).get("fc_prevention", 0.0) for B in (2, 3))
        if fc_base < 0.10:
            regime = "benign"           # nothing to prevent
        elif oracle_any <= 0.05:
            regime = "ceiling_bound"
        elif best1[0] >= 2 * max(uni1, 0.02) and best1[0] >= 0.10:
            regime = "placement_critical"
        elif uni23 >= oracle_any - 0.05:
            regime = "budget_saturated"
        else:
            regime = "intermediate"
        cell["regime"] = regime
        cell["best_B1"] = {"policy": best1[1], "fc_prevention": best1[0]}
        regime_counts[regime] += 1

        # H2 material: windowed cells
        if med_c is not None and 3 <= med_c <= 10:
            bestp = best1[1]
            tc_best1 = pol.get(f"{bestp}@B1", {}).get("timely_catch", 0.0)
            tc_uni2 = pol.get("uniform@B2", {}).get("timely_catch", 0.0)
            h2_cells.append({"cell": f"{cc}|{ev}|{topo}", "best_B1_policy": bestp,
                             "timely_best_B1": tc_best1, "timely_uniform_B2": tc_uni2,
                             "delta": tc_best1 - tc_uni2})
        summary[f"{cc}|{ev}|{topo}"] = cell

    out = {"regime_counts": dict(regime_counts), "h2_windowed_cells": h2_cells,
           "cells": summary}
    (RES / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("=== REGIME COUNTS (60 committed cells) ===")
    for k, v in sorted(regime_counts.items()):
        print(f"  {k}: {v}")
    print(f"\n=== H2 WINDOWED CELLS ({len(h2_cells)}) ===")
    pos = sum(1 for h in h2_cells if h["delta"] > 0)
    for h in h2_cells:
        print(f"  {h['cell']}: best-B1 {h['best_B1_policy']} timely {h['timely_best_B1']:.2f} "
              f"vs uniform-B2 {h['timely_uniform_B2']:.2f} (d={h['delta']:+.2f})")
    print(f"  positive deltas: {pos}/{len(h2_cells)}")
    print("\n=== CAPTURE-SPEED CONTINUUM (median capture by committed count, random topo, ev .66) ===")
    for cc in (1, 2, 3, 4, 6):
        c = summary.get(f"{cc}|0.66|random", {})
        print(f"  committed={cc}: rate={c.get('capture_rate'):.2f} median_round={c.get('median_capture_round')} fc={c.get('fc_baseline'):.2f}")


if __name__ == "__main__":
    main()
