r"""S1 verify-grade statistics (PREREG H2/H3): exact sign tests, seed-level
bootstrap CIs, per-topology regime stability, clean-arm false interventions.

Output: ..\results\s1\stats.json (the file verify_p1_numbers.py pins against).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import median

RES = Path(__file__).resolve().parent.parent / "results" / "s1"
BUDGETS = (1, 2, 3)
NONORACLE = ("uniform", "random", "front", "back", "geometric", "suspicion")
BOOT_N = 10_000
BOOT_SEED = 20260810


def sign_test_p(k_pos: int, n: int) -> float:
    """Exact one-sided binomial P(X >= k_pos | p=.5, n)."""
    return sum(comb(n, k) for k in range(k_pos, n + 1)) / 2 ** n


def boot_ci(values: list[float], stat=lambda v: sum(v) / len(v),
            n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    rng = random.Random(seed)
    pt = stat(values)
    bs = sorted(stat(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return pt, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def main():
    rows = [json.loads(x) for x in (RES / "trials.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["cell_committed"], r["cell_evidence"], r["cell_topology"])][(r["policy"], r["budget"])].append(r)

    out = {"h2": {}, "h3": {}, "clean": {}, "cells": {}}

    # ---- per-cell aggregates ----
    windowed = []
    per_topo_regimes = defaultdict(lambda: defaultdict(int))
    for key in sorted(k for k in cells if k[0] > 0):
        cc, ev, topo = key
        arms = cells[key]
        base = arms[("baseline", 0)]
        n = len(base)
        fc_base = sum(t["false_consensus"] for t in base) / n
        captured = [t["capture_round"] for t in base if t["capture_round"] is not None]
        med_c = median(captured) if len(captured) >= n / 2 else None

        def agg(pid, B):
            ts = arms.get((pid, B), [])
            if not ts:
                return None
            return {"fc_prevention": fc_base - sum(t["false_consensus"] for t in ts) / len(ts),
                    "timely": sum(t["timely_committed_prune"] for t in ts) / len(ts)}

        pol = {f"{p}@B{B}": agg(p, B) for p in NONORACLE + ("oracle",) for B in BUDGETS}
        pol = {k: v for k, v in pol.items() if v}
        oracle_any = max(pol[f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS)
        uni1 = pol["uniform@B1"]["fc_prevention"]
        best1_val, best1_pid = max((pol[f"{p}@B1"]["fc_prevention"], p) for p in NONORACLE)
        uni23 = max(pol[f"uniform@B{B}"]["fc_prevention"] for B in (2, 3))
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
        per_topo_regimes[topo][regime] += 1
        out["cells"][f"{cc}|{ev}|{topo}"] = {
            "fc_baseline": fc_base, "median_capture": med_c, "regime": regime,
            "best_B1": {"policy": best1_pid, "fc_prevention": best1_val},
            "oracle_any": oracle_any, "policies": pol}

        if med_c is not None and 3 <= med_c <= 10:
            # paired per-seed deltas: best-B1 (registered) and geometric-B1 (fixed a priori)
            b_best = {t["trial"]: t["timely_committed_prune"] for t in arms[(best1_pid, 1)]}
            b_geo = {t["trial"]: t["timely_committed_prune"] for t in arms[("geometric", 1)]}
            u2 = {t["trial"]: t["timely_committed_prune"] for t in arms[("uniform", 2)]}
            deltas_best = [int(b_best[s]) - int(u2[s]) for s in sorted(u2)]
            deltas_geo = [int(b_geo[s]) - int(u2[s]) for s in sorted(u2)]
            # PREREG H2 registers Holm over TWO budget contrasts (B=1 vs uniform at
            # B=2 and at B=3), on both endpoints. Both are carried here so the
            # correction has a family to run over instead of a single reported test.
            contrasts = {}
            for opp_B in (2, 3):
                u = {t["trial"]: t for t in arms[("uniform", opp_B)]}
                b = {t["trial"]: t for t in arms[(best1_pid, 1)]}
                seeds = sorted(u)
                contrasts[f"vs_uniform@B{opp_B}"] = {
                    "timely": [int(b[s]["timely_committed_prune"])
                               - int(u[s]["timely_committed_prune"]) for s in seeds],
                    # prevention delta per seed: uniform captured and the single
                    # check did not (+1), the reverse (-1), agreement (0)
                    "prevention": [int(u[s]["false_consensus"])
                                   - int(b[s]["false_consensus"]) for s in seeds]}
            windowed.append({
                "cell": f"{cc}|{ev}|{topo}", "best_B1_policy": best1_pid,
                "delta_best": sum(deltas_best) / len(deltas_best),
                "delta_geo": sum(deltas_geo) / len(deltas_geo),
                "ci_best": boot_ci(deltas_best), "ci_geo": boot_ci(deltas_geo),
                "budget_contrasts": {k: {e: sum(v) / len(v) for e, v in d.items()}
                                     for k, d in contrasts.items()}})

    # ---- H2: sign tests across windowed cells ----
    for tag in ("best", "geo"):
        ds = [w[f"delta_{tag}"] for w in windowed]
        pos = sum(1 for d in ds if d > 0)
        neg = sum(1 for d in ds if d < 0)
        p = sign_test_p(pos, pos + neg) if pos + neg else 1.0
        out["h2"][tag] = {"n_windowed": len(ds), "k_positive": pos, "k_negative": neg,
                          "n_ties": len(ds) - pos - neg, "sign_p_one_sided": p,
                          "mean_delta": sum(ds) / len(ds) if ds else None}
    out["h2"]["cells"] = windowed
    # ---- the registered H2 family: two budget contrasts x two endpoints ----
    # PREREG H2: "exact one-sided sign test across cells, Holm over the two budget
    # contrasts". Both contrasts are run and the correction is applied within each
    # endpoint, so nothing in the paper reports a single test out of a family of two.
    fam = {}
    for opp in ("vs_uniform@B2", "vs_uniform@B3"):
        for endpoint in ("timely", "prevention"):
            ds = [w["budget_contrasts"][opp][endpoint] for w in windowed]
            pos = sum(1 for d in ds if d > 1e-12)
            neg = sum(1 for d in ds if d < -1e-12)
            pt, lo, hi = boot_ci(ds)
            fam[f"{opp}|{endpoint}"] = {
                "n_windowed": len(ds), "k_positive": pos, "k_negative": neg,
                "n_ties": len(ds) - pos - neg,
                "sign_p_one_sided": sign_test_p(pos, pos + neg) if pos + neg else 1.0,
                "mean_delta": pt, "paired_boot_ci": [lo, hi]}
    holm = {}
    for endpoint in ("timely", "prevention"):
        keys = [f"vs_uniform@B{B}|{endpoint}" for B in (2, 3)]
        order = sorted(keys, key=lambda k: fam[k]["sign_p_one_sided"])
        run = 0.0
        for i, k in enumerate(order):                     # Holm step-down, m = 2
            run = max(run, fam[k]["sign_p_one_sided"] * (len(order) - i))
            holm[k] = min(1.0, run)
    out["h2"]["registered_family"] = {
        "note": "PREREG H2 family: best single check vs uniform at B=2 and at B=3, "
                "on timely catch and on capture prevention; Holm within endpoint. "
                "The latency arms in Section 4 are robustness, not family members.",
        "tests": fam, "holm_adjusted_p": holm,
        "max_adjusted_p": max(holm.values())}
    out["h2"]["registered_pass"] = (out["h2"]["best"]["sign_p_one_sided"] < 0.05
                                    and max(holm.values()) < 0.05)

    # ---- H3: regime shares + topology stability ----
    totals = defaultdict(int)
    for topo, d in per_topo_regimes.items():
        for k, v in d.items():
            totals[k] += v
    n_cells = sum(totals.values())
    shares = {k: v / n_cells for k, v in totals.items()}
    named = ("ceiling_bound", "placement_critical", "budget_saturated")
    stability = {r: sum(1 for topo in per_topo_regimes if per_topo_regimes[topo].get(r, 0) > 0)
                 for r in named}
    out["h3"] = {"counts": dict(totals), "shares": shares,
                 "per_topology": {t: dict(d) for t, d in per_topo_regimes.items()},
                 "ge10pct": {r: shares.get(r, 0) >= 0.10 for r in named},
                 "present_in_topologies": stability,
                 "registered_pass": all(shares.get(r, 0) >= 0.10 for r in named)
                                    and all(stability[r] >= 3 for r in named)}

    # ---- clean arms ----
    for key in sorted(k for k in cells if k[0] == 0):
        _, ev, topo = key
        arms = cells[key]
        out["clean"][f"{ev}|{topo}"] = {
            f"{p}@B{B}": sum(1 for t in v if t["prune_events"]) / len(v)
            for (p, B), v in arms.items() if p != "baseline" and v}

    (RES / "stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    h2 = out["h2"]
    print(f"H2 registered (best-B1 vs uniform-B2): {h2['best']['k_positive']}+/"
          f"{h2['best']['k_negative']}- of {h2['best']['n_windowed']}, "
          f"p={h2['best']['sign_p_one_sided']:.2e}, mean delta={h2['best']['mean_delta']:+.3f} "
          f"-> {'PASS' if h2['registered_pass'] else 'FAIL'}")
    print(f"H2 fixed-policy (geometric-B1): {h2['geo']['k_positive']}+/{h2['geo']['k_negative']}- "
          f"p={h2['geo']['sign_p_one_sided']:.2e}")
    print(f"H3: shares={ {k: round(v, 3) for k, v in shares.items()} }")
    print(f"H3 registered: {'PASS' if out['h3']['registered_pass'] else 'FAIL'} "
          f"(ge10pct={out['h3']['ge10pct']}, topo-presence={stability})")
    clean_max = max((v for d in out["clean"].values() for v in d.values()), default=0)
    print(f"clean-arm max false-intervention rate: {clean_max:.3f}")


if __name__ == "__main__":
    main()
