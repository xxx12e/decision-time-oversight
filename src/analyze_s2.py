r"""S2 analysis: policy x budget outcomes under randomized onset; amended-H5
(geometric vs uniform, high-rho well-specified, B<=4; misspecification clause).

Output: ..\results\s2\stats.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "s2"
BUDGETS = (1, 2, 4, 8, 16)
POLICIES = ("uniform", "random", "front", "back", "geometric", "oracle")


def sign_p(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def main():
    rows = [json.loads(x) for x in (RES / "trials.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["condition"], r["policy"], r["budget"])].append(r)

    cells = sorted({r["cell"] for r in rows})
    rho = {r["cell"]: r["rho"] for r in rows}
    out = {"cells": {}, "h5": {}, "front_collapse": {}}

    for cell in cells:
        for cond in ("well_specified", "misspecified"):
            entry = {}
            for pid in POLICIES:
                for B in BUDGETS:
                    ts = by.get((cell, cond, pid, B), [])
                    if not ts:
                        continue
                    entry[f"{pid}@B{B}"] = {
                        "miss": sum(t["miss"] for t in ts) / len(ts),
                        "mean_pen": sum(t["penalized_delay"] for t in ts) / len(ts),
                        "fa": sum(t["false_alarm"] for t in ts) / len(ts)}
            out["cells"][f"{cell}|{cond}"] = entry

    # amended H5: high-rho (>=0.25) well-specified, B in {1,2,4}
    hi = [c for c in cells if rho[c] >= 0.25]
    for B in (1, 2, 4):
        k_pos = k_neg = 0
        details = []
        for cell in hi:
            g = {t["seed"]: t["penalized_delay"] for t in by[(cell, "well_specified", "geometric", B)]}
            u = {t["seed"]: t["penalized_delay"] for t in by[(cell, "well_specified", "uniform", B)]}
            deltas = [u[s] - g[s] for s in sorted(u)]   # positive => geometric better
            mean_d = sum(deltas) / len(deltas)
            details.append({"cell": cell, "mean_delta_pen": mean_d})
            k_pos += mean_d > 0
            k_neg += mean_d < 0
        # misspecification clause: does any geometric advantage reverse/vanish?
        mis = []
        for cell in hi:
            g = {t["seed"]: t["penalized_delay"] for t in by[(cell, "misspecified", "geometric", B)]}
            u = {t["seed"]: t["penalized_delay"] for t in by[(cell, "misspecified", "uniform", B)]}
            mis.append(sum(u[s] - g[s] for s in sorted(u)) / len(u))
        out["h5"][f"B{B}"] = {
            "n_cells": len(hi), "k_geometric_better": k_pos, "k_uniform_better": k_neg,
            "sign_p_geometric": sign_p(k_pos, k_pos + k_neg),
            "sign_p_uniform": sign_p(k_neg, k_pos + k_neg),
            "well_details": details, "mis_mean_deltas": mis}

    # front-collapse: front/geometric miss vs uniform at each budget (all cells, well)
    diagnosable = [c for c in cells if rho[c] > 0]
    for B in BUDGETS:
        fr = [t for c in diagnosable for t in by[(c, "well_specified", "front", B)]]
        ge = [t for c in diagnosable for t in by[(c, "well_specified", "geometric", B)]]
        un = [t for c in diagnosable for t in by[(c, "well_specified", "uniform", B)]]
        orc = [t for c in diagnosable for t in by[(c, "well_specified", "oracle", B)]]
        out["front_collapse"][f"B{B}"] = {
            "front_miss": sum(t["miss"] for t in fr) / len(fr),
            "geometric_miss": sum(t["miss"] for t in ge) / len(ge),
            "uniform_miss": sum(t["miss"] for t in un) / len(un),
            "oracle_miss": sum(t["miss"] for t in orc) / len(orc)}

    (RES / "stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("=== amended H5 (geometric vs uniform, high-rho well-specified) ===")
    for B in (1, 2, 4):
        h = out["h5"][f"B{B}"]
        print(f"  B={B}: geo-better {h['k_geometric_better']}/{h['n_cells']} "
              f"(p_geo={h['sign_p_geometric']:.3f}, p_uni={h['sign_p_uniform']:.3f})")
    print("=== miss rates pooled over diagnosable cells (well-specified) ===")
    for B in BUDGETS:
        f = out["front_collapse"][f"B{B}"]
        print(f"  B={B}: front={f['front_miss']:.2f} geometric={f['geometric_miss']:.2f} "
              f"uniform={f['uniform_miss']:.2f} oracle={f['oracle_miss']:.2f}")


if __name__ == "__main__":
    main()
