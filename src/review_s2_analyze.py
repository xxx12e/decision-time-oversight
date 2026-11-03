r"""Analysis for protocol supplement S2 (reads results/s1/review_s2/*.jsonl).

Usage: python -X utf8 review_s2_analyze.py
Output: ../results/s1/review_s2/analysis.json (and a printed summary)
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from review_s2_runs import ALL_SCHEDS, SCHED_INDEX, ROUNDS, BUDGETS, library

RES = Path(__file__).resolve().parent.parent / "results" / "s1"
D = RES / "review_s2"
NONORACLE = ("uniform", "random", "front", "back", "geometric", "suspicion")
BOOT_N, BOOT_SEED = 20_000, 20260808


def sign_p(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def boot(ds, seed=BOOT_SEED):
    rng = random.Random(seed)
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(BOOT_N))
    return [sum(ds) / len(ds), bs[int(0.025 * BOOT_N)], bs[int(0.975 * BOOT_N)]]


def contrast(cells, a, b):
    """per-cell a-b; sign test over cells excluding ties; bootstrap CI over cells."""
    ds = [a[c] - b[c] for c in cells]
    pos, neg = sum(d > 1e-12 for d in ds), sum(d < -1e-12 for d in ds)
    return {"k_pos": pos, "k_neg": neg, "ties": len(ds) - pos - neg, "n": len(ds),
            "p_one_sided": sign_p(pos, pos + neg), "mean_ci": boot(ds),
            "per_cell": {c: round(d, 4) for c, d in zip(cells, ds)}}


def load(arm):
    f = D / f"{arm}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[r["cell"]][r["trial"]] = r
    return by


def outcome(row, key_or_sched):
    """(fc, tc, iv) for a library key 'name@B' or a frozenset schedule."""
    if isinstance(key_or_sched, str):
        if key_or_sched in row["library"]:
            return tuple(row["library"][key_or_sched][:3])
        name, B = key_or_sched.split("@B")
        cap = row["capture_round"]
        s = library(int(B), row["trial"], cap)[name]
    else:
        s = key_or_sched
    if "fc" in row:
        i = SCHED_INDEX[s]
        return int(row["fc"][i]), int(row["tc"][i]), int(row["iv"][i])
    if len(s) == 1:
        return tuple(row["singles"][str(min(s))])
    for k, v in row["library"].items():
        if v[3] is not None and frozenset(v[3]) == s and not k.startswith("suspicion"):
            return tuple(v[:3])
    raise KeyError(s)


def cell_stats(by, keys):
    out = {}
    for cell, trials in by.items():
        rows = [trials[t] for t in sorted(trials)]
        n = len(rows)
        fcb = sum(r["base_fc"] for r in rows) / n
        pol = {}
        for k in keys:
            o = [outcome(r, k) for r in rows]
            pol[k] = {"fc_prevention": fcb - sum(x[0] for x in o) / n, "timely": sum(x[1] for x in o) / n}
        out[cell] = {"fc_base": fcb, "pol": pol, "n": n,
                     "captures": sum(r["base_fc"] for r in rows)}
    return out


def classify(fc_base, pol):
    oracle_any = max(pol[f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS)
    uni1 = pol["uniform@B1"]["fc_prevention"]
    best1 = max(pol[f"{p}@B1"]["fc_prevention"] for p in NONORACLE)
    uni23 = max(pol[f"uniform@B{B}"]["fc_prevention"] for B in (2, 3))
    if fc_base < 0.10:
        return "benign"
    if oracle_any <= 0.05:
        return "ceiling_bound"
    if best1 >= 2 * max(uni1, 0.02) and best1 >= 0.10:
        return "placement_critical"
    if uni23 >= oracle_any - 0.05:
        return "budget_saturated"
    return "intermediate"


def candidates(row):
    if "fc" in row:
        return list(ALL_SCHEDS)
    return [frozenset({r}) for r in ROUNDS]


def clairvoyant(by):
    """per cell: share of runs in which some schedule (B<=3, rounds 2..15) prevents capture."""
    out = {}
    for cell, trials in by.items():
        rows = list(trials.values())
        k = 0
        for r in rows:
            if r["base_fc"]:
                if "fc" in r:
                    k += "0" in r["fc"]
                else:
                    k += any(v[0] == 0 for v in r["singles"].values())
        out[cell] = k / len(rows)
    return out


def best_fixed_heldout(by, B):
    """choose the best fixed schedule of size B on even seeds, score on odd, and vice versa."""
    out = {}
    for cell, trials in by.items():
        rows = [trials[t] for t in sorted(trials)]
        scheds = [s for s in candidates(rows[0]) if len(s) == B]
        halves = ([r for r in rows if r["trial"] % 2 == 0], [r for r in rows if r["trial"] % 2 == 1])

        def prev(rs, s):
            return (sum(r["base_fc"] for r in rs) - sum(outcome(r, s)[0] for r in rs)) / len(rs)
        vals = []
        for a, b in ((0, 1), (1, 0)):
            s_best = max(scheds, key=lambda s: (prev(halves[a], s), -min(s)))
            vals.append(prev(halves[b], s_best))
        out[cell] = sum(vals) / 2
    return out


def main():
    strict = json.loads((RES / "strict_regime_map.json").read_text(encoding="utf-8"))
    windowed = sorted(c for c, v in strict["cells"].items() if v.get("windowed"))
    assert len(windowed) == 12, windowed
    lib_keys = [f"{p}@B{B}" for p in NONORACLE + ("oracle", "uniform2") for B in BUDGETS]
    A = {"windowed_cells": windowed}
    arms = {}
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD", "SINGLE_L0_STD", "SINGLE_L1_STD", "BENCH_L0_DELAY",
                "BENCH_L1_DELAY", "PERFECT_L0_STD", "PERFECT_L1_STD", "BENCH_L0_EVADE", "BENCH_L1_EVADE"):
        by = load(arm)
        if by is None or sum(len(v) for v in by.values()) < 3000:
            print("missing/incomplete", arm)
            continue
        if arm.endswith("EVADE"):       # EVADE cannot anticipate an online trigger: STD's suspicion rows
            std = load(arm.replace("EVADE", "STD"))
            for cell in by:
                for t in by[cell]:
                    for B in BUDGETS:
                        by[cell][t]["library"][f"suspicion@B{B}"] = std[cell][t]["library"][f"suspicion@B{B}"]
        arms[arm] = by
    S = {arm: cell_stats(by, lib_keys) for arm, by in arms.items()}
    benign = sorted(c for c, v in S["BENCH_L0_STD"].items() if v["fc_base"] < 0.10)
    nonbenign = sorted(c for c in S["BENCH_L0_STD"] if c not in benign)
    A["n_nonbenign"] = len(nonbenign)

    # ---- validation gate against the stored sweeps
    val = {"L0_pairs_checked": 0, "L0_mismatch": 0, "L1_cells_checked": 0, "L1_mismatch": []}
    want = {}
    with open(RES / "trials.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["policy"] in ("baseline",) or r["cell_committed"] == 0:
                continue
            cell = f"{r['cell_committed']}|{r['cell_evidence']}|{r['cell_topology']}"
            want[(cell, r["trial"], f"{r['policy']}@B{r['budget']}")] = (
                int(r["false_consensus"]), int(r["timely_committed_prune"]))
    for (cell, t, k), v in want.items():
        got = outcome(arms["BENCH_L0_STD"][cell][t], k)[:2]
        val["L0_pairs_checked"] += 1
        val["L0_mismatch"] += got != v
    for cell, v in strict["cells"].items():
        pol_old = v["by_convention"]["strict"]["policies"]
        for k, pv in pol_old.items():
            if k.split("@")[0] in NONORACLE + ("oracle",):
                new = S["BENCH_L1_STD"][cell]["pol"][k]
                val["L1_cells_checked"] += 1
                if abs(new["fc_prevention"] - pv["fc_prevention"]) > 1e-9 or abs(new["timely"] - pv["timely"]) > 1e-9:
                    val["L1_mismatch"].append([cell, k])
    A["validation"] = val
    print("validation:", {k: (v if not isinstance(v, list) else len(v)) for k, v in val.items()})

    # ---- regime maps
    A["regimes"] = {}
    for arm, st in S.items():
        cnt = defaultdict(int)
        lab = {}
        for c, v in st.items():
            lab[c] = classify(S["BENCH_L0_STD"][c]["fc_base"] if arm.endswith("STD") else v["fc_base"], v["pol"])
            cnt[lab[c]] += 1
        A["regimes"][arm] = {"counts": dict(cnt), "labels": lab}

    # ---- P1: uniform-from-2 (B=2) vs audit-early (B=1), BENCH L0 STD
    st = S["BENCH_L0_STD"]
    pv = lambda arm, k: {c: S[arm][c]["pol"][k]["fc_prevention"] for c in S[arm]}  # noqa: E731
    tm = lambda arm, k: {c: S[arm][c]["pol"][k]["timely"] for c in S[arm]}  # noqa: E731
    A["P1"] = {}
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD", "PERFECT_L0_STD", "SINGLE_L0_STD"):
        if arm in S:
            c = contrast(windowed, pv(arm, "uniform2@B2"), pv(arm, "geometric@B1"))
            c["within_0.05"] = sum(abs(d) <= 0.05 + 1e-12 for d in c["per_cell"].values())
            A["P1"][arm] = c
    A["P1"]["uniform2B2_vs_uniformB2_L0"] = contrast(windowed, pv("BENCH_L0_STD", "uniform2@B2"), pv("BENCH_L0_STD", "uniform@B2"))
    A["P1"]["geoB1_vs_uniformB2_L0"] = contrast(windowed, pv("BENCH_L0_STD", "geometric@B1"), pv("BENCH_L0_STD", "uniform@B2"))
    A["P1"]["uniform2B3_vs_geoB1_L0"] = contrast(windowed, pv("BENCH_L0_STD", "uniform2@B3"), pv("BENCH_L0_STD", "geometric@B1"))

    # ---- P2: best fixed schedule (held-out) by budget; clairvoyant bound
    A["P2"] = {}
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD", "SINGLE_L0_STD", "SINGLE_L1_STD", "BENCH_L0_DELAY", "BENCH_L1_DELAY"):
        if arm not in arms:
            continue
        bf = {B: best_fixed_heldout(arms[arm], B) for B in BUDGETS}
        cv = clairvoyant(arms[arm])
        A["P2"][arm] = {
            "heldout_mean_windowed": {B: sum(bf[B][c] for c in windowed) / 12 for B in BUDGETS},
            "B3_minus_B1": contrast(windowed, bf[3], bf[1]),
            "B2_minus_B1": contrast(windowed, bf[2], bf[1]),
            "clairvoyant_mean_windowed": sum(cv[c] for c in windowed) / 12,
            "clairvoyant_minus_bestfixedB3_windowed": contrast(windowed, cv, bf[3])["mean_ci"]}

    # ---- P3: latency decomposition of the clairvoyant bound over non-benign cells
    A["P3"] = {}
    for det in ("BENCH", "SINGLE", "PERFECT"):
        a0, a1 = f"{det}_L0_STD", f"{det}_L1_STD"
        if a0 in arms and a1 in arms:
            c0, c1 = clairvoyant(arms[a0]), clairvoyant(arms[a1])
            r0 = {c: max(S[a0][c]["pol"][f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS) for c in nonbenign}
            r1 = {c: max(S[a1][c]["pol"][f"oracle@B{B}"]["fc_prevention"] for B in BUDGETS) for c in nonbenign}
            A["P3"][det] = {"median_bound_L0": median(c0[c] for c in nonbenign),
                            "median_bound_L1": median(c1[c] for c in nonbenign),
                            "D_median_of_bound": median(c0[c] for c in nonbenign) - median(c1[c] for c in nonbenign),
                            "median_reference_L0": median(r0.values()), "median_reference_L1": median(r1.values()),
                            "windowed_bound_L0": sum(c0[c] for c in windowed) / 12,
                            "windowed_bound_L1": sum(c1[c] for c in windowed) / 12}
    # perfect detector: multi-check outcome == earliest single check (pruning is complete at once)?
    if "PERFECT_L0_STD" in arms:
        chk = bad = 0
        for arm in ("PERFECT_L0_STD", "PERFECT_L1_STD"):
            for trials in arms[arm].values():
                for r in trials.values():
                    for k, v in r["library"].items():
                        if k.startswith("suspicion") or not v[3] or len(v[3]) < 2:
                            continue
                        chk += 1
                        bad += v[0] != r["singles"][str(min(v[3]))][0] if str(min(v[3])) in r["singles"] else 0
        A["P3"]["perfect_earliest_check_property"] = {"checked": chk, "violations": bad}

    # ---- P4: effective deadline (PERFECT L0)
    if "PERFECT_L0_STD" in arms:
        gaps, never, n_capt, r1_only = [], 0, 0, 0
        per_cell = defaultdict(list)
        for cell, trials in arms["PERFECT_L0_STD"].items():
            for r in trials.values():
                if not r["base_fc"]:
                    continue
                n_capt += 1
                prev_rounds = [int(k) for k, v in r["singles"].items() if v[0] == 0]
                if r["library"].get("round1@B1", [1])[0] == 0:
                    prev_rounds.append(1)
                if not prev_rounds:
                    never += 1
                    continue
                if prev_rounds == [1]:
                    r1_only += 1
                t_eff = max(prev_rounds)
                cap = r["capture_round"] or 16
                gaps.append(cap - t_eff)
                per_cell[cell].append(cap - t_eff)
        A["P4"] = {"captured_pairs": n_capt, "with_preventing_round": len(gaps), "never_preventable": never,
                   "preventable_only_at_round1": r1_only,
                   "median_gap": median(gaps), "share_gap_ge1": sum(g >= 1 for g in gaps) / len(gaps),
                   "gap_hist": {str(g): gaps.count(g) for g in sorted(set(gaps))}}

    # ---- window model (Proposition): W = rounds at which a single check prevents capture.
    # (i) is W an interval; (ii) does a multi-check schedule prevent iff it hits W (union rule)?
    A["window_model"] = {}
    for arm in ("BENCH_L0_STD", "BENCH_L1_STD", "SINGLE_L0_STD", "SINGLE_L1_STD", "BENCH_L0_DELAY", "BENCH_L1_DELAY"):
        if arm not in arms:
            continue
        n_w = n_int = agree = tot = 0
        cover2 = defaultdict(list)
        for cell, trials in arms[arm].items():
            for r in trials.values():
                if not r["base_fc"]:
                    continue
                W = {min(s) for s in ALL_SCHEDS if len(s) == 1 and r["fc"][SCHED_INDEX[s]] == "0"}
                if W:
                    n_w += 1
                    n_int += (max(W) - min(W) + 1) == len(W)
                for s in ALL_SCHEDS:
                    if len(s) > 1:
                        tot += 1
                        agree += (r["fc"][SCHED_INDEX[s]] == "0") == bool(W & s)
        A["window_model"][arm] = {"captured_with_window": n_w, "window_is_interval": n_int / n_w if n_w else None,
                                  "union_rule_agreement": agree / tot if tot else None}

    # ---- relative prevention in ceiling-bound cells (BENCH L0 STD) and the round-1 question
    cv = clairvoyant(arms["BENCH_L0_STD"])
    ceil = [c for c, l in A["regimes"]["BENCH_L0_STD"]["labels"].items() if l == "ceiling_bound"]
    rows = []
    for c in sorted(ceil):
        capt = S["BENCH_L0_STD"][c]["captures"]
        prevented = round(cv[c] * 50)
        r = {"cell": c, "captures": capt, "bound_prevented": prevented,
             "relative": prevented / capt if capt else None,
             "median_capture": strict["cells"][c]["median_capture"]}
        if "PERFECT_L0_STD" in arms:
            pr = arms["PERFECT_L0_STD"][c]
            r["perfect_round1_prevented"] = sum(1 for x in pr.values() if x["base_fc"] and x["library"]["round1@B1"][0] == 0)
            r["perfect_bound_prevented"] = round(clairvoyant(arms["PERFECT_L0_STD"])[c] * 50)
        rows.append(r)
    A["ceiling_relative"] = rows

    # ---- P5: EVADE
    if "BENCH_L0_EVADE" in S:
        A["P5"] = {}
        for lag in ("L0", "L1"):
            ev, sd = f"BENCH_{lag}_EVADE", f"BENCH_{lag}_STD"
            A["P5"][lag] = {
                "geoB1_evade_minus_std": contrast(windowed, pv(ev, "geometric@B1"), pv(sd, "geometric@B1")),
                "randomB1_minus_geoB1_evade": contrast(windowed, pv(ev, "random@B1"), pv(ev, "geometric@B1")),
                "geoB1_timely_evade_minus_std": contrast(windowed, tm(ev, "geometric@B1"), tm(sd, "geometric@B1")),
                "uniformB2_evade_minus_std": contrast(windowed, pv(ev, "uniform@B2"), pv(sd, "uniform@B2")),
                "randomB3_minus_geoB1_evade": contrast(windowed, pv(ev, "random@B3"), pv(ev, "geometric@B1"))}
            # capture under evasion of a round-2 check alone (no pruning): does camouflage cost the adversary?
            by = arms[ev]
            stalled = 0
            tot = 0
            for c in windowed:
                for r in by[c].values():
                    fc, tc, iv = outcome(r, "geometric@B1")
                    if r["base_fc"] and not iv:
                        tot += 1
                        stalled += fc == 0
            A["P5"][lag]["prevented_without_any_prune_share"] = stalled / tot if tot else None

    # ---- P6: DELAY
    if "BENCH_L0_DELAY" in S:
        A["P6"] = {}
        for lag in ("L0", "L1"):
            dl = f"BENCH_{lag}_DELAY"
            A["P6"][lag] = {"uniformB2_minus_geoB1": contrast(windowed, pv(dl, "uniform@B2"), pv(dl, "geometric@B1")),
                            "uniform2B2_minus_geoB1": contrast(windowed, pv(dl, "uniform2@B2"), pv(dl, "geometric@B1")),
                            "base_capture_windowed": sum(S[dl][c]["fc_base"] for c in windowed) / 12}

    # ---- suspicion-triggered and library summary in windowed cells, every arm
    A["library_windowed"] = {arm: {k: round(sum(S[arm][c]["pol"][k]["fc_prevention"] for c in windowed) / 12, 4)
                                   for k in lib_keys} for arm in S}
    # ---- fault-free false intervention per detector
    fi = defaultdict(lambda: defaultdict(list))
    for line in (D / "clean.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = f"{r['det']}_L{r['lag']}"
        for k, v in r["library"].items():
            fi[key][k].append(v)
        fi[key]["single_any_round"].append(sum(r["singles"].values()) / len(r["singles"]))
    A["false_intervention"] = {k: {kk: sum(vv) / len(vv) for kk, vv in v.items()} for k, v in fi.items()}

    (D / "analysis.json").write_text(json.dumps(A, indent=1), encoding="utf-8")
    brief = {k: A[k] for k in ("regimes",) if k in A}
    for arm, v in brief["regimes"].items():
        print(arm, v["counts"])
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        if k in A:
            print("==", k)
            print(json.dumps(A[k], indent=0, default=str)[:3000])


if __name__ == "__main__":
    main()
