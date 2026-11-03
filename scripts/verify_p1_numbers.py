r"""verify_p1_numbers.py — every number in paper/main.tex must be derivable from
results/*.json (house rule). XX...XX placeholders are counted and reported;
the script FAILS if any pinned number mismatches, and lists remaining
placeholders so the backfill state is always visible.

Usage: python -X utf8 verify_p1_numbers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent.parent
import os
TEX_PATH = Path(os.environ.get("P1_TEX", P / "paper" / "main.tex")).resolve()
TEX = TEX_PATH.read_text(encoding="utf-8")
IS_JAAMAS = TEX_PATH.parent.name == "jaamas"
s1 = json.loads((P / "results" / "s1" / "stats.json").read_text(encoding="utf-8"))
s2 = json.loads((P / "results" / "s2" / "stats.json").read_text(encoding="utf-8"))
zoo = json.loads((P / "results" / "s4" / "zoo.json").read_text(encoding="utf-8"))

FAILS = []
PASSES = []


def pin(desc, claim, actual, tol=0.0):
    ok = (abs(claim - actual) <= tol) if isinstance(claim, float) else (claim == actual)
    shown = round(actual, 6) if isinstance(actual, float) else actual
    (PASSES if ok else FAILS).append(f"{desc}: tex={claim} data={shown}")


# venue builds may reword a pinned phrase; the reworded literal is then pinned instead
VENUE_RETARGET = {}  # empty since R10: the JAAMAS abstract is the source's own

def intext(pattern, desc):
    pattern = VENUE_RETARGET.get(pattern, pattern)
    if not re.search(pattern, TEX, re.DOTALL):
        FAILS.append(f"MISSING IN TEX: {desc} (pattern {pattern!r})")
    else:
        PASSES.append(f"present: {desc}")


# ---- S1 ----
h2 = s1["h2"]["best"]
pin("H2 k_positive", 11, h2["k_positive"])
pin("H2 n_windowed", 12, h2["n_windowed"])
pin("H2 p", 4.9e-4, h2["sign_p_one_sided"], tol=5e-5)
pin("H2 mean delta", 0.41, h2["mean_delta"], tol=0.005)
deltas = [w["delta_best"] for w in s1["h2"]["cells"]]
pin("H2 max delta", 0.92, max(deltas), tol=0.005)
pin("H2 min positive delta", 0.02, min(d for d in deltas if d > 0), tol=0.005)
# ---- H2 on the causal endpoint (capture prevention) ----
import math as _math
import random as _random
_dtc, _dpv = [], []
for _w in s1["h2"]["cells"]:
    _pols = s1["cells"][_w["cell"]]["policies"]
    _b1tc = max(v["timely"] for k, v in _pols.items()
                if k.endswith("@B1") and not k.startswith("oracle"))
    _b1pv = max(v["fc_prevention"] for k, v in _pols.items()
                if k.endswith("@B1") and not k.startswith("oracle"))
    _dtc.append(_b1tc - _pols["uniform@B2"]["timely"])
    _dpv.append(_b1pv - _pols["uniform@B2"]["fc_prevention"])
_pos = sum(d > 0 for d in _dpv)
_n = sum(d != 0 for d in _dpv)
_p1s = sum(_math.comb(_n, k) for k in range(_pos, _n + 1)) / 2 ** _n
pin("H2-prevention wins 12/12", 12, _pos)
pin("H2-prevention one-sided p 2.4e-4", 2.4e-4, _p1s, tol=1e-5)
pin("H2-prevention mean delta +0.39", 0.39, sum(_dpv) / len(_dpv), tol=0.005)
_rng = _random.Random(20260808)
for _nm, _ds, _lo, _hi in (("timely", _dtc, 0.25, 0.58), ("prevention", _dpv, 0.26, 0.54)):
    _bs = sorted(sum(_ds[_rng.randrange(len(_ds))] for _ in _ds) / len(_ds)
                 for _ in range(20000))
    # tol 0.0051: the timely CI's upper bound lands exactly on a half-ULP (0.575)
    pin(f"H2 {_nm} paired CI lo {_lo}", _lo, _bs[500], tol=0.0051)
    pin(f"H2 {_nm} paired CI hi {_hi}", _hi, _bs[19500], tol=0.0051)

sh = s1["h3"]["shares"]
pin("regime share placement", 0.47, sh["placement_critical"], tol=0.005)
pin("regime share ceiling", 0.17, sh["ceiling_bound"], tol=0.005)
pin("regime share saturated", 0.05, sh["budget_saturated"], tol=0.005)
cnt = s1["h3"]["counts"]
pin("placement cells 28/60", 28, cnt["placement_critical"])
pin("ceiling cells 10/60", 10, cnt["ceiling_bound"])
pin("saturated cells 3/60", 3, cnt["budget_saturated"])
clean_max = max(v for d in s1["clean"].values() for v in d.values())
pin("clean-arm max false intervention 0.08", 0.08, clean_max, tol=0.001)
intext(r"72\{,\}000", "72,000 trials")

# ---- S2 ----
fc = s2["front_collapse"]
for B in (1, 2, 4):
    pin(f"S2 front miss B{B} = 1.0", 1.0, fc[f"B{B}"]["front_miss"])
    pin(f"S2 geometric miss B{B} = 1.0", 1.0, fc[f"B{B}"]["geometric_miss"])
    pin(f"S2 oracle miss B{B} = 0.0", 0.0, fc[f"B{B}"]["oracle_miss"])
pin("S2 uniform miss B1", 0.48, fc["B1"]["uniform_miss"], tol=0.005)
pin("S2 uniform miss B2", 0.18, fc["B2"]["uniform_miss"], tol=0.005)
pin("S2 uniform miss B4", 0.00, fc["B4"]["uniform_miss"], tol=0.005)
h5_fail_all = all(s2["h5"][f"B{B}"]["k_geometric_better"] == 0 for B in (1, 2, 4))
pin("S2 amended-H5 geometric better 0 cells at all budgets", True, h5_fail_all)
intext(r"14\{,\}400", "14,400 trials")

# ---- S4 (final H4: 90 archived + 40 fresh S3 trials) ----
pin("S4 includes both S3 families", "gemma4,qwen35",
    ",".join(sorted(zoo.get("included_s3") or [])))
tr = zoo["kendall"]["timely_recall_mfpr"]
pin("S4 tau (recall, registered pair)", 0.51, tr["tau"], tol=0.005)
pin("S4 tau recall CI lo", 0.36, tr["ci95"][0], tol=0.005)
pin("S4 tau recall CI hi", 0.56, tr["ci95"][1], tol=0.005)
pin("H4 registered criterion: tau<=0.6 and CI excludes 0.8", True,
    tr["tau"] <= 0.6 and tr["ci95"][1] < 0.8)
ta = zoo["kendall"]["timely_auc"]
pin("S4 tau (timely AUC)", 0.67, ta["tau"], tol=0.005)
pin("S4 tau AUC CI lo", 0.62, ta["ci95"][0], tol=0.005)
pin("S4 tau AUC CI hi", 0.72, ta["ci95"][1], tol=0.005)
max_rec = max((d["timely_recall_mfpr"] or 0) for d in zoo["monitors"].values())
pin("S4 best timely recall 0.24", 0.24, max_rec, tol=0.005)
pin("S4 zoo size 13", 13, len(zoo["monitors"]))
pin("S4 trials 130", 130, zoo["n_trials"])
top_post = zoo["rank_posthoc"][:2]
pin("S4 hindsight top is constancy-family", True,
    set(top_post) <= {"conf_constancy", "s2_only", "conf_mean"})
fpr = zoo["deployed_fpr_sensitivity"]
pin("S4 FPR sweep low end 0.01", 0.01, fpr["0.02"], tol=0.005)
pin("S4 FPR sweep high end 0.07", 0.07, fpr["0.125"], tol=0.005)
lat = zoo["latency_ablation"]
pin("S4 latency pairs k=0", 862, lat["lead_0"]["n_pairs"])
pin("S4 latency pairs k=1", 151, lat["lead_1"]["n_pairs"])
pin("S4 latency pairs k=2", 36, lat["lead_2"]["n_pairs"])
pin("S4 lead-0 deployed AUC 0.43 (below chance)", 0.43,
    lat["lead_0"]["deployed_auc"], tol=0.005)
pin("S4 deployed never reaches 0.7 under lead", True,
    max(lat[k]["deployed_auc"] for k in lat) < 0.7)
# the S4 table must match zoo.json row for row (rounded to 2 dp)
_TBL_NAMES = {
    "deployed_442": "benchmark detector ($0.4\\,s_1 + 0.4\\,s_2$)",
    "s1_only": "provenance term $s_1$", "s2_only": "behavioral term $s_2$",
    "s1s2_equal": "$0.5\\,s_1 + 0.5\\,s_2$", "w622": "$0.6\\,s_1 + 0.2\\,s_2$",
    "w262": "$0.2\\,s_1 + 0.6\\,s_2$", "conf_mean": "mean confidence",
    "conf_constancy": "confidence constancy", "claim_constancy": "claim constancy",
    "high_conf_share": "high-confidence share", "round1_conf": "round-1 confidence",
    "evidence_rate": "evidence rate", "s1_raw_reuse": "raw signature reuse",
}
# recall column is the tie-aware matched-FPR recall (zoo_tie_aware.py): ties at the 6.25%
# threshold are flagged fractionally so the realized FPR on completed transcripts is exactly 6.25%
zta = json.loads((P / "results" / "s4" / "zoo_tie_aware.json").read_text(encoding="utf-8"))
pin("tie-aware zoo covers the same thirteen monitors", sorted(zoo["monitors"]), sorted(zta["monitors"]))
for _k, _d in zoo["monitors"].items():
    _rec = zta["monitors"][_k]["recall_tie_aware"]
    _row = (f"{_TBL_NAMES[_k]} & {_d['posthoc_auc']:.2f} & "
            f"{_d['timely_auc']:.2f} & {_rec:.2f}")
    intext(re.escape(_row), f"zoo table row: {_k}")

# ---- timing conventions (strict = 1 round latency; intercept = 0 latency) ----
tc_f = P / "results" / "s3" / "timing_conventions.json"
if tc_f.exists():
    tc = json.loads(tc_f.read_text(encoding="utf-8"))
    pin("qwen35 plain intercept recall 0.30", 0.30,
        tc["s3"]["qwen35"]["plain"]["intercept"]["timely_recall"], tol=0.005)
    pin("gemma4 plain intercept recall 0.85", 0.85,
        tc["s3"]["gemma4"]["plain"]["intercept"]["timely_recall"], tol=0.005)
    pin("jitter recall 0 under both conventions/families", True,
        all(tc["s3"][f]["jitter"][c]["timely_recall"] == 0.0
            for f in ("qwen35", "gemma4") for c in ("strict", "intercept")))
    pin("S4 intercept best recall 0.63", 0.63,
        tc["s4"]["intercept"]["best_recall"], tol=0.005)
    pin("S4 intercept best monitor is conf_constancy", "conf_constancy",
        tc["s4"]["intercept"]["best_monitor"])
    pin("S4 intercept tau 0.41", 0.41, tc["s4"]["tau_recall_intercept"], tol=0.005)
    pin("intercept deepens the inversion (tau falls)", True,
        tc["s4"]["tau_recall_intercept"] < zoo["kendall"]["timely_recall_mfpr"]["tau"])

# scoreable decomposition + conf_constancy hindsight AUC quoted as 0.95
pin("scoreable pairs 862 of 948", True,
    zoo["monitors"]["conf_constancy"]["n_timely"] == 862
    and zoo["monitors"]["conf_constancy"]["n_post"] == 948)
pin("hindsight best AUC 0.95 (conf_constancy)", 0.95,
    zoo["monitors"]["conf_constancy"]["posthoc_auc"], tol=0.005)
# the level-collapse sentence must quote the LEADER's numbers, not a splice of
# rows: 0.95 -> 0.56 is conf_constancy; the best tie-aware recall, 0.36, is
# high_conf_share at hindsight 0.90 (a 0.94/0.95 mix shipped in rounds 1..6 and was caught in 7)
pin("hindsight leader is conf_constancy", "conf_constancy", zoo["rank_posthoc"][0])
pin("leader deadline AUC 0.56", 0.56,
    zoo["monitors"]["conf_constancy"]["timely_auc"], tol=0.005)
pin("best tie-aware recall is high_conf_share", "high_conf_share", zta["best_tie_aware"]["monitor"])
pin("best tie-aware recall 0.36", 0.36, zta["best_tie_aware"]["recall_tie_aware"], tol=0.005)
pin("best tie-aware recall CI [0.29, 0.47]", (0.29, 0.47),
    tuple(round(x, 2) for x in zta["best_tie_aware"]["recall_tie_aware_ci95"]))
pin("high_conf_share AUC 0.90 -> 0.88", (0.90, 0.88),
    (round(zoo["monitors"]["high_conf_share"]["posthoc_auc"], 2),
     round(zoo["monitors"]["high_conf_share"]["timely_auc"], 2)))
pin("best recall with thresholds set on deadline data 0.27", 0.27,
    max(v["recall_tie_aware_deadline_threshold"] for v in zta["monitors"].values()), tol=0.005)
pin("FPR sweep best recall 0.12 / 0.29 / 0.67 at 2% / 5% / 10%", (0.12, 0.29, 0.67),
    tuple(round(zta["fpr_sweep_best"][k]["best"], 2) for k in ("0.02", "0.05", "0.1")))
intext(re.escape("its ranking accuracy falls from 0.95 to 0.56"),
       "abstract level-collapse quotes the leader's own numbers")
intext(re.escape("the best monitor on transcripts falls from 0.95 to 0.56 AUC at the deadline, "
                 "and the best recall at the deadline at a 6.25\\% false-positive rate is 0.36"),
       "intro level-collapse and best timely recall")
intext(re.escape("confidence constancy, falls from 0.95 AUC to 0.56 at the deadline"),
       "study-4 level-collapse attributes the AUC to the right monitor")
intext(re.escape("high-confidence share falls only from 0.90 to 0.88 and reaches the highest "
                 "timely recall, 0.36 (CI $[0.29, 0.47]$)"), "study-4 best recall in prose")
intext(re.escape("Setting each threshold on deadline data instead lowers the best recall to 0.27"),
       "deadline-threshold recall in prose")
intext(re.escape("the best timely recall is 0.12 at a 2\\% rate, 0.29 at 5\\%, and 0.67 at 10\\%"),
       "FPR sweep in prose")
pin("no stale 0.94-AUC leader claim anywhere", 0, TEX.count("falls from $0.94$ AUC"))
# the round-7 detector spec must match the replay code: s2 carries constancy,
# s3 is divergence, and the replayed deployed score drops s3 because the
# archived logs record citation presence without sign
intext(re.escape("undefined before two messages exist"), "s2 constancy property in spec")
intext(re.escape("a divergence term $s_3$"), "s3 named divergence, not constancy")
intext(re.escape("the replayed benchmark detector is $0.4\\,s_1 + 0.4\\,s_2$"),
       "replay drops s3, matching the zoo's 0.2*0.0 term")
intext(re.escape("orders agents exactly as the equal-weight variant"),
       "deployed/equal-weights identity explained")
intext(re.escape("three reweightings of them, a raw signature-reuse count"),
       "zoo roster arithmetic sums to 13")
intext(re.escape("Figure~\\ref{fig:dissoc}"), "study-3 figure is referenced")

lofo_f = P / "results" / "s4" / "lofo_tau.json"
if lofo_f.exists():
    lofo = json.loads(lofo_f.read_text(encoding="utf-8"))
    _rr = lofo["range"]["tau_recall"]
    pin("LOFO recall tau range lo 0.44", 0.44, _rr[0], tol=0.005)
    pin("LOFO recall tau range hi 0.54", 0.54, _rr[1], tol=0.005)
    pin("LOFO recall tau < 0.6 in every drop", True,
        all(r["tau_recall"] < 0.6 for r in lofo["leave_one_out"].values()))
    pin("LOFO covers six populations", 6, len(lofo["leave_one_out"]))

bf = zoo["best_recall_by_fpr"]
pin("S4 zoo sweep: best recall <=0.24 up to 5% FPR", True,
    max(bf["0.01"], bf["0.02"], bf["0.05"]) <= 0.24)
pin("S4 zoo sweep: best recall 0.67 at 10% FPR", 0.67, bf["0.1"], tol=0.005)

# ---- exhaustive placement oracle (true bound) ----
to_f = P / "results" / "s1" / "true_oracle.json"
if to_f.exists():
    to = json.loads(to_f.read_text(encoding="utf-8"))
    tt = to["totals"]
    pin("oracle pairs 3000", 3000, to["n_pairs"])
    pin("oracle baseline captures 1412", 1412, tt["baseline_captured"])
    pin("exhaustive timely catches 2298 (B-invariant)", True,
        tt["B1"]["exh_tc"] == tt["B2"]["exh_tc"] == tt["B3"]["exh_tc"] == 2298)
    pin("exhaustive preventions 694 (B-invariant)", True,
        tt["B1"]["exh_prev"] == tt["B2"]["exh_prev"] == tt["B3"]["exh_prev"] == 694)
    pin("heuristic TC share at B1 98.5%", 0.985,
        tt["B1"]["onset_share_of_exhaustive_tc"], tol=0.005)
    pin("heuristic prevention share B1 50%", 0.50,
        tt["B1"]["onset_share_of_exhaustive_prev"], tol=0.005)
    pin("heuristic prevention share B2 72%", 0.72,
        tt["B2"]["onset_share_of_exhaustive_prev"], tol=0.005)
    pin("heuristic prevention share B3 86%", 0.86,
        tt["B3"]["onset_share_of_exhaustive_prev"], tol=0.005)
    _cp = _cc2 = 0
    _shift = []
    for _cell, _c in to["cells"].items():
        _reg = s1["cells"][_cell]["regime"]
        if _reg == "ceiling_bound":
            _cp += _c["B3"]["exhaustive_prevented"] * _c["n"]
            _cc2 += _c["baseline_captured"]
            if _c["B3"]["exhaustive_prevented"] > 0.05:
                _shift.append(_cell)
    pin("ceiling cells exhaustive preventable 13", 13, round(_cp))
    pin("ceiling cells captures 326", 326, _cc2)
    pin("ceiling exhaustive share 4.0%", 0.04, _cp / _cc2, tol=0.002)
    pin("regime labels hold 57/60 (3 border cells)", 3, len(_shift))
    pin("ledger vs exhaustive: forfeits 13 of 694 = 1.9%", 0.019, _cp / 694, tol=0.001)

# ---- artifact-section replay claims ----
# "timely catch" credits any committed prune in a run that never reaches
# irreversibility, so benign cells inflate the catch denominator. The paper
# discloses that composition rather than quoting the inflated share, and these
# pins hold it to the disclosure.
tot_c = tot_p = ceil_c = ceil_p = benign_c = benign_p = 0.0
at_stake_cells = 0
for key, cell in s1["cells"].items():
    oc = max(cell["policies"].get(f"oracle@B{B}", {}).get("timely", 0.0) for B in (1, 2, 3))
    c, p = oc * 50, cell["oracle_any"] * 50
    tot_c += c
    tot_p += p
    if cell["regime"] == "ceiling_bound":
        ceil_c += c
        ceil_p += p
    if cell["regime"] == "benign":
        benign_c += c
        benign_p += p
    else:
        at_stake_cells += 1
at_stake_c = tot_c - benign_c
pin("artifact total catches 2271", 2271, round(tot_c))
pin("artifact catches given up 194", 194, round(ceil_c))
pin("artifact catches given up 8.5%", 0.085, ceil_c / tot_c, tol=0.002)
pin("artifact forfeited preventable 6", 6, round(ceil_p))
pin("artifact total preventable 595", 595, round(tot_p))
pin("artifact forfeited share 1.0%", 0.010, ceil_p / tot_p, tol=0.001)
pin("artifact benign catches 750", 750, round(benign_c))
pin("artifact benign preventable 3", 3, round(benign_p))
pin("artifact at-stake cells 41", 41, at_stake_cells)
pin("artifact at-stake catches 1521", 1521, round(at_stake_c))
pin("artifact at-stake retained 1327", 1327, round(at_stake_c - ceil_c))
pin("artifact at-stake retained share 87%", 0.87, (at_stake_c - ceil_c) / at_stake_c, tol=0.005)

# ---- scoreboard sentence: the zoo ranks it names ----
def _rank(field):
    return [k for k, _ in sorted(zoo["monitors"].items(),
                                 key=lambda kv: (kv[1].get(field) or 0.0), reverse=True)]


_rp, _ra, _rr = _rank("posthoc_auc"), _rank("timely_auc"), _rank("timely_recall_mfpr")
pin("claim_constancy hindsight rank 5", 5, _rp.index("claim_constancy") + 1)
pin("claim_constancy timely-recall rank 9", 9, _rr.index("claim_constancy") + 1)
pin("claim_constancy timely recall 0.00", 0.0,
    zoo["monitors"]["claim_constancy"]["timely_recall_mfpr"] or 0.0)
pin("conf_constancy hindsight rank 1", 1, _rp.index("conf_constancy") + 1)
pin("conf_constancy timely-AUC rank 5", 5, _ra.index("conf_constancy") + 1)
pin("conf_constancy timely-recall rank 2", 2, _rr.index("conf_constancy") + 1)
pin("conf_constancy timely recall 0.12", 0.12,
    zoo["monitors"]["conf_constancy"]["timely_recall_mfpr"], tol=0.005)
pin("high_conf_share timely-AUC rank 1", 1, _ra.index("high_conf_share") + 1)
pin("high_conf_share timely recall 0.00", 0.0,
    zoo["monitors"]["high_conf_share"]["timely_recall_mfpr"] or 0.0)
# The paper no longer claims the scoreboard ranks by penalized delay: zoo.json
# carries no per-monitor delay, so that ranking is not computable.
pin("no per-monitor delay field exists", True,
    all("delay" not in f for d in zoo["monitors"].values() for f in d))
# (scoreboard sentence removed with the reference-implementation section, 2026-10-29)

# ---- S3 (per-family metrics, pinned as families complete) ----
s3q_f = P / "results" / "s3" / "metrics_qwen35.json"
if s3q_f.exists():
    s3q = json.loads(s3q_f.read_text(encoding="utf-8"))
    pl = s3q["by_adv"]["plain"]
    ji = s3q["by_adv"]["jitter"]
    pin("S3 qwen35 trials 20", 20, s3q["n_trials"])
    pin("S3 qwen35 captured 20", 20, pl["n_captured"] + ji["n_captured"])
    pin("S3 qwen35 plain posthoc AUC 0.91", 0.91, pl["posthoc_AUC"], tol=0.005)
    pin("S3 qwen35 plain timely recall 0.0", 0.0, pl["timely_recall_at_C"])
    pin("S3 qwen35 jitter posthoc AUC 0.53", 0.53, ji["posthoc_AUC"], tol=0.005)
    pin("S3 qwen35 plain AUC CI lo 0.77", 0.77, pl["posthoc_AUC_ci95"][0], tol=0.005)
    pin("S3 qwen35 plain AUC CI hi 1.00", 1.00, pl["posthoc_AUC_ci95"][1], tol=0.005)
    pin("S3 qwen35 recall zero in every resample", True,
        pl["timely_recall_ci95"] == [0.0, 0.0])
    pin("S3 qwen35 jitter CI lo 0.37", 0.37, ji["posthoc_AUC_ci95"][0], tol=0.0051)
    pin("S3 qwen35 jitter CI hi 0.70", 0.70, ji["posthoc_AUC_ci95"][1], tol=0.005)

s3g_f = P / "results" / "s3" / "metrics_gemma4.json"
if s3g_f.exists():
    s3g = json.loads(s3g_f.read_text(encoding="utf-8"))
    pl = s3g["by_adv"]["plain"]
    ji = s3g["by_adv"]["jitter"]
    pin("S3 gemma4 trials 20", 20, s3g["n_trials"])
    pin("S3 gemma4 captured 20", 20, pl["n_captured"] + ji["n_captured"])
    pin("S3 gemma4 plain posthoc AUC 0.85", 0.85, pl["posthoc_AUC"], tol=0.005)
    pin("S3 gemma4 plain timely recall 0.22", 0.22, pl["timely_recall_at_C"], tol=0.005)
    # data is exactly 0.375; tex prints the 2-dp rounding 0.38, a half-ULP case
    pin("S3 gemma4 round-2 AUC 0.38 (below chance)", 0.38,
        pl["roundwise"]["2"]["AUC"], tol=0.0051)
    pin("S3 gemma4 round-2 TPR 0 (no round-2 catches)", 0.0,
        pl["roundwise"]["2"]["TPR"])
    pin("S3 gemma4 round-3 AUC 0.99", 0.99, pl["roundwise"]["3"]["AUC"], tol=0.005)
    pin("S3 gemma4 jitter posthoc AUC 0.38 (inverted)", 0.38,
        ji["posthoc_AUC"], tol=0.005)
    pin("S3 gemma4 modal capture round is 2",
        2, max(set(pl["capture_rounds"]), key=pl["capture_rounds"].count))
    pin("S3 gemma4 plain AUC CI lo 0.70", 0.70, pl["posthoc_AUC_ci95"][0], tol=0.005)
    pin("S3 gemma4 plain recall CI lo 0.00", 0.00, pl["timely_recall_ci95"][0])
    pin("S3 gemma4 plain recall CI hi 0.50", 0.50, pl["timely_recall_ci95"][1], tol=0.005)
    pin("S3 gemma4 jitter CI lo 0.18", 0.18, ji["posthoc_AUC_ci95"][0], tol=0.005)
    pin("S3 gemma4 jitter CI hi 0.62", 0.62, ji["posthoc_AUC_ci95"][1], tol=0.005)
    pin("S3 gemma4 jitter CI includes chance (claim: at or below)", True,
        ji["posthoc_AUC_ci95"][0] < 0.5 <= ji["posthoc_AUC_ci95"][1])

# ---- latency (strict-convention) regime map ---------------------------------
# Every number the paper states about overseer latency is pinned here. The arm
# earns its keep by reproducing the published map under the intercept
# convention, so that block is pinned first: if the harness drifts, the
# reproduction fails before any strict number is quoted.
srm_f = P / "results" / "s1" / "strict_regime_map.json"
if srm_f.exists():
    import statistics as _st
    srm = json.loads(srm_f.read_text(encoding="utf-8"))
    val = srm["validation"]
    pin("strict-map harness reproduces the published regime counts", True,
        val["regime_counts_match"])
    pin("strict-map harness reproduces every published cell label", 60,
        val["per_cell_regime_match"])
    pin("strict-map harness reproduces the published H2 sign test", True,
        val["h2_timely_match"])
    ic = srm["h3_regimes"]["intercept"]["counts"]
    st = srm["h3_regimes"]["strict"]["counts"]
    la = srm["h3_latency_aware_library"]["strict"]["counts"]
    pin("intercept placement-critical 28", 28, ic["placement_critical"])
    pin("intercept budget-saturated 3", 3, ic["budget_saturated"])
    pin("intercept ceiling-bound 10", 10, ic["ceiling_bound"])
    pin("strict placement-critical 1", 1, st["placement_critical"])
    pin("strict budget-saturated 13", 13, st["budget_saturated"])
    pin("strict ceiling-bound 17", 17, st["ceiling_bound"])
    pin("latency-aware library placement-critical 19", 19, la["placement_critical"])
    pin("latency-aware library budget-saturated 4", 4, la["budget_saturated"])
    pin("latency-aware library ceiling-bound unchanged at 17", 17, la["ceiling_bound"])
    h2s = srm["h2_placement_vs_budget"]["strict"]["registered_recipe"]
    pin("strict H2 timely: no cell positive", 0, h2s["timely"]["k_positive"])
    pin("strict H2 timely: 11 negative", 11, h2s["timely"]["k_negative"])
    pin("strict H2 timely mean -0.15", -0.15, h2s["timely"]["mean_delta"], tol=0.0051)
    pin("strict H2 prevention: no cell positive", 0, h2s["prevention"]["k_positive"])
    pin("strict H2 prevention: 9 negative", 9, h2s["prevention"]["k_negative"])
    pin("strict H2 prevention: 3 ties", 3, h2s["prevention"]["n_ties"])
    pin("strict H2 prevention mean -0.09", -0.09, h2s["prevention"]["mean_delta"], tol=0.0051)
    ho = srm["heldout_placement_selection"]["by_convention"]["strict"]["any_single_round"]
    pin("held-out strict timely wins 11 of 12", 11, ho["timely"]["k_positive"])
    pin("held-out strict timely mean +0.26", 0.26, ho["timely"]["mean_delta"], tol=0.0051)
    pin("held-out strict timely CI lo +0.18", 0.18, ho["timely"]["paired_boot_ci"][0], tol=0.0051)
    pin("held-out strict timely CI hi +0.34", 0.34, ho["timely"]["paired_boot_ci"][1], tol=0.0051)
    pin("held-out strict timely p 4.9e-4", 4.9e-4, ho["timely"]["sign_p_one_sided"], tol=2e-5)
    pin("held-out strict prevention wins 12 of 12", 12, ho["prevention"]["k_positive"])
    pin("held-out strict prevention mean +0.20", 0.20, ho["prevention"]["mean_delta"], tol=0.0051)
    pin("held-out strict prevention CI lo +0.13", 0.13, ho["prevention"]["paired_boot_ci"][0], tol=0.0051)
    pin("held-out strict prevention CI hi +0.30", 0.30, ho["prevention"]["paired_boot_ci"][1], tol=0.0051)
    pin("held-out strict prevention p 2.4e-4", 2.4e-4, ho["prevention"]["sign_p_one_sided"], tol=2e-5)
    picks = {c: {p["picked"] for p in
                 srm["heldout_placement_selection"]["by_convention"][c]["any_single_round"]["timely"]["per_cell"]}
             for c in ("intercept", "strict")}
    pin("held-out picks round 2 in every windowed cell at zero latency", {2}, picks["intercept"])
    pin("held-out picks round 3 in every windowed cell under strict timing", {3}, picks["strict"])
    r2 = srm["exhaustive_single_round"]["convention_divergence_by_check_round"]["strict_round2_identical_to_no_check"]
    pin("strict round-2 check identical to no check, 3000 pairs", 3000, r2["n_pairs"])
    pin("strict round-2 check identical to no check, all of them", 3000, r2["n_identical"])
    med = {c: _st.median([x["by_convention"][c]["oracle_any"] for x in srm["cells"].values()
                          if x["by_convention"][c]["regime"] != "benign"])
           for c in ("intercept", "strict")}
    n_risk = sum(1 for x in srm["cells"].values()
                 if x["by_convention"]["intercept"]["regime"] != "benign")
    pin("risk cells 41", 41, n_risk)
    pin("median oracle prevention 0.18 at zero latency", 0.18, med["intercept"], tol=0.0051)
    pin("median oracle prevention halves to 0.08 under strict", 0.08, med["strict"], tol=0.0051)
    # intext() takes a regex; these literals carry TeX metacharacters
    for lit, desc in (("placement-critical cells fall from 28 to 1", "collapse in prose"),
                      ("$+0.26$", "held-out timely gain in prose"),
                      ("$+0.20$", "held-out prevention gain in prose"),
                      ("identical to no check in all 3{,}000 cell--seed pairs",
                       "round-2 identity in prose"),
                      ("the median prevention of the onset-informed reference from 0.18 to 0.08",
                       "median oracle prevention in prose")):
        intext(re.escape(lit), desc)
else:
    FAILS.append("results/s1/strict_regime_map.json missing (run src/strict_regime_map.py)")

# ---- PREREG H2's registered family: two budget contrasts, Holm within endpoint ----
# The preregistration registers Holm over the two budget contrasts; reporting only
# the doubled-budget one and calling it corrected would be a family of one.
_fam = s1["h2"]["registered_family"]
_t = _fam["tests"]
pin("H2 family has four tests (2 contrasts x 2 endpoints)", 4, len(_t))
pin("B3 timely wins 11 of 12 (one tie)", (11, 0, 1),
    (_t["vs_uniform@B3|timely"]["k_positive"], _t["vs_uniform@B3|timely"]["k_negative"],
     _t["vs_uniform@B3|timely"]["n_ties"]))
pin("B3 timely mean +0.30", 0.30, _t["vs_uniform@B3|timely"]["mean_delta"], tol=0.005)
pin("B3 timely CI lo +0.16", 0.16, _t["vs_uniform@B3|timely"]["paired_boot_ci"][0], tol=0.0051)
pin("B3 timely CI hi +0.45", 0.45, _t["vs_uniform@B3|timely"]["paired_boot_ci"][1], tol=0.0051)
pin("B3 prevention wins 12 of 12", 12, _t["vs_uniform@B3|prevention"]["k_positive"])
pin("B3 prevention mean +0.31", 0.31, _t["vs_uniform@B3|prevention"]["mean_delta"], tol=0.005)
pin("B3 prevention CI lo +0.18", 0.18, _t["vs_uniform@B3|prevention"]["paired_boot_ci"][0], tol=0.0051)
pin("B3 prevention CI hi +0.47", 0.47, _t["vs_uniform@B3|prevention"]["paired_boot_ci"][1], tol=0.0051)
pin("B2 contrast reproduces the headline mean +0.41", 0.41,
    _t["vs_uniform@B2|timely"]["mean_delta"], tol=0.005)
pin("Holm max adjusted p 9.8e-4", 9.8e-4, _fam["max_adjusted_p"], tol=5e-5)
pin("every Holm-adjusted p stays under .001", True,
    all(v < 1e-3 for v in _fam["holm_adjusted_p"].values()))
for _lit, _desc in (("$9.8\\times10^{-4}$", "Holm-adjusted bound in prose"),
                    ("All four Holm-adjusted $p$-values are at most", "Holm bound in prose"),
                    ("11 of 12 on timely catch, with one tie (mean $+0.30$)", "B3 timely in prose"),
                    ("12 of 12 cells on prevention (mean $+0.31$)", "B3 prevention in prose")):
    intext(re.escape(_lit), _desc)

# ---- ceiling-bound composition (the prose describes ten cells; check the split) ----
_cb = {k: v for k, v in s1["cells"].items() if v["regime"] == "ceiling_bound"}
pin("ceiling-bound cells 10", 10, len(_cb))
pin("nine of ten ceiling-bound cells at the weakest evidence", 9,
    sum(1 for k in _cb if k.split("|")[1] == "0.58"))
_fast = {k: v for k, v in _cb.items() if v["median_capture"] is not None}
pin("six ceiling-bound cells capture by round 1-2", 6, len(_fast))
pin("those six are five hub and one small-world", (5, 1),
    (sum(1 for k in _fast if k.endswith("hub")),
     sum(1 for k in _fast if k.endswith("small_world"))))
pin("fast ceiling-bound cells capture by round 2.5 or sooner", True,
    max(v["median_capture"] for v in _fast.values()) <= 2.5)
pin("fast ceiling-bound baselines 0.96-1.00", True,
    all(0.96 <= v["fc_baseline"] <= 1.00 for v in _fast.values()))
_slow = {k: v for k, v in _cb.items() if v["median_capture"] is None}
pin("other four ceiling-bound cells capture in 10-24% of trials", True,
    len(_slow) == 4 and all(0.10 <= v["fc_baseline"] <= 0.24 for v in _slow.values()))

# ---- S2: the misspecified half is a floor, not a replication ----
_mis = {k: v for k, v in s2["cells"].items() if k.endswith("misspecified")}
pin("ten misspecified cells", 10, len(_mis))
pin("at B<=4 every misspecified arm misses every shift", True,
    all(a["miss"] == 1.0 for v in _mis.values() for k, a in v.items()
        if k.rsplit("@B", 1)[1] in ("1", "2", "4")))
pin("the onset reference misses too (no policy separates)", True,
    all(v[f"oracle@B{B}"]["miss"] == 1.0 for v in _mis.values() for B in (1, 2, 4)))
pin("registered misspecification clause is an exact tie at every budget", True,
    all(d == 0.0 for B in ("B1", "B2", "B4") for d in s2["h5"][B]["mis_mean_deltas"]))
pin("front-collapse curves come from nine well-specified cells", 9,
    len({k.rsplit("|", 1)[0] for k, v in s2["cells"].items()
         if k.endswith("well_specified")
         and any(a["miss"] < 1.0 for a in v.values())}))
for _lit, _desc in (("in all ten such cells at budgets up to four, every policy",
                     "S2 floor scope in prose"),
                    ("on the nine cells whose candidate set contains the target rule",
                     "S2 curve denominator in prose")):
    intext(re.escape(_lit), _desc)

# ---- S4: matched FPR drifts in BOTH directions after transfer ----
_mps_f = P / "results" / "s4" / "metric_pair_split.json"
if _mps_f.exists():
    _mps = json.loads(_mps_f.read_text(encoding="utf-8"))
    _op = _mps["summary"]["strict"]["operating_point_is_not_matched_after_transfer"]
    pin("nine of thirteen variants realize exactly zero FPR", 9,
        sum(1 for v in _op.values() if v == 0.0))
    pin("thirteen variants in the transfer table", 13, len(_op))
    pin("conf_mean realizes 0.156 FPR", 0.156, _op["conf_mean"], tol=0.0005)
    pin("conf_mean is the variant carrying the best same-threshold recall", "conf_mean",
        _mps["summary"]["strict"]["best"]["timely_recall_same_threshold"]["monitor"])
    pin("best recall at a genuine 6.25% deadline FPR is 0.18", 0.18,
        _mps["summary"]["strict"]["best"]["timely_recall_at_own_fpr"]["value"], tol=0.005)
    pin("re-setting thresholds lowers the wall, not raises it", True,
        _mps["summary"]["strict"]["best"]["timely_recall_at_own_fpr"]["value"]
        < _mps["summary"]["strict"]["best"]["timely_recall_same_threshold"]["value"])
else:
    FAILS.append("results/s4/metric_pair_split.json missing (run src/metric_pair_split.py)")


# ---- instrument sensitivity: is the map the dynamics or one detector? ----
_is_f = P / "results" / "s1" / "instrument_sensitivity.json"
if _is_f.exists():
    _is = json.loads(_is_f.read_text(encoding="utf-8"))
    pin("instrument sweep reproduces the published map at the deployed threshold",
        True, _is["reproduction_check"]["passes"])
    # "both sign tests to the digit" -- assert the prevention test directly from
    # the shipped data, independent of which gate version wrote the JSON
    _pv = _is["arms"]["theta=0.50"]["h2"]["fc_prevention"]
    _pub_pv = s1["h2"]["registered_family"]["tests"]["vs_uniform@B2|prevention"]
    pin("sweep reproduces the prevention sign test to the digit", True,
        _pv["k_positive"] == _pub_pv["k_positive"]
        and _pv["n_ties"] == _pub_pv["n_ties"]
        and abs(_pv["mean_delta"] - _pub_pv["mean_delta"]) < 1e-9
        and _pv["sign_p_one_sided"] == _pub_pv["sign_p_one_sided"])
    _dep, _perf = _is["arms"]["theta=0.50"], _is["arms"]["perfect"]
    pin("deployed arm regimes match the paper", (28, 10, 3, 19),
        tuple(_dep["regime_counts"][k] for k in ("placement_critical", "ceiling_bound",
                                                 "budget_saturated", "benign")))
    pin("perfect detector leaves 8 cells ceiling-bound", 8,
        _perf["regime_counts"]["ceiling_bound"])
    pin("eight of the ten ceiling-bound cells survive a perfect detector", 8,
        len([c for c in _dep["ceiling_bound_cells"] if c in _perf["ceiling_bound_cells"]]))
    pin("the two freed cells become placement-critical", ["placement_critical"] * 2,
        [_perf["cells"][c]["regime"] for c in _dep["ceiling_bound_cells"]
         if c not in _perf["ceiling_bound_cells"]])
    pin("all five round-1-capture ceiling cells survive a perfect detector", True,
        all(c in _perf["ceiling_bound_cells"] for c in _dep["ceiling_bound_cells"]
            if s1["cells"][c]["median_capture"] == 1.0))
    pin("placement-critical grows 28 -> 30 under a perfect detector", 30,
        _perf["regime_counts"]["placement_critical"])
    pin("H2 timely is unchanged under a perfect detector", True,
        _perf["h2"]["timely"]["k_positive"] == _dep["h2"]["timely"]["k_positive"] == 11
        and abs(_perf["h2"]["timely"]["mean_delta"] - _dep["h2"]["timely"]["mean_delta"]) < 1e-12)
    pin("H2 prevention is unchanged under a perfect detector", True,
        _perf["h2"]["fc_prevention"]["k_positive"] == 12
        and abs(_perf["h2"]["fc_prevention"]["mean_delta"]
                - _dep["h2"]["fc_prevention"]["mean_delta"]) < 1e-12)
    pin("perfect-detector H2 timely mean +0.41", 0.41,
        _perf["h2"]["timely"]["mean_delta"], tol=0.005)
    pin("perfect-detector H2 prevention mean +0.39", 0.39,
        _perf["h2"]["fc_prevention"]["mean_delta"], tol=0.005)
    pin("median oracle prevention 0.18 under both instruments", True,
        abs(_dep["median_oracle_prevention_over_risk_cells"] - 0.18) < 0.0051
        and abs(_perf["median_oracle_prevention_over_risk_cells"] - 0.18) < 0.0051)
    for _th in ("theta=0.55", "theta=0.60", "theta=0.65"):
        pin(f"raising theta to {_th[6:]} grows ceiling-bound to 14", 14,
            _is["arms"][_th]["regime_counts"]["ceiling_bound"])
        pin(f"raising theta to {_th[6:]} weakens the contrast to 8 of 11", (8, 3),
            (_is["arms"][_th]["h2"]["timely"]["k_positive"],
             _is["arms"][_th]["h2"]["timely"]["k_negative"]))
    pin("high-theta contrast mean +0.28", 0.28,
        _is["arms"]["theta=0.55"]["h2"]["timely"]["mean_delta"], tol=0.0051)
    pin("theta=0.35 shrinks ceiling-bound to 6", 6,
        _is["arms"]["theta=0.35"]["regime_counts"]["ceiling_bound"])
    pin("theta 0.45 and 0.50 agree on every regime count", True,
        _is["arms"]["theta=0.45"]["regime_counts"] == _dep["regime_counts"])
    pin("theta 0.45 and 0.50 agree on both sign tests", True,
        _is["arms"]["theta=0.45"]["h2"] == _dep["h2"])
    _cfi = _is.get("clean_false_intervention", {}).get("arms", {})
    if _cfi:
        pin("theta=0.35 fires on every clean trial", 1.00,
            _cfi["theta=0.35"]["max_false_intervention_rate"], tol=1e-9)
        pin("theta=0.40 fires on every clean trial", 1.00,
            _cfi["theta=0.40"]["max_false_intervention_rate"], tol=1e-9)
        pin("theta=0.45 clean false-intervention 0.36", 0.36,
            _cfi["theta=0.45"]["max_false_intervention_rate"], tol=0.0051)
        pin("deployed theta clean false-intervention 0.02", 0.02,
            _cfi["theta=0.50"]["max_false_intervention_rate"], tol=0.0051)
        pin("the perfect detector never false-intervenes", 0.0,
            _cfi["perfect"]["max_false_intervention_rate"], tol=1e-9)
    else:
        FAILS.append("instrument_sensitivity.json lacks clean_false_intervention "
                     "(run src/instrument_clean_arms.py)")
    for _lit, _desc in (("two ceiling-bound cells become placement-critical",
                         "perfect-detector ceiling claim in prose"),
                        ("grows from 28 to 30 cells", "placement-critical growth in prose"),
                        ("they flag an honest agent in every fault-free run",
                         "low-theta cost in prose"),
                        ("Under the perfect detector, still checking from round 2",
                         "perfect-arm scope condition disclosed"),
                        ("Their windows close before round 2, the first check we allow",
                         "round-1-capture mechanism in prose"),
                        ("captured in the first rounds or rarely.",
                         "abstract states both ceiling causes")):
        intext(re.escape(_lit), _desc)
else:
    FAILS.append("results/s1/instrument_sensitivity.json missing "
                 "(run src/instrument_sensitivity.py)")

# ---- variance components behind the cell-level test ----
_vc_f = P / "results" / "s1" / "variance_components.json"
if _vc_f.exists():
    _vc = json.loads(_vc_f.read_text(encoding="utf-8"))["contrasts"]
    _t = _vc["vs_uniform@B2|timely"]
    pin("timely ICC 0.36", 0.36, _t["icc"], tol=0.005)
    pin("timely design effect 18.6", 18.6, _t["design_effect"], tol=0.05)
    pin("600 seed-level observations", 600, _t["n_obs"])
    pin("effective n 32", 32.0, _t["effective_n"], tol=0.5)
    pin("seed-level SE understates by 4.3x", 4.3, _t["naive_se_understates_by_factor"], tol=0.05)
    pin("random-effects SE 0.088", 0.088, _t["se_random_effects"], tol=0.0005)
    pin("random-effects mean reproduces the published +0.41", 0.4067,
        _t["mean"], tol=1e-4)
    _p = _vc["vs_uniform@B2|prevention"]
    pin("prevention ICC 0.26", 0.26, _p["icc"], tol=0.005)
    pin("prevention understatement 3.7x", 3.7, _p["naive_se_understates_by_factor"], tol=0.05)
    intext(re.escape("carry the information of about 32 independent observations"),
           "effective n in prose")
    intext(re.escape("too small by a factor of 4.3"), "SE understatement in prose")
    intext(re.escape("the intraclass correlation of the per-seed paired difference is 0.36"),
           "ICC in prose")
else:
    FAILS.append("results/s1/variance_components.json missing "
                 "(run src/variance_components_s1.py)")

# ---- REWRITE 2026-10-29: numbers that first appear in the rewritten text ----
# Every row of the sensitivity table is generated from the JSON and must appear
# verbatim, so a hand edit to the table fails here.
_is = json.loads((P / "results" / "s1" / "instrument_sensitivity.json").read_text(encoding="utf-8"))
_ct = json.loads((P / "results" / "s1" / "capture_threshold_sensitivity.json").read_text(encoding="utf-8"))
_fi = _is["clean_false_intervention"]["arms"]
pin("capture-threshold sweep reproduces the registered 80% arm", True,
    _ct["reproduction_check"]["passes"])


def _row(label, arm, fi):
    rc, pv = arm["regime_counts"], arm["h2"]["fc_prevention"]
    tail = f"{fi:.2f}" if fi is not None else "--"
    return (f"{label} & {rc.get('placement_critical', 0)} & {rc.get('ceiling_bound', 0)} & "
            f"{rc.get('budget_saturated', 0)} & {pv['n_windowed']} & "
            f"{pv['k_positive']}/{pv['n_windowed']}, ${pv['mean_delta']:+.2f}$ & {tail} \\\\")


for _k, _lab in (("theta=0.35", "detector threshold 0.35"),
                 ("theta=0.40", "detector threshold 0.40"),
                 ("theta=0.45", "detector threshold 0.45"),
                 ("theta=0.50", "detector threshold 0.50 (benchmark)"),
                 ("theta=0.55", "detector threshold 0.55--0.65"),
                 ("perfect", "perfect detector")):
    intext(re.escape(_row(_lab, _is["arms"][_k], _fi[_k]["max_false_intervention_rate"])),
           f"sensitivity table row: {_k}")
pin("theta 0.55, 0.60, 0.65 are identical, so one merged row is exact", True,
    all(_is["arms"][k]["regime_counts"] == _is["arms"]["theta=0.55"]["regime_counts"]
        and _is["arms"][k]["h2"] == _is["arms"]["theta=0.55"]["h2"]
        and _fi[k]["max_false_intervention_rate"] == _fi["theta=0.55"]["max_false_intervention_rate"]
        for k in ("theta=0.60", "theta=0.65")))
for _k, _lab in (("0.60", "capture threshold 60\\%"), ("0.70", "capture threshold 70\\%"),
                 ("0.80", "capture threshold 80\\% (registered)"),
                 ("0.90", "capture threshold 90\\%")):
    intext(re.escape(_row(_lab, _ct["arms"][_k], None)), f"sensitivity table row: capture {_k}")

_wins = [_ct["arms"][k]["h2"]["fc_prevention"]["n_windowed"] for k in ("0.60", "0.70", "0.80", "0.90")]
pin("windowed cells 7, 8, 12, 16 across capture thresholds", [7, 8, 12, 16], _wins)
intext(re.escape("(7, 8, 12, and 16 at 60\\% through 90\\%)"), "windowed counts in prose")
_pm = [_ct["arms"][k]["h2"]["fc_prevention"]["mean_delta"] for k in ("0.60", "0.70", "0.80", "0.90")]
pin("prevention gains span +0.31..+0.43", (0.31, 0.43), (round(min(_pm), 2), round(max(_pm), 2)))
intext(re.escape("mean gains between $+0.31$ and $+0.43$"), "gain range in prose")
pin("only loss-free arms: every windowed cell wins or ties on prevention", True,
    all(_ct["arms"][k]["h2"]["fc_prevention"]["k_negative"] == 0 for k in _ct["arms"]))
pin("exactly one tie, at 90%", (1, 0, 0, 0),
    tuple(_ct["arms"][k]["h2"]["fc_prevention"]["n_ties"] for k in ("0.90", "0.60", "0.70", "0.80")))
intext(re.escape("except for one tie at 90\\%"), "tie disclosed in prose")
_pcs = [_ct["arms"][k]["regime_counts"].get("placement_critical", 0) for k in _ct["arms"]]
pin("placement-critical 25..30 across thresholds", (25, 30), (min(_pcs), max(_pcs)))
intext(re.escape("Placement-critical cells stay between 25 and 30 of 60"), "P range in prose")

# high-threshold detector arms
_hi = _is["arms"]["theta=0.55"]
pin("theta>=0.55 ceiling 14, prevention 9 of 12", (14, 9, 12),
    (_hi["regime_counts"]["ceiling_bound"], _hi["h2"]["fc_prevention"]["k_positive"],
     _hi["h2"]["fc_prevention"]["n_windowed"]))
intext(re.escape("the ceiling grows to 14 cells, and the single check wins in 9 of 12 windowed cells on prevention"),
       "high-threshold arm in prose")
intext(re.escape("the false-intervention rate on fault-free runs is 0.36 at 0.45 against 0.02 at 0.50"),
       "plateau false-intervention contrast in prose")
pin("0.36 vs 0.02 false intervention", (0.36, 0.02),
    (round(_fi["theta=0.45"]["max_false_intervention_rate"], 2),
     round(_fi["theta=0.50"]["max_false_intervention_rate"], 2)))

# latency block
_srm = json.loads((P / "results" / "s1" / "strict_regime_map.json").read_text(encoding="utf-8"))
pin("ten placement-critical cells turn intermediate under latency", 10,
    _srm["regime_transitions"]["matrix"]["placement_critical->intermediate"])
intext(re.escape("ten former placement-critical cells become intermediate"), "intermediate cells in prose")
_st = _srm["h2_placement_vs_budget"]["strict"]["registered_recipe"]["timely"]
pin("fixed library loses 11 of 12 under latency, mean -0.15", (11, -0.15),
    (_st["k_negative"], round(_st["mean_delta"], 2)))
intext(re.escape("loses on timely catch in 11 of 12 (mean $-0.15$"), "latency loss in prose")
_ho = _srm["heldout_placement_selection"]["by_convention"]
intext(re.escape("the chosen round is 2 in all twelve windowed cells at zero latency and 3 in all "
                 "twelve at one round of latency"), "held-out chosen rounds in prose")

# exhaustive bound
_to = json.loads((P / "results" / "s1" / "true_oracle.json").read_text(encoding="utf-8"))
intext(re.escape("the clairvoyant bound prevents 13 of 326 captures (4.0\\%)"), "ceiling prevention in prose")
intext(re.escape("agree with the exhaustive bound in 57 of 60 cells"), "label agreement in prose")
intext(re.escape("makes 98.5\\% of the achievable timely catches but only 50\\%"),
       "detect-vs-prevent timing in prose")

# rank agreement (metric_pair_split, matched units)
_tl = json.loads((P / "results" / "s4" / "metric_pair_split.json").read_text(encoding="utf-8"))["summary"]
_s, _z = _tl["strict"]["tau_ladder"], _tl["intercept"]["tau_ladder"]
pin("tau ladder 0.64/0.97/0.72/0.90/0.51/0.46",
    (0.64, 0.97, 0.72, 0.90, 0.51, 0.46),
    tuple(round(x, 2) for x in (_s["like_for_like_auc_pair"], _z["like_for_like_auc_pair"],
                                _s["like_for_like_recall_pair_same_fpr"],
                                _z["like_for_like_recall_pair_same_fpr"],
                                _s["headline_mixed_auc_vs_recall"],
                                _s["control_metric_type_only_both_in_hindsight"])))
for _lit in ("Kendall $\\tau = 0.64$ with one round of latency and 0.97 with zero latency",
             "it is 0.72 and 0.90",
             "gives $\\tau = 0.51$ $[0.36, 0.56]$",
             "already gives $\\tau = 0.46$"):
    intext(re.escape(_lit), "tau sentence: " + _lit[:30])

# abstract numbers tied to their sources
for _lit, _d in (("evenly spaced checks that start later have prevention rates 0.39 lower",
                  "abstract prevention gain"),
                 ("all but 8 of 642{,}460 multi-check outcomes", "abstract window-rule fit"),
                 ("early schedules miss every shift at budgets up to four", "abstract S2 collapse")):
    intext(re.escape(_lit), _d)

# ---- REWRITE 2026-10-29 design pins: counts and the onset sweep ----
def _jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


_t1 = _jsonl(P / "results" / "s1" / "trials.jsonl")
pin("S1 trial count 72,000", 72000, len(_t1))
_t2 = _jsonl(P / "results" / "s2" / "trials.jsonl")
pin("S2 trial count 14,400", 14400, len(_t2))
intext(re.escape("24 seeds (14{,}400 trials)"), "S2 seeds and trials")
pin("S2 seeds 24", 24, len({t["seed"] for t in _t2}))
pin("S2 cells 10", 10, len({t["cell"] for t in _t2}))
pin("S2 budgets 1,2,4,8,16", [1, 2, 4, 8, 16], sorted({t["budget"] for t in _t2}))
pin("S2 separability floor 0", 0.0, min(t["rho"] for t in _t2))
pin("S2 separability ceiling 0.71", 0.71, max(t["rho"] for t in _t2), tol=0.005)
intext(re.escape("over ten cells ranging from 0 to 0.71"), "S2 separability range")
pin("S2 onset range 40..120", (40, 120), (min(t["tau"] for t in _t2), max(t["tau"] for t in _t2)))
pin("S2 stream length 160", {160}, {t["tau"] + t["horizon"] for t in _t2})
intext(re.escape("drawn uniformly from $[40, 120]$"), "S2 onset law")
pin("regime share placement-critical 47% (28 of 60)", 47, round(100 * 28 / 60))
intext(re.escape("(28 of 60, 47\\%)"), "placement-critical share")

sys.path.insert(0, str(P / "src"))
import s2_engine as _s2                                              # noqa: E402
pin("geometric checks at 2,4,8,16 for B<=4", [2, 4, 8, 16], sorted(_s2.s2_geometric(4)))
pin("geometric B1, B2 are prefixes", ([2], [2, 4]),
    (sorted(_s2.s2_geometric(1)), sorted(_s2.s2_geometric(2))))
intext(re.escape("puts all its checks at steps 2--16"), "geometric window in prose")
intext(re.escape("geometric spacing puts the checks at steps 2, 4, 8, and 16"), "geometric steps in prose")

_os = json.loads((P / "results" / "s2" / "onset_accuracy_sweep.json").read_text(encoding="utf-8"))
pin("onset sweep seeds 50", 50, _os["meta"]["seeds"]["n"])
pin("onset sweep uses the nine diagnosable cells", 9, len(_os["meta"]["diagnosable_cells"]))
_osd = _os["meta"]["onset_distributions"]
pin("onset sweep early law 2..30", (2, 30), (_osd["early_u2_30"]["lo"], _osd["early_u2_30"]["hi"]))
pin("onset sweep broad law 5..155", (5, 155), (_osd["broad_u5_155"]["lo"], _osd["broad_u5_155"]["hi"]))
intext(re.escape("In a separate sweep with 50 new seeds on the same nine cells"), "onset sweep design")
_hl_ = _os["headline"]
for _law, _want in (("early_u2_30", (0.97, 0.87, 0.46)), ("broad_u5_155", (1.00, 1.00, 0.91)),
                    ("bimodal_6_20__95_135", (1.00, 1.00, 0.64))):
    for _B, _w in zip(("B1", "B2", "B4"), _want):
        pin(f"onset sweep {_law} audit-early miss {_B} {_w:.2f}", _w,
            _hl_[f"audit_early_miss|{_law}"][_B], tol=0.005)
pin("onset sweep early law: uniform misses none at B<=4", 0.0,
    max(_hl_["uniform_miss|early_u2_30"][b] for b in ("B1", "B2", "B4")))
intext(re.escape("still leaves audit-early missing 0.97, 0.87, and 0.46 of shifts"), "early-law misses")
intext(re.escape("leaves audit-early at 1.00, 1.00, and 0.91, and 1.00, 1.00, and 0.64"),
       "broad and bimodal misses")
_cx = _os["crossover"]["paper_u40_120|B1|noise|aware_point"]["penalized_delay"]
pin("onset tolerance 8.2 steps", 8.2, _cx["crossover_E"], tol=0.05)
pin("onset tolerance CI [3.0, 19.1]", (3.0, 19.1),
    tuple(round(x, 1) for x in _cx["crossover_E_ci95"]))
pin("crossover exists in every bootstrap draw", 1.0, _cx["boot_frac_with_crossover"])
intext(re.escape("8.2 steps (95\\% CI $[3.0, 19.1]$, over 24 seeds), about a tenth of the 80-step range"),
       "onset tolerance in prose")
pin("8.2 is about a tenth of the 80-step onset range", 0.10, 8.2 / (120 - 40), tol=0.005)

# ---- the compiled figure is the regenerated one, byte for byte ----
# main.tex reads figures/*.pdf relative to paper/, i.e. a COPY. A regenerated
# figure that was never copied ships stale art under fresh prose, and no other
# gate can see it.
import hashlib as _hl
# both builds use figures/make_figs.py output, drawn at the printed size
_srcdir = P / "figures" / "final"
_texdir = TEX_PATH.parent / "figures"
_missing, _stale = [], []
_used = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", TEX)
pin("the manuscript includes figures", True, len(_used) > 0)
for _name in sorted(set(_used)):
    _f, _t = _srcdir / _name, _texdir / _name
    if not (_f.exists() and _t.exists()):
        _missing.append(_name)
    elif _hl.sha256(_t.read_bytes()).hexdigest() != _hl.sha256(_f.read_bytes()).hexdigest():
        _stale.append(_name)
pin("every included figure exists in figures/ and next to the manuscript", [], _missing)
pin("no included figure is stale against its source", [], _stale)

# ---- R10 review supplement S2 (prereg/PROTOCOL_SUPPLEMENT_S2_review_experiments.md) ----
_R2 = P / "results" / "s1" / "review_s2"
_A = json.loads((_R2 / "analysis.json").read_text(encoding="utf-8"))
_WD = json.loads((_R2 / "window_details.json").read_text(encoding="utf-8"))
_CAL = json.loads((_R2 / "calibration.json").read_text(encoding="utf-8"))
pin("S2 re-run reproduces every stored L0 pair", (63000, 0),
    (_A["validation"]["L0_pairs_checked"], _A["validation"]["L0_mismatch"]))
pin("S2 re-run reproduces every stored L1 value", (1260, []),
    (_A["validation"]["L1_cells_checked"], _A["validation"]["L1_mismatch"]))
# window rule
pin("windows are intervals in every captured run (all enumerated arms)", True,
    all(v["window_is_interval"] == 1.0 for v in _A["window_model"].values()))
pin("union rule misses 8 of 642,460 at zero latency", (8, 642460),
    (_WD["BENCH_L0_STD"]["union_rule_disagreements"], _WD["BENCH_L0_STD"]["multi_check_outcomes"]))
pin("every windowed-cell window opens at round 2 at zero latency", ["2"],
    list(_WD["BENCH_L0_STD"]["window_open_round_windowed"]))
pin("best single round matches the bound in 38 of 41, one run short elsewhere", (41, 38, 1),
    (_WD["BENCH_L0_STD"]["cells_with_preventable_capture"], _WD["BENCH_L0_STD"]["best_single_equals_bound"],
     _WD["BENCH_L0_STD"]["best_single_max_shortfall_runs"]))
pin("held-out best fixed schedule 0.62 at B=1,2,3 equals the clairvoyant bound", True,
    all(round(v, 2) == 0.62 for v in _A["P2"]["BENCH_L0_STD"]["heldout_mean_windowed"].values())
    and abs(_A["P2"]["BENCH_L0_STD"]["clairvoyant_mean_windowed"]
            - _A["P2"]["BENCH_L0_STD"]["heldout_mean_windowed"]["1"]) < 1e-12)
for _lit, _desc in (("every window is an interval and the rule predicts all but 8 of 642{,}460",
                     "abstract window-rule claim"),
                    ("every window that exists opens at round 2, the first allowed check",
                     "window opening in prose"),
                    ("its held-out prevention rate is 0.62 with one, two, or three checks",
                     "held-out schedule in prose"),
                    ("the best single round matches the bound in 38 and falls one run short in the other three",
                     "best single round in prose")):
    intext(re.escape(_lit), _desc)
# latency by detector (Table tab:latency)
_AB = {"P": "placement_critical", "C": "ceiling_bound", "S": "budget_saturated", "I": "intermediate"}
for _det, _name in (("BENCH", "benchmark"), ("SINGLE", "single-message"), ("PERFECT", "perfect")):
    for _l, _lname in ((0, "zero"), (1, "one round")):
        _cnt = _A["regimes"][f"{_det}_L{_l}_STD"]["counts"]
        _p3 = _A["P3"][_det]
        _row = (f"{_name} & {_lname} & " + " & ".join(str(_cnt.get(_AB[k], 0)) for k in "PCSI")
                + f" & {_p3[f'windowed_bound_L{_l}']:.2f} & {_p3[f'median_bound_L{_l}']:.2f} & "
                + f"{_A['false_intervention'][f'{_det}_L{_l}']['single_any_round']:.3f}")
        intext(re.escape(_row), f"latency table row: {_det} L{_l}")
pin("single-message thresholds 0.72 / 0.70", (0.72, 0.70),
    (_CAL["theta_single"]["0"], _CAL["theta_single"]["1"]))
intext(re.escape("(0.72 at zero latency, 0.70 at one round)"), "single-message thresholds in prose")
pin("median bound halves 0.24 -> 0.12; reference 0.18 -> 0.08", (0.24, 0.12, 0.18, 0.08),
    tuple(round(_A["P3"]["BENCH"][k], 2) for k in
          ("median_bound_L0", "median_bound_L1", "median_reference_L0", "median_reference_L1")))
intext(re.escape("the median clairvoyant bound halves from 0.24 to 0.12"), "median bound under latency")
# capture round vs end of window
_p4 = _A["P4"]
pin("captured runs 1,412; window end after capture in 125", (1412, 125),
    (_p4["captured_pairs"], sum(v for g, v in _p4["gap_hist"].items() if int(g) < 0)))
pin("window end before capture in 45% of runs; median gap 0", (0.45, 0),
    (round(_p4["share_gap_ge1"], 2), _p4["median_gap"]))
intext(re.escape("falls at least one round earlier in 45\\% of captured runs, and lies after the "
                 "capture round in 125 of 1{,}412"), "window end vs capture round in prose")
# ceiling cells
_cr = {r["cell"]: r for r in _A["ceiling_relative"]}
pin("ten ceiling cells: bound 13 of 326", (10, 13, 326),
    (len(_cr), sum(r["bound_prevented"] for r in _cr.values()), sum(r["captures"] for r in _cr.values())))
_r1 = [r for r in _cr.values() if r["median_capture"] == 1.0]
_early = [r for r in _cr.values() if r["median_capture"] == 2.5]
_rare = [r for r in _cr.values() if r["median_capture"] is None]
pin("five round-1 cells: perfect round-1 check prevents 246 of 249", (5, 246, 249),
    (len(_r1), sum(r["perfect_round1_prevented"] for r in _r1), sum(r["captures"] for r in _r1)))
pin("round-1 cells are four hub cells and one small-world cell", (4, 1),
    (sum(r["cell"].endswith("hub") for r in _r1), sum(r["cell"].endswith("small_world") for r in _r1)))
pin("the sixth early cell (median 2.5): 48 of 48", ("1|0.58|hub", 48, 48),
    (_early[0]["cell"], _early[0]["perfect_round1_prevented"], _early[0]["captures"]))
pin("four rare cells: bound 11 of 29, perfect 22", (4, 11, 29, 22),
    (len(_rare), sum(r["bound_prevented"] for r in _rare), sum(r["captures"] for r in _rare),
     sum(r["perfect_bound_prevented"] for r in _rare)))
pin("rare cells captured in 10-24% of runs", (0.10, 0.24),
    (min(r["captures"] for r in _rare) / 50, max(r["captures"] for r in _rare) / 50))
pin("perfect detector turns 1|0.58|hub and 1|0.58|random placement-critical", True,
    all(_A["regimes"]["PERFECT_L0_STD"]["labels"][c] == "placement_critical"
        for c in ("1|0.58|hub", "1|0.58|random")))
for _lit, _desc in (("prevents 246 of the 249 captures in the five round-1 cells and all 48 in the sixth",
                     "round-1 perfect check in prose"),
                    ("it prevents 11 of their 29 captures (38\\%), and the perfect detector 22 (76\\%)",
                     "rare-cell relative prevention in prose"),
                    ("captured in only 10--24\\% of runs", "rare-cell capture rates in prose")):
    intext(re.escape(_lit), _desc)
# adversaries that move the window
_e = _A["P5"]["L0"]
pin("EVADE cuts audit-early prevention 0.44 [0.59, 0.31], 12 of 12", (-0.44, -0.59, -0.31, 12),
    tuple(round(x, 2) for x in _e["geoB1_evade_minus_std"]["mean_ci"]) + (_e["geoB1_evade_minus_std"]["k_neg"],))
pin("one random check vs evaded check: 5 wins, 4 losses, 3 ties", (5, 4, 3),
    (_e["randomB1_minus_geoB1_evade"]["k_pos"], _e["randomB1_minus_geoB1_evade"]["k_neg"],
     _e["randomB1_minus_geoB1_evade"]["ties"]))
pin("three random checks beat it 12 of 12, +0.15 [0.11, 0.20]", (12, 0.15, 0.11, 0.20),
    (_e["randomB3_minus_geoB1_evade"]["k_pos"],)
    + tuple(round(x, 2) for x in _e["randomB3_minus_geoB1_evade"]["mean_ci"]))
pin("camouflage cost 24% (L0) and 87% (L1)", (0.24, 0.87),
    (round(_e["prevented_without_any_prune_share"], 2),
     round(_A["P5"]["L1"]["prevented_without_any_prune_share"], 2)))
pin("delayed onset: windowed capture 0.745 -> 0.113", (0.745, 0.113),
    tuple(round(_WD["base_capture_windowed"][k] + 1e-9, 3) for k in ("BENCH_L0_STD", "BENCH_L0_DELAY")))
pin("delayed onset: uniform B2 loses to round-2 check in 10 of 12", (0, 10),
    (_A["P6"]["L0"]["uniformB2_minus_geoB1"]["k_pos"], _A["P6"]["L0"]["uniformB2_minus_geoB1"]["k_neg"]))
for _lit, _desc in (("cuts the prevention of the audit-early check by 0.44 on the windowed cells "
                     "(CI $[-0.59, -0.31]$, 12 of 12)", "EVADE cost in prose"),
                    ("(5 wins, 4 losses, 3 ties; we had predicted at least 9 wins)", "random B1 in prose"),
                    ("12 of 12 cells (mean $+0.15$, CI $[+0.11, +0.20]$)", "random B3 in prose"),
                    ("24\\% are no longer captured", "camouflage cost L0 in prose"),
                    ("this share rises to 87\\%", "camouflage cost L1 in prose"),
                    ("falls from 0.745 to 0.113", "delayed-onset capture in prose"),
                    ("loses to it in 10 of 12 cells, against our prediction", "delayed-onset uniform in prose")):
    intext(re.escape(_lit), _desc)
# suspicion trigger in the S2 fault-free runs
pin("suspicion trigger intervenes in 2.7% of fault-free runs", 0.027,
    _A["false_intervention"]["BENCH_L0"]["suspicion@B1"], tol=0.0005)
intext(re.escape("it intervened in 2.7\\% of fault-free runs"), "suspicion false intervention in prose")

# the window figure's example cell and its counts
_ex = _WD["example_cell"]
pin("window figure: example cell 4|0.74|hub, 44 captured", ("4|0.74|hub", 44),
    (_ex["BENCH_L0_STD"]["cell"], _ex["BENCH_L0_STD"]["captured"]))
pin("window figure: round 2 prevents 44 (L0) and 0 (L1); rounds 5+10 prevent 24 at both", (44, 0, 24, 24),
    (_ex["BENCH_L0_STD"]["prevented_round2"], _ex["BENCH_L1_STD"]["prevented_round2"],
     _ex["BENCH_L0_STD"]["prevented_rounds_5_10"], _ex["BENCH_L1_STD"]["prevented_rounds_5_10"]))
pin("window figure: two runs lose their window under latency", 2,
    _ex["BENCH_L0_STD"]["with_window"] - _ex["BENCH_L1_STD"]["with_window"])
pin("windows open at round 2 (374 runs, L0) and round 3 (256 runs, L1)", ({"2": 374}, {"3": 256}),
    (_WD["BENCH_L0_STD"]["window_open_round_windowed"], _WD["BENCH_L1_STD"]["window_open_round_windowed"]))
intext(re.escape("every window opens at round 2 at zero latency (374 runs) and at round 3 with one round of "
                 "latency (256 runs)"), "window figure caption pooled counts")
intext(re.escape("Prevention windows of the 44 captured runs (of 50 seeds) in one windowed cell, with 4 committed "
                 "agents, evidence 0.74 and a hub topology"), "window figure caption cell")

# ---- JAAMAS Methods: panel decoding settings are the code's own constants ----
if IS_JAAMAS:
    _s3 = (P / "src" / "s3_panels.py").read_text(encoding="utf-8")
    pin("panel decoding temperature 0.7 / top_p 0.9", True,
        '"temperature": 0.7, "top_p": 0.9' in _s3)
    pin("panel max new tokens 1024", True, "MAX_NEW_TOKENS = 1024" in _s3)
    intext(re.escape("(temperature 0.7, top-$p$ 0.9, at most 1{,}024 new tokens per message)"),
           "JAAMAS Methods decoding settings")

# ---- placeholders ----
# The acmart template's own DOI/ISBN boilerplate matches the placeholder
# pattern; TAPS fills those at camera-ready, so they are not backfill gaps.
BOILERPLATE = {"XXXXXXX"}
ph = sorted(set(re.findall(r"XX[A-Z0-9]{2,}XX", TEX)) - BOILERPLATE)

print(f"PASS {len(PASSES)} pins")
for f in FAILS:
    print("FAIL", f)
print(f"content placeholders remaining ({len(ph)}): {ph}")

# --strict is the packaging gate: an unfilled number is a failure, not a note.
strict = "--strict" in sys.argv
if strict and ph:
    print(f"FAIL strict: {len(ph)} content placeholder(s) still unfilled")
sys.exit(1 if (FAILS or (strict and ph)) else 0)
