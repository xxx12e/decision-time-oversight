r"""S4 addendum -- like-for-like metric pairs and held-out threshold calibration.

Two reviewer items, both about comparing like with like.

R1. The headline pair is mixed: hindsight transcript AUC (a separability
    statistic on the full transcript) against timely recall at a fixed
    operating point (a deployment statistic on a causal prefix). This module
    splits it into two homogeneous pairs:

      (a) hindsight AUC  -> timely AUC          [information separability
                                                 vanishing at the deadline]
      (b) hindsight recall@FPR -> timely recall@FPR
                                                [deployment degradation at a
                                                 matched operating point]

    A third confound is removed at the same time: the paper's hindsight
    numbers are pooled over all 948 agent-trial units, while the timely
    numbers live on the 862 units that are still scoreable at the deadline.
    Every like-for-like quantity here is additionally computed on the
    *matched* unit set (the units scoreable under that convention), so the
    only thing that differs between the two sides of a pair is the
    information available, not the population.

    Both pairs are reported under BOTH timing conventions defined in
    timing_conventions.py: STRICT (a check at round t sees 1..t-1) and
    INTERCEPT (a check at round t sees 1..t).

R2. Thresholds and the matched-FPR evaluation share the replay corpus. Every
    recall is therefore also reported cross-fitted: K trial-level folds, the
    operating point calibrated on the out-of-fold negatives only and applied
    to the held-out fold's positives, pooled over folds so every unit is
    scored out-of-sample.

Additionally, the rank disagreement is converted into a decision loss an
interface designer can act on: over trial-level bootstrap resamples, a
dashboard selects the top-1 (and top-3) monitor by hindsight AUC on the
in-bag corpus, and the realized timely recall is read off the OUT-OF-BAG
trials (selection and evaluation never share a trial), against the best
out-of-bag recall available in the zoo.

Reuses the S4 substrate unmodified: MONITORS, prefix_pairs, load_trials,
rank_by and the ranked-list kendall from s4_zoo; the matched-FPR threshold
construction is the same order statistic s4_zoo.evaluate uses.

Usage:  python -X utf8 metric_pair_split.py
Output: ..\results\s4\metric_pair_split.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from s4_zoo import (MONITORS, ROUNDS, kendall, load_trials,  # noqa: E402
                    prefix_pairs, rank_by)

OUT = Path(__file__).resolve().parent.parent / "results" / "s4"
INCLUDE_S3 = ["qwen35", "gemma4"]      # same corpus as results/s4/zoo.json
FPR = 0.0625                           # deployed false-positive rate
SEED = 20260808                        # project seed convention
BOOT_N = 5000                          # matches the S4 bootstrap in s4_zoo.py
K_FOLDS = 5
CONVENTIONS = {"strict": 0, "intercept": 1}
MON = list(MONITORS)
NM = len(MON)


# ------------------------------------------------------------------ units --
def unit_ids(mz, t):
    """Agent ids, in the order prefix_pairs(mz, t, .) emits them.

    Mirrors prefix_pairs' own selection rule (sorted ids, non-empty history);
    every caller asserts alignment against the labels prefix_pairs returns, so
    a divergence is a hard failure rather than a silent mismatch.
    """
    hist_rounds = [r for r in range(1, t) if r in mz]
    if not hist_rounds:
        return []
    ids = set()
    for r in hist_rounds:
        ids |= set(mz[r].keys())
    out = []
    for j in sorted(ids):
        if any(j in mz[r] and mz[r][j][0] is not None and mz[r][j][1] is not None
               for r in hist_rounds):
            out.append(j)
    return out


class Pool:
    """A set of scored agent-trial units: score matrix, labels, trial index."""

    def __init__(self, S, y, keys, n_trials):
        self.S = np.asarray(S, dtype=float).reshape(len(y), NM) if len(y) else \
            np.zeros((0, NM))
        self.y = np.asarray(y, dtype=bool)
        self.keys = list(keys)
        rows = {}
        for r, (i, _j) in enumerate(self.keys):
            rows.setdefault(i, []).append(r)
        self.rows = {i: np.asarray(v, dtype=int) for i, v in rows.items()}
        self.n_trials = n_trials

    def gather(self, trial_ids):
        parts = [self.rows[i] for i in trial_ids if i in self.rows]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=int)

    def subset(self, row_idx):
        row_idx = np.asarray(row_idx, dtype=int)
        return Pool(self.S[row_idx], self.y[row_idx],
                    [self.keys[r] for r in row_idx], self.n_trials)


def build_views(trials):
    """hind = full transcript (t = T+1); strict = t = C; intercept = t = C+1."""
    views = {}
    for view in ("hind", "strict", "intercept"):
        S, y, keys = [], [], []
        for i, tr in enumerate(trials):
            if view == "hind":
                t = ROUNDS + 1
            else:
                c = tr["capture"]
                if c is None:
                    continue
                t = c + CONVENTIONS[view]
            pairs = prefix_pairs(tr["mz"], t, tr["committed"])
            if not pairs:
                continue
            ids = unit_ids(tr["mz"], t)
            assert len(ids) == len(pairs), (view, i, len(ids), len(pairs))
            comm = set(tr["committed"])
            for j, (hist, sigs, is_c) in zip(ids, pairs):
                assert (j in comm) == bool(is_c), (view, i, j)
                S.append([MONITORS[m](hist, sigs) for m in MON])
                y.append(bool(is_c))
                keys.append((i, j))
        views[view] = Pool(S, y, keys, len(trials))
    return views


# ---------------------------------------------------------------- metrics --
def auc_cols(S, y):
    """Mann-Whitney AUC per monitor column, ties averaged."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(NM, np.nan)
    out = np.empty(NM)
    for c in range(NM):
        r = rankdata(S[:, c])
        out[c] = (r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def thr_cols(S, y, fpr=FPR):
    """Per-monitor threshold at the given FPR on this pool's negatives.

    Same order statistic as s4_zoo.evaluate: negatives sorted descending,
    index int(fpr * n_neg) - 1, plus a 1e-9 nudge so the point is strictly
    above the k-th negative.
    """
    neg = S[~y]
    if len(neg) == 0:
        return np.full(NM, 0.5)
    srt = np.sort(neg, axis=0)[::-1]
    return srt[max(0, int(fpr * len(neg)) - 1)] + 1e-9


def recall_cols(S, y, thr):
    pos = S[y]
    if len(pos) == 0:
        return np.full(NM, np.nan)
    return (pos >= thr).mean(axis=0)


def fpr_cols(S, y, thr):
    """Realized false-positive rate of `thr` on this pool's negatives.

    The diagnostic for reviewer item R1: transferring the hindsight operating
    point to the deadline does NOT in general preserve the false-positive rate,
    so 'matched FPR' is matched only on the side it was calibrated on.
    """
    neg = S[~y]
    if len(neg) == 0:
        return np.full(NM, np.nan)
    return (neg >= thr).mean(axis=0)


def _folds(trial_ids, k, rng):
    ids = list(trial_ids)
    rng.shuffle(ids)
    return [ids[f::k] for f in range(k)]


def cv_recall(calib: Pool, evalp: Pool, folds, sample_ids, fpr=FPR):
    """Cross-fitted recall: threshold from out-of-fold negatives of `calib`,
    applied to the held-out fold's positives in `evalp`; pooled over folds.

    `sample_ids` is the trial list under analysis (with multiplicity, so this
    works unchanged inside a trial-level bootstrap replicate); `folds` is a
    partition of the distinct trial ids in it."""
    tp = np.zeros(NM)
    n_pos = 0
    for f in folds:
        held = set(f)
        cal_rows = calib.gather([i for i in sample_ids if i not in held])
        ev_rows = evalp.gather([i for i in sample_ids if i in held])
        if len(cal_rows) == 0 or len(ev_rows) == 0:
            continue
        thr = thr_cols(calib.S[cal_rows], calib.y[cal_rows], fpr)
        pos = evalp.S[ev_rows][evalp.y[ev_rows]]
        if len(pos) == 0:
            continue
        tp += (pos >= thr).sum(axis=0)
        n_pos += len(pos)
    return (tp / n_pos) if n_pos else np.full(NM, np.nan), n_pos


def _clean(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def tau_pair(vals_a, vals_b):
    """Both statistics the paper uses: the ranked-list kendall from s4_zoo
    (reused verbatim) and the tie-corrected tau_b on the raw values."""
    ra = rank_by({m: {"x": vals_a[i]} for i, m in enumerate(MON)}, "x")
    rb = rank_by({m: {"x": vals_b[i]} for i, m in enumerate(MON)}, "x")
    common = [m for m in ra if m in set(rb)]
    t_list = kendall([m for m in ra if m in common], [m for m in rb if m in common])
    a, b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    tb = None
    if ok.sum() >= 3 and len(set(a[ok])) > 1 and len(set(b[ok])) > 1:
        tb = float(kendalltau(a[ok], b[ok]).statistic)
    return t_list, tb


# ------------------------------------------------- per-convention analysis --
def convention_block(views, conv, folds_all):
    hind_full = views["hind"]
    tim = views[conv]
    hkey = {k: r for r, k in enumerate(hind_full.keys)}
    for k in tim.keys:
        assert k in hkey, k
    hind_m = hind_full.subset([hkey[k] for k in tim.keys])

    thr_full = thr_cols(hind_full.S, hind_full.y)      # paper's construction
    thr_hm = thr_cols(hind_m.S, hind_m.y)              # matched-unit hindsight
    thr_t = thr_cols(tim.S, tim.y)                     # same FPR, at the deadline

    a_hf = auc_cols(hind_full.S, hind_full.y)
    a_hm = auc_cols(hind_m.S, hind_m.y)
    a_t = auc_cols(tim.S, tim.y)
    r_hf = recall_cols(hind_full.S, hind_full.y, thr_full)
    r_hm = recall_cols(hind_m.S, hind_m.y, thr_hm)
    r_t_same = recall_cols(tim.S, tim.y, thr_full)     # paper's headline recall
    r_t_own = recall_cols(tim.S, tim.y, thr_t)         # same FPR at the deadline
    f_hf = fpr_cols(hind_full.S, hind_full.y, thr_full)
    f_hm = fpr_cols(hind_m.S, hind_m.y, thr_hm)
    f_t = fpr_cols(tim.S, tim.y, thr_t)
    f_t_same = fpr_cols(tim.S, tim.y, thr_full)        # is the transfer really matched?

    ids_all = list(range(hind_full.n_trials))
    cv_hm, npos_hm = cv_recall(hind_m, hind_m, folds_all, ids_all)
    cv_t_same, npos_t = cv_recall(hind_full, tim, folds_all, ids_all)
    cv_t_own, _ = cv_recall(tim, tim, folds_all, ids_all)
    cv_hf, _ = cv_recall(hind_full, hind_full, folds_all, ids_all)

    mons = {}
    for i, m in enumerate(MON):
        mons[m] = {
            "hindsight_auc_full": _clean(a_hf[i]),
            "hindsight_auc_matched": _clean(a_hm[i]),
            "timely_auc": _clean(a_t[i]),
            "auc_drop_matched": _clean(a_hm[i] - a_t[i]),
            "hindsight_recall_at_fpr_full": _clean(r_hf[i]),
            "hindsight_recall_at_fpr_matched": _clean(r_hm[i]),
            "timely_recall_same_threshold": _clean(r_t_same[i]),
            "timely_recall_at_own_fpr": _clean(r_t_own[i]),
            "recall_drop_matched": _clean(r_hm[i] - r_t_own[i]),
            "thr_hindsight_full": _clean(thr_full[i]),
            "thr_hindsight_matched": _clean(thr_hm[i]),
            "thr_timely": _clean(thr_t[i]),
            "realized_fpr_hindsight_full_at_own_thr": _clean(f_hf[i]),
            "realized_fpr_matched_at_own_thr": _clean(f_hm[i]),
            "realized_fpr_timely_at_own_thr": _clean(f_t[i]),
            "realized_fpr_timely_at_transferred_thr": _clean(f_t_same[i]),
            "heldout": {
                "hindsight_recall_cv_full": _clean(cv_hf[i]),
                "hindsight_recall_cv_matched": _clean(cv_hm[i]),
                "timely_recall_same_threshold_cv": _clean(cv_t_same[i]),
                "timely_recall_at_own_fpr_cv": _clean(cv_t_own[i]),
                "delta_hindsight_recall_matched": _clean(cv_hm[i] - r_hm[i]),
                "delta_timely_recall_same_threshold": _clean(cv_t_same[i] - r_t_same[i]),
                "delta_timely_recall_own_fpr": _clean(cv_t_own[i] - r_t_own[i]),
            },
        }

    pairs = {
        "auc_pair_matched": (a_hm, a_t),
        "auc_pair_full_hindsight": (a_hf, a_t),
        "recall_pair_own_fpr": (r_hm, r_t_own),
        "recall_pair_same_threshold": (r_hf, r_t_same),
        "headline_mixed_pair": (a_hf, r_t_same),
        # controls: metric-type disagreement with the time axis held fixed.
        # If these are already low, the headline pair's low tau is mostly
        # AUC-vs-recall, not hindsight-vs-deadline.
        "metric_type_pair_hindsight_only": (a_hm, r_hm),
        "metric_type_pair_timely_only": (a_t, r_t_own),
        "heldout_recall_pair_own_fpr": (cv_hm, cv_t_own),
        "heldout_recall_pair_same_threshold": (cv_hf, cv_t_same),
        "heldout_headline_mixed_pair": (a_hf, cv_t_same),
    }
    taus = {}
    for name, (va, vb) in pairs.items():
        tl, tb = tau_pair(va, vb)
        taus[name] = {"tau_ranked_list": _clean(tl), "tau_b": _clean(tb)}

    ranks = {
        "by_hindsight_auc_full": rank_by({m: {"x": a_hf[i]} for i, m in enumerate(MON)}, "x"),
        "by_hindsight_auc_matched": rank_by({m: {"x": a_hm[i]} for i, m in enumerate(MON)}, "x"),
        "by_timely_auc": rank_by({m: {"x": a_t[i]} for i, m in enumerate(MON)}, "x"),
        "by_hindsight_recall_matched": rank_by({m: {"x": r_hm[i]} for i, m in enumerate(MON)}, "x"),
        "by_timely_recall_own_fpr": rank_by({m: {"x": r_t_own[i]} for i, m in enumerate(MON)}, "x"),
        "by_timely_recall_same_threshold": rank_by(
            {m: {"x": r_t_same[i]} for i, m in enumerate(MON)}, "x"),
    }
    counts = {
        "n_units_hindsight_full": int(len(hind_full.y)),
        "n_units_matched": int(len(hind_m.y)),
        "n_units_timely": int(len(tim.y)),
        "n_pos_hindsight_full": int(hind_full.y.sum()),
        "n_pos_matched": int(hind_m.y.sum()),
        "n_pos_timely": int(tim.y.sum()),
        "n_pos_heldout_matched": int(npos_hm),
        "n_pos_heldout_timely": int(npos_t),
    }
    return {"monitors": mons, "kendall": taus, "ranks": ranks, "counts": counts}, \
        (hind_full, hind_m, tim)


# ----------------------------------------------------------- bootstrap ----
def _replicate_metrics(hind_full, hind_m, tim, trial_ids, folds=None):
    """All like-for-like quantities on one (possibly resampled) trial set."""
    rh = hind_full.gather(trial_ids)
    rm = hind_m.gather(trial_ids)
    rt = tim.gather(trial_ids)
    if len(rt) == 0 or len(rm) == 0:
        return None
    Sh, yh = hind_full.S[rh], hind_full.y[rh]
    Sm, ym = hind_m.S[rm], hind_m.y[rm]
    St, yt = tim.S[rt], tim.y[rt]
    if ym.sum() == 0 or yt.sum() == 0 or (~yt).sum() == 0 or (~yh).sum() == 0:
        return None
    th_full = thr_cols(Sh, yh)
    th_m = thr_cols(Sm, ym)
    th_t = thr_cols(St, yt)
    out = {
        "a_hf": auc_cols(Sh, yh), "a_hm": auc_cols(Sm, ym), "a_t": auc_cols(St, yt),
        "r_hf": recall_cols(Sh, yh, th_full), "r_hm": recall_cols(Sm, ym, th_m),
        "r_t_same": recall_cols(St, yt, th_full), "r_t_own": recall_cols(St, yt, th_t),
        "thr_full": th_full,
    }
    if folds is not None:
        out["cv_hm"] = cv_recall(hind_m, hind_m, folds, trial_ids)[0]
        out["cv_hf"] = cv_recall(hind_full, hind_full, folds, trial_ids)[0]
        out["cv_t_same"] = cv_recall(hind_full, tim, folds, trial_ids)[0]
        out["cv_t_own"] = cv_recall(tim, tim, folds, trial_ids)[0]
    return out


def bootstrap_block(hind_full, hind_m, tim, n_trials, boot_n=BOOT_N):
    """One pass produces (a) tau CIs for every pair and (b) the out-of-bag
    decision-loss ledger. Thresholds are recalibrated inside every replicate,
    and out-of-bag evaluation never shares a trial with selection."""
    rng = random.Random(SEED)
    pair_defs = (
        ("auc_pair_matched", "a_hm", "a_t"),
        ("auc_pair_full_hindsight", "a_hf", "a_t"),
        ("recall_pair_own_fpr", "r_hm", "r_t_own"),
        ("recall_pair_same_threshold", "r_hf", "r_t_same"),
        ("headline_mixed_pair", "a_hf", "r_t_same"),
        ("heldout_recall_pair_own_fpr", "cv_hm", "cv_t_own"),
        ("heldout_recall_pair_same_threshold", "cv_hf", "cv_t_same"),
        ("heldout_headline_mixed_pair", "a_hf", "cv_t_same"),
        ("metric_type_pair_hindsight_only", "a_hm", "r_hm"),
        ("metric_type_pair_timely_only", "a_t", "r_t_own"),
    )
    tau_draws = {p: {"tau_ranked_list": [], "tau_b": []} for p, _, _ in pair_defs}
    keys = ["hindsight_auc", "hindsight_recall", "timely_auc", "timely_recall"]
    dl = {k: {"top1_in": [], "top3_in": [], "best_in": [],
              "top1_oob": [], "top3_oob": [], "best_oob": []} for k in keys}
    n_ok = 0
    for _ in range(boot_n):
        samp = [rng.randrange(n_trials) for _ in range(n_trials)]
        oob = sorted(set(range(n_trials)) - set(samp))
        folds = _folds(sorted(set(samp)), K_FOLDS, rng)
        mi = _replicate_metrics(hind_full, hind_m, tim, samp, folds)
        if mi is None:
            continue
        for name, ka, kb in pair_defs:
            tl, tb = tau_pair(mi[ka], mi[kb])
            tau_draws[name]["tau_ranked_list"].append(tl)
            if tb is not None:
                tau_draws[name]["tau_b"].append(tb)
        # out-of-bag timely recall at the in-bag-calibrated operating point
        rt_o = tim.gather(oob)
        if len(rt_o) == 0:
            continue
        So, yo = tim.S[rt_o], tim.y[rt_o]
        if yo.sum() == 0:
            continue
        rec_oob = recall_cols(So, yo, mi["thr_full"])
        rec_in = mi["r_t_same"]
        sel = {"hindsight_auc": mi["a_hf"], "hindsight_recall": mi["r_hf"],
               "timely_auc": mi["a_t"], "timely_recall": mi["r_t_same"]}
        best_in, best_oob = float(np.nanmax(rec_in)), float(np.nanmax(rec_oob))
        for k, v in sel.items():
            order = np.argsort(-np.nan_to_num(v, nan=-np.inf), kind="stable")
            t1, t3 = order[0], order[:3]
            dl[k]["top1_in"].append(float(rec_in[t1]))
            dl[k]["top3_in"].append(float(np.nanmax(rec_in[t3])))
            dl[k]["best_in"].append(best_in)
            dl[k]["top1_oob"].append(float(rec_oob[t1]))
            dl[k]["top3_oob"].append(float(np.nanmax(rec_oob[t3])))
            dl[k]["best_oob"].append(best_oob)
        n_ok += 1

    def _ci(v):
        v = sorted(float(x) for x in v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]] if v else None

    taus = {p: {s: {"ci95": _ci(lst), "n_draws": len(lst)} for s, lst in dd.items()}
            for p, dd in tau_draws.items()}

    def _summ(real, best):
        real, best = np.asarray(real), np.asarray(best)
        reg = best - real
        return {"mean_realized_timely_recall": float(real.mean()),
                "mean_best_available_timely_recall": float(best.mean()),
                "mean_regret": float(reg.mean()),
                "median_regret": float(np.median(reg)),
                "regret_ci95": _ci(list(reg)),
                "p_selection_optimal": float((reg <= 1e-12).mean()),
                "mean_recall_ratio": (float((real[best > 0] / best[best > 0]).mean())
                                      if (best > 0).any() else None)}

    loss = {}
    for k in keys:
        loss[k] = {
            "out_of_bag": {"top1": _summ(dl[k]["top1_oob"], dl[k]["best_oob"]),
                           "top3": _summ(dl[k]["top3_oob"], dl[k]["best_oob"])},
            "in_bag": {"top1": _summ(dl[k]["top1_in"], dl[k]["best_in"]),
                       "top3": _summ(dl[k]["top3_in"], dl[k]["best_in"])}}
    return taus, loss, n_ok


# ------------------------------------------------------------------ main --
def main():
    trials = load_trials(INCLUDE_S3)
    views = build_views(trials)
    rng = random.Random(SEED)
    folds_all = _folds(range(len(trials)), K_FOLDS, rng)

    res = {"meta": {
        "n_trials": len(trials), "included_s3": INCLUDE_S3, "fpr": FPR,
        "seed": SEED, "boot_n": BOOT_N, "k_folds": K_FOLDS,
        "monitors": MON,
        "definitions": {
            "hindsight": "score on the full transcript, t = T+1 (5 rounds visible)",
            "strict": "decision at t = C sees rounds 1..C-1 (one round of latency)",
            "intercept": "decision at t = C sees rounds 1..C (zero latency)",
            "matched": ("hindsight quantities restricted to the agent-trial units "
                        "that are still scoreable at the deadline under that "
                        "convention, so the two sides of a pair differ only in the "
                        "information available"),
            "timely_recall_same_threshold": ("threshold fixed at FPR=0.0625 on the "
                                             "hindsight negatives and transferred to "
                                             "the deadline -- the paper's headline"),
            "timely_recall_at_own_fpr": ("threshold re-set to FPR=0.0625 on the "
                                         "deadline negatives -- literally the same "
                                         "false-positive rate on both sides"),
            "heldout": (f"{K_FOLDS}-fold cross-fitting over trials: operating point "
                        "from out-of-fold negatives, recall on the held-out fold, "
                        "pooled -- no unit is scored at a threshold its own trial "
                        "helped set"),
            "decision_loss": ("dashboard selects on in-bag trials, realized timely "
                              "recall read off the out-of-bag trials at the in-bag "
                              "operating point"),
        }}, "conventions": {}}

    for conv in ("strict", "intercept"):
        block, (hf, hm, tm) = convention_block(views, conv, folds_all)
        taus_ci, loss, n_ok = bootstrap_block(hf, hm, tm, len(trials))
        for p, d in taus_ci.items():
            if p in block["kendall"]:
                block["kendall"][p].update(
                    {"ci95_ranked_list": d["tau_ranked_list"]["ci95"],
                     "ci95_tau_b": d["tau_b"]["ci95"]})
        block["decision_loss"] = {"select_by": loss, "n_valid_replicates": n_ok}
        res["conventions"][conv] = block

    # compact, quotable summary -----------------------------------------
    res["summary"] = {}
    for conv in ("strict", "intercept"):
        b = res["conventions"][conv]
        mons, k = b["monitors"], b["kendall"]

        def _best(field):
            m = max(MON, key=lambda x: (mons[x][field] if mons[x][field] is not None
                                        else -np.inf))
            return {"monitor": m, "value": mons[m][field]}

        d_same = [mons[m]["heldout"]["delta_timely_recall_same_threshold"] for m in MON]
        d_own = [mons[m]["heldout"]["delta_timely_recall_own_fpr"] for m in MON]
        top_h = b["ranks"]["by_hindsight_auc_matched"][0]
        dlh = b["decision_loss"]["select_by"]["hindsight_auc"]["out_of_bag"]
        res["summary"][conv] = {
            "tau_ladder": {
                "headline_mixed_auc_vs_recall": k["headline_mixed_pair"]["tau_ranked_list"],
                "control_metric_type_only_both_in_hindsight":
                    k["metric_type_pair_hindsight_only"]["tau_ranked_list"],
                "like_for_like_auc_pair": k["auc_pair_matched"]["tau_ranked_list"],
                "like_for_like_recall_pair_same_fpr":
                    k["recall_pair_own_fpr"]["tau_ranked_list"],
                "like_for_like_recall_pair_transferred_threshold":
                    k["recall_pair_same_threshold"]["tau_ranked_list"],
                "heldout_recall_pair_same_fpr":
                    k["heldout_recall_pair_own_fpr"]["tau_ranked_list"],
            },
            "best": {"timely_auc": _best("timely_auc"),
                     "timely_recall_at_own_fpr": _best("timely_recall_at_own_fpr"),
                     "timely_recall_same_threshold": _best("timely_recall_same_threshold"),
                     "hindsight_auc_matched": _best("hindsight_auc_matched")},
            "level_collapse_of_hindsight_leader": {
                "monitor": top_h,
                "hindsight_auc_matched": mons[top_h]["hindsight_auc_matched"],
                "timely_auc": mons[top_h]["timely_auc"],
                "hindsight_recall_at_fpr_matched": mons[top_h]["hindsight_recall_at_fpr_matched"],
                "timely_recall_at_own_fpr": mons[top_h]["timely_recall_at_own_fpr"]},
            "calibration_leakage": {
                "max_abs_delta_timely_recall_same_threshold": float(np.max(np.abs(d_same))),
                "mean_signed_delta_timely_recall_same_threshold": float(np.mean(d_same)),
                "max_abs_delta_timely_recall_own_fpr": float(np.max(np.abs(d_own))),
                "mean_signed_delta_timely_recall_own_fpr": float(np.mean(d_own)),
                "zoo_best_timely_recall_in_sample":
                    max(mons[m]["timely_recall_same_threshold"] for m in MON),
                "zoo_best_timely_recall_heldout":
                    max(mons[m]["heldout"]["timely_recall_same_threshold_cv"] for m in MON)},
            "operating_point_is_not_matched_after_transfer": {
                m: mons[m]["realized_fpr_timely_at_transferred_thr"] for m in MON},
            "decision_loss_hindsight_auc_dashboard_oob": {
                "top1_realized": dlh["top1"]["mean_realized_timely_recall"],
                "top1_best_available": dlh["top1"]["mean_best_available_timely_recall"],
                "top1_regret": dlh["top1"]["mean_regret"],
                "top1_regret_ci95": dlh["top1"]["regret_ci95"],
                "top1_p_optimal": dlh["top1"]["p_selection_optimal"],
                "top3_realized": dlh["top3"]["mean_realized_timely_recall"],
                "top3_regret": dlh["top3"]["mean_regret"]},
        }

    # faithfulness check against the existing results/s4/zoo.json
    zoo = json.loads((OUT / "zoo.json").read_text(encoding="utf-8"))
    chk = {"max_abs_diff_posthoc_auc": 0.0, "max_abs_diff_timely_auc": 0.0,
           "max_abs_diff_timely_recall": 0.0}
    for m in MON:
        a = res["conventions"]["strict"]["monitors"][m]
        z = zoo["monitors"][m]
        chk["max_abs_diff_posthoc_auc"] = max(
            chk["max_abs_diff_posthoc_auc"], abs(a["hindsight_auc_full"] - z["posthoc_auc"]))
        chk["max_abs_diff_timely_auc"] = max(
            chk["max_abs_diff_timely_auc"], abs(a["timely_auc"] - z["timely_auc"]))
        chk["max_abs_diff_timely_recall"] = max(
            chk["max_abs_diff_timely_recall"],
            abs(a["timely_recall_same_threshold"] - z["timely_recall_mfpr"]))
    chk["zoo_tau_timely_recall"] = zoo["kendall"]["timely_recall_mfpr"]["tau"]
    chk["here_tau_headline_mixed_strict"] = \
        res["conventions"]["strict"]["kendall"]["headline_mixed_pair"]["tau_ranked_list"]
    res["reproduction_check"] = chk

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metric_pair_split.json").write_text(json.dumps(res, indent=1),
                                                encoding="utf-8")

    for conv in ("strict", "intercept"):
        b = res["conventions"][conv]
        k = b["kendall"]
        print(f"\n=== {conv} ({b['counts']['n_units_matched']} matched units, "
              f"{b['counts']['n_pos_timely']} committed) ===")
        print(f"  tau AUC pair (matched)      {k['auc_pair_matched']['tau_ranked_list']:.3f} "
              f"{k['auc_pair_matched']['ci95_ranked_list']}")
        print(f"  tau recall pair (same FPR)  {k['recall_pair_own_fpr']['tau_ranked_list']:.3f} "
              f"{k['recall_pair_own_fpr']['ci95_ranked_list']}")
        print(f"  tau headline (mixed)        {k['headline_mixed_pair']['tau_ranked_list']:.3f} "
              f"{k['headline_mixed_pair']['ci95_ranked_list']}")
        print(f"  tau heldout recall pair     "
              f"{k['heldout_recall_pair_own_fpr']['tau_ranked_list']:.3f}")
        best_a = max(MON, key=lambda m: b["monitors"][m]["timely_auc"])
        best_r = max(MON, key=lambda m: b["monitors"][m]["timely_recall_at_own_fpr"])
        print(f"  best timely AUC   {best_a} {b['monitors'][best_a]['timely_auc']:.3f} "
              f"(its hindsight matched AUC {b['monitors'][best_a]['hindsight_auc_matched']:.3f})")
        print(f"  best timely recall@ownFPR {best_r} "
              f"{b['monitors'][best_r]['timely_recall_at_own_fpr']:.3f} "
              f"(hindsight recall {b['monitors'][best_r]['hindsight_recall_at_fpr_matched']:.3f})")
        d = b["decision_loss"]["select_by"]["hindsight_auc"]["out_of_bag"]
        print(f"  OOB decision loss  top1 realized {d['top1']['mean_realized_timely_recall']:.3f} "
              f"vs best {d['top1']['mean_best_available_timely_recall']:.3f} "
              f"regret {d['top1']['mean_regret']:.3f} {d['top1']['regret_ci95']}")
        print(f"                     top3 realized {d['top3']['mean_realized_timely_recall']:.3f} "
              f"regret {d['top3']['mean_regret']:.3f}")
    print("\nreproduction check vs zoo.json:", res["reproduction_check"])


if __name__ == "__main__":
    main()
