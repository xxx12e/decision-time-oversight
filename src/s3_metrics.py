r"""S3 metrics — hindsight vs decision-time scoring of new-family panels, using
the established causal semantics (scores_at_decision_round reimplementation
documented in the prior benchmark's gen_roundwise_metrics.py) and the deployed
anomaly score imported unmodified from that repo.

Input: ..\results\s3\panels_<family>.json
Output: ..\results\s3\metrics_<family>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _paths import belief_bench, add_to_syspath          # noqa: E402

BENCH = add_to_syspath(belief_bench())
from scripts.gate6_e3 import anomaly_for  # noqa: E402  (deployed score, unmodified)

OUT = Path(__file__).resolve().parent.parent / "results" / "s3"
ROUNDS = 5
THETA = 0.5
CONF_ROUND = 2


def _norm_messages(mz):
    return {int(r): {int(i): v for i, v in d.items()} for r, d in mz.items()}


def scores_at_decision_round(mz, t, committed):
    """History = rounds 1..t-1; signature map from round t-1 only (senders with
    non-empty evidence), key (claim_pm1, round(conf, 2))."""
    comm = set(committed)
    hist_rounds = [r for r in range(1, t) if r in mz]
    if not hist_rounds:
        return []
    sig_round = t - 1
    round_sigs: dict = {}
    if sig_round in mz:
        for j, (cl, cf, ev) in mz[sig_round].items():
            if cl is not None and cf is not None and ev:
                k = (1 if cl == "A" else -1, round(float(cf), CONF_ROUND))
                round_sigs[k] = round_sigs.get(k, 0) + 1
    res = []
    all_ids = set()
    for r in hist_rounds:
        all_ids |= set(mz[r].keys())
    for j in sorted(all_ids):
        hist = []
        for r in hist_rounds:
            cl, cf, ev = mz[r].get(j, (None, None, None))
            if cl is None or cf is None:
                continue
            hist.append((r, 1 if cl == "A" else -1, float(cf), bool(ev)))
        if not hist:
            continue
        res.append((anomaly_for(hist, round_sigs), j in comm))
    return res


def roc_auc(pairs):
    pos = [s for s, c in pairs if c]
    neg = [s for s, c in pairs if not c]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def operating_point(pairs, theta=THETA):
    tp = sum(1 for s, c in pairs if c and s >= theta)
    fp = sum(1 for s, c in pairs if not c and s >= theta)
    n_pos = sum(1 for _, c in pairs if c)
    n_neg = len(pairs) - n_pos
    return {"TP": tp, "FP": fp, "TPR": tp / n_pos if n_pos else None,
            "FPR": fp / n_neg if n_neg else None}


def analyze(family: str):
    trials = json.loads((OUT / f"panels_{family}.json").read_text(encoding="utf-8"))
    res = {"family": family, "n_trials": len(trials), "by_adv": {}}
    for adv in ("plain", "jitter"):
        tr = [t for t in trials if t["adv"] == adv]
        if not tr:
            continue
        caps = [t["capture_round"] for t in tr]
        n_cap = sum(1 for c in caps if c is not None)
        roundwise = {}
        for t_dec in list(range(2, ROUNDS + 1)) + ["posthoc"]:
            td = ROUNDS + 1 if t_dec == "posthoc" else t_dec
            pairs = []
            for t in tr:
                pairs += scores_at_decision_round(_norm_messages(t["messages"]), td, t["committed"])
            op = operating_point(pairs)
            roundwise[str(t_dec)] = {"AUC": roc_auc(pairs), **op, "n_pairs": len(pairs)}
        # timely recall: at last timely round t <= C, per captured trial
        timely_tp = timely_pos = 0
        for t in tr:
            c = t["capture_round"]
            if c is None:
                continue
            pairs = scores_at_decision_round(_norm_messages(t["messages"]), c, t["committed"])
            timely_pos += sum(1 for _, is_c in pairs if is_c)
            timely_tp += sum(1 for s, is_c in pairs if is_c and s >= THETA)
        # trial-level cluster bootstrap (panels are the exchangeable unit; agents
        # and rounds within a panel are not iid) for the two headline numbers
        import random as _random
        rng = _random.Random(20260808)
        bs_auc, bs_rec = [], []
        for _ in range(4000):
            samp = [tr[rng.randrange(len(tr))] for _ in tr]
            pairs = []
            ttp = tpos = 0
            for t in samp:
                mz = _norm_messages(t["messages"])
                pairs += scores_at_decision_round(mz, ROUNDS + 1, t["committed"])
                c = t["capture_round"]
                if c is not None:
                    pp = scores_at_decision_round(mz, c, t["committed"])
                    tpos += sum(1 for _, is_c in pp if is_c)
                    ttp += sum(1 for s, is_c in pp if is_c and s >= THETA)
            a = roc_auc(pairs)
            if a is not None:
                bs_auc.append(a)
            if tpos:
                bs_rec.append(ttp / tpos)
        bs_auc.sort()
        bs_rec.sort()

        def _ci(v):
            return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]] if v else None

        res["by_adv"][adv] = {
            "n_trials": len(tr), "n_captured": n_cap, "capture_rounds": caps,
            "parse_fail_total": sum(t["parse_fail"] for t in tr),
            "roundwise": roundwise,
            "timely_recall_at_C": (timely_tp / timely_pos) if timely_pos else None,
            "posthoc_AUC": roundwise["posthoc"]["AUC"],
            "posthoc_AUC_ci95": _ci(bs_auc),
            "timely_recall_ci95": _ci(bs_rec),
        }
    (OUT / f"metrics_{family}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for adv, d in res["by_adv"].items():
        print(f"[{family}|{adv}] captured {d['n_captured']}/{d['n_trials']} "
              f"posthoc AUC={d['posthoc_AUC']} timely_recall@C={d['timely_recall_at_C']} "
              f"pf={d['parse_fail_total']}")
    return res


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "qwen35")
