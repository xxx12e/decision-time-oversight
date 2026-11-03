r"""Leave-one-population-out robustness for the H4 Kendall taus (review item).

Drops each of the six agent-population configurations in turn, recomputes both
tau pairs on the remaining trials, and reports the range. Populations map to
trial tags: qwen_fast_* / qwen_slow_* / qwen_expand_* / ministral_* /
s3_qwen35_* / s3_gemma4_*.

Usage: python -X utf8 lofo_tau.py
"""
from __future__ import annotations

import json
from pathlib import Path

from s4_zoo import load_trials, score_trials, evaluate, rank_by, kendall

OUT = Path(__file__).resolve().parent.parent / "results" / "s4"

POPS = ("qwen_fast", "qwen_slow", "qwen_expand", "ministral",
        "s3_qwen35", "s3_gemma4")


def taus_for(cache):
    res = evaluate(cache)
    r_post = rank_by(res, "posthoc_auc")
    out = {}
    for key in ("timely_auc", "timely_recall_mfpr"):
        r_t = rank_by(res, key)
        common = [m for m in r_post if m in set(r_t)]
        out[key] = kendall([m for m in r_post if m in common],
                           [m for m in r_t if m in common])
    return out


def main():
    trials = load_trials(["qwen35", "gemma4"])
    cache = score_trials(trials)
    tags = [t["tag"] for t in trials]
    full = taus_for(cache)
    print(f"full ({len(cache)} trials): "
          f"recall tau {full['timely_recall_mfpr']:.3f}, AUC tau {full['timely_auc']:.3f}")

    rows = {}
    for pop in POPS:
        sub = [c for c, tag in zip(cache, tags) if not tag.startswith(pop)]
        t = taus_for(sub)
        rows[pop] = {"n_trials": len(sub),
                     "tau_recall": t["timely_recall_mfpr"],
                     "tau_auc": t["timely_auc"]}
        print(f"drop {pop:<12} ({len(sub):3d} trials): "
              f"recall tau {t['timely_recall_mfpr']:.3f}, AUC tau {t['timely_auc']:.3f}")

    recs = [r["tau_recall"] for r in rows.values()]
    aucs = [r["tau_auc"] for r in rows.values()]
    summary = {"full": {"tau_recall": full["timely_recall_mfpr"],
                        "tau_auc": full["timely_auc"], "n_trials": len(cache)},
               "leave_one_out": rows,
               "range": {"tau_recall": [min(recs), max(recs)],
                         "tau_auc": [min(aucs), max(aucs)]}}
    (OUT / "lofo_tau.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"recall tau range [{min(recs):.3f}, {max(recs):.3f}] | "
          f"AUC tau range [{min(aucs):.3f}, {max(aucs):.3f}] -> lofo_tau.json")


if __name__ == "__main__":
    main()
