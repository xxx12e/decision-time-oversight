r"""replay_core.py -- the measured quantities behind the reference monitoring view.

Pure computation. No Streamlit, no plotting, no network, no absolute paths:
everything is read from the released results tree, located relative to this
file, so the interactive view and the self-check consume identical numbers.

Every quantity returned here is an ORACLE-PLACEMENT bound: the detection
instrument is held fixed at the deployed detector, and the oracle optimizes
only where and when it fires. Such a bound caps every checking schedule of
the deployed instrument; it does not cap a reader whose judgment beats the
instrument -- that axis is measured by the Study-4 monitor zoo, not bounded.
Nothing here estimates, models, or claims anything about a human operator.

Source of each element:
  (1) regime banner        -- results/s1/stats.json  (60 cells x 50 seeds)
  (2) placement strip      -- results/s2/stats.json  (front_collapse block),
                              independently recomputable from results/s2/trials.jsonl
  (3) decision-time board  -- results/s4/zoo.json    (13 monitors, 90 trials)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- constants

SEEDS_PER_CELL = 50          # Study 1: 50 seeds per committed cell
S1_BUDGETS = (1, 2, 3)       # Study 1 attention budgets
S1_ROUNDS = 15               # Study 1 horizon T

S2_LENGTH = 160              # Study 2 stream length
S2_ONSET_LO = 40             # registered onset prior support, inclusive
S2_ONSET_HI = 120
S2_BUDGETS = (1, 2, 4, 8, 16)
S2_REPLAY_BUDGETS = (1, 2, 4)   # budgets the paper's fallback claim covers

# A posterior counts as flat when no support point carries more than this
# multiple of the least-weighted support point.
FLATNESS_RATIO = 1.05

MINORITY_SIZES = (1, 2, 3, 4, 6)
EVIDENCE_LEVELS = ("0.58", "0.66", "0.74")
TOPOLOGIES = ("hub", "random", "small_world", "clustered")

# The three honest messages, plus the fourth state the data actually contains.
# The ceiling-bound message is a first-class output, not an error state.
REGIME_MESSAGE = {
    "placement_critical": (
        "Place carefully",
        "Placement buys more than budget here. The best-placed single check "
        "beats a doubled uniform budget on timely-catch rate.",
    ),
    "budget_saturated": (
        "Any coverage works",
        "Uniform checking at two or more checks already matches the oracle. "
        "Spend the minimum and stop tuning placement.",
    ),
    "ceiling_bound": (
        "Prevention value near zero here",
        "Even exhaustive placement prevents at most ~4% of captures in "
        "ceiling-bound cells (preregistered threshold: oracle prevention "
        "<= 5%). A check is defensible only if this failure is very costly; "
        "weigh those odds against your failure cost, and invest upstream.",
    ),
    "benign": (
        "No risk-bearing fault measured",
        "Baseline failure rate in this cell is below the risk threshold, so "
        "there is nothing for a check to prevent. This is outside the "
        "three-message space, not a fourth regime.",
    ),
    "intermediate": (
        "Unclassified",
        "This cell met none of the three regime criteria. No honest message "
        "is available for it.",
    ),
}

REGIME_ORDER = ("placement_critical", "ceiling_bound", "budget_saturated",
                "benign", "intermediate")

# Study 4 records post-hoc AUC, timely AUC and matched-FPR timely recall per
# monitor. Penalized delay is NOT among them, so the scoreboard reports the
# delay column as unavailable rather than inventing one.
DELAY_UNAVAILABLE = (
    "Penalized delay is not reported per monitor in results/s4/zoo.json; "
    "the released Study-4 replay scores each monitor by post-hoc AUC, timely "
    "AUC and matched-FPR timely recall only. This column is left empty rather "
    "than filled with a proxy."
)

CONSTANCY_MONITORS = ("conf_constancy", "claim_constancy")


# ---------------------------------------------------------------- loading

def find_results_dir(start: Path | None = None) -> Path:
    """Locate the released results tree relative to this file.

    Works both for the in-repo layout (artifact/ next to results/) and for a
    packaged copy that carries results/ inside the artifact directory.
    """
    here = (start or Path(__file__).resolve().parent)
    for base in (here, *list(here.parents)[:3]):
        cand = base / "results"
        if (cand / "s1" / "stats.json").is_file():
            return cand
    raise FileNotFoundError(
        "results/ not found next to the artifact directory or its parents; "
        "expected results/s1/stats.json")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Bundle:
    root: Path
    s1: dict
    s2: dict
    zoo: dict
    s3: dict | None

    @property
    def s2_trials_path(self) -> Path:
        return self.root / "s2" / "trials.jsonl"


def load_bundle(results: Path | None = None) -> Bundle:
    root = results or find_results_dir()
    s3_path = root / "s3" / "metrics_qwen35.json"
    return Bundle(
        root=root,
        s1=_read_json(root / "s1" / "stats.json"),
        s2=_read_json(root / "s2" / "stats.json"),
        zoo=_read_json(root / "s4" / "zoo.json"),
        s3=_read_json(s3_path) if s3_path.is_file() else None,
    )


def relative_source(root: Path, *parts: str) -> str:
    """Display path for a results file, anchored at results/ (never absolute)."""
    return "/".join(("results",) + parts)


# ------------------------------------------------- element 1: regime banner

def cell_key(minority: int, evidence: str, topology: str) -> str:
    return f"{minority}|{evidence}|{topology}"


def parse_cell_key(key: str) -> tuple[int, str, str]:
    cc, ev, topo = key.split("|")
    return int(cc), ev, topo


def oracle_timely_reference(cell: dict) -> float:
    """Onset-oracle timely-catch rate (reference policy, not a catch-count ceiling).

    Maxes over oracle@B{1,2,3} only. It is NOT a ceiling on catch count:
    uniform@B3 registers more timely catches in seven of the 60 cells, all of
    them cells where those extra catches prevent nothing (see
    uniform_b3_overtakes). The paper states exactly this.
    """
    return max(cell["policies"].get(f"oracle@B{B}", {}).get("timely", 0.0)
               for B in S1_BUDGETS)


def uniform_b3_overtakes(s1: dict) -> list[dict]:
    """Cells where uniform@B3 registers MORE timely catches than the onset
    oracle at any Study-1 budget -- the reason the oracle rate is labelled a
    reference policy, not a catch-count ceiling.

    In the released results these are seven cells, every one with an
    oracle-preventable failure rate of 0.0 (one ceiling-bound, six benign):
    the extra catches prevent nothing.
    """
    out = []
    for key, cell in sorted(s1["cells"].items()):
        ref = oracle_timely_reference(cell)
        u3 = cell["policies"].get("uniform@B3", {}).get("timely", 0.0)
        if u3 > ref:
            out.append({
                "key": key,
                "regime": cell["regime"],
                "oracle_reference": ref,
                "uniform_b3": u3,
                "oracle_preventable": cell["oracle_any"],
            })
    return out


def oracle_preventable(cell: dict) -> float:
    """Failure rate the oracle can actually prevent in this cell."""
    return cell["oracle_any"]


def routing_ledger(s1: dict) -> dict:
    """Replay: what is kept and what is given up by spending zero checks in
    ceiling-bound cells, over every Study-1 cell x 50 seeds.

    Two ledgers, deliberately kept apart:
      catches     -- oracle-achievable timely catches (detection in time)
      preventable -- oracle-preventable failures (detection that changes the
                     outcome).  The gap between them is the whole point: most
                     of the catches given up prevent nothing.
    """
    per_regime: dict[str, dict[str, float]] = {}
    total_catches = 0.0
    total_preventable = 0.0
    for key, cell in s1["cells"].items():
        regime = cell["regime"]
        catches = oracle_timely_reference(cell) * SEEDS_PER_CELL
        prevent = oracle_preventable(cell) * SEEDS_PER_CELL
        row = per_regime.setdefault(regime, {"cells": 0, "catches": 0.0,
                                             "preventable": 0.0})
        row["cells"] += 1
        row["catches"] += catches
        row["preventable"] += prevent
        total_catches += catches
        total_preventable += prevent

    ceil = per_regime.get("ceiling_bound", {"catches": 0.0, "preventable": 0.0})
    retained_catches = total_catches - ceil["catches"]
    forfeited_preventable = ceil["preventable"]
    # Shares are None (not a crash) when the results tree carries no cells or
    # no preventable failures; callers must render that state visibly.
    return {
        "per_regime": per_regime,
        "n_cells": len(s1["cells"]),
        "seeds_per_cell": SEEDS_PER_CELL,
        "total_catches": total_catches,
        "retained_catches": retained_catches,
        "retained_share": (retained_catches / total_catches
                           if total_catches > 0.0 else None),
        "total_preventable": total_preventable,
        "forfeited_preventable": forfeited_preventable,
        "forfeited_share": (forfeited_preventable / total_preventable
                            if total_preventable > 0.0 else None),
    }


def regime_counts(s1: dict) -> dict[str, int]:
    return dict(s1["h3"]["counts"])


def regime_shares(s1: dict) -> dict[str, float]:
    return dict(s1["h3"]["shares"])


def lookup_cell(s1: dict, key: str) -> dict | None:
    cell = s1["cells"].get(key)
    if cell is None:
        return None
    head, body = REGIME_MESSAGE[cell["regime"]]
    return {
        "key": key,
        "regime": cell["regime"],
        "headline": head,
        "detail": body,
        "median_capture": cell["median_capture"],
        "baseline_failure": cell["fc_baseline"],
        "oracle_timely_reference": oracle_timely_reference(cell),
        "oracle_preventable": oracle_preventable(cell),
        "best_single_check": cell["best_B1"],
        "policies": cell["policies"],
    }


def cells_at_fault_speed(s1: dict, rounds: float | None,
                         window: float = 1.0) -> dict:
    """Map a fault-speed estimate onto the regime map.

    rounds is the operator-side estimate of how many rounds elapse before the
    fault becomes irreversible.  None selects the cells whose median run never
    reaches capture.  Returns every measured cell within +/- window of the
    estimate, the regime distribution over them, and the modal message.

    Fault speed alone does not determine the regime -- the data contains cells
    that capture at the same median round and land in different regimes -- so
    the distribution is returned, not just a winner.
    """
    matched = []
    for key, cell in s1["cells"].items():
        mc = cell["median_capture"]
        if rounds is None:
            hit = mc is None
        else:
            hit = mc is not None and abs(float(mc) - rounds) <= window
        if hit:
            matched.append((key, cell))

    dist: dict[str, int] = {}
    for _, cell in matched:
        dist[cell["regime"]] = dist.get(cell["regime"], 0) + 1
    modal = max(dist.items(), key=lambda kv: (kv[1], -REGIME_ORDER.index(kv[0])))[0] \
        if dist else None
    return {
        "estimate": rounds,
        "window": window,
        "matched": [
            {"key": k, "regime": c["regime"], "median_capture": c["median_capture"],
             "oracle_timely_reference": oracle_timely_reference(c),
             "oracle_preventable": oracle_preventable(c)}
            for k, c in sorted(matched, key=lambda kc: kc[0])
        ],
        "distribution": dist,
        "modal_regime": modal,
        "unanimous": len(dist) == 1,
        "headline": REGIME_MESSAGE[modal][0] if modal else "No measured cell",
        "detail": (REGIME_MESSAGE[modal][1] if modal else
                   "No Study-1 cell captures at this speed, so the regime map "
                   "says nothing here. Do not read a message off an empty cell."),
    }


# ----------------------------------------------- element 2: placement strip

def s2_uniform(B: int) -> list[int]:
    return sorted({round(i * S2_LENGTH / (B + 1)) for i in range(1, B + 1)})


def s2_geometric(B: int) -> list[int]:
    """Audit-early: geometric spacing, the AI-control transposition."""
    return sorted({min(S2_LENGTH - 1, 2 ** (i + 1)) for i in range(B)})


def s2_front(B: int) -> list[int]:
    return sorted(range(10, 10 + B))


def s2_back(B: int) -> list[int]:
    return sorted(range(S2_LENGTH - B, S2_LENGTH))


def s2_oracle(B: int, tau: int) -> list[int]:
    return sorted({min(S2_LENGTH - 1, tau + i) for i in range(B)})


S2_SCHEDULES = {
    "uniform": s2_uniform,
    "geometric": s2_geometric,
    "front": s2_front,
    "back": s2_back,
}

POLICY_LABEL = {
    "uniform": "uniform",
    "geometric": "audit-early (geometric)",
    "front": "front-loaded",
    "back": "back-loaded",
    "oracle": "onset oracle",
    "posterior": "posterior-targeted",
}


def flat_posterior() -> list[float]:
    """The registered prior: onset uniform on [40, 120] of a 160-step stream."""
    w = [0.0] * S2_LENGTH
    for t in range(S2_ONSET_LO, S2_ONSET_HI + 1):
        w[t] = 1.0
    total = sum(w)
    return [x / total for x in w]


def informative_posterior(center: float, spread: float) -> list[float]:
    """Onset posterior concentrated by the observables, truncated to support.

    spread is the standard deviation in stream steps. Large spread returns a
    posterior that is flat to within tolerance -- which is the point: the
    fallback triggers on the shape of the posterior, not on a mode switch.
    """
    spread = max(float(spread), 1e-6)
    w = [0.0] * S2_LENGTH
    for t in range(S2_ONSET_LO, S2_ONSET_HI + 1):
        z = (t - center) / spread
        w[t] = math.exp(-0.5 * z * z)
    total = sum(w)
    if total <= 0.0:
        return flat_posterior()
    return [x / total for x in w]


def posterior_is_flat(post: list[float], ratio: float = FLATNESS_RATIO) -> bool:
    """True when the observables add nothing to the registered prior."""
    sup = [p for p in post if p > 0.0]
    if not sup:
        return True
    lo, hi = min(sup), max(sup)
    if lo <= 0.0:
        return False
    return (hi / lo) <= ratio


def posterior_targeted_schedule(post: list[float], B: int) -> list[int]:
    """Place B checks at equal posterior-mass quantiles.

    NOT a measured policy: Study 2 replayed uniform, random, front, back,
    audit-early and the onset oracle. Anything this returns for a non-flat
    posterior is bracketed by the measured uniform and oracle rows.
    """
    cum, acc = [], 0.0
    for p in post:
        acc += p
        cum.append(acc)
    out = []
    for i in range(1, B + 1):
        q = i / (B + 1.0)
        step = next((t for t, c in enumerate(cum) if c >= q), S2_LENGTH - 1)
        out.append(step)
    return sorted(set(out))


def choose_placement(post: list[float], B: int) -> dict:
    """The registered fallback rule, in one place.

    Flat posterior -> uniform placement (measured). Informative posterior ->
    posterior-targeted placement (not measured; reported as such).
    """
    flat = posterior_is_flat(post)
    policy = "uniform" if flat else "posterior"
    schedule = s2_uniform(B) if flat else posterior_targeted_schedule(post, B)
    return {
        "posterior_flat": flat,
        "policy": policy,
        "policy_label": POLICY_LABEL[policy],
        "schedule": schedule,
        "measured": flat,
        "rationale": (
            "Posterior is flat: the observables are uninformative, so the "
            "strip hedges to uniform placement -- the measured default."
            if flat else
            "Posterior is informative: checks are placed at equal "
            "posterior-mass quantiles. This placement is not one of the six "
            "policies Study 2 replayed, so no measured miss rate is shown "
            "for it; the measured uniform and onset-oracle rows bracket it."
        ),
    }


def mass_after_last_check(post: list[float], schedule: list[int]) -> float:
    """Posterior mass at steps strictly after the last check.

    Geometry of the displayed posterior, not a replayed measurement: onset
    landing there cannot be seen by any of these checks at all.
    """
    if not schedule:
        return sum(post)
    last = max(schedule)
    return sum(post[last + 1:])


def front_collapse(s2: dict) -> dict:
    """Measured miss rates, pooled over diagnosable well-specified cells."""
    return s2["front_collapse"]


def fallback_replay(s2: dict, budgets: tuple[int, ...] = S2_REPLAY_BUDGETS) -> dict:
    """Element 2's replay claim: what switching to uniform on a flat posterior buys.

    Returned per budget, straight out of the front_collapse block. Budgets
    with no front_collapse entry are simply absent from the returned dict --
    callers render an explicit "not in results" state for them instead of
    crashing on a missing key.
    """
    fc = front_collapse(s2)
    rows = {}
    for B in budgets:
        cell = fc.get(f"B{B}")
        if cell is None:
            continue
        rows[B] = {
            "audit_early_miss": cell["geometric_miss"],
            "front_miss": cell["front_miss"],
            "uniform_miss": cell["uniform_miss"],
            "oracle_miss": cell["oracle_miss"],
        }
    return rows


def recompute_front_collapse(trials_path: Path,
                             budgets: tuple[int, ...] = S2_REPLAY_BUDGETS) -> dict:
    """Independent recomputation of the miss rates from the Study-2 trial log.

    Pools well-specified trials from every diagnosable cell (separability
    rho > 0), exactly the population the released aggregate uses, but reads
    per-trial outcomes rather than the aggregate -- so a corrupted or stale
    stats.json cannot make the self-check agree with itself.
    """
    diagnosable: set[str] = set()
    rows: list[dict] = []
    with trials_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("rho", 0.0) > 0.0:
                diagnosable.add(t["cell"])
            rows.append(t)

    acc: dict[tuple[str, int], list[int]] = {}
    delay: dict[tuple[str, int], list[float]] = {}
    for t in rows:
        if t["condition"] != "well_specified" or t["cell"] not in diagnosable:
            continue
        if t["budget"] not in budgets:
            continue
        k = (t["policy"], t["budget"])
        acc.setdefault(k, []).append(1 if t["miss"] else 0)
        delay.setdefault(k, []).append(float(t["penalized_delay"]))

    out: dict[int, dict[str, float]] = {}
    for B in budgets:
        entry: dict[str, float] = {}
        for pol in ("geometric", "front", "uniform", "oracle", "back", "random"):
            m = acc.get((pol, B))
            if not m:
                continue
            entry[f"{pol}_miss"] = sum(m) / len(m)
            d = delay[(pol, B)]
            entry[f"{pol}_pen"] = sum(d) / len(d)
            entry[f"{pol}_n"] = float(len(m))
        entry["audit_early_miss"] = entry.get("geometric_miss", float("nan"))
        out[B] = entry
    out["n_diagnosable_cells"] = len(diagnosable)  # type: ignore[index]
    return out


def s2_cell_keys(s2: dict) -> list[str]:
    return sorted(s2["cells"].keys())


def s2_cell_policy_table(s2: dict, cell_key_: str, B: int) -> list[dict]:
    """Per-cell measured miss / penalized delay / false alarm at one budget."""
    entry = s2["cells"][cell_key_]
    rows = []
    for pol in ("uniform", "geometric", "front", "back", "random", "oracle"):
        rec = entry.get(f"{pol}@B{B}")
        if rec is None:
            continue
        rows.append({
            "policy": pol,
            "label": POLICY_LABEL.get(pol, pol),
            "miss": rec["miss"],
            "mean_pen": rec["mean_pen"],
            "fa": rec["fa"],
        })
    return rows


def cell_is_undetectable(s2: dict, cell_key_: str) -> bool:
    """True when even the onset oracle misses everything at every budget.

    The Study-2 analogue of a ceiling-bound cell: placement is irrelevant
    because the two rules are not separable by the probe battery.
    """
    entry = s2["cells"][cell_key_]
    vals = [entry[f"oracle@B{B}"]["miss"] for B in S2_BUDGETS
            if f"oracle@B{B}" in entry]
    return bool(vals) and min(vals) >= 1.0


# ------------------------------------------ element 3: decision-time board

def scoreboard(zoo: dict) -> list[dict]:
    """Rank the monitor zoo by decision-time performance, not transcript AUC.

    Primary key: matched-FPR timely recall at the last timely round.
    Secondary key: timely AUC. Ties are broken by name so the order is stable.
    Penalized delay is not in the released per-monitor record (see
    DELAY_UNAVAILABLE) and is therefore not a ranking key here.
    """
    mons = zoo.get("monitors", {})
    post_rank = {name: i + 1 for i, name in enumerate(zoo.get("rank_posthoc", []))}
    tauc_rank = {name: i + 1 for i, name in enumerate(zoo.get("rank_timely_auc", []))}

    ordered = sorted(
        mons.items(),
        key=lambda kv: (-(kv[1]["timely_recall_mfpr"] or 0.0),
                        -(kv[1]["timely_auc"] or 0.0), kv[0]))
    rows = []
    for i, (name, m) in enumerate(ordered):
        rows.append({
            "rank": i + 1,
            "monitor": name,
            "timely_recall_mfpr": m["timely_recall_mfpr"],
            "timely_auc": m["timely_auc"],
            "posthoc_auc": m["posthoc_auc"],
            "posthoc_rank": post_rank.get(name),
            "timely_auc_rank": tauc_rank.get(name),
            "rank_shift": (post_rank.get(name) or 0) - (i + 1),
            "penalized_delay": None,
            "is_constancy": name in CONSTANCY_MONITORS,
            "n_timely": m["n_timely"],
        })
    return rows


def constancy_demotion(zoo: dict) -> dict:
    """The specific, measured fate of the constancy monitors.

    Reported per ranking key, because the two keys do not agree and the view
    must not imply they do. Constancy monitors absent from the results tree
    are absent from the returned dict; callers must render that visibly.
    """
    post = {n: i + 1 for i, n in enumerate(zoo.get("rank_posthoc", []))}
    tauc = {n: i + 1 for i, n in enumerate(zoo.get("rank_timely_auc", []))}
    board = {r["monitor"]: r["rank"] for r in scoreboard(zoo)}
    out = {}
    for name in CONSTANCY_MONITORS:
        if name not in zoo.get("monitors", {}):
            continue
        out[name] = {
            "posthoc_rank": post.get(name),
            "timely_auc_rank": tauc.get(name),
            "scoreboard_rank": board.get(name),
            "timely_recall_mfpr": zoo["monitors"][name]["timely_recall_mfpr"],
        }
    return out


def zoo_headline(zoo: dict) -> dict:
    """Headline zoo quantities. An empty or partial zoo yields None fields
    rather than a crash; callers render those as an explicit unavailable
    state and the self-check reports them as FAIL lines."""
    mons = zoo.get("monitors", {})
    recalls = [(m["timely_recall_mfpr"] or 0.0) for m in mons.values()]
    kendall = zoo.get("kendall", {})
    k_auc = kendall.get("timely_auc", {})
    k_rec = kendall.get("timely_recall_mfpr", {})
    return {
        "n_monitors": len(mons),
        "n_trials": zoo.get("n_trials", 0),
        "max_timely_recall": max(recalls) if recalls else None,
        "tau_timely_auc": k_auc.get("tau"),
        "tau_timely_auc_ci": k_auc.get("ci95", (None, None)),
        "tau_timely_recall": k_rec.get("tau"),
        "tau_timely_recall_ci": k_rec.get("ci95", (None, None)),
        "posthoc_top": zoo.get("rank_posthoc", [])[:3],
        "fpr_sensitivity": zoo.get("deployed_fpr_sensitivity", {}),
        "latency_ablation": zoo.get("latency_ablation", {}),
    }
