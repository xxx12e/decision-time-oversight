r"""selfcheck.py -- assert the view's three elements against the numbers the
paper prints.

Rule of construction: the expected value of every check is a LITERAL --
transcribed from the paper, or hand-derived from the src/s2_engine.py
definitions with the derivation in a comment -- never the output of the
expression under test. The replay side recomputes from the released results
tree. If the two ever disagree the check fails loudly and the process exits
non-zero. A degraded results tree (missing cells, monitors, or budgets)
produces FAIL lines, never an uncaught exception.

Run it:
    python -X utf8 monitoring_view.py --selfcheck
    python -X utf8 monitoring_view.py --demo-fault   (proves the checks bite)
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path

import replay_core as rc

# --------------------------------------------------------------- literals
#
# Transcribed by hand from the artifact section of the paper and the study
# sections it cites. Nothing below is computed.

# Element 1 -- regime banner replay ledger.
PAPER_RETAINED_CATCHES = 2077        # "2{,}077 of 2{,}271"
PAPER_TOTAL_CATCHES = 2271
PAPER_RETAINED_SHARE = 0.91          # "retains 91\%"
PAPER_FORFEITED_PREVENTABLE = 6      # "(6 of 595)"
PAPER_TOTAL_PREVENTABLE = 595
PAPER_FORFEITED_SHARE = 0.010        # "forfeiting only 1.0\%"
PAPER_CELLS_TOTAL = 60
PAPER_PLACEMENT_CELLS = 28           # "28/60 cells, 47\%"
PAPER_CEILING_CELLS = 10             # "10/60, 17\%"
PAPER_SATURATED_CELLS = 3            # "covers 3 cells (5\%)"
PAPER_PLACEMENT_SHARE = 0.47
PAPER_CEILING_SHARE = 0.17
PAPER_SATURATED_SHARE = 0.05

# Element 1 -- the catch ledger as the paper now phrases it: the oracle is a
# reference policy, not a proven ceiling on catch count.
PAPER_FORFEITED_CATCHES = 194        # "giving up ... (194 of 2{,}271"
PAPER_FORFEITED_CATCH_SHARE = 0.085  # "giving up 8.5\%"
# The seven cells where uniform@B3 registers MORE timely catches than the
# onset oracle at any Study-1 budget -- transcribed from results/s1/stats.json
# and asserted against the paper's sentence "uniform coverage at $B{=}3$
# registers more catches in seven cells, all of them cells where those catches
# prevent nothing". Every one has oracle-preventable rate 0.0 (one
# ceiling-bound, six benign).
UNIFORM_B3_OVERTAKE_CELLS = [
    "1|0.66|clustered", "1|0.66|hub", "1|0.66|small_world",
    "1|0.74|clustered", "1|0.74|hub", "1|0.74|random", "1|0.74|small_world",
]

# Element 2 -- placement strip, flat-posterior fallback.
PAPER_AUDIT_EARLY_MISS = {1: 1.00, 2: 1.00, 4: 1.00}   # "from $1.00$"
PAPER_UNIFORM_MISS = {1: 0.48, 2: 0.18, 4: 0.00}       # "to $0.48/0.18/0.00$"
PAPER_FRONT_MISS = {1: 1.00, 2: 1.00, 4: 1.00}         # "Front-loading fails identically"
PAPER_ORACLE_MISS = {1: 0.00, 2: 0.00, 4: 0.00}        # "the onset oracle never misses"
PAPER_FALLBACK_POLICY = "uniform"                      # "hedges to uniform placement"

# Element 2 -- schedule geometry, pinned to LITERALS transcribed by hand from
# src/s2_engine.py (L_TOTAL = 160). Never computed by calling the functions
# under test -- that is the whole point of these pins.
#
#   uniform:   round(i * 160 / (B+1)), i = 1..B
#              B=1: round(80.0) = 80
#              B=2: round(53.33) = 53, round(106.67) = 107
#              B=4: round(32/64/96/128) exact
#   geometric: min(159, 2**(i+1)), i = 0..B-1  ->  2, 4, 8, 16, ...
#   front:     range(10, 10+B)
#   back:      range(160-B, 160)
#   oracle:    min(159, tau+i), i = 0..B-1; the min() clamps at the stream end
#              and the set collapses the clamped duplicates.
EXPECTED_UNIFORM = {1: [80], 2: [53, 107], 4: [32, 64, 96, 128]}
EXPECTED_GEOMETRIC = {1: [2], 2: [2, 4], 4: [2, 4, 8, 16]}
EXPECTED_FRONT = {1: [10], 2: [10, 11], 4: [10, 11, 12, 13]}
EXPECTED_BACK = {1: [159], 2: [158, 159], 4: [156, 157, 158, 159]}
EXPECTED_ORACLE_TAU100 = {1: [100], 2: [100, 101], 4: [100, 101, 102, 103]}
EXPECTED_ORACLE_TAU158_B4 = [158, 159]   # 158+2=160 and 158+3=161 clamp to 159

# Element 2 -- posterior geometry, hand-derived literals.
#
# flat_posterior: uniform over the REGISTERED onset support [40, 120] of the
# 160-step stream -- 81 support points (120 - 40 + 1), weight 1/81 each, and
# zero everywhere else. NOT uniform over all 160 steps.
FLAT_SUPPORT = list(range(40, 121))
FLAT_WEIGHT = 1.0 / 81.0
#
# posterior_targeted_schedule places B checks at the first steps where the
# cumulative posterior reaches i/(B+1):
#   * point mass at 60 (literal input list): every quantile crosses at 60,
#     the set collapses -> [60] at any B.
#   * half mass at 50, half at 100 (literal input list), B=3: quantiles are
#     1/4, 1/2, 3/4; cum(50) = 0.5 crosses 1/4 and 1/2, cum(100) = 1.0
#     crosses 3/4 -> [50, 100].
#   * informative_posterior(80, 40), B=1: support [40,120] is symmetric about
#     centre 80, so cum(79) = (1 - w80)/2 < 1/2 and cum(80) = (1 + w80)/2
#     >= 1/2 -> the median check lands exactly at [80].
EXPECTED_TARGETED_POINTMASS_60 = [60]
EXPECTED_TARGETED_TWOPOINT_B3 = [50, 100]
EXPECTED_TARGETED_SYMMETRIC_B1 = [80]
#
# mass_after_last_check on the flat posterior:
#   * after uniform B=2 = [53, 107]: steps 108..120 remain -> 13 of the 81
#     support points -> 13/81 = 0.160494.
#   * after audit-early B=4 = [2, 4, 8, 16]: the last check (16) fires before
#     the earliest possible onset (40), so ALL posterior mass is unseen -> 1.0.
#   * empty schedule: nothing is checked -> 1.0.
EXPECTED_MASS_AFTER_UNIFORM_B2 = 13.0 / 81.0
EXPECTED_MASS_AFTER_GEOMETRIC_B4 = 1.0
EXPECTED_MASS_EMPTY_SCHEDULE = 1.0
#
# posterior_is_flat boundary, derived from FLATNESS_RATIO = 1.05: for
# informative_posterior(c, s) the max/min ratio over the support is
# exp(0.5 * (d_max/s)^2) with d_max = max(c - 40, 120 - c), so flat iff
# s >= d_max * sqrt(0.5 / ln 1.05) = d_max * 3.2013.
#   centre 80 (d_max = 40): threshold 128.05 -> spread 129 flat, 128 not.
#   centre 40 (d_max = 80): threshold 256.10 -> spread 257 flat.
# All of these spreads are reachable on the strip's slider (1..260), so the
# "widening the spread flattens the posterior" caption describes a state the
# view can actually reach.
FLAT_SPREAD_AT_C80 = 129.0       # classifies flat
NONFLAT_SPREAD_AT_C80 = 128.0    # does not
FLAT_SPREAD_AT_C40 = 257.0       # worst-case centre, still reachable

# Element 3 -- decision-time scoreboard.
PAPER_ZOO_SIZE = 13                  # "thirteen monitor variants"
PAPER_ZOO_TRIALS = 130               # "130 real-agent panel trials" (90 archived + 40 S3)
PAPER_MAX_TIMELY_RECALL = 0.24       # "best variant reaches only $0.24$ timely recall"
PAPER_TAU_TIMELY_AUC = 0.67          # "$\tau = 0.67$ $[0.62, 0.72]$ on timely AUC"
PAPER_TAU_TIMELY_AUC_CI = (0.62, 0.72)
PAPER_TAU_TIMELY_RECALL = 0.51       # "$\tau = 0.51$ ... the preregistered pair"
PAPER_POSTHOC_TOP = "conf_constancy"  # hindsight winner is a constancy monitor

# Regression pins for the demotion sentences the VIEW itself prints. These are
# not paper-printed numbers; they are pinned so the view's wording cannot drift
# away from the data behind it. Marked (view) in the report.
VIEW_SCOREBOARD_TOP3 = ["conf_mean", "conf_constancy", "s2_only"]
VIEW_CONF_CONSTANCY_RANKS = (1, 5, 2)     # posthoc, timely-AUC, scoreboard
VIEW_CLAIM_CONSTANCY_RANKS = (5, 6, 10)
VIEW_CLAIM_CONSTANCY_RECALL = 0.00
VIEW_HIGH_CONF_SHARE_RANKS = (4, 1, 8)

TOL_SHARE = 0.005
TOL_TIGHT = 0.001
TOL_RATE = 0.005


@dataclass
class Check:
    group: str
    label: str
    paper: object
    replay: object
    ok: bool
    note: str = ""


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


class Runner:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def num(self, group: str, label: str, paper: float, replay: float,
            tol: float = 0.0, note: str = "") -> None:
        # A replay value that is missing or non-numeric (None from a degraded
        # results tree) is a FAIL line, never an exception.
        try:
            ok = abs(float(paper) - float(replay)) <= tol
        except (TypeError, ValueError):
            ok = False
            note = (note + "; " if note else "") + "replay value not numeric"
        self.checks.append(Check(group, label, paper, replay, ok, note))

    def exact(self, group: str, label: str, paper: object, replay: object,
              note: str = "") -> None:
        self.checks.append(Check(group, label, paper, replay, paper == replay, note))

    def guarded(self, group: str, label: str, fn) -> None:
        """Run a block of checks; an exception inside it becomes a FAIL line
        naming the exception instead of aborting the whole self-check."""
        try:
            fn()
        except Exception as exc:               # noqa: BLE001 -- report, never raise
            self.checks.append(Check(group, label, "no exception",
                                     f"{type(exc).__name__}: {exc}", False))


# ----------------------------------------------------------------- checks

G1 = "1. regime banner"
G2 = "2. placement strip"
G3 = "3. decision-time scoreboard"
G0 = "0. paper text"


def run_geometry_checks(r: Runner) -> None:
    """Pin every schedule generator and posterior-geometry function against
    hand-derived literals (see the EXPECTED_* blocks above for derivations).

    Pure -- needs no results bundle. The fault demo runs exactly this suite
    with each generator monkeypatched in turn, to prove the pins bite:
    sabotaging any of these functions now produces FAIL lines.
    """
    for B in rc.S2_REPLAY_BUDGETS:
        r.exact(G2, f"schedule pin: s2_uniform({B}) == {EXPECTED_UNIFORM[B]}",
                EXPECTED_UNIFORM[B], rc.s2_uniform(B),
                note="literal from src/s2_engine.py: round(i*160/(B+1))")
        r.exact(G2, f"schedule pin: s2_geometric({B}) == {EXPECTED_GEOMETRIC[B]}",
                EXPECTED_GEOMETRIC[B], rc.s2_geometric(B),
                note="literal: min(159, 2**(i+1))")
        r.exact(G2, f"schedule pin: s2_front({B}) == {EXPECTED_FRONT[B]}",
                EXPECTED_FRONT[B], rc.s2_front(B),
                note="literal: range(10, 10+B)")
        r.exact(G2, f"schedule pin: s2_back({B}) == {EXPECTED_BACK[B]}",
                EXPECTED_BACK[B], rc.s2_back(B),
                note="literal: range(160-B, 160)")
        r.exact(G2, f"schedule pin: s2_oracle({B}, tau=100) == "
                    f"{EXPECTED_ORACLE_TAU100[B]}",
                EXPECTED_ORACLE_TAU100[B], rc.s2_oracle(B, 100),
                note="literal: min(159, tau+i)")
    r.exact(G2, "schedule pin: s2_oracle(4, tau=158) clamps at the stream end",
            EXPECTED_ORACLE_TAU158_B4, rc.s2_oracle(4, 158),
            note="158+2 and 158+3 clamp to 159 and dedup")

    # flat_posterior: support and weights, pinned to the registered onset
    # range [40, 120] -- NOT all 160 steps.
    fp = rc.flat_posterior()
    r.exact(G2, "flat_posterior support == registered onset steps 40..120",
            FLAT_SUPPORT, [t for t, p in enumerate(fp) if p > 0.0],
            note="81 support points, zero elsewhere")
    r.num(G2, "flat_posterior weight on each support point == 1/81",
          FLAT_WEIGHT, max(fp), 1e-12,
          note="uniform over 81 points; max weight equals the literal")
    r.num(G2, "flat_posterior total mass == 1", 1.0, sum(fp), 1e-9)

    # posterior_targeted_schedule against literal inputs and derived outputs.
    pointmass = [0.0] * rc.S2_LENGTH
    pointmass[60] = 1.0
    r.exact(G2, "targeted schedule pin: point mass at 60, B=2 -> [60]",
            EXPECTED_TARGETED_POINTMASS_60,
            rc.posterior_targeted_schedule(pointmass, 2),
            note="all quantiles cross at the point mass; set collapses")
    twopoint = [0.0] * rc.S2_LENGTH
    twopoint[50] = 0.5
    twopoint[100] = 0.5
    r.exact(G2, "targeted schedule pin: half mass at 50/100, B=3 -> [50, 100]",
            EXPECTED_TARGETED_TWOPOINT_B3,
            rc.posterior_targeted_schedule(twopoint, 3),
            note="quantiles 1/4,1/2,3/4; cum crosses at 50, 50, 100")
    r.exact(G2, "targeted schedule pin: informative_posterior(80, 40), B=1 -> [80]",
            EXPECTED_TARGETED_SYMMETRIC_B1,
            rc.posterior_targeted_schedule(rc.informative_posterior(80.0, 40.0), 1),
            note="support symmetric about 80, so the median check is 80")

    # mass_after_last_check against hand-derived literals.
    r.num(G2, "mass pin: flat posterior after uniform B=2 [53,107] == 13/81",
          EXPECTED_MASS_AFTER_UNIFORM_B2,
          rc.mass_after_last_check(fp, [53, 107]), 1e-9,
          note="support steps 108..120 remain: 13 of 81")
    r.num(G2, "mass pin: flat posterior after audit-early B=4 [2,4,8,16] == 1.0",
          EXPECTED_MASS_AFTER_GEOMETRIC_B4,
          rc.mass_after_last_check(fp, [2, 4, 8, 16]), 1e-9,
          note="last check fires before the earliest possible onset")
    r.num(G2, "mass pin: empty schedule leaves all mass unseen == 1.0",
          EXPECTED_MASS_EMPTY_SCHEDULE, rc.mass_after_last_check(fp, []), 1e-9)

    # posterior_is_flat boundary: the flat classification must be genuinely
    # reachable on the strip's sliders (spread 1..260), and must not fire on
    # a sharp posterior.
    r.exact(G2, "flatness pin: centre 80, spread 129 (reachable) classifies flat",
            True, rc.posterior_is_flat(rc.informative_posterior(80.0, 129.0)),
            note="ratio exp(800/129^2) = 1.0492 <= 1.05")
    r.exact(G2, "flatness pin: centre 80, spread 128 does NOT classify flat",
            False, rc.posterior_is_flat(rc.informative_posterior(80.0, 128.0)),
            note="ratio exp(800/128^2) = 1.0500 > 1.05")
    r.exact(G2, "flatness pin: centre 40, spread 257 (worst-case centre) is flat",
            True, rc.posterior_is_flat(rc.informative_posterior(40.0, 257.0)),
            note="d_max=80 -> threshold 256.1; still on the slider")
    r.exact(G2, "sharp posterior is not classified flat", False,
            rc.posterior_is_flat(rc.informative_posterior(60.0, 6.0)))
    r.exact(G2, "very wide posterior falls back to flat", True,
            rc.posterior_is_flat(rc.informative_posterior(80.0, 400.0)))


def run_checks(bundle: rc.Bundle, trials_fc: dict | None = None,
               use_trials: bool = True, paper_tex: Path | None = None) -> list[Check]:
    r = Runner()

    # ---------------- element 1: regime banner --------------------------
    def _element_one() -> None:
        led = rc.routing_ledger(bundle.s1)
        r.exact(G1, "cells in the regime map", PAPER_CELLS_TOTAL, led["n_cells"])
        r.exact(G1, "oracle-achievable timely catches, all cells",
                PAPER_TOTAL_CATCHES, round(led["total_catches"]))
        r.exact(G1, "timely catches retained when ceiling-bound cells get zero checks",
                PAPER_RETAINED_CATCHES, round(led["retained_catches"]))
        r.num(G1, "share of timely catches retained",
              PAPER_RETAINED_SHARE, led["retained_share"], TOL_SHARE)
        r.exact(G1, "timely catches given up (paper: 194 of 2,271)",
                PAPER_FORFEITED_CATCHES,
                round(led["total_catches"] - led["retained_catches"]))
        r.num(G1, "share of timely catches given up (paper: 8.5%)",
              PAPER_FORFEITED_CATCH_SHARE,
              None if led["retained_share"] is None else 1.0 - led["retained_share"],
              TOL_SHARE)
        r.exact(G1, "oracle-preventable failures, all cells",
                PAPER_TOTAL_PREVENTABLE, round(led["total_preventable"]))
        r.exact(G1, "oracle-preventable failures forfeited in ceiling-bound cells",
                PAPER_FORFEITED_PREVENTABLE, round(led["forfeited_preventable"]))
        r.num(G1, "share of oracle-preventable failures forfeited",
              PAPER_FORFEITED_SHARE, led["forfeited_share"], TOL_TIGHT)

        counts = rc.regime_counts(bundle.s1)
        shares = rc.regime_shares(bundle.s1)
        r.exact(G1, "placement-critical cells", PAPER_PLACEMENT_CELLS,
                counts["placement_critical"])
        r.exact(G1, "ceiling-bound cells", PAPER_CEILING_CELLS, counts["ceiling_bound"])
        r.exact(G1, "budget-saturated cells", PAPER_SATURATED_CELLS,
                counts["budget_saturated"])
        r.num(G1, "placement-critical share", PAPER_PLACEMENT_SHARE,
              shares["placement_critical"], TOL_SHARE)
        r.num(G1, "ceiling-bound share", PAPER_CEILING_SHARE,
              shares["ceiling_bound"], TOL_SHARE)
        r.num(G1, "budget-saturated share", PAPER_SATURATED_SHARE,
              shares["budget_saturated"], TOL_SHARE)

        # The oracle rate is a reference policy, NOT a catch-count ceiling:
        # exactly these seven cells see more uniform@B3 catches, and in every
        # one the extra catches prevent nothing.
        over = rc.uniform_b3_overtakes(bundle.s1)
        r.exact(G1, "cells where uniform@B3 out-catches the onset-oracle reference",
                UNIFORM_B3_OVERTAKE_CELLS, [o["key"] for o in over],
                note="oracle rate is a reference policy, not a catch-count ceiling")
        r.num(G1, "oracle-preventable rate in every overtaken cell (prevent nothing)",
              0.0, max((o["oracle_preventable"] for o in over), default=None), 0.0)

        # The banner must actually emit the ceiling-bound message for a
        # ceiling-bound cell -- the message is an output, not an error state.
        ceiling_key = next((k for k, c in sorted(bundle.s1["cells"].items())
                            if c["regime"] == "ceiling_bound"), None)
        if ceiling_key is None:
            r.exact(G1, "a ceiling-bound cell exists in results/s1/stats.json",
                    "at least one", "none found")
            return
        banner = rc.lookup_cell(bundle.s1, ceiling_key)
        r.exact(G1, f"banner headline for ceiling-bound cell {ceiling_key}",
                "Prevention value near zero here", banner["headline"])
        r.num(G1, f"oracle-preventable rate in {ceiling_key} (nothing to prevent)",
              0.0, banner["oracle_preventable"], 0.05,
              note="ceiling-bound criterion: oracle prevention <= 0.05")

    r.guarded(G1, "element 1 checks aborted by exception", _element_one)

    # ---------------- element 2: placement strip ------------------------
    def _element_two() -> None:
        fb = rc.fallback_replay(bundle.s2)
        for B in rc.S2_REPLAY_BUDGETS:
            row = fb.get(B)
            if row is None:
                r.exact(G2, f"front_collapse block carries budget B={B}",
                        "present", "not in results")
                continue
            r.num(G2, f"audit-early miss at B={B}", PAPER_AUDIT_EARLY_MISS[B],
                  row["audit_early_miss"], 0.0)
            r.num(G2, f"front-loaded miss at B={B}", PAPER_FRONT_MISS[B],
                  row["front_miss"], 0.0)
            r.num(G2, f"uniform miss at B={B}", PAPER_UNIFORM_MISS[B],
                  row["uniform_miss"], TOL_RATE)
            r.num(G2, f"onset-oracle miss at B={B}", PAPER_ORACLE_MISS[B],
                  row["oracle_miss"], 0.0)

        # The fallback wiring itself: a flat posterior must select uniform
        # placement, the schedule it selects must equal the HAND-TRANSCRIBED
        # engine schedule (a literal -- never rc.s2_uniform(B), which is the
        # expression under test), and the number the strip renders for it must
        # be the number the paper prints.
        flat = rc.flat_posterior()
        for B in rc.S2_REPLAY_BUDGETS:
            chosen = rc.choose_placement(flat, B)
            r.exact(G2, f"flat posterior selects a policy at B={B}",
                    PAPER_FALLBACK_POLICY, chosen["policy"])
            r.exact(G2, f"flat-posterior schedule at B={B} equals the engine literal",
                    EXPECTED_UNIFORM[B], chosen["schedule"],
                    note="expected side is a hand-transcribed literal")
            row = fb.get(B)
            if row is None:
                r.exact(G2, f"miss rate the strip renders at B={B}",
                        PAPER_UNIFORM_MISS[B], "not in results")
                continue
            rendered = row["uniform_miss"] if chosen["policy"] == "uniform" \
                else row["audit_early_miss"]
            r.num(G2, f"miss rate the strip renders at B={B}",
                  PAPER_UNIFORM_MISS[B], rendered, TOL_RATE)

        # Independent recomputation from the trial log: the same literals must
        # come out of per-trial outcomes, not just out of the stored aggregate.
        if use_trials:
            fc2 = trials_fc if trials_fc is not None else \
                rc.recompute_front_collapse(bundle.s2_trials_path)
            for B in rc.S2_REPLAY_BUDGETS:
                r.num(G2, f"uniform miss at B={B}, recomputed from trials.jsonl",
                      PAPER_UNIFORM_MISS[B], fc2[B]["uniform_miss"], TOL_RATE)
                r.num(G2, f"audit-early miss at B={B}, recomputed from trials.jsonl",
                      PAPER_AUDIT_EARLY_MISS[B], fc2[B]["audit_early_miss"], 0.0)

    r.guarded(G2, "element 2 checks aborted by exception", _element_two)
    r.guarded(G2, "element 2 geometry checks aborted by exception",
              lambda: run_geometry_checks(r))

    # ---------------- element 3: decision-time scoreboard ---------------
    def _element_three() -> None:
        head = rc.zoo_headline(bundle.zoo)
        r.exact(G3, "monitor variants ranked", PAPER_ZOO_SIZE, head["n_monitors"])
        r.exact(G3, "panel trials replayed", PAPER_ZOO_TRIALS, head["n_trials"])
        r.num(G3, "best timely recall in the whole zoo", PAPER_MAX_TIMELY_RECALL,
              head["max_timely_recall"], TOL_RATE)
        r.num(G3, "Kendall tau, hindsight vs timely AUC", PAPER_TAU_TIMELY_AUC,
              head["tau_timely_auc"], TOL_SHARE)
        r.num(G3, "Kendall tau CI low", PAPER_TAU_TIMELY_AUC_CI[0],
              head["tau_timely_auc_ci"][0], TOL_SHARE)
        r.num(G3, "Kendall tau CI high", PAPER_TAU_TIMELY_AUC_CI[1],
              head["tau_timely_auc_ci"][1], TOL_SHARE)
        r.num(G3, "Kendall tau, hindsight vs matched-FPR timely recall",
              PAPER_TAU_TIMELY_RECALL, head["tau_timely_recall"], TOL_SHARE)
        r.exact(G3, "hindsight winner", PAPER_POSTHOC_TOP,
                head["posthoc_top"][0] if head["posthoc_top"] else "not in results")

        board = rc.scoreboard(bundle.zoo)
        r.exact(G3, "(view) scoreboard top three by timely recall",
                VIEW_SCOREBOARD_TOP3, [row["monitor"] for row in board[:3]],
                note="regression pin on the view's own ordering")
        dem = rc.constancy_demotion(bundle.zoo)
        cc = dem.get("conf_constancy")
        if cc is None:
            r.exact(G3, "conf_constancy present in results/s4/zoo.json",
                    "present", "not in results")
        else:
            r.exact(G3, "(view) conf_constancy ranks (hindsight, timely AUC, scoreboard)",
                    list(VIEW_CONF_CONSTANCY_RANKS),
                    [cc["posthoc_rank"], cc["timely_auc_rank"], cc["scoreboard_rank"]],
                    note="demoted on timely AUC; NOT demoted on timely recall")
        cl = dem.get("claim_constancy")
        if cl is None:
            r.exact(G3, "claim_constancy present in results/s4/zoo.json",
                    "present", "not in results")
        else:
            r.exact(G3, "(view) claim_constancy ranks (hindsight, timely AUC, scoreboard)",
                    list(VIEW_CLAIM_CONSTANCY_RANKS),
                    [cl["posthoc_rank"], cl["timely_auc_rank"], cl["scoreboard_rank"]])
            r.num(G3, "(view) claim_constancy timely recall", VIEW_CLAIM_CONSTANCY_RECALL,
                  cl["timely_recall_mfpr"], 0.0)
        hcs = next((row for row in board if row["monitor"] == "high_conf_share"),
                   None)
        if hcs is None:
            r.exact(G3, "high_conf_share present in results/s4/zoo.json",
                    "present", "not in results")
        else:
            r.exact(G3, "(view) high_conf_share ranks (hindsight, timely AUC, scoreboard)",
                    list(VIEW_HIGH_CONF_SHARE_RANKS),
                    [hcs["posthoc_rank"], hcs["timely_auc_rank"], hcs["rank"]])
        r.exact(G3, "penalized delay is reported as unavailable, not imputed",
                True, all(row["penalized_delay"] is None for row in board),
                note="results/s4/zoo.json carries no per-monitor delay")

    r.guarded(G3, "element 3 checks aborted by exception", _element_three)

    # ---------------- paper text cross-check ----------------------------
    # Guards against transcription drift in the literals above. Skipped when
    # the paper source is not shipped alongside the artifact.
    if paper_tex is not None and paper_tex.is_file():
        tex = paper_tex.read_text(encoding="utf-8")
        needles = [
            (r"giving up 8\.5\\% of the timely catches it makes \(194 of 2\{,\}271",
             "194 of 2,271 catches given up (8.5%)"),
            # "onset oracle" was renamed "onset reference" in review: it is a
            # reference policy, not a proven ceiling, and the paper now says so
            (r"forfeits 1\.0\\% of the failures the onset reference prevents \(6 of 595\)",
             "forfeits 1.0% (6 of 595)"),
            (r"uniform coverage at \$B\{=\}3\$ registers more catches in seven cells",
             "uniform@B3 out-catches in seven cells"),
            (r"all of them cells where those catches prevent nothing",
             "the seven cells prevent nothing"),
            (r"oracle-placement bound", "oracle-placement bound"),
            (r"\$1\.00\$ to \$0\.48/0\.18/0\.00\$", "1.00 to 0.48/0.18/0.00"),
            (r"\$B = 1/2/4\$", "at B = 1/2/4"),
            (r"thirteen monitor variants", "thirteen monitor variants"),
        ]
        for pat, label in needles:
            r.exact(G0, f"literal present in paper source: {label}",
                    True, bool(re.search(pat, tex)))
    return r.checks


# ---------------------------------------------------------------- report

def format_report(checks: list[Check], header: str = "") -> str:
    lines: list[str] = []
    if header:
        lines.append(header)
    width = max((len(c.label) for c in checks), default=10)
    width = min(width, 74)
    seen: list[str] = []
    for c in checks:
        if c.group not in seen:
            seen.append(c.group)
            lines.append("")
            lines.append(f"[{c.group}]")
        tag = "PASS" if c.ok else "FAIL"
        lines.append(f"  {tag}  {c.label:<{width}}  paper={_fmt(c.paper):>10}  "
                     f"replay={_fmt(c.replay):>10}"
                     + (f"   ({c.note})" if c.note else ""))
    n_fail = sum(1 for c in checks if not c.ok)
    lines.append("")
    lines.append(f"{len(checks) - n_fail} PASS, {n_fail} FAIL")
    return "\n".join(lines)


# ------------------------------------------------------------ fault demo

def _corrupt(bundle: rc.Bundle) -> rc.Bundle:
    """Deliberately break one input per element, to prove the checks bite."""
    s1 = copy.deepcopy(bundle.s1)
    key = next(k for k, c in sorted(s1["cells"].items())
               if c["regime"] == "ceiling_bound")
    s1["cells"][key]["regime"] = "placement_critical"       # moves the ledger
    s2 = copy.deepcopy(bundle.s2)
    s2["front_collapse"]["B2"]["uniform_miss"] = 0.30       # wrong miss rate
    zoo = copy.deepcopy(bundle.zoo)
    zoo["monitors"]["conf_mean"]["timely_recall_mfpr"] = 0.30  # wrong ceiling
    return rc.Bundle(root=bundle.root, s1=s1, s2=s2, zoo=zoo, s3=bundle.s3)


# Each schedule/geometry function the geometry pins cover, with a stub that
# returns a plausibly-typed wrong answer. Monkeypatching rc.<name> to the stub
# must make at least one geometry pin FAIL -- the literals cannot move with
# the code under test, which is exactly what the pins exist to prove.
GENERATOR_SABOTAGES: list[tuple[str, object]] = [
    ("s2_uniform", lambda B: [7]),
    ("s2_geometric", lambda B: [7]),
    ("s2_front", lambda B: [7]),
    ("s2_back", lambda B: [7]),
    ("s2_oracle", lambda B, tau: [7]),
    ("flat_posterior", lambda: [1.0 / rc.S2_LENGTH] * rc.S2_LENGTH),
    ("posterior_targeted_schedule", lambda post, B: [7]),
    ("mass_after_last_check", lambda post, schedule: 0.0),
    ("posterior_is_flat", lambda post, ratio=rc.FLATNESS_RATIO: True),
]


def sabotage_generators() -> tuple[list[str], bool]:
    """Monkeypatch each generator in turn and rerun the geometry pins.

    Returns one report line per sabotage and True only when every sabotage
    was detected (>= 1 FAIL among the geometry checks while patched).
    """
    lines: list[str] = []
    all_detected = True
    for name, stub in GENERATOR_SABOTAGES:
        orig = getattr(rc, name)
        setattr(rc, name, stub)
        try:
            r = Runner()
            r.guarded(G2, f"geometry checks aborted under sabotage of {name}",
                      lambda: run_geometry_checks(r))
            n_fail = sum(1 for c in r.checks if not c.ok)
        finally:
            setattr(rc, name, orig)
        if n_fail > 0:
            lines.append(f"  DETECTED      {name:28s} -> "
                         f"{n_fail} geometry check(s) FAIL")
        else:
            lines.append(f"  NOT DETECTED  {name:28s} -> "
                         f"all geometry checks still pass (pin is vacuous)")
            all_detected = False
    return lines, all_detected


def demo_fault(bundle: rc.Bundle) -> tuple[str, bool]:
    """Corrupt the data and sabotage the generators; failures are expected.

    Phase 1 runs the identical check suite over corrupted inputs. Phase 2
    monkeypatches every schedule/geometry generator in turn and reruns the
    geometry pins. The demo passes only if BOTH phases are detected.
    """
    bad = _corrupt(bundle)
    bad_trials = copy.deepcopy(rc.recompute_front_collapse(bundle.s2_trials_path))
    bad_trials[1]["uniform_miss"] = 0.90
    checks = run_checks(bad, trials_fc=bad_trials, use_trials=True, paper_tex=None)
    n_fail = sum(1 for c in checks if not c.ok)
    report = format_report(
        checks,
        "FAULT DEMO, phase 1 -- three inputs deliberately corrupted (one per element)\n"
        "  s1: one ceiling-bound cell relabelled placement-critical\n"
        "  s2: front_collapse uniform miss at B=2 set to 0.30\n"
        "  s2: trials-recomputed uniform miss at B=1 set to 0.90\n"
        "  s4: conf_mean timely recall set to 0.30")
    sab_lines, sab_ok = sabotage_generators()
    report += (
        "\n\nFAULT DEMO, phase 2 -- every schedule/geometry generator "
        "monkeypatched in turn;\nthe hand-transcribed literal pins must FAIL "
        "under each sabotage:\n" + "\n".join(sab_lines) + "\n"
        + ("all sabotages detected"
           if sab_ok else "SOME SABOTAGE WENT UNDETECTED"))
    return report, n_fail > 0 and sab_ok
