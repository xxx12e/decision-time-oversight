# Attention-routing reference monitoring view

A working implementation of the reference monitoring view described in the
paper's artifact section. Three elements, each a direct rendering of a measured
bound from the released results tree:

1. **Regime banner** - takes a fault-speed estimate (and, optionally, the
   observables the regime map is parameterized by) and states which of the three
   honest messages applies: *place carefully* (placement-critical), *any
   coverage works* (budget-saturated), or *checking cannot help here; prevent
   upstream* (ceiling-bound). It also shows the routing ledger: what spending
   zero checks in ceiling-bound cells keeps and what it gives up.
2. **Placement strip** - renders the chosen policy's check times against an
   onset posterior over the 160-step stream, next to the measured miss rate of
   each replayed policy. When the observables are uninformative the posterior is
   flat and the strip hedges to uniform placement, which is the registered
   fallback.
3. **Decision-time scoreboard** - ranks the thirteen monitor variants by
   matched-FPR timely recall at the last round where intervention still prevents
   the failure, with the hindsight (transcript AUC) rank alongside so the
   reordering is visible.

Everything is computed locally from JSON and JSONL files. No network access, no
GPU, no model inference, no external services.

## Running it

Interactive view (needs `streamlit`):

    streamlit run monitoring_view.py

Replay self-check (standard library only - no Streamlit, no server):

    python -X utf8 monitoring_view.py --selfcheck

Prove the self-check can fail (phase 1 corrupts three inputs on purpose, one
per element; phase 2 monkeypatches every schedule/geometry generator in turn
and shows the literal pins failing under each sabotage; exits 0 only when both
phases are detected):

    python -X utf8 monitoring_view.py --demo-fault

Files: `monitoring_view.py` (entry point and interface), `replay_core.py` (all
computation), `selfcheck.py` (the assertions and their literals). Exclude any
`__pycache__/` directory when packaging: compiled files embed the build
machine's source paths.

Data is read from the released results tree, located relative to this file:
`results/s1/stats.json`, `results/s2/stats.json`, `results/s2/trials.jsonl`,
`results/s4/zoo.json`. No path in this package is absolute.

## Exactly what the self-check asserts

Each check compares a **literal** - transcribed from the paper, or hand-derived
from the `src/s2_engine.py` schedule definitions with the derivation in a
comment - against a value **recomputed from the results tree or the artifact's
own functions**. The expected side is never produced by the expression under
test, so a change in the data, in the loading code, in the schedule geometry,
or in the routing logic breaks the check rather than moving with it. Non-zero
exit on any failure. A degraded results tree (missing cells, monitors, or
budgets) produces FAIL lines, never an uncaught exception.

**Check count: 98 when `paper/main.tex` is shipped next to the artifact
(19 + 57 + 14 + 8); 90 when it is not** - a packaged zip without the paper
source skips the 8 paper-text checks and runs the other 90 unchanged.

**Element 1 - regime banner (19 checks).** Recomputes the routing ledger over
all 60 Study-1 cells x 50 seeds: for every cell, the onset-oracle timely-catch
rate (a reference policy, not a catch-count ceiling) and the failure rate the
oracle can prevent. Asserts, against the paper's printed numbers: 2,271
oracle-achievable timely catches in total; 2,077 retained / 194 given up
(8.5%, tolerance 0.005) when ceiling-bound cells receive zero checks; 595
oracle-preventable failures in total; 6 forfeited (1.0%, tolerance 0.001);
regime counts 28 / 10 / 3 of 60 and shares 0.47 / 0.17 / 0.05. It pins the
exact seven cells in which uniform@B3 registers more timely catches than the
onset-oracle reference, and that every one of them has an oracle-preventable
rate of 0.0 - the extra catches prevent nothing, which is why the oracle rate
is labelled a reference and not a ceiling. It also asserts that the banner
emits the literal string `Checking cannot help here` for a ceiling-bound cell
- the message is an output of the view, not an error state.

**Element 2 - placement strip (57 checks).** Asserts the measured miss rates
behind the fallback: audit-early (geometric) 1.00 and front-loaded 1.00 at
B = 1 / 2 / 4; uniform 0.48 / 0.18 / 0.00 (tolerance 0.005); onset oracle 0.00.
It then asserts the *wiring*: a flat posterior selects the policy string
`uniform`, the schedule it selects equals the hand-transcribed engine literal
(B=1 -> [80], B=2 -> [53, 107], B=4 -> [32, 64, 96, 128] - never the output of
`s2_uniform` itself), and the miss rate the strip renders for it equals the
paper's 0.48 / 0.18 / 0.00. Thirty further geometry pins hold every schedule
generator and posterior function to hand-derived literals: uniform, geometric,
front, back and oracle schedules at B = 1 / 2 / 4 (plus the oracle's
stream-end clamp); `flat_posterior` supported exactly on the registered onset
steps 40..120 with weight 1/81 (not all 160 steps); posterior-targeted
schedules for a point mass, a two-point posterior, and a symmetric
`informative_posterior(80, 40)`; posterior mass after the last check (13/81
after uniform B=2, 1.0 after audit-early B=4, 1.0 for an empty schedule); and
the flatness boundary - spread 129 at centre 80 classifies flat, spread 128
does not, and the worst-case centres flatten at spread 257, all reachable on
the strip's spread slider (1..260). Finally it recomputes the miss rates
**independently from `results/s2/trials.jsonl`**, pooling per-trial outcomes
over the diagnosable well-specified cells, so a stale or edited aggregate
cannot make the check agree with itself.

**Element 3 - decision-time scoreboard (14 checks).** Asserts 13 monitor
variants over 90 panel trials; best timely recall in the whole zoo 0.14
(tolerance 0.005); Kendall tau 0.62 with CI [0.54, 0.67] on timely AUC and 0.49
on matched-FPR timely recall; the hindsight winner is `conf_constancy`. Five
further checks are regression pins on the ranks the view itself prints (marked
`(view)` in the output; these are not paper-printed numbers): the scoreboard's
top three, and the hindsight / timely-AUC / scoreboard ranks of
`conf_constancy`, `claim_constancy` and `high_conf_share`. One check asserts
that the penalized-delay column stays empty rather than being imputed.

**Paper text (8 checks, only when the paper source is present).** The literals
above are located in `paper/main.tex` by regular expression - including the
oracle-placement-bound phrasing and the seven-cell reference-not-ceiling
sentence - so a transcription slip in this file cannot pass silently. Skipped,
and subtracted from the check count, when the source is absent.

Sample output, tail (98 with the paper source, 90 without):

    98 PASS, 0 FAIL
    SELFCHECK PASS

## What this does not claim

- **No claim about human operators, anywhere.** Every bound displayed is an
  oracle-placement bound: the detection instrument is held fixed at the
  deployed detector, and the oracle optimizes only where and when it fires.
  The bound caps every checking *schedule* of the deployed instrument -
  whatever the best-placed schedule of that detector cannot catch, no schedule
  of it can, and an overseer who additionally pays reading latency sits below
  it on that scheduling axis. It does not cap a reader whose judgment beats
  the instrument; that axis is measured by the Study-4 monitor zoo, not
  bounded. Nothing here is a usability result, an attention model, or an
  evaluation with users.
- **Penalized delay is not in the scoreboard.** The released Study-4 record
  scores each monitor by post-hoc AUC, timely AUC and matched-FPR timely recall;
  it carries no per-monitor penalized delay. The column is displayed as
  *not measured* rather than filled with a proxy. Penalized delay does appear in
  the placement strip's per-cell table, where it was measured.
- **The constancy demotion is stated per ranking key, not averaged.**
  `conf_constancy` is rank 1 of 13 in hindsight and falls to rank 5 on timely
  AUC - the mechanism being that a constancy signal needs two rounds of history
  and only one exists at the last timely round. But on matched-FPR timely
  recall, the scoreboard's primary key, it is rank 2 (0.123 against a zoo best
  of 0.142), so on that key it is not demoted. `claim_constancy` is demoted on
  both keys (rank 5 to 10, timely recall 0.00). The view prints both columns.
- **Posterior-targeted placement is not a measured policy.** Study 2 replayed
  uniform, random, front-loaded, back-loaded, audit-early and an onset oracle.
  When the posterior is informative the strip places checks at posterior-mass
  quantiles and shows no miss rate for that schedule; the measured uniform and
  onset-oracle rows bracket it. Only the flat-posterior branch carries a
  replayed number.
- **Posterior mass after the last check is geometry, not a measurement.** It is
  computed from the posterior currently displayed and is labelled as such.
- **Fault speed alone does not determine the regime.** Cells that capture at the
  same median round fall into different regimes, so the banner shows the regime
  distribution over matching cells and says when the estimate is not enough.
- **The regime boundaries are properties of these two substrates.** They are
  quantitative here and qualitative elsewhere; what transfers is the
  three-regime structure, the onset dependence and the metric inversion. A
  deployment re-measures its own boundaries with the released replay tooling.
- **The whole monitor zoo fails the deployment test.** No variant exceeds 0.14
  timely recall at the deployed operating point. The scoreboard orders
  survivors, not solutions.
