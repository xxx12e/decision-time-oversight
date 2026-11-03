# Protocol supplement S1-CT: capture-threshold sensitivity (written before running, 2026-10-29)

Motivation: a reviewer asked whether the Study 1 regime map and the placement-vs-budget
contrast depend on the registered capture threshold (capture = at least 80% of honest
agents adopt the false claim).

Design. Re-run the full Study 1 sweep (60 committed cells x 50 seeds, six non-suspicion
policies x B in {1,2,3}, plus the no-check baseline that defines each cell's capture round)
with the capture threshold set to 0.60, 0.70, 0.80 and 0.90. Everything else is unchanged:
deployed anomaly score at theta = 0.50, intercept timing convention, registered regime rules,
windowed-cell definition (median capture round in [3, 10], recomputed per threshold because
the capture round moves with the threshold), and the H2 contrast (per-cell best single check
vs uniform at B = 2, sign test over windowed cells).

Gate. The 0.80 arm must reproduce results/s1/stats.json label for label and both H2 sign
tests exactly; otherwise no other arm is reported.

Reporting. All four arms are reported in full (regime counts, number of windowed cells, H2
timely and prevention contrasts), whatever they show. No arm is dropped and no subset of cells
is re-selected after seeing results.
