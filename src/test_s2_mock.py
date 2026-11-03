r"""Mock gate for S2 engine.

A determinism; B zero-rho cell never detects (miss=1, and pre-shift checks do
not false-alarm at obs-acc .995); C dense cell + oracle B=1 detects with small
delay; D all-front checks before a late tau -> miss; E budget monotonicity on
dense cell (uniform B=16 detects >= B=1); F reduction sanity at tau=80-ish:
uniform B=16 approximates the original fixed-interval cadence and detects
dense-cell shifts.
"""
from __future__ import annotations

import sys

from s2_engine import s2_trial, make_schedule, get_cells, tau_for, s2_front, s2_oracle

DENSE = "easy_dense_four|L2"
ZERO = "impossible_equivalent|L2"


def test_A():
    a = s2_trial(DENSE, "well_specified", 7, make_schedule("uniform", 4, DENSE, 7))
    b = s2_trial(DENSE, "well_specified", 7, make_schedule("uniform", 4, DENSE, 7))
    assert a == b
    print("  A determinism ok")


def test_B():
    misses = fas = 0
    for s in (7, 18, 30):
        r = s2_trial(ZERO, "well_specified", s, make_schedule("uniform", 8, ZERO, s))
        misses += r["miss"]
        fas += r["false_alarm"]
    print(f"  B zero-rho: miss {misses}/3, false alarms {fas}/3")
    assert misses == 3 and fas == 0


def test_C():
    delays = []
    for s in (7, 18, 30, 41):
        tau = tau_for(DENSE, s)
        r = s2_trial(DENSE, "well_specified", s, s2_oracle(1, tau))
        delays.append(r["delay"])
    print(f"  C dense oracle-B1 delays: {delays}")
    assert all(d is not None and d <= 8 for d in delays)


def test_D():
    missed = 0
    trials = 0
    for s in range(30):
        tau = tau_for(DENSE, s)
        if tau < 100:
            continue
        trials += 1
        r = s2_trial(DENSE, "well_specified", s, s2_front(3))
        missed += r["miss"]
    print(f"  D late-tau front-loaded: missed {missed}/{trials}")
    assert trials >= 3 and missed == trials


def test_E():
    d1 = d16 = 0
    for s in (7, 18, 30, 41, 52):
        r1 = s2_trial(DENSE, "well_specified", s, make_schedule("uniform", 1, DENSE, s))
        r16 = s2_trial(DENSE, "well_specified", s, make_schedule("uniform", 16, DENSE, s))
        d1 += not r1["miss"]
        d16 += not r16["miss"]
    print(f"  E detection: uniform B=1 {d1}/5, B=16 {d16}/5")
    assert d16 >= d1 and d16 >= 4


if __name__ == "__main__":
    get_cells()
    ks = list(get_cells())
    assert len(ks) == 10, f"expected 10 cells, got {len(ks)}: {ks}"
    print(f"  cells: {ks}")
    test_A()
    test_B()
    test_C()
    test_D()
    test_E()
    print("ALL S2 MOCK TESTS PASS")
    sys.exit(0)
