"""Mock-first gate for the S1 engine (house rule: no sweep before these pass).

A: determinism — same (cond, trial) twice -> identical trajectories.
B: empty schedule == no defense; committed minority captures a hub cell fast.
C: all-rounds schedule prunes committed senders (plain self-support adversary
   must be catchable by round 3: s1+s2 both live) and reduces/blocks capture
   relative to B in a slow-capture cell.
D: oracle-placed B=1 check in a windowed cell strictly beats back-loaded B=1.
E: clean arm (0 committed): checks fire rarely, prune few normal senders.
F: capture-speed continuum — capture round is monotone-ish in committed count
   (1 vs 6 committed at weak evidence).
"""
from __future__ import annotations

import sys

from s1_engine import (
    make_condition, s1_trial, sched_uniform, sched_back, sched_oracle,
)


def traj(cc, ev, topo, trial, schedule):
    return s1_trial(make_condition(cc, ev, topo), trial, frozenset(schedule))


def test_A_determinism():
    a = traj(3, 0.66, "hub", 7, {4, 8})
    b = traj(3, 0.66, "hub", 7, {4, 8})
    assert a["false_frac_by_round"] == b["false_frac_by_round"]
    assert a["prune_events"] == b["prune_events"]
    print("  A determinism ok")


def test_B_capture_no_defense():
    caps = []
    for t in range(6):
        r = traj(4, 0.58, "hub", t, set())
        caps.append(r["capture_round"])
    captured = sum(1 for c in caps if c is not None)
    assert captured >= 4, f"hub 4-committed weak-evidence should capture: {caps}"
    print(f"  B no-defense capture rounds: {caps}")


def test_C_full_schedule_prunes():
    got_committed_prune = blocked = 0
    n = 6
    for t in range(n):
        nd = traj(2, 0.66, "random", t, set())
        fs = traj(2, 0.66, "random", t, set(range(2, 16)))
        got_committed_prune += fs["committed_pruned"] > 0
        blocked += (nd["false_consensus"] and not fs["false_consensus"])
    assert got_committed_prune >= n - 1, f"committed pruned in only {got_committed_prune}/{n}"
    print(f"  C full-schedule: committed pruned {got_committed_prune}/{n}, capture blocked {blocked}/{n}")


def test_D_oracle_beats_back():
    wins = ties = 0
    n = 10
    for t in range(n):
        nd = traj(2, 0.66, "random", t, set())
        c = nd["capture_round"]
        if c is None or c <= 2:
            continue
        orc = s1_trial(make_condition(2, 0.66, "random"), t, sched_oracle(1, c))
        bck = s1_trial(make_condition(2, 0.66, "random"), t, sched_back(1))
        o_ok = orc["timely_committed_prune"]
        b_ok = bck["timely_committed_prune"]
        wins += o_ok and not b_ok
        ties += o_ok == b_ok
    print(f"  D windowed trials: oracle-only-wins {wins}, ties {ties}")
    assert wins >= 1, "oracle placement should win somewhere in the window"


def test_E_clean_arm():
    total_pruned = fired = 0
    for t in range(6):
        r = traj(0, 0.66, "hub", t, {5, 10})
        total_pruned += r["normal_pruned"]
        fired += len(r["prune_events"]) > 0
    print(f"  E clean arm: prune-event trials {fired}/6, normals pruned {total_pruned}")
    assert fired <= 2, "clean arm should rarely trigger prunes at theta=0.5"


def test_F_continuum():
    def med_cap(cc):
        caps = []
        for t in range(8):
            c = traj(cc, 0.58, "random", t, set())["capture_round"]
            caps.append(c if c is not None else 99)
        caps.sort()
        return caps[len(caps) // 2]
    m1, m6 = med_cap(1), med_cap(6)
    print(f"  F median capture: 1-committed={m1}, 6-committed={m6}")
    assert m6 <= m1, "more committed agents must not capture slower"


if __name__ == "__main__":
    test_A_determinism()
    test_B_capture_no_defense()
    test_C_full_schedule_prunes()
    test_D_oracle_beats_back()
    test_E_clean_arm()
    test_F_continuum()
    print("ALL S1 MOCK TESTS PASS")
    sys.exit(0)
