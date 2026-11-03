r"""Runs for protocol supplement S2 (prereg/PROTOCOL_SUPPLEMENT_S2_review_experiments.md).

Arms = detector x latency x adversary on the belief-capture benchmark. The substrate is reused
verbatim: detectors are passed to s1_engine.s1_trial as `instrument`, and the two strategic
adversaries rebind s1_engine._make_public_messages to a wrapper that calls the original first
(so every random draw of the substrate is unchanged) and then overwrites committed agents'
messages with camouflage copies in the rounds the adversary chooses.

Stages (each resumable; outputs in results/s1/review_s2/):
  python -X utf8 review_s2_runs.py calibrate          # SINGLE threshold, suspicion deltas (fault-free only)
  python -X utf8 review_s2_runs.py run --workers 20   # all committed-cell arms + fault-free arms
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s1_engine  # noqa: E402
from s1_engine import (  # noqa: E402
    T_ROUNDS, make_condition, s1_trial, anomaly_scores, THETA,
    sched_uniform, sched_random, sched_front, sched_back, sched_geometric, sched_oracle,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "s1" / "review_s2"
CALIB_F = OUT / "calibration.json"

COUNTS = (1, 2, 3, 4, 6)
EVIDENCE = (0.58, 0.66, 0.74)
TOPOS = ("hub", "random", "small_world", "clustered")
N_SEEDS = 50
BUDGETS = (1, 2, 3)
ROUNDS = tuple(range(2, T_ROUNDS + 1))
ALL_SCHEDS = [frozenset(c) for B in BUDGETS for c in itertools.combinations(ROUNDS, B)]
SCHED_INDEX = {s: i for i, s in enumerate(ALL_SCHEDS)}
assert len(ALL_SCHEDS) == 469
RAND_SEED_BASE = 20260810          # sweep_s1.RAND_SEED_BASE
SINGLE_GRID = [round(0.01 * i, 2) for i in range(101)]
_ORIG_MAKE = s1_engine._make_public_messages


def sched_uniform2(B: int) -> frozenset[int]:
    """uniform-from-2: {2 + round(i*13/B)} -> {2}, {2,8}, {2,6,11}"""
    return frozenset(2 + round(i * (T_ROUNDS - 2) / B) for i in range(B))


assert [sorted(sched_uniform2(B)) for B in BUDGETS] == [[2], [2, 8], [2, 6, 11]]


def library(B: int, t: int, cap) -> dict[str, frozenset]:
    return {"uniform": sched_uniform(B), "uniform2": sched_uniform2(B), "front": sched_front(B),
            "back": sched_back(B), "geometric": sched_geometric(B),
            "random": sched_random(B, RAND_SEED_BASE + t * 100 + B), "oracle": sched_oracle(B, cap)}


# ------------------------------------------------------------------ detectors
class Detector:
    """kind in BENCH / PERFECT / SINGLE; lag in {0, 1}. A check at round t reads rounds 1..t-lag."""

    def __init__(self, kind: str, lag: int, theta_single: float | None = None):
        self.kind, self.lag, self.theta_single = kind, lag, theta_single

    def _visible(self, public_log, t):
        r = t - self.lag
        return r, {s: [m for m in h if m.round_index <= r] for s, h in public_log.items()}

    def single_scores(self, vis):
        out = {}
        for s, h in vis.items():
            if h:
                mc = sum(float(m.confidence) for m in h) / len(h)
                out[int(s)] = min(1.0, max(0.0, (mc - 0.9) / 0.09))
        return out

    def flag(self, public_log, round_index, committed_agents) -> set:
        r, vis = self._visible(public_log, round_index)
        if r < 1:
            return set()
        if self.kind == "PERFECT":
            return {int(s) for s in committed_agents if vis[int(s)]}
        if self.kind == "BENCH":
            sc = anomaly_scores(vis if self.lag else public_log, r)
            return {s for s, v in sc.items() if v >= THETA}
        sc = self.single_scores(vis)
        return {s for s, v in sc.items() if v >= self.theta_single}


class Recorder(Detector):
    """Fault-free calibration: never flags; records the max SINGLE score per (lag, round)."""

    def __init__(self):
        super().__init__("SINGLE", 0)
        self.rec = {0: {}, 1: {}}

    def flag(self, public_log, round_index, committed_agents) -> set:
        for lag in (0, 1):
            r = round_index - lag
            if r < 1:
                continue
            vis = {s: [m for m in h if m.round_index <= r] for s, h in public_log.items()}
            sc = self.single_scores(vis)
            self.rec[lag][round_index] = max(sc.values()) if sc else 0.0
        return set()


# ------------------------------------------------------------------ adversaries
_ADV = {"mode": "STD", "camo_rounds": frozenset()}


def _camo_make(agents, graph, stress, round_index, n_agents):
    outbound, counts = _ORIG_MAKE(agents, graph, stress, round_index, n_agents)
    if round_index in _ADV["camo_rounds"]:
        honest = sorted(int(a.agent_id) for a in agents if not a.is_committed)
        rng = random.Random(f"camo|{stress.pair_id}|{round_index}")
        for a in agents:
            if a.is_committed:
                own = outbound[a.agent_id]
                src = outbound[rng.choice(honest)]
                outbound[a.agent_id] = dataclasses.replace(
                    src, sender_id=own.sender_id, source_group=own.source_group,
                    sender_degree_centrality=own.sender_degree_centrality, evidence_signal=None)
    return outbound, counts


def delay_onset(cell: str, t: int) -> int:
    return random.Random(f"delay|{cell}|{t}").randint(1, 8)


def _set_adv(mode: str, camo: frozenset):
    _ADV["mode"], _ADV["camo_rounds"] = mode, camo
    s1_engine._make_public_messages = _ORIG_MAKE if mode == "STD" or not camo else _camo_make


def camo_rounds(adv: str, sched: frozenset, lag: int, onset: int) -> frozenset:
    if adv == "DELAY":
        return frozenset(range(1, onset))
    if adv == "EVADE":
        return frozenset(t - lag for t in sched if t - lag >= 1)
    return frozenset()


# ------------------------------------------------------------------ workers
def _calib_job(job):
    ev, topo, t = job
    _set_adv("STD", frozenset())
    rec = Recorder()
    base = s1_trial(make_condition(0, ev, topo), t, frozenset(range(1, T_ROUNDS + 1)), instrument=rec)
    return {"ev": ev, "topo": topo, "t": t, "rec": {str(k): v for k, v in rec.rec.items()},
            "mean_conf_by_round": base["mean_conf_by_round"]}


def _pair_job(job):
    arm, cc, ev, topo, t, theta_single, deltas = job
    det_kind, lagname, adv = arm.split("_")
    lag = int(lagname[1])
    cell = f"{cc}|{ev}|{topo}"
    onset = delay_onset(cell, t) if adv == "DELAY" else 1
    cond = make_condition(cc, ev, topo)
    det = Detector(det_kind, lag, theta_single)

    _set_adv(adv, camo_rounds(adv, frozenset(), lag, onset))   # baseline: no checks
    base = s1_trial(cond, t, frozenset())
    cap = base["capture_round"]
    row = {"arm": arm, "cell": cell, "trial": t, "onset": onset, "capture_round": cap,
           "base_fc": bool(base["false_consensus"])}

    enumerate_all = det_kind in ("BENCH", "SINGLE") and adv in ("STD", "DELAY")
    scheds = list(ALL_SCHEDS) if enumerate_all else [frozenset({r}) for r in ROUNDS]
    extra = {}
    for B in BUDGETS:
        for name, s in library(B, t, cap).items():
            if s not in SCHED_INDEX or not enumerate_all:
                extra[f"{name}@B{B}"] = s
    if det_kind == "PERFECT" and lag == 0:
        extra["round1@B1"] = frozenset({1})

    def one(s, known=True):
        # EVADE cannot see the draw of the random policy: it pushes normally there
        _set_adv(adv, camo_rounds(adv, s, lag, onset) if known else frozenset())
        r = s1_trial(cond, t, s, instrument=det)
        return (int(bool(r["false_consensus"])), int(bool(r["timely_committed_prune"])),
                int(bool(r["prune_events"])), [list(e) for e in r["prune_events"]])

    if enumerate_all:
        res = [one(s) for s in scheds]
        row["fc"] = "".join(str(x[0]) for x in res)
        row["tc"] = "".join(str(x[1]) for x in res)
        row["iv"] = "".join(str(x[2]) for x in res)
    else:
        row["singles"] = {}
        for s in scheds:
            fc, tc, iv, _ = one(s)
            row["singles"][str(min(s))] = [fc, tc, iv]
    row["library"] = {}
    for key, s in extra.items():
        fc, tc, iv, pe = one(s, known=not key.startswith("random@"))
        row["library"][key] = [fc, tc, iv, sorted(s), pe if det_kind == "PERFECT" else None]
    # suspicion (online; EVADE cannot anticipate an online trigger, so it is not run there)
    if adv != "EVADE":
        from strict_regime_map import StrictSuspicionPolicy
        _set_adv(adv, camo_rounds(adv, frozenset(), lag, onset))
        for B in BUDGETS:
            pol = (StrictSuspicionPolicy if lag else s1_engine.SuspicionPolicy)(B, deltas[lag][f"{topo}|{ev}"])
            r = s1_trial(cond, t, frozenset(), policy=pol, instrument=det)
            row["library"][f"suspicion@B{B}"] = [int(bool(r["false_consensus"])),
                                                 int(bool(r["timely_committed_prune"])),
                                                 int(bool(r["prune_events"])), r["checked_rounds"], None]
    return row


def _clean_job(job):
    det_kind, lag, ev, topo, t, theta_single, deltas = job
    det = Detector(det_kind, lag, theta_single)
    cond = make_condition(0, ev, topo)
    _set_adv("STD", frozenset())
    row = {"det": det_kind, "lag": lag, "cell": f"0|{ev}|{topo}", "trial": t, "singles": {}, "library": {}}
    for r in ROUNDS:
        row["singles"][str(r)] = int(bool(s1_trial(cond, t, frozenset({r}), instrument=det)["prune_events"]))
    for B in BUDGETS:
        for name, s in library(B, t, None).items():
            if name != "oracle":
                row["library"][f"{name}@B{B}"] = int(bool(s1_trial(cond, t, s, instrument=det)["prune_events"]))
        from strict_regime_map import StrictSuspicionPolicy
        pol = (StrictSuspicionPolicy if lag else s1_engine.SuspicionPolicy)(B, deltas[lag][f"{topo}|{ev}"])
        row["library"][f"suspicion@B{B}"] = int(bool(
            s1_trial(cond, t, frozenset(), policy=pol, instrument=det)["prune_events"]))
    return row


# ------------------------------------------------------------------ stages
def calibrate(workers: int):
    jobs = [(ev, topo, t) for ev in EVIDENCE for topo in TOPOS for t in range(N_SEEDS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_calib_job, jobs, chunksize=4))
    out = {"n_runs": len(rows), "theta_single": {}, "fi_by_theta": {}}
    for lag in (0, 1):
        def fi(th):
            return sum(sum(1 for row in rows if row["rec"][str(lag)].get(r, 0.0) >= th) / len(rows)
                       for r in ROUNDS) / len(ROUNDS)
        curve = {str(th): fi(th) for th in SINGLE_GRID}
        out["fi_by_theta"][str(lag)] = curve
        out["theta_single"][str(lag)] = next(th for th in SINGLE_GRID if curve[str(th)] <= 0.02)
    # suspicion deltas: verbatim strict_regime_map recipe (95th pct of clean max shifts)
    shifts = {0: {}, 1: {}}
    for row in rows:
        mc = row["mean_conf_by_round"]
        k = f"{row['topo']}|{row['ev']}"
        shifts[0].setdefault(k, []).append(max(m - mc[0] for m in mc))
        shifts[1].setdefault(k, []).append(max(m - mc[0] for m in mc[:T_ROUNDS - 1]))
    out["suspicion_deltas"] = {str(lag): {k: sorted(v)[min(len(v) - 1, int(0.95 * len(v)))]
                                          for k, v in shifts[lag].items()} for lag in (0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    CALIB_F.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("theta_single:", out["theta_single"])


ARMS = (["BENCH_L0_STD", "BENCH_L1_STD", "SINGLE_L0_STD", "SINGLE_L1_STD",
         "BENCH_L0_DELAY", "BENCH_L1_DELAY",
         "PERFECT_L0_STD", "PERFECT_L1_STD", "BENCH_L0_EVADE", "BENCH_L1_EVADE"])


def run(workers: int, arms):
    cal = json.loads(CALIB_F.read_text(encoding="utf-8"))
    deltas = {int(k): v for k, v in cal["suspicion_deltas"].items()}
    th = {int(k): v for k, v in cal["theta_single"].items()}
    t0 = time.time()
    # fault-free arms first (cheap)
    cf = OUT / "clean.jsonl"
    if not cf.exists():
        jobs = [(d, lag, ev, topo, t, th[lag], deltas) for d in ("BENCH", "SINGLE", "PERFECT")
                for lag in (0, 1) for ev in EVIDENCE for topo in TOPOS for t in range(N_SEEDS)]
        with ProcessPoolExecutor(max_workers=workers) as ex, open(cf, "w", encoding="utf-8") as fh:
            for r in ex.map(_clean_job, jobs, chunksize=8):
                fh.write(json.dumps(r) + "\n")
        print(f"clean arms done ({time.time() - t0:.0f}s)", flush=True)
    for arm in arms:
        f = OUT / f"{arm}.jsonl"
        done = set()
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["cell"], r["trial"]))
        lag = int(arm.split("_")[1][1])
        jobs = [(arm, cc, ev, topo, t, th[lag], deltas) for cc in COUNTS for ev in EVIDENCE
                for topo in TOPOS for t in range(N_SEEDS) if (f"{cc}|{ev}|{topo}", t) not in done]
        print(f"{arm}: {len(jobs)} pairs to run ({len(done)} cached)", flush=True)
        ta = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex, open(f, "a", encoding="utf-8") as fh:
            for i, r in enumerate(ex.map(_pair_job, jobs, chunksize=2)):
                fh.write(json.dumps(r) + "\n")
                if (i + 1) % 250 == 0:
                    fh.flush()
                    el = time.time() - ta
                    print(f"  {arm} {i + 1}/{len(jobs)} ({el / 60:.1f} min, "
                          f"ETA {el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min)", flush=True)
    print(f"all arms done ({(time.time() - t0) / 60:.0f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("calibrate", "run", "probe"))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    if a.stage == "calibrate":
        calibrate(a.workers)
    elif a.stage == "probe":
        cal = json.loads(CALIB_F.read_text(encoding="utf-8"))
        deltas = {int(k): v for k, v in cal["suspicion_deltas"].items()}
        th = {int(k): v for k, v in cal["theta_single"].items()}
        for arm in a.arms.split(","):
            lag = int(arm.split("_")[1][1])
            tp = time.time()
            r = _pair_job((arm, 3, 0.66, "random", 0, th[lag], deltas))
            print(arm, f"{time.time() - tp:.1f}s", r["capture_round"], r["onset"],
                  {k: v[:3] for k, v in list(r["library"].items())[:4]})
    else:
        run(a.workers, a.arms.split(","))


if __name__ == "__main__":
    main()
