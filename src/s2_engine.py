r"""S2 engine — check-placement sweep on latent rule-shift streams.

Faithful schedule-parameterized variant of v4.simulate_symbolic_monitor:
same formula pools, EIG probe battery (8 probes/check), Bayesian posterior
(obs acc 0.995), softening 0.12, alarm rule (step>=10, best!=source,
posterior[source]<0.25). Changes: (1) checks fire at scheduled steps instead of
every probe_interval; (2) shift step tau randomized per (cell, seed) in [40,120]
on a 160-step stream, matched inputs via stream[t] = base[t % tau].
Originals never modified.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from _paths import rule_shift_parent, add_to_syspath     # noqa: E402

SRS_PARENT = add_to_syspath(rule_shift_parent())

from semantic_rule_shift import v3, v4  # noqa: E402
from semantic_rule_shift import v7_oracle_free as v7  # noqa: E402

L_TOTAL = 160
TAU_LO, TAU_HI = 40, 120
PROBES_PER_CHECK = 8

_cells_cache: dict | None = None


def get_cells() -> dict[str, dict]:
    """10 established cells keyed by 'label|lang' (built once per process)."""
    global _cells_cache
    if _cells_cache is None:
        pools = v4.build_formula_pools(max_depth=2)
        cells = v7.regime_cells(pools) + v7.discover_low_density_cells(pools, want=4)
        _cells_cache = {f"{c['label']}|{c['lang']}": c for c in cells}
    return _cells_cache


def tau_for(cell_key: str, seed: int) -> int:
    return random.Random(f"tau|{cell_key}|{seed}").randint(TAU_LO, TAU_HI)


def s2_trial(cell_key: str, condition: str, seed: int, schedule: frozenset[int]) -> dict:
    cell = v7.restrict(get_cells()[cell_key], condition)
    cand = cell["candidate_names_full"]
    formulas, vectors = cell["formulas"], cell["vectors"]
    source, target = cell["source"], cell["target"]
    tau = tau_for(cell_key, seed)

    rng = random.Random(seed + v4.stable_offset(source, target, "sched", str(sorted(schedule)), str(len(formulas))))
    base = [rng.randrange(len(formulas)) for _ in range(tau)]
    posterior = {n: 1.0 / len(cand) for n in cand}
    used: set[int] = set()
    probes_used = 0
    alarm = None

    for t in range(L_TOTAL):
        true_name = source if t < tau else target
        if t in schedule:
            posterior = v4._soften_posterior(posterior, cand, mixing=0.12)
            for _ in range(PROBES_PER_CHECK):
                fi = v3.information_gain_probe(cand, formulas, vectors, posterior, used)
                used.add(fi)
                probes_used += 1
                posterior = v3.update_bayesian_posterior(
                    cand, vectors, posterior, fi, vectors[true_name][fi])
        best = max(posterior, key=posterior.get)
        if alarm is None and t >= 10 and best != source and posterior[source] < 0.25:
            alarm = t

    post_alarm = alarm is not None and alarm >= tau
    delay = (alarm - tau) if post_alarm else None
    horizon = L_TOTAL - tau
    return {
        "cell": cell_key, "condition": condition, "seed": seed, "tau": tau,
        "rho": cell["rho"], "region": cell["region"],
        "schedule": sorted(schedule), "alarm": alarm,
        "delay": delay, "miss": delay is None,
        "false_alarm": alarm is not None and alarm < tau,
        "penalized_delay": float(delay) if delay is not None else float(horizon),
        "horizon": horizon, "probes_used": probes_used,
    }


# ---------------- schedules ----------------

def s2_uniform(B: int) -> frozenset[int]:
    return frozenset(round(i * L_TOTAL / (B + 1)) for i in range(1, B + 1))


def s2_random(B: int, seed: int) -> frozenset[int]:
    return frozenset(random.Random(seed).sample(range(2, L_TOTAL), B))


def s2_front(B: int) -> frozenset[int]:
    return frozenset(range(10, 10 + B))


def s2_back(B: int) -> frozenset[int]:
    return frozenset(range(L_TOTAL - B, L_TOTAL))


def s2_geometric(B: int) -> frozenset[int]:
    return frozenset(min(L_TOTAL - 1, 2 ** (i + 1)) for i in range(B))


def s2_oracle(B: int, tau: int) -> frozenset[int]:
    return frozenset(min(L_TOTAL - 1, tau + i) for i in range(B))


S2_POLICIES = ("uniform", "random", "front", "back", "geometric", "oracle")


def make_schedule(policy: str, B: int, cell_key: str, seed: int) -> frozenset[int]:
    if policy == "uniform":
        return s2_uniform(B)
    if policy == "random":
        return s2_random(B, 20260810 + seed * 1000 + B)
    if policy == "front":
        return s2_front(B)
    if policy == "back":
        return s2_back(B)
    if policy == "geometric":
        return s2_geometric(B)
    if policy == "oracle":
        return s2_oracle(B, tau_for(cell_key, seed))
    raise ValueError(policy)
