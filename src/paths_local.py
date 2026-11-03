"""Substrate locations for this repository (vendored under ../substrates/).

Environment variables of the same name take precedence over these values.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "substrates"

# Root of the multi-agent belief-capture benchmark (Studies 1, 3, 4).
BELIEF_BENCH_ROOT = str(_ROOT / "belief_bench")

# Parent directory containing the symbolic rule-shift substrate (Study 2).
RULE_SHIFT_ROOT = str(_ROOT / "rule_shift")
