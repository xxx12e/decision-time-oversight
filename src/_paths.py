"""Substrate locations, resolved outside the source tree.

Both study substrates live in their own repositories; this project imports them
rather than vendoring them. Their location is deployment-specific, so no module
hard-codes it. Resolution order:

  1. environment variable  -- BELIEF_BENCH_ROOT, RULE_SHIFT_ROOT
  2. ``paths_local.py`` sitting next to this file (not part of the release
     package; copy ``paths_local.example.py`` and fill in your checkout paths)
  3. a clear error naming exactly what to set

Anything that resolves to a real directory is also pushed onto ``sys.path`` by
``add_to_syspath`` so the substrate packages import normally.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

try:                                    # pragma: no cover - deployment-specific
    import paths_local as _local        # noqa: F401
except ImportError:                     # pragma: no cover
    try:
        sys.path.insert(0, str(_HERE))
        import paths_local as _local    # noqa: F401
    except ImportError:
        _local = None


def _resolve(env_var: str, local_attr: str, what: str) -> Path:
    raw = os.environ.get(env_var) or getattr(_local, local_attr, None)
    if not raw:
        raise RuntimeError(
            f"{what} not configured. Set the {env_var} environment variable to "
            f"your checkout of it, or create {_HERE / 'paths_local.py'} from "
            f"paths_local.example.py."
        )
    p = Path(raw)
    if not p.exists():
        raise RuntimeError(f"{env_var} points at {p}, which does not exist.")
    return p


def belief_bench() -> Path:
    """Root of the multi-agent belief-capture benchmark (Studies 1, 3, 4)."""
    return _resolve("BELIEF_BENCH_ROOT", "BELIEF_BENCH_ROOT",
                    "The belief-capture benchmark")


def rule_shift_parent() -> Path:
    """Parent directory of the symbolic rule-shift substrate (Study 2)."""
    return _resolve("RULE_SHIFT_ROOT", "RULE_SHIFT_ROOT",
                    "The rule-shift substrate")


def add_to_syspath(p: Path) -> Path:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return p
