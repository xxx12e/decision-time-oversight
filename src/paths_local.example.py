"""Copy to ``paths_local.py`` and point these at your own checkouts.

``paths_local.py`` is deployment-specific and is excluded from the release
package; the environment variables of the same name override it.
"""

# Root of the multi-agent belief-capture benchmark (Studies 1, 3, 4).
BELIEF_BENCH_ROOT = r"/path/to/collective-belief-benchmark"

# Parent directory containing the symbolic rule-shift substrate (Study 2).
RULE_SHIFT_ROOT = r"/path/to/rule-shift-substrate"
