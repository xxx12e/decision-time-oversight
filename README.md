# Decision-time bounds on checking LLM agents

Code, data, and preregistration for the manuscript
*Looking in the Right Place, at the Right Time: Decision-Time Bounds on Checking LLM Agents*
(Haoyang Li, Irvine Valley College; under review).

An overseer of a multi-agent run can afford only a few checks, and a check helps only before an
irreversible deadline. This repository contains everything needed to recompute every number in
the paper from the result files, and to regenerate the result files themselves.

## Layout

| Path | Contents |
|---|---|
| `src/` | Experiment and analysis code. `S1` = belief capture (Sections 4–6), `S2` = rule-shift streams (Section 7), `S3` = LLM-agent panels and `S4` = the monitor comparison (Section 8); `review_s2_*` = the detector, latency, and adversary arms of protocol supplement S2 (Sections 4–6) |
| `results/` | Every result file the paper reports from (`s1/`–`s4/`) |
| `scripts/verify_p1_numbers.py` | Checks every number in the manuscript against `results/` (429 pins) |
| `jaamas/` | Manuscript source (the verification script reads it) |
| `figures/` | `make_figs.py` draws every figure from `results/`; `figures/final/` holds them as printed |
| `prereg/` | Preregistration: frozen original with its SHA-256 sidecars, the amendment hash, the release copy, and the two protocol supplements (capture threshold; detector, latency, and adversary arms) |
| `artifact/` | Reference monitoring view (regime banner, placement strip, decision-time scoreboard) with a self-check |
| `substrates/` | The parts of the two simulation substrates this paper imports (`collective_belief_benchmark`, `semantic_rule_shift`), their configs, and the eight archived panel transcripts replayed in S4 |

## Setup

Python 3.12. Install the dependencies:

```bash
pip install -r requirements.txt
```

`src/paths_local.py` points at the vendored `substrates/`; the environment variables
`BELIEF_BENCH_ROOT` and `RULE_SHIFT_ROOT` override it.

## Check the paper's numbers

```bash
P1_TEX=jaamas/main.tex python -X utf8 scripts/verify_p1_numbers.py --strict
```

Expected: `PASS 429 pins`.

## Regenerate results

All simulations are CPU-only and seeded; reruns reproduce the stored files (row order in the
`.jsonl` files can differ between parallel runs, the rows do not).

```bash
cd src
python -X utf8 test_s1_mock.py && python -X utf8 test_s2_mock.py   # engine gates
python -X utf8 sweep_s1.py            # S1 sweep -> results/s1/trials.jsonl
python -X utf8 true_oracle_s1.py      # exhaustive placement oracle (469 schedules per pair)
python -X utf8 stats_s1.py            # S1 statistics
python -X utf8 sweep_s2.py            # S2 sweep -> results/s2/trials.jsonl
python -X utf8 analyze_s2.py
python -X utf8 s3_metrics.py qwen35 && python -X utf8 s3_metrics.py gemma4
python -X utf8 s4_zoo.py --include-s3 qwen35,gemma4
python -X utf8 review_s2_runs.py calibrate && python -X utf8 review_s2_runs.py run   # supplement S2 arms
python -X utf8 review_s2_analyze.py && python -X utf8 review_s2_window_details.py
```

Each remaining script in `src/` produces one supplementary analysis; its docstring names its
output file. `s3_panels.py` collected the S3 transcripts from local models served through
LM Studio (Qwen3.5-9B, gemma-4-26B-A4B); rerunning it needs those models, while
`results/s3/panels_*.json` holds the transcripts used in the paper.

Figures:

```bash
python -X utf8 figures/make_figs.py
```

Monitoring view:

```bash
python -X utf8 artifact/monitoring_view.py --selfcheck
streamlit run artifact/monitoring_view.py
```

## License

MIT (see `LICENSE`).
