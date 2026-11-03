# PREREG — Paper 1: Decision-Time Bounds on Agent Oversight Interfaces (v1.0)

Status: to be FROZEN (SHA-256 sidecar) before any new data collection. Target: ACM IUI 2027, full paper 2026-08-20 AoE. Working title: "Looking in the Right Place, at the Right Time: Decision-Time Bounds on Agent Oversight Interfaces".

## 0. Framing constraints (binding)

- All bounds are **oracle-overseer bounds**: a perfect reader of everything the monitoring surface exposes. No claims about human psychology, attention, vigilance, or fatigue. LLM panels model **agent populations under oversight**, never simulated humans.
- Vocabulary: "check placement", "attention budget", "regime map". Never "phase diagram", "probe density × budget" (in-review adjacency), "challenge budget". Disjoint from Paper 2's triage/routing/queue vocabulary.
- Evidence class per CFP taxonomy: computational analysis + system evaluation (replay-evaluated reference implementation). Simulation is the instrument of choice because placement × budget × regime sweeps are unsweepable by user studies (TiiS 2024 QA4ML argument).

## 1. Formal model (check semantics — the paper's Def. 1)

An agent execution runs t = 1..T with latent fault onset. A **check at time t** reveals exactly the causal prefix information I_t available at t (never future rounds). A fired check triggers the intervention available at t. Intervention is **effective iff t ≤ t\***, the last timely time (fault-specific). A **placement policy** π maps (budget B, T, optionally cheap online signals) to check times; B is the attention budget. Outcomes: **timely-catch rate** (fire at t ≤ t\* on faulty runs), **false-intervention rate** (fire on clean runs / pre-onset), **penalized delay**. Hindsight metrics (post-hoc AUC over completed transcripts) are the foil throughout.

## 2. Substrates (all local, all existing engines; originals never modified — schedule-capable copies live in our repo)

### S1 — Multi-agent belief capture (irreversible regime; scripted, CPU)
Engine: `collective_belief_benchmark` scripted arm (deterministic, sha256-stable seeds). **Capture-speed continuum** (de-circularization): committed count {1,2,3,4,6} of N=32 (top-degree placement) × evidence strength {0.58, 0.66, 0.74} × topology {hub, random, small_world, clustered}, T=15 rounds, balanced private evidence, adversary fabricated_self_support. Fault onset = committed agents active from round 1; t\* = capture round C (first round false-fraction ≥ 0.8 among non-committed); clean runs = committed count 0 arms for false-intervention rates.
Check at round t: compute content-anomaly scores (`anomaly_scores`, θ=0.5 fixed as the prior benchmark's untuned default) on the causal message prefix; prune flagged senders from that round onward.
Policies (B ∈ {1, 2, 3} checks): uniform-spaced; seeded random; front-loaded (earliest B rounds ≥ 2); back-loaded (last B rounds); front-weighted geometric (AI-control-style audit-early baseline); suspicion-triggered (cheap passive signal — round-mean confidence shift ≥ registered threshold — spends a check when triggered, else per fallback uniform); oracle (checks at rounds C−B+1..C; upper bound). Boundary-aligned placement is **not defined** in S1 (no subtask boundaries) — stated in paper.
Seeds: 50 per cell. Volume ≈ 4×3×5×8 policies×3 budgets×50 ≈ 72k trials + no-defense/clean arms; CPU minutes-to-tens-of-minutes via multiprocessing.

### S2 — Latent rule-shift streams (detectable regime; symbolic, CPU-deterministic)
Engine: schedule-capable copy of `simulate_symbolic_monitor` (fixed-interval condition replaced by `step in schedule`; everything else byte-preserved: 0.995 observation accuracy, posterior softening 0.12, alarm rule best≠source ∧ posterior<0.25 after warmup 10). The 10 established cells (ρ 0.0–0.71) × {well-, mis-}specified; within-check probe battery = 8 probes, EIG-adaptive selection (established); **shift step τ randomized per seed, uniform [40, 120]** on a 160-step stream (registered de-trivialization — placement must hedge over unknown onset). t\* = end of stream; effectiveness decays via penalized delay (miss = remaining-horizon penalty).
Policies (B ∈ {1, 2, 4, 8, 16} checks): uniform; random; front-loaded; back-loaded; suspicion-triggered (passive confidence/entropy drift detector from `detectors.py` triggers the check); oracle-at-τ. Seeds: 24 (7..261 stride 11, the established list). False alarms recorded separately (full return dict; the prior pipeline's folding of FA into miss is repaired here and noted).

### S3 — Dissociation premise, self-contained on two current families (LM Studio)
Panels replicate the CLAIM/CONFIDENCE/EVIDENCE protocol byte-identically (system+user prompts, parse regexes, θ=0.5, P=0.70, temperature 0.7) with generation via LM Studio (models: qwen/qwen3.5-9b, google/gemma-4-26b-a4b; max_new_tokens raised to fit reasoning-style output, registered as operational parameter; parse retry once). Cells: N ∈ {5,7} × {plain, jitter} × nodef, T=5, seeds {11,22,33,44,55} → 20 trials/family/adversary-arm, ≥15-captured kill bar per family. Metrics via the causal `scores_at_decision_round` semantics: post-hoc AUC (t=6) vs decision-time recall/AUC at t ≤ C.

### S4 — Monitor zoo for metric inversion (pure replay, CPU)
≥12 monitor variants, each a deterministic function over causal prefixes, replayed over existing Qwen2.5-7B/Ministral-8B panel logs + new S3 logs: full anomaly score; s1-only (provenance); s2-only (behaviour); s3-only (divergence); 4 reweightings (0.6/0.2/0.2 permutations + published reweight grid); confidence-mean threshold; claim-constancy count; round-1 confidence gate; first-round-echo heuristic. Each ranked by (a) post-hoc transcript AUC, (b) timely recall at t ≤ C, (c) penalized delay.

## 3. Hypotheses (registered directions, subjective P(pass), exits)

- **H1 (dissociation, self-contained)**: each new family shows post-hoc AUC ≥ 0.85 while timely recall at t ≤ C = 0 in FAST cells (capture at round 2). [P=.65. EXIT if a family shows nonzero timely recall: reported as a **boundary finding** — the ceiling is population-dependent — and the regime map absorbs it as a slower-capture regime; paper claim softens from "ceiling binds current families" to "ceiling binds capture-speed regimes", carried by S1.]
- **H2 (placement dominates budget in window regimes)**: restricted to S1 cells with median C ∈ [3, 10] (window exists), best non-oracle placement at B=1 achieves higher timely-catch than uniform at B=2; exact one-sided sign test across cells, Holm over the two budget contrasts. [P=.6. EXIT: report the measured ordering; the regime map remains the headline regardless of which policy wins.]
- **H3 (regime map tri-partition)**: the S1 map contains all three registered regions — **ceiling-bound** (median C ≤ 2: no policy incl. oracle prevents false consensus), **placement-critical** (best placement at B=1 succeeds ≥ 2× uniform B=1 rate), **budget-saturated** (uniform B ≥ 2 within 5pp of oracle) — each covering ≥ 10% of cells, stable across ≥ 3 of 4 topologies. [P=.7]
- **H4 (metric inversion)**: Kendall τ between hindsight-AUC ranking and timely-recall ranking of the ≥12-monitor zoo ≤ 0.6, bootstrap 95% CI (trial-level resampling) excluding 0.8. [P=.65. EXIT: if rankings agree (τ > 0.8), the honest finding is "hindsight AUC is a safe proxy in this class" — reported; C3 demotes to a robustness section and the paper carries on H1-H3.]
- **H5 (suspicion-triggering is regime- and specification-sensitive)**: in S2 high-ρ (≥0.25) well-specified cells, suspicion-triggered beats uniform at B ≤ 4 (lower penalized delay, sign test); under misspecification the advantage reverses or vanishes in ≥ half of those cells. [P=.55]

## 4. Statistics

Sign tests exact one-sided with registered directions; Holm within hypothesis families; bootstrap 10⁴ percentile CIs (trial/seed-level clustering); Clopper-Pearson for small-n panel rates (shown, per house rule, as existence evidence — statistical weight rides on S1/S2). All numbers script-emitted to results/*.json; verify_p1_numbers.py asserts every paper number; XX...XX placeholders until backfilled. rng seed 20260810 for all new randomization.

## 5. Artifact (system-evaluation leg)

Reference implementation: an attention-routing monitor view driven by decision-time metrics (timely-catch projection per candidate check placement, regime annotation, ceiling banner), rendered over replayed trajectories from S1/S3 + the 7 verbatim capture case studies (cited third-person). Evaluated as **policy evaluation by trace replay** at each budget; explicitly not an interface evaluation. 2-day timebox; if it slips, the paper ships design rules + mockup walkthrough and says so.

## 6. Dual-submission hygiene

The reaction-latency ceiling and the diagnosability/probe-selection results are in-review elsewhere by others-to-us (cited third person if public; otherwise established self-containedly here — S3 does not cite them). This paper's claims — check-placement regime map, metric inversion on a monitor zoo, cross-family self-contained dissociation, oversight-interface design rules — overlap none of them. S2 reuses a testbed for a placement sweep only; no diagnosability claims. θ=0.5 and the anomaly score are treated as fixed instruments, not contributions.

## AMENDMENT A1 (2026-08-08, before any S2 analysis; S2 sweep in flight)

S2 policy set deviations from §2-S2, recorded before any S2 result was inspected:
(a) **suspicion-triggered dropped in S2**: the referenced passive detectors are blind by construction on matched streams (the original monitor hardcodes their decision to False in the symbolic path); a suspicion arm would degenerate to "budget never spent". The construction-level blindness is stated in the paper as an instrument property, not tested as a policy. S1 retains its suspicion arm (calibrated cheap signal exists there).
(b) **geometric (audit-early) added to S2** to mirror S1's AI-control-transposition baseline across substrates.
H5 is re-scoped accordingly: the suspicion-vs-uniform contrast is evaluated in S1 only; in S2 the corresponding registered contrast becomes geometric (audit-early) vs uniform under randomized onset, same directions and misspecification reversal clause. Timing note: amendment written after S2 sweep launch but before any S2 output was read; sweep outputs untouched at freeze of this amendment.

## 7. Known limitations (pre-committed wording)

Scripted populations + small open-weight panels model agent-side dynamics, not deployments; θ fixed untuned; anomaly instrument single-family; S2 rule streams are synthetic model systems; oracle-overseer bounds upper-bound any real overseer but do not estimate one.
