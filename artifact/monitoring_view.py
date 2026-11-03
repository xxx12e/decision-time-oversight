r"""monitoring_view.py -- the attention-routing reference monitoring view.

Three elements, each a direct rendering of a measured bound:

  1. regime banner          which of the three honest messages applies, and
                            what routing attention by regime costs and keeps
  2. placement strip        the chosen policy's check times against an onset
                            posterior, with the registered flat-posterior
                            fallback to uniform placement
  3. decision-time board    the thirteen monitor variants ranked by timely
                            recall instead of transcript AUC

Every bound displayed here is an ORACLE-PLACEMENT bound: the detection
instrument is held fixed at the deployed detector, and the oracle optimizes
only where and when it fires. Whatever the best-placed schedule of that
instrument cannot catch, no schedule of it can -- the bound caps checking
schedules of the deployed instrument, not readers whose judgment beats the
instrument; that axis is measured by the Study-4 monitor zoo, not bounded.
The view makes no claim about human operators -- it does not model one,
measure one, or assume one.

Usage:
    streamlit run monitoring_view.py                  interactive view
    python -X utf8 monitoring_view.py --selfcheck     replay assertions
    python -X utf8 monitoring_view.py --demo-fault    proves the assertions bite
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import replay_core as rc          # noqa: E402
import selfcheck as sc            # noqa: E402

PAPER_TEX = _HERE.parent / "paper" / "main.tex"

# Palette chosen to read on light and dark backgrounds alike; structural
# strokes and all body text inherit the host theme through currentColor.
ACCENT = "#5b9dff"
GOOD = "#3fb950"
WARN = "#d29922"
STOP = "#f85149"
MUTED = "#8a8f98"

REGIME_COLOR = {
    "placement_critical": ACCENT,
    "ceiling_bound": STOP,
    "budget_saturated": GOOD,
    "benign": MUTED,
    "intermediate": WARN,
}


# ----------------------------------------------------------------- format

def pct(x: float | None, digits: int = 0) -> str:
    """None (a metric absent from the results tree) renders as a visible
    'n/a', never as a crash."""
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def f2(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}"


def f3(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def cell_label(key: str) -> str:
    cc, ev, topo = rc.parse_cell_key(key)
    return f"{cc} committed / evidence {ev} / {topo.replace('_', '-')}"


# -------------------------------------------------------------- svg strip

def strip_svg(post: list[float], rows: list[dict]) -> str:
    """Render check times against the onset posterior over the 160-step stream."""
    left, right = 210, 900
    span = right - left
    post_h = 56
    row_h = 30
    top = 26
    height = top + post_h + 22 + row_h * len(rows) + 34
    peak = max(post) or 1.0

    def x_of(step: float) -> float:
        return left + span * (step / rc.S2_LENGTH)

    out = [f'<svg viewBox="0 0 940 {height}" width="100%" '
           f'role="img" aria-label="onset posterior and check placement" '
           f'style="font-family:inherit;font-size:12px;overflow:visible">']

    # posterior area
    base_y = top + post_h
    pts = [f"{x_of(0):.1f},{base_y:.1f}"]
    for t, p in enumerate(post):
        pts.append(f"{x_of(t):.1f},{base_y - post_h * (p / peak) * 0.92:.1f}")
    pts.append(f"{x_of(rc.S2_LENGTH):.1f},{base_y:.1f}")
    out.append(f'<polygon points="{" ".join(pts)}" fill="{ACCENT}" '
               f'fill-opacity="0.22" stroke="{ACCENT}" stroke-width="1.5"/>')
    out.append(f'<text x="{left}" y="{top - 10}" fill="currentColor" '
               f'opacity="0.75">onset posterior over the 160-step stream</text>')

    # axis
    axis_y = base_y + 1
    out.append(f'<line x1="{left}" y1="{axis_y}" x2="{right}" y2="{axis_y}" '
               f'stroke="currentColor" stroke-opacity="0.35"/>')
    for t in (0, 40, 80, 120, 160):
        out.append(f'<line x1="{x_of(t):.1f}" y1="{axis_y}" x2="{x_of(t):.1f}" '
                   f'y2="{axis_y + 5}" stroke="currentColor" stroke-opacity="0.35"/>')
        out.append(f'<text x="{x_of(t):.1f}" y="{axis_y + 18}" fill="currentColor" '
                   f'opacity="0.7" text-anchor="middle">{t}</text>')

    # policy rows
    y = axis_y + 34
    for row in rows:
        color = row.get("color", MUTED)
        weight = "600" if row.get("emphasis") else "400"
        opacity = "1" if row.get("emphasis") else "0.85"
        out.append(f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" '
                   f'fill="currentColor" opacity="{opacity}" '
                   f'font-weight="{weight}">{row["label"]}</text>')
        out.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" '
                   f'stroke="currentColor" stroke-opacity="0.18"/>')
        for step in row["schedule"]:
            out.append(f'<rect x="{x_of(step) - 1.6:.1f}" y="{y - 9}" width="3.2" '
                       f'height="18" rx="1.2" fill="{color}"/>')
        note = row.get("note", "")
        if note:
            out.append(f'<text x="{right + 10}" y="{y + 4}" fill="currentColor" '
                       f'opacity="0.8" font-weight="{weight}">{note}</text>')
        y += row_h

    out.append(f'<text x="{left}" y="{height - 8}" fill="currentColor" '
               f'opacity="0.6">check times marked; onset can only be seen by a '
               f'check that fires at or after it</text>')
    out.append("</svg>")
    return "".join(out)


CSS = """
<style>
.rv-banner { border-radius: 10px; padding: 16px 18px; margin: 6px 0 14px 0;
             border: 1px solid; }
.rv-banner h3 { margin: 0 0 6px 0; font-size: 1.25rem; letter-spacing: -0.01em; }
.rv-banner p { margin: 0; opacity: 0.92; }
.rv-tag { display:inline-block; font-size: 0.72rem; letter-spacing: 0.08em;
          text-transform: uppercase; opacity: 0.8; margin-bottom: 6px; }
table.rv { width: 100%; border-collapse: collapse; font-size: 0.88rem;
           font-variant-numeric: tabular-nums; }
table.rv th { text-align: left; font-weight: 600; padding: 6px 10px;
              border-bottom: 1px solid rgba(128,128,128,0.45); opacity: 0.85; }
table.rv td { padding: 5px 10px; border-bottom: 1px solid rgba(128,128,128,0.2); }
table.rv td.num, table.rv th.num { text-align: right; }
table.rv tr.hi td { font-weight: 600; }
.rv-note { font-size: 0.85rem; opacity: 0.75; margin-top: 6px; }
.rv-scroll { overflow-x: auto; }
</style>
"""


def banner_html(headline: str, detail: str, color: str, tag: str) -> str:
    return (f'<div class="rv-banner" style="border-color:{color};'
            f'background:{color}1f">'
            f'<div class="rv-tag" style="color:{color}">{tag}</div>'
            f'<h3>{headline}</h3><p>{detail}</p></div>')


def table_html(headers: list[str], rows: list[list[str]],
               numeric: set[int] | None = None,
               highlight: set[int] | None = None) -> str:
    numeric = numeric or set()
    highlight = highlight or set()
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{h}</th>'
                   for i, h in enumerate(headers))
    body = []
    for r_i, row in enumerate(rows):
        cls = ' class="hi"' if r_i in highlight else ""
        cells = "".join(f'<td class="{"num" if i in numeric else ""}">{c}</td>'
                        for i, c in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    return ('<div class="rv-scroll"><table class="rv"><thead><tr>' + head +
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


# ------------------------------------------------------------------- ui

def run_ui() -> int:
    import streamlit as st

    st.set_page_config(page_title="Attention-routing monitoring view",
                       layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    bundle = rc.load_bundle()

    st.title("Attention-routing reference monitoring view")
    st.markdown(
        "Every number on this page is an **oracle-placement bound** replayed "
        "from the released measurements: the detection instrument is held "
        "fixed at the deployed detector, and the oracle optimizes only where "
        "and when it fires. The bound caps every checking **schedule** of the "
        "deployed instrument -- it does not cap a reader whose judgment beats "
        "the instrument; that axis is measured by the Study-4 monitor zoo, "
        "not bounded. Nothing here models, measures, or assumes a human "
        "operator.")

    with st.sidebar:
        st.header("Data")
        st.markdown(
            "- `results/s1/stats.json` (regime map, 60 cells x 50 seeds)\n"
            "- `results/s2/stats.json` (+ `trials.jsonl`, onset placement)\n"
            "- `results/s4/zoo.json` (13 monitors, 90 panel trials)")
        st.caption("Paths resolve relative to this file; no absolute paths, "
                   "no network access, no GPU.")
        st.header("Replay self-check")
        if st.button("Run self-check"):
            checks = sc.run_checks(bundle, paper_tex=PAPER_TEX)
            n_fail = sum(1 for c in checks if not c.ok)
            (st.success if n_fail == 0 else st.error)(
                f"{len(checks) - n_fail} PASS, {n_fail} FAIL")
            st.code(sc.format_report(checks), language="text")

    _element_one(st, bundle)
    _element_two(st, bundle)
    _element_three(st, bundle)

    st.divider()
    st.markdown(
        "**What this view does not claim.** No human-operator claim of any "
        "kind: no attention model, no latency, no fallibility, no usability "
        "result. The bounds are oracle-placement bounds on a fixed "
        "instrument: they cap every checking schedule of the deployed "
        "detector, and an overseer who additionally pays reading latency sits "
        "below them on that scheduling axis. Judgment better than the "
        "instrument's is not bounded here -- it is measured, by the Study-4 "
        "monitor zoo. Regime boundaries are quantitative "
        "for these two substrates and qualitative beyond them; a deployment "
        "re-measures them with the released replay tooling.")
    return 0


def _element_one(st, bundle: rc.Bundle) -> None:
    st.header("1. Regime banner")
    st.caption("Maps a fault-speed estimate onto the regime map and states "
               "which of the three honest messages applies.")

    left, right = st.columns([1, 2], gap="large")
    with left:
        no_capture = st.checkbox("Median run never reaches irreversibility",
                                 value=False)
        speed = st.slider("Fault-speed estimate: rounds to irreversibility",
                          min_value=1, max_value=15, value=2, step=1,
                          disabled=no_capture)
        window = st.select_slider("Match window (rounds)", options=[0.5, 1.0, 2.0],
                                  value=1.0, disabled=no_capture)
        st.markdown("**Observables** (the same ones the regime map is "
                    "parameterized by)")
        minority = st.selectbox("Committed minority size", rc.MINORITY_SIZES,
                                index=1)
        evidence = st.selectbox("Evidence strength", rc.EVIDENCE_LEVELS, index=0)
        topology = st.selectbox("Topology", rc.TOPOLOGIES, index=0)

    match = rc.cells_at_fault_speed(bundle.s1, None if no_capture else float(speed),
                                    window=float(window))
    key = rc.cell_key(minority, evidence, topology)
    cell = rc.lookup_cell(bundle.s1, key)

    with right:
        color = REGIME_COLOR.get(match["modal_regime"] or "intermediate", MUTED)
        speed_txt = ("median run never captures" if no_capture
                     else f"about {speed} rounds (+/- {window:g})")
        tag = (f"fault speed: {speed_txt} -- "
               f"{len(match['matched'])} measured cell(s)")
        st.markdown(banner_html(match["headline"], match["detail"], color, tag),
                    unsafe_allow_html=True)
        if match["matched"] and not match["unanimous"]:
            dist = ", ".join(f"{k.replace('_', '-')}: {v}"
                             for k, v in sorted(match["distribution"].items(),
                                                key=lambda kv: -kv[1]))
            st.markdown(
                f'<div class="rv-note">Fault speed alone does not fix the '
                f'regime: cells at this speed split {dist}. Supply the '
                f'observables below to land on a cell.</div>',
                unsafe_allow_html=True)

        if cell is None:
            st.warning("No measured cell for that combination.")
        else:
            c_color = REGIME_COLOR[cell["regime"]]
            st.markdown(
                banner_html(
                    cell["headline"], cell["detail"], c_color,
                    f"cell {cell_label(key)} -- "
                    f"{cell['regime'].replace('_', '-')}"),
                unsafe_allow_html=True)
            mc = cell["median_capture"]
            rows = [[
                "median round of irreversibility",
                "none at median" if mc is None else f2(float(mc))],
                ["baseline failure rate without checking",
                 pct(cell["baseline_failure"])],
                ["onset-oracle timely-catch rate (reference policy, not a "
                 "catch-count ceiling)",
                 pct(cell["oracle_timely_reference"])],
                ["failures the oracle can actually prevent",
                 pct(cell["oracle_preventable"])],
                ["best single-check policy",
                 f'{cell["best_single_check"]["policy"]} '
                 f'({pct(cell["best_single_check"]["fc_prevention"])} prevented)'],
            ]
            st.markdown(table_html(["measured quantity", "value"], rows, {1}),
                        unsafe_allow_html=True)

    st.subheader("What routing by regime costs and keeps")
    led = rc.routing_ledger(bundle.s1)
    if led["n_cells"] == 0 or led["total_catches"] <= 0.0 \
            or led["total_preventable"] <= 0.0:
        st.warning(
            "results/s1/stats.json carries no cells (or no oracle catches / "
            "preventable failures), so the routing ledger cannot be computed. "
            "Run the self-check: it reports this state as FAIL lines.")
        return
    st.markdown(
        f"Spending **zero checks** in ceiling-bound cells across all "
        f"{led['n_cells']} Study-1 cells x {led['seeds_per_cell']} seeds "
        f"retains **{round(led['retained_catches']):,} of "
        f"{round(led['total_catches']):,}** oracle-achievable timely catches "
        f"(**{pct(led['retained_share'])}**) while forfeiting "
        f"**{round(led['forfeited_preventable'])} of "
        f"{round(led['total_preventable'])}** oracle-preventable failures "
        f"(**{pct(led['forfeited_share'], 1)}**). The detections given up are "
        f"almost entirely ones that prevent nothing.")

    over = rc.uniform_b3_overtakes(bundle.s1)
    if over:
        cells_txt = ", ".join(o["key"] for o in over)
        n_ceil = sum(1 for o in over if o["regime"] == "ceiling_bound")
        n_benign = sum(1 for o in over if o["regime"] == "benign")
        st.markdown(
            f'<div class="rv-note">The onset-oracle rate is a <b>reference '
            f'policy, not a catch-count ceiling</b>: it maxes over the oracle '
            f'at B = 1/2/3 only, and uniform@B3 registers more timely catches '
            f'in {len(over)} of the 60 cells -- {cells_txt}. Every one is a '
            f'cell where those extra catches prevent nothing '
            f'(oracle-preventable rate 0.0: {n_ceil} ceiling-bound, '
            f'{n_benign} benign on prevention).</div>',
            unsafe_allow_html=True)

    order = [r for r in rc.REGIME_ORDER if r in led["per_regime"]]
    rows = []
    hi = set()
    for i, reg in enumerate(order):
        row = led["per_regime"][reg]
        if reg == "ceiling_bound":
            hi.add(i)
        rows.append([
            reg.replace("_", "-"),
            str(row["cells"]),
            f"{round(row['catches']):,}",
            f"{row['catches'] / led['total_catches'] * 100:.1f}%",
            f"{round(row['preventable']):,}",
            f"{row['preventable'] / led['total_preventable'] * 100:.1f}%",
            "zero checks" if reg == "ceiling_bound" else "checks spent",
        ])
    st.markdown(table_html(
        ["regime", "cells", "oracle timely catches", "share of catches",
         "oracle-preventable failures", "share of preventable", "routing"],
        rows, {1, 2, 3, 4, 5}, hi), unsafe_allow_html=True)
    ceil_row = led["per_regime"].get("ceiling_bound")
    if ceil_row is None:
        st.markdown(
            '<div class="rv-note">No ceiling-bound cells in this results '
            'tree, so routing gives nothing up.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="rv-note">Ceiling-bound cells carry '
            f'{pct(ceil_row["catches"] / led["total_catches"], 1)} of the catches '
            f'and {pct(led["forfeited_share"], 1)} of the preventable failures: '
            f'checking there detects the fault and changes nothing, because '
            f'irreversibility has already passed. That is why the honest message '
            f'is "checking cannot help here" and not "look harder".</div>',
            unsafe_allow_html=True)


def _element_two(st, bundle: rc.Bundle) -> None:
    st.header("2. Placement strip")
    st.caption("The chosen policy's check times against the onset posterior, "
               "with the registered fallback when the observables say nothing.")

    left, right = st.columns([1, 2], gap="large")
    with left:
        B = st.select_slider("Attention budget (checks)",
                             options=list(rc.S2_BUDGETS), value=2)
        mode = st.radio("Onset posterior",
                        ["Flat (observables uninformative)",
                         "Informative (observables sharpen onset)"], index=0)
        informative = mode.startswith("Informative")
        # Spread range must reach the flat classification: with
        # FLATNESS_RATIO = 1.05 the posterior classifies flat at
        # spread >= 129 for centre 80 and >= 257 for the worst-case centres
        # (40 or 120), so the cap sits above both. The self-check pins that
        # flat is reachable on this range and that spread 128 is not flat.
        center = st.slider("Posterior centre (stream step)",
                           float(rc.S2_ONSET_LO), float(rc.S2_ONSET_HI), 60.0,
                           step=1.0, disabled=not informative)
        spread = st.slider("Posterior spread (steps, std)", 1.0, 260.0, 6.0,
                           step=1.0, disabled=not informative)
        st.caption("Widening the spread flattens the posterior; the fallback "
                   "fires on the shape of the posterior, not on this switch.")

    post = rc.informative_posterior(center, spread) if informative \
        else rc.flat_posterior()
    choice = rc.choose_placement(post, B)
    fb = rc.fallback_replay(bundle.s2, budgets=rc.S2_BUDGETS)
    measured = fb.get(B)   # None when this budget is not in front_collapse
    if measured is None:
        st.info(f"Budget B={B} has no front_collapse entry in "
                f"results/s2/stats.json; measured miss rates at this budget "
                f"are shown as 'not in results'.")

    def measured_miss(key: str) -> str:
        if measured is None:
            return "not in results"
        return f"miss {f2(measured[key])}"

    with right:
        color = GOOD if choice["posterior_flat"] else ACCENT
        head = ("Posterior is flat -- hedge to uniform placement"
                if choice["posterior_flat"]
                else "Posterior is informative -- place on posterior mass")
        st.markdown(banner_html(head, choice["rationale"], color,
                                f"budget {B} -- policy: {choice['policy_label']}"),
                    unsafe_allow_html=True)

    # Illustrative true onset for the oracle row: the posterior's own mean.
    posterior_mean = int(round(sum(t * p for t, p in enumerate(post))))
    rows = [{
        "label": f"chosen: {choice['policy_label']}",
        "schedule": choice["schedule"],
        "color": color, "emphasis": True,
        "note": (measured_miss("uniform_miss")
                 if choice["policy"] == "uniform" else "not replayed"),
    }, {
        "label": "audit-early (geometric)",
        "schedule": rc.s2_geometric(B), "color": STOP,
        "note": measured_miss("audit_early_miss"),
    }, {
        "label": "front-loaded",
        "schedule": rc.s2_front(B), "color": STOP,
        "note": measured_miss("front_miss"),
    }, {
        "label": "uniform",
        "schedule": rc.s2_uniform(B), "color": ACCENT,
        "note": measured_miss("uniform_miss"),
    }, {
        "label": "back-loaded",
        "schedule": rc.s2_back(B), "color": WARN, "note": "detects at the end",
    }, {
        "label": "onset oracle (exact onset)",
        "schedule": rc.s2_oracle(B, posterior_mean), "color": GOOD,
        "note": measured_miss("oracle_miss"),
    }]
    st.markdown(strip_svg(post, rows), unsafe_allow_html=True)

    unseen = rc.mass_after_last_check(post, choice["schedule"])
    geo_unseen = rc.mass_after_last_check(post, rc.s2_geometric(B))
    st.markdown(
        f'<div class="rv-note">Posterior mass falling after the last check -- '
        f'onset that no check in the schedule can see: '
        f'<b>{pct(unseen, 1)}</b> for the chosen placement, '
        f'<b>{pct(geo_unseen, 1)}</b> for audit-early. This is geometry of the '
        f'displayed posterior, not a replayed measurement; the measured miss '
        f'rates are in the table below.</div>', unsafe_allow_html=True)

    st.subheader("Replay: switching to uniform whenever the posterior is flat")
    rows2 = []
    for b in rc.S2_REPLAY_BUDGETS:
        m = fb.get(b)
        if m is None:
            rows2.append([f"B = {b}"] + ["not in results"] * 4)
            continue
        rows2.append([f"B = {b}", f2(m["audit_early_miss"]), f2(m["front_miss"]),
                      f2(m["uniform_miss"]), f2(m["oracle_miss"])])
    st.markdown(table_html(
        ["budget", "audit-early miss", "front-loaded miss", "uniform miss",
         "onset-oracle miss"], rows2, {1, 2, 3, 4}), unsafe_allow_html=True)
    st.markdown(
        "Switching from audit-early to uniform whenever the posterior is flat "
        "cuts the miss rate from **1.00** to **0.48 / 0.18 / 0.00** at "
        "B = 1 / 2 / 4, eliminating the total-miss failure mode entirely at "
        "four checks. Audit-early misses everything at these budgets because "
        "geometric spacing puts every check before the earliest possible "
        "onset. Pooled over the diagnosable well-specified Study-2 cells.")

    with st.expander("Per-cell placement outcomes at this budget"):
        cell_keys = rc.s2_cell_keys(bundle.s2)
        if not cell_keys:
            st.warning("results/s2/stats.json carries no per-cell entries.")
        else:
            ck = st.selectbox("Study-2 cell", cell_keys, index=0)
            table = rc.s2_cell_policy_table(bundle.s2, ck, B)
            if rc.cell_is_undetectable(bundle.s2, ck):
                st.markdown(banner_html(
                    "Detection out of reach in this cell",
                    "Even the onset-informed reference misses every trial at "
                    "every budget here: the two rules are not separable by the "
                    "probe battery, so placement is irrelevant. The Study-2 "
                    "analogue of a ceiling-bound cell.", STOP, "cell ceiling"),
                    unsafe_allow_html=True)
            st.markdown(table_html(
                ["policy", "miss rate", "penalized delay", "false alarm rate"],
                [[r["label"], f2(r["miss"]), f"{r['mean_pen']:.1f}",
                  f2(r["fa"])] for r in table], {1, 2, 3}),
                unsafe_allow_html=True)
            st.caption("Penalized delay charges a miss the remaining horizon: "
                       "back-loaded placement almost never misses and still "
                       "pays close to the full stream.")


def _element_three(st, bundle: rc.Bundle) -> None:
    st.header("3. Decision-time scoreboard")
    st.caption("Thirteen monitor variants ranked by timely recall at the last "
               "round where intervention still prevents the failure -- not by "
               "transcript AUC.")

    head = rc.zoo_headline(bundle.zoo)
    board = rc.scoreboard(bundle.zoo)
    if not board:
        st.warning(
            "results/s4/zoo.json carries no monitors; the scoreboard cannot "
            "be rendered. Run the self-check: it reports this state as FAIL "
            "lines.")
        return
    st.markdown(
        f"{head['n_monitors']} monitors over {head['n_trials']} real-agent "
        f"panel trials. Hindsight and decision-time rankings agree only at "
        f"Kendall tau = {f2(head['tau_timely_auc'])} "
        f"[{f2(head['tau_timely_auc_ci'][0])}, "
        f"{f2(head['tau_timely_auc_ci'][1])}] on timely AUC and "
        f"{f2(head['tau_timely_recall'])} on matched-FPR timely recall. "
        f"**No variant exceeds {f2(head['max_timely_recall'])} timely recall "
        f"at the deployed operating point** -- the whole zoo fails the "
        f"deployment test, so this ranking orders survivors, not solutions.")

    rows, hi = [], set()
    for i, r in enumerate(board):
        shift = r["rank_shift"]
        arrow = "--" if shift == 0 else (f"up {shift}" if shift > 0
                                         else f"down {-shift}")
        if r["is_constancy"]:
            hi.add(i)
        rows.append([
            str(r["rank"]),
            r["monitor"] + (" (constancy)" if r["is_constancy"] else ""),
            f3(r["timely_recall_mfpr"]),
            f3(r["timely_auc"]),
            "not measured",
            f3(r["posthoc_auc"]),
            "n/a" if r["posthoc_rank"] is None else str(r["posthoc_rank"]),
            arrow,
        ])
    st.markdown(table_html(
        ["#", "monitor", "timely recall (matched FPR)", "timely AUC",
         "penalized delay", "transcript AUC", "hindsight rank",
         "move vs hindsight"],
        rows, {0, 2, 3, 5, 6}, hi), unsafe_allow_html=True)

    st.markdown(f'<div class="rv-note">{rc.DELAY_UNAVAILABLE}</div>',
                unsafe_allow_html=True)

    dem = rc.constancy_demotion(bundle.zoo)
    cc = dem.get("conf_constancy")
    cl = dem.get("claim_constancy")
    hcs = next((r for r in board if r["monitor"] == "high_conf_share"), None)
    if cc is None or cl is None or hcs is None:
        missing = [n for n, v in (("conf_constancy", cc), ("claim_constancy", cl),
                                  ("high_conf_share", hcs)) if v is None]
        st.warning(
            "Monitors missing from results/s4/zoo.json: "
            + ", ".join(missing)
            + ". The constancy-demotion note is not rendered; the self-check "
              "reports this state as FAIL lines.")
    else:
        st.markdown(
            f"**What the reordering does to the constancy monitors.** "
            f"`conf_constancy` is the best monitor in hindsight "
            f"(rank {cc['posthoc_rank']} of {head['n_monitors']} by transcript "
            f"AUC) and falls to rank {cc['timely_auc_rank']} on timely AUC: at the "
            f"last timely round exactly one round of history exists, and a "
            f"constancy signal needs at least two, so its discriminative power is "
            f"structurally zero when it matters. `claim_constancy` goes "
            f"{cl['posthoc_rank']} -> {cl['timely_auc_rank']} on timely AUC and "
            f"{cl['scoreboard_rank']} on this scoreboard, with timely recall "
            f"{f2(cl['timely_recall_mfpr'])}. Single-round-readable signals move "
            f"the other way: `high_conf_share` goes {hcs['posthoc_rank']} -> "
            f"{hcs['timely_auc_rank']} on timely AUC.")
        st.markdown(
            f'<div class="rv-note">Stated exactly: under the scoreboard\'s primary '
            f'key, matched-FPR timely recall, <code>conf_constancy</code> is '
            f'rank {cc["scoreboard_rank"]} '
            f'({f3(cc["timely_recall_mfpr"])} against a zoo best of '
            f'{f3(head["max_timely_recall"])}), so on that key it is not demoted. '
            f'The demotion of the constancy family is unambiguous on timely AUC '
            f'(rank {cc["posthoc_rank"]} -> {cc["timely_auc_rank"]}) and for '
            f'<code>claim_constancy</code> on both keys. The view reports both '
            f'columns rather than the one that reads better.</div>',
            unsafe_allow_html=True)

    with st.expander("Does the threshold or the deadline rescue the zoo?"):
        fpr = head["fpr_sensitivity"]
        st.markdown(table_html(
            ["matched false-positive rate", "deployed monitor timely recall"],
            [[f"{float(k) * 100:.1f}%", f3(v)] for k, v in sorted(
                fpr.items(), key=lambda kv: float(kv[0]))], {0, 1}),
            unsafe_allow_html=True)
        lat = head["latency_ablation"]
        st.markdown(table_html(
            ["lead time required (rounds)", "deployed timely AUC",
             "best monitor timely AUC", "scoreable pairs left"],
            [[k.replace("lead_", ""), f3(v["deployed_auc"]),
              f3(v["best_auc"]), str(v["n_pairs"])]
             for k, v in sorted(lat.items())], {0, 1, 2, 3}),
            unsafe_allow_html=True)
        st.caption("Sweeping the operating point moves the deployed monitor's "
                   "timely recall from 0.01 to 0.04. Demanding lead time only "
                   "shrinks the set of trials where any decision is possible.")


# ------------------------------------------------------------------ cli

USAGE = """\
attention-routing reference monitoring view

  streamlit run monitoring_view.py               interactive view
  python -X utf8 monitoring_view.py --selfcheck  replay assertions, PASS/FAIL
  python -X utf8 monitoring_view.py --demo-fault corrupt the inputs AND
                                                 monkeypatch every schedule
                                                 generator, showing the
                                                 checks failing in both cases
"""


def main(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(USAGE)
        return 0

    if "--selfcheck" in argv:
        bundle = rc.load_bundle()
        checks = sc.run_checks(bundle, paper_tex=PAPER_TEX)
        header = ("reference monitoring view -- replay self-check\n"
                  "expected values are literals transcribed from the paper; "
                  "replay values are recomputed from results/")
        print(sc.format_report(checks, header))
        n_fail = sum(1 for c in checks if not c.ok)
        print("SELFCHECK " + ("PASS" if n_fail == 0 else "FAIL"))
        return 1 if n_fail else 0

    if "--demo-fault" in argv:
        bundle = rc.load_bundle()
        report, did_fail = sc.demo_fault(bundle)
        print(report)
        if did_fail:
            print("FAULT DEMO PASS: corrupted inputs were rejected, so the "
                  "self-check can fail.")
            return 0
        print("FAULT DEMO FAIL: corrupted inputs still passed -- the "
              "self-check is not testing anything.")
        return 1

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            running = get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:      # older Streamlit without the keyword
            running = get_script_run_ctx() is not None
    except Exception:
        running = False
    if not running:
        print(USAGE)
        print("Not running inside a Streamlit runtime; nothing to render.")
        return 0
    return run_ui()


if __name__ == "__main__":
    # Streamlit executes this file as "__main__" too, so only a genuine
    # command-line failure raises SystemExit; a rendered page just returns.
    _status = main(sys.argv[1:])
    if _status:
        sys.exit(_status)
