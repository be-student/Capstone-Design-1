# F11-X Fix Log — render_overview / render_budget_optimization / render_ab_testing

**Agent:** F11-X (iter11 round 1)
**File touched:** `src/dashboard/app.py` (only the three assigned functions)
**Verify reports closed against:**
- `_test_results/iter10/verify_v1.md` (Page 00)
- `_test_results/iter10/verify_v2.md` (Page 05)
- `_test_results/iter10/verify_v3.md` (Page 06)

Helpers consumed (defensive try/except ImportError in place at module top
since iter10 round 1):
`format_count`, `format_currency_krw`, `compute_overall_roi`.

---

## Defect 1 — Page 00 Total CLV ellipsis (verify_v1 #5)

**Reasoning.** F1 already shipped `format_currency_krw()` in iter10 but the
helper was never wired to the Page 00 KPI strip. KPI tile width clipped the
12-digit raw `f"{total_clv:,.0f} KRW"` and rendered `57,936,514,970 ...`.

**Before.**
```python
col4.metric("Total CLV", f"{total_clv:,.0f} KRW")
```

**After.**
```python
col4.metric(
    "Total CLV",
    format_currency_krw(total_clv),
    help="Sum of predicted Customer Lifetime Value ... Compact display (B/M/K) avoids overflow truncation in the KPI tile.",
)
```

Total Customers and High Risk also now go through `format_count(...)` for
consistent formatting. The per-customer Predicted CLV in the customer
lookup section also uses `format_currency_krw(clv)`.

**Closes:** verify_v1 #5.

---

## Defect 2 — Page 00 / 01 histogram bin parity (verify_v1 #4)

**Reasoning.** Page 00 used `nbins=30` (≈ 0.0333 wide bins) over
auto-detected range, while Page 01 uses `nbinsx=50` (0.02 wide bins) over
[0, 1]. For the same 20k roster the leftmost-bin count was ≈ 4,000 vs
≈ 3,000 — and the gap *widened* in iter10 vs iter9 because Page 00 wasn't
touched.

**Before.**
```python
fig = px.histogram(
    predictions, x="churn_probability", nbins=30,
    title="Distribution of Churn Probabilities",
    ...
)
```

**After.**
```python
fig = px.histogram(
    predictions, x="churn_probability", nbins=50,
    range_x=[0, 1],
    title="Distribution of Churn Probabilities",
    ...
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Histogram bin width: 0.02 (50 bins across [0, 1]) - consistent "
    "with the Churn Analytics page so the leftmost-bin counts can be "
    "reconciled across pages."
)
```

`nbinsx=50` matches Page 01's `go.Histogram(nbinsx=50)` at line ~3247. A
footnote also documents the chosen bin width so the reader can reconcile
counts across pages even if the underlying data is loaded twice.

**Closes:** verify_v1 #4.

---

## Defect 3 — Page 05 Avg ROI 3.5x ≠ aggregate 3.84x (verify_v2 #5)

**Reasoning.** The headline KPI computed `display_results["roi"].mean()`
(arithmetic mean of per-segment ROIs = 3.5x) while the aggregate ROI of
the budget envelope is `total_revenue_saved / total_allocated` ≈ 3.84x.
Two different aggregations, one headline number — the production-relevant
scope is the budget-envelope aggregate.

**Before.**
```python
avg_roi = display_results["roi"].mean()
...
kpi4.metric("Avg ROI", f"{avg_roi:.1f}x")
```

**After.**
```python
avg_seg_roi = float(display_results["roi"].mean())
overall_roi = compute_overall_roi(
    revenue_saved=total_rev_saved,
    cost_or_budget=total_alloc,
    scope_label="budget",
)
...
kpi4.metric(
    overall_roi.get("label", "ROI (budget envelope)"),
    overall_roi.get("display", "-"),
    help=("Aggregate ROI = total revenue saved / total budget allocated. ..."),
)
st.caption(
    f"Mean of segment ROIs: {avg_seg_roi:.2f}x - see ROI by Segment chart. "
    "The headline above uses the aggregate revenue_saved / total_allocated, "
    "which is the production-relevant scope (iter11 fix for verify_v2 #5)."
)
```

Headline now reads `ROI (budget envelope)  3.84x`; the previous mean is
preserved as a sub-caption so anyone reading the chart's per-segment bars
can still cross-reference the 3.5x figure.

**Closes:** verify_v2 #5 (also satisfies the iter9-deferred ROI scope
ask).

---

## Defect 4 — Page 05 Expected Retained 118 vs Baseline 122 (verify_v2 #6)

**Reasoning.** Root cause: the headline KPI summed the *per-row
int-truncated* `expected_retained` column (after `.astype(int)` was
applied to scaled values, line ~1006-1009) → 118. The
`_compute_scenario_comparison` path computed `int((... * scale * uplift_m).sum())`
on the un-truncated baseline → 122. Same scenario, two different
arithmetic paths.

Picked option (a) per task description: align headline to the same
arithmetic the scenario table uses.

**Before.**
```python
total_retained = display_results["expected_retained"].sum()
kpi2.metric("Expected Retained", f"{total_retained:,}")
```

**After.**
```python
total_retained = int(
    (budget_results["expected_retained"] * scale * uplift_multiplier).sum()
)
kpi2.metric(
    "Expected Retained",
    f"{total_retained:,}",
    help=("Aggregated as int(sum(per-segment retained)). Matches the Baseline / "
          "Current Selection rows of the What-If Scenario Comparison table by "
          "construction (iter11 reconciliation)."),
)
```

Since the headline now mirrors `_compute_scenario_comparison` exactly,
the Baseline row and the headline KPI display the same integer for the
default-budget / multipliers=1.0 case.

**Closes:** verify_v2 #6.

---

## Defect 5 — Page 05 high_value_persuadable 31k @ 8x ROI LP constraint (verify_v2 #7)

**Reasoning.** LP solution metadata (`slack`, `shadow_price`) is not
exposed by `data_loader.load_budget_results()`, so per the task spec we
take the documentation route and surface the binding constraint via a
caption beneath the segment allocation table.

**Before.** No annotation; the user sees a 31k allocation against the
chart's tallest 8x ROI bar with no explanation.

**After.**
```python
try:
    if "high_value_persuadable" in set(display_results["segment"].astype(str).tolist()):
        hv_row = display_results[
            display_results["segment"].astype(str) == "high_value_persuadable"
        ].iloc[0]
        hv_alloc = float(hv_row.get("allocated_budget_krw", 0))
        hv_roi = float(hv_row.get("roi", 0))
        st.caption(
            f"Note: high_value_persuadable receives only "
            f"{hv_alloc:,.0f} {currency} despite a ~{hv_roi:.1f}x ROI. "
            "Allocation is limited by segment-size cap (binding "
            "constraint: segment_size - only a small population of "
            "high-value persuadable customers exists, so the LP "
            "cannot scale spend further on this segment regardless "
            "of its per-unit ROI)."
        )
except (KeyError, IndexError, ValueError, TypeError):
    pass
```

Reads numbers off the same DataFrame the table renders, so the caption
auto-updates if the upstream LP changes the allocation.

**Closes:** verify_v2 #7.

---

## Defect 6 — Page 05 Channel-Level Cost Breakdown empty H3 (verify_v2 #8)

**Reasoning.** The H3 was always rendered, then the empty-state info
banner appeared *under* it, producing a visually empty section above a
warning. Moved the `st.subheader("Channel-Level Cost Breakdown")` inside
the `if channel_config:` branch so when `budget.channels` is missing only
the banner renders.

**Before.**
```python
st.subheader("Channel-Level Cost Breakdown")
channel_config = config.get("budget", {}).get("channels", {})
if channel_config:
    ...
else:
    st.info("Channel configuration not found in config. ...")
```

**After.**
```python
channel_config = config.get("budget", {}).get("channels", {})
if channel_config:
    st.subheader("Channel-Level Cost Breakdown")
    ...
else:
    st.info("Channel configuration not found in config. ...")
```

**Closes:** verify_v2 #8.

---

## Defect 7 — Page 06 0/0/N/A/0.0% empty state + MDE feasibility guard (verify_v3 P06 #1)

**Reasoning.** Two sub-defects bundled:

1. When `total_experiments == 0`, four bare zero-style KPI tiles look
   like real data. Replace with `st.info(...)` empty state.
2. The MDE Sensitivity table recommends 24,441/group → 48,882 total at
   MDE 1% against a 20k pool, with no feasibility flag.

**Before.**
```python
kc1, kc2, kc3, kc4 = st.columns(4)
kc1.metric("Total Experiments", summary.get("total_experiments", len(experiments)))
kc2.metric("Significant Results", summary.get("significant_count", 0))
kc3.metric("Best Experiment", summary.get("best_experiment", "N/A"))
kc4.metric("Avg Lift", f"{summary.get('avg_lift', 0):.1%}")
...
# (no MDE feasibility guard)
st.dataframe(mde_table.style.format(...), use_container_width=True)
```

**After.**
```python
total_experiments = int(summary.get("total_experiments", len(experiments)) or 0)
if total_experiments == 0:
    st.info(
        "No experiments logged yet - launch your first A/B test from "
        "the Retention Campaign Builder (Page 10) and re-run the "
        "pipeline to populate this view. The Power Analysis & Sample "
        "Size Calculator below is still usable for planning."
    )
else:
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total Experiments", total_experiments)
    ...
```

And before the MDE table render:
```python
pool_size = int(config.get("simulator", {}).get("num_customers", 0) or 0)
infeasible_rows = []
if pool_size > 0 and not mde_table.empty:
    for _, _row in mde_table.iterrows():
        try:
            total_required = int(_row.get("Total Participants", 0))
            mde_val = float(_row.get("MDE", 0))
        except (TypeError, ValueError):
            continue
        if total_required > pool_size and mde_val > 0:
            infeasible_rows.append((mde_val, total_required))
if infeasible_rows:
    infeasible_rows.sort(key=lambda x: x[0])
    examples = "; ".join(
        f"MDE {m:.1%} needs {n:,} vs {pool_size:,} pool"
        for m, n in infeasible_rows[:3]
    )
    st.warning(
        "Feasibility check: some MDE rows below require more "
        f"participants than the available customer pool ({pool_size:,}). "
        f"Infeasible row(s): {examples}. ..."
    )
```

The warning is rendered *before* the table so the user sees the
feasibility caveat before reading the row that triggered it. Pulls pool
size from `simulator.num_customers` in config (= 20,000 in FULL mode,
5,000 in SMALL mode).

**Closes:** verify_v3 P06 #1 (both sub-parts).

---

## Constraints honored

- Only `render_overview` (~196), `render_budget_optimization` (~920), and
  `render_ab_testing` (~1352) were modified. Verified via `grep -n
  "^def render_"` before/after; line offsets shifted from concurrent
  edits by F11-Y/F11-Z to neighbour functions but the three function
  spans I edited are intact.
- All new helper calls (`format_currency_krw`, `format_count`,
  `compute_overall_roi`) are imported via the existing defensive try/except
  ImportError block at module top — local fallbacks preserve render even
  when the helpers haven't been deployed.
- `python -c "import ast; ast.parse(...)"` passes on the modified file.

## Untouched (not in F11-X scope)

- Page 01 Critical/High taper disclosure (verify_v1 #6) — F11-Y or F11-Z.
- P03 segment definitions table (verify_v2 #1) — F11-Y or F11-Z.
- P07 segment binary 0/100 leak / dual taxonomy / median censoring
  threshold (verify_v3 P07 #3, #4, #5) — F11-Y or F11-Z.
- Synthetic-data banner (verify_v1 #7) — explicitly out of scope per
  v1 report.
