# V5 — Verification of iter9 audit A5 (Recommendations / CLV / Retention Campaign)

**Verifier stance:** independent, post-fix. Each verdict is grounded in (a) the
fix log claims, (b) actual code in `src/dashboard/recommendations_view.py` /
`src/dashboard/app.py` / `src/dashboard/utils/dashboard_helpers.py`, and (c)
the rendered PNGs under `_test_results/dashboard_pages/`.

**Inputs consulted:**
- `_test_results/iter9/iter9_audit_a5.md`
- `_test_results/iter10/fix_logs/{f1_helpers.md, f4_recommendations.md, f5_app.md}`
- `src/dashboard/recommendations_view.py`
- `src/dashboard/app.py` (render_clv at L2173, render_retention_campaign at L2706, render_budget_optimization at L898)
- `src/dashboard/utils/dashboard_helpers.py`
- `_test_results/dashboard_pages/{09_recommendations.png, 10_clv_prediction.png, 12_clv_retention_campaign.png}`

---

## 1. P09 — two "Avg Expected Uplift" labels (6.36% vs 10.88%)

**Verdict: FIXED.**

- F4 fix log claims the top KPI was renamed to **"Avg Predicted Uplift (all customers)"** and the mid-page card to **"Avg Treated Uplift"**, with reciprocal `help=` tooltips.
- `recommendations_view.py:213` confirms the top card uses `kc2.metric("Avg Predicted Uplift (all customers)", ...)` with help text that explicitly cross-references the mid-page treated-only card.
- `recommendations_view.py:569` confirms the mid-page card uses `oc4.metric("Avg Treated Uplift", ...)` with help text pointing back at the top card.
- Labels are now distinct strings, populations are stated, and an analyst can disambiguate from either tile without reading source.

---

## 2. P09 — 16,106 high priority vs 3,398 coupons reconciliation (12,708 silent no_action)

**Verdict: FIXED.**

- F4 fix log claims an inline "Priority vs treated reconciliation" `st.info` banner immediately under the top KPI strip.
- `recommendations_view.py:250-281` confirms the implementation: it computes `high_priority_count` (priority_score ≥ 0.7), `hp_treated`, and `hp_no_action`, then renders an info banner with the literal sentence "Of N high-priority customers, T (X%) receive a treatment offer; H (Y%) get `no_action` because their **predicted uplift × CLV did not exceed the cost threshold** for any offer in the catalog. Total treated across all priorities: M." All counts route through `format_count`.
- The banner is computed from the same `recs` dataframe used by the KPI cards, so the numbers are guaranteed consistent (no separate query path).

---

## 3. Cross-page Overall ROI (P05 = 3.5x vs P09 = 9.0x vs P12 = 3.8x) — single helper / disambiguated labels

**Verdict: PARTIAL.**

- F1 added the canonical `compute_overall_roi(revenue_saved, cost_or_budget, scope_label=...)` helper (`dashboard_helpers.py:189-250`). It returns `{value, display, label, tooltip}` and forces the caller to declare a scope. Helper unit tests pass (F1: 26 legacy tests).
- **P09 (FIXED):** `recommendations_view.py:535` calls `compute_overall_roi(..., scope_label="treated")`, renders `roi_info["label"]` ("ROI (treated only)"), and the help tooltip explicitly references both Page 05 and Page 12 to reconcile the three values.
- **P12 (FIXED):** `app.py:2934` calls `compute_overall_roi(total_rev_saved, total_allocated, scope_label="budget")`, renders `roi_info["label"]` ("ROI (budget envelope)"), and the help tooltip points the user at Page 09 / Page 13's treated-only scope.
- **P05 (NOT FIXED):** `app.py:1002,1007` still computes `avg_roi = display_results["roi"].mean()` and renders `kpi4.metric("Avg ROI", f"{avg_roi:.1f}x")`. Page 05 was deliberately left out of F5 scope ("owned by F2/F3/F4") but no F2/F3 fix log was supplied and no other agent appears to have touched Page 05's KPI strip. The "Avg ROI 3.5x" headline (mean-of-per-segment-ROIs) is therefore still a third, undocumented ROI definition. The cross-page disambiguation only holds for two of the three pages; a buyer clicking P05 → P09 → P12 still sees three labels with no glossary on P05's tile.
- Until P05's "Avg ROI" tile is also routed through `compute_overall_roi(scope_label="segment_avg")` (or a footnote is added), the iter9 buyer-trust blocker is only 2/3 closed.

---

## 4. P10 — high_value_persuadable n=2, lost_cause n=1 hidden in headline (count column emphasis)

**Verdict: PARTIAL.**

- The "CLV by Segment" aggregation at `app.py:2280-2286` already requests `["mean", "sum", "count", "median", "std"]` and renames to `"Count"` — a Count column is therefore present in the segment dataframe shown via `st.dataframe(seg_clv.style.format(...))` at L2314. PNG 10_clv_prediction.png confirms the numeric column is rendered (visible n=4000-ish numbers next to high_value_sure_thing, n=1 next to high_value_lost_cause, n=2 next to high_value_persuadable).
- However, **the "Mean CLV by Segment" bar chart at L2289-2299 still treats n=1 and n=4,000 segments as visually equivalent** — no explicit count overlay, no opacity-by-count, no sample-size warning banner above the chart, and no outlier flag for the n=1/n=2 segments. The buyer-trust hazard the audit raised ("a buyer who sorts or budgets by mean CLV") is only addressed in the table, not in the chart that the eye lands on first.
- No new "Sample-size warning" banner or `st.caption` was added under the segment bar chart. The fix logs (f4, f5) do not mention Page 10 segment-count emphasis at all.
- Verdict: partial because the Count column exists in the table (the audit's literal "no segment counts column" complaint), but the headline/bar-chart visual still does not emphasize tiny segments.

---

## 5. P10 — CLV vs Churn x-axis −0.5 to 1.5 (negative probability) — clamped to [0,1]

**Verdict: NOT FIXED.**

- `app.py:2328-2348` (`render_clv`, "CLV vs Churn Risk" scatter) constructs `px.scatter(predictions, x="churn_probability", y="clv_predicted", ...)` with no `range_x=[0, 1]` argument and no upstream `predictions["churn_probability"].clip(0, 1)`. The only x-axis modification is `add_vline(x=0.5, ...)` (a threshold annotation), which does not constrain the axis.
- A grep over `app.py` for `range_x`, `xaxis_range`, or any `clip(... churn_probability ...)` returns no hits inside `render_clv`.
- PNG `10_clv_prediction.png` confirms the scatter still renders an x-axis that extends beyond [0, 1] (the right edge sits past the "Churn Threshold" vline at 0.5 by significantly more than 0.5 — auto-padding from plotly is unchanged).
- Neither f4 nor f5 fix logs mention an x-axis clamp on this scatter.

---

## 6. P10 — NaN in churn_probability for top/bottom 10 tables

**Verdict: NOT FIXED.**

- `app.py:2413-2427` (`render_clv` Top/Bottom 10 tables) selects columns `["customer_id", "clv_predicted", "segment", "churn_probability"]` directly from `predictions`. There is no `.dropna(subset=["churn_probability"])`, no `fillna(...)`, and no fallback when the column is all-NaN.
- Upstream at `app.py:2222-2224`, when no predictions data is returned the code defensively assigns `predictions["churn_probability"] = np.nan` — meaning the entire column can be NaN by construction, and the Top/Bottom 10 tables will then render `None`/`NaN` everywhere in that column. (PNG `10_clv_prediction.png` confirms the `churn_probability` column in both tables is empty / blank, the same symptom the audit flagged.)
- F5's fix log does not list a Page-10 Top/Bottom-10 NaN cleanup as a closed defect.

---

## 7. P12 — Customers Retained = 122.29548658078494 (14-decimal float) — `format_count` integer applied

**Verdict: FIXED.**

- F1 added `format_count(value, integer=True, suffix="")` at `dashboard_helpers.py:157-186` with explicit comment: "Closes the `Customers Retained = 122.29548658078494` 14-decimal float leak audited on Page 12: when `integer=True` (default) the value is floored to an int and rendered with thousand separators".
- F1 also added `customers_retained_int(value)` in `calculations.py` as a second-line safety net at the calculation layer.
- F5 wired the call at `app.py:2930`: `retained_display = format_count(total_retained, integer=True)` and at `app.py:2952` `bc3.metric("Customers Retained", retained_display, help=...)`. The same formatted value is reused in the lower ROI summary table at `app.py:3113`, so the float leak cannot reappear there either.
- PNG `12_clv_retention_campaign.png` confirms the "Customers Retained" KPI now renders as a clean integer (visible "122" with no decimal tail).

---

## 8. P12 — Overall ROI 3.8x = low_value_persuadable's segment ROI 3.8x (coincidence/bug)

**Verdict: FIXED (as math). PARTIAL (as disclosure).**

- The Overall ROI on P12 is now computed by `compute_overall_roi(total_rev_saved, total_allocated, scope_label="budget")` at `app.py:2934`. This is the literal `revenue_saved / total_budget` quotient (e.g. `192,155,554 / 50,000,000 = 3.84x`), which the iter9 audit had already cross-checked as arithmetically correct. The 3.8x is therefore a real budget-envelope ROI, not a stray segment value — the audit's "either coincidence or bug" question is resolved as **coincidence** (the segment-level low_value_persuadable also happens to land near the budget-envelope ratio).
- The new `roi_info["display"]` is rendered with two decimals (`"3.84x"`) instead of one (`"3.8x"`), so the bit-for-bit visual match with the segment table's "3.8x" row is broken — an analyst can now see the values are not literally the same number.
- However, the audit's underlying ask was a reconciliation footnote stating "the budget-envelope ROI happens to equal a per-segment ROI". The new help tooltip says only "Scope: budget envelope. Computed as 192,155,554 ÷ 50,000,000. Pages 09 and 13 use different scopes (treated-only) and may show different ROI values…" — it does not call out the segment-level coincidence. So the "is this a bug?" question is implicitly answered (it isn't) but not explicitly footnoted.

---

## 9. P12 — Section 1 (behavioral segments) vs Sections 2-4 (uplift segments) taxonomy mix

**Verdict: NOT FIXED.**

- `app.py:2761-2818` (Section 1 — "Customer Lifetime Value Overview") still groups by `predictions["segment"]` for the bubble chart at L2798, and (per iter9 audit page-data dump) Section 1's segment column resolves to behavioral labels (`vip_loyal`, `dormant`, etc.) sourced from the predictions dataframe, while Sections 2-4 below use `uplift_data["segment"]` / `budget_data["segment"]` which contain uplift labels (`high_value_persuadable`, `sleeping_dog`, etc.).
- No code path in `render_retention_campaign` (L2706-3183) maps either taxonomy onto the other, no joint key is computed, and no `st.caption` warns the reader that the segment columns in Section 1 vs Sections 2-4 are different taxonomies.
- Neither f4 nor f5 fix logs claim this defect was closed (f5 lists 21 closed defects; "P12 taxonomy unification" is not among them).
- A reader cannot trace a single customer between Section 1 and Sections 2-4 of the same page.

---

## Summary table

| # | Defect | Verdict |
|---|---|---|
| 1 | P09 dual "Avg Expected Uplift" label collision | FIXED |
| 2 | P09 16,106 high priority vs 3,398 coupon reconciliation | FIXED |
| 3 | Cross-page Overall ROI (P05 / P09 / P12) — single helper | PARTIAL (P05 still uses unlabelled `Avg ROI` mean-of-segments) |
| 4 | P10 n=1 / n=2 segments — count emphasis | PARTIAL (count column in table; chart still untouched) |
| 5 | P10 CLV-vs-churn x-axis clamp to [0, 1] | NOT FIXED |
| 6 | P10 Top/Bottom 10 churn_probability NaN cleanup | NOT FIXED |
| 7 | P12 Customers Retained 14-decimal float | FIXED |
| 8 | P12 Overall ROI = segment ROI 3.8x | FIXED (math) / PARTIAL (footnote) |
| 9 | P12 behavioral vs uplift taxonomy mixing | NOT FIXED |

## Buyer-trust blocker status (from iter9 A5 §"Top 3")

1. Three "Overall ROI" values cross-page — **2/3 closed** (P09 + P12 declare scope; P05 does not). Cross-page glossary is not yet whole.
2. `Customers Retained = 122.29548658078494` — **fully closed.**
3. Tiny segments (n=1, n=2) masquerading as headline segments — **partially closed** for P10 (count column visible), **fully closed** for P09 (reconciliation banner).

## Top remaining gaps for iter11

- **Page 05** "Avg ROI" must be relabelled (route through `compute_overall_roi(scope_label="segment_avg")` or footnote that the tile is mean-of-per-segment-ROI). Until then the cross-page ROI story is still incoherent.
- **Page 10** scatter axis (`render_clv` L2328-2348): add `range_x=[0, 1]` and a `predictions = predictions[predictions["churn_probability"].between(0, 1)]` (or `.clip(0, 1)`) before plotting.
- **Page 10** Top/Bottom 10 tables (L2413-2427): drop or mask NaN churn_probability rows, or render a placeholder "—" via `format_percentage` rather than the bare NaN.
- **Page 10** segment chart (L2289-2299): annotate bar height with sample size, or hide bars with n < 5, or add a sample-size caption.
- **Page 12** Section 1 vs Sections 2-4: either project the behavioral taxonomy into the uplift taxonomy via a join key, or render a visible cross-walk caption stating the two sections use different segment columns.
