# F4 — Page 09 Recommendations Fix Log (iter10)

**Agent:** F4 (parallel remediation lane 4 of 5)
**File modified (only):** `C:\Users\yoonc\Capstone-Design-1\src\dashboard\recommendations_view.py`
**Audit input:** `_test_results\iter9\iter9_audit_a5.md` (§ Page 09)
**Page dump input:** `_test_results\iter9\page_data\09_recommendations.md`

## Defects addressed

### D1 — Two KPI cards labeled identically "Avg Expected Uplift"
**Audit:** Top of page shows 6.36% (across all 20,000 customers). Mid-page Cost-Benefit strip shows 10.88% (across the 3,398 treated customers). Same label, different values, no scope qualifier. (A5 §"Wrong / Unreliable" #1.)

**Fix:**
- Top KPI strip (`_render_kpi_cards`): renamed to **"Avg Predicted Uplift (all customers)"** with a `help=` tooltip stating the population is the full base including `no_action`, and explicitly cross-referencing the mid-page treated-only card.
- Mid KPI strip (`_render_cost_benefit_analysis`): renamed to **"Avg Treated Uplift"** with a `help=` tooltip declaring the population is the treated subset only and pointing back to the top card.

The two KPIs now have distinct labels, distinct populations, and reciprocal tooltips so an analyst can audit the gap from either card without reading source.

### D2 — High Priority 16,106 vs Coupon recipients 3,398 (12,708-customer gap unexplained)
**Audit:** 79% of "high priority" customers silently land in `no_action`. (A5 §"Wrong / Unreliable" #2.)

**Fix:** Added an inline **"Priority vs treated reconciliation"** info banner immediately under the top KPI strip. It computes:
- `high_priority_count` (priority_score ≥ 0.7),
- `hp_treated` (intersection of high-priority and `recommendation_type != "no_action"`),
- `hp_no_action = high_priority_count - hp_treated`,
and renders sentence: "Of N high-priority customers, T (X%) receive a treatment offer; H (Y%) get `no_action` because their predicted uplift × CLV did not exceed the cost threshold for any offer in the catalog. Total treated across all priorities: M." All counts go through `format_count`.

### D3 — "Overall ROI 9.0x" without denominator footnote
**Audit:** Page 05 reports 3.5x, Page 09 reports 9.0x, Page 12 reports 3.8x — same label, three different denominators, no glossary. (A5 §"Wrong / Unreliable" #3 + cross-page table.)

**Fix:** The mid-page ROI tile is now built by `compute_overall_roi(revenue_saved, total_cost, scope_label="treated")`:
- The metric **label** is sourced from `roi_info["label"]` → renders as **"ROI (treated only)"**, replacing the ambiguous "Overall ROI".
- The metric **value** uses `roi_info["display"]` (e.g. `"9.00x"`).
- The metric **help** tooltip surfaces `roi_info["tooltip"]` (the literal `revenue ÷ cost` division) **and** explicitly states that Page 05 uses the full budget envelope and Page 12 uses the planned budget — so the three values are reconciled, not contradictory.

### D4 — `Avg Uplift by Action: no_action = 4.36%`
**Audit:** Realized uplift on customers who received no treatment is 0 by definition; the 4.36% is "predicted uplift if treated" for the not-treated population, plotted on an axis labeled "Average Expected Uplift". (A5 §"Wrong / Unreliable" #5.)

**Fix:** In `_render_uplift_analysis` the right-hand bar chart (`Average Expected Uplift by Action`):
- The `no_action` row is **dropped** from the dataframe before plotting.
- The chart title is renamed to **"Average Predicted Uplift by Treated Action"** and the y-axis label to **"Avg Predicted Uplift (treated)"**.
- A `st.caption(...)` immediately under the chart explains why `no_action` is excluded. The companion box plot keeps `no_action` for distribution visibility but is now titled **"Predicted Uplift Distribution by Action Type"** with the y-axis explicitly labeled **"Predicted Uplift (if treated)"** so the meaning is unambiguous.

### D5 — Float-precision / number rendering
**Audit:** Customer-facing KPI cards must not leak full IEEE-754 floats (cf. Page 12's `122.29548658078494`). All counts/currency on Page 09 should route through helpers.

**Fix:**
- `Total Recommendations`, `High Priority`, and the reconciliation banner counts use `format_count(...)`.
- `Total Campaign Cost` and `Est. Revenue Saved` cards now render as `format_currency_krw(...)` (e.g. `₩1.21M (KRW)`, `₩10.89M (KRW)`) — eliminates the previous raw `1,211,055 KRW` long-string rendering and matches the formatter standardised by F1.
- All remaining f-strings on the page use explicit width/precision specifiers (`:.2%`, `:,.0f`, `:,.1f`); a grep audit confirms no `f"{x}"` raw-float leaks.

## Defensive helper imports
`compute_overall_roi`, `format_count`, `format_currency_krw` are imported from `src.dashboard.utils.dashboard_helpers` inside a `try/except`, with inline minimal fallbacks defined at module load time so the page renders even if F1's helpers move or temporarily break. The fallbacks preserve the same return shapes (dict for `compute_overall_roi`, str for the formatters).

## Constraints honoured
- Modified ONLY `src/dashboard/recommendations_view.py`. No edits to `app.py`, `monitoring_view.py`, `system_health_view.py`, or `dashboard_helpers.py`.
- No new features added — only the five audit defects closed.
- Module byte-compiles cleanly (`python -c "import ast; ast.parse(open(...).read())"` → OK).

## Verification
- `python -c "import ast; ast.parse(open('src/dashboard/recommendations_view.py').read())"` → **OK**.
- Grep for raw f-string number renders (`f"\{[^}]*\}"`): all 7 remaining matches use explicit format specs (`:.2%`, `:,.0f`, `:,.1f`, `:,`) — no precision leaks.
- The reconciliation banner is computed from the same `recs` dataframe used by the KPI cards, so the numbers it shows are guaranteed consistent with the cards above it (no separate query path).

## Files changed
1. `C:\Users\yoonc\Capstone-Design-1\src\dashboard\recommendations_view.py` — defensive helper import block (top), `_render_kpi_cards` (relabel + reconciliation banner), `_render_uplift_analysis` (drop `no_action` from avg-by-action bar, retitle box plot), `_render_cost_benefit_analysis` (`compute_overall_roi`, "Avg Treated Uplift" rename, `format_count` / `format_currency_krw` everywhere).

## Files written
1. `C:\Users\yoonc\Capstone-Design-1\_test_results\iter10\fix_logs\f4_recommendations.md` (this log).
