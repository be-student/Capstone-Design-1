# iter10 Verification V2 — Pages 03 / 04 / 05

**Agent:** Verification Agent V2
**Inputs cross-referenced:**
- Issues source: `_test_results/iter9/iter9_audit_a2.md`
- Fix logs: `_test_results/iter10/fix_logs/f5_app.md`, `_test_results/iter10/fix_logs/f1_helpers.md` (also referenced f4 for P05 ownership confirmation)
- PNGs (iter10): `_test_results/dashboard_pages/03_customer_segmentation.png`, `04_cohort_analysis.png`, `05_budget_optimization.png`
- Code spot-checks: `src/dashboard/app.py` (`render_customer_segmentation` ~line 886, `render_budget_optimization` lines 898–1180)

Verdict legend: **FIXED / PARTIAL / NOT FIXED / REGRESSION**

---

## Page 03 — Customer Segmentation

### Issue 1 — Definitions table mismatch (`regular_loyal` & `dormant` missing; 4 of 6 chart segments un-defined)

- **Expected fix evidence:** definitions table widened to include `regular_loyal` and `dormant` (the headline "Highest Risk Segment"), or at minimum a mapping row that ties the 8 config-driven names to the 6 charted segments.
- **PNG evidence:** "Segment Definitions & Retention Actions" table renders the same 8 rows as iter9: `vip_loyal, loyal_customer, potential_loyalist, at_risk, hibernating, explorer, new_customer, bargain_hunter`. `regular_loyal` and `dormant` are still absent. Chart segments still read `vip_loyal, regular_loyal, bargain_hunter, new_customer, explorer, dormant` (donut + bar + statistics table). Same 4-of-6 overlap as iter9.
- **Fix-log evidence:** Neither `f5_app.md` nor `f1_helpers.md` lists a P03 entry. The fix list in F5 explicitly skips P03 (no P03 sections among "P0 defects closed" or "P1 defects closed").
- **Code evidence:** `src/dashboard/app.py:884–895` still builds the table directly from `config["segmentation"]["segments"]` with no augmentation for `regular_loyal` / `dormant` and no schism-bridging mapping.

**Verdict: NOT FIXED** — defect carried forward verbatim from iter9.

---

## Page 04 — Cohort Analysis

### Issue 2 — Retention monotonicity violation (Apr 2024 P7 91.0% → P8 92.1%) clamped or flagged

- **Expected fix:** clamp the cell, OR display an explicit warning / asterisk so the violation is acknowledged.
- **PNG evidence:** A yellow warning banner is rendered above the heatmap: *"Retention monotonicity violations detected — retention must be non-increasing within a cohort by construction. Affected cells are flagged with red asterisks in the heatmap below: 2024-04: P7 91.0% → P8 92.1%."* Apr 2024 row in the Retention Matrix (Raw Data) still shows 91.0 and 92.1 (un-clamped, but called out).
- **Fix-log evidence:** F5 entry #16 ("Page 04 — retention monotonicity violations flagged") describes exactly this implementation: walk `(prev → curr)` pairs, render `st.warning` listing up to three offending pairs, render `"—"` for unobserved cells.

**Verdict: FIXED** — flagged (not clamped), which matches the agreed remediation pattern. The exact Apr 2024 P7 → P8 pair from iter9 is named in the banner.

### Issue 3 — "Avg Final Retention 2.5%" recomputed without zero-fill

- **Expected fix:** mask trailing zeroed/unobserved cells out of the headline KPI; ideally fall back to "Avg Deepest-Observed Retention" wording when every cohort's last column is unobserved.
- **PNG evidence:** Headline KPI now reads **"Avg Final Retention 10.2%"** with an info icon. This matches iter9's only-real-cell value (Jan 2024 P12 = 10.2%), confirming the unobserved Feb/Mar/Apr trailing zeros were excluded from the average. Heatmap unobserved cells now render as `"—"` instead of `0.0`.
- **Fix-log evidence:** F5 entry #17 documents `unobserved_mask`, conditional fallback to "Avg Deepest-Observed Retention" wording, and the `"—"` display for unobserved cells.
- **Caveat:** The label still reads "Avg Final Retention" rather than the documented "Avg Deepest-Observed Retention" fallback wording, because at least one cohort (Jan 2024) has an observed P12. Per F5 spec this is the intended behavior, not a regression.

**Verdict: FIXED** — KPI no longer 2.5%; reads 10.2% which is the truthful unobserved-mask average.

### Issue 4 — Only 4 cohorts banner

- **Expected fix:** explanatory banner when `n_cohorts < 6`.
- **PNG evidence:** A blue info banner is rendered between the KPI strip and the monotonicity warning: *"Limited cohort window — only 4 monthly cohorts are available. Production cohort analysis typically uses ≥6–12 cohorts; generate more historical data for trend reliability."*
- **Fix-log evidence:** F5 entry #18 documents this exact banner copy.

**Verdict: FIXED.**

---

## Page 05 — Budget Optimization

### Issue 5 — "Avg ROI 3.5x" vs aggregate 3.84x (single ROI helper)

- **Expected fix:** route the headline through `compute_overall_roi(scope_label=...)` so the displayed value either equals the aggregate `revenue_saved / total_allocated = 3.84x` OR carries an explicit "mean of segment ROIs" label and tooltip.
- **PNG evidence:** Allocation Summary KPI strip still reads **"Avg ROI 3.5x"** as a bare label (no scope qualifier, no tooltip surfaced in screenshot). Total Allocated 50,000,000 KRW and Revenue Saved 192,155,551 KRW are unchanged, so the aggregate is still 3.843x. Same 3.5x vs 3.84x split as iter9.
- **Code evidence:** `src/dashboard/app.py:1002,1007` still computes `avg_roi = display_results["roi"].mean()` and renders `f"{avg_roi:.1f}x"` with metric label `"Avg ROI"`. No call to `compute_overall_roi` inside `render_budget_optimization` (the only `compute_overall_roi` call sites in app.py are at lines 2934 and 4211 — Pages 12 and 13).
- **Fix-log evidence:** F5 explicitly disclaims P05: *"P05/P09 ROI labels — owned by F2/F3/F4. F5 only fixes the Page 12 side …"* And F4's log only modifies `recommendations_view.py` (Page 09). No agent in this iter touched `render_budget_optimization`.

**Verdict: NOT FIXED** — defect carried forward; explicitly out-of-scope per F5, and no other agent picked it up.

### Issue 6 — Expected Retained 118 vs Baseline-scenario 122 reconciled

- **Expected fix:** make the headline KPI agree with the "Current Selection" / Baseline row of the scenario table, OR explain the 4-customer gap.
- **PNG evidence:** Allocation Summary still shows **Expected Retained 118**. What-If Scenario Comparison table still shows Baseline = 122 and Current Selection = 122. Same 118 vs 122 discrepancy as iter9. No reconciliation banner.
- **Code evidence:** `total_retained = display_results["expected_retained"].sum()` (line 1000) is rendered as `f"{total_retained:,}"` (line 1005) — no integer-helper, no reconciliation against `comparison_df`. The comparison table is built from a separate `_compute_scenario_comparison` path (line 1164) and has not been aligned.
- **Fix-log evidence:** No P05 entry in any fix log addresses this 118 vs 122 mismatch.

**Verdict: NOT FIXED.**

### Issue 7 — `high_value_persuadable` 31,000 KRW (8x ROI) — LP constraint surfaced

- **Expected fix:** an annotation, badge, or banner explaining why a segment with the chart's highest ROI receives 0.062% of spend (segment-size cap, min-spend rule, etc.), or surface the binding LP constraint.
- **PNG evidence:** Budget Allocation by Segment table still shows `high_value_persuadable` at 31,000 KRW. ROI by Segment chart still shows it as the tallest bar (≈ 8x). No "constraint binding" annotation, no caption, no banner explaining the 31k floor anywhere on the page.
- **Fix-log evidence:** No fix log addresses this.

**Verdict: NOT FIXED.**

### Issue 8 — Channel-Level Cost Breakdown empty H3 hidden or filled

- **Expected fix:** either (a) hide the section when `budget.channels` is missing from config, or (b) populate it via config additions and a working visualization.
- **PNG evidence:** "Channel-Level Cost Breakdown" H3 is still rendered with the same blue info box: *"Channel configuration not found in config. Add budget.channels to simulator_config.yaml for multi-channel allocation views."* Empty section persists.
- **Code evidence:** `src/dashboard/app.py:1068` always renders the subheader; the `else:` branch at line 1140 emits the same info banner unchanged. No hide-when-empty logic.
- **Config note:** `config/simulator_config.yaml` is in the repo's modified list at session start, but no fix log claims `budget.channels` was added; the PNG would render channel charts (not the info banner) if config had been populated.

**Verdict: NOT FIXED.**

---

## Summary table

| # | Page | Issue | Verdict |
|---|------|-------|---------|
| 1 | 03 | Definitions table omits `regular_loyal` & `dormant` | NOT FIXED |
| 2 | 04 | Apr 2024 P7→P8 monotonicity violation flagged | **FIXED** |
| 3 | 04 | Avg Final Retention recomputed (no zero-fill) | **FIXED** (now 10.2%) |
| 4 | 04 | Only 4 cohorts — banner | **FIXED** |
| 5 | 05 | Avg ROI 3.5x vs aggregate 3.84x | NOT FIXED |
| 6 | 05 | Expected Retained 118 vs Baseline 122 | NOT FIXED |
| 7 | 05 | high_value_persuadable 31k @ 8x ROI — LP surfacing | NOT FIXED |
| 8 | 05 | Channel breakdown empty H3 | NOT FIXED |

**Tally:** 3 FIXED / 0 PARTIAL / 5 NOT FIXED / 0 REGRESSION.

**Cross-cutting observation:** all three remediated issues sit on Page 04, which F5 explicitly claimed (entries #16, #17, #18 in `f5_app.md`). Page 03 and Page 05 have zero remediation activity in iter10 — F5 disclaims P05 ROI ownership to "F2/F3/F4", but F2 only touches `system_health_view.py`, F3 only touches `monitoring_view.py`, and F4 only touches `recommendations_view.py` (Page 09). No agent in iter10 owned `render_budget_optimization` or `render_customer_segmentation`, so all five P03/P05 defects from iter9 are carried forward intact.
