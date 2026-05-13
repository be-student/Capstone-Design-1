# V6 — Real-Time Scoring (Page 13, 3 tabs) — iter10 verification

**Inputs**
- Issues source: `_test_results/iter9/iter9_audit_a6.md`
- Fix logs: `_test_results/iter10/fix_logs/f5_app.md`, `_test_results/iter10/fix_logs/f1_helpers.md`
- Screenshots: `_test_results/dashboard_pages/13_realtime_scoring_a_live.png`,
  `13_realtime_scoring_b_offers.png`, `13_realtime_scoring_c_monitoring.png`

Verdict legend: FIXED / PARTIAL / NOT FIXED / REGRESSION.

---

## 1. Tab a — Stream 0/0 vs Total Scores 200 label disambiguation
**Verdict: FIXED.**

Screenshot `13_realtime_scoring_a_live.png` Service Health row now reads:
- `Request queue depth (current)` = **0** (with ⓘ tooltip)
- `Response queue depth (current)` = **0** (with ⓘ tooltip)
- `Consumer Group` = `scoring_consu...`
And Recent Scoring History card is now labeled `Total Scores (lifetime)` = **200**.

The two units (queue depth vs lifetime processed) are now linguistically separated on the surface. Matches `f5_app.md` defect 11 ("`app.py:3563-3567` and `:3654` … renamed to Request queue depth (current), Response queue depth (current), Total Scores (lifetime), each with help= tooltip"). The headline contradiction is resolved.

---

## 2. Tab a — throughput dated Oct 15–16 2024 on a "real-time" page
**Verdict: NOT FIXED.**

Screenshot Tab a still plots `Scoring Requests per Minute` and `Response Latency & Error Rate` against an x-axis labeled **Oct 15, 2024 → Oct 16, 2024**. No "as of" caption, no staleness banner, no fixture-data warning.

`f5_app.md` explicitly lists this under "Defects skipped (out of scope or owned by another agent)" — *"Realism / fixture-data issues on Page 13 (Oct-2024 timestamps, uniform-4 scoring volume) — the dashboard correctly displays the loader output; replacing the fixtures with live telemetry is outside the dashboard's responsibility."* The dashboard surface still presents 19-month-old timestamps as live throughput with no warning.

---

## 3. Tab b — ROI 8.0x card vs math 8.985x
**Verdict: FIXED.**

Screenshot `13_realtime_scoring_b_offers.png` headline KPI strip now reads:
- Total Offers = `44 / 200`
- Total Cost = `1,196,659 KRW`
- Expected Revenue Saved = `10,752,341 KRW`
- ROI (treated only) = **`8.99x`**

10,752,341 / 1,196,659 = 8.985 → displays correctly as 8.99x. The label "ROI (treated only)" is visible on the card. Matches `f5_app.md` defect 9 (`compute_overall_roi(total_revenue, total_cost, scope_label="treated")`). The off-by-one `(roi - 1)` unit error from iter9 is gone.

---

## 4. Tab b — Total Offers 44 denominator missing
**Verdict: FIXED.**

Screenshot Tab b card now reads `Total Offers` = **`44 / 200`**. The "200 recently scored" denominator is on the card itself; matches `f5_app.md` defect 10 (`f"{offers_n:,} / {scored_n:,}"` with help= tooltip showing the percentage).

---

## 5. Tab b — Quick Lookup Priority 1.00 + no_action + uplift 1.46%
**Verdict: FIXED (deferred-render variant).**

Screenshot Tab b now shows the Quick Recommendation Lookup section with:
- Header `Quick Recommendation Lookup`
- Selectbox showing `— Select a customer to see their recommendation —`
- A caption (light grey) prompting the user to pick a customer.

No customer is currently selected in the screenshot, so the contradictory `Priority 1.00 + no_action + uplift 1.46%` cluster is not rendered. Per `f5_app.md` defect 12, when a customer *is* picked, the offending "Priority Score" card is renamed to **Risk Score** (with tooltip "this is the customer's churn-risk score — formerly mislabelled Priority") and a fourth card `Action EV (uplift × CLV)` is added that drives offer selection. The semantic re-wiring is documented and the visible regression vector (auto-selected first customer) is removed in this screenshot. Verdict treats both halves of the iter9 issue as closed: (a) Priority renamed to Risk Score, (b) Action EV column introduced.

---

## 6. Tab b — Banner "Recommended Offer: no_action" defaulting globally
**Verdict: FIXED.**

In iter9 the Page 13 surface carried a top-of-page banner `Recommended Offer: no_action` rendered before any selection. Iter10 screenshots (all three tabs) show only the synthetic-data banner: `Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated.` The Quick Lookup selectbox now defaults to the sentinel `— Select a customer to see their recommendation —` and the recommendation banner is suppressed until the user explicitly picks a customer. Matches `f5_app.md` defect 13 ("Prepended a sentinel … Banner only fires after the user explicitly picks a customer").

---

## 7. Tab c — drift trend 1 datapoint on 1.5 ms axis
**Verdict: FIXED.**

Screenshot `13_realtime_scoring_c_monitoring.png` shows immediately under the KPI strip a blue st.info banner:

> *Insufficient history — need ≥5 observations, have 1. Drift trend charts require at least 5 observations spanning ≥ 1 hour. Showing the alert summary above; trend lines will appear once history accumulates.*

The previous Drift Alert Timeline / PSI Trend / KS Statistic Trend charts (which iter9 rendered as single points on a 1.5 ms axis) are no longer drawn. Drift Detection History expander is preserved. Matches `f5_app.md` defect 14 + `f1_helpers.md` `drift_trend_guard(timeseries, min_points=5)` requiring n ≥ 5 AND span ≥ 3600s.

---

## 8. Tab c — Scoring Volume = 4 uniformly across ~50 buckets
**Verdict: NOT FIXED.**

Screenshot Tab c "Scoring Quality Metrics" still renders **`Scoring Volume Over Time`** as a bar chart of ~50 hourly buckets that visually appear uniform (~4 each, dated Oct 11-15 2024). No "synthetic placeholder — replace with live consumer-group counters" caption, no relabel. The `drift_trend_guard` was applied to the drift trio but the same guard / staleness label was *not* surfaced on the Scoring Volume chart even though `f5_app.md` defect 14 claims "Same guard then applied to the Scoring Volume Over Time / Mean Churn Probability Over Time block lower on the same tab." Visually the block still draws. The fixture-data nature is also explicitly out-of-scope per the F5 "Defects skipped" section.

---

## 9. Tab a vs Tab c clock mismatch (Oct 2024 vs May 2026)
**Verdict: NOT FIXED.**

Tab a charts: x-axis labeled **Oct 15-16 2024**.
Tab c "Mean Churn Probability Over Time": x-axis labeled **Oct 11-15 2024**.
Tab c drift KPI metric (`Total Drift Checks = 1`, `Latest Alert Level = RED`) is the only surface dated to system clock (May 2026). The two-clock incoherence iter9 flagged is unchanged on the visible surface. Same root cause as #2 / #8: F5 declared this fixture-data class out-of-scope.

---

## 10. All tabs — no model_version pinned
**Verdict: NOT FIXED.**

Across all three screenshots:
- Tab a Recent Scoring History `Primary Model` = **`ensemble`** (no version string).
- Tab b headline KPI strip carries no policy or model version.
- Tab c `Model Type Usage in Recent Scoring` pie shows ensemble/lightgbm/xgboost mix but no version, no training date, no last-retrain timestamp.

`f5_app.md` does not list a "model_version stamp" defect among the 21 closed entries. This iter9 ask is still open.

---

## Cross-tab consistency snapshot (post-iter10)

| Surface | Iter10 timestamp / count | Iter9 issue addressed? |
|---|---|---|
| Tab a — Throughput chart x-axis | Oct 15–16 2024 (unchanged) | No |
| Tab a — Service Health card | "queue depth (current)" labels | Yes (#1) |
| Tab a — Recent Scoring | "Total Scores (lifetime) = 200" | Yes (#1) |
| Tab b — Total Offers card | `44 / 200` | Yes (#4) |
| Tab b — ROI card | `8.99x` (label: ROI (treated only)) | Yes (#3) |
| Tab b — Quick Lookup default | sentinel "— Select a customer —" | Yes (#5, #6) |
| Tab b — Page banner | synthetic-data banner only | Yes (#6) |
| Tab c — Drift trends | suppressed + st.info guard banner | Yes (#7) |
| Tab c — Scoring Volume | still uniform ~4 bars, Oct 2024 | No (#8) |
| Tab c — Mean Churn over Time | Oct 11–15 2024 (unchanged) | No (#9) |
| All tabs — model_version stamp | absent | No (#10) |

---

## Tally

| Verdict | Issues |
|---|---|
| **FIXED** | #1, #3, #4, #5, #6, #7 (6) |
| **PARTIAL** | (0) |
| **NOT FIXED** | #2, #8, #9, #10 (4) |
| **REGRESSION** | (0) |

Closed-by-fix issues are all UI-layer label / math / guard fixes attributable to F5
(`app.py`) and F1 (`compute_overall_roi`, `drift_trend_guard`). Open issues are all
the data-realism / version-stamp class that F5's log explicitly deferred as
"loader output, not dashboard responsibility." From a SaaS-readiness lens, the
KPI-integrity blockers iter9 called out (stream contradiction, ROI math, lookup
contradiction, single-point drift trend) are now closed; the time-anchor
incoherence and the missing model-version stamp remain open and continue to
disqualify the page from "production-ready" status.
