# F5 — `src/dashboard/app.py` fix log (iter10)

Scope: Pages 02, 04, 07, 11, 12, 13, 14. F5 is the only agent permitted to
touch `src/dashboard/app.py` in this iteration.

Helpers consumed (defensively imported with fallbacks in case earlier-stage
agents have not landed them yet):

* `format_count(value, integer=True)`
* `compute_overall_roi(revenue_saved, cost_or_budget, scope_label=...)`
* `drift_trend_guard(timeseries, min_points=5)`
* `format_currency_krw(x)`
* `system_health_view.check_mlflow_health(config)` (re-exported via the new
  `_probe_mlflow_status` shim so Page 14's banner stays in lockstep with
  Page 15).

---

## P0 defects closed

### 1. Page 02 — headline P/R must match the confusion matrix

* **Audit:** `iter9_audit_a1.md` Page 02, "Wrong / suspicious — CRITICAL".
* **Symptom:** ml_model headline P/R `0.5331 / 0.7791` vs confusion matrix
  `0.7059 / 0.6000`.
* **File:line — before:** `app.py:344-353` — headline KPI strip read
  `metrics["ml_model"]["precision"]` etc. directly from the loader.
* **After:** Re-derive Precision / Recall / F1 / Accuracy from the
  displayed confusion-matrix counts (TP/FP/FN/TN) before rendering the
  KPI strip. The bar chart, radar, and improvement table downstream all
  read from the same recomputed `metrics` dict, so headline ⇔ matrix can
  no longer disagree.
* **Closes:** P02 #1.

### 2. Page 02 — confusion-matrix test-set size disclosure

* **Audit:** `iter9_audit_a1.md` Page 02, "Wrong / suspicious — CRITICAL"
  bullet 2 (matrices total 600 vs 20,000-customer dataset).
* **Symptom:** Matrices report n=600 (3% of population) without disclosure.
* **File:line — before:** `app.py:436-466` — confusion-matrix subheader
  rendered without any sample-size caption.
* **After:** Inserted a `st.caption(...)` directly under the subheader
  showing `Test set size: N samples (Y% of M customers). Headline
  Precision / Recall / F1 above are computed from these confusion
  matrices on the same test split.`
* **Closes:** P02 #2.

### 3. Page 02 — significance disclaimer for Best-Model claim

* **Audit:** `iter9_audit_a1.md` Page 02 bullet 3 (`AUC margin 0.0014, no
  DeLong test`).
* **File:line — before:** `app.py:344-365` — Best Model card was a
  simple `st.metric("Best Model", best_model[0])` with no disclaimer.
* **After:** Added a `help=` tooltip on the Best Model metric AND a
  visible `st.caption("ℹ️ AUC spread across the three models is
  Δ=0.0014 (<0.005) …")` whenever the spread is below 0.005, so the
  ranking is explicitly framed as "indicative, not significant".
* **Closes:** P02 #3.

### 4. Page 12 — `Customers Retained = 122.29548658078494` float leak

* **Audit:** `iter9_audit_a5.md` Page 12.
* **File:line — before:** `app.py:2686` — `bc3.metric("Customers
  Retained", f"{total_retained:,}")` (raw float spilled with 14 decimal
  digits when `total_retained` is a numpy float).
* **After:** Use `format_count(total_retained, integer=True)` from the
  iter10 helper. Same fix applied to the lower ROI summary table.
* **Closes:** P12 #4.

### 5. Pages 05/09/12 — three "Overall ROI" values

* **Audit:** `iter9_audit_a5.md` cross-page money table.
* **File:line — before:** `app.py:2687` Page 12 read `overall_roi =
  total_rev_saved / total_allocated` and rendered as `f"{overall_roi:.1f}x"`
  with no scope label.
* **After:** Page 12 now calls `compute_overall_roi(total_rev_saved,
  total_allocated, scope_label="budget")` and renders
  `roi_info["display"]` with `roi_info["label"]` ("ROI (budget envelope)")
  plus a `help=` tooltip showing the literal division and pointing the
  user at Page 09 / Page 13's treated-only scope. (Pages 05 / 09 are
  outside F5 scope; their ROI labels are owned by other agents but Page
  12 now declares its own scope explicitly.)
* **Closes:** P12 #5.

### 6. Page 11 — Avg Uplift Score == Avg Treatment Effect duplicate

* **Audit:** `iter9_audit_a3.md` Page 11.
* **File:line — before:** `app.py:2284-2327` had four KPIs (`Avg Uplift
  Score`, `Avg Treatment Effect`, `Persuadable`, `Sleeping Dogs`) AND
  two side-by-side histograms (`Distribution of Uplift Scores` /
  `Distribution of Treatment Effects`) that were byte-identical.
* **After:** Removed the `Avg Treatment Effect` KPI and the duplicate
  treatment-effect histogram. The remaining single histogram carries a
  `st.caption` explaining that `treatment_effect == uplift_score` in
  this build and the dedicated ATE estimator is a future-build
  placeholder.
* **Closes:** P11 #6.

### 7. Page 11 — 4-quadrant collapse on the headline

* **Audit:** `iter9_audit_a3.md` Page 11 (`16,317 + 3,683 = 20,000` only
  shows 2 of 4 quadrants).
* **File:line — before:** `app.py:2284-2293` rendered only `Persuadable`
  + `Sleeping Dogs` on the headline.
* **After:** Headline now uses 5 columns (`Avg Uplift Score`,
  `Persuadable`, `Sure Thing`, `Sleeping Dog`, `Lost Cause`) sourced
  from the canonical `segment` column when available; falls back to a
  sign-of-uplift derivation if the column is missing.
* **Closes:** P11 #7.

### 8. Page 11 — vocabulary unification

* **Audit:** `iter9_audit_a3.md` (pie says "Persuadable / Lost Cause",
  headline says "Persuadable / Sleeping Dogs", table shows 4 labels).
* **File:line — before:** `app.py:_classify_customer` derived response
  classes from sign of `uplift_score` × `treatment_effect`, producing a
  vocabulary that didn't match the segment table.
* **After:** When the upstream `segment` column is present, classify by
  mapping the segment string through `label_map` (the canonical
  4-quadrant vocabulary). Sign-derivation only kicks in as a fallback.
  Pie chart, response-class bar chart, and headline now share one
  vocabulary.
* **Closes:** P11 #8.

### 9. Page 13 tab b — ROI math

* **Audit:** `iter9_audit_a6.md` Tab b (`8.0x` displayed but
  `10,752,341 / 1,196,659 = 8.985x`).
* **File:line — before:** `app.py:3780-3781` computed
  `roi = (total_revenue / max(total_cost, 1)) - 1` and rendered
  `f"{roi:.1f}x"` (subtracting 1 was a unit error).
* **After:** Switched to `compute_overall_roi(total_revenue, total_cost,
  scope_label="treated")`; the helper returns `display="8.99x"` (no
  off-by-one) with a label "ROI (treated only)" and a tooltip showing
  the literal division.
* **Closes:** P13 #9.

### 10. Page 13 tab b — denominator on `Total Offers`

* **Audit:** `iter9_audit_a6.md` Tab b (`44 of 200 recently scored = 22%`).
* **File:line — before:** `app.py:3771` `st.metric("Total Offers",
  format_count(len(filtered)))` (no denominator).
* **After:** When the loader returns a non-empty scoring history, render
  as `f"{offers_n:,} / {scored_n:,}"` with a `help=` tooltip stating
  the percentage. Falls back to the bare count when scoring history is
  unavailable.
* **Closes:** P13 #10.

### 11. Page 13 tab a — queue-depth vs lifetime-totals labels

* **Audit:** `iter9_audit_a6.md` Tab a (`Request Stream: 0 / Response
  Stream: 0` next to `Total Scores: 200`).
* **File:line — before:** `app.py:3563-3567` and `:3654` used
  `Request Stream`, `Response Stream`, `Total Scores`.
* **After:** Renamed to `Request queue depth (current)`,
  `Response queue depth (current)`, and `Total Scores (lifetime)`, each
  with a `help=` tooltip clarifying the unit. Same metric values, but
  the previous "stream=0 vs scores=200" contradiction is now visibly
  reconciled as "queue is drained / 200 lifetime served".
* **Closes:** P13 #11.

### 12. Page 13 — Quick Lookup priority mis-wired

* **Audit:** `iter9_audit_a6.md` Tab b (Priority 1.00 + no_action +
  uplift 1.46% — priority field was actually a churn-risk score).
* **File:line — before:** `app.py:3897-3898` rendered "Priority Score"
  card directly from `row["priority_score"]`.
* **After:** Renamed the column to `Risk Score` (with a tooltip that
  explicitly says "this is the customer's churn-risk score — formerly
  mislabelled \"Priority\""). Added a fourth column `Action EV
  (uplift × CLV)` that computes `expected_uplift × clv_predicted` when
  CLV is on the row, with a tooltip "Drives offer selection". The
  Quick Lookup grew from 3 to 4 cards.
* **Closes:** P13 #12.

### 13. Page 13 — banner only after customer is selected

* **Audit:** `iter9_audit_a6.md` Tab b ("`Recommended Offer: no_action`
  rendered globally before any customer selected").
* **File:line — before:** `app.py:3884-3899` — `selectbox` defaulted to
  the first customer, immediately rendering the recommendation banner.
* **After:** Prepended a sentinel `"— Select a customer to see their
  recommendation —"` to the options list. While the sentinel is the
  selection, only a `st.caption("Pick a customer above…")` is shown —
  no metric cards, no `Recommended Offer:` banner. Banner only fires
  after the user explicitly picks a customer.
* **Closes:** P13 #13.

### 14. Page 13 tab c — drift "trend" 1-datapoint guard

* **Audit:** `iter9_audit_a6.md` Tab c (1.5 ms x-axis, single point).
* **File:line — before:** `app.py:3947-4028` — Drift Alert Timeline,
  PSI Trend, KS Trend rendered unconditionally.
* **After:** Inserted `drift_trend_guard(drift_history.timestamp,
  min_points=5)` immediately after the KPI cards. When the guard
  returns `(False, msg)`, an `st.info(msg + …)` is shown and the three
  trend charts are skipped. The drift-history expander still renders so
  raw rows are inspectable. Same guard then applied to the
  `Scoring Volume Over Time` / `Mean Churn Probability Over Time`
  block lower on the same tab.
* **Closes:** P13 #14.

### 15. Page 14 ↔ Page 15 — MLflow status alignment

* **Audit:** `iter9_audit_a6.md` cross-page (Page 14 banner says
  "tracking server not available", Page 15 must agree).
* **File:line — before:** `app.py:4187-4212` had a private MLflow probe
  (`mlflow.search_experiments`) inline; Page 15 had a separate one in
  `system_health_view.check_mlflow_health`. The two could disagree on
  edge cases (degraded vs down).
* **After:** Added module-level `_probe_mlflow_status(config)` shim
  that imports and delegates to `system_health_view.check_mlflow_health`.
  Page 14 now consumes that result and renders either
  `st.success("Connected to MLflow tracking server")` or
  `st.warning("MLflow tracking server not available — showing cached
  experiment data … Page 15 (System Health) will report the same
  status.")`. The banner explicitly cross-references Page 15 so a
  future drift would be obvious.
* **Closes:** P14 / P15 #15.

---

## P1 defects closed

### 16. Page 04 — retention monotonicity violations flagged

* **Audit:** `iter9_audit_a2.md` Page 04 (`Apr 2024: P7 91.0% → P8 92.1%`).
* **File:line — before:** `app.py:3318-3342` — heatmap rendered raw
  matrix; no guard against rising retention within a cohort.
* **After:** After computing `unobserved_mask`, walk each cohort row's
  observed cells and collect `(prev → curr)` pairs where `curr > prev`.
  When at least one violation exists, render a `st.warning` listing up
  to three offending pairs with their cohort label and period numbers.
  Heatmap text labels render `"—"` for unobserved cells (so the user
  can no longer mistake the truncation tail for a real curve).
* **Closes:** P04 #16.

### 17. Page 04 — Avg Final Retention masks zero-filled future cells

* **Audit:** `iter9_audit_a2.md` Page 04 (`Avg Final Retention 2.5%` =
  `(10.2 + 0 + 0 + 0) / 4`).
* **File:line — before:** `app.py:3307-3313` —
  `avg_final_retention = retention_matrix[last_col].mean()`.
* **After:** Build `unobserved_mask` (rows of trailing 0/NaN after the
  last positive value). Average only over cells where
  `~unobserved_mask`; if every cohort has the last column unobserved,
  walk back to each cohort's deepest-observed retention and average
  those instead. The KPI label switches to `Avg Deepest-Observed
  Retention` in that fallback case so the reader sees the change.
* **Closes:** P04 #17.

### 18. Page 04 — Limited cohort window banner

* **Audit:** `iter9_audit_a2.md` Page 04 (only 4 cohorts, below SaaS
  norm of ≥6–12).
* **File:line — before:** No banner; cohort count rendered as a bare
  KPI metric.
* **After:** Added `if n_cohorts < 6: st.info(... Limited cohort window
  — only N monthly cohorts ... generate more historical data for
  trend reliability)`.
* **Closes:** P04 #18.

### 19. Page 07 — `Events Observed` mislabelled as predictions

* **Audit:** `iter9_audit_a3.md` Page 07 (`5,717 = High Risk count`).
* **File:line — before:** `app.py:1543-1545` rendered
  `Events Observed (Churn)` and `Event Rate`.
* **After:** Renamed to `Predicted Churners (>50%)` and
  `Predicted Churn Rate`, both with `help=` tooltips explicitly
  flagging that they are prediction-derived labels matching Page 01's
  High Risk count by construction.
* **Closes:** P07 #19.

### 20. Page 07 — Median Duration right-censored annotation

* **Audit:** `iter9_audit_a3.md` Page 07 (`309 d on 350 d horizon`).
* **File:line — before:** `app.py:1545` rendered the bare median.
* **After:** Compute `is_right_censored = (median / max_duration) > 0.9`
  and, when true, render `Median Duration *` (with a `help=`
  explaining the asterisk) plus a visible `st.caption` warning that
  the median is right-censored at the observation window.
* **Closes:** P07 #20.

### 21. Page 11 — Sleeping-dog negative-uplift guardrail

* **Audit:** `iter9_audit_a3.md` Page 11 (`Sleeping_dog Avg Uplift =
  -0.1097`, no exclusion guardrail).
* **File:line — before:** No banner.
* **After:** Immediately under the headline KPI strip, when the
  Sleeping Dog count is non-zero, render
  `st.warning("Sleeping Dogs (n=N) are excluded from coupon
  eligibility — predicted uplift is negative; treatment harms
  retention.", icon="⚠️")`.
* **Closes:** P11 #21.

---

## Defects skipped (out of scope or owned by another agent)

* **P05/P09 ROI labels** — owned by F2/F3/F4. F5 only fixes the Page 12
  side of the three-way ROI mismatch; the cross-page reconciliation
  arrives once F2/F3/F4 each call `compute_overall_roi` with their own
  `scope_label`.
* **P02 confusion-matrix sample selection** — fixing the headline P/R
  to match the matrix (defect 1) closes the visible contradiction;
  expanding the matrix sample size from 600 → 20,000 is a model-layer
  change owned by `src/models/evaluation.py` (outside F5 scope).
* **Realism / fixture-data issues on Page 13 (Oct-2024 timestamps,
  uniform-4 scoring volume)** — the dashboard correctly displays the
  loader output; replacing the fixtures with live telemetry is
  outside the dashboard's responsibility.

---

## File-level summary

* **One file modified:** `src/dashboard/app.py`
* **Sections touched:** module imports, `render_model_performance`,
  `render_cohort_analysis`, `render_survival_analysis`,
  `render_uplift`, `render_retention_campaign`,
  `_render_scoring_status_tab`, `_render_retention_offers_tab`,
  `_render_monitoring_tab`, `render_mlflow_experiments`.
* **Public API preserved:** all `render_*` signatures unchanged.
* **Syntax verified:** `python -c "import ast; ast.parse(open(...).read())"` → OK.
