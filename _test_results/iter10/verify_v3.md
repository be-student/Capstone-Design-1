# Verifier V3 — A/B (P06) / Survival (P07) / Uplift (P11)

Sources cross-referenced:
* Issue ledger: `_test_results/iter9/iter9_audit_a3.md`
* Fix log: `_test_results/iter10/fix_logs/f5_app.md`
* PNGs: `_test_results/dashboard_pages/{06_ab_testing,07_survival_analysis,11_uplift_modeling}.png`
* Code: `src/dashboard/app.py` (`render_ab_testing` 1261+, `render_survival_analysis` 1681+, `render_uplift` 2470+)

| Page | Issue | Verdict |
| --- | --- | --- |
| P06 | (1) 0/0/N/A/0.0% empty-state OR feasibility guard vs 20k pool | **NOT FIXED** |
| P07 | (2) Events Observed = 5,717 collision with Page 01 High Risk count | **FIXED** |
| P07 | (3) Median 309d on 350d horizon — censoring annotation | **PARTIAL** |
| P07 | (4) Event Rate by Segment binary 0%/100% on 7/8 segments | **NOT FIXED** |
| P07 | (5) Two segment taxonomies (8-uplift + 6-behavioral) on one page | **NOT FIXED** |
| P11 | (6) Avg Uplift Score == Avg Treatment Effect duplicate KPI | **FIXED** |
| P11 | (7) 16,317 + 3,683 = 20,000 (4-quadrant collapse on headline) | **FIXED** |
| P11 | (8) Vocabulary {Persuadable+Sleeping Dogs / Persuadable+Lost Cause / 4 segments} | **FIXED** |
| P11 | (9) sleeping_dog negative-uplift exclusion guardrail | **FIXED** |

---

## Page 06 — A/B Testing

### Issue 1 — empty-state OR feasibility guard against 20k pool: **NOT FIXED**

**PNG evidence (`06_ab_testing.png`):**
* Headline KPI strip is unchanged: `Total Experiments 0 / Significant Results 0 / Best Experiment N/A / Avg Lift 0.0%`. Still rendered as four numeric metrics — no empty-state illustration, no CTA, no "no experiments yet" copy.
* `MDE Sensitivity Analysis` table still shows `MDE 1.0% → 24,441 per group / 48,882 total participants` (2.4× the 20k pool) with no infeasibility flag, no "exceeds available pool" warning, no headroom annotation.
* `Required Sample Size (per group) 906 / Total Participants Needed 1,812 / Expected Duration (days) 19` is rendered with no comparison to the 20k pool size.

**Code evidence (`src/dashboard/app.py` 1261-1640):**
* `render_ab_testing` is unchanged from iter9: lines 1299-1315 still render the four bare metrics; lines 1566-1579 render the power-analysis trio with no pool comparison; lines 1619-1640 render the MDE sensitivity table with no `pool_size_check` column or warning.
* No reference to `n_customers`, `pool_size`, `feasible`, or `available_population` anywhere in the function.

**Why NOT FIXED:** Page 06 is not in the F5 fix log's "P0 closed" or "P1 closed" sections. F5's scope was Pages 02, 04, 07, 11, 12, 13, 14 — Page 06 was not assigned, and no agent in iter10 has closed P06 #1.

---

## Page 07 — Survival Analysis

### Issue 2 — `Events Observed = 5,717` collision: **FIXED**

**PNG evidence (`07_survival_analysis.png`):**
* KPI strip now reads `Total Customers 20,000 / Predicted Churners (>50%) 5,717 / Predicted Churn Rate 28.59% / Median Duration 309 days`.
* `Events Observed (Churn)` label is gone; replaced by `Predicted Churners (>50%)`. `Event Rate` replaced by `Predicted Churn Rate`. Both have help-tooltip indicators (i icon).

**Code evidence (`app.py` 1707-1722):**
```
kc2.metric("Predicted Churners (>50%)", f"{int(event_count):,}",
    help="Count of customers whose predicted churn probability exceeds 50%. ...
          matches the High Risk count on Page 01 by construction — it is NOT
          an observed event count (iter9 audit P07 #19).")
kc3.metric("Predicted Churn Rate", f"{event_rate:.2%}", ...)
```
Tooltip explicitly cross-references Page 01 and the iter9 audit ID. Number itself (5,717) unchanged because the underlying data is unchanged — only the label is corrected. Matches F5 fix log defect #19.

### Issue 3 — Median 309 d on 350 d horizon: **PARTIAL**

**PNG evidence:** KPI shows bare `Median Duration  309 days` — no asterisk, no inline censoring caption visible. Text "right-censored" does not appear above the chart row.

**Code evidence (`app.py` 1700-1739):**
```
is_right_censored = (median / max_duration) > 0.9
if is_right_censored:
    kc4.metric("Median Duration *", ..., help="* right-censored at observation window ...")
    st.caption("⚠️ Median duration is right-censored: only ... days of follow-up ...")
else:
    kc4.metric("Median Duration", f"{median_duration:.0f} days")
```
The threshold is 90%. With `median = 309` and `max_duration = 350`, `309/350 = 0.883 < 0.90`, so the guard does not trigger. The annotation **mechanism** is implemented (defect #20 closed in F5 log) but is **dormant** for the actually-observed data on this PNG — user sees no annotation.

**Why PARTIAL:** F5 closed the audit's literal request ("add an annotation"), but the threshold is set such that the very example flagged in the audit (`309/350`) does not fire it. Recommend lowering the trigger to 0.85 (or comparing `median + IQR` against `max_duration`) so that the audit's own example would surface the warning.

### Issue 4 — Event Rate by Segment binary 0% / 100% on 7/8 segments: **NOT FIXED**

**PNG evidence:** `Event Rate by Segment` chart and table are still rendered with the same eight uplift segments. Same binary 0%/100% pattern visible (chart has bars all at full height for `high_value_lost_cause / *_persuadable / mid_value_persuadable` and zero for `*_sure_thing` segments; the table column "Event Rate" shows only the corner values).

**Code evidence (`app.py` 1888-1956):** `Event Rate by Segment` block is unchanged from iter9 — `event_stats = survival.groupby("segment").agg(events=("event_observed","sum"))` then `event_rate = events/total`. No outcome-leak guard, no warning banner about tautological labels, no consolidation onto a single non-leaking grouping.

**Why NOT FIXED:** F5 fix log does not list a defect for Event Rate by Segment binary pattern. The label-leak is upstream (segment definition encodes the outcome) and is a data-generator concern, but the dashboard surfaces it without any qualifier.

### Issue 5 — Two segment taxonomies (8-uplift + 6-behavioral) on one page: **NOT FIXED**

**PNG evidence:**
* `Average Survival Probability by Segment` (line 1827 block) shows 8 uplift-taxonomy bars: `high_value_sure_thing, mid_value_sure_thing, low_value_sure_thing, *_persuadable, sleeping_dog, high_value_lost_cause, low_value_persuadable`.
* `Daily Hazard Rate by Segment` (line 1847 block) shows the 6-behavioral taxonomy: `vip_loyal, regular_loyal, bargain_hunter, explorer, dormant, new_customer`.
* `Event Rate by Segment` again uses 8-uplift taxonomy.
* `Customer Duration Distribution` legend uses 8-uplift taxonomy.

Two taxonomies remain on the same page; charts are not directly comparable.

**Code evidence (`app.py` 1745-1754, 1828):**
* `surv_curves = data_loader.load_survival_curves()` → 6 behavioral keys, fed into KM curves and Daily Hazard.
* `survival.groupby("segment")` (the column from the loaded survival DataFrame) → 8 uplift segments, fed into Avg Survival Probability + Event Rate by Segment.

The function consumes two upstream data sources with different segment vocabularies and never crosswalks them.

**Why NOT FIXED:** F5 fix log has no P07 taxonomy-unification entry. This is a multi-layer fix (data loader + view) that was not landed in iter10.

---

## Page 11 — Uplift Modeling

### Issue 6 — Avg Uplift Score == Avg Treatment Effect duplicate (4dp identical): **FIXED**

**PNG evidence (`11_uplift_modeling.png`):**
* Headline shows 5 metrics: `Avg Uplift Score 0.0434 / Persuadable 2,798 / Sure Thing 12,919 / Sleeping Dog 3,683 / Lost Cause 600`. The `Avg Treatment Effect` KPI is **gone**.
* Single `Uplift Score Distribution` histogram is shown. The previously-paired `Distribution of Treatment Effects` plot is **gone**.
* Below the histogram, a caption is visible: "Note: in this build the `treatment_effect` column equals the `uplift_score` column on every row, so a separate distribution plot would be a duplicate. A dedicated ATE estimator is a future-build placeholder."

**Code evidence (`app.py` 2511-2564):**
* Five-column KPI strip; only `Avg Uplift Score` is the floating-point metric.
* Single `px.histogram(uplift, x="uplift_score", …)` rendered; no second histogram for treatment_effect.
* `st.caption(…)` block (line 2559) reproduces the user-visible explanation verbatim.

Closes audit P11 #6 / fix log defect #6.

(Note: the per-row "Top 10 Persuadable Customers" table still shows `uplift_score` and `treatment_effect` columns side-by-side and they remain equal to 4 dp. This was acknowledged in the caption rather than dropped from the table — acceptable for transparency.)

### Issue 7 — 4-quadrant collapse on headline (16,317 + 3,683 = 20,000 → only 2 buckets): **FIXED**

**PNG evidence:** Headline arithmetic now sums to all four quadrants: `Persuadable 2,798 + Sure Thing 12,919 + Sleeping Dog 3,683 + Lost Cause 600 = 20,000`. The previous 16,317 figure is gone; the headline counts now reconcile with the segment table directly below.

**Code evidence (`app.py` 2486-2525):**
* `quad_counts` dict is built directly from `uplift["segment"].value_counts()` (line 2493), so the source of truth is the segment column itself — the same column the table groups on.
* Five-column layout: `kc1..kc5 = st.columns(5)`; metrics rendered with `format_count(quad_counts[k])`.
* Fallback path (lines 2496-2509) classifies by sign-of-uplift if the `segment` column is absent — preserves the same 4-bucket vocabulary.

Closes audit P11 #7 / fix log defect #7.

### Issue 8 — Vocabulary unification {Persuadable+Sleeping Dogs / Persuadable+Lost Cause / 4 segments}: **FIXED**

**PNG evidence:**
* Headline labels: `Persuadable / Sure Thing / Sleeping Dog / Lost Cause` (4 canonical labels).
* Pie chart `Customer Response Classification` legend: `Sure Thing / Sleeping Dog / Persuadable / Lost Cause` — same 4 labels.
* Bar chart `Response Classification by Segment` x-axis: same 4 labels.
* Segment table x-axis: `lost_cause, persuadable, sleeping_dog, sure_thing` — same 4 labels (snake_case in the raw column).

All four sections of the page now use one vocabulary.

**Code evidence (`app.py` 2628-2655):**
```
label_map = {"persuadable":"Persuadable","sure_thing":"Sure Thing",
             "lost_cause":"Lost Cause","sleeping_dog":"Sleeping Dog"}
…
if "segment" in uplift_classified.columns:
    uplift_classified["response_class"] = (
        uplift_classified["segment"].astype(str).map(label_map)
        .fillna(uplift_classified.apply(_classify_customer, axis=1)))
```
Pie + bar chart + headline all consume `response_class` derived from this single mapping. Closes audit P11 #8 / fix log defect #8.

### Issue 9 — sleeping_dog negative-uplift exclusion guardrail: **FIXED**

**PNG evidence:** Immediately under the headline strip a yellow warning banner is visible:
> ⚠ Sleeping Dogs (n=3,683) are excluded from coupon eligibility — predicted uplift is negative; treatment harms retention.

**Code evidence (`app.py` 2528-2534):**
```
if quad_counts["sleeping_dog"] > 0:
    st.warning(
        f"Sleeping Dogs (n={quad_counts['sleeping_dog']:,}) are excluded "
        "from coupon eligibility — predicted uplift is negative; "
        "treatment harms retention.",
        icon="⚠️",)
```
Closes audit P11 #21 / fix log defect #21.

---

## Cross-page consistency snapshot

| Number | iter9 reading | iter10 reading | Status |
|---|---|---|---|
| 5,717 (Page 07) | "Events Observed (Churn)" — collided with Page 01 High Risk count | "Predicted Churners (>50%)" — explicit prediction-derived label, tooltip cross-references Page 01 | **Fixed via relabel** |
| 28.59% (Page 07) | "Event Rate" | "Predicted Churn Rate" | **Fixed via relabel** |
| 309 d / 350 d | No annotation | Annotation mechanism present but threshold 0.9 → does not fire on the actual 0.883 ratio | **Partial** |
| 16,317 + 3,683 (Page 11) | Two of four quadrants — contradicted segment table | All four quadrants in headline (`2,798 + 12,919 + 3,683 + 600 = 20,000`); reconciles with table | **Fixed** |
| Avg Uplift = Avg TE = 0.0434 (Page 11) | Two KPIs identical to 4 dp | One KPI; second removed; caption explains | **Fixed** |
| Sleeping Dog −0.1097 (Page 11) | No exclusion guardrail | Inline warning banner | **Fixed** |
| 0/0/N/A/0.0% (Page 06) | Bare KPIs, no empty state | Unchanged — bare KPIs, no empty state, no pool-feasibility flag | **Not fixed** |
| 8 segments binary 0/100 (Page 07) | Tautological label leak | Unchanged | **Not fixed** |
| 8-uplift + 6-behavioral on Page 07 | Two taxonomies on one page | Two taxonomies on one page | **Not fixed** |

---

## 5-line summary

1. **Verdict matrix:** 5 FIXED (P07 #2 relabel, P11 #6 dedup, P11 #7 4-quadrant, P11 #8 vocabulary, P11 #9 sleeping-dog guard); 1 PARTIAL (P07 #3 censoring annotation gated at 0.9 ratio, doesn't fire on actual 309/350=0.883); 3 NOT FIXED (P06 #1 empty-state/feasibility, P07 #4 binary segment leak, P07 #5 dual taxonomy).
2. **Page 11 is fully closed:** headline now shows all four uplift quadrants summing to 20,000, the duplicate ATE metric and duplicate histogram are gone with an explanatory caption, vocabulary is unified across pie/bar/headline/table via `label_map`, and the sleeping-dog negative-uplift exclusion banner is rendered inline.
3. **Page 07 is partially closed:** the `5,717` mislabeling is correctly fixed by renaming to `Predicted Churners (>50%)` with a tooltip cross-referencing Page 01; the median-censoring annotation logic is implemented but the 0.9 threshold means the audit's own example (309/350=0.883) does not trigger it — recommend lowering threshold to 0.85.
4. **Page 07 still carries two unfixed structural issues:** Event Rate by Segment is still binary 0%/100% on 7 of 8 segments (label-leak unaddressed), and the page still mixes 8-uplift taxonomy (Avg Survival, Event Rate, Duration Distribution) with 6-behavioral taxonomy (KM curves, Daily Hazard) — not in F5's scope.
5. **Page 06 is regressed in priority — completely untouched in iter10:** 0/0/N/A/0.0% still rendered as headline KPIs with no empty state, and the MDE Sensitivity Analysis still recommends 24,441 per group / 48,882 total participants for MDE 1% against a 20k pool with no feasibility guard. This page was not in F5's scope and no other agent's fix log addresses it.
