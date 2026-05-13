# A3 — A/B / Survival / Uplift

Auditor: independent (no prior context). Source: PNG screenshots + MD data dumps in `_test_results/dashboard_pages/` and `_test_results/page_data/`. All three pages carry the banner "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

---

## Page 06 — A/B Testing

### Visible KPIs (verbatim)
- Total Experiments: **0**
- Significant Results: **0**
- Best Experiment: **N/A**
- Avg Lift: **0.0%**
- Required Sample Size (per group): **906**
- Total Participants Needed: **1,812**
- Expected Duration (days): **19**
- Power inputs: Baseline Churn 0.20, MDE 0.05, alpha 0.05, power 0.8
- MDE Sensitivity table: MDE 1% → 24,441/group (48,882 total); 2% → 6,059/12,118; 3% → 2,629/5,258; 5% → 906/1,812; 8% → 329/658; 10% → 199/398; 15% → 70/152

### Wrong
- **Best Experiment "N/A" with Avg Lift "0.0%"** is contradictory framing: 0.0% is a *measured* number with one decimal of precision, but there are 0 experiments to measure. Should be "—" or hidden, not formatted as a numeric KPI.
- **MDE 1% requires 24,441/group = 48,882 total** which exceeds the 20,000 customer pool. Page presents this with no "infeasible at current pool size" guardrail.

### Unreliable
- The four headline KPIs (0/0/N/A/0.0%) read as a broken model rather than as a "no experiments yet" empty state. There is no empty-state illustration, CTA ("Launch first experiment"), or copy explaining the zeros.
- Power calc treats the 20,000 customer pool as unlimited — it never asks "do you have enough customers?". With MDE 5% it needs 1,812; with 1% it needs 48,882 (i.e. 2.4× the pool) but no warning is shown.

### Missing
- No reconciliation of `Required Sample Size = 1,812` vs. the 20,000-customer pool (no headroom indicator like "11× capacity available").
- No experiment audit log / history of past experiments.
- No multiple-comparisons control (Bonferroni / BH FDR), no two-sided vs one-sided toggle, no variance-adjusted (CUPED) option.
- No SRM (sample-ratio mismatch) check, no minimum runtime / weekly seasonality guidance.
- No segmentation toggle (power calc is global only).
- No real-world lift sanity disclaimer (typical 5–15% range).

---

## Page 07 — Survival Analysis

### Visible KPIs (verbatim)
- Total Customers: **20,000**
- Events Observed (Churn): **5,717**
- Event Rate: **28.59%**
- Median Duration: **309 days**
- Avg Survival by Segment: high_value_sure_thing 97.68%, mid_value_sure_thing 86.98%, low_value_sure_thing 73.04%, high_value_persuadable 39.84%, mid_value_persuadable 39.78%, sleeping_dog 38.81%, high_value_lost_cause 37.23%, low_value_persuadable 22.28%
- Daily Hazard by Segment: dormant 0.00254, new_customer 0.00166, bargain_hunter 0.00120, explorer 0.00099, regular_loyal 0.00062, vip_loyal 0.00023
- Event Rate by Segment: high_value_lost_cause **100.0%**, high_value_persuadable **100.0%**, low_value_persuadable **100.0%**, mid_value_persuadable **100.0%**, sleeping_dog **65.8%**, high_value_sure_thing **0.0%**, low_value_sure_thing **0.0%**, mid_value_sure_thing **0.0%** (8 segments total — answering the prompt's "how many segments?" → **8**)
- Model config: penalizer 0.01, l1_ratio 0, alpha 0.05

### Wrong
- **Events Observed = 5,717** is the same number reported on the Overview page as "High Risk (>50%) count = 5,717" (per the analyst note in the MD). This is "predicted high risk count" being mislabeled as "observed events" — a category error: predictions are not outcomes.
- **Event Rate by Segment is binary**: 7 of 8 segments report exactly 0.0% or 100.0% (only `sleeping_dog` 65.8% breaks the pattern). That is a tautological labeling artifact (segment definition is leaking the outcome label) — not a credible production signal.
- **Two segment taxonomies on a single page**: Avg Survival uses the 8-bucket uplift taxonomy (`high/mid/low × value × persuadable/sure_thing/lost_cause/sleeping_dog`), while Daily Hazard uses the 6-bucket behavioral taxonomy (`vip_loyal / regular_loyal / bargain_hunter / explorer / dormant / new_customer`). Charts are not directly comparable.
- **Median 309 d on a 350 d horizon** = right-censoring artifact; the median is bumping the observation window. There is no annotation, no "horizon-bounded" warning, and no rerun on a longer follow-up.

### Unreliable
- KM curves shown without 95% confidence bands and without an "at-risk" table beneath, making the curves unverifiable.
- The Customer Duration Distribution shows a massive spike of `mid_value_sure_thing` in the 350-day bin — confirms heavy right-censoring at the window boundary; survival probabilities for "sure_thing" segments are therefore optimistic by construction.
- 28.59% event rate is plausible, but given the 5,717 = predicted-high-risk-count collision, it cannot be trusted as an *observed* rate.

### Missing
- No log-rank / Cox PH test comparing segments.
- No confidence intervals on KM curves or median durations.
- No at-risk numbers under the curve.
- No censoring annotation / treatment.
- No restricted mean survival time (RMST), no hazard-ratio table.
- No definition of what counts as the "event" (is it churn label, last_purchase>X days, or risk_score>0.5?).

---

## Page 11 — Uplift Modeling

### Visible KPIs (verbatim)
- Avg Uplift Score: **0.0434**
- Avg Treatment Effect: **0.0434**
- Persuadable Customers: **16,317**
- Sleeping Dogs: **3,683**
- Avg Uplift by Segment table: persuadable 0.1902 (count 600 → table actually shows lost_cause 0.0258 count 600, persuadable 0.1902 count 2,708, sleeping_dog -0.1097 count 3,683, sure_thing 0.0560 count 12,929; Persuadable% column is 100% / 0% / 0% / 100%)
- Customer Response Classification pie: **Persuadable 81.6% / Lost Cause 18.4%**
- Top-10 Persuadable customers table (uplift_score == treatment_effect to 4 dp on every row, e.g. 0.6874/0.6874, 0.6343/0.6343, 0.64/0.64, …)

### Wrong
- **Avg Uplift Score (0.0434) == Avg Treatment Effect (0.0434)** to 4 decimals. The "Top 10 Persuadable" table shows uplift_score == treatment_effect on every single row to 4 dp (0.6874=0.6874, 0.6343=0.6343, 0.64=0.64, 0.6386=0.6386, …). These are not two metrics — it is one variable rendered twice with two labels. The two distribution histograms ("Distribution of Uplift Scores" vs "Distribution of Treatment Effects") are visually identical for the same reason.
- **16,317 + 3,683 = 20,000** → the entire customer base has been forced into just 2 of the 4 quadrants in the headline KPIs. The bar/table below shows all 4 (lost_cause 600, persuadable 2,708, sleeping_dog 3,683, sure_thing 12,929 = 20,000) — so the data has 4 quadrants, but the headline collapses to 2. The 16,317 in the headline does not match any single segment count: it equals `persuadable (2,708) + sure_thing (12,929) + lost_cause (600) + ...` no, actually 2,708 + 12,929 + 600 = 16,237, not 16,317. The 16,317 number reconciles with **none** of the segment counts shown below it on the same page.
- **Naming inconsistency on one page**: headline says "Persuadable / Sleeping Dogs"; pie chart legend says "Persuadable / Lost Cause"; segment table shows all four labels {lost_cause, persuadable, sleeping_dog, sure_thing}. Three different vocabularies in three sections of the same page.
- **Page 11 uses 4-bucket uplift taxonomy**, while (per analyst note in MD) Pages 05/10 use the 8-bucket value × uplift taxonomy. Cross-page taxonomy drift.
- **Sleeping_dog Avg Uplift = -0.1097 (negative)** — these customers are *harmed* by treatment. There is no inline guardrail flagging that they must be excluded from any coupon/retention campaign.

### Unreliable
- Persuadable% column reports `100.0%` for lost_cause and persuadable, `0.0%` for sleeping_dog and sure_thing — another binary 0/100 segment-leak pattern (mirrors Page 07 Event Rate by Segment).
- "Avg Uplift" of 0.1902 for `persuadable` with sample size 2,708 has no CI shown.
- Distribution of Uplift Scores and Distribution of Treatment Effects are presented side-by-side as if they were independent insights; they are the same plot.

### Missing
- No CIs / standard errors on uplift scores.
- No Qini curve or uplift gain curve (the canonical uplift validation chart).
- No treatment/control split disclosure (sample sizes per arm, randomization check).
- No barrier/treatment cost ROI integration (table shows `selected_barrier = t_barrier` for all top-10 rows but no cost or expected value).
- No exclusion guardrail for negative-uplift customers (sleeping_dogs).
- No reconciliation of headline `Persuadable=16,317` with table `persuadable count=2,708`.

---

## Cross-page consistency check (especially with the 5,717 / 16,317 / 3,683 / 20,000 numbers)

| Number | Where it appears | What it is claimed to be | Reality |
|---:|---|---|---|
| 20,000 | Page 06 banner, Page 07 Total Customers, Page 11 (16,317+3,683) | Customer pool size | Consistent — this is the n. |
| 5,717 | Page 07 "Events Observed (Churn)"; per MD note, Page 01 reports same value as "High Risk (>50%) count" | Survival page calls it observed events; Overview calls it predicted high-risk count | These are different concepts. Either it is observed churn (post-hoc) OR it is the count of customers with predicted churn prob > 0.5. It cannot be both. The collision strongly suggests Page 07 is treating prediction-derived labels as if they were observed events — which would invalidate the entire Kaplan-Meier curve. |
| 16,317 | Page 11 headline "Persuadable Customers" | Count of persuadable customers | Page 11's own segment table shows persuadable count = 2,708 (not 16,317). 16,317 does not equal any sum of subsets in the table (closest is persuadable + sure_thing = 15,637; persuadable + sure_thing + lost_cause = 16,237). The headline figure is unreconciled with the supporting table on the same page. |
| 3,683 | Page 11 headline "Sleeping Dogs" | Count of sleeping_dogs | Matches the segment table sleeping_dog count = 3,683. Internally consistent. |
| 16,317 + 3,683 = 20,000 | Page 11 headline arithmetic | "Whole base classified as Persuadable or Sleeping Dog" | Contradicts the same page's table which shows 4 segments summing to 20,000 (sure_thing 12,929 + persuadable 2,708 + sleeping_dog 3,683 + lost_cause 600). The headline collapsed 4 → 2 incorrectly. |
| Avg Uplift = Avg Treatment Effect = 0.0434 | Page 11 headline + every row of Top-10 table | Two distinct metrics | One variable, two labels. False equivalence presented as corroboration. |
| 0% / 100% segment patterns | Page 07 Event Rate by Segment (7/8 segments are 0 or 100), Page 11 Persuadable% column (4/4 are 0 or 100) | Real per-segment rates | Tautological — segment definition leaks the outcome label. Not a production signal. |
| Segment taxonomies | Page 03 (6-behavioral); Page 07 (mixes 8-uplift in Avg Survival + 6-behavioral in Daily Hazard); Page 10/05 (8-uplift); Page 11 (4-quadrant uplift) | Should be one canonical taxonomy with documented crosswalks | Four different taxonomies appear across the dashboard with no crosswalk. Page 07 uses two of them on a single screen. |

---

## SaaS-readiness verdict

**Verdict: NOT READY FOR SaaS / paying customers.** The three analytics pages put numerically wrong, internally contradictory, and conceptually confused content directly above the "Deploy" button. A single screenshot of Page 11 (Persuadable=16,317 in the headline, persuadable count=2,708 in the table directly below; "Avg Uplift" identical to "Avg Treatment Effect" to 4 decimals across every row) is enough to lose enterprise trust. Page 07 reuses a prediction count (5,717) as observed events — that is statistically invalid, not just cosmetic. Page 06 leads with 0/0/N/A/0.0% framed as KPIs instead of an empty state.

**Top 3 blockers:**
1. **Page 11: Avg Uplift Score == Avg Treatment Effect (0.0434) and uplift_score == treatment_effect on every customer row.** Either the model literally returns one value rendered twice (then remove one), or the two computations are mistakenly aliased (then fix the bug). Plus the headline "16,317 Persuadable + 3,683 Sleeping Dogs = 20,000" contradicts the segment table on the same page (persuadable=2,708, four segments summing to 20,000) and uses a different vocabulary than the pie chart ("Lost Cause") on the same page.
2. **Page 07: Events Observed = 5,717 collides with "High Risk count = 5,717" from the Overview page.** Predictions are being treated as observations; this invalidates the Kaplan-Meier curves, hazard rates, median 309 d, and 28.59% event rate. Compounded by `Event Rate by Segment` being binary 0%/100% on 7 of 8 segments — a tautological label-leak pattern that also appears on Page 11's Persuadable% column. Two distinct segment taxonomies (8-uplift and 6-behavioral) are mixed on one page.
3. **Page 06: 0/0/N/A/0.0% headline KPIs with no empty state, and a power calculator that cheerfully recommends sample sizes (24,441/group at MDE 1%) larger than the entire 20,000-customer pool.** Reads as a broken model, not as "no experiments yet"; offers no feasibility guard against the actual customer base.

---

## 5-line summary

1. Page 11 fatal flaw: Avg Uplift Score == Avg Treatment Effect (both 0.0434) and identical on every customer row — one variable presented as two metrics.
2. Page 11 arithmetic break: headline "Persuadable 16,317 + Sleeping Dogs 3,683 = 20,000" contradicts the same page's 4-segment table where persuadable = 2,708, and the pie legend further mislabels the second group "Lost Cause".
3. Page 07 fatal flaw: Events Observed = 5,717 is the same number Page 01 reports as "High Risk count" — predictions are being booked as observed events, invalidating the KM/hazard analysis.
4. Page 07 secondary: 8 segments with Event Rate binary 0%/100% on 7/8 (tautological label leak); two segment taxonomies (8-uplift + 6-behavioral) mixed on one page; median 309 d on a 350 d horizon is a censoring artifact with no annotation.
5. Page 06: 0/0/N/A/0.0% framed as KPIs instead of empty state; power calc recommends up to 48,882 participants against a 20,000 pool with no feasibility guard. Verdict: NOT SaaS-ready.
