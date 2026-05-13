# A3 — A/B Testing / Survival / Uplift Modeling

Banner on every page: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## Page 06 — A/B Testing

### Visible KPIs
- Total Experiments: **0**
- Significant Results: **0**
- Best Experiment: **N/A**
- Avg Lift: **0.0%**
- Power Analysis inputs: Baseline Churn Rate **0.20**, MDE **0.05**, Significance Level (alpha) **0.05**, Target Power (1-beta) **0.8**
- Required Sample Size (per group): **906**
- Total Participants Needed: **1,812**
- Expected Duration (days): **19**
- Power Curve chart with marker n=906 crossing the Target: 80% line

### Wrong / suspicious
- The four headline KPIs are all zero/N/A while the dataset advertises n=20,000. A SaaS buyer reads "Total Experiments 0 / Avg Lift 0.0%" as a broken pipeline, not a deliberate empty state. There is no callout that says "no experiments configured" — the zeros sit in the same KPI strip as the populated power-analysis numbers, which is misleading.
- Power-analysis sample size 906/group implies 1,812 total; but the simulator population is 20,000. There is no reconciliation between "what we'd need" (1,812) and "what we have" (20,000) — buyers expect the page to recommend N actually achievable experiments given the 20k pool.

### Unreliable
- Expected Duration "19 days" is shown without an arrival-rate assumption. 1,812 / 19 = ~95 enrollments/day is implied but never stated; with no traffic-rate input the duration figure is not trustworthy.
- Baseline Churn Rate 0.20 is hard-coded on a slider — it does not appear to be wired to the actual churn rate computed elsewhere in the simulator, so the power calc may be inconsistent with the rest of the app.

### Missing
- No experiment list / table (variant, n_control, n_treatment, conversion_control, conversion_treatment, p-value, CI, lift).
- No CUPED / variance-reduction note, no sequential-testing / peeking guard, no multiple-testing correction.
- No empty-state CTA ("Create experiment") to disambiguate "0 experiments" from "broken".
- No SRM (sample-ratio mismatch) check, no guardrail metrics, no minimum-runtime guard.
- No literature-band disclaimer for "Avg Lift" (typical retention-experiment lifts sit in the 5–15% band; with zero data the figure of 0.0% is uninformative).

---

## Page 07 — Survival Analysis

### Visible KPIs
- Total Customers: **20,000**
- Events Observed (Churn): **5,717**
- Event Rate: **28.59%**
- Median Duration: **309 days**
- Kaplan–Meier Survival Curves by Customer Segment (multi-line chart, several segments listed in the legend)
- Median Survival Time by Segment table (partially visible — at least one row "no_login" with a median value visible around 200; the rest of the table is below the fold of the screenshot)

### Wrong / suspicious
- Median duration **309 days** is suspiciously close to a 1-year (365 d) observation window. If the simulator's horizon is 365 d, this is almost certainly a right-censoring artifact: half the cohort hasn't churned yet, so the KM estimator returns the last observed time. There is no annotation of the observation window, no "censored" tick marks called out, and no caveat that the median may not be reached for some segments.
- Event rate 28.59% with median 309 d is internally tense: if only ~29% of customers experienced the event, the median survival time should typically be undefined (>observation horizon) rather than 309 d. The page does not explain how the median was computed when S(t) likely never crosses 0.5 for the full cohort.

### Unreliable
- Per-segment medians risk small-sample artifacts (medians of 0 or = horizon) — visible row appears truncated and there is no n per segment shown alongside the median, no confidence interval, and no log-rank p-value comparing segments.
- KM curves shown without confidence bands and without a "number-at-risk" table beneath the x-axis — standard survival-analysis hygiene that a SaaS buyer in pharma/insurance/SaaS-retention will look for.

### Missing
- No annotation of the observation/administrative-censoring window (start date, end date, max follow-up).
- No censoring indicator on the KM curves (tick marks) or in a summary ("X censored, Y events").
- No Cox PH / hazard-ratio output, no Schoenfeld residual / PH-assumption check.
- No log-rank or pairwise segment comparison.
- No restricted-mean-survival-time (RMST) fallback for cases where median is not reached.
- No disclaimer that medians near the horizon are right-censoring artifacts.

---

## Page 11 — Uplift Modeling

### Visible KPIs
- Avg Uplift Score: **0.0434**
- Avg Treatment Effect: **0.0434**
- Persuadable Customers: **16,317**
- Sleeping Dogs: **3,683**
- Distribution of Uplift Scores (histogram, x-range ~ -0.5 to 0.5, peak count ~10k at 0)
- Distribution of Treatment Effects (histogram, identical shape and range, peak ~10k at 0)
- "Uplift Score vs Treatment Effect by Segment" chart heading visible

### Wrong / suspicious
- **Avg Uplift Score 0.0434 == Avg Treatment Effect 0.0434** to 4 decimal places. The two histograms are visually identical (same peak height ~10k, same x-range, same shape, same "Zero" reference line). This strongly suggests one variable is being plotted twice under different labels — a false-equivalence issue. In a real uplift model, `uplift_score` is the model's predicted CATE while `treatment_effect` is the realized/observed lift; they should not be numerically identical.
- Persuadable + Sleeping Dogs = 16,317 + 3,683 = **20,000 exactly** — the entire population. That leaves **0** customers in the "Sure Things" and "Lost Causes" quadrants of the standard 4-quadrant uplift segmentation. Either the segmentation is binarized on the sign of one score (which collapses 4 quadrants to 2) or the categorization logic is wrong.
- 3,683 Sleeping Dogs (18.4% of base) is a large negative-uplift cohort. There is no visible callout that these customers must be **excluded** from coupon/treatment eligibility — without that guardrail, a buyer treating them would actively destroy revenue.

### Unreliable
- A peak of ~10,000 at exactly 0 in both histograms (half the population sitting at zero uplift) suggests a degenerate model output (e.g., a regressor outputting near-zero for the bulk of users) or a thresholding artifact. The KPI "Avg Uplift 0.0434" averages over this mass-at-zero and is not actionable.
- No model validation metric shown (Qini coefficient, AUUC, uplift-by-decile lift table, policy value vs random) — so the 0.0434 average is not benchmarkable.
- The "by Segment" chart is only partially loaded in the screenshot; cannot verify segment-level reliability.

### Missing
- No Qini / AUUC score, no uplift decile chart, no cumulative-gain curve.
- No explicit Sleeping-Dog **exclusion rule** wired to the recommendation/coupon engine; no warning banner.
- No 4-quadrant breakdown (Persuadable / Sure Thing / Lost Cause / Sleeping Dog) with counts and CIs.
- No confidence intervals on the headline 0.0434 (and no n behind it).
- No mention of which uplift learner was used (T-learner, S-learner, X-learner, DR-learner, causal forest), no propensity-score diagnostics, no overlap/positivity check.
- No disclaimer that uplift_score and treatment_effect are distinct constructs.

---

## SaaS-readiness verdict
**Verdict:** DO-NOT-SHIP

**Top 3 blockers:**
1. **Uplift page is plotting one variable as two** — Avg Uplift Score and Avg Treatment Effect are identical to 4 decimals (0.0434 / 0.0434) with visually identical histograms. This is a false-equivalence bug that invalidates the entire page's claim to causal inference. Compounded by the Sleeping-Dogs cohort (3,683 customers, 18.4% of base) having no visible exclusion-from-treatment guardrail.
2. **A/B Testing page reads as broken** — Total Experiments 0, Significant Results 0, Best Experiment N/A, Avg Lift 0.0% on a 20k-row dataset, with no empty-state explanation and no reconciliation against the 20,000 available users vs the 1,812 the power calc says are needed. Buyers will assume the experiment service is down.
3. **Survival median 309 d is a right-censoring artifact, unannotated** — with only a 28.59% event rate the cohort median should likely be undefined; reporting 309 d (close to a 1-year horizon) without a censoring caveat, KM CIs, or a number-at-risk table is statistically misleading.

---

5-line summary:
- A/B Testing shows 0 experiments / 0 significant / 0.0% lift as headline KPIs on n=20,000 with no empty-state — reads as broken; power calc needs 1,812 but page never reconciles against the 20k pool.
- Survival reports median 309 d at 28.59% event rate near a likely 365-day horizon — classic right-censoring artifact with no annotation, no CIs, no at-risk table, no log-rank.
- Uplift shows Avg Uplift Score 0.0434 = Avg Treatment Effect 0.0434 with visually identical histograms — one variable plotted twice (false equivalence).
- Persuadable 16,317 + Sleeping Dogs 3,683 = 20,000 exactly (4-quadrant segmentation collapsed to 2); no Sleeping-Dog exclusion guardrail for coupon eligibility.
- Verdict: DO-NOT-SHIP until the uplift double-plot bug, the A/B empty-state framing, and the survival censoring disclosure are fixed.
