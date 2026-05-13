# A1 — Overview / Churn / Model Performance

**Auditor stance:** Independent SaaS-buyer review prior to a paid pilot. Numbers cited verbatim from the MD dumps in `_test_results/page_data/` and cross-checked against the rendered PNGs in `_test_results/dashboard_pages/`.

---

## Page 00 — Overview

### Visible KPIs (from MD)
- Total Customers: **20,000**
- Avg Churn Prob: **31.31%**
- High Risk: **5,717**
- Total CLV: **57,936,514,970 KRW**
- Selected customer C000000: Churn Probability **3.09%**, Risk Level **LOW**, Segment **bargain_hunter**, Predicted CLV **2,716,186 KRW**, Recommended Action **N/A**, Days Since Purchase **0**.
- Risk donut shares: low **57.9%** · critical **18%** · medium **13.5%** · high **10.6%** (= 100.0%, OK).
- Histogram: leftmost bin (0–0.05) ≈ **4,000**; right tail spike ≈ 1,700 at p≈0.9 (bimodal).

### Wrong / suspicious
- **Recommended Action = "N/A"** for the very first customer (C000000) on a "Recommendations"-grade product; a buyer expects every scored row to carry an action. This signals an unimplemented action-engine path.
- **Days Since Purchase = 0** for a customer whose churn probability is 3.09% and whose donut-driven cohort is "low" — plausible, but the value 0 with no timestamp invites suspicion that the field is defaulted rather than computed.
- The histogram x-axis only labels up to ~0.9; the right-tail spike sits at 0.9+ but is not separately labeled, hiding whether bin 0.9–0.95 vs 0.95–1.0 is the spike.

### Unreliable
- **Synthetic data banner ("All KPIs are simulator-generated")** — every number on this page is from a simulator, not from production telemetry. A buyer cannot infer real-world performance from this view.
- **No period-over-period delta**: 31.31% Avg Churn Prob, 5,717 High Risk are point-in-time only, no Δ vs last week / last cohort, so trend-judgment is impossible.
- **Hardcoded thresholds**: "High Risk" definition (>50%) is not surfaced on this page (only revealed on Page 01); the 5,717 figure therefore depends on an undisclosed threshold.
- **Total CLV = 57.9B KRW** with no methodology note — a single large monetary KPI with zero confidence band on a synthetic dataset is a red flag.

### Missing
- "as of" timestamp / data freshness / scoring batch ID
- Model version (which scorer produced the 5,717?)
- Confidence interval on Avg Churn Prob (31.31% is a point estimate)
- Class-balance disclosure (positive class rate)
- Threshold definition tooltip on "High Risk" KPI
- Segment / date / cohort filter (the page is global only)
- Any backtest panel (yesterday's prediction vs today's outcome)
- Methodology / model-card link

---

## Page 01 — Churn Analytics

### Visible KPIs (from MD)
- Summary: Total Customers **20,000** · Avg Churn Prob **31.31%** · Median Churn Prob **15.39%** · High Risk (>50%) **5,717** · Critical (>75%) **3,596**.
- ml_model: AUC **0.8852**, F1 **0.6331**, Precision **0.5331**, Recall **0.7791**.
- dl_model: AUC **0.8860**, F1 **0.6531**, Precision **0.6759**, Recall **0.6318**.
- ensemble: AUC **0.8866**, F1 **0.6522**, Precision **0.6426**, Recall **0.6621**.
- Banner: "At-Risk Revenue (churn prob > 50%): **2,997,471,916 KRW (5.2% of total CLV)**".
- Risk donut: low 57.9% · critical 18% · medium 13.5% · high 10.6%.
- Histogram leftmost bin (0–0.05) ≈ **3,500**.
- High-Risk slider: "**5717 customers above threshold (50%)**".

### Wrong / suspicious
- **Histogram leftmost bin disagrees with Page 00 for the same 20k roster.** Page 00 shows ~**4,000** in the leftmost bin; Page 01 shows ~**3,500**. Same dataset, same population, two different first-bin counts → either different binning (undisclosed) or different filtering. Not buyer-acceptable without a footnote.
- **Critical / High ratio is implausibly high.** Critical(>75%) = 3,596 vs High(>50%) = 5,717 ⇒ **62.9%** of the High bucket is actually Critical. In other words only 5,717 − 3,596 = **2,121 customers (37.1% of High)** sit in the 50–75% band. A healthy churn distribution typically tapers; here Critical dwarfs the 50–75% middle, consistent with the bimodal histogram and suggesting that the simulator emits two clumps (≈0 and ≈0.9), not a realistic risk continuum.
- **Mean (31.31%) vs Median (15.39%) ratio = 2.03x.** Mean is more than double the median → strong right-skew driven by the 0.9-spike. Acceptable to display, but no skewness/IQR disclosure means a customer reading "Avg Churn Prob 31.31%" is misled about the typical customer (who is closer to 15%).
- **At-Risk Revenue 2.997B KRW = 5.2% of Total CLV (57.94B)** while High Risk customers are **5,717 / 20,000 = 28.6%** of the population. So 28.6% of customers represent only 5.2% of CLV — internally consistent (high-risk customers have low CLV) but the framing "At-Risk Revenue" without that qualifier overstates risk concentration; needs a tooltip.
- **Heatmap row "dormant"** in the MD dump shows `low 0.05 / medium 0.94 / high 0.06 / critical 0.02` summing to **1.07** — rows are not normalized (or the dump cell-order is wrong); either way the numbers as displayed do not sum to 1.0.
- **Heatmap row "new_customer"** shows `low 0.41 / medium 0.10 / high 0.05 / critical 0.44` — sum = 1.00 OK, but a segment that is 41% low-risk and simultaneously 44% critical-risk is bimodal in a way that contradicts the bar chart's "new_customer avg ~0.50" single-point summary.

### Unreliable
- **Per-model Precision/Recall here will conflict with the confusion matrix on Page 02** (see Page 02 section). The Precision/Recall numbers on this page are therefore unverifiable in isolation.
- **No sample size on the AUC table** — we are told AUC=0.8852 but not on what n, what split, what time window. Same critique for F1/Precision/Recall.
- **Threshold cuts (0.25 / 0.50 / 0.75)** are hardcoded and not labeled as policy choices. A SaaS buyer needs to be able to tune these per business cost matrix.
- **Cumulative feature importance "reaches 80% around feature index 5"** — no error bars, no permutation-importance vs gain-based disclosure, no model-version stamp.

### Missing
- Class balance / positive rate (is the trained label 28.6%? 18%? 5%? — undisclosed)
- Confidence intervals or bootstrap SEs on AUC/F1/Precision/Recall (all reported to 4 decimals as if exact)
- Test-set size and split methodology (random / time-based / customer-grouped)
- Calibration curve (high AUC + bad precision suggests miscalibration)
- Lift/gain chart for the top-decile (the actual operational view)
- Training date and model version per row of the Model Performance Summary
- Definition of "Critical" / "High" / "Medium" / "Low" cutoffs adjacent to the donut

---

## Page 02 — Model Performance

### Visible KPIs (from MD)
- ML Model AUC **0.8852** · DL Model AUC **0.8860** · Ensemble AUC **0.8866** · Best Model **ensemble**.
- ROC: ml=0.885, dl=0.886, ensemble=0.887, random=0.5.
- Confusion matrices (n=600 each):
  - ml_model: TN 350 / FP 50 / FN 80 / TP 120 → Acc 78.33% / Prec **70.59%** / Rec **60.00%**.
  - dl_model: TN 340 / FP 60 / FN 90 / TP 110 → Acc 75.00% / Prec **64.71%** / Rec **55.00%**.
  - ensemble: TN 360 / FP 40 / FN 70 / TP 130 → Acc 81.67% / Prec **76.47%** / Rec **65.00%**.
- Training Time: ml 1.0s, dl 1.0s, ensemble 1.0s.
- Ensemble weights: ML 60% / DL 40%.

### Wrong / suspicious — CRITICAL
- **Headline Precision/Recall ≠ Confusion-matrix Precision/Recall for the SAME model**:

  | Model | Headline P | Matrix P | Headline R | Matrix R | Δ Precision | Δ Recall |
  |---|---:|---:|---:|---:|---:|---:|
  | ml_model | 0.5331 | 0.7059 | 0.7791 | 0.6000 | **+17.3 pts** | **−17.9 pts** |
  | dl_model | 0.6759 | 0.6471 | 0.6318 | 0.5500 | −2.9 pts | −8.2 pts |
  | ensemble | 0.6426 | 0.7647 | 0.6621 | 0.6500 | **+12.2 pts** | −1.2 pts |

  At least one of {headline KPI, confusion matrix} is wrong, or they were computed on different test splits / thresholds with no disclosure. **This alone is a do-not-ship defect for a model-performance page.**
- **Confusion matrix totals = 600 cases**, but the dataset has **20,000 customers**. The confusion matrices are evaluated on **3% of the population** — undisclosed sub-sample, no statement of which 600. A buyer cannot reproduce or trust the matrix-level numbers.
- **"Best Model: ensemble"** is declared on AUC differences of **0.8852 vs 0.8860 vs 0.8866** — i.e. **0.0014 between best and worst**. With no confidence intervals and no DeLong test, this difference is statistically meaningless. The "Best Model" label fails any reasonable significance bar.
- **AUC vs Training Time scatter is degenerate**: all three points sit at x=1.0s, y∈[0.8855, 0.886]. The "trade-off" chart conveys no information.

### Unreliable
- **Training Time = 1.0s for ALL three models** including a deep-learning model and an ensemble — this is the simulator floor, not a measurement. Any infrastructure-cost analysis built on this is fictional.
- **MLflow Experiment Runs table** — only 1 run per model type, so no run-to-run variance is visible; "current_1" is the lone tag.
- **Radar / capability chart** displays a single test-time snapshot per model with no axis scale shown.
- **Ensemble Improvement Over Individual Models** table reports gains to 4 decimals (e.g. +0.0014 on AUC) without a CI, again giving false precision.

### Missing
- Confidence intervals on every metric (AUC SE, P/R via bootstrap, CI on F1)
- Statistical-significance test between models (DeLong for AUC, McNemar for paired errors)
- Test-set size, split strategy, and class balance — all absent
- Calibration curve / Brier score (high AUC alone is not enough for a probability product)
- Threshold used for the confusion matrix (default 0.5? optimal-F1?) is undisclosed
- Per-segment performance (does AUC hold for vip_loyal vs dormant?)
- Training-data window and model-card link
- Drift / monitoring panel (today vs last training)

---

## Cross-page consistency check

KPIs that **should** match across the three pages:

| KPI | Page 00 | Page 01 | Page 02 | Match? |
|---|---|---|---|---|
| Total Customers | 20,000 | 20,000 | (n/a) | OK |
| Avg Churn Prob | 31.31% | 31.31% | (n/a) | OK |
| High Risk count | 5,717 | 5,717 | (n/a) | OK |
| Risk donut shares | 57.9 / 18 / 13.5 / 10.6 | 57.9 / 18 / 13.5 / 10.6 | (n/a) | OK |
| ML AUC | (n/a) | 0.8852 | 0.8852 | OK |
| DL AUC | (n/a) | 0.8860 | 0.8860 | OK |
| Ensemble AUC | (n/a) | 0.8866 | 0.8866 | OK |
| ML Precision | (n/a) | 0.5331 | headline 0.5331 / **matrix 0.7059** | **CONFLICT inside Page 02** |
| ML Recall | (n/a) | 0.7791 | headline 0.7791 / **matrix 0.6000** | **CONFLICT inside Page 02** |
| Ensemble Precision | (n/a) | 0.6426 | headline 0.6426 / **matrix 0.7647** | **CONFLICT inside Page 02** |
| Ensemble Recall | (n/a) | 0.6621 | headline 0.6621 / **matrix 0.6500** | **CONFLICT inside Page 02** |
| Histogram leftmost bin (0–0.05) | ≈ **4,000** | ≈ **3,500** | (n/a) | **CONFLICT** — same 20k roster, two different bin-0 counts |
| Critical count | donut 18% ⇒ 3,600 | 3,596 | (n/a) | OK (rounding) |
| High-only (50–75%) | donut 10.6% ⇒ 2,120 | 5,717 − 3,596 = 2,121 | (n/a) | OK (consistent within Page 01 if you subtract Critical from High≥50%) |

**Mismatches found:**
1. Page 00 histogram bin-0 ~4,000 vs Page 01 histogram bin-0 ~3,500 (same population of 20,000).
2. Page 02 headline P/R contradicts Page 02 confusion-matrix P/R — large gaps for ml_model (Δ Precision +17.3 pts, Δ Recall −17.9 pts) and ensemble (Δ Precision +12.2 pts).
3. Page 02 confusion-matrix sample size (600) is not the test-set size that backs the headline numbers (20k or unspecified split).

---

## SaaS-readiness verdict

**Verdict: DO-NOT-SHIP.**

**Top 3 blockers:**
1. **Internal Page 02 contradiction.** Headline Precision/Recall and the confusion-matrix Precision/Recall report different values for the same model — ml_model: 0.5331/0.7791 vs 0.7059/0.6000. A model-performance page that disagrees with itself cannot survive a customer review.
2. **Confusion matrix evaluated on n=600 (3% of the 20,000 population) with no disclosure**, while a "Best Model: ensemble" label is asserted on AUC differences of 0.0014 with no significance test, no CI, and an identical 1.0s training-time floor for all three models. These are simulator artefacts presented as production metrics.
3. **Histogram for the same 20,000-customer roster shows ~4,000 in the leftmost bin on Page 00 but ~3,500 on Page 01.** Cross-page numerical inconsistency on the most prominent chart kills buyer trust before the model details are even read. Add to this the absence of data freshness, model version, class balance, confidence intervals, and threshold definitions across all three pages.

Until (1)–(3) are resolved and a "Synthetic / Demo" watermark is replaced with real telemetry plus a model card, the dashboard should not be put in front of paying pilots.
