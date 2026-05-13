# A1 — Overview / Churn Analytics / Model Performance

## Page 00 — Overview

### Visible KPIs
- Banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- Title: "Churn Prediction Overview"
- Green coverage banner: "Churn predictions cover all 20,000 customers."
- Total Customers: 20,000
- Avg Churn Prob: 31.31%
- High Risk: 5,717
- Total CLV: 57,936,514,970 ... (truncated with ellipsis, currency unit not shown)
- Chart: "Churn Probability Distribution" / "Distribution of Churn Probabilities" — histogram, x-axis Churn Probability 0-0.9+, y-axis count up to ~4000. Mass concentrated near 0 (~4000), declining, then a notable secondary peak near 0.9 (~1700).
- Section heading "Risk Level Distribution" visible but chart cut off.
- Sidebar: Churn Definition (No purchase: 30 days, No login: 60 days, Operator: OR), Budget Total: 50,000,000 KRW, Ensemble Weights ML: 0.6 | DL: 0.4.

### Wrong / suspicious
- "Total CLV 57,936,514,970 ..." is visually truncated with an ellipsis — the number is not fully readable and has no currency symbol. For 20k customers this implies ~2.9M per customer; given Budget is in KRW, 2.9M KRW per customer is plausible, but the value being clipped on a headline KPI is a defect.
- Avg Churn Prob 31.31% suggests average risk; yet 5,717 / 20,000 = 28.6% are flagged "High Risk." The "High Risk" threshold is undefined on this page, so the KPI cannot be interpreted.
- Histogram is bimodal with a hard spike at the rightmost bin (~1,700 customers at p≈0.9). That shape is unusual for a calibrated churn model and suggests probability saturation/clipping rather than a natural distribution.
- The leftmost bin alone contains ~4,000 customers — 20% of the entire base sits in a single 0.05-wide probability bucket; this hints at score collapse to near-zero.

### Unreliable
- Top-of-page banner explicitly says "Synthetic data — FULL mode … All KPIs are simulator-generated." Every number on this page is therefore non-production by the vendor's own admission.
- All KPIs are point estimates — no confidence interval, no comparison vs. prior period, no delta arrow.
- Ensemble weights "ML: 0.6 | DL: 0.4" appear hardcoded in a sidebar with no provenance (who set them, when, on what validation set).
- Churn definition (30/60-day OR rule) is hardcoded in the sidebar — looks like a config dump, not a governed business rule.

### Missing
- No "as of" timestamp / data freshness indicator.
- No model version, training date, or scoring run ID.
- No currency label on Total CLV (we infer KRW only from the Budget panel).
- No definition of "High Risk" threshold on this page (the analytics page uses >50%, but Overview never says so).
- No segment selector, no date range, no cohort filter.
- No comparison to previous period / no trend.
- No customer-count reconciliation footnote (active vs. total, opted-out, etc.).
- No link to methodology, model card, or data lineage.

## Page 01 — Churn Analytics

### Visible KPIs
- Same synthetic-data banner and 20,000-customer coverage banner.
- Title: "Churn Prediction Analytics"
- Section: "Churn Risk Summary"
  - Total Customers: 20,000
  - Avg Churn Prob: 31.31%
  - Median Churn Prob: 15.39%
  - High Risk (>50%): 5,717
  - Critical (>75%): 3,596
- Chart: "Churn Risk Score Distribution — Distribution of Churn Risk Scores with Threshold Boundaries"
  - X-axis Churn Probability 0-0.9, Y-axis Customer Count up to 3,500
  - Threshold lines drawn at 0.25 (Low/Medium, green dashed), 0.50 (Medium/High, orange dashed), 0.75 (High/Critical, red dashed)
  - Leftmost bin ~3,500 count; long flat plateau roughly 0.1-0.7 around 200-300 count; secondary hump around 0.85-0.9 (~700 count).

### Wrong / suspicious
- KPI inconsistency vs. Overview: Overview's histogram leftmost bin reached ~4,000 customers, but the same probability axis on this page maxes the leftmost bin at ~3,500. The two pages purport to show the same 20,000-customer distribution and should have consistent extreme bins or explicit different binning notes — neither is provided.
- Threshold story is contradictory: "High Risk (>50%)" = 5,717 and "Critical (>75%)" = 3,596 means 62.9% of "High Risk" customers are also "Critical." When the critical bucket is the majority of the high-risk bucket, the labels stop discriminating and look ill-calibrated.
- Mean (31.31%) is more than 2× the median (15.39%). That's a heavily right-skewed score — consistent with the visible spike near p≈0.9 — and again hints that the model is producing a bimodal, possibly poorly-calibrated, score, not a smooth probability.
- 3,596 / 20,000 = 18.0% of customers labeled "Critical (>75%)". An 18% critical-churn rate is operationally implausible for any going-concern SaaS book of business.

### Unreliable
- "Synthetic data … All KPIs are simulator-generated" banner persists — vendor self-discloses that none of these figures reflect real customers.
- Thresholds 0.25 / 0.50 / 0.75 are evenly spaced round numbers — almost certainly hardcoded defaults rather than calibrated to business cost/value curves.
- No calibration plot, no Brier score, no reliability diagram — yet the page is selling probability cutoffs.
- Identical "Total Customers 20,000" / "Avg 31.31%" / "High Risk 5,717" reproduced verbatim from Overview suggests a single shared snapshot, not independent computations.

### Missing
- No confidence intervals or bootstrap bands on the histogram.
- No segmentation (tier, geography, tenure, plan).
- No "as of" date, model version, scoring batch ID.
- No threshold-tuning rationale (why 25/50/75 and not data-driven cutpoints).
- No counts for Low and Medium tiers — only High and Critical are surfaced.
- No churn outcome backtest (predicted vs. observed) on this analytics page.
- No export / drill-down affordance visible.

## Page 02 — Model Performance

### Visible KPIs
- Synthetic data banner present.
- Title: "Model Performance"
- Headline metrics:
  - ML Model AUC: 0.8852
  - DL Model AUC: 0.8860
  - Ensemble AUC: 0.8866
  - Best Model: ensemble
- Green callout: "Ensemble AUC: 0.8866 (>= 0.78 threshold)"
- Performance Comparison table (yellow-highlighted cells indicate per-column best):

| Model     | auc    | precision | recall | f1_score | accuracy |
|-----------|--------|-----------|--------|----------|----------|
| ml_model  | 0.8852 | 0.5331    | 0.7791 (hi) | 0.6331 | 0.8209 |
| dl_model  | 0.8860 | 0.6759 (hi) | 0.6318 | 0.6531 (hi) | 0.8671 (hi) |
| ensemble  | 0.8866 (hi) | 0.6426 | 0.6621 | 0.6522 | 0.8602 |

- Bar chart "Model Metrics Comparison" with three series (ml_model blue, dl_model orange, ensemble green) — first metric grouping all near 0.88, second group ~0.53/0.67/0.64, third ~0.78/0.63/0.66, fourth ~0.63/0.65/0.65, fifth ~0.82/0.86/0.86.

### Wrong / suspicious
- "Best Model: ensemble" is declared, but the table shows the ensemble wins **only AUC (0.8866)**. The DL model wins precision (0.6759), f1_score (0.6531), and accuracy (0.8671); the ML model wins recall (0.7791). The "best model" pick rests on a 0.0006 AUC edge over DL and a 0.0014 edge over ML — well within statistical noise — while the ensemble loses 3 of 5 head-to-head metrics. This is a defensible-on-AUC-only claim presented as a definitive winner.
- All three AUCs are within 0.0014 of each other (0.8852 / 0.8860 / 0.8866). Differences of 4 decimal places at this magnitude almost never survive a confidence interval; reporting them to four decimals without CIs is over-precision.
- "Ensemble AUC: 0.8866 (>= 0.78 threshold)" — the 0.78 deployment gate is unsourced. No link to the policy that set it.
- ml_model precision 0.5331 with recall 0.7791 implies a low operating threshold; dl_model has the inverse trade-off. The "ensemble" is numerically just an average and does not demonstrate a true Pareto improvement.
- Accuracy of 0.82-0.87 is not informative without class-balance disclosure. With 18% "Critical" churn (per page 01), a trivial "predict no churn" baseline would already score ~0.71-0.82 accuracy depending on the positive class definition. Accuracy is shown without that baseline.

### Unreliable
- Synthetic data banner: by the vendor's own statement these AUCs are computed on simulator output, not real holdout data.
- No train/validation/test split disclosed; no fold count; no date range of evaluation.
- No standard errors, no DeLong test for AUC differences, no McNemar for paired classifiers.
- Threshold used to compute precision/recall/f1/accuracy is not stated — the same model can be tuned to top any of these columns.
- Hardcoded-looking ensemble weights (0.6/0.4 from sidebar) suggest the ensemble isn't optimized, just blended.

### Missing
- No confidence intervals on AUC (or any metric).
- No ROC / PR curves visible on this page; only a bar chart of point estimates.
- No calibration plot, Brier score, or log loss.
- No confusion matrix.
- No model version, training date, dataset version, feature count, or run ID.
- No drift / stability metric (PSI, KS) vs. prior version.
- No business-cost-weighted metric (expected value, profit curve).
- No champion/challenger history — three models compared once with no time series.
- No fairness / segment-level breakdown of metrics.
- No statement of evaluation horizon (prospective vs. retrospective, observation window length).

## SaaS-readiness verdict for these 3 pages
**Verdict:** DO-NOT-SHIP

**Top 3 blockers:**
1. The product self-labels every KPI as "Synthetic data — FULL mode … simulator-generated" at the top of every page. No paying customer will accept dashboards that explicitly disclaim their own numbers; procurement will reject on data provenance alone.
2. "Best Model: ensemble" is declared on a 0.0006 AUC margin while the DL model wins precision, F1, and accuracy and the ML model wins recall — with no confidence intervals, no statistical test, no operating threshold, and no calibration. The "winner" claim does not survive enterprise model-risk review.
3. Risk segmentation is internally inconsistent and likely miscalibrated: Critical (>75%) at 18% of the base, Critical being 63% of High-Risk, mean (31.31%) at >2× median (15.39%), a hard spike at p≈0.9 in the histogram, and a leftmost-bin count that differs between Overview (~4,000) and Churn Analytics (~3,500) for the same 20,000 customers. Combined with hardcoded 25/50/75 thresholds and 0.6/0.4 ensemble weights, the risk tiers are not defensible as production segments.

Supporting blockers: no data freshness timestamp anywhere, no model version, no currency label on Total CLV (and the headline value is visually truncated with "…"), no confidence intervals, no segment/date filters, no class-balance disclosure for accuracy.
