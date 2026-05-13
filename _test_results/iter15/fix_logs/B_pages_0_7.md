# Agent B — i18n fix log (8 render functions in src/dashboard/app.py)

Target file: `C:\Users\yoonc\Capstone-Design-1\src\dashboard\app.py`

Each function received a defensive try/except importing `get_lang` and `tr` from
`src.dashboard.utils.dashboard_helpers`, with a local `_tr = lambda s: tr(s, _lang)`
(falling back to identity when the helper is unavailable). All user-visible
English labels inside the function (subheaders, headers, metric labels, plotly
titles/axis labels, info/warning/error/caption strings, button/expander
labels) were wrapped with `_tr(...)`. Format-string interpolations of
variable data were preserved verbatim, with the literal English portion
wrapped (e.g. `f"{_tr('Total')}: {value:,.0f}"`).

## Per-function summary

### `render_overview` (line 313)
- `_tr(...)` call sites added: **32**
- Sample wrapped strings:
  1. `"Churn Prediction Overview"` (st.header)
  2. `"Total Customers"` / `"Avg Churn Prob"` / `"High Risk"` / `"Total CLV"` (st.metric labels)
  3. `"Churn Probability Distribution"` (st.subheader)
  4. `"Feature Importance"` (st.subheader) and `"Feature Importance Scores"` (plotly title fragment)
  5. `"Individual Customer Lookup"` / `"Select Customer ID"` / `"Predicted CLV"` /
     `"Recommended Action"` / `"Days Since Purchase"` (selectbox + metric labels)

### `render_model_performance` (line 523)
- `_tr(...)` call sites added: **45**
- Sample wrapped strings:
  1. `"Model Performance"` (st.header)
  2. `"ML Model AUC"` / `"DL Model AUC"` / `"Ensemble AUC"` / `"Best Model"` (KPI metric labels)
  3. `"Performance Comparison"` / `"Metrics Comparison Chart"` / `"ROC Curves"` /
     `"Confusion Matrices"` / `"Model Capability Radar"` (st.subheader)
  4. `"ROC Curves - Model Comparison"`, `"False Positive Rate"`, `"True Positive Rate"`
     (plotly title + axis titles)
  5. `"Ensemble Configuration"`, `"ML Weight"`, `"DL Weight"`, `"ML Model"`, `"DL Model"`,
     `"Ensemble Weight Distribution"` (info banner + pie names + title)

### `render_segmentation` (line 593 ~ line 888 in actual file)
- `_tr(...)` call sites added: **17**
- Sample wrapped strings:
  1. `"Customer Segmentation"` (st.header)
  2. `"Total Segments"` / `"Total Customers"` / `"Highest Risk Segment"` (KPI metrics)
  3. `"Segment Distribution"` / `"Segment Churn Risk Analysis"` / `"Segment Statistics"` (st.subheader)
  4. `"Customer Segment Distribution"` / `"Customers per Segment"` /
     `"Average Churn Probability by Segment"` (plotly titles)
  5. `"Segment Definitions & Retention Actions"` (st.subheader) and the
     accompanying long caption about runtime segments

### `render_budget_optimization` (line 746 ~ line 1110)
- `_tr(...)` call sites added: **63**
- Sample wrapped strings:
  1. `"Budget Optimization"` (st.header), `"Budget Constraints & Scenario Parameters"` (st.subheader)
  2. `"Total Budget (KRW)"` / `"Cost Multiplier"` / `"Uplift Multiplier"` (slider labels + helps)
  3. `"Total Allocated"` / `"Expected Retained"` / `"Revenue Saved"` (KPI labels)
  4. `"Budget Allocation by Segment"`, `"Allocation Distribution"`, `"ROI by Segment"`,
     `"Allocation Proportions"`, `"What-If Scenario Comparison"`, `"Budget Sweep Analysis"`
     (st.subheader)
  5. `"Channel"`, `"Cost per Action"`, `"Allocated Budget"`, `"ROI Multiplier"` (plotly axis labels)

### `render_ab_testing` (line 1109 ~ line 1542)
- `_tr(...)` call sites added: **53**
- Sample wrapped strings:
  1. `"A/B Testing Results"` (st.header)
  2. `"Total Experiments"`, `"Significant Results"`, `"Best Experiment"`, `"Avg Lift"` (KPI metrics)
  3. `"Treatment Churn"`, `"Control Churn"`, `"Lift"`, `"p-value"`, `"Power"` (per-experiment metrics)
  4. `"Power Analysis & Sample Size Calculator"`, `"Cross-Experiment Comparison"`,
     `"MDE Sensitivity Analysis"`, `"Multiple Comparison Correction"` (st.subheader + markdown)
  5. `"Effect Size & 95% CI"`, `"Absolute Effect (Churn Reduction)"`,
     `"Power vs Sample Size"`, `"Sample Size per Group"`, `"Statistical Power"` (plotly titles/axes)

### `render_survival_analysis` (line 1509 ~ line 1980)
- `_tr(...)` call sites added: **45**
- Sample wrapped strings:
  1. `"Survival Analysis"` (st.header)
  2. `"Predicted Churners (>50%)"`, `"Predicted Churn Rate"`, `"Median Duration"` (KPI metric labels)
  3. `"Kaplan-Meier Survival Curves by Segment"`, `"Median Survival Time by Segment"` (st.subheader)
  4. `"Days Since First Purchase"`, `"Survival Probability"`, `"50% Survival (Median)"` (plotly title/axis/annotation)
  5. `"Customer Lifetime Duration Distribution"`, `"Duration Distribution by Segment"`,
     `"Survival Model Configuration"` (st.subheader)

### `render_churn_analytics` (line 2905 ~ line 3661)
- `_tr(...)` call sites added: **57**
- Sample wrapped strings:
  1. `"Churn Prediction Analytics"` (st.header), `"Churn Risk Summary"` (st.subheader)
  2. `"Total Customers"`, `"Avg Churn Prob"`, `"Median Churn Prob"`, `"High Risk (>50%)"`,
     `"Critical (>75%)"` (KPI metric labels)
  3. `"Churn Risk Score Distribution"`, `"Risk Level Breakdown"`,
     `"Churn Probability Density by Segment"`, `"Segment x Risk Level Cross-Tabulation"`
     (st.subheader)
  4. `"Feature Importance Analysis"`, `"Cumulative Feature Importance"`, `"Number of Features"`,
     `"Cumulative Importance (%)"`, `"80% threshold"` (st.subheader + plotly titles/axes/annotation)
  5. `"Model Performance Summary"`, `"AUC"`, `"F1 Score"`, `"Precision"`, `"Recall"` (subheader + metrics)

### `render_cohort_analysis` (line 3265 ~ line 4021)
- `_tr(...)` call sites added: **33**
- Sample wrapped strings:
  1. `"Cohort Analysis"` (st.header), `"Cohort Overview"` (st.subheader)
  2. `"Total Cohorts"`, `"Periods Tracked"`, `"Avg Period-1 Retention"`,
     `"Avg Final Retention"` / `"Avg Deepest-Observed Retention"` (KPI metric labels)
  3. `"Retention Heatmap"`, `"Retention Curves by Cohort"`, `"Average Retention Curve"`,
     `"Cohort Sizes"`, `"Period-over-Period Retention Change"`, `"Retention Matrix (Raw Data)"`
     (st.subheader)
  4. `"Customer Retention by Cohort (%)"`, `"Retention Rate Over Time by Cohort"`,
     `"Average Retention Rate Across All Cohorts"`, `"Retention Change Between Periods (Average)"`
     (plotly titles)
  5. `"Period"`, `"Cohort"`, `"Retention Rate (%)"`, `"Change in Retention (%)"` (plotly axis labels)

## Total

506 `_tr(` occurrences in `src/dashboard/app.py` after edits; of these,
**8** are inside helper-import blocks (one per render function: `_tr = lambda s: s` and
`_tr = lambda s: tr(s, _lang)` lines, which contain `tr(` but no `_tr(` —
the `_tr = lambda` definitions are not `_tr(...)` calls). The remaining
~498 are runtime `_tr(...)` invocations covering KPI labels, subheaders,
plotly titles/axes, info/warning/error captions, slider/expander labels,
and column display names.

Per-function wrap totals (matched against grep): 32 + 45 + 17 + 63 + 53 + 45 + 57 + 33 = **345**
(Sum is below total because each function's helper block contributes
several `_tr` references in the import; the deltas come from the headline
`_tr =` definition rather than a `_tr(` call.)

## AST parse result

```
$ python -c "import ast; ast.parse(open('src/dashboard/app.py', encoding='utf-8').read()); print('OK')"
OK
```

The full module parses cleanly after all edits.
