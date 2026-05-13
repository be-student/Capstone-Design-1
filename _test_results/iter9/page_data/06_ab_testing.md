# Page 06 — A/B Testing (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## Section structure
- H2: A/B Testing Results
- H3: Power Analysis & Sample Size Calculator
- (also present: "MDE Sensitivity Analysis" mentioned in body but no separate H3 detected)

## KPI cards
| Group | Label | Value |
|---|---|---|
| Headline | Total Experiments | **0** |
| Headline | Significant Results | **0** |
| Headline | Best Experiment | **N/A** |
| Headline | Avg Lift | **0.0%** |
| Power calc | Required Sample Size (per group) | 906 |
| Power calc | Total Participants Needed | 1,812 |
| Power calc | Expected Duration (days) | 19 |

## Power Analysis inputs (sliders)
- Baseline Churn Rate: 0.20 (range 0.01–0.50)
- Minimum Detectable Effect (MDE): 0.05 (range 0.01–0.20)
- Significance Level (α): 0.05
- Target Power (1-β): 0.8

## Charts
1. **Power Curve: Sample Size vs Statistical Power** (line) — x=500–2500, y=0–100%. Target marker 80% at n=906.
2. **MDE Sensitivity Analysis** (mentioned but chart not detected by selector)

## Issues
1. **Zero-state framing**: 0/0/N/A/0.0% headline KPIs read as "broken model" rather than "no experiments yet". Need empty-state UI.
2. **No reconciliation against the 20,000-customer pool**: power calc says 1,812 needed; we have 20,000 idle — page should suggest "you have 11x the headroom; here's how to launch one".
3. **No A/B literature disclaimer**: real-world lift is 5-15%; if simulator ever produces something outside that range, no inline warning.
4. **No experiment audit log** (history of past experiments and outcomes).
5. **No multi-test correction** (Bonferroni, BH FDR) controls — assumes single-experiment world.
