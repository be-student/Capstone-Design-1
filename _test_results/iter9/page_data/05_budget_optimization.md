# Page 05 — Budget Optimization (full data dump)

## Banners
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- "Channel configuration not found in config. Add budget.channels to simulator_config.yaml for multi-channel allocation views."

## Section structure
- H2: Budget Optimization
- H3: Budget Constraints & Scenario Parameters
- H3: Allocation Summary
- H3: Budget Allocation by Segment
- H3: Allocation Distribution
- H3: ROI by Segment
- H3: Allocation Proportions
- H3: Channel-Level Cost Breakdown (empty — banner explains)
- H3: What-If Scenario Comparison
- H3: Budget Sweep Analysis

## Allocation Summary KPI cards
| Label | Value |
|---|---|
| Total Allocated | 50,000,000 KRW |
| Expected Retained | 118 |
| Revenue Saved | 192,155,551 KRW |
| Avg ROI | **3.5x** |

**Aggregate ROI cross-check:** 192,155,551 / 50,000,000 = **3.843x**. Headline says **3.5x** → ROI definition trap (mean-of-segment-ROIs vs aggregate).

## Budget Allocation by Segment (8 uplift segments)
| Segment | Allocated (KRW) | % of total |
|---|---:|---:|
| mid_value_persuadable | 21,062,000 | 42.1% |
| low_value_persuadable | 13,922,000 | 27.8% |
| mid_value_sure_thing | 12,781,000 | 25.6% |
| low_value_sure_thing | 1,524,000 | 3.05% |
| high_value_sure_thing | 680,000 | 1.36% |
| high_value_persuadable | 31,000 | 0.062% |
| high_value_lost_cause | 0 | 0% |
| sleeping_dog | 0 | 0% |

Sum: 49,000,000 → discrepancy of 1,000,000 vs "Total Allocated 50,000,000" KPI ❓

## Expected ROI by Customer Segment (bar)
ROI (x-axis 0–8, no per-segment numbers extracted from bar; 8x is the visual max). high_value_persuadable shows ROI ≈ 8x but only gets 31,000 KRW — suggests LP is constrained on segment-size, not ROI.

## What-If Scenario Comparison
Scenarios visible:
- Baseline
- Current Selection
- Conservative (-30%)
- Aggressive (+50%)
- Cost Reduction

ROI axis 3 to 5. Total Allocated bars 0–60M.

## Expected Retained Customers by Scenario
| Scenario | Retained |
|---|---:|
| Baseline | 122 |
| Current Selection | 122 |
| Conservative (-30%) | 68 |
| Aggressive (+50%) | 220 |
| Cost Reduction | 122 |

**Baseline 122 ≠ headline "Expected Retained 118"** — 4-customer mismatch unexplained.

## Budget Sweep Analysis
Dual-axis plot: x=Budget 0–200M, y1=Expected Retained 0–500, y2=Revenue Saved 0–800M.

## Issues identified
1. **Avg ROI 3.5x vs aggregate 3.84x** — same page, two computations.
2. **high_value_persuadable receives 31,000 (0.062%)** while showing the highest ROI (≈8x) — LP allocation looks misaligned with displayed ROI.
3. **sum(allocations) = 49M ≠ Total Allocated 50M** — 1M unaccounted.
4. **Baseline scenario 122 retained vs "Current Selection / Allocation Summary" 118 retained** — same baseline, two different headcounts.
5. Channel-level breakdown is empty (graceful banner, but H3 leaves a hole on the page).
6. No CIs on retained-customer point estimates; "Aggressive +50% retains 220" is a single number with no uncertainty.
