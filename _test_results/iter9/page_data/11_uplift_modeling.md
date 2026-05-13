# Page 11 — Uplift Modeling (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## KPI cards
| Label | Value |
|---|---|
| Avg Uplift Score | **0.0434** |
| Avg Treatment Effect | **0.0434** |
| Persuadable Customers | **16,317** |
| Sleeping Dogs | **3,683** |

🚨 **Avg Uplift Score == Avg Treatment Effect to 4 decimals**.
🚨 16,317 + 3,683 = 20,000 → **4-quadrant collapsed to 2** (Sure Thing / Lost Cause missing as headline KPIs).

## Distribution of Uplift Scores
- x: -0.5 → 0.5
- y: 0 → 10k
- Mean annotation: 0.0434

## Distribution of Treatment Effects (visually identical to Uplift)
- Same x range, same y range, same mean shown — just a different label

## Uplift Score vs Treatment Effect scatter
- x = Uplift Score (-0.5 → 0.5)
- y = Treatment Effect (-0.8 → 0.6)
- Color by 4 segments (sure_thing, persuadable, lost_cause, sleeping_dog)

## Average Uplift by Segment (4 quadrant uplift segments — different from Page 10's 8 uplift segments)
| Segment | Avg Uplift |
|---|---:|
| persuadable | 0.1902 |
| sure_thing | 0.0560 |
| lost_cause | 0.0258 |
| sleeping_dog | -0.1097 |

🚨 **Page 11 uses 4 quadrant labels** {persuadable, sure_thing, lost_cause, sleeping_dog} while Page 10 / Page 05 use 8 labels {high/mid/low × value × persuadable/sure_thing/lost_cause/sleeping_dog}. **Different taxonomies on adjacent pages**.

## Customer Response Classification
| Class | Share |
|---|---:|
| Persuadable | 81.6% |
| Lost Cause | 18.4% |

🚨 The headline says "16,317 Persuadable + 3,683 Sleeping Dogs", but the pie chart calls the second group "Lost Cause", not "Sleeping Dog". And Sure Thing + Lost Cause segments do show up in the bar chart below ("lost_cause", "persuadable", "sure_thing", "sleeping_dog" all visible). **Inconsistent vocabulary** within the same page.

## Issues
1. **Avg Uplift Score == Avg Treatment Effect (0.0434)** — false equivalence; one variable plotted as two.
2. **4-quadrant collapsed in headline** to Persuadable + Sleeping Dogs.
3. **Pie says "Persuadable / Lost Cause"** but headline KPI says "Persuadable / Sleeping Dogs" — naming inconsistency on same page.
4. **Sleeping_dog avg uplift = -0.1097 (negative)** — they should be excluded from coupon eligibility, but no inline guardrail visible.
5. **Page 11 taxonomy (4 categories) ≠ Page 10/05 taxonomy (8 categories)**.
6. **No CIs on uplift scores**.
