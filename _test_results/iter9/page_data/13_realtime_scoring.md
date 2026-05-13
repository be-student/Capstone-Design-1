# Page 13 — Real-Time Scoring & Recommendations (full data dump, all 3 tabs)

## Banners
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- "Redis: Connected"
- "Recommended Offer: no_action"

## Tabs
- Live Scoring Status (a)
- Retention Offer Recommendations (b)
- Model Monitoring (c)

## All KPIs (all 3 tabs DOM-rendered eagerly)

### Tab a — Live Scoring Status
| Group | Label | Value |
|---|---|---|
| Service Health | Request Stream | **0** |
| Service Health | Response Stream | **0** |
| Service Health | Consumer Group | scoring_consumers |
| Recent Scores | Total Scores | **200** |
| Recent Scores | Avg Churn Prob | 27.30% |
| Recent Scores | High/Critical Risk | 17 |
| Recent Scores | Primary Model | ensemble |

🚨 **Request Stream = 0 / Response Stream = 0** but **Total Scores = 200** — KPI cards contradict each other (queue depth vs lifetime processed labels not differentiated).

### Tab b — Retention Offer Recommendations
| Label | Value |
|---|---|
| Total Offers | **44** |
| Total Cost | 1,196,659 KRW |
| Expected Revenue Saved | 10,752,341 KRW |
| Expected ROI | **8.0x** |
| Recommendation | no_action |
| Expected Uplift | **1.46%** |
| Priority Score | **1.00** |

🚨 **ROI math check**: 10,752,341 / 1,196,659 = **8.985x**. Card says **8.0x**. Off by ~1.

🚨 **44 offers / 200 scored** = 22% (or 44 / 20,000 = 0.22% if denominator is total population) — denominator not stated.

🚨 **Recommendation "no_action" with Priority 1.00** — "high priority but no action" contradiction at the customer level.

### Tab c — Model Monitoring
| Label | Value |
|---|---|
| Total Drift Checks | **1** |
| Red Alerts | 1 |
| Yellow Warnings | 0 |
| Latest Alert Level | **RED** |

## Charts (tab a)
1. **Scoring Requests per Minute** — x = Oct 15 2024 timestamps, y = 0–80
2. **Response Latency & Error Rate** dual-axis — x = Oct 15→Oct 16 2024, y1 = Latency 15–35 ms, y2 = Error Rate 0–5%
3. **Churn Probability Distribution (Recent Scores)** — histogram of 200 scores
4. **Risk Level Distribution** — bar {low, medium, high, critical}

## Charts (tab b)
5. **Offer Type Distribution** — pie
6. **Average Expected Uplift by Segment** — bar
7. **Cost vs Expected Revenue Saved by Segment** — grouped bar
8. **Churn Probability vs Expected Uplift** — scatter

## Charts (tab c)
9. **Drift Alerts Over Time** — single point on May 10 2026 11:56:50 (1.5ms x-axis)
10. **PSI Trend** — single point
11. **KS Statistic Trend** — single point
12. **Scoring Volume Over Time** — bar, ~50 hourly buckets, all uniformly value 4 (synthetic)
13. **Mean Churn Probability Over Time** — line + ±1σ band, ~50 points
14. **Model Type Usage in Recent Scoring** — pie

## Issues
1. **Time anchor mismatch**: throughput/latency charts show **Oct 15-16, 2024** while drift charts show **May 10, 2026 11:56:50** (1.5ms span). Same "real-time" page, ~19 months apart.
2. **Stream depth 0 vs Total Scores 200** — KPI label disambiguation missing.
3. **ROI math 8.0x card vs 8.985x actual** — off by ~10%.
4. **Drift trends with 1.5ms x-axis = single observation**.
5. **Scoring Volume = 4 uniformly across all buckets** — synthetic, not real telemetry.
6. **Total Drift Checks = 1 yet Status = RED with no PSI/KS values surfaced** in KPI cards.
7. **Customer lookup returns "no_action / Priority 1.00 / Uplift 1.46%"** — no inline explanation why high priority gets no action.
