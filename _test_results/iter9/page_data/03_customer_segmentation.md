# Page 03 — Customer Segmentation (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## Section structure
- H2: Customer Segmentation
- H3: Segment Distribution
- H3: Segment Churn Risk Analysis
- H3: Segment Statistics
- H3: CLV by Segment
- H3: Risk Level Distribution by Segment
- H3: Segment Definitions & Retention Actions

## KPI cards
| Label | Value |
|---|---|
| Total Segments | 6 |
| Total Customers | 20,000 |
| Highest Risk Segment | dormant |

## Segment Distribution (donut)
| Segment | Share |
|---|---:|
| regular_loyal | 24.7% |
| bargain_hunter | 20.4% |
| new_customer | 15.1% |
| explorer | 14.9% |
| dormant | 14.7% |
| vip_loyal | 10.2% |

## Segment Churn Risk (bar w/ sample sizes)
| Segment | n |
|---|---:|
| vip_loyal | 2,030 |
| regular_loyal | 4,949 |
| bargain_hunter | 4,087 |
| explorer | 2,975 |
| new_customer | 3,014 |
| dormant | 2,945 |

(Sum: 20,000 ✓)

Avg Churn ordering (low → high): vip_loyal, regular_loyal, bargain_hunter, explorer, new_customer, dormant.

## Average CLV by Segment
| Segment | Mean CLV (KRW) |
|---|---:|
| vip_loyal | 12,760,815 |
| regular_loyal | 3,248,249 |
| bargain_hunter | 1,932,498 |
| new_customer | 1,503,177 |
| explorer | 1,120,117 |
| dormant | 66,362 |

## Risk Level Distribution within Segments
Stacked-bar with risk_level dimensions {critical, high, low, medium}. Counts on each bar reach up to 4,000.

## Headline KPI says "Highest Risk Segment = dormant"
- Visible from Avg Churn chart, dormant ~0.85+
- BUT no explicit risk score shown next to the segment name in the KPI

## Segment Definitions section
The H3 "Segment Definitions & Retention Actions" exists but no table content extracted via stMetric/stDataFrame selectors — likely a custom st.markdown table; user-visible but not surfaced in DOM extractor.

## Cross-page consistency
- Total Customers 20,000 — matches Page 00, 01.
- Sample sizes per segment (2,030 / 4,949 / 4,087 / 2,975 / 3,014 / 2,945) match the share donut percentages × 20,000:
  - vip_loyal 10.15% (matches 10.2%)
  - regular_loyal 24.74% (matches 24.7%)
  - bargain_hunter 20.43% (matches 20.4%)
  - explorer 14.87% (matches 14.9%)
  - new_customer 15.07% (matches 15.1%)
  - dormant 14.72% (matches 14.7%)

→ Internal consistency on this page: ✓
