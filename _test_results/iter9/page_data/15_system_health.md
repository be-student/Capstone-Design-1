# Page 15 — System Overview & Health (full data dump)

## Banners
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- ✅ Redis Streaming
- ✅ MLflow Tracking
- ✅ ML Pipeline
- "Connected to MLflow tracking server"
- "No experiments found on MLflow server."
- "Current Drift Status: RED"
- "Best model: ensemble (AUC = 0.8866)"

## Status header
- "✅ System Status: All Systems Operational"
- "Last checked: 2026-05-10T14:20:53.742949"
- "3/3 services healthy"

## Service Health KPIs
| Service | Label | Value |
|---|---|---|
| Redis Streaming | Connected | Yes |
| Redis Streaming | Stream (requests) | **0** |
| Redis Streaming | Stream (responses) | **0** |
| MLflow Tracking | Connected | **Yes** |
| MLflow Tracking | Experiments | **0** |
| ML Pipeline | Artifacts | 44 |
| ML Pipeline | Models | 4 |

🚨 **DIRECT CONTRADICTION with Page 14**:
- Page 14 banner: "MLflow tracking server **not available**. Showing cached experiment data from artifacts."
- Page 15 banner: "Connected to MLflow tracking server" + KPI "MLflow Tracking Connected: **Yes**"
- **Two different pages of the same dashboard report opposite states of MLflow.**

🚨 **MLflow KPI says Experiments=0 but "Total Runs=3"** — same page, two MLflow facts in mutual contradiction.

## Streaming Pipeline Configuration (visible JSON)
```json
{
  "host": "redis",
  "port": 6379,
  "request_stream": "scoring_requests",
  "response_stream": "scoring_responses",
  "consumer_group": "scoring_consumers",
  "batch_size": 10,
  "max_stream_length": 10000,
  "cache_ttl_seconds": 3600
}
```

## Throughput KPIs
| Label | Value |
|---|---|
| Avg Throughput | 49.0 req/min |
| Avg Latency | 19.1 ms |
| Avg Error Rate | **0.0103** (1.03%) |

🚨 Same numbers as Page 08 — page reuses without re-deriving.

## Experiment Run History KPIs
| Label | Value |
|---|---|
| Total Runs | **3** |
| Best AUC | 0.8866 |
| Best Model | ensemble |
| Total Train Time | 3s |

## Charts
1. **Stream Lengths** (bar) — both 0 for scoring_requests and scoring_responses.
2. **Scoring Throughput (24h)** (line) — Oct 15 2024 timestamps.
3. **AUC by Model Type** (bar) — same 0.8866/0.8860/0.8852.
4. **Model Performance Over Time** — x = `11:56:50.2892 → 11:56:50.2893` May 10 2026 (0.1ms span).
5. **PSI Drift Trend** — x = May 10 2026 11:56:50.282 → .2835 (1.5ms span), single point.

## Issues
1. **Page 14 vs Page 15 contradict on MLflow status** (direct, same dashboard).
2. **Page 15 internal contradiction**: Experiments=0 vs Total Runs=3.
3. **All-green checkmarks** on services that have nothing inside them (0 streams, 0 experiments).
4. **Time anchor split**: throughput dated Oct 15 2024, drift dated May 10 2026 — same page.
5. **Drift Status RED** while header says "All Systems Operational" with green ✅.
6. **No "last checked" timestamps per service** — only one global timestamp.
7. **Service "health" reduced to "Connected: Yes/No" + counter** — no probe age, no disk space, no queue lag, no stale-model alarm.
