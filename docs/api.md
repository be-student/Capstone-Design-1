# API Reference

> **Customer Churn Prediction & Retention Optimization System — Python Module API Reference**

This document provides the complete API reference for all major Python modules in the churn prediction system. For REST API endpoint documentation, see the Swagger UI at `http://localhost:8000/docs` when running the pipeline service.

---

## Table of Contents

1. [Uplift Modeling](#1-uplift-modeling) — `src.models.uplift_model`
2. [CLV Prediction](#2-clv-prediction) — `src.models.clv_model`
3. [A/B Testing](#3-ab-testing) — `src.models.ab_testing` / `src.analysis.ab_testing`
4. [Budget Optimization](#4-budget-optimization) — `src.models.budget_optimizer`
5. [Survival Analysis](#5-survival-analysis) — `src.models.survival_model`
6. [Recommendations](#6-recommendations) — `src.models.recommendations`
7. [Streaming](#7-streaming) — `src.streaming`
8. [Monitoring](#8-monitoring) — `src.monitoring`
9. [Cohort Analysis](#9-cohort-analysis) — `src.analysis.cohort_analysis`

---

## 1. Uplift Modeling

**Module:** `src.models.uplift_model`

Uplift modeling estimates individual treatment effects (ITE) to identify which customers are most likely to respond positively to retention interventions.

### Class: `UpliftModel`

```python
from src.models.uplift_model import UpliftModel

model = UpliftModel(config: dict, learner: str = "t_learner")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | — | Configuration dict; uses `simulation.random_seed` and `uplift` section |
| `learner` | `str` | `"t_learner"` | Meta-learner approach: `"t_learner"` or `"s_learner"` |

#### Methods

##### `fit(X, treatment, y) → UpliftModel`

Train the uplift model on labeled data with treatment assignments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `array-like (n_samples, n_features)` | Feature matrix |
| `treatment` | `array-like (n_samples,)` | Binary treatment indicator (1 = treated, 0 = control) |
| `y` | `array-like (n_samples,)` | Binary outcome variable (1 = churned, 0 = retained) |

**Returns:** `self` (fitted model instance)

##### `predict_uplift(X) → np.ndarray`

Predict individual treatment effects for new customers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `array-like (n_samples, n_features)` | Feature matrix |

**Returns:** `np.ndarray` of shape `(n_samples,)` — uplift scores per customer. Positive values indicate treatment is beneficial.

##### `segment_customers(uplift_scores) → np.ndarray`

Classify customers into four uplift quadrants.

| Parameter | Type | Description |
|-----------|------|-------------|
| `uplift_scores` | `array-like (n_samples,)` | Predicted uplift scores |

**Returns:** `np.ndarray` of shape `(n_samples,)` — string labels from `{"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}`

| Segment | Description |
|---------|-------------|
| `persuadable` | Positive uplift — customer responds to treatment |
| `sure_thing` | Will retain regardless of treatment |
| `lost_cause` | Will churn regardless of treatment |
| `sleeping_dog` | Treatment has negative effect — do not contact |

##### `compute_auuc(y, uplift, treatment) → float`

Calculate Area Under the Uplift Curve metric.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `array-like` | True outcome labels |
| `uplift` | `array-like` | Predicted uplift scores |
| `treatment` | `array-like` | Treatment assignment indicators |

**Returns:** `float` — AUUC value (higher is better)

##### `save(path) → None`

Persist the fitted model to disk as a pickle file.

##### `load(path) → UpliftModel` *(classmethod)*

Load a previously saved model.

**Example:**

```python
from src.models.uplift_model import UpliftModel

config = {"simulation": {"random_seed": 42}}
model = UpliftModel(config, learner="t_learner")
model.fit(X_train, treatment_train, y_train)

uplift_scores = model.predict_uplift(X_test)
segments = model.segment_customers(uplift_scores)
auuc = model.compute_auuc(y_test, uplift_scores, treatment_test)

model.save("models/uplift_model.pkl")
loaded = UpliftModel.load("models/uplift_model.pkl")
```

---

## 2. CLV Prediction

**Module:** `src.models.clv_model`

Customer Lifetime Value prediction using ML-based 12-month value regression with gradient boosting. Supports churn-adjusted CLV and proportional budget allocation.

### Class: `CLVModel`

```python
from src.models.clv_model import CLVModel

model = CLVModel(config: dict)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict` | Configuration dict with optional `simulation.random_seed` and `clv` section |

#### Methods

##### `fit(X, y) → CLVModel`

Train the CLV regression model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `pd.DataFrame` | Customer features (RFM features auto-engineered if present) |
| `y` | `pd.Series` | Target CLV values in KRW |

**Returns:** `self`

##### `predict(X) → np.ndarray`

Predict CLV for each customer. Values are clipped to be non-negative.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `pd.DataFrame` | Customer features |

**Returns:** `np.ndarray` — predicted CLV values (non-negative, in KRW)

##### `rank_customers(customer_ids, X) → pd.DataFrame`

Rank customers by predicted CLV in descending order.

| Parameter | Type | Description |
|-----------|------|-------------|
| `customer_ids` | `Sequence[str]` | Customer identifiers |
| `X` | `pd.DataFrame` | Customer features |

**Returns:** `pd.DataFrame` with columns `["customer_id", "predicted_clv"]`, sorted descending by CLV

##### `allocate_budget(customer_ids, X, total_budget) → pd.DataFrame`

Allocate retention budget proportionally to predicted CLV.

| Parameter | Type | Description |
|-----------|------|-------------|
| `customer_ids` | `Sequence[str]` | Customer identifiers |
| `X` | `pd.DataFrame` | Customer features |
| `total_budget` | `float` | Total budget to allocate (KRW) |

**Returns:** `pd.DataFrame` with columns `["customer_id", "predicted_clv", "allocated_budget"]`. Budget sums to `total_budget`, all values non-negative.

##### `adjust_for_churn(predicted_clv, churn_prob) → np.ndarray` *(static method)*

Compute churn-adjusted CLV: `CLV × (1 − churn_prob)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `predicted_clv` | `array-like` | Raw predicted CLV values |
| `churn_prob` | `array-like` | Churn probabilities (0–1) |

**Returns:** `np.ndarray` — adjusted CLV values. Higher churn probability → lower adjusted CLV.

##### `save(path) → None` / `load(path) → CLVModel` *(classmethod)*

Persist and restore fitted model state.

**Example:**

```python
from src.models.clv_model import CLVModel

model = CLVModel(config)
model.fit(X_train, y_train)

clv = model.predict(X_test)
adjusted = CLVModel.adjust_for_churn(clv, churn_probs)
ranking = model.rank_customers(ids, X_test)
budget = model.allocate_budget(ids, X_test, total_budget=10_000_000)
```

---

## 3. A/B Testing

**Modules:** `src.models.ab_testing`, `src.analysis.ab_testing`

Statistical A/B testing framework with power analysis, multiple comparison corrections, and experiment management.

### Class: `PowerAnalysis`

```python
from src.models.ab_testing import PowerAnalysis
```

All methods are static.

##### `PowerAnalysis.required_sample_size(baseline_rate, mde, alpha, power) → int`

Calculate required sample size per group for a two-proportion z-test.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline_rate` | `float` | — | Baseline conversion/churn rate |
| `mde` | `float` | — | Minimum detectable effect (absolute) |
| `alpha` | `float` | `0.05` | Significance level |
| `power` | `float` | `0.80` | Statistical power |

**Returns:** `int` — required sample size per group

##### `PowerAnalysis.minimum_detectable_effect(baseline_rate, sample_size, alpha, power) → float`

Calculate the minimum detectable effect given a fixed sample size.

##### `PowerAnalysis.compute_power(baseline_rate, mde, sample_size, alpha) → float`

Compute statistical power for given parameters.

### Class: `ABTestFramework`

```python
from src.models.ab_testing import ABTestFramework

framework = ABTestFramework(config: dict)
```

#### Methods

##### `create_experiment(name, hypothesis, variants, ...) → dict`

Create and register a new experiment.

##### `assign_groups(data, experiment_name, id_column) → pd.DataFrame`

Randomly assign customers to control/treatment groups.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Customer data |
| `experiment_name` | `str` | Registered experiment name |
| `id_column` | `str` | Column name for customer IDs |

**Returns:** `pd.DataFrame` — data with added `group` column

##### `compute_significance(data, metric, group_column) → dict`

Perform statistical significance test (z-test, t-test, or chi-square depending on metric type).

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Experiment data with outcomes |
| `metric` | `str` | Column name of metric to test |
| `group_column` | `str` | Column identifying control/treatment |

**Returns:** `dict` with keys: `statistic`, `p_value`, `significant`, `effect_size`, `test_type`

##### `analyze(data, metric, group_column) → dict`

Full experiment analysis with significance testing and confidence intervals.

##### `get_summary() → dict`

Return summary of all registered experiments.

##### `save(path) → None` / `load(path) → ABTestFramework` *(classmethod)*

Persist and restore framework state.

### Class: `MultipleComparisonCorrection`

```python
from src.analysis.ab_testing import MultipleComparisonCorrection
```

All methods are static. Used when running multiple simultaneous tests.

##### `bonferroni(p_values, alpha=0.05) → dict`

Apply Bonferroni correction to control family-wise error rate (FWER).

##### `fdr_bh(p_values, alpha=0.05) → dict`

Apply Benjamini-Hochberg procedure for false discovery rate (FDR) control.

##### `holm_bonferroni(p_values, alpha=0.05) → dict`

Apply Holm-Bonferroni step-down correction.

**Return dict keys:** `adjusted_p_values`, `rejected`, `alpha`, `method`, `n_rejected`

**Example:**

```python
from src.models.ab_testing import ABTestFramework, PowerAnalysis
from src.analysis.ab_testing import MultipleComparisonCorrection

# Power analysis
n = PowerAnalysis.required_sample_size(baseline_rate=0.15, mde=0.03)

# Run experiment
fw = ABTestFramework(config)
fw.create_experiment("retention_coupon", hypothesis="Coupons reduce churn")
data = fw.assign_groups(customers, "retention_coupon", "customer_id")
result = fw.analyze(data, metric="churned", group_column="group")

# Multiple comparison correction
p_vals = [result1["p_value"], result2["p_value"], result3["p_value"]]
corrected = MultipleComparisonCorrection.bonferroni(p_vals, alpha=0.05)
```

---

## 4. Budget Optimization

**Module:** `src.models.budget_optimizer`

Linear programming (LP) based budget allocation optimizing expected retained value across customers and channels.

### Class: `BudgetOptimizer`

```python
from src.models.budget_optimizer import BudgetOptimizer

optimizer = BudgetOptimizer(config: dict)
```

#### Methods

##### `optimize(data, total_budget=None) → pd.DataFrame`

Solve the LP problem for optimal budget allocation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Customer data (see required columns below) |
| `total_budget` | `float \| None` | Total budget constraint; uses config default if None |

**Required DataFrame columns:**

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | `str` | Unique customer identifier |
| `clv` | `float` | Predicted customer lifetime value |
| `uplift_score` | `float` | Treatment uplift score |
| `churn_prob` | `float` | Churn probability |
| `cost_per_action` | `float` | Cost of retention action |
| `expected_retention_lift` | `float` | Expected retention probability increase |

**Returns:** `pd.DataFrame` with columns `["customer_id", "allocated_budget"]`

##### `optimize_multi_channel(data, channels, channel_costs, channel_budgets, total_budget) → pd.DataFrame`

Optimize allocation across multiple communication channels with per-channel budget constraints.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Customer data |
| `channels` | `List[str]` | Channel names (e.g., `["email", "sms", "push"]`) |
| `channel_costs` | `Dict[str, float]` | Cost per contact per channel |
| `channel_budgets` | `Dict[str, float]` | Per-channel budget limits |
| `total_budget` | `float` | Overall budget constraint |

##### `simulate_scenario(data, scenario_name, total_budget, ...) → dict`

Run what-if scenario analysis with parameter multipliers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario_name` | `str` | — | Descriptive scenario name |
| `total_budget` | `float \| None` | `None` | Budget override |
| `cost_multiplier` | `float` | `1.0` | Scale costs by this factor |
| `uplift_multiplier` | `float` | `1.0` | Scale uplift scores |
| `churn_multiplier` | `float` | `1.0` | Scale churn probabilities |
| `clv_multiplier` | `float` | `1.0` | Scale CLV values |

**Returns:** `dict` with keys: `scenario_name`, `allocation`, `total_allocated`, `expected_value`, `roi`

##### `vary_parameter(data, parameter, values, base_budget) → pd.DataFrame`

Sensitivity analysis: sweep a single parameter across a range of values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `parameter` | `str` | One of: `"cost"`, `"uplift"`, `"churn"`, `"clv"` |
| `values` | `List[float]` | Multiplier values to test |
| `base_budget` | `float` | Base budget for all scenarios |

**Returns:** `pd.DataFrame` — one row per value with ROI and allocation metrics

##### `compare_strategies(data, total_budget) → pd.DataFrame`

Compare LP-optimal, proportional, and uniform allocation strategies side by side.

##### `compare_budget_scenarios(data, scenarios) → pd.DataFrame`

Run and compare multiple named budget scenarios.

##### `compute_roi(allocation, data) → float`

Compute expected retained value (ROI) for a given allocation.

##### `get_lp_diagnostics() → Optional[dict]`

Return LP solver status and diagnostics from the last optimization run.

##### `save(path) → None` / `load(path) → BudgetOptimizer` *(classmethod)*

Persist and restore optimizer state (JSON format).

**Example:**

```python
from src.models.budget_optimizer import BudgetOptimizer

opt = BudgetOptimizer(config)
allocation = opt.optimize(customer_data, total_budget=50_000_000)
roi = opt.compute_roi(allocation, customer_data)

# What-if analysis
scenario = opt.simulate_scenario(
    customer_data, "pessimistic",
    cost_multiplier=1.5, uplift_multiplier=0.8
)

# Strategy comparison
comparison = opt.compare_strategies(customer_data, total_budget=50_000_000)
```

---

## 5. Survival Analysis

**Module:** `src.models.survival_model`

Cox Proportional Hazards model for time-to-churn prediction with Kaplan-Meier survival curves.

### Class: `SurvivalModel`

```python
from src.models.survival_model import SurvivalModel

model = SurvivalModel(config: Optional[dict] = None)
```

#### Methods

##### `fit(X, duration, event) → SurvivalModel`

Fit a Cox Proportional Hazards model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `pd.DataFrame` | Covariate features |
| `duration` | `pd.Series` | Observed time durations (days) |
| `event` | `pd.Series` | Event indicator (1 = churned, 0 = censored) |

**Returns:** `self`

##### `predict_survival(X, t) → np.ndarray`

Predict survival probability at a specific time point.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `pd.DataFrame` | Customer features |
| `t` | `float` | Time point (days) |

**Returns:** `np.ndarray` of shape `(n_samples,)` — survival probabilities in [0, 1]

##### `predict_hazard(X) → np.ndarray`

Predict partial hazard scores `exp(X @ β)`. Higher values indicate higher churn risk.

##### `median_survival_time(X) → np.ndarray`

Predict median time-to-churn for each customer (in days).

##### `get_survival_curve(duration, event) → pd.DataFrame`

Compute Kaplan-Meier survival curve from population data.

**Returns:** `pd.DataFrame` with columns `["timeline", "survival_probability"]`

##### `summary() → pd.DataFrame`

Return Cox model coefficient summary including hazard ratios and p-values.

#### Properties

##### `concordance_index → float` *(read-only)*

Concordance index of the fitted model (0.5 = random, 1.0 = perfect).

##### `save(path) → None` / `load(path) → SurvivalModel` *(classmethod)*

Persist and restore fitted model.

**Example:**

```python
from src.models.survival_model import SurvivalModel

model = SurvivalModel(config)
model.fit(X_train, duration_train, event_train)

# Predict 30-day survival probability
surv_30 = model.predict_survival(X_test, t=30)

# Hazard scores for risk ranking
hazard = model.predict_hazard(X_test)

# Median survival time
median_t = model.median_survival_time(X_test)

# Population-level KM curve
km_curve = model.get_survival_curve(duration_all, event_all)

print(f"C-index: {model.concordance_index:.3f}")
print(model.summary())
```

---

## 6. Recommendations

**Module:** `src.models.recommendations`

Personalized retention action recommendation engine combining churn risk, CLV, uplift scores, and customer segments.

### Class: `RecommendationEngine`

```python
from src.models.recommendations import RecommendationEngine

engine = RecommendationEngine(config: dict)
```

#### Methods

##### `get_available_actions() → List[str]`

Return list of available retention action types.

**Default actions (6 types):**

| Action | Base Cost (KRW) |
|--------|----------------|
| `coupon` | 5,000 |
| `push_notification` | 100 |
| `email_campaign` | 200 |
| `loyalty_points` | 3,000 |
| `personal_outreach` | 10,000 |
| `exclusive_offer` | 8,000 |

##### `recommend(data) → pd.DataFrame`

Generate the single best recommendation for each customer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Customer data (see required columns below) |

**Required DataFrame columns:**

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `customer_id` | `str` | Yes | Customer identifier |
| `churn_prob` | `float` | Yes | Churn probability |
| `clv` | `float` | Yes | Predicted CLV |
| `uplift_score` | `float` | Yes | Uplift score |
| `segment` | `str` | Yes | Customer segment label |
| `push_opt_in` | `bool` | No | Push notification opt-in |
| `email_opt_in` | `bool` | No | Email opt-in |

**Returns:** `pd.DataFrame` with columns: `customer_id`, `action_type`, `score`, `estimated_cost`, `reason`

##### `recommend_top_k(data, k=3) → pd.DataFrame`

Return top-k recommendations per customer, sorted by score descending.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Customer data |
| `k` | `int` | `3` | Number of recommendations per customer |

**Returns:** Same columns as `recommend()`, with up to `k` rows per customer.

**Scoring weights:**

| Weight | Value | Description |
|--------|-------|-------------|
| `w_churn` | 0.35 | Churn probability weight |
| `w_clv` | 0.25 | Customer lifetime value weight |
| `w_uplift` | 0.20 | Uplift score weight |
| `w_segment` | 0.20 | Segment priority weight |

##### `save(path) → None` / `load(path) → RecommendationEngine` *(classmethod)*

Persist and restore engine state (JSON format).

**Example:**

```python
from src.models.recommendations import RecommendationEngine

engine = RecommendationEngine(config)

# Single best recommendation per customer
recs = engine.recommend(customer_data)

# Top-3 recommendations per customer
top3 = engine.recommend_top_k(customer_data, k=3)

# Available actions
print(engine.get_available_actions())
# ['coupon', 'push_notification', 'email_campaign', ...]
```

---

## 7. Streaming

**Module:** `src.streaming`

Redis Streams-based real-time scoring pipeline with producer/consumer architecture.

### Class: `RedisStreamProducer`

```python
from src.streaming.redis_producer import RedisStreamProducer

producer = RedisStreamProducer(config: dict)
```

**Configuration keys** (under `config["redis"]`):

| Key | Env Override | Default | Description |
|-----|-------------|---------|-------------|
| `host` | `REDIS_HOST` | `"localhost"` | Redis server host |
| `port` | `REDIS_PORT` | `6379` | Redis server port |
| `db` | `REDIS_DB` | `0` | Redis database number |
| `stream_name` | — | `"scoring_requests"` | Stream name for requests |
| `stream_maxlen` | — | `10000` | Maximum stream length |

#### Methods

##### `connect() → bool`

Establish connection to Redis server.

##### `publish(features) → str`

Publish a single scoring request to the stream.

| Parameter | Type | Description |
|-----------|------|-------------|
| `features` | `Dict[str, Any]` | Customer feature dictionary |

**Returns:** `str` — Redis stream message ID

##### `publish_batch(features_list) → List[str]`

Publish multiple scoring requests atomically.

##### `get_stream_length() → int`

Return current number of messages in the stream.

##### `health_check() → dict`

Check producer health and Redis connectivity.

**Returns:** `dict` with keys: `status`, `connected`, `stream_length`, `timestamp`

##### `close() → None`

Close the Redis connection.

---

### Class: `RedisStreamConsumer`

```python
from src.streaming.redis_consumer import RedisStreamConsumer

consumer = RedisStreamConsumer(config: dict)
```

**Additional configuration keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `response_stream` | `"scoring_responses"` | Stream for publishing results |
| `consumer_group` | `"scoring_group"` | Consumer group name |
| `consumer_name` | `"consumer_1"` | Consumer instance name |
| `consumer_batch_size` | `10` | Messages per read |
| `consumer_block_ms` | `1000` | Blocking read timeout (ms) |

#### Methods

##### `connect() → bool`

Establish connection and create consumer group.

##### `set_scoring_api(scoring_api) → None`

Attach a scoring API instance for processing predictions.

##### `process_message(message_id, data) → dict`

Process a single message and return prediction result.

##### `process_one() → Optional[dict]`

Read and process the next message from the stream.

##### `start(max_iterations=None) → None`

Start blocking event loop for continuous message processing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_iterations` | `int \| None` | Max messages to process; `None` = infinite |

##### `stop() → None`

Signal the consumer to stop the event loop.

##### `health_check() → dict`

Check consumer health and pending message count.

#### Properties

##### `processed_count → int`

Total number of messages processed by this consumer instance.

**Example:**

```python
from src.streaming.redis_producer import RedisStreamProducer
from src.streaming.redis_consumer import RedisStreamConsumer

# Producer side
producer = RedisStreamProducer(config)
producer.connect()
msg_id = producer.publish({"customer_id": "C00847", "recency": 15, "frequency": 8})
producer.close()

# Consumer side
consumer = RedisStreamConsumer(config)
consumer.connect()
consumer.set_scoring_api(scoring_api)
consumer.start(max_iterations=100)  # Process 100 messages then stop
consumer.close()
```

---

## 8. Monitoring

**Module:** `src.monitoring`

Model drift detection using Population Stability Index (PSI) and Kolmogorov-Smirnov tests with configurable alert thresholds.

### Enum: `AlertLevel`

```python
from src.monitoring.monitoring_service import AlertLevel

AlertLevel.GREEN   # "green"  — No significant drift
AlertLevel.YELLOW  # "yellow" — Moderate drift, monitor closely
AlertLevel.RED     # "red"    — Significant drift, action required
```

### Dataclass: `MonitoringResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `psi_report` | `dict` | PSI drift detection report |
| `ks_report` | `dict` | KS drift detection report |
| `overall_alert_level` | `AlertLevel` | Highest alert level across all checks |
| `drifted_features` | `List[str]` | Feature names exceeding drift thresholds |
| `timestamp` | `str` | ISO 8601 timestamp |

**Properties:**
- `has_drift → bool` — `True` if any features exceeded drift thresholds

**Methods:**
- `to_dict() → dict` — JSON-serializable representation

---

### Class: `ModelMonitoringService`

```python
from src.monitoring.monitoring_service import ModelMonitoringService

monitor = ModelMonitoringService(config: dict)
```

#### Methods

##### `fit(reference) → ModelMonitoringService`

Fit drift detectors with reference (training) data distributions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | `pd.DataFrame` | Reference dataset (e.g., training data) |

##### `check(production) → MonitoringResult`

Run drift detection on production data against the reference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `production` | `pd.DataFrame` | Current production data |

**Returns:** `MonitoringResult` — complete drift analysis with alert levels

##### `register_alert_callback(callback) → None`

Register a callback function triggered when drift is detected.

| Parameter | Type | Description |
|-----------|------|-------------|
| `callback` | `Callable[[MonitoringResult], None]` | Function called on drift alerts |

---

### Class: `DriftDetector` (PSI-based)

```python
from src.monitoring.drift_detection import DriftDetector

detector = DriftDetector.from_config(config)
```

#### Methods

##### `from_config(config) → DriftDetector` *(classmethod)*

Create detector from configuration dictionary.

##### `fit(reference) → None`

Fit reference distributions and bin edges.

##### `detect(production, features=None) → DriftReport`

Detect drift using PSI metric.

| Parameter | Type | Description |
|-----------|------|-------------|
| `production` | `pd.DataFrame` | Production data |
| `features` | `List[str] \| None` | Specific features to check; `None` = all |

### Dataclass: `DriftAlert`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `psi_value` | `float` | — | Computed PSI value |
| `yellow_threshold` | `float` | `0.10` | Warning threshold |
| `red_threshold` | `float` | `0.25` | Critical threshold |
| `level` | `str` | computed | `"green"`, `"yellow"`, or `"red"` |
| `is_drifted` | `bool` | computed | `True` if PSI ≥ red threshold |

### Dataclass: `DriftReport`

| Attribute | Type | Description |
|-----------|------|-------------|
| `feature_psi` | `Dict[str, float]` | PSI value per feature |
| `feature_alerts` | `Dict[str, DriftAlert]` | Alert status per feature |

**Methods:**
- `summary() → dict` — Keys: `total_features`, `drifted_features`, `yellow_features`, `green_features`, `max_psi_feature`, `max_psi_value`
- `to_dict() → dict`

---

### Class: `KSDriftDetector`

```python
from src.monitoring.ks_drift import KSDriftDetector

detector = KSDriftDetector(
    numerical_features: List[str],
    categorical_features: List[str],
    warning_threshold: float = 0.05,
    drift_threshold: float = 0.01
)
```

#### Methods

##### `auto_detect(data, warning_threshold=0.05, drift_threshold=0.01) → KSDriftDetector` *(classmethod)*

Create detector with auto-detected feature types.

##### `detect(production) → KSDriftReport`

Run KS tests on production data.

### Function: `calculate_psi()`

```python
from src.monitoring.drift_detection import calculate_psi

psi = calculate_psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
    bin_edges: Optional[np.ndarray] = None,
    binning_strategy: str = "quantile",
    epsilon: float = 1e-6
) → float
```

**Example:**

```python
from src.monitoring.monitoring_service import ModelMonitoringService

monitor = ModelMonitoringService(config)
monitor.fit(training_data)

# Check production batch for drift
result = monitor.check(production_batch)

if result.has_drift:
    print(f"Alert: {result.overall_alert_level.value}")
    print(f"Drifted features: {result.drifted_features}")

# Register automated alert
monitor.register_alert_callback(lambda r: send_slack_alert(r))
```

---

## 9. Cohort Analysis

**Module:** `src.analysis.cohort_analysis`

Lifecycle cohort analysis with retention matrices, churn rates, half-life computation, and visualization.

### Class: `CohortAnalyzer`

```python
from src.analysis.cohort_analysis import CohortAnalyzer

analyzer = CohortAnalyzer(config: Optional[dict] = None)
```

#### Methods

##### `assign_cohorts(data, cohort_type=None) → pd.DataFrame`

Assign cohort labels to customer data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Customer transaction/event data |
| `cohort_type` | `str \| None` | Config default | `"monthly"`, `"weekly"`, or `"behavioral"` |

**Returns:** `pd.DataFrame` — input data with added `cohort` column

##### `compute_retention_matrix(cohort_data, max_periods=None) → pd.DataFrame`

Compute retention rates as a cohort × period matrix.

**Returns:** `pd.DataFrame` — rows = cohorts, columns = period indices, values = retention rates (0–1)

##### `compute_cohort_metrics(cohort_data, metrics=None) → Dict[str, pd.DataFrame]`

Compute multiple metrics aggregated by cohort.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `List[str] \| None` | Metrics to compute; default: all available |

**Available metrics:** `retention_rate`, `revenue`, `avg_order_value`, `churn_rate`, `customer_count`

**Returns:** `Dict[str, pd.DataFrame]` — one DataFrame per metric

##### `get_cohort_summary(cohort_data) → pd.DataFrame`

Return per-cohort summary statistics.

**Returns:** `pd.DataFrame` with columns: `total_customers`, `total_events`, `avg_lifetime_periods`, `total_revenue`

##### `filter_cohorts(cohort_data, cohorts=None, min_size=None) → pd.DataFrame`

Filter data to specific cohorts or by minimum cohort size.

##### `get_retention_curves(retention_matrix) → Dict[str, np.ndarray]`

Extract individual retention curves per cohort.

##### `get_average_retention_curve(retention_matrix) → np.ndarray`

Compute the average retention curve across all cohorts.

##### `compute_half_life(retention_matrix) → pd.Series`

Compute retention half-life (periods until retention drops below 50%) per cohort.

##### `compute_churn_rates(retention_matrix) → pd.DataFrame`

Compute incremental churn rates: `1 − retention(t) / retention(t-1)`.

##### `compute_cumulative_revenue(cohort_data, max_periods=None) → pd.DataFrame`

Compute cumulative revenue per cohort over time. Values are non-decreasing.

##### `full_analysis(data, cohort_type=None) → dict`

Run the complete cohort analysis pipeline.

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `cohort_data` | `pd.DataFrame` | Data with cohort assignments |
| `retention_matrix` | `pd.DataFrame` | Retention rate matrix |
| `retention_curves` | `Dict[str, np.ndarray]` | Per-cohort curves |
| `avg_retention_curve` | `np.ndarray` | Average curve |
| `churn_rates` | `pd.DataFrame` | Incremental churn rates |
| `half_life` | `pd.Series` | Half-life per cohort |
| `summary` | `pd.DataFrame` | Cohort summary statistics |
| `metrics` | `Dict[str, pd.DataFrame]` | All computed metrics |

#### Visualization Methods

##### `plot_retention_heatmap(retention_matrix, ...) → plt.Figure`

Plot retention matrix as a color-coded heatmap.

##### `plot_retention_lines(retention_matrix, ...) → plt.Figure`

Plot retention curves as line charts.

##### `plot_cohort_sizes(cohort_data, ...) → plt.Figure`

Plot bar chart of cohort sizes.

**Example:**

```python
from src.analysis.cohort_analysis import CohortAnalyzer

analyzer = CohortAnalyzer(config)
results = analyzer.full_analysis(transaction_data, cohort_type="monthly")

print(results["summary"])
print(f"Average half-life: {results['half_life'].mean():.1f} periods")

# Visualization
fig = analyzer.plot_retention_heatmap(results["retention_matrix"])
fig.savefig("results/retention_heatmap.png")
```

---

## Appendix A: Module Summary

| Module | Import Path | Primary Class | Key Method |
|--------|-------------|---------------|------------|
| Uplift | `src.models.uplift_model` | `UpliftModel` | `predict_uplift()` |
| CLV | `src.models.clv_model` | `CLVModel` | `predict()` |
| A/B Testing | `src.models.ab_testing` | `ABTestFramework` | `analyze()` |
| Budget | `src.models.budget_optimizer` | `BudgetOptimizer` | `optimize()` |
| Survival | `src.models.survival_model` | `SurvivalModel` | `predict_survival()` |
| Recommendations | `src.models.recommendations` | `RecommendationEngine` | `recommend_top_k()` |
| Streaming (Producer) | `src.streaming.redis_producer` | `RedisStreamProducer` | `publish()` |
| Streaming (Consumer) | `src.streaming.redis_consumer` | `RedisStreamConsumer` | `start()` |
| Monitoring | `src.monitoring.monitoring_service` | `ModelMonitoringService` | `check()` |
| Drift (PSI) | `src.monitoring.drift_detection` | `DriftDetector` | `detect()` |
| Drift (KS) | `src.monitoring.ks_drift` | `KSDriftDetector` | `detect()` |
| Cohort | `src.analysis.cohort_analysis` | `CohortAnalyzer` | `full_analysis()` |

## Appendix B: Common Data Structures

All modules use standard **pandas DataFrames** and **numpy arrays** as primary data structures. Models follow a consistent interface:

- **`fit()`** — Train/fit the model on data
- **`predict*()`** — Generate predictions
- **`save(path)`** — Persist model to disk
- **`load(path)`** *(classmethod)* — Restore model from disk

Configuration is passed as Python `dict` objects, typically loaded from YAML files in the `config/` directory (e.g., `config/simulator_config.yaml`).

## Appendix C: Configuration Files

| File | Purpose |
|------|---------|
| `config/simulator_config.yaml` | Main configuration for all modules |
| `config/api_config.yaml` | REST API server settings |

All configurable parameters (thresholds, hyperparameters, budget defaults) are defined in these YAML files and passed to modules via the `config` dict parameter.
