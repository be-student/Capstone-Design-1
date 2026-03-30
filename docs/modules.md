# Module Documentation

> Detailed API reference and usage examples for all analytics and optimization modules in the E-Commerce Customer Churn Prediction and Retention Optimization System.

---

## Table of Contents

1. [Uplift Modeling](#1-uplift-modeling)
2. [Customer Lifetime Value (CLV)](#2-customer-lifetime-value-clv)
3. [A/B Testing](#3-ab-testing)
4. [Budget Optimization](#4-budget-optimization)
5. [Survival Analysis](#5-survival-analysis)
6. [Recommendations](#6-recommendations)
7. [Cohort Analysis](#7-cohort-analysis)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. Uplift Modeling

**Module**: `src/models/uplift_model.py`
**Class**: `UpliftModel`
**Purpose**: Estimate the causal effect of a treatment (e.g., coupon, campaign) on individual customer churn probability using T-Learner or S-Learner meta-learner strategies.

### API Reference

#### Constructor

```python
UpliftModel(config: dict, learner: str = "t_learner")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | — | Configuration dictionary (see [Configuration Reference](#8-configuration-reference)) |
| `learner` | `str` | `"t_learner"` | Meta-learner strategy: `"t_learner"` (two separate models) or `"s_learner"` (single model with treatment as feature) |

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `fit(X, treatment, y)` | `self` | Train the uplift model on features, treatment indicator, and outcome |
| `predict_uplift` | `predict_uplift(X)` | `np.ndarray` | Predict individual treatment effects. Positive = treatment reduces churn |
| `segment_customers` | `segment_customers(uplift_scores)` | `np.ndarray` | Classify into 4 quadrants based on configurable thresholds |
| `compute_auuc` | `compute_auuc(y, uplift, treatment)` | `float` | Area Under the Uplift Curve metric |
| `save` | `save(path)` | `None` | Persist model to disk (pickle) |
| `load` | `load(path)` (classmethod) | `UpliftModel` | Restore model from disk |

**`fit(X, treatment, y)`**

```python
def fit(
    X: Union[pd.DataFrame, np.ndarray],      # Feature matrix
    treatment: Union[pd.Series, np.ndarray],  # Binary treatment indicator (0/1)
    y: Union[pd.Series, np.ndarray]           # Binary outcome (0/1, e.g., churned)
) -> UpliftModel
```

**`predict_uplift(X)`**

```python
def predict_uplift(
    X: Union[pd.DataFrame, np.ndarray]  # Feature matrix (same schema as training)
) -> np.ndarray  # Uplift scores; positive means treatment helps
```

**`segment_customers(uplift_scores)`**

Returns one of four segment labels per customer:

| Segment | Meaning |
|---------|---------|
| `"persuadable"` | High uplift — treatment reduces churn (target these) |
| `"sure_thing"` | Low uplift, low base churn — will stay regardless |
| `"lost_cause"` | Low uplift, high base churn — treatment won't help |
| `"sleeping_dog"` | Negative uplift — treatment increases churn (avoid!) |

### Usage Example

```python
import yaml
from src.models.uplift_model import UpliftModel

# Load config
with open("config/simulator_config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize and train
model = UpliftModel(config, learner="t_learner")
model.fit(X_train, treatment_train, y_train)

# Predict uplift and segment
uplift_scores = model.predict_uplift(X_test)
segments = model.segment_customers(uplift_scores)

# Evaluate
auuc = model.compute_auuc(y_test, uplift_scores, treatment_test)
print(f"AUUC: {auuc:.4f}")

# Persist
model.save("models/uplift_model.pkl")
loaded = UpliftModel.load("models/uplift_model.pkl")
```

### Config Keys

```yaml
simulation:
  random_seed: 42
uplift:
  n_estimators: 100
  max_depth: 4
  learning_rate: 0.1
  segment_thresholds:
    high_uplift_quantile: 0.75
    low_uplift_quantile: 0.25
```

---

## 2. Customer Lifetime Value (CLV)

**Module**: `src/models/clv_model.py`
**Class**: `CLVModel`
**Purpose**: Predict customer lifetime value using BG/NBD-inspired feature engineering with Gradient Boosting regression. Supports churn-adjusted CLV and proportional budget allocation.

### API Reference

#### Constructor

```python
CLVModel(config: dict)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | — | Configuration dictionary with `simulation.random_seed` |

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `fit(X, y)` | `self` | Train CLV model on RFM + behavioral features |
| `predict` | `predict(X)` | `np.ndarray` | Predict non-negative CLV values (KRW) |
| `rank_customers` | `rank_customers(customer_ids, X)` | `pd.DataFrame` | Rank customers by predicted CLV descending |
| `adjust_for_churn` | `adjust_for_churn(predicted_clv, churn_prob)` (static) | `np.ndarray` | Compute churn-adjusted CLV = CLV × (1 − churn_prob) |
| `allocate_budget` | `allocate_budget(customer_ids, X, total_budget)` | `pd.DataFrame` | Proportional budget allocation by CLV share |
| `save` | `save(path)` | `None` | Persist model to disk (pickle) |
| `load` | `load(path)` (classmethod) | `CLVModel` | Restore model from disk |

**`rank_customers(customer_ids, X)`**

```python
def rank_customers(
    customer_ids: Sequence[str],  # Customer identifiers
    X: pd.DataFrame               # Feature matrix
) -> pd.DataFrame  # Columns: customer_id, predicted_clv (sorted descending)
```

**`allocate_budget(customer_ids, X, total_budget)`**

```python
def allocate_budget(
    customer_ids: Sequence[str],
    X: pd.DataFrame,
    total_budget: float   # Total budget in KRW
) -> pd.DataFrame  # Columns: customer_id, allocated_budget
```

### Usage Example

```python
from src.models.clv_model import CLVModel

model = CLVModel(config)
model.fit(X_train, y_train)

# Predict CLV
clv_predictions = model.predict(X_test)

# Adjust for churn probability
adjusted_clv = CLVModel.adjust_for_churn(clv_predictions, churn_probabilities)

# Rank and allocate budget
rankings = model.rank_customers(customer_ids, X_test)
budget_plan = model.allocate_budget(customer_ids, X_test, total_budget=50_000_000)

# Save/Load
model.save("models/clv_model.pkl")
loaded = CLVModel.load("models/clv_model.pkl")
```

### Config Keys

```yaml
simulation:
  random_seed: 42
budget:
  total_krw: 50000000
```

---

## 3. A/B Testing

**Modules**:
- `src/models/ab_testing.py` — Core framework (`ABTestFramework`, `PowerAnalysis`)
- `src/analysis/ab_testing.py` — Statistical analysis (`StatisticalTestSuite`, `MultipleComparisonCorrection`, `ExperimentManager`)

### 3.1 Power Analysis

**Class**: `PowerAnalysis`
**Purpose**: Pre-experiment power calculations for sample size planning.

#### Static Methods

```python
PowerAnalysis.required_sample_size(
    baseline_rate: float,       # Expected baseline conversion/churn rate
    mde: float,                 # Minimum detectable effect (absolute)
    alpha: float = 0.05,        # Significance level
    power: float = 0.80,        # Statistical power
    two_sided: bool = True
) -> int  # Required sample size per group

PowerAnalysis.minimum_detectable_effect(
    n: int,                     # Sample size per group
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True
) -> float  # Smallest detectable effect

PowerAnalysis.compute_power(
    n: int,
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    two_sided: bool = True
) -> float  # Statistical power (0–1)
```

#### Usage Example

```python
from src.models.ab_testing import PowerAnalysis

# How many customers per group?
n = PowerAnalysis.required_sample_size(baseline_rate=0.15, mde=0.03)
print(f"Need {n} per group")

# What effect can we detect with 5000 per group?
mde = PowerAnalysis.minimum_detectable_effect(n=5000, baseline_rate=0.15)
print(f"MDE: {mde:.4f}")
```

### 3.2 A/B Test Framework

**Class**: `ABTestFramework`
**Purpose**: End-to-end A/B experiment lifecycle management with deterministic group assignment.

#### Constructor

```python
ABTestFramework(config: Dict[str, Any])
```

#### Key Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `create_experiment` | `create_experiment(name, description, metrics)` | `str` | Create experiment, returns experiment ID |
| `assign_groups` | `assign_groups(customer_ids, n_variants, seed)` | `pd.DataFrame` | Deterministic group assignment via hashing |
| `compute_significance` | `compute_significance(data, metric, alpha, test_type)` | `dict` | Auto-detect binary/continuous, run appropriate test |
| `analyze` | `analyze(data, metric, alpha, confidence_level)` | `dict` | Full analysis: effect size, CI, p-value |
| `get_summary` | `get_summary(data, metric, alpha)` | `dict` | Human-readable summary with lift and CI |
| `save` / `load` | — | — | JSON persistence |

**`compute_significance` return dict:**

```python
{
    "test_statistic": float,
    "p_value": float,
    "is_significant": bool,
    "test_used": str,    # "chi_square" or "t_test"
    "alpha": float
}
```

**`get_summary` return dict:**

```python
{
    "treatment_mean": float,
    "control_mean": float,
    "absolute_difference": float,
    "relative_lift": float,
    "confidence_interval": (float, float),
    "p_value": float
}
```

#### Usage Example

```python
from src.models.ab_testing import ABTestFramework

framework = ABTestFramework(config)

# Create experiment and assign groups
exp_id = framework.create_experiment("retention_campaign", metrics=["churn_rate"])
groups = framework.assign_groups(customer_ids, n_variants=2)

# After collecting data...
result = framework.analyze(experiment_data, metric="churn_rate")
summary = framework.get_summary(experiment_data, metric="churn_rate")
print(f"Lift: {summary['relative_lift']:.2%}, p={summary['p_value']:.4f}")
```

### 3.3 Statistical Test Suite

**Class**: `StatisticalTestSuite`
**Purpose**: Comprehensive statistical tests with effect size calculations and multiple comparison corrections.

#### Methods

| Method | Use Case | Effect Size |
|--------|----------|-------------|
| `t_test(group_a, group_b)` | Continuous metrics (Welch's t-test) | Cohen's d |
| `chi_square_test(group_a, group_b)` | Categorical outcomes (2×2 table) | Cramér's V |
| `mann_whitney_u_test(group_a, group_b)` | Non-parametric comparison | Rank-biserial |
| `z_test_proportions(group_a, group_b)` | Binary proportions | Cohen's h |
| `run_multiple_tests(data, metrics, ...)` | Test multiple metrics with correction | Per-metric |

#### Usage Example

```python
from src.analysis.ab_testing import StatisticalTestSuite, MultipleComparisonCorrection

suite = StatisticalTestSuite(alpha=0.05)

# Single test
result = suite.t_test(treatment_revenue, control_revenue)
print(f"Cohen's d = {result['effect_size']:.3f}, p = {result['p_value']:.4f}")

# Multiple metrics with correction
results = suite.run_multiple_tests(
    data=experiment_df,
    metrics=["revenue", "sessions", "churn"],
    group_column="group",
    correction_method="bonferroni"
)

# Direct multiple comparison correction
corrected = MultipleComparisonCorrection.fdr_bh([0.01, 0.04, 0.08])
```

### 3.4 Experiment Manager

**Class**: `ExperimentManager`
**Purpose**: Full experiment lifecycle with sequential testing (alpha-spending) support.

#### Key Methods

| Method | Description |
|--------|-------------|
| `create_experiment(name, ...)` | Create with optional sequential testing config |
| `record_result(exp_id, data, metric)` | Record interim result snapshot |
| `sequential_test(exp_id, data, metric)` | Run test with O'Brien-Fleming alpha spending |
| `get_summary(exp_id)` | Get experiment status and all results |
| `complete_experiment(exp_id)` | Mark experiment as complete |

#### Usage Example

```python
from src.analysis.ab_testing import ExperimentManager

mgr = ExperimentManager(config, alpha=0.05)

# Sequential experiment
exp_id = mgr.create_experiment(
    name="coupon_test",
    sequential=True,
    spending_function="obrien_fleming",
    n_interim_analyses=5
)

# Interim checks (e.g., every week)
for week_data in weekly_snapshots:
    result = mgr.sequential_test(exp_id, week_data, metric="churn_rate")
    if result["stop_early"]:
        print(f"Early stopping: {result['decision']}")
        break

mgr.complete_experiment(exp_id)
```

---

## 4. Budget Optimization

**Module**: `src/optimization/budget_optimizer.py`
**Classes**: `LPBudgetOptimizer`, `CostConfig`, `ChannelConstraint`, `OptimizationResult`, `WhatIfScenario`
**Purpose**: Linear programming-based budget allocation across channels and customers, maximizing expected retained CLV subject to channel and budget constraints.

### API Reference

#### CostConfig

```python
@dataclass
class CostConfig:
    channels: Dict[str, Dict[str, Any]]  # Channel definitions
    discount_rate: float = 0.10           # Annual discount rate
    time_horizon_months: int = 12
    currency: str = "KRW"
    total_budget: float = 50_000_000.0
```

| Method | Returns | Description |
|--------|---------|-------------|
| `get_channel_constraints()` | `List[ChannelConstraint]` | Convert config to constraint objects |
| `get_monthly_discount_factor()` | `float` | Monthly discount factor |
| `get_npv_factor()` | `float` | Net present value factor over horizon |
| `from_config(config)` (classmethod) | `CostConfig` | Load from config dict |

#### ChannelConstraint

```python
@dataclass
class ChannelConstraint:
    name: str
    min_budget: float = 0.0
    max_budget: Optional[float] = None
    cost_per_action: float = 1000.0
    expected_roi_multiplier: float = 1.0
```

#### LPBudgetOptimizer

```python
LPBudgetOptimizer(config: Dict[str, Any], channels: Optional[List[ChannelConstraint]] = None)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `solve` | `solve(data, total_budget, channels)` | `OptimizationResult` | Solve LP allocation problem |
| `get_customer_allocations` | `get_customer_allocations(result)` | `pd.DataFrame` | Per-customer budget totals |
| `get_channel_summary` | `get_channel_summary(result)` | `dict` | Per-channel budget totals |
| `compute_expected_value` | `compute_expected_value(result, data)` | `float` | Expected value of allocations |
| `simulate_budget_change` | `simulate_budget_change(data, budget_multiplier)` | `dict` | What-if for budget changes |
| `compare_scenarios` | `compare_scenarios(data, scenarios)` | `pd.DataFrame` | Side-by-side scenario comparison |
| `run_budget_sweep` | `run_budget_sweep(data, budget_range, n_steps)` | `pd.DataFrame` | Sweep across budget levels |
| `save` / `load` | — | — | JSON/pickle persistence |

#### OptimizationResult

```python
@dataclass
class OptimizationResult:
    allocations: pd.DataFrame      # Per-customer, per-channel allocations
    total_allocated: float
    total_budget: float
    objective_value: float
    channel_summary: Dict[str, float]
    status: str                    # "optimal", "infeasible", etc.
    solver_message: str
```

#### Required Input Data Columns

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | `str` | Unique customer identifier |
| `clv` | `float` | Predicted customer lifetime value |
| `uplift_score` | `float` | Treatment effect estimate |
| `churn_prob` | `float` | Churn probability |
| `cost_per_action` | `float` | Per-customer cost of intervention |

### Usage Example

```python
from src.optimization.budget_optimizer import LPBudgetOptimizer, WhatIfScenario

optimizer = LPBudgetOptimizer(config)

# Solve allocation
result = optimizer.solve(customer_data, total_budget=50_000_000)
print(f"Status: {result.status}, Objective: {result.objective_value:,.0f}")

# Per-customer breakdown
allocations = optimizer.get_customer_allocations(result)

# What-if analysis
impact = optimizer.simulate_budget_change(customer_data, budget_multiplier=1.2)

# Compare scenarios
scenarios = [
    WhatIfScenario(name="base", total_budget=50_000_000),
    WhatIfScenario(name="aggressive", total_budget=80_000_000, uplift_multiplier=1.1),
    WhatIfScenario(name="conservative", total_budget=30_000_000),
]
comparison = optimizer.compare_scenarios(customer_data, scenarios)

# Budget sweep
sweep = optimizer.run_budget_sweep(
    customer_data,
    budget_range=(10_000_000, 100_000_000),
    n_steps=10
)
```

### Config Keys

```yaml
optimization:
  total_budget: 50000000
  discount_rate: 0.10
  time_horizon_months: 12
  channels:
    email:
      cost_per_action: 1000
      expected_roi_multiplier: 1.0
      min_budget: 0
      max_budget: null
    sms:
      cost_per_action: 500
      expected_roi_multiplier: 0.8
    push_notification:
      cost_per_action: 100
      expected_roi_multiplier: 0.6
    coupon:
      cost_per_action: 5000
      expected_roi_multiplier: 1.5
    call_center:
      cost_per_action: 10000
      expected_roi_multiplier: 2.0
```

---

## 5. Survival Analysis

**Module**: `src/models/survival_analysis.py`
**Class**: `SurvivalModel`
**Purpose**: Time-to-churn modeling using Cox Proportional Hazards with Kaplan-Meier curve estimation. Predicts survival probabilities, hazard rates, and median survival times.

### API Reference

#### Constructor

```python
SurvivalModel(config: Optional[Dict[str, Any]] = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | `None` | Configuration with survival parameters |

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `fit(X, duration, event)` | `self` | Fit Cox PH model on censored data |
| `predict_survival` | `predict_survival(X, t)` | `np.ndarray` | Survival probability at time t ∈ [0, 1] |
| `predict_hazard` | `predict_hazard(X)` | `np.ndarray` | Partial hazard exp(Xβ), non-negative |
| `median_survival_time` | `median_survival_time(X)` | `np.ndarray` | Time when S(t) = 0.5 (inf if never) |
| `get_survival_curve` | `get_survival_curve(duration, event)` | `pd.DataFrame` | Kaplan-Meier curve data |
| `summary` | `summary()` | `pd.DataFrame` | Cox model coefficients and p-values |
| `concordance_index` | (property) | `float` | C-index model quality metric |
| `save` / `load` | — | — | Pickle persistence |

**`fit(X, duration, event)`**

```python
def fit(
    X: pd.DataFrame,       # Feature matrix
    duration: pd.Series,   # Time-to-event or censoring time
    event: pd.Series       # 1 = event (churn), 0 = censored (still active)
) -> SurvivalModel
```

**`get_survival_curve(duration, event)`**

```python
def get_survival_curve(
    duration: pd.Series,
    event: pd.Series
) -> pd.DataFrame  # Columns: timeline, survival_probability
```

### Usage Example

```python
from src.models.survival_analysis import SurvivalModel

model = SurvivalModel(config)
model.fit(X_train, duration_train, event_train)

# Survival at 6 months
surv_6m = model.predict_survival(X_test, t=6.0)

# Hazard rates
hazards = model.predict_hazard(X_test)

# Median survival time
medians = model.median_survival_time(X_test)

# Kaplan-Meier curve
km_curve = model.get_survival_curve(duration_train, event_train)

# Model summary
print(model.summary())
print(f"C-index: {model.concordance_index:.4f}")

# Persist
model.save("models/survival_model.pkl")
```

### Config Keys

```yaml
survival:
  penalizer: 0.01    # L2 regularization penalty
  l1_ratio: 0.0      # Elastic net mixing (0 = pure L2)
  alpha: 0.05         # Confidence interval significance level
```

---

## 6. Recommendations

**Module**: `src/models/recommendations.py`
**Class**: `RecommendationEngine`
**Purpose**: Generate personalized retention action recommendations per customer, scored by a weighted combination of churn probability, CLV, uplift score, and customer segment.

### API Reference

#### Constructor

```python
RecommendationEngine(config: Dict[str, Any])
```

**Scoring weights** (internal defaults):

| Weight | Value | Description |
|--------|-------|-------------|
| `w_churn` | 0.35 | Churn probability weight |
| `w_clv` | 0.25 | CLV weight |
| `w_uplift` | 0.20 | Uplift score weight |
| `w_segment` | 0.20 | Segment affinity weight |

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_available_actions` | `get_available_actions()` | `List[str]` | List all action types |
| `recommend` | `recommend(*, data)` | `pd.DataFrame` | Single best recommendation per customer |
| `recommend_top_k` | `recommend_top_k(*, data, k)` | `pd.DataFrame` | Top-k recommendations per customer |
| `save` / `load` | — | — | JSON persistence |

#### Input Data Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `customer_id` | `str` | Yes | Unique customer identifier |
| `churn_prob` | `float` | Yes | Churn probability (0–1) |
| `clv` | `float` | Yes | Customer lifetime value |
| `uplift_score` | `float` | Yes | Uplift effect estimate |
| `segment` | `str` | Yes | Customer segment label |
| `push_opt_in` | `bool` | No | Opted into push notifications |
| `email_opt_in` | `bool` | No | Opted into email communications |

#### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | `str` | Customer identifier |
| `action_type` | `str` | Recommended action |
| `score` | `float` | Composite recommendation score |
| `estimated_cost` | `float` | Cost of the recommended action (KRW) |
| `reason` | `str` | Human-readable explanation |

#### Built-in Action Catalogue

| Action | Base Cost (KRW) | Constraint | Best For |
|--------|----------------|------------|----------|
| `coupon` | 5,000 | — | High-churn customers |
| `push_notification` | 100 | Requires `push_opt_in` | Low-cost engagement |
| `email_campaign` | 200 | Requires `email_opt_in` | Broad outreach |
| `loyalty_points` | 3,000 | — | Reward loyalty |
| `personal_outreach` | 10,000 | — | High-value customers |
| `exclusive_offer` | 8,000 | — | Premium retention |

### Usage Example

```python
from src.models.recommendations import RecommendationEngine

engine = RecommendationEngine(config)

# Available actions
print(engine.get_available_actions())
# ['coupon', 'push_notification', 'email_campaign', 'loyalty_points', 'personal_outreach', 'exclusive_offer']

# Single best recommendation per customer
recs = engine.recommend(data=customer_df)
print(recs[["customer_id", "action_type", "score", "estimated_cost"]].head())

# Top-3 recommendations per customer
top3 = engine.recommend_top_k(data=customer_df, k=3)

# Persist
engine.save("models/recommendation_engine.json")
loaded = RecommendationEngine.load("models/recommendation_engine.json")
```

---

## 7. Cohort Analysis

**Module**: `src/analysis/cohort_analysis.py`
**Class**: `CohortAnalyzer`
**Purpose**: Assign customers to cohorts (monthly, weekly, or behavioral), compute retention matrices, aggregate metrics, and generate visualizations.

### API Reference

#### Constructor

```python
CohortAnalyzer(config: Optional[Dict[str, Any]] = None)
```

Default config values:

| Key | Default | Description |
|-----|---------|-------------|
| `cohort_type` | `"monthly"` | `"monthly"`, `"weekly"`, or `"behavioral"` |
| `periods` | `12` | Number of periods to track |
| `min_cohort_size` | `5` | Minimum customers per cohort |
| `behavioral_column` | `"segment"` | Column for behavioral cohorts |
| `date_column` | `"event_date"` | Timestamp column |
| `customer_column` | `"customer_id"` | Customer ID column |
| `revenue_column` | `"revenue"` | Revenue column |

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `assign_cohorts` | `assign_cohorts(data, cohort_type)` | `pd.DataFrame` | Add `cohort` and `cohort_period` columns |
| `compute_retention_matrix` | `compute_retention_matrix(cohort_data, max_periods)` | `pd.DataFrame` | Cohort × period retention rates |
| `compute_cohort_metrics` | `compute_cohort_metrics(cohort_data, metrics)` | `Dict[str, pd.DataFrame]` | Aggregate metrics per cohort |
| `get_cohort_summary` | `get_cohort_summary(cohort_data)` | `pd.DataFrame` | Per-cohort summary statistics |
| `filter_cohorts` | `filter_cohorts(cohort_data, cohorts, min_size)` | `pd.DataFrame` | Filter by name or minimum size |
| `get_retention_curves` | `get_retention_curves(retention_matrix)` | `Dict[str, np.ndarray]` | Per-cohort retention arrays |
| `get_average_retention_curve` | `get_average_retention_curve(retention_matrix)` | `np.ndarray` | Mean retention across cohorts |
| `compute_half_life` | `compute_half_life(retention_matrix)` | `Dict[str, float]` | Period when retention drops to 50% |
| `compute_churn_rates` | `compute_churn_rates(retention_matrix)` | `pd.DataFrame` | Period-over-period churn rates |
| `compute_cumulative_revenue` | `compute_cumulative_revenue(cohort_data)` | `pd.DataFrame` | Cumulative revenue over time |
| `plot_retention_heatmap` | `plot_retention_heatmap(retention_matrix, ...)` | `str` | Path to saved heatmap PNG |
| `plot_retention_lines` | `plot_retention_lines(retention_matrix, ...)` | `str` | Path to saved line plot PNG |
| `plot_cohort_sizes` | `plot_cohort_sizes(cohort_data, ...)` | `str` | Path to saved bar chart PNG |
| `full_analysis` | `full_analysis(data, cohort_type, ...)` | `dict` | Complete end-to-end analysis pipeline |

**`full_analysis` return structure:**

```python
{
    "cohorts": pd.DataFrame,           # Data with cohort assignments
    "retention_matrix": pd.DataFrame,  # Retention matrix
    "metrics": Dict[str, pd.DataFrame],# All computed metrics
    "summary": pd.DataFrame,           # Cohort summary
    "plots": Dict[str, str]            # Paths to generated plot files
}
```

### Usage Example

```python
from src.analysis.cohort_analysis import CohortAnalyzer

analyzer = CohortAnalyzer({
    "cohort_type": "monthly",
    "periods": 12,
    "min_cohort_size": 10
})

# Full analysis pipeline
results = analyzer.full_analysis(transaction_data)

# Or step by step:
cohort_data = analyzer.assign_cohorts(transaction_data, cohort_type="monthly")
retention = analyzer.compute_retention_matrix(cohort_data)
metrics = analyzer.compute_cohort_metrics(cohort_data)
summary = analyzer.get_cohort_summary(cohort_data)

# Derived analytics
half_lives = analyzer.compute_half_life(retention)
churn_rates = analyzer.compute_churn_rates(retention)
avg_retention = analyzer.get_average_retention_curve(retention)

# Visualization
heatmap_path = analyzer.plot_retention_heatmap(retention, output_dir="results/")
lines_path = analyzer.plot_retention_lines(retention, output_dir="results/")
sizes_path = analyzer.plot_cohort_sizes(cohort_data, output_dir="results/")
```

---

## 8. Configuration Reference

All modules read from `config/simulator_config.yaml`. Below is the complete configuration schema for the new modules:

```yaml
# === Core Settings ===
simulation:
  random_seed: 42              # Global random seed for reproducibility
  num_customers: 20000         # Number of simulated customers
  simulation_months: 12        # Simulation time horizon

# === Treatment / A/B Testing ===
treatment:
  treatment_ratio: 0.50        # Fraction assigned to treatment
  min_group_size: 10000        # Minimum per-group sample size

# === Uplift Model ===
uplift:
  n_estimators: 100            # GBM boosting rounds
  max_depth: 4                 # Max tree depth
  learning_rate: 0.1           # GBM learning rate
  segment_thresholds:
    high_uplift_quantile: 0.75 # Top quantile → "persuadable"
    low_uplift_quantile: 0.25  # Bottom quantile → "sleeping_dog"

# === Survival Analysis ===
survival:
  penalizer: 0.01              # L2 regularization
  l1_ratio: 0.0                # Elastic net mixing parameter
  alpha: 0.05                  # CI significance level

# === Budget / Optimization ===
budget:
  total_krw: 50000000          # Total budget (KRW)
  currency: "KRW"

optimization:
  total_budget: 50000000
  discount_rate: 0.10          # Annual discount rate for NPV
  time_horizon_months: 12
  channels:
    email:
      cost_per_action: 1000
      expected_roi_multiplier: 1.0
      min_budget: 0
      max_budget: null
    sms:
      cost_per_action: 500
      expected_roi_multiplier: 0.8
    push_notification:
      cost_per_action: 100
      expected_roi_multiplier: 0.6
    coupon:
      cost_per_action: 5000
      expected_roi_multiplier: 1.5
    call_center:
      cost_per_action: 10000
      expected_roi_multiplier: 2.0
```

---

## Module Integration Map

The modules work together in the following pipeline:

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Churn Predict   │────▶│ Uplift Model │────▶│  Recommendations │
│  (existing)      │     │              │     │                  │
└────────┬────────┘     └──────┬───────┘     └────────┬─────────┘
         │                     │                       │
         ▼                     ▼                       ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  CLV Model       │────▶│ Budget Opt.  │◀────│  Cost Config     │
│                  │     │  (LP Solver) │     │                  │
└────────┬────────┘     └──────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Survival Model   │     │ A/B Testing  │     │ Cohort Analysis  │
│ (Cox PH + KM)   │     │ (Sequential) │     │ (Retention)      │
└─────────────────┘     └──────────────┘     └──────────────────┘
```

### Typical End-to-End Workflow

```python
import yaml

with open("config/simulator_config.yaml") as f:
    config = yaml.safe_load(f)

# 1. Train churn model (existing)
# 2. Train uplift model
from src.models.uplift_model import UpliftModel
uplift = UpliftModel(config, learner="t_learner")
uplift.fit(X, treatment, y)

# 3. Train CLV model
from src.models.clv_model import CLVModel
clv = CLVModel(config)
clv.fit(X_clv, y_clv)

# 4. Train survival model
from src.models.survival_analysis import SurvivalModel
survival = SurvivalModel(config)
survival.fit(X, duration, event)

# 5. Generate recommendations
from src.models.recommendations import RecommendationEngine
engine = RecommendationEngine(config)
recs = engine.recommend(data=customer_df)

# 6. Optimize budget allocation
from src.optimization.budget_optimizer import LPBudgetOptimizer
optimizer = LPBudgetOptimizer(config)
result = optimizer.solve(customer_df)

# 7. Run A/B test on recommendations
from src.models.ab_testing import ABTestFramework
ab = ABTestFramework(config)
groups = ab.assign_groups(customer_ids)

# 8. Analyze cohorts
from src.analysis.cohort_analysis import CohortAnalyzer
cohorts = CohortAnalyzer(config)
analysis = cohorts.full_analysis(transaction_data)
```

### CLI Integration

All modules are accessible via the CLI entrypoint:

```bash
# Train uplift model
python -m src --mode uplift

# Run budget optimization
python -m src --mode optimize --budget 50000000

# Run survival analysis
python -m src --mode survival

# Generate recommendations
python -m src --mode recommend

# Run A/B test analysis
python -m src --mode ab_test

# Run cohort analysis
python -m src --mode cohort

# Launch dashboard
python -m src --mode dashboard
```
