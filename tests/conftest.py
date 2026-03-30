"""
Shared test configuration, fixtures, and mock data for the
e-commerce churn prediction system test suite.

Provides:
- Python path setup (PROJECT_ROOT on sys.path)
- config fixture (loads config/simulator_config.yaml)
- Common sample data generators for uplift, CLV, A/B testing,
  survival analysis, recommendations, cohort analysis, and budget
  optimization tests
- Shared utility fixtures (project_root, config_path, tmp model dirs)
- Mock Streamlit module for dashboard tests
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Path setup — ensures `import src.*` works regardless of working directory
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_path() -> Path:
    """Return the path to the simulator config YAML."""
    return CONFIG_PATH


@pytest.fixture
def config() -> dict:
    """Load the full simulator configuration from YAML.

    This is the most widely used fixture — shared by uplift, CLV,
    A/B testing, survival, recommendation, budget, dashboard,
    and CLI tests.
    """
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Sample data fixtures — uplift
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_uplift_data() -> pd.DataFrame:
    """Synthetic data with treatment/control labels for uplift tests.

    Contains 2 000 customers with 20 features, heterogeneous treatment
    effects (persuadables, sleeping dogs, sure things, lost causes),
    binary churn label, and ground-truth uplift for validation.
    """
    np.random.seed(42)
    n = 2000
    n_features = 20

    X = np.random.randn(n, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    df["customer_id"] = [f"C{i:05d}" for i in range(n)]

    # Assign treatment/control (50/50 split)
    df["treatment_group"] = np.random.choice(
        ["treatment", "control"], size=n, p=[0.5, 0.5]
    )
    df["is_treatment"] = (df["treatment_group"] == "treatment").astype(int)

    # Generate heterogeneous treatment effects
    base_churn_prob = 1 / (1 + np.exp(-(0.5 * X[:, 2] - 0.3 * X[:, 3])))
    treatment_effect = np.where(
        X[:, 0] > 0,
        -0.2,  # Persuadables: treatment reduces churn
        np.where(X[:, 1] > 1, 0.15, 0.0),  # Sleeping dogs / no effect
    )
    churn_prob = base_churn_prob + df["is_treatment"].values * treatment_effect
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    df["churn_label"] = (np.random.rand(n) < churn_prob).astype(int)
    df["true_uplift"] = -treatment_effect  # Positive = treatment helps

    return df


# ---------------------------------------------------------------------------
# Sample data fixtures — CLV
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_clv_data() -> pd.DataFrame:
    """Synthetic customer data with RFM features for CLV prediction tests.

    Contains 1 500 customers with purchase-history features that correlate
    with lifetime value, plus a churn_prob column for integration testing.
    """
    np.random.seed(42)
    n = 1500

    recency = np.random.exponential(30, n)
    frequency = np.random.poisson(5, n).astype(float)
    monetary = np.random.lognormal(10, 1, n)
    tenure_days = np.random.uniform(30, 365, n)
    avg_order_value = monetary / np.maximum(frequency, 1)
    purchase_cycle_days = np.random.exponential(14, n)
    coupon_usage_rate = np.random.beta(2, 5, n)
    review_rate = np.random.beta(1, 10, n)
    visit_frequency = np.random.poisson(10, n).astype(float)
    cart_conversion = np.random.beta(3, 7, n)
    session_duration = np.random.exponential(15, n)
    search_count = np.random.poisson(8, n).astype(float)
    cs_contact_count = np.random.poisson(1, n).astype(float)
    weekend_ratio = np.random.beta(2, 3, n)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "tenure_days": tenure_days,
        "avg_order_value": avg_order_value,
        "purchase_cycle_days": purchase_cycle_days,
        "coupon_usage_rate": coupon_usage_rate,
        "review_rate": review_rate,
        "visit_frequency": visit_frequency,
        "cart_conversion": cart_conversion,
        "session_duration": session_duration,
        "search_count": search_count,
        "cs_contact_count": cs_contact_count,
        "weekend_ratio": weekend_ratio,
    })

    df["clv_target"] = (
        0.4 * frequency * avg_order_value
        + 0.3 * (365 / np.maximum(purchase_cycle_days, 1)) * avg_order_value
        - 0.1 * recency * 1000
        + np.random.randn(n) * 50000
    ).clip(0)

    df["churn_prob"] = 1 / (1 + np.exp(
        -(0.02 * recency - 0.1 * frequency + np.random.randn(n) * 0.5)
    ))

    return df


# ---------------------------------------------------------------------------
# Sample data fixtures — A/B testing
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_experiment_data() -> pd.DataFrame:
    """Synthetic A/B experiment data with treatment/control outcomes.

    Contains 4 000 customers split into treatment and control groups
    with churn, revenue, and conversion outcome metrics.
    """
    np.random.seed(42)
    n = 4000

    group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
    np.random.shuffle(group)

    base_churn = np.random.beta(2, 5, n)
    treatment_effect = np.where(group == "treatment", -0.05, 0.0)
    actual_churn = np.clip(base_churn + treatment_effect, 0, 1)
    churned = np.random.binomial(1, actual_churn)

    revenue = np.where(
        churned == 0,
        np.random.lognormal(10, 1, n),
        np.random.lognormal(8, 1, n) * 0.1,
    )
    converted = np.random.binomial(
        1, np.where(group == "treatment", 0.15, 0.10)
    )

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "group": group,
        "churned": churned,
        "revenue": revenue,
        "converted": converted,
        "churn_prob": base_churn,
        "days_active": np.random.poisson(30, n),
    })


# ---------------------------------------------------------------------------
# Sample data fixtures — survival analysis
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_survival_data() -> pd.DataFrame:
    """Synthetic survival / time-to-event data for survival analysis tests.

    Contains 2 000 customers with Weibull-distributed durations,
    right-censoring (~30 %), and 8 covariates.
    """
    np.random.seed(42)
    n = 2000

    recency = np.random.exponential(30, n)
    frequency = np.random.poisson(5, n).astype(float)
    monetary = np.random.lognormal(10, 1, n)
    tenure_days = np.random.uniform(30, 365, n)
    visit_frequency = np.random.poisson(10, n).astype(float)
    coupon_usage_rate = np.random.beta(2, 5, n)
    session_duration = np.random.exponential(15, n)
    cs_contact_count = np.random.poisson(1, n).astype(float)

    scale = np.exp(
        3.5
        + 0.1 * frequency
        - 0.02 * recency
        + 0.3 * coupon_usage_rate
        + np.random.randn(n) * 0.3
    )
    shape = 1.5
    duration = np.random.weibull(shape, n) * scale
    duration = np.clip(duration, 1, 365).astype(float)

    censoring_time = np.random.uniform(60, 365, n)
    event = (duration <= censoring_time).astype(int)
    observed_duration = np.minimum(duration, censoring_time)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "duration": observed_duration,
        "event": event,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "tenure_days": tenure_days,
        "visit_frequency": visit_frequency,
        "coupon_usage_rate": coupon_usage_rate,
        "session_duration": session_duration,
        "cs_contact_count": cs_contact_count,
    })


@pytest.fixture
def feature_cols() -> list:
    """Standard feature column names for survival analysis models."""
    return [
        "recency", "frequency", "monetary", "tenure_days",
        "visit_frequency", "coupon_usage_rate", "session_duration",
        "cs_contact_count",
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures — recommendations
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_customer_data() -> pd.DataFrame:
    """Synthetic customer data for recommendation engine tests.

    Includes upstream model predictions (churn_prob, clv, uplift_score),
    behavioral features, channel opt-in flags, and segment labels.
    """
    np.random.seed(42)
    n = 1500

    churn_prob = np.random.beta(2, 5, n)
    clv = np.random.lognormal(10, 1, n)
    uplift_score = np.random.randn(n) * 0.1

    recency = np.random.exponential(30, n)
    frequency = np.random.poisson(5, n).astype(float)
    monetary = np.random.lognormal(10, 1, n)
    coupon_usage_rate = np.random.beta(2, 5, n)
    visit_frequency = np.random.poisson(10, n).astype(float)
    session_duration = np.random.exponential(15, n)

    preferred_category = np.random.choice(
        ["electronics", "fashion", "food", "beauty", "home"], size=n,
    )
    avg_discount_used = np.random.beta(3, 7, n) * 50
    push_opt_in = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
    email_opt_in = np.random.choice([0, 1], size=n, p=[0.1, 0.9])
    segments = np.random.choice(
        ["vip_loyal", "regular_loyal", "bargain_hunter",
         "explorer", "dormant", "new_customer"],
        size=n,
    )

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_prob": churn_prob,
        "clv": clv,
        "uplift_score": uplift_score,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "coupon_usage_rate": coupon_usage_rate,
        "visit_frequency": visit_frequency,
        "session_duration": session_duration,
        "preferred_category": preferred_category,
        "avg_discount_used": avg_discount_used,
        "push_opt_in": push_opt_in,
        "email_opt_in": email_opt_in,
        "segment": segments,
    })


# ---------------------------------------------------------------------------
# Sample data fixtures — budget optimization
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_optimization_data() -> pd.DataFrame:
    """Synthetic customer data with churn, CLV, and uplift scores
    for budget optimization tests.
    """
    np.random.seed(42)
    n = 1500

    churn_prob = np.random.beta(2, 5, n)
    clv = np.random.lognormal(10, 1, n)
    uplift_score = np.random.randn(n) * 0.1
    cost_per_action = np.random.choice([5000, 10000, 20000], size=n)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_prob": churn_prob,
        "clv": clv,
        "uplift_score": uplift_score,
        "cost_per_action": cost_per_action.astype(float),
    })


# ---------------------------------------------------------------------------
# Sample data fixtures — cohort analysis
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_events() -> pd.DataFrame:
    """Sample event data spanning multiple months for cohort analysis tests."""
    np.random.seed(42)
    data = []
    # 3 cohorts: Jan, Feb, Mar 2024
    for cid in range(1, 11):
        for month in range(1, 4):
            n_events = np.random.randint(1, 4)
            for _ in range(n_events):
                day = np.random.randint(1, 28)
                data.append({
                    "customer_id": f"C{cid:03d}",
                    "event_date": pd.Timestamp(2024, month, day),
                    "revenue": np.random.uniform(10000, 100000),
                    "segment": "loyal" if cid <= 5 else "at_risk",
                })

    # Feb cohort: 8 customers, active for 2 months
    for cid in range(11, 19):
        for month in range(2, 4):
            n_events = np.random.randint(1, 3)
            for _ in range(n_events):
                day = np.random.randint(1, 28)
                data.append({
                    "customer_id": f"C{cid:03d}",
                    "event_date": pd.Timestamp(2024, month, day),
                    "revenue": np.random.uniform(10000, 80000),
                    "segment": "new",
                })

    # Mar cohort: 6 customers, active for 1 month only
    for cid in range(19, 25):
        n_events = np.random.randint(1, 3)
        for _ in range(n_events):
            day = np.random.randint(1, 28)
            data.append({
                "customer_id": f"C{cid:03d}",
                "event_date": pd.Timestamp(2024, 3, day),
                "revenue": np.random.uniform(10000, 60000),
                "segment": "new",
            })

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Model instance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uplift_model(config):
    """Create an UpliftModel instance from config."""
    from src.models.uplift_model import UpliftModel
    return UpliftModel(config)


@pytest.fixture
def clv_model(config):
    """Create a CLVModel instance from config."""
    from src.models.clv_model import CLVModel
    return CLVModel(config)


@pytest.fixture
def ab_test_framework(config):
    """Create an ABTestFramework instance from config."""
    from src.models.ab_testing import ABTestFramework
    return ABTestFramework(config)


@pytest.fixture
def survival_model(config):
    """Create a SurvivalModel instance from config."""
    from src.models.survival_analysis import SurvivalModel
    return SurvivalModel(config)


@pytest.fixture
def recommendation_engine(config):
    """Create a RecommendationEngine instance from config."""
    from src.models.recommendations import RecommendationEngine
    return RecommendationEngine(config)


# ---------------------------------------------------------------------------
# Mock Streamlit fixture for dashboard tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_st():
    """Create a mock Streamlit module with common methods.

    Useful for dashboard tests that need to call st.write(),
    st.plotly_chart(), st.sidebar, etc. without a real browser.
    """
    st = MagicMock()
    st.sidebar = MagicMock()
    st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    st.expander = MagicMock()
    st.container = MagicMock()
    st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    st.session_state = {}
    return st


# ---------------------------------------------------------------------------
# Docker / infrastructure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def docker_compose_path() -> Path:
    """Return the docker-compose.yml path."""
    return PROJECT_ROOT / "docker-compose.yml"


@pytest.fixture
def docker_compose(docker_compose_path) -> dict:
    """Load and parse docker-compose.yml."""
    with open(docker_compose_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_output_dir(tmp_path) -> Path:
    """Provide a temporary directory for model artifacts."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def results_dir(tmp_path) -> Path:
    """Provide a temporary directory for results / reports."""
    d = tmp_path / "results"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Sample data fixtures — dashboard display data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_churn_predictions() -> pd.DataFrame:
    """Sample churn prediction data for dashboard display tests.

    Contains 500 customers with churn probability, risk level, segment,
    recommended action, CLV, and recency metrics.
    """
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_probability": np.random.beta(2, 5, n),
        "risk_level": np.random.choice(
            ["low", "medium", "high", "critical"],
            n,
            p=[0.4, 0.3, 0.2, 0.1],
        ),
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "explorer", "dormant", "new_customer"],
            n,
        ),
        "recommended_action": np.random.choice(
            ["coupon", "push_notification", "email", "no_action"],
            n,
        ),
        "clv_predicted": np.random.lognormal(11, 1, n),
        "days_since_last_purchase": np.random.exponential(15, n),
        "days_since_last_login": np.random.exponential(8, n),
    })


@pytest.fixture
def sample_model_metrics() -> dict:
    """Sample model performance metrics for dashboard display tests."""
    return {
        "ml_model": {
            "auc": 0.82, "precision": 0.76, "recall": 0.70,
            "f1_score": 0.73, "accuracy": 0.81,
        },
        "dl_model": {
            "auc": 0.79, "precision": 0.72, "recall": 0.67,
            "f1_score": 0.69, "accuracy": 0.78,
        },
        "ensemble": {
            "auc": 0.84, "precision": 0.78, "recall": 0.72,
            "f1_score": 0.75, "accuracy": 0.83,
        },
    }


@pytest.fixture
def sample_ab_test_results() -> dict:
    """Sample A/B test results for dashboard display tests."""
    return {
        "experiment_name": "retention_coupon_campaign",
        "treatment_size": 500,
        "control_size": 500,
        "treatment_churn_rate": 0.12,
        "control_churn_rate": 0.20,
        "lift": 0.40,
        "p_value": 0.003,
        "is_significant": True,
        "confidence_interval": (0.03, 0.13),
    }


@pytest.fixture
def sample_budget_results() -> pd.DataFrame:
    """Sample budget optimization results for dashboard display tests."""
    return pd.DataFrame({
        "segment": ["vip_loyal", "regular_loyal", "bargain_hunter",
                     "explorer", "dormant", "new_customer"],
        "allocated_budget_krw": [5_000_000, 12_000_000, 8_000_000,
                                  10_000_000, 3_000_000, 12_000_000],
        "expected_retained_customers": [50, 120, 80, 100, 30, 60],
        "expected_roi": [2.5, 1.8, 1.5, 2.0, 0.8, 3.0],
    })


# ---------------------------------------------------------------------------
# Dashboard data loader fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dashboard_data_loader(config):
    """Create a DashboardDataLoader instance for dashboard tests."""
    try:
        from src.dashboard.data_loader import DashboardDataLoader
        return DashboardDataLoader(config)
    except ImportError:
        return MagicMock()


# ---------------------------------------------------------------------------
# Docker file content fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dockerfile_mlflow() -> str:
    """Read Dockerfile.mlflow contents."""
    path = PROJECT_ROOT / "Dockerfile.mlflow"
    return path.read_text() if path.exists() else ""


@pytest.fixture
def dockerfile_pipeline() -> str:
    """Read Dockerfile.pipeline contents."""
    path = PROJECT_ROOT / "Dockerfile.pipeline"
    return path.read_text() if path.exists() else ""


@pytest.fixture
def dockerfile_dashboard() -> str:
    """Read Dockerfile.dashboard contents."""
    path = PROJECT_ROOT / "Dockerfile.dashboard"
    return path.read_text() if path.exists() else ""


# ---------------------------------------------------------------------------
# CLI fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> dict:
    """Minimal valid config dict for CLI / pipeline tests."""
    return {
        "simulation": {
            "random_seed": 42,
            "num_customers": 100,
            "simulation_months": 6,
            "simulation_days": 180,
            "start_date": "2024-01-01",
            "small_mode": {
                "num_customers": 50,
                "simulation_months": 3,
                "simulation_days": 90,
            },
        },
        "churn_definition": {
            "no_purchase_days": 30,
            "no_login_days": 60,
            "operator": "OR",
        },
        "pipeline": {
            "train_months": 4,
            "test_months": 2,
            "ensemble_weight_ml": 0.6,
            "ensemble_weight_dl": 0.4,
        },
        "segmentation": {"method": "rfm_behavioral", "n_rfm_bins": 5},
        "optimization": {
            "total_budget": 50_000_000,
            "channels": {"email": {"cost_per_action": 1000}},
        },
        "monitoring": {"enabled": True},
        "treatment": {"treatment_ratio": 0.5},
        "budget": {"total_krw": 50_000_000, "currency": "KRW"},
    }


# ---------------------------------------------------------------------------
# Cohort analyzer fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cohort_analyzer():
    """Create a CohortAnalyzer instance with default config."""
    try:
        from src.analysis.cohort_analysis import CohortAnalyzer
        return CohortAnalyzer()
    except ImportError:
        return MagicMock()
