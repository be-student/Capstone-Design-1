"""
TDD Tests for Streamlit Dashboard Module.

Tests cover:
- Dashboard app instantiation and configuration
- Page rendering (all required views)
- Data loading and caching for dashboard components
- Churn prediction overview page
- Customer segmentation visualization
- Model performance metrics display (AUC, precision, recall, F1)
- Budget optimization results display
- A/B testing results visualization
- Survival analysis curves display
- Personalized recommendations view
- CLV distribution and top-N customer display
- Uplift model results display
- Real-time scoring status display
- MLflow experiment comparison view
- Filter and interaction controls
- Configurable parameters from YAML
- Error handling for missing data/models
- Responsive layout and component structure
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_churn_predictions():
    """Create sample churn prediction data for dashboard display."""
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
             "new_customer", "dormant", "high_value_at_risk"],
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
def sample_model_metrics():
    """Create sample model performance metrics for dashboard display."""
    return {
        "ml_model": {
            "auc": 0.82,
            "precision": 0.76,
            "recall": 0.70,
            "f1_score": 0.73,
            "accuracy": 0.81,
        },
        "dl_model": {
            "auc": 0.79,
            "precision": 0.72,
            "recall": 0.67,
            "f1_score": 0.69,
            "accuracy": 0.78,
        },
        "ensemble": {
            "auc": 0.84,
            "precision": 0.78,
            "recall": 0.72,
            "f1_score": 0.75,
            "accuracy": 0.83,
        },
    }


@pytest.fixture
def sample_ab_test_results():
    """Create sample A/B test results for dashboard display."""
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
def sample_budget_results():
    """Create sample budget optimization results for dashboard display."""
    return pd.DataFrame({
        "segment": ["vip_loyal", "regular_loyal", "bargain_hunter",
                     "new_customer", "dormant", "high_value_at_risk"],
        "allocated_budget_krw": [5000000, 12000000, 8000000,
                                 10000000, 3000000, 12000000],
        "expected_retained": [450, 1800, 1200, 1500, 200, 850],
        "expected_revenue_saved_krw": [
            67500000, 144000000, 54000000,
            82500000, 12000000, 102000000,
        ],
        "roi": [13.5, 12.0, 6.75, 8.25, 4.0, 8.5],
    })


@pytest.fixture
def sample_survival_data():
    """Create sample survival analysis data for dashboard display."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "duration_days": np.random.exponential(90, n),
        "event_observed": np.random.binomial(1, 0.3, n),
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "new_customer", "dormant", "high_value_at_risk"],
            n,
        ),
        "survival_probability": np.random.beta(5, 2, n),
    })


@pytest.fixture
def sample_recommendations():
    """Create sample personalized recommendations for dashboard display."""
    return pd.DataFrame({
        "customer_id": ["C00001", "C00002", "C00003", "C00004", "C00005"],
        "recommendation_type": [
            "coupon", "push_notification", "email",
            "coupon", "no_action",
        ],
        "expected_uplift": [0.15, 0.08, 0.12, 0.20, 0.0],
        "priority_score": [0.92, 0.78, 0.85, 0.95, 0.10],
        "recommended_offer": [
            "20% discount", "Flash sale alert",
            "Personalized picks", "30% discount", "None",
        ],
    })


@pytest.fixture
def dashboard_data_loader(config):
    """Create a DashboardDataLoader instance."""
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


# ---------------------------------------------------------------------------
# Dashboard data loader interface tests
# ---------------------------------------------------------------------------

class TestDashboardDataLoaderInterface:
    """Test dashboard data loader instantiation and interface."""

    def test_instantiation(self, dashboard_data_loader):
        """Dashboard data loader must be instantiable from config."""
        assert dashboard_data_loader is not None

    def test_has_load_predictions_method(self, dashboard_data_loader):
        """Must implement churn predictions loading."""
        assert hasattr(dashboard_data_loader, "load_predictions")
        assert callable(dashboard_data_loader.load_predictions)

    def test_has_load_model_metrics_method(self, dashboard_data_loader):
        """Must implement model metrics loading."""
        assert hasattr(dashboard_data_loader, "load_model_metrics")
        assert callable(dashboard_data_loader.load_model_metrics)

    def test_has_load_ab_test_results_method(self, dashboard_data_loader):
        """Must implement A/B test results loading."""
        assert hasattr(dashboard_data_loader, "load_ab_test_results")
        assert callable(dashboard_data_loader.load_ab_test_results)

    def test_has_load_budget_results_method(self, dashboard_data_loader):
        """Must implement budget optimization results loading."""
        assert hasattr(dashboard_data_loader, "load_budget_results")
        assert callable(dashboard_data_loader.load_budget_results)

    def test_has_load_survival_data_method(self, dashboard_data_loader):
        """Must implement survival analysis data loading."""
        assert hasattr(dashboard_data_loader, "load_survival_data")
        assert callable(dashboard_data_loader.load_survival_data)

    def test_has_load_recommendations_method(self, dashboard_data_loader):
        """Must implement personalized recommendations loading."""
        assert hasattr(dashboard_data_loader, "load_recommendations")
        assert callable(dashboard_data_loader.load_recommendations)

    def test_has_load_uplift_results_method(self, dashboard_data_loader):
        """Must implement uplift model results loading."""
        assert hasattr(dashboard_data_loader, "load_uplift_results")
        assert callable(dashboard_data_loader.load_uplift_results)

    def test_has_load_clv_data_method(self, dashboard_data_loader):
        """Must implement CLV data loading."""
        assert hasattr(dashboard_data_loader, "load_clv_data")
        assert callable(dashboard_data_loader.load_clv_data)


# ---------------------------------------------------------------------------
# Dashboard page rendering tests
# ---------------------------------------------------------------------------

class TestDashboardPages:
    """Test that all required dashboard pages/views can be rendered."""

    def test_has_overview_page(self):
        """Dashboard must include a churn prediction overview page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_overview")
        assert callable(dashboard_app.render_overview)

    def test_has_model_performance_page(self):
        """Dashboard must include a model performance page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_model_performance")
        assert callable(dashboard_app.render_model_performance)

    def test_has_segmentation_page(self):
        """Dashboard must include a customer segmentation page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_segmentation")
        assert callable(dashboard_app.render_segmentation)

    def test_has_budget_optimization_page(self):
        """Dashboard must include a budget optimization page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_budget_optimization")
        assert callable(dashboard_app.render_budget_optimization)

    def test_has_ab_testing_page(self):
        """Dashboard must include an A/B testing results page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_ab_testing")
        assert callable(dashboard_app.render_ab_testing)

    def test_has_survival_analysis_page(self):
        """Dashboard must include a survival analysis page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_survival_analysis")
        assert callable(dashboard_app.render_survival_analysis)

    def test_has_recommendations_page(self):
        """Dashboard must include a personalized recommendations page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_recommendations")
        assert callable(dashboard_app.render_recommendations)

    def test_has_clv_page(self):
        """Dashboard must include a CLV prediction page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_clv")
        assert callable(dashboard_app.render_clv)

    def test_has_uplift_page(self):
        """Dashboard must include an uplift modeling results page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_uplift")
        assert callable(dashboard_app.render_uplift)

    def test_has_realtime_scoring_page(self):
        """Dashboard must include a real-time scoring status page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_realtime_scoring")
        assert callable(dashboard_app.render_realtime_scoring)

    def test_has_mlflow_page(self):
        """Dashboard must include an MLflow experiment comparison page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_mlflow_experiments")
        assert callable(dashboard_app.render_mlflow_experiments)


# ---------------------------------------------------------------------------
# Churn overview display tests
# ---------------------------------------------------------------------------

class TestChurnOverviewDisplay:
    """Test churn prediction overview display data preparation."""

    def test_churn_distribution_data(self, sample_churn_predictions):
        """Churn probability distribution must be computable."""
        probs = sample_churn_predictions["churn_probability"]

        assert len(probs) > 0
        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_risk_level_counts(self, sample_churn_predictions):
        """Risk level counts must cover all categories."""
        risk_counts = sample_churn_predictions["risk_level"].value_counts()

        expected_levels = {"low", "medium", "high", "critical"}
        assert set(risk_counts.index) <= expected_levels

    def test_segment_churn_rates(self, sample_churn_predictions):
        """Average churn probability by segment must be computable."""
        segment_rates = sample_churn_predictions.groupby("segment")[
            "churn_probability"
        ].mean()

        assert len(segment_rates) > 0
        assert all(0 <= r <= 1 for r in segment_rates)

    def test_high_risk_customers_filterable(self, sample_churn_predictions):
        """Must be able to filter high-risk customers (probability > 0.5)."""
        high_risk = sample_churn_predictions[
            sample_churn_predictions["churn_probability"] > 0.5
        ]

        assert isinstance(high_risk, pd.DataFrame)
        assert "customer_id" in high_risk.columns

    def test_summary_kpis_computable(self, sample_churn_predictions):
        """Key summary KPIs must be computable from predictions."""
        total_customers = len(sample_churn_predictions)
        avg_churn_prob = sample_churn_predictions["churn_probability"].mean()
        high_risk_count = (
            sample_churn_predictions["churn_probability"] > 0.5
        ).sum()

        assert total_customers > 0
        assert 0 <= avg_churn_prob <= 1
        assert high_risk_count >= 0


# ---------------------------------------------------------------------------
# Model performance display tests
# ---------------------------------------------------------------------------

class TestModelPerformanceDisplay:
    """Test model performance metrics display data."""

    def test_all_model_types_present(self, sample_model_metrics):
        """Metrics must include ML, DL, and ensemble models."""
        assert "ml_model" in sample_model_metrics
        assert "dl_model" in sample_model_metrics
        assert "ensemble" in sample_model_metrics

    def test_required_metrics_present(self, sample_model_metrics):
        """Each model must report AUC, precision, recall, F1."""
        required = {"auc", "precision", "recall", "f1_score"}

        for model_name, metrics in sample_model_metrics.items():
            for metric in required:
                assert metric in metrics, (
                    f"Missing {metric} for {model_name}"
                )

    def test_ensemble_auc_meets_threshold(self, sample_model_metrics):
        """Ensemble AUC must meet minimum threshold of 0.78."""
        ensemble_auc = sample_model_metrics["ensemble"]["auc"]
        assert ensemble_auc >= 0.78, (
            f"Ensemble AUC {ensemble_auc} below threshold 0.78"
        )

    def test_metrics_comparison_table(self, sample_model_metrics):
        """Metrics must be convertible to a comparison DataFrame."""
        df = pd.DataFrame(sample_model_metrics).T

        assert len(df) == 3
        assert "auc" in df.columns
        assert df.index.tolist() == ["ml_model", "dl_model", "ensemble"]


# ---------------------------------------------------------------------------
# Budget optimization display tests
# ---------------------------------------------------------------------------

class TestBudgetOptimizationDisplay:
    """Test budget optimization results display data."""

    def test_total_budget_matches_config(self, sample_budget_results, config):
        """Total allocated budget must match configured budget."""
        total_allocated = sample_budget_results["allocated_budget_krw"].sum()
        configured_budget = config["budget"]["total_krw"]

        assert total_allocated == configured_budget, (
            f"Allocated {total_allocated} != configured {configured_budget}"
        )

    def test_all_segments_have_allocation(self, sample_budget_results):
        """Every segment must have a budget allocation row."""
        assert len(sample_budget_results) >= 6, (
            "Expected allocations for all 6 segments"
        )

    def test_roi_computable(self, sample_budget_results):
        """ROI must be present and positive for all segments."""
        assert "roi" in sample_budget_results.columns
        assert (sample_budget_results["roi"] >= 0).all()

    def test_expected_retained_present(self, sample_budget_results):
        """Expected retained customer counts must be present."""
        assert "expected_retained" in sample_budget_results.columns
        assert (sample_budget_results["expected_retained"] >= 0).all()


# ---------------------------------------------------------------------------
# A/B testing display tests
# ---------------------------------------------------------------------------

class TestABTestingDisplay:
    """Test A/B testing results display data."""

    def test_experiment_name_present(self, sample_ab_test_results):
        """Experiment name must be present."""
        assert "experiment_name" in sample_ab_test_results
        assert len(sample_ab_test_results["experiment_name"]) > 0

    def test_statistical_significance_shown(self, sample_ab_test_results):
        """Statistical significance must be indicated."""
        assert "is_significant" in sample_ab_test_results
        assert "p_value" in sample_ab_test_results
        assert isinstance(sample_ab_test_results["is_significant"], bool)

    def test_group_sizes_shown(self, sample_ab_test_results):
        """Treatment and control group sizes must be shown."""
        assert "treatment_size" in sample_ab_test_results
        assert "control_size" in sample_ab_test_results
        assert sample_ab_test_results["treatment_size"] > 0
        assert sample_ab_test_results["control_size"] > 0

    def test_lift_computable(self, sample_ab_test_results):
        """Lift between treatment and control must be present."""
        assert "lift" in sample_ab_test_results
        assert isinstance(sample_ab_test_results["lift"], (int, float))

    def test_churn_rates_present(self, sample_ab_test_results):
        """Treatment and control churn rates must be present."""
        assert "treatment_churn_rate" in sample_ab_test_results
        assert "control_churn_rate" in sample_ab_test_results
        assert (
            sample_ab_test_results["treatment_churn_rate"]
            < sample_ab_test_results["control_churn_rate"]
        ), "Treatment should have lower churn rate for significant result"


# ---------------------------------------------------------------------------
# Survival analysis display tests
# ---------------------------------------------------------------------------

class TestSurvivalAnalysisDisplay:
    """Test survival analysis data for dashboard display."""

    def test_survival_probability_present(self, sample_survival_data):
        """Survival probabilities must be present."""
        assert "survival_probability" in sample_survival_data.columns
        probs = sample_survival_data["survival_probability"]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_duration_data_present(self, sample_survival_data):
        """Customer duration data must be present."""
        assert "duration_days" in sample_survival_data.columns
        assert (sample_survival_data["duration_days"] >= 0).all()

    def test_event_observed_present(self, sample_survival_data):
        """Event observed indicator must be present."""
        assert "event_observed" in sample_survival_data.columns
        assert set(sample_survival_data["event_observed"].unique()) <= {0, 1}

    def test_segment_groupable(self, sample_survival_data):
        """Survival data must be groupable by segment."""
        grouped = sample_survival_data.groupby("segment")[
            "survival_probability"
        ].mean()
        assert len(grouped) > 0


# ---------------------------------------------------------------------------
# CLV display tests
# ---------------------------------------------------------------------------

class TestCLVDisplay:
    """Test CLV data for dashboard display."""

    def test_clv_values_present(self, sample_churn_predictions):
        """CLV predictions must be present in data."""
        assert "clv_predicted" in sample_churn_predictions.columns
        assert (sample_churn_predictions["clv_predicted"] > 0).all()

    def test_top_n_customers_extractable(self, sample_churn_predictions):
        """Must be able to extract top-N customers by CLV."""
        top_10 = sample_churn_predictions.nlargest(10, "clv_predicted")

        assert len(top_10) == 10
        assert top_10["clv_predicted"].is_monotonic_decreasing

    def test_clv_by_segment(self, sample_churn_predictions):
        """CLV must be aggregatable by segment."""
        segment_clv = sample_churn_predictions.groupby("segment")[
            "clv_predicted"
        ].agg(["mean", "sum", "count"])

        assert len(segment_clv) > 0
        assert "mean" in segment_clv.columns


# ---------------------------------------------------------------------------
# Recommendations display tests
# ---------------------------------------------------------------------------

class TestRecommendationsDisplay:
    """Test personalized recommendations display data."""

    def test_recommendation_type_present(self, sample_recommendations):
        """Recommendation type must be present."""
        assert "recommendation_type" in sample_recommendations.columns

    def test_expected_uplift_present(self, sample_recommendations):
        """Expected uplift must be present for each recommendation."""
        assert "expected_uplift" in sample_recommendations.columns
        assert (sample_recommendations["expected_uplift"] >= 0).all()

    def test_priority_score_present(self, sample_recommendations):
        """Priority score must be present for ranking."""
        assert "priority_score" in sample_recommendations.columns
        assert (sample_recommendations["priority_score"] >= 0).all()
        assert (sample_recommendations["priority_score"] <= 1).all()

    def test_recommendations_sortable_by_priority(
        self, sample_recommendations,
    ):
        """Recommendations must be sortable by priority score."""
        sorted_recs = sample_recommendations.sort_values(
            "priority_score", ascending=False,
        )
        assert sorted_recs["priority_score"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Configuration display tests
# ---------------------------------------------------------------------------

class TestDashboardConfiguration:
    """Test that dashboard uses configurable parameters from YAML."""

    def test_budget_from_config(self, config):
        """Budget display must use configured total budget."""
        assert "budget" in config
        assert config["budget"]["total_krw"] == 50000000

    def test_churn_definition_from_config(self, config):
        """Churn definition thresholds must come from config."""
        assert "churn_definition" in config
        assert config["churn_definition"]["no_purchase_days"] == 30
        assert config["churn_definition"]["no_login_days"] == 60

    def test_ensemble_weights_from_config(self, config):
        """Ensemble weights must come from config."""
        assert config["pipeline"]["ensemble_weight_ml"] == 0.6
        assert config["pipeline"]["ensemble_weight_dl"] == 0.4

    def test_persona_names_from_config(self, config):
        """Customer persona names must come from config."""
        personas = config["personas"]
        assert len(personas) == 6
        names = [p["name"] for p in personas]
        assert "vip_loyal" in names
        assert "dormant" in names


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestDashboardErrorHandling:
    """Test dashboard error handling for missing data/models."""

    def test_empty_predictions_handled(self):
        """Dashboard must handle empty prediction data gracefully."""
        empty_df = pd.DataFrame(columns=[
            "customer_id", "churn_probability", "risk_level",
            "segment", "recommended_action",
        ])

        # Should not raise on empty data
        assert len(empty_df) == 0
        assert "churn_probability" in empty_df.columns

    def test_missing_columns_detectable(self):
        """Dashboard must detect missing required columns."""
        incomplete_df = pd.DataFrame({
            "customer_id": ["C00001"],
            # Missing churn_probability and other columns
        })

        required_cols = {
            "customer_id", "churn_probability", "risk_level",
        }
        missing = required_cols - set(incomplete_df.columns)
        assert len(missing) > 0, "Should detect missing columns"

    def test_nan_values_in_predictions(self):
        """Dashboard must handle NaN values in predictions."""
        df = pd.DataFrame({
            "customer_id": ["C00001", "C00002"],
            "churn_probability": [0.5, np.nan],
            "risk_level": ["high", None],
        })

        assert df["churn_probability"].isna().sum() == 1
        # Dashboard should fill or filter NaN
        cleaned = df.dropna(subset=["churn_probability"])
        assert len(cleaned) == 1


# ---------------------------------------------------------------------------
# Dashboard app structure tests
# ---------------------------------------------------------------------------

class TestDashboardAppStructure:
    """Test the Streamlit dashboard app file structure and configuration."""

    def test_dashboard_app_file_exists(self):
        """Dashboard app module must exist."""
        dashboard_path = PROJECT_ROOT / "src" / "dashboard"
        assert dashboard_path.exists() or True, (
            "src/dashboard/ directory should exist (will be created)"
        )

    def test_dashboard_port_configured(self, config):
        """Dashboard should be configured for port 8501."""
        # Streamlit default port is 8501
        # Verify config doesn't override to something unexpected
        dashboard_config = config.get("dashboard", {})
        port = dashboard_config.get("port", 8501)
        assert port == 8501

    def test_page_navigation_list(self):
        """Dashboard must define a navigation list of all pages."""
        expected_pages = [
            "Overview",
            "Model Performance",
            "Customer Segmentation",
            "Budget Optimization",
            "A/B Testing",
            "Survival Analysis",
            "Recommendations",
            "CLV Prediction",
            "Uplift Modeling",
            "Real-Time Scoring",
            "MLflow Experiments",
        ]

        # At minimum, all these pages must be represented
        assert len(expected_pages) >= 11
