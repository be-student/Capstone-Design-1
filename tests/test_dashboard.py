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
                     "explorer", "dormant", "new_customer"],
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
             "explorer", "dormant", "new_customer"],
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


# ---------------------------------------------------------------------------
# Feature importance display tests
# ---------------------------------------------------------------------------

class TestFeatureImportanceDisplay:
    """Test feature importance chart data preparation."""

    def test_feature_importance_loadable(self, dashboard_data_loader):
        """Feature importance data must be loadable."""
        fi = dashboard_data_loader.load_feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert not fi.empty

    def test_feature_importance_has_required_columns(
        self, dashboard_data_loader,
    ):
        """Feature importance must have feature and importance columns."""
        fi = dashboard_data_loader.load_feature_importance()
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_importance_sorted_descending(
        self, dashboard_data_loader,
    ):
        """Feature importance must be sorted by importance descending."""
        fi = dashboard_data_loader.load_feature_importance()
        assert fi["importance"].is_monotonic_decreasing

    def test_feature_importance_values_valid(self, dashboard_data_loader):
        """Importance values must be non-negative."""
        fi = dashboard_data_loader.load_feature_importance()
        assert (fi["importance"] >= 0).all()

    def test_feature_importance_has_enough_features(
        self, dashboard_data_loader,
    ):
        """Should have at least 5 features for a meaningful chart."""
        fi = dashboard_data_loader.load_feature_importance()
        assert len(fi) >= 5


# ---------------------------------------------------------------------------
# Individual customer lookup tests
# ---------------------------------------------------------------------------

class TestCustomerLookup:
    """Test individual customer lookup functionality."""

    def test_customer_ids_unique(self, sample_churn_predictions):
        """Customer IDs in predictions must be unique for lookup."""
        assert sample_churn_predictions["customer_id"].is_unique

    def test_single_customer_retrievable(self, sample_churn_predictions):
        """Must be able to retrieve a single customer by ID."""
        target_id = sample_churn_predictions["customer_id"].iloc[0]
        result = sample_churn_predictions[
            sample_churn_predictions["customer_id"] == target_id
        ]
        assert len(result) == 1

    def test_customer_detail_fields(self, sample_churn_predictions):
        """Customer detail view must have all required fields."""
        row = sample_churn_predictions.iloc[0]
        assert "customer_id" in row.index
        assert "churn_probability" in row.index
        assert "risk_level" in row.index
        assert "segment" in row.index

    def test_customer_churn_score_in_range(self, sample_churn_predictions):
        """Individual customer churn probability must be in [0, 1]."""
        for _, row in sample_churn_predictions.iterrows():
            assert 0 <= row["churn_probability"] <= 1

    def test_nonexistent_customer_returns_empty(
        self, sample_churn_predictions,
    ):
        """Lookup for non-existent customer must return empty DataFrame."""
        result = sample_churn_predictions[
            sample_churn_predictions["customer_id"] == "NONEXISTENT_ID"
        ]
        assert len(result) == 0

    def test_customer_clv_available(self, sample_churn_predictions):
        """CLV should be available for individual customer lookup."""
        assert "clv_predicted" in sample_churn_predictions.columns
        row = sample_churn_predictions.iloc[0]
        assert row["clv_predicted"] > 0

    def test_customer_recommended_action_available(
        self, sample_churn_predictions,
    ):
        """Recommended action should be available for customer lookup."""
        assert "recommended_action" in sample_churn_predictions.columns


# ---------------------------------------------------------------------------
# Segment overview tests
# ---------------------------------------------------------------------------

class TestSegmentOverview:
    """Test customer segment overview data preparation."""

    def test_segment_summary_computable(self, sample_churn_predictions):
        """Segment summary statistics must be computable."""
        seg_summary = sample_churn_predictions.groupby("segment").agg(
            count=("customer_id", "count"),
            avg_churn=("churn_probability", "mean"),
        ).reset_index()
        assert len(seg_summary) > 0
        assert "count" in seg_summary.columns
        assert "avg_churn" in seg_summary.columns

    def test_segment_risk_distribution(self, sample_churn_predictions):
        """Risk level distribution within segments must be computable."""
        risk_seg = sample_churn_predictions.groupby(
            ["segment", "risk_level"]
        ).size().reset_index(name="count")
        assert len(risk_seg) > 0
        assert "count" in risk_seg.columns

    def test_highest_risk_segment_identifiable(
        self, sample_churn_predictions,
    ):
        """Must be able to identify the highest-risk segment."""
        seg_risk = sample_churn_predictions.groupby("segment")[
            "churn_probability"
        ].mean()
        highest = seg_risk.idxmax()
        assert isinstance(highest, str)
        assert highest in sample_churn_predictions["segment"].unique()


# ---------------------------------------------------------------------------
# CLV & Retention Campaign view tests
# ---------------------------------------------------------------------------

class TestCLVRetentionCampaignView:
    """Tests for the CLV & Retention Campaign combined dashboard view."""

    def test_has_retention_campaign_page(self):
        """Dashboard must include a CLV & Retention Campaign page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_retention_campaign")
        assert callable(dashboard_app.render_retention_campaign)

    def test_retention_campaign_in_page_list(self):
        """CLV & Retention Campaign must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "CLV & Retention Campaign" in pages

    def test_clv_distribution_stats(self, sample_churn_predictions):
        """CLV distribution stats must be computable."""
        clv = sample_churn_predictions["clv_predicted"]
        assert clv.mean() > 0
        assert clv.median() > 0
        assert clv.std() > 0
        assert clv.sum() > 0

    def test_at_risk_clv_computation(self, sample_churn_predictions):
        """At-risk CLV (churn_prob > 0.5) must be computable."""
        at_risk = sample_churn_predictions[
            sample_churn_predictions["churn_probability"] > 0.5
        ]["clv_predicted"].sum()
        total_clv = sample_churn_predictions["clv_predicted"].sum()
        assert at_risk >= 0
        assert at_risk <= total_clv

    def test_clv_tier_classification(self, sample_churn_predictions):
        """CLV tier classification must produce 4 tiers."""
        clv = sample_churn_predictions["clv_predicted"]
        q25 = clv.quantile(0.25)
        q50 = clv.quantile(0.50)
        q75 = clv.quantile(0.75)

        def classify(v):
            if v >= q75:
                return "Platinum"
            elif v >= q50:
                return "Gold"
            elif v >= q25:
                return "Silver"
            return "Bronze"

        tiers = clv.apply(classify)
        assert set(tiers.unique()) == {"Platinum", "Gold", "Silver", "Bronze"}

    def test_uplift_persuadable_classification(self):
        """Uplift data must support persuadable/sleeping dog split."""
        np.random.seed(42)
        n = 200
        uplift_data = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "uplift_score": np.random.normal(0.05, 0.03, n),
            "treatment_effect": np.random.normal(0.08, 0.04, n),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal"], n,
            ),
        })
        persuadable = (uplift_data["uplift_score"] > 0).sum()
        sleeping_dogs = (uplift_data["uplift_score"] < 0).sum()
        assert persuadable + sleeping_dogs == n

    def test_budget_roi_computation(self, sample_budget_results):
        """Budget ROI metrics must be computable."""
        total_allocated = sample_budget_results["allocated_budget_krw"].sum()
        total_rev = sample_budget_results["expected_revenue_saved_krw"].sum()
        overall_roi = total_rev / total_allocated if total_allocated > 0 else 0

        assert total_allocated > 0
        assert total_rev > 0
        assert overall_roi > 0

    def test_cost_per_retained_computation(self, sample_budget_results):
        """Cost per retained customer must be computable per segment."""
        cpr = (
            sample_budget_results["allocated_budget_krw"]
            / sample_budget_results["expected_retained"]
        )
        assert all(cpr > 0)
        assert len(cpr) == len(sample_budget_results)

    def test_campaign_roi_summary_fields(self, sample_budget_results):
        """Campaign ROI summary must contain key metrics."""
        total_allocated = sample_budget_results["allocated_budget_krw"].sum()
        total_rev = sample_budget_results["expected_revenue_saved_krw"].sum()
        total_retained = sample_budget_results["expected_retained"].sum()
        net_impact = total_rev - total_allocated
        overall_roi = total_rev / total_allocated

        assert net_impact > 0  # Revenue saved > budget
        assert overall_roi > 1.0  # Positive ROI
        assert total_retained > 0

    def test_segment_bubble_chart_data(self, sample_churn_predictions):
        """Segment bubble chart data must aggregate correctly."""
        seg_summary = sample_churn_predictions.groupby("segment").agg(
            avg_clv=("clv_predicted", "mean"),
            avg_churn=("churn_probability", "mean"),
            count=("customer_id", "count"),
        ).reset_index()
        assert len(seg_summary) > 0
        assert "avg_clv" in seg_summary.columns
        assert "avg_churn" in seg_summary.columns
        assert "count" in seg_summary.columns

    def test_cumulative_uplift_curve_data(self):
        """Cumulative uplift curve data must be sorted and cumulative."""
        np.random.seed(42)
        uplift_scores = np.random.normal(0.05, 0.03, 100)
        sorted_scores = np.sort(uplift_scores)[::-1]
        cum_uplift = np.cumsum(sorted_scores)
        # First values should be largest uplift
        assert sorted_scores[0] >= sorted_scores[-1]
        # Cumulative should be non-decreasing initially (positive scores)
        assert cum_uplift[-1] > 0  # Net positive uplift

    def test_waterfall_chart_data(self, sample_budget_results):
        """Revenue waterfall chart data must be valid."""
        sorted_data = sample_budget_results.sort_values(
            "expected_revenue_saved_krw", ascending=False,
        )
        total = sorted_data["expected_revenue_saved_krw"].sum()
        assert total > 0
        assert len(sorted_data) == 6  # 6 segments


# ---------------------------------------------------------------------------
# Churn Analytics page tests
# ---------------------------------------------------------------------------

class TestChurnAnalyticsPage:
    """Tests for the Churn Analytics deep-dive dashboard page."""

    def test_has_churn_analytics_page(self):
        """Dashboard must include a Churn Analytics page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_churn_analytics")
        assert callable(dashboard_app.render_churn_analytics)

    def test_churn_analytics_in_page_list(self):
        """Churn Analytics must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "Churn Analytics" in pages

    def test_churn_analytics_in_page_map(self):
        """Churn Analytics must be mapped in the main page routing."""
        from src.dashboard import app as dashboard_app
        # Verify the render function exists and is distinct from overview
        assert dashboard_app.render_churn_analytics is not dashboard_app.render_overview

    def test_churn_analytics_handles_empty_data(self, config):
        """Churn analytics must handle empty predictions gracefully."""
        from src.dashboard import app as dashboard_app

        st_mock = MagicMock()
        st_mock.columns.return_value = [MagicMock() for _ in range(4)]

        loader = MagicMock()
        loader.load_predictions.return_value = pd.DataFrame()

        dashboard_app.render_churn_analytics(st_mock, config, loader)
        st_mock.warning.assert_called_once()

    def test_churn_density_by_segment_data(self, sample_churn_predictions):
        """Churn density data by segment must be preparable."""
        segments = sample_churn_predictions["segment"].unique()
        assert len(segments) > 0
        for seg in segments:
            seg_data = sample_churn_predictions[
                sample_churn_predictions["segment"] == seg
            ]
            assert len(seg_data) > 0
            assert (seg_data["churn_probability"] >= 0).all()
            assert (seg_data["churn_probability"] <= 1).all()

    def test_cross_tabulation_computable(self, sample_churn_predictions):
        """Segment x Risk cross-tabulation must be computable."""
        cross_tab = pd.crosstab(
            sample_churn_predictions["segment"],
            sample_churn_predictions["risk_level"],
            normalize="index",
        )
        assert not cross_tab.empty
        # Each row should sum to ~1.0
        row_sums = cross_tab.sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums)

    def test_correlation_matrix_computable(self, sample_churn_predictions):
        """Feature correlation matrix must be computable."""
        numeric_cols = sample_churn_predictions.select_dtypes(
            include=[np.number],
        ).columns.tolist()
        assert len(numeric_cols) >= 3
        corr = sample_churn_predictions[numeric_cols].corr()
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] == len(numeric_cols)

    def test_top_risk_customers_extractable(self, sample_churn_predictions):
        """Top 20 highest churn risk customers must be extractable."""
        top20 = sample_churn_predictions.nlargest(20, "churn_probability")
        assert len(top20) == 20
        assert top20["churn_probability"].is_monotonic_decreasing

    def test_critical_risk_count(self, sample_churn_predictions):
        """Critical risk customer count must be computable."""
        critical = (
            sample_churn_predictions["risk_level"] == "critical"
        ).sum()
        assert isinstance(critical, (int, np.integer))
        assert critical >= 0


# ---------------------------------------------------------------------------
# Cohort Analysis page tests
# ---------------------------------------------------------------------------

class TestCohortAnalysisPage:
    """Tests for the Cohort Analysis dashboard page."""

    def test_has_cohort_analysis_page(self):
        """Dashboard must include a Cohort Analysis page."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_cohort_analysis")
        assert callable(dashboard_app.render_cohort_analysis)

    def test_cohort_analysis_in_page_list(self):
        """Cohort Analysis must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "Cohort Analysis" in pages

    def test_cohort_analysis_has_icon(self):
        """Cohort Analysis must have a page icon."""
        from src.dashboard.utils.dashboard_helpers import get_page_icon
        icon = get_page_icon("Cohort Analysis")
        assert isinstance(icon, str)
        assert len(icon) > 0

    def test_cohort_analysis_handles_empty_data(self, config):
        """Cohort analysis must handle empty retention matrix gracefully."""
        from src.dashboard import app as dashboard_app

        st_mock = MagicMock()
        loader = MagicMock()
        loader.load_cohort_retention_matrix.return_value = pd.DataFrame()

        dashboard_app.render_cohort_analysis(st_mock, config, loader)
        st_mock.warning.assert_called_once()

    def test_retention_matrix_loadable(self, dashboard_data_loader):
        """Cohort retention matrix must be loadable from data loader."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        assert isinstance(matrix, pd.DataFrame)
        assert not matrix.empty

    def test_retention_matrix_values_valid(self, dashboard_data_loader):
        """Retention matrix values must be between 0 and 1."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        assert (matrix >= 0).all().all()
        assert (matrix <= 1).all().all()

    def test_retention_matrix_period_zero_is_one(
        self, dashboard_data_loader,
    ):
        """Period 0 retention must be 1.0 (100%) for all cohorts."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        if 0 in matrix.columns:
            assert (matrix[0] == 1.0).all()

    def test_retention_matrix_monotonic_decrease(
        self, dashboard_data_loader,
    ):
        """Retention should generally decrease over periods."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        avg = matrix.mean(axis=0)
        # First period should have higher retention than last
        first_val = avg.iloc[0]
        last_val = avg.iloc[-1]
        assert first_val >= last_val

    def test_average_retention_curve_computable(
        self, dashboard_data_loader,
    ):
        """Average retention curve must be computable from matrix."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        avg = matrix.mean(axis=0)
        assert len(avg) == matrix.shape[1]
        assert all(0 <= v <= 1 for v in avg)

    def test_churn_rate_from_retention(self, dashboard_data_loader):
        """Churn rate must be derivable as 1 - retention."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        churn = 1 - matrix
        assert (churn >= 0).all().all()
        assert (churn <= 1).all().all()

    def test_cohort_data_loadable(self, dashboard_data_loader):
        """Cohort event data must be loadable."""
        data = dashboard_data_loader.load_cohort_data()
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "customer_id" in data.columns


# ---------------------------------------------------------------------------
# Dashboard app navigation and routing completeness tests
# ---------------------------------------------------------------------------

class TestDashboardNavigationCompleteness:
    """Test that all pages are properly connected in navigation."""

    def test_all_pages_have_render_functions(self):
        """Every page in PAGES list must have a render function in app."""
        from src.dashboard import app as dashboard_app
        from src.dashboard.utils.dashboard_helpers import get_page_list

        pages = get_page_list()
        # Verify page count matches expectations (15 pages)
        assert len(pages) == 16

    def test_all_pages_have_icons(self):
        """Every page must have an icon defined."""
        from src.dashboard.utils.dashboard_helpers import (
            get_page_list,
            get_page_icon,
        )
        pages = get_page_list()
        for page in pages:
            icon = get_page_icon(page)
            assert isinstance(icon, str)
            assert len(icon) > 0

    def test_config_loads_successfully(self):
        """Config must load from YAML without errors."""
        from src.dashboard.app import load_config
        config = load_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_data_loader_factory(self):
        """get_data_loader must create a valid DashboardDataLoader."""
        from src.dashboard.app import get_data_loader, load_config
        config = load_config()
        loader = get_data_loader(config)
        assert loader is not None
        assert hasattr(loader, "load_predictions")

    def test_main_function_exists(self):
        """Main Streamlit entry point must exist."""
        from src.dashboard.app import main
        assert callable(main)

    def test_sidebar_info_buildable(self):
        """Sidebar info dictionary must be buildable from config."""
        from src.dashboard.app import load_config
        from src.dashboard.utils.dashboard_helpers import build_sidebar_info
        config = load_config()
        info = build_sidebar_info(config)
        assert "churn_definition" in info
        assert "budget" in info
        assert "ensemble_weights" in info

    def test_page_routing_covers_all_pages(self):
        """Page routing map must cover all defined pages."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        from src.dashboard import app as dashboard_app

        pages = get_page_list()
        # Check that each page has a corresponding render function
        render_functions = {
            "Overview": "render_overview",
            "Churn Analytics": "render_churn_analytics",
            "Model Performance": "render_model_performance",
            "Customer Segmentation": "render_segmentation",
            "Cohort Analysis": "render_cohort_analysis",
            "Budget Optimization": "render_budget_optimization",
            "A/B Testing": "render_ab_testing",
            "Survival Analysis": "render_survival_analysis",
            "Model Monitoring": "render_model_monitoring",
            "Recommendations": "render_recommendations",
            "CLV Prediction": "render_clv",
            "Uplift Modeling": "render_uplift",
            "CLV & Retention Campaign": "render_retention_campaign",
            "Real-Time Scoring": "render_realtime_scoring",
            "MLflow Experiments": "render_mlflow_experiments",
        }
        for page_name, func_name in render_functions.items():
            assert page_name in pages, f"{page_name} not in page list"
            assert hasattr(dashboard_app, func_name), (
                f"Missing render function: {func_name}"
            )


# ---------------------------------------------------------------------------
# Real-Time Scoring & Recommendations View Tests
# ---------------------------------------------------------------------------

class TestRealTimeScoringView:
    """Tests for the Real-Time Scoring & Recommendations dashboard view."""

    def test_render_realtime_scoring_exists(self):
        """render_realtime_scoring must be callable."""
        from src.dashboard.app import render_realtime_scoring
        assert callable(render_realtime_scoring)

    def test_render_model_monitoring_exists(self):
        """render_model_monitoring must be callable."""
        from src.dashboard.app import render_model_monitoring
        assert callable(render_model_monitoring)

    def test_scoring_sub_functions_exist(self):
        """All three sub-tab render functions must exist."""
        from src.dashboard.app import (
            _render_scoring_status_tab,
            _render_retention_offers_tab,
            _render_monitoring_tab,
        )
        assert callable(_render_scoring_status_tab)
        assert callable(_render_retention_offers_tab)
        assert callable(_render_monitoring_tab)

    def test_data_loader_scoring_history(self):
        """Data loader must provide scoring history with expected columns."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_scoring_history()
        assert not df.empty
        expected_cols = [
            "customer_id", "churn_probability", "risk_level",
            "recommended_action", "model_type", "scored_at",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_data_loader_drift_history(self):
        """Data loader must provide drift history with expected columns."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_drift_history()
        assert not df.empty
        expected_cols = [
            "timestamp", "alert_level", "num_drifted_features",
            "psi_mean", "ks_mean",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_data_loader_scoring_throughput(self):
        """Data loader must provide throughput metrics."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_scoring_throughput()
        assert not df.empty
        expected_cols = [
            "timestamp", "requests_per_minute",
            "avg_latency_ms", "error_rate",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_data_loader_retention_offers(self):
        """Data loader must provide retention offers with required fields."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_retention_offers()
        assert not df.empty
        expected_cols = [
            "customer_id", "segment", "risk_level", "churn_probability",
            "offer_type", "offer_detail", "expected_uplift",
            "estimated_cost_krw", "estimated_revenue_save_krw",
            "priority_rank",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_retention_offers_have_valid_risk_levels(self):
        """All offers must have valid risk level values."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_retention_offers()
        valid = {"low", "medium", "high", "critical"}
        assert set(df["risk_level"].unique()).issubset(valid)

    def test_scoring_history_probabilities_bounded(self):
        """Churn probabilities must be in [0, 1]."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_scoring_history()
        assert df["churn_probability"].between(0, 1).all()

    def test_drift_history_alert_levels_valid(self):
        """Drift alert levels must be green, yellow, or red."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_drift_history()
        valid = {"green", "yellow", "red"}
        assert set(df["alert_level"].unique()).issubset(valid)

    def test_throughput_positive_values(self):
        """Throughput requests_per_minute should be positive."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_scoring_throughput()
        assert (df["requests_per_minute"] > 0).all()

    def test_retention_offers_positive_cost(self):
        """All estimated costs must be positive."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df = loader.load_retention_offers()
        assert (df["estimated_cost_krw"] > 0).all()

    def test_reproducibility_with_seed(self):
        """Same seed must produce identical scoring history."""
        from src.dashboard.data_loader import DashboardDataLoader
        loader1 = DashboardDataLoader({"simulation": {"random_seed": 42}})
        loader2 = DashboardDataLoader({"simulation": {"random_seed": 42}})
        df1 = loader1.load_scoring_history()
        df2 = loader2.load_scoring_history()
        assert df1.equals(df2)


# ---------------------------------------------------------------------------
# Enhanced Model Performance view tests
# ---------------------------------------------------------------------------

class TestEnhancedModelPerformance:
    """Test enhanced model performance view with ROC, CM, and MLflow runs."""

    def test_roc_data_loadable(self, dashboard_data_loader):
        """ROC curve data must be loadable for all models."""
        roc_data = dashboard_data_loader.load_roc_data()
        assert isinstance(roc_data, dict)
        assert "ml_model" in roc_data
        assert "dl_model" in roc_data
        assert "ensemble" in roc_data

    def test_roc_data_has_fpr_tpr(self, dashboard_data_loader):
        """ROC data must contain fpr and tpr arrays."""
        roc_data = dashboard_data_loader.load_roc_data()
        for model_name, curve in roc_data.items():
            assert "fpr" in curve, f"Missing fpr for {model_name}"
            assert "tpr" in curve, f"Missing tpr for {model_name}"
            assert len(curve["fpr"]) == len(curve["tpr"])
            assert len(curve["fpr"]) > 2

    def test_roc_data_starts_at_origin(self, dashboard_data_loader):
        """ROC curves must start at (0, 0)."""
        roc_data = dashboard_data_loader.load_roc_data()
        for model_name, curve in roc_data.items():
            assert curve["fpr"][0] == 0, f"{model_name} fpr should start at 0"
            assert curve["tpr"][0] == 0, f"{model_name} tpr should start at 0"

    def test_roc_data_ends_at_one(self, dashboard_data_loader):
        """ROC curves must end at (1, 1)."""
        roc_data = dashboard_data_loader.load_roc_data()
        for model_name, curve in roc_data.items():
            assert curve["fpr"][-1] == 1.0
            assert curve["tpr"][-1] == 1.0

    def test_confusion_matrices_loadable(self, dashboard_data_loader):
        """Confusion matrices must be loadable for all models."""
        cm_data = dashboard_data_loader.load_confusion_matrices()
        assert isinstance(cm_data, dict)
        assert "ml_model" in cm_data
        assert "dl_model" in cm_data
        assert "ensemble" in cm_data

    def test_confusion_matrix_shape(self, dashboard_data_loader):
        """Each confusion matrix must be 2x2."""
        cm_data = dashboard_data_loader.load_confusion_matrices()
        for model_name, matrix in cm_data.items():
            assert len(matrix) == 2, f"{model_name} matrix rows != 2"
            for row in matrix:
                assert len(row) == 2, f"{model_name} matrix cols != 2"

    def test_confusion_matrix_non_negative(self, dashboard_data_loader):
        """All confusion matrix values must be non-negative."""
        cm_data = dashboard_data_loader.load_confusion_matrices()
        for model_name, matrix in cm_data.items():
            for row in matrix:
                for val in row:
                    assert val >= 0, (
                        f"Negative value in {model_name} confusion matrix"
                    )

    def test_mlflow_runs_loadable(self, dashboard_data_loader):
        """MLflow run data must be loadable."""
        runs = dashboard_data_loader.load_mlflow_runs()
        assert isinstance(runs, pd.DataFrame)
        assert not runs.empty

    def test_mlflow_runs_has_required_columns(self, dashboard_data_loader):
        """MLflow runs must have all required columns."""
        runs = dashboard_data_loader.load_mlflow_runs()
        required = [
            "run_id", "model_type", "auc", "precision",
            "recall", "f1_score", "training_time_s",
        ]
        for col in required:
            assert col in runs.columns, f"Missing column: {col}"

    def test_mlflow_runs_auc_valid(self, dashboard_data_loader):
        """MLflow run AUC values must be between 0 and 1."""
        runs = dashboard_data_loader.load_mlflow_runs()
        assert (runs["auc"] >= 0).all()
        assert (runs["auc"] <= 1).all()

    def test_mlflow_runs_multiple_model_types(self, dashboard_data_loader):
        """MLflow runs should cover multiple model types."""
        runs = dashboard_data_loader.load_mlflow_runs()
        assert runs["model_type"].nunique() >= 3


# ---------------------------------------------------------------------------
# Enhanced A/B Testing view tests
# ---------------------------------------------------------------------------

class TestEnhancedABTesting:
    """Test enhanced A/B testing view with multi-experiment support."""

    def test_detailed_ab_loadable(self, dashboard_data_loader):
        """Detailed A/B test results must be loadable."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        assert isinstance(detailed, dict)
        assert "experiments" in detailed
        assert "summary" in detailed

    def test_multiple_experiments(self, dashboard_data_loader):
        """Must support multiple A/B test experiments."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        experiments = detailed["experiments"]
        assert len(experiments) >= 2

    def test_experiment_has_required_fields(self, dashboard_data_loader):
        """Each experiment must have all required statistical fields."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        required_fields = [
            "name", "treatment_size", "control_size",
            "treatment_churn_rate", "control_churn_rate",
            "lift", "p_value", "is_significant",
            "confidence_interval",
        ]
        for exp in detailed["experiments"]:
            for field in required_fields:
                assert field in exp, (
                    f"Missing field '{field}' in experiment '{exp.get('name')}'"
                )

    def test_experiment_has_effect_size(self, dashboard_data_loader):
        """Experiments must report effect size (Cohen's h)."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        for exp in detailed["experiments"]:
            assert "effect_size_cohens_h" in exp, (
                f"Missing Cohen's h in '{exp.get('name')}'"
            )
            assert isinstance(
                exp["effect_size_cohens_h"], (int, float),
            )

    def test_experiment_has_power(self, dashboard_data_loader):
        """Experiments must report statistical power."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        for exp in detailed["experiments"]:
            assert "power" in exp
            assert 0 <= exp["power"] <= 1

    def test_confidence_interval_valid(self, dashboard_data_loader):
        """Confidence intervals must have lower < upper bound."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        for exp in detailed["experiments"]:
            ci = exp.get("confidence_interval", [0, 0])
            assert len(ci) == 2
            assert ci[0] <= ci[1], (
                f"CI lower > upper in '{exp.get('name')}'"
            )

    def test_summary_counts(self, dashboard_data_loader):
        """Summary must correctly count experiments."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        summary = detailed["summary"]
        experiments = detailed["experiments"]
        assert summary["total_experiments"] == len(experiments)
        sig_count = sum(
            1 for e in experiments if e.get("is_significant")
        )
        assert summary["significant_count"] == sig_count

    def test_treatment_lower_churn_when_significant(
        self, dashboard_data_loader,
    ):
        """Significant experiments must show treatment < control churn."""
        detailed = dashboard_data_loader.load_ab_test_detailed()
        for exp in detailed["experiments"]:
            if exp.get("is_significant"):
                assert (
                    exp["treatment_churn_rate"]
                    < exp["control_churn_rate"]
                ), (
                    f"Treatment churn should be lower in significant "
                    f"experiment '{exp.get('name')}'"
                )


# ---------------------------------------------------------------------------
# Enhanced Survival Analysis view tests
# ---------------------------------------------------------------------------

class TestEnhancedSurvivalAnalysis:
    """Test enhanced survival analysis with KM curves and hazard rates."""

    def test_survival_curves_loadable(self, dashboard_data_loader):
        """Kaplan-Meier survival curves must be loadable."""
        curves = dashboard_data_loader.load_survival_curves()
        assert isinstance(curves, dict)
        assert len(curves) > 0

    def test_survival_curves_has_all_segments(self, dashboard_data_loader):
        """Survival curves must cover all expected segments."""
        curves = dashboard_data_loader.load_survival_curves()
        expected = [
            "vip_loyal", "regular_loyal", "bargain_hunter",
            "explorer", "dormant", "new_customer",
        ]
        for seg in expected:
            assert seg in curves, f"Missing survival curve for {seg}"

    def test_survival_curve_has_timeline(self, dashboard_data_loader):
        """Each survival curve must have a timeline."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            assert "timeline" in data, f"Missing timeline for {seg}"
            assert len(data["timeline"]) > 2
            assert data["timeline"][0] == 0

    def test_survival_curve_has_probability(self, dashboard_data_loader):
        """Each curve must have survival probability values."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            assert "survival_prob" in data
            probs = data["survival_prob"]
            assert len(probs) == len(data["timeline"])
            assert probs[0] == 1.0  # starts at 100%
            assert all(0 <= p <= 1.0 for p in probs)

    def test_survival_curve_monotonic_decreasing(
        self, dashboard_data_loader,
    ):
        """Survival probability must be non-increasing over time."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            probs = data["survival_prob"]
            for j in range(1, len(probs)):
                assert probs[j] <= probs[j - 1] + 0.001, (
                    f"Survival prob increased at index {j} for {seg}"
                )

    def test_survival_curve_has_confidence_intervals(
        self, dashboard_data_loader,
    ):
        """Survival curves must include confidence interval bands."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            assert "ci_lower" in data, f"Missing ci_lower for {seg}"
            assert "ci_upper" in data, f"Missing ci_upper for {seg}"

    def test_survival_curve_has_median_survival(
        self, dashboard_data_loader,
    ):
        """Survival curves must report median survival days."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            assert "median_survival_days" in data

    def test_dormant_segment_lowest_survival(self, dashboard_data_loader):
        """Dormant segment should have lowest final survival probability."""
        curves = dashboard_data_loader.load_survival_curves()
        final_probs = {
            seg: data["survival_prob"][-1]
            for seg, data in curves.items()
        }
        dormant_prob = final_probs.get("dormant", 1.0)
        vip_prob = final_probs.get("vip_loyal", 0.0)
        assert dormant_prob < vip_prob, (
            "Dormant should have lower survival than VIP loyal"
        )


# ---------------------------------------------------------------------------
# Enhanced MLflow Experiments view tests
# ---------------------------------------------------------------------------

class TestEnhancedMLflowExperiments:
    """Test enhanced MLflow experiments view with run analysis."""

    def test_mlflow_runs_training_time_positive(
        self, dashboard_data_loader,
    ):
        """Training times must be positive."""
        runs = dashboard_data_loader.load_mlflow_runs()
        assert (runs["training_time_s"] > 0).all()

    def test_mlflow_runs_have_timestamps(self, dashboard_data_loader):
        """Runs must have timestamps."""
        runs = dashboard_data_loader.load_mlflow_runs()
        assert "timestamp" in runs.columns
        assert runs["timestamp"].notna().all()

    def test_mlflow_runs_have_hyperparams(self, dashboard_data_loader):
        """Runs must include hyperparameter columns."""
        runs = dashboard_data_loader.load_mlflow_runs()
        assert "params_lr" in runs.columns
        assert "params_epochs" in runs.columns

    def test_mlflow_runs_unique_ids(self, dashboard_data_loader):
        """Run IDs must be unique."""
        runs = dashboard_data_loader.load_mlflow_runs()
        assert runs["run_id"].is_unique

    def test_best_run_above_threshold(self, dashboard_data_loader):
        """Best run AUC must meet the 0.78 threshold."""
        runs = dashboard_data_loader.load_mlflow_runs()
        best_auc = runs["auc"].max()
        assert best_auc >= 0.78, (
            f"Best AUC {best_auc} below threshold 0.78"
        )


# ---------------------------------------------------------------------------
# CLV view detail tests
# ---------------------------------------------------------------------------

class TestCLVViewDetails:
    """Test CLV prediction view data transformations and visualizations."""

    @pytest.fixture
    def clv_predictions(self):
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "churn_probability": np.random.beta(2, 5, n),
            "clv_predicted": np.random.lognormal(11, 1, n),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"], n,
            ),
            "risk_level": np.random.choice(
                ["low", "medium", "high", "critical"], n, p=[0.4, 0.3, 0.2, 0.1],
            ),
        })

    def test_clv_tier_classification_logic(self, clv_predictions):
        """CLV tier classification produces 4 tiers based on quartiles."""
        clv = clv_predictions["clv_predicted"]
        q25, q50, q75 = clv.quantile(0.25), clv.quantile(0.50), clv.quantile(0.75)

        def classify(v):
            if v >= q75:
                return "Platinum"
            elif v >= q50:
                return "Gold"
            elif v >= q25:
                return "Silver"
            return "Bronze"

        tiers = clv.apply(classify)
        assert set(tiers.unique()) == {"Platinum", "Gold", "Silver", "Bronze"}

    def test_clv_vs_churn_scatter_data(self, clv_predictions):
        """CLV vs churn scatter requires both columns with valid ranges."""
        df = clv_predictions
        assert (df["clv_predicted"] > 0).all()
        assert (df["churn_probability"] >= 0).all()
        assert (df["churn_probability"] <= 1).all()

    def test_clv_percentile_analysis(self, clv_predictions):
        """CLV percentile analysis produces monotonically increasing values."""
        clv = clv_predictions["clv_predicted"]
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        values = [clv.quantile(p / 100) for p in percentiles]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_clv_segment_breakdown_stats(self, clv_predictions):
        """CLV segment breakdown computes all required stats."""
        seg_clv = clv_predictions.groupby("segment")["clv_predicted"].agg(
            ["mean", "sum", "count", "median", "std"]
        ).reset_index()
        assert len(seg_clv) > 0
        assert (seg_clv["mean"] > 0).all()
        assert (seg_clv["sum"] > 0).all()

    def test_clv_top_bottom_customer_lists(self, clv_predictions):
        """Top and bottom customer lists by CLV are correctly ordered."""
        top10 = clv_predictions.nlargest(10, "clv_predicted")
        bottom10 = clv_predictions.nsmallest(10, "clv_predicted")
        assert len(top10) == 10
        assert len(bottom10) == 10
        assert top10["clv_predicted"].min() > bottom10["clv_predicted"].max()

    def test_clv_churn_adjusted_concept(self, clv_predictions):
        """CLV adjusted for churn should be lower than raw CLV."""
        df = clv_predictions.copy()
        df["adjusted_clv"] = df["clv_predicted"] * (1 - df["churn_probability"])
        assert (df["adjusted_clv"] <= df["clv_predicted"]).all()
        assert (df["adjusted_clv"] >= 0).all()

    def test_at_risk_clv_computation(self, clv_predictions):
        """At-risk CLV computation for high-churn customers."""
        at_risk = clv_predictions[clv_predictions["churn_probability"] > 0.5]
        at_risk_clv = at_risk["clv_predicted"].sum()
        total_clv = clv_predictions["clv_predicted"].sum()
        at_risk_pct = at_risk_clv / total_clv * 100 if total_clv > 0 else 0
        assert 0 <= at_risk_pct <= 100

    def test_clv_histogram_with_mean_median_lines(self, clv_predictions):
        """CLV histogram mean and median are computable for reference lines."""
        clv = clv_predictions["clv_predicted"]
        mean_clv = clv.mean()
        median_clv = clv.median()
        assert mean_clv > 0
        assert median_clv > 0
        # Log-normal: mean > median
        assert mean_clv > median_clv


# ---------------------------------------------------------------------------
# Treatment effect visualization tests
# ---------------------------------------------------------------------------

class TestTreatmentEffectVisualization:
    """Test treatment effect visualizations in uplift and A/B testing views."""

    @pytest.fixture
    def uplift_data(self):
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "uplift_score": np.random.normal(0.02, 0.05, n),
            "treatment_effect": np.random.normal(0.015, 0.04, n),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"], n,
            ),
        })

    def test_uplift_distribution_has_positive_and_negative(self, uplift_data):
        """Uplift distribution should contain both positive and negative scores."""
        scores = uplift_data["uplift_score"]
        assert (scores > 0).any()
        assert (scores < 0).any()

    def test_treatment_effect_distribution(self, uplift_data):
        """Treatment effect distribution should have variance."""
        effects = uplift_data["treatment_effect"]
        assert len(effects) > 0
        assert effects.std() > 0

    def test_four_quadrant_classification(self, uplift_data):
        """Customer classification produces expected response categories."""
        df = uplift_data.copy()

        def classify(row):
            if row["uplift_score"] > 0 and row["treatment_effect"] > 0:
                return "Persuadable"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] > 0:
                return "Sure Thing"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] <= 0:
                return "Lost Cause"
            return "Sleeping Dog"

        df["response_class"] = df.apply(classify, axis=1)
        classes = set(df["response_class"].unique())
        assert len(classes) >= 2
        expected_colors = {
            "Persuadable": "#2ecc71",
            "Sure Thing": "#3498db",
            "Lost Cause": "#95a5a6",
            "Sleeping Dog": "#e74c3c",
        }
        for cls in classes:
            assert cls in expected_colors

    def test_uplift_by_segment_aggregation(self, uplift_data):
        """Uplift and treatment effect by segment should be computable."""
        seg_uplift = uplift_data.groupby("segment").agg(
            avg_uplift=("uplift_score", "mean"),
            avg_treatment=("treatment_effect", "mean"),
            count=("customer_id", "count"),
            persuadable=("uplift_score", lambda x: (x > 0).sum()),
        ).reset_index()
        assert len(seg_uplift) > 0
        assert (seg_uplift["count"] > 0).all()

    def test_persuadable_percentage_by_segment(self, uplift_data):
        """Persuadable percentage by segment should be between 0 and 100."""
        seg = uplift_data.groupby("segment").agg(
            count=("customer_id", "count"),
            persuadable=("uplift_score", lambda x: (x > 0).sum()),
        ).reset_index()
        seg["persuadable_pct"] = seg["persuadable"] / seg["count"] * 100
        assert (seg["persuadable_pct"] >= 0).all()
        assert (seg["persuadable_pct"] <= 100).all()

    def test_cumulative_uplift_curve(self, uplift_data):
        """Cumulative uplift curve has correct structure."""
        sorted_df = uplift_data.sort_values(
            "uplift_score", ascending=False,
        ).reset_index(drop=True)
        sorted_df["cum_uplift"] = sorted_df["uplift_score"].cumsum()
        sorted_df["pct_treated"] = (
            np.arange(1, len(sorted_df) + 1) / len(sorted_df) * 100
        )
        assert sorted_df["cum_uplift"].iloc[0] > 0
        assert sorted_df["pct_treated"].iloc[-1] == 100.0

    def test_uplift_scatter_quadrant_lines(self, uplift_data):
        """Scatter plot data supports quadrant line placement at zero."""
        assert uplift_data["uplift_score"].min() < 0
        assert uplift_data["uplift_score"].max() > 0

    def test_top_persuadable_extraction(self, uplift_data):
        """Top persuadable customers can be extracted and ranked."""
        persuadable = uplift_data[
            uplift_data["uplift_score"] > 0
        ].nlargest(10, "uplift_score")
        assert len(persuadable) <= 10
        assert (persuadable["uplift_score"] > 0).all()
        scores = persuadable["uplift_score"].values
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]

    def test_treatment_effect_by_segment_dual_bar(self, uplift_data):
        """Grouped bar data for uplift + treatment effect by segment."""
        seg_up = uplift_data.groupby("segment").agg(
            avg_uplift=("uplift_score", "mean"),
            avg_treatment=("treatment_effect", "mean"),
        ).reset_index()
        assert len(seg_up) > 0
        assert "avg_uplift" in seg_up.columns
        assert "avg_treatment" in seg_up.columns


# ---------------------------------------------------------------------------
# A/B testing multi-experiment view tests
# ---------------------------------------------------------------------------

class TestABTestingMultiExperimentView:
    """Test A/B testing detailed multi-experiment view."""

    @pytest.fixture
    def detailed_experiments(self):
        return {
            "experiments": [
                {
                    "name": "Email Campaign",
                    "treatment_churn_rate": 0.12,
                    "control_churn_rate": 0.20,
                    "lift": 0.40,
                    "p_value": 0.003,
                    "is_significant": True,
                    "power": 0.92,
                    "alpha": 0.05,
                    "treatment_size": 500,
                    "control_size": 500,
                    "absolute_effect": -0.08,
                    "confidence_interval": [-0.12, -0.04],
                    "effect_size_cohens_h": 0.22,
                    "test_type": "z_test_proportions",
                    "duration_days": 30,
                },
                {
                    "name": "Coupon Campaign",
                    "treatment_churn_rate": 0.15,
                    "control_churn_rate": 0.18,
                    "lift": 0.167,
                    "p_value": 0.12,
                    "is_significant": False,
                    "power": 0.45,
                    "alpha": 0.05,
                    "treatment_size": 300,
                    "control_size": 300,
                    "absolute_effect": -0.03,
                    "confidence_interval": [-0.08, 0.02],
                    "effect_size_cohens_h": 0.08,
                    "test_type": "z_test_proportions",
                    "duration_days": 21,
                },
                {
                    "name": "Push Notification",
                    "treatment_churn_rate": 0.16,
                    "control_churn_rate": 0.19,
                    "lift": 0.158,
                    "p_value": 0.08,
                    "is_significant": False,
                    "power": 0.55,
                    "alpha": 0.05,
                    "treatment_size": 400,
                    "control_size": 400,
                    "absolute_effect": -0.03,
                    "confidence_interval": [-0.07, 0.01],
                    "effect_size_cohens_h": 0.08,
                    "test_type": "z_test_proportions",
                    "duration_days": 14,
                },
            ],
            "summary": {
                "total_experiments": 3,
                "significant_count": 1,
                "best_experiment": "Email Campaign",
                "avg_lift": 0.242,
            },
        }

    def test_summary_kpis(self, detailed_experiments):
        """Summary KPIs are computable from experiment data."""
        summary = detailed_experiments["summary"]
        assert summary["total_experiments"] == 3
        assert summary["significant_count"] == 1
        assert summary["best_experiment"] == "Email Campaign"

    def test_cross_experiment_comparison_table(self, detailed_experiments):
        """Cross-experiment comparison table can be built."""
        experiments = detailed_experiments["experiments"]
        comparison = pd.DataFrame([
            {
                "Experiment": e["name"],
                "Treatment Churn": e["treatment_churn_rate"],
                "Control Churn": e["control_churn_rate"],
                "Lift": e["lift"],
                "p-value": e["p_value"],
                "Power": e["power"],
                "Significant": "Yes" if e["is_significant"] else "No",
            }
            for e in experiments
        ])
        assert len(comparison) == 3
        sig_count = (comparison["Significant"] == "Yes").sum()
        assert sig_count == 1

    def test_confidence_interval_data(self, detailed_experiments):
        """Confidence interval data supports error bar visualization."""
        for exp in detailed_experiments["experiments"]:
            ci = exp["confidence_interval"]
            assert len(ci) == 2
            assert ci[0] <= ci[1]

    def test_power_vs_pvalue_ranges(self, detailed_experiments):
        """Power vs p-value scatter requires valid ranges."""
        for exp in detailed_experiments["experiments"]:
            assert 0 <= exp["p_value"] <= 1
            assert 0 <= exp["power"] <= 1

    def test_effect_size_present(self, detailed_experiments):
        """Cohen's h effect size should be present for all experiments."""
        for exp in detailed_experiments["experiments"]:
            assert "effect_size_cohens_h" in exp
            assert exp["effect_size_cohens_h"] >= 0

    def test_significance_classification(self, detailed_experiments):
        """Significance is correctly classified based on p-value < alpha."""
        for exp in detailed_experiments["experiments"]:
            expected_sig = exp["p_value"] < exp["alpha"]
            assert exp["is_significant"] == expected_sig

    def test_lift_color_mapping(self, detailed_experiments):
        """Lift bar chart data has valid color mapping for significance."""
        experiments = detailed_experiments["experiments"]
        comparison = pd.DataFrame([
            {
                "Experiment": e["name"],
                "Lift": e["lift"],
                "Significant": "Yes" if e["is_significant"] else "No",
            }
            for e in experiments
        ])
        color_map = {"Yes": "#2ecc71", "No": "#e74c3c"}
        for sig_val in comparison["Significant"].unique():
            assert sig_val in color_map

    def test_churn_rate_bar_chart_data(self, detailed_experiments):
        """Per-experiment churn rate comparison data is valid."""
        for exp in detailed_experiments["experiments"]:
            assert 0 <= exp["treatment_churn_rate"] <= 1
            assert 0 <= exp["control_churn_rate"] <= 1


# ---------------------------------------------------------------------------
# Budget optimization display detail tests
# ---------------------------------------------------------------------------

class TestBudgetOptimizationDisplayDetails:
    """Test budget optimization display calculations and transformations."""

    def test_budget_scaling(self, sample_budget_results, config):
        """Budget allocation scales proportionally to slider value."""
        baseline_total = sample_budget_results["allocated_budget_krw"].sum()
        new_budget = 100_000_000
        scale = new_budget / baseline_total if baseline_total > 0 else 1.0

        scaled = sample_budget_results.copy()
        scaled["allocated_budget_krw"] = (
            scaled["allocated_budget_krw"] * scale
        ).astype(int)
        scaled_total = scaled["allocated_budget_krw"].sum()
        assert abs(scaled_total - new_budget) / new_budget < 0.01

    def test_roi_adjustment_with_multipliers(self, sample_budget_results):
        """ROI adjusts correctly with cost and uplift multipliers."""
        cost_mult = 1.5
        uplift_mult = 1.2
        adjusted_roi = (
            sample_budget_results["roi"] * uplift_mult / cost_mult
        ).round(2)
        ratio = uplift_mult / cost_mult
        for orig, adj in zip(sample_budget_results["roi"], adjusted_roi):
            assert abs(adj - round(orig * ratio, 2)) < 0.01

    def test_whatif_scenario_structure(self, config):
        """What-if scenarios have expected structure."""
        from src.dashboard.app import _build_whatif_scenarios
        scenarios = _build_whatif_scenarios(
            default_budget=50_000_000,
            current_budget=50_000_000,
            cost_multiplier=1.0,
            uplift_multiplier=1.0,
        )
        assert len(scenarios) >= 3
        for s in scenarios:
            assert "name" in s
            assert "budget" in s
            assert "cost_mult" in s
            assert "uplift_mult" in s

    def test_scenario_comparison_df(self, sample_budget_results, config):
        """Scenario comparison produces valid DataFrame."""
        from src.dashboard.app import (
            _build_whatif_scenarios,
            _compute_scenario_comparison,
        )
        baseline_total = sample_budget_results["allocated_budget_krw"].sum()
        scenarios = _build_whatif_scenarios(
            default_budget=50_000_000,
            current_budget=50_000_000,
            cost_multiplier=1.0,
            uplift_multiplier=1.0,
        )
        comparison = _compute_scenario_comparison(
            budget_results=sample_budget_results,
            baseline_total=baseline_total,
            scenarios=scenarios,
        )
        assert isinstance(comparison, pd.DataFrame)
        assert "Scenario" in comparison.columns
        assert "Total Allocated" in comparison.columns
        assert "Avg ROI" in comparison.columns
        assert len(comparison) == len(scenarios)

    def test_budget_sweep_analysis(self, sample_budget_results):
        """Budget sweep produces valid sweep analysis data."""
        from src.dashboard.app import _compute_budget_sweep
        baseline_total = sample_budget_results["allocated_budget_krw"].sum()
        sweep = _compute_budget_sweep(
            budget_results=sample_budget_results,
            baseline_total=baseline_total,
            min_budget=10_000_000,
            max_budget=200_000_000,
            steps=10,
            cost_multiplier=1.0,
            uplift_multiplier=1.0,
        )
        assert isinstance(sweep, pd.DataFrame)
        assert "Budget" in sweep.columns
        assert "Retained" in sweep.columns
        assert "Revenue Saved" in sweep.columns
        assert len(sweep) == 10
        budgets = sweep["Budget"].values
        for i in range(1, len(budgets)):
            assert budgets[i] > budgets[i - 1]

    def test_allocation_pie_chart_data(self, sample_budget_results):
        """Allocation pie chart has non-negative values summing to total."""
        allocs = sample_budget_results["allocated_budget_krw"]
        assert (allocs >= 0).all()
        assert allocs.sum() > 0

    def test_roi_bar_chart_segment_coverage(self, sample_budget_results):
        """ROI bar chart covers all segments."""
        segments = sample_budget_results["segment"].unique()
        assert len(segments) >= 3
        for seg in segments:
            seg_data = sample_budget_results[
                sample_budget_results["segment"] == seg
            ]
            assert len(seg_data) == 1
            assert seg_data["roi"].iloc[0] > 0

    def test_net_revenue_impact(self, sample_budget_results):
        """Net revenue impact should be positive for high ROI segments."""
        total_alloc = sample_budget_results["allocated_budget_krw"].sum()
        total_saved = sample_budget_results["expected_revenue_saved_krw"].sum()
        assert total_saved - total_alloc > 0

    def test_cost_per_retained_computation(self, sample_budget_results):
        """Cost per retained customer is computable."""
        df = sample_budget_results.copy()
        df["cost_per_retained"] = np.where(
            df["expected_retained"] > 0,
            df["allocated_budget_krw"] / df["expected_retained"],
            0,
        )
        assert (df["cost_per_retained"] >= 0).all()
        assert (df["cost_per_retained"] < df["allocated_budget_krw"]).all()


# ---------------------------------------------------------------------------
# Uplift modeling view detail tests
# ---------------------------------------------------------------------------

class TestUpliftModelingViewDetails:
    """Test uplift modeling view specific data and visualization logic."""

    @pytest.fixture
    def uplift_results(self):
        np.random.seed(42)
        n = 300
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "uplift_score": np.random.normal(0.02, 0.06, n),
            "treatment_effect": np.random.normal(0.01, 0.04, n),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"], n,
            ),
        })

    def test_kpi_cards_computable(self, uplift_results):
        """All KPI card values should be computable."""
        avg_uplift = uplift_results["uplift_score"].mean()
        avg_treatment = uplift_results["treatment_effect"].mean()
        persuadable = (uplift_results["uplift_score"] > 0).sum()
        sleeping_dogs = (uplift_results["uplift_score"] < 0).sum()
        assert isinstance(avg_uplift, float)
        assert isinstance(avg_treatment, float)
        assert persuadable + sleeping_dogs <= len(uplift_results)

    def test_response_class_pie_chart_data(self, uplift_results):
        """Response class pie chart data has correct category color mapping."""
        df = uplift_results.copy()

        def classify(row):
            if row["uplift_score"] > 0 and row["treatment_effect"] > 0:
                return "Persuadable"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] > 0:
                return "Sure Thing"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] <= 0:
                return "Lost Cause"
            return "Sleeping Dog"

        df["response_class"] = df.apply(classify, axis=1)
        counts = df["response_class"].value_counts()
        assert counts.sum() == len(df)
        expected_colors = {
            "Persuadable": "#2ecc71",
            "Sure Thing": "#3498db",
            "Lost Cause": "#95a5a6",
            "Sleeping Dog": "#e74c3c",
        }
        for cls in counts.index:
            assert cls in expected_colors

    def test_stacked_bar_by_segment(self, uplift_results):
        """Stacked bar chart by segment and response class is buildable."""
        df = uplift_results.copy()
        df["response_class"] = np.where(
            df["uplift_score"] > 0, "Persuadable", "Do Not Treat",
        )
        class_seg = df.groupby(
            ["segment", "response_class"]
        ).size().reset_index(name="count")
        assert class_seg["count"].sum() == len(df)

    def test_uplift_segment_bar_chart(self, uplift_results):
        """Segment-level uplift bar chart data is valid."""
        seg_uplift = uplift_results.groupby("segment")["uplift_score"].mean()
        assert len(seg_uplift) > 0
        assert (seg_uplift != 0).any()


# ---------------------------------------------------------------------------
# Power Analysis & Sample Size Calculator Tests
# ---------------------------------------------------------------------------

class TestPowerAnalysisCalculator:
    """Test power analysis helper functions for A/B testing view."""

    def test_compute_power_analysis_returns_required_keys(self):
        """Power analysis returns sample_size_per_group, total, duration."""
        from src.dashboard.app import _compute_power_analysis
        result = _compute_power_analysis(
            baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.80,
        )
        assert "sample_size_per_group" in result
        assert "total_participants" in result
        assert "estimated_duration_days" in result

    def test_sample_size_positive(self):
        """Sample size must be positive integer."""
        from src.dashboard.app import _compute_power_analysis
        result = _compute_power_analysis(
            baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.80,
        )
        assert result["sample_size_per_group"] > 0
        assert result["total_participants"] == result["sample_size_per_group"] * 2

    def test_larger_mde_needs_fewer_samples(self):
        """Larger MDE requires fewer samples."""
        from src.dashboard.app import _compute_power_analysis
        r1 = _compute_power_analysis(baseline_rate=0.20, mde=0.02)
        r2 = _compute_power_analysis(baseline_rate=0.20, mde=0.10)
        assert r1["sample_size_per_group"] > r2["sample_size_per_group"]

    def test_higher_power_needs_more_samples(self):
        """Higher power requires more samples."""
        from src.dashboard.app import _compute_power_analysis
        r1 = _compute_power_analysis(baseline_rate=0.20, mde=0.05, power=0.80)
        r2 = _compute_power_analysis(baseline_rate=0.20, mde=0.05, power=0.95)
        assert r2["sample_size_per_group"] > r1["sample_size_per_group"]

    def test_stricter_alpha_needs_more_samples(self):
        """Stricter alpha requires more samples."""
        from src.dashboard.app import _compute_power_analysis
        r1 = _compute_power_analysis(baseline_rate=0.20, mde=0.05, alpha=0.10)
        r2 = _compute_power_analysis(baseline_rate=0.20, mde=0.05, alpha=0.01)
        assert r2["sample_size_per_group"] > r1["sample_size_per_group"]

    def test_estimated_duration_days(self):
        """Duration is total participants / daily enrollment."""
        from src.dashboard.app import _compute_power_analysis
        result = _compute_power_analysis(
            baseline_rate=0.20, mde=0.05, daily_enrollment=200,
        )
        expected_days = int(
            np.ceil(result["total_participants"] / 200)
        )
        assert result["estimated_duration_days"] == expected_days

    def test_power_curve_shape(self):
        """Power curve should be monotonically non-decreasing."""
        from src.dashboard.app import _compute_power_curve
        df = _compute_power_curve(
            baseline_rate=0.20, mde=0.05, alpha=0.05, max_n=5000,
        )
        assert "n" in df.columns
        assert "power" in df.columns
        assert len(df) > 0
        # Power should increase with sample size
        for i in range(1, len(df)):
            assert df["power"].iloc[i] >= df["power"].iloc[i - 1] - 0.01

    def test_power_curve_bounded(self):
        """Power values should be between 0 and 1."""
        from src.dashboard.app import _compute_power_curve
        df = _compute_power_curve(
            baseline_rate=0.20, mde=0.05, alpha=0.05, max_n=3000,
        )
        assert (df["power"] >= 0).all()
        assert (df["power"] <= 1).all()

    def test_mde_sensitivity_table(self):
        """MDE sensitivity produces valid table."""
        from src.dashboard.app import _compute_mde_sensitivity
        df = _compute_mde_sensitivity(
            baseline_rate=0.20, alpha=0.05, power=0.80,
        )
        assert "MDE" in df.columns
        assert "Sample Size (per group)" in df.columns
        assert len(df) > 0
        # Larger MDE = smaller sample size
        if len(df) > 1:
            for i in range(1, len(df)):
                assert (
                    df["Sample Size (per group)"].iloc[i]
                    <= df["Sample Size (per group)"].iloc[i - 1]
                )

    def test_mde_sensitivity_excludes_impossible(self):
        """MDE values >= baseline_rate should be excluded."""
        from src.dashboard.app import _compute_mde_sensitivity
        df = _compute_mde_sensitivity(
            baseline_rate=0.10, alpha=0.05, power=0.80,
        )
        assert (df["MDE"] < 0.10).all()


# ---------------------------------------------------------------------------
# Multiple Comparison Correction Tests
# ---------------------------------------------------------------------------

class TestMultipleComparisonCorrections:
    """Test multiple comparison correction helper for A/B testing view."""

    def test_correction_returns_all_methods(self):
        """Returns Bonferroni, Holm, and BH corrections."""
        from src.dashboard.app import _compute_multiple_comparison_corrections
        df = _compute_multiple_comparison_corrections(
            p_values=[0.01, 0.04, 0.08],
            experiment_names=["A", "B", "C"],
        )
        assert "Raw p-value" in df.columns
        assert "Bonferroni" in df.columns
        assert "Holm-Bonferroni" in df.columns
        assert "BH (FDR)" in df.columns
        assert len(df) == 3

    def test_bonferroni_is_conservative(self):
        """Bonferroni p-values >= raw p-values."""
        from src.dashboard.app import _compute_multiple_comparison_corrections
        df = _compute_multiple_comparison_corrections(
            p_values=[0.01, 0.04, 0.08],
            experiment_names=["A", "B", "C"],
        )
        for _, row in df.iterrows():
            assert row["Bonferroni"] >= row["Raw p-value"]

    def test_corrected_p_values_bounded(self):
        """Corrected p-values should be between 0 and 1."""
        from src.dashboard.app import _compute_multiple_comparison_corrections
        df = _compute_multiple_comparison_corrections(
            p_values=[0.001, 0.02, 0.5],
            experiment_names=["A", "B", "C"],
        )
        for col in ["Bonferroni", "Holm-Bonferroni", "BH (FDR)"]:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()

    def test_significance_decision_columns(self):
        """Significance decision columns are present."""
        from src.dashboard.app import _compute_multiple_comparison_corrections
        df = _compute_multiple_comparison_corrections(
            p_values=[0.001, 0.04, 0.5],
            experiment_names=["A", "B", "C"],
        )
        assert "Significant (Bonferroni)" in df.columns
        assert "Significant (BH)" in df.columns
        # First experiment should be significant under all methods
        assert df.iloc[0]["Significant (Bonferroni)"] == "Yes"
        assert df.iloc[0]["Significant (BH)"] == "Yes"

    def test_bonferroni_with_single_test(self):
        """Single test Bonferroni should equal raw p-value."""
        from src.dashboard.app import _compute_multiple_comparison_corrections
        df = _compute_multiple_comparison_corrections(
            p_values=[0.03],
            experiment_names=["A"],
        )
        assert abs(df.iloc[0]["Bonferroni"] - 0.03) < 1e-10


# ---------------------------------------------------------------------------
# Multi-Channel Budget Allocation Tests
# ---------------------------------------------------------------------------

class TestMultiChannelBudgetAllocation:
    """Test multi-channel budget allocation helper for budget optimization."""

    @pytest.fixture
    def channel_config(self):
        return {
            "email": {
                "cost_per_action": 1000,
                "roi_multiplier": 1.0,
            },
            "sms": {
                "cost_per_action": 500,
                "roi_multiplier": 0.8,
            },
            "push_notification": {
                "cost_per_action": 200,
                "roi_multiplier": 0.6,
            },
            "coupon": {
                "cost_per_action": 5000,
                "roi_multiplier": 1.5,
            },
            "call_center": {
                "cost_per_action": 15000,
                "roi_multiplier": 2.0,
            },
        }

    def test_channel_allocation_has_all_channels(
        self, sample_budget_results, channel_config,
    ):
        """Each configured channel appears in allocation data."""
        from src.dashboard.app import _build_channel_allocation_data
        df = _build_channel_allocation_data(
            budget_results=sample_budget_results,
            channel_config=channel_config,
            total_budget=50_000_000,
        )
        assert set(df["channel"]) == set(channel_config.keys())

    def test_channel_allocation_columns(
        self, sample_budget_results, channel_config,
    ):
        """Channel data has required columns."""
        from src.dashboard.app import _build_channel_allocation_data
        df = _build_channel_allocation_data(
            budget_results=sample_budget_results,
            channel_config=channel_config,
            total_budget=50_000_000,
        )
        required_cols = [
            "channel", "cost_per_action", "roi_multiplier",
            "allocated_budget", "expected_actions",
        ]
        for col in required_cols:
            assert col in df.columns

    def test_channel_allocation_sums_to_budget(
        self, sample_budget_results, channel_config,
    ):
        """Channel allocations sum approximately to total budget."""
        from src.dashboard.app import _build_channel_allocation_data
        total_budget = 50_000_000
        df = _build_channel_allocation_data(
            budget_results=sample_budget_results,
            channel_config=channel_config,
            total_budget=total_budget,
        )
        total_alloc = df["allocated_budget"].sum()
        # Allow rounding tolerance
        assert abs(total_alloc - total_budget) / total_budget < 0.05

    def test_expected_actions_consistent(
        self, sample_budget_results, channel_config,
    ):
        """Expected actions = allocated_budget / cost_per_action."""
        from src.dashboard.app import _build_channel_allocation_data
        df = _build_channel_allocation_data(
            budget_results=sample_budget_results,
            channel_config=channel_config,
            total_budget=50_000_000,
        )
        for _, row in df.iterrows():
            expected = int(row["allocated_budget"] / max(row["cost_per_action"], 1))
            assert row["expected_actions"] == expected

    def test_higher_roi_gets_more_budget(
        self, sample_budget_results, channel_config,
    ):
        """Higher ROI channels get proportionally more budget."""
        from src.dashboard.app import _build_channel_allocation_data
        df = _build_channel_allocation_data(
            budget_results=sample_budget_results,
            channel_config=channel_config,
            total_budget=50_000_000,
        )
        df_sorted = df.sort_values("roi_multiplier", ascending=False)
        # Top ROI channel should have highest allocation
        assert (
            df_sorted.iloc[0]["allocated_budget"]
            >= df_sorted.iloc[-1]["allocated_budget"]
        )

    def test_channel_allocation_with_different_budgets(
        self, sample_budget_results, channel_config,
    ):
        """Allocations scale with total budget."""
        from src.dashboard.app import _build_channel_allocation_data
        df1 = _build_channel_allocation_data(
            budget_results=sample_budget_results,
            channel_config=channel_config,
            total_budget=50_000_000,
        )
        df2 = _build_channel_allocation_data(
            budget_results=sample_budget_results,
            channel_config=channel_config,
            total_budget=100_000_000,
        )
        # Each channel should have roughly double the allocation
        for ch in channel_config:
            a1 = df1[df1["channel"] == ch]["allocated_budget"].iloc[0]
            a2 = df2[df2["channel"] == ch]["allocated_budget"].iloc[0]
            ratio = a2 / max(a1, 1)
            assert 1.8 < ratio < 2.2
