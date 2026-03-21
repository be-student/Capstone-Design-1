"""
TDD Tests for Churn Prediction Analytics and Cohort Analysis Dashboard Views.

Tests cover:
- Churn analytics view data preparation (risk scores, feature importance, model predictions)
- Cohort analysis visualization data (retention matrix, cohort sizes, retention curves)
- Integration between dashboard data loader and new views
- Render function existence and callability
- Data validation for new views
- Configurable parameters from YAML
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

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
def dashboard_data_loader(config):
    """Create a DashboardDataLoader instance."""
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


@pytest.fixture
def sample_predictions():
    """Create sample churn prediction data."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_probability": np.random.beta(2, 5, n),
        "risk_level": np.random.choice(
            ["low", "medium", "high", "critical"],
            n, p=[0.4, 0.3, 0.2, 0.1],
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
def sample_feature_importance():
    """Create sample feature importance data."""
    features = [
        "days_since_last_purchase", "days_since_last_login",
        "purchase_frequency", "avg_order_value", "total_revenue",
        "page_views_30d", "cart_abandonment_rate", "coupon_usage_rate",
        "customer_tenure_days", "review_count",
    ]
    np.random.seed(42)
    importances = np.sort(np.random.dirichlet(np.ones(len(features))))[::-1]
    return pd.DataFrame({
        "feature": features,
        "importance": importances,
    })


@pytest.fixture
def sample_retention_matrix():
    """Create sample cohort retention matrix."""
    np.random.seed(42)
    cohorts = [f"2024-{m:02d}" for m in range(1, 7)]
    periods = list(range(7))

    data = {}
    for cohort in cohorts:
        retention = [1.0]
        for p in range(1, len(periods)):
            decay = np.random.uniform(0.80, 0.95)
            retention.append(round(retention[-1] * decay, 4))
        data[cohort] = retention

    df = pd.DataFrame(data, index=periods).T
    df.index.name = "cohort"
    df.columns = periods
    return df


@pytest.fixture
def sample_cohort_data():
    """Create sample cohort event data."""
    np.random.seed(42)
    n_customers = 100
    customers = [f"C{i:05d}" for i in range(n_customers)]
    rows = []
    base_date = pd.Timestamp("2024-01-01")

    for cust in customers:
        first_offset = np.random.randint(0, 180)
        first_date = base_date + pd.Timedelta(days=int(first_offset))
        segment = np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter"]
        )
        for j in range(np.random.randint(1, 4)):
            event_offset = np.random.randint(0, 365)
            event_date = first_date + pd.Timedelta(days=int(event_offset))
            revenue = np.random.lognormal(9, 1)
            rows.append({
                "customer_id": cust,
                "event_date": event_date,
                "revenue": revenue,
                "segment": segment,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Churn Analytics Page Tests
# ---------------------------------------------------------------------------

class TestChurnAnalyticsPageExists:
    """Test that the churn analytics page exists in the dashboard."""

    def test_render_churn_analytics_exists(self):
        """Dashboard must include a render_churn_analytics function."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_churn_analytics")
        assert callable(dashboard_app.render_churn_analytics)

    def test_churn_analytics_in_page_list(self):
        """Churn Analytics must be in the navigation page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "Churn Analytics" in pages

    def test_churn_analytics_has_icon(self):
        """Churn Analytics page must have an icon."""
        from src.dashboard.utils.dashboard_helpers import get_page_icon
        icon = get_page_icon("Churn Analytics")
        assert isinstance(icon, str)
        assert len(icon) > 0


class TestChurnRiskScores:
    """Test churn risk score computation for analytics view."""

    def test_risk_score_distribution(self, sample_predictions):
        """Churn risk scores must have valid distribution stats."""
        probs = sample_predictions["churn_probability"]
        assert probs.mean() > 0
        assert probs.mean() < 1
        assert probs.std() > 0

    def test_risk_level_classification(self, sample_predictions):
        """Risk levels must classify all customers."""
        risk_counts = sample_predictions["risk_level"].value_counts()
        assert risk_counts.sum() == len(sample_predictions)

    def test_risk_level_percentages(self, sample_predictions):
        """Risk level percentages must sum to 100%."""
        risk_counts = sample_predictions["risk_level"].value_counts()
        pcts = risk_counts / risk_counts.sum() * 100
        assert abs(pcts.sum() - 100.0) < 0.01

    def test_high_risk_count_computable(self, sample_predictions):
        """High risk and critical customer counts must be computable."""
        high_risk = (sample_predictions["churn_probability"] > 0.5).sum()
        critical = (sample_predictions["churn_probability"] > 0.75).sum()
        assert high_risk >= critical
        assert critical >= 0

    def test_risk_threshold_filtering(self, sample_predictions):
        """Must be able to filter by custom risk thresholds."""
        for threshold in [0.25, 0.5, 0.75]:
            filtered = sample_predictions[
                sample_predictions["churn_probability"] >= threshold
            ]
            assert isinstance(filtered, pd.DataFrame)
            assert len(filtered) <= len(sample_predictions)


class TestFeatureImportanceAnalytics:
    """Test feature importance data for analytics view."""

    def test_feature_importance_loadable(self, dashboard_data_loader):
        """Feature importance must be loadable from data loader."""
        fi = dashboard_data_loader.load_feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert not fi.empty

    def test_feature_importance_sorted(self, sample_feature_importance):
        """Features must be sorted by importance descending."""
        assert sample_feature_importance["importance"].is_monotonic_decreasing

    def test_cumulative_importance_computable(self, sample_feature_importance):
        """Cumulative importance must be computable."""
        cum = sample_feature_importance["importance"].cumsum()
        total = sample_feature_importance["importance"].sum()
        cum_pct = cum / total * 100
        assert cum_pct.iloc[-1] == pytest.approx(100.0, abs=0.01)
        assert cum_pct.is_monotonic_increasing

    def test_top_features_identifiable(self, sample_feature_importance):
        """Must identify top-N most important features."""
        top_5 = sample_feature_importance.head(5)
        assert len(top_5) == 5
        assert top_5["importance"].iloc[0] >= top_5["importance"].iloc[-1]


class TestSegmentChurnAnalysis:
    """Test segment-level churn analysis for analytics view."""

    def test_segment_aggregation(self, sample_predictions):
        """Segment-level churn stats must be computable."""
        seg_analysis = sample_predictions.groupby("segment").agg(
            customer_count=("customer_id", "count"),
            avg_churn=("churn_probability", "mean"),
            median_churn=("churn_probability", "median"),
        ).reset_index()
        assert len(seg_analysis) > 0
        assert "avg_churn" in seg_analysis.columns
        assert all(0 <= v <= 1 for v in seg_analysis["avg_churn"])

    def test_segment_high_risk_count(self, sample_predictions):
        """High risk count per segment must be computable."""
        seg_hr = sample_predictions.groupby("segment")["churn_probability"].agg(
            lambda x: (x > 0.5).sum()
        )
        assert all(v >= 0 for v in seg_hr)

    def test_churn_vs_clv_correlation(self, sample_predictions):
        """Churn vs CLV scatter data must be preparable."""
        assert "churn_probability" in sample_predictions.columns
        assert "clv_predicted" in sample_predictions.columns
        # Both columns exist and have data
        assert len(sample_predictions) > 0


class TestModelPredictionSummary:
    """Test model prediction summary for analytics view."""

    def test_model_metrics_loadable(self, dashboard_data_loader):
        """Model metrics must be loadable."""
        metrics = dashboard_data_loader.load_model_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_model_metrics_has_required_keys(self, dashboard_data_loader):
        """Model metrics must have AUC, precision, recall, F1 for each model."""
        metrics = dashboard_data_loader.load_model_metrics()
        required = {"auc", "precision", "recall", "f1_score"}
        for model_name, model_metrics in metrics.items():
            for key in required:
                assert key in model_metrics, f"Missing {key} for {model_name}"


# ---------------------------------------------------------------------------
# Cohort Analysis Page Tests
# ---------------------------------------------------------------------------

class TestCohortAnalysisPageExists:
    """Test that the cohort analysis page exists in the dashboard."""

    def test_render_cohort_analysis_exists(self):
        """Dashboard must include a render_cohort_analysis function."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_cohort_analysis")
        assert callable(dashboard_app.render_cohort_analysis)

    def test_cohort_analysis_in_page_list(self):
        """Cohort Analysis must be in the navigation page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "Cohort Analysis" in pages

    def test_cohort_analysis_has_icon(self):
        """Cohort Analysis page must have an icon."""
        from src.dashboard.utils.dashboard_helpers import get_page_icon
        icon = get_page_icon("Cohort Analysis")
        assert isinstance(icon, str)
        assert len(icon) > 0


class TestCohortDataLoader:
    """Test cohort data loading for dashboard."""

    def test_has_load_cohort_data_method(self, dashboard_data_loader):
        """Data loader must have load_cohort_data method."""
        assert hasattr(dashboard_data_loader, "load_cohort_data")
        assert callable(dashboard_data_loader.load_cohort_data)

    def test_cohort_data_returns_dataframe(self, dashboard_data_loader):
        """load_cohort_data must return a DataFrame."""
        data = dashboard_data_loader.load_cohort_data()
        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_cohort_data_has_required_columns(self, dashboard_data_loader):
        """Cohort data must have customer_id, event_date, revenue columns."""
        data = dashboard_data_loader.load_cohort_data()
        required = {"customer_id", "event_date", "revenue"}
        assert required.issubset(set(data.columns))

    def test_has_load_retention_matrix_method(self, dashboard_data_loader):
        """Data loader must have load_cohort_retention_matrix method."""
        assert hasattr(dashboard_data_loader, "load_cohort_retention_matrix")
        assert callable(dashboard_data_loader.load_cohort_retention_matrix)

    def test_retention_matrix_returns_dataframe(self, dashboard_data_loader):
        """load_cohort_retention_matrix must return a DataFrame."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        assert isinstance(matrix, pd.DataFrame)
        assert not matrix.empty


class TestRetentionMatrix:
    """Test retention matrix properties for visualization."""

    def test_retention_values_between_0_and_1(self, sample_retention_matrix):
        """All retention values must be between 0 and 1."""
        assert (sample_retention_matrix >= 0).all().all()
        assert (sample_retention_matrix <= 1).all().all()

    def test_period_0_is_100_percent(self, sample_retention_matrix):
        """Period 0 retention must always be 1.0 (100%)."""
        if 0 in sample_retention_matrix.columns:
            assert (sample_retention_matrix[0] == 1.0).all()

    def test_retention_generally_decreasing(self, sample_retention_matrix):
        """Retention should generally decrease over periods."""
        for _, row in sample_retention_matrix.iterrows():
            values = row.dropna().values
            # Overall: last value should be <= first value
            assert values[-1] <= values[0]

    def test_multiple_cohorts_present(self, sample_retention_matrix):
        """Retention matrix must have multiple cohorts."""
        assert len(sample_retention_matrix) >= 2

    def test_multiple_periods_present(self, sample_retention_matrix):
        """Retention matrix must have multiple periods."""
        assert len(sample_retention_matrix.columns) >= 2

    def test_heatmap_data_preparable(self, sample_retention_matrix):
        """Retention data must be convertible to heatmap format."""
        heatmap_data = sample_retention_matrix * 100
        assert heatmap_data.values.max() <= 100
        assert heatmap_data.values.min() >= 0


class TestRetentionCurves:
    """Test retention curve data for visualization."""

    def test_average_retention_curve(self, sample_retention_matrix):
        """Average retention curve must be computable."""
        avg = sample_retention_matrix.mean(axis=0)
        assert len(avg) == len(sample_retention_matrix.columns)
        assert all(0 <= v <= 1 for v in avg)

    def test_per_cohort_curves(self, sample_retention_matrix):
        """Each cohort must have a plottable retention curve."""
        for cohort in sample_retention_matrix.index:
            curve = sample_retention_matrix.loc[cohort].dropna()
            assert len(curve) > 0
            assert all(0 <= v <= 1 for v in curve)

    def test_period_over_period_drops(self, sample_retention_matrix):
        """Period-over-period retention drops must be computable."""
        avg = sample_retention_matrix.mean(axis=0) * 100
        if len(avg) > 1:
            drops = [
                avg.iloc[i] - avg.iloc[i - 1]
                for i in range(1, len(avg))
            ]
            # Most drops should be negative (retention decreasing)
            negative_drops = sum(1 for d in drops if d < 0)
            assert negative_drops > 0


class TestCohortSizes:
    """Test cohort size data for visualization."""

    def test_cohort_size_from_event_data(self, sample_cohort_data):
        """Cohort sizes must be derivable from event data."""
        sample_cohort_data["event_date"] = pd.to_datetime(
            sample_cohort_data["event_date"]
        )
        first_event = sample_cohort_data.groupby("customer_id")[
            "event_date"
        ].min().reset_index()
        first_event["cohort"] = (
            first_event["event_date"].dt.to_period("M").astype(str)
        )
        cohort_sizes = first_event["cohort"].value_counts().sort_index()
        assert len(cohort_sizes) > 0
        assert all(v > 0 for v in cohort_sizes)


class TestCohortRetentionIntegration:
    """Test integration between CohortAnalyzer and dashboard."""

    def test_analyzer_produces_valid_retention(self, sample_cohort_data):
        """CohortAnalyzer must produce a valid retention matrix for display."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(
            sample_cohort_data, cohort_type="monthly"
        )
        retention = analyzer.compute_retention_matrix(assigned)
        assert isinstance(retention, pd.DataFrame)
        # All values should be between 0 and 1
        assert (retention >= 0).all().all()
        assert (retention <= 1.0 + 1e-9).all().all()


# ---------------------------------------------------------------------------
# Page Navigation Tests
# ---------------------------------------------------------------------------

class TestPageNavigation:
    """Test page navigation includes new views."""

    def test_page_list_includes_churn_analytics(self):
        """Page list must include Churn Analytics."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Churn Analytics" in PAGES

    def test_page_list_includes_cohort_analysis(self):
        """Page list must include Cohort Analysis."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Cohort Analysis" in PAGES

    def test_page_map_in_main(self):
        """Main page map must include new pages."""
        from src.dashboard import app as dashboard_app
        # Verify the render functions exist and would be in the page map
        assert hasattr(dashboard_app, "render_churn_analytics")
        assert hasattr(dashboard_app, "render_cohort_analysis")

    def test_total_page_count(self):
        """Total page count should be at least 13 (original 12 + 2 new)."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert len(PAGES) >= 13


# ---------------------------------------------------------------------------
# Render Function Smoke Tests (with mock Streamlit)
# ---------------------------------------------------------------------------

class TestRenderChurnAnalyticsMock:
    """Test render_churn_analytics with mocked Streamlit."""

    def test_render_with_valid_data(self, config, dashboard_data_loader):
        """render_churn_analytics must not raise with valid data."""
        from src.dashboard.app import render_churn_analytics

        st_mock = MagicMock()
        # Return correct number of columns for each call
        st_mock.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        st_mock.slider.return_value = 0.5

        # Should not raise
        render_churn_analytics(st_mock, config, dashboard_data_loader)

        # Verify it called st.header
        st_mock.header.assert_called_once_with("Churn Prediction Analytics")

    def test_render_with_empty_data(self, config):
        """render_churn_analytics must handle empty predictions gracefully."""
        from src.dashboard.app import render_churn_analytics

        st_mock = MagicMock()
        loader_mock = MagicMock()
        loader_mock.load_predictions.return_value = pd.DataFrame()

        render_churn_analytics(st_mock, config, loader_mock)
        st_mock.warning.assert_called()


class TestRenderCohortAnalysisMock:
    """Test render_cohort_analysis with mocked Streamlit."""

    def test_render_with_valid_data(self, config, sample_retention_matrix):
        """render_cohort_analysis must not raise with valid data."""
        from src.dashboard.app import render_cohort_analysis

        st_mock = MagicMock()
        st_mock.columns.return_value = [MagicMock() for _ in range(4)]

        loader_mock = MagicMock()
        loader_mock.load_cohort_retention_matrix.return_value = (
            sample_retention_matrix
        )
        loader_mock.load_cohort_data.return_value = pd.DataFrame({
            "customer_id": ["C00001", "C00002"],
            "event_date": pd.to_datetime(["2024-01-15", "2024-02-20"]),
            "revenue": [10000, 20000],
            "segment": ["vip_loyal", "regular_loyal"],
        })

        render_cohort_analysis(st_mock, config, loader_mock)
        st_mock.header.assert_called_once_with("Cohort Analysis")

    def test_render_with_empty_matrix(self, config):
        """render_cohort_analysis must handle empty matrix gracefully."""
        from src.dashboard.app import render_cohort_analysis

        st_mock = MagicMock()
        loader_mock = MagicMock()
        loader_mock.load_cohort_retention_matrix.return_value = pd.DataFrame()

        render_cohort_analysis(st_mock, config, loader_mock)
        st_mock.warning.assert_called()


# ---------------------------------------------------------------------------
# Data Loader Enhanced Tests
# ---------------------------------------------------------------------------

class TestDataLoaderCohortMethods:
    """Test data loader cohort-related methods return valid data."""

    def test_cohort_data_event_dates_valid(self, dashboard_data_loader):
        """Cohort event dates must be valid datetime."""
        data = dashboard_data_loader.load_cohort_data()
        assert pd.api.types.is_datetime64_any_dtype(data["event_date"])

    def test_cohort_data_revenue_positive(self, dashboard_data_loader):
        """Cohort revenue values must be positive."""
        data = dashboard_data_loader.load_cohort_data()
        assert (data["revenue"] > 0).all()

    def test_cohort_data_has_customers(self, dashboard_data_loader):
        """Cohort data must have multiple customers."""
        data = dashboard_data_loader.load_cohort_data()
        n_customers = data["customer_id"].nunique()
        assert n_customers >= 10

    def test_retention_matrix_valid_shape(self, dashboard_data_loader):
        """Retention matrix must have valid shape (rows x cols)."""
        matrix = dashboard_data_loader.load_cohort_retention_matrix()
        assert matrix.shape[0] >= 2  # At least 2 cohorts
        assert matrix.shape[1] >= 2  # At least 2 periods
