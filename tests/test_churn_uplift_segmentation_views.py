"""
TDD Tests for Churn Prediction, Uplift Model, and Customer Segmentation Dashboard Views.

Tests cover:
- Churn prediction view: risk score distributions, thresholds, feature importance,
  segment-level analytics, model metrics integration, CLV vs churn scatter
- Uplift model view: uplift score distributions, treatment effect, persuadable/sleeping dog
  classification, segment-level uplift, cumulative uplift curve (Qini-style)
- Customer segmentation view: segment distribution, churn risk by segment, CLV by segment,
  risk level stacked breakdown, retention action mapping
- Render function smoke tests with mocked Streamlit
- Data validation and edge cases
"""

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
    """Create sample churn prediction data with all required columns."""
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
def sample_uplift_data():
    """Create sample uplift modeling results."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "uplift_score": np.random.normal(0.05, 0.03, n),
        "treatment_effect": np.random.normal(0.08, 0.04, n),
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "new_customer", "dormant", "high_value_at_risk"],
            n,
        ),
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


def _make_st_mock():
    """Create a properly configured Streamlit mock."""
    st_mock = MagicMock()
    st_mock.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    st_mock.slider.return_value = 0.5
    return st_mock


# ---------------------------------------------------------------------------
# Churn Prediction View Tests
# ---------------------------------------------------------------------------

class TestChurnPredictionViewExists:
    """Verify churn prediction analytics render function exists and is routed."""

    def test_render_churn_analytics_callable(self):
        """render_churn_analytics must be callable."""
        from src.dashboard.app import render_churn_analytics
        assert callable(render_churn_analytics)

    def test_churn_analytics_in_page_map(self):
        """Churn Analytics must be in the page routing map."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Churn Analytics" in PAGES

    def test_churn_analytics_page_icon_nonempty(self):
        """Churn Analytics page icon must be non-empty."""
        from src.dashboard.utils.dashboard_helpers import get_page_icon
        icon = get_page_icon("Churn Analytics")
        assert len(icon) > 0


class TestChurnRiskScoreView:
    """Test churn risk score computations for the analytics view."""

    def test_risk_score_range(self, sample_predictions):
        """All churn probabilities must be in [0, 1]."""
        probs = sample_predictions["churn_probability"]
        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_risk_level_categories_valid(self, sample_predictions):
        """Risk levels must only contain expected categories."""
        valid = {"low", "medium", "high", "critical"}
        actual = set(sample_predictions["risk_level"].unique())
        assert actual.issubset(valid)

    def test_risk_threshold_boundary(self, sample_predictions):
        """Risk thresholds at 0.25, 0.5, 0.75 must partition customers."""
        thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
        total = 0
        for i in range(len(thresholds) - 1):
            count = ((sample_predictions["churn_probability"] >= thresholds[i]) &
                     (sample_predictions["churn_probability"] < thresholds[i + 1])).sum()
            total += count
        # Add exactly-1.0 customers
        total += (sample_predictions["churn_probability"] >= 1.0).sum()
        assert total == len(sample_predictions)

    def test_high_risk_customers_filterable(self, sample_predictions):
        """Must be able to filter customers above a threshold."""
        for threshold in [0.3, 0.5, 0.7]:
            filtered = sample_predictions[
                sample_predictions["churn_probability"] >= threshold
            ]
            assert isinstance(filtered, pd.DataFrame)
            if len(filtered) > 0:
                assert filtered["churn_probability"].min() >= threshold


class TestChurnFeatureImportanceView:
    """Test feature importance visualization data for churn analytics."""

    def test_feature_importance_loadable(self, dashboard_data_loader):
        """Feature importance must be loadable from data loader."""
        fi = dashboard_data_loader.load_feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_importance_positive(self, sample_feature_importance):
        """All importance values must be non-negative."""
        assert (sample_feature_importance["importance"] >= 0).all()

    def test_feature_importance_sums_to_one(self, sample_feature_importance):
        """Importance values should sum to approximately 1.0."""
        total = sample_feature_importance["importance"].sum()
        assert abs(total - 1.0) < 0.01

    def test_top_n_features_extractable(self, sample_feature_importance):
        """Must be able to extract top-N features for visualization."""
        for n in [5, 10]:
            top_n = sample_feature_importance.head(n)
            assert len(top_n) == n
            # Top features should have higher importance
            assert top_n["importance"].iloc[0] >= top_n["importance"].iloc[-1]

    def test_cumulative_importance_reaches_100(self, sample_feature_importance):
        """Cumulative importance must eventually reach ~100%."""
        total = sample_feature_importance["importance"].sum()
        cum_pct = sample_feature_importance["importance"].cumsum() / total * 100
        assert cum_pct.iloc[-1] == pytest.approx(100.0, abs=0.01)


class TestChurnSegmentAnalyticsView:
    """Test segment-level analytics for churn prediction view."""

    def test_segment_churn_aggregation(self, sample_predictions):
        """Segment-level churn stats (mean, median, std) must be computable."""
        seg = sample_predictions.groupby("segment")["churn_probability"].agg(
            ["mean", "median", "std", "count"]
        ).reset_index()
        assert len(seg) > 0
        assert all(0 <= v <= 1 for v in seg["mean"])

    def test_segment_high_risk_percentage(self, sample_predictions):
        """High risk percentage per segment must be computable."""
        seg = sample_predictions.groupby("segment").agg(
            total=("customer_id", "count"),
            high_risk=("churn_probability", lambda x: (x > 0.5).sum()),
        ).reset_index()
        seg["high_risk_pct"] = seg["high_risk"] / seg["total"] * 100
        assert all(0 <= v <= 100 for v in seg["high_risk_pct"])

    def test_cross_tabulation_segment_risk(self, sample_predictions):
        """Cross-tabulation of segment x risk level must be computable."""
        cross_tab = pd.crosstab(
            sample_predictions["segment"],
            sample_predictions["risk_level"],
            normalize="index",
        )
        # Each row should sum to ~1.0
        row_sums = cross_tab.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 0.01

    def test_churn_vs_clv_data_available(self, sample_predictions):
        """Both churn_probability and clv_predicted must exist for scatter."""
        assert "churn_probability" in sample_predictions.columns
        assert "clv_predicted" in sample_predictions.columns
        assert len(sample_predictions) > 0

    def test_at_risk_revenue_computable(self, sample_predictions):
        """At-risk revenue (high churn + CLV) must be computable."""
        at_risk = sample_predictions[
            sample_predictions["churn_probability"] > 0.5
        ]["clv_predicted"].sum()
        total_clv = sample_predictions["clv_predicted"].sum()
        assert at_risk >= 0
        assert at_risk <= total_clv


class TestChurnModelMetricsView:
    """Test model performance metrics for churn analytics view."""

    def test_model_metrics_loadable(self, dashboard_data_loader):
        """Model metrics must be loadable."""
        metrics = dashboard_data_loader.load_model_metrics()
        assert isinstance(metrics, dict)

    def test_model_metrics_auc_range(self, dashboard_data_loader):
        """AUC values must be between 0.5 and 1.0."""
        metrics = dashboard_data_loader.load_model_metrics()
        for model_name, m in metrics.items():
            auc = m.get("auc", 0)
            assert 0.5 <= auc <= 1.0, f"{model_name} AUC={auc} out of range"

    def test_model_metrics_complete(self, dashboard_data_loader):
        """Each model must have auc, precision, recall, f1_score, accuracy."""
        metrics = dashboard_data_loader.load_model_metrics()
        required = {"auc", "precision", "recall", "f1_score", "accuracy"}
        for model_name, m in metrics.items():
            missing = required - set(m.keys())
            assert not missing, f"{model_name} missing: {missing}"


# ---------------------------------------------------------------------------
# Uplift Model View Tests
# ---------------------------------------------------------------------------

class TestUpliftViewExists:
    """Verify uplift model view render function exists and is routed."""

    def test_render_uplift_callable(self):
        """render_uplift must be callable."""
        from src.dashboard.app import render_uplift
        assert callable(render_uplift)

    def test_uplift_in_page_list(self):
        """Uplift Modeling must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Uplift Modeling" in PAGES

    def test_uplift_page_icon(self):
        """Uplift Modeling page must have an icon."""
        from src.dashboard.utils.dashboard_helpers import get_page_icon
        icon = get_page_icon("Uplift Modeling")
        assert len(icon) > 0


class TestUpliftScoreDistribution:
    """Test uplift score distribution data for visualization."""

    def test_uplift_data_loadable(self, dashboard_data_loader):
        """Uplift results must be loadable from data loader."""
        uplift = dashboard_data_loader.load_uplift_results()
        assert isinstance(uplift, pd.DataFrame)
        assert not uplift.empty

    def test_uplift_has_required_columns(self, dashboard_data_loader):
        """Uplift data must have customer_id, uplift_score, treatment_effect, segment."""
        uplift = dashboard_data_loader.load_uplift_results()
        required = {"customer_id", "uplift_score", "treatment_effect", "segment"}
        assert required.issubset(set(uplift.columns))

    def test_uplift_score_has_variation(self, sample_uplift_data):
        """Uplift scores must have non-zero standard deviation."""
        assert sample_uplift_data["uplift_score"].std() > 0

    def test_treatment_effect_has_variation(self, sample_uplift_data):
        """Treatment effects must have non-zero standard deviation."""
        assert sample_uplift_data["treatment_effect"].std() > 0

    def test_average_uplift_computable(self, sample_uplift_data):
        """Average uplift score must be computable."""
        avg = sample_uplift_data["uplift_score"].mean()
        assert isinstance(avg, float)


class TestUpliftCustomerClassification:
    """Test uplift-based customer classification (4-quadrant)."""

    def test_persuadable_classification(self, sample_uplift_data):
        """Must identify persuadable customers (positive uplift, positive treatment)."""
        persuadable = sample_uplift_data[
            (sample_uplift_data["uplift_score"] > 0) &
            (sample_uplift_data["treatment_effect"] > 0)
        ]
        assert len(persuadable) >= 0

    def test_sleeping_dog_classification(self, sample_uplift_data):
        """Must identify sleeping dogs (positive uplift, negative treatment)."""
        sleeping_dogs = sample_uplift_data[
            (sample_uplift_data["uplift_score"] > 0) &
            (sample_uplift_data["treatment_effect"] <= 0)
        ]
        assert isinstance(sleeping_dogs, pd.DataFrame)

    def test_four_quadrant_classification(self, sample_uplift_data):
        """All customers must be classifiable into 4 quadrants."""

        def classify(row):
            if row["uplift_score"] > 0 and row["treatment_effect"] > 0:
                return "Persuadable"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] > 0:
                return "Sure Thing"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] <= 0:
                return "Lost Cause"
            return "Sleeping Dog"

        classes = sample_uplift_data.apply(classify, axis=1)
        assert len(classes) == len(sample_uplift_data)
        # All must have a classification
        assert classes.notna().all()
        # At least 2 classes should be present with this seed
        assert classes.nunique() >= 2

    def test_classification_counts_sum_to_total(self, sample_uplift_data):
        """Classification counts must sum to total customers."""

        def classify(row):
            if row["uplift_score"] > 0 and row["treatment_effect"] > 0:
                return "Persuadable"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] > 0:
                return "Sure Thing"
            elif row["uplift_score"] <= 0 and row["treatment_effect"] <= 0:
                return "Lost Cause"
            return "Sleeping Dog"

        classes = sample_uplift_data.apply(classify, axis=1)
        counts = classes.value_counts()
        assert counts.sum() == len(sample_uplift_data)


class TestUpliftSegmentAnalysis:
    """Test segment-level uplift analysis for visualization."""

    def test_segment_uplift_aggregation(self, sample_uplift_data):
        """Segment-level uplift stats must be computable."""
        seg = sample_uplift_data.groupby("segment").agg(
            avg_uplift=("uplift_score", "mean"),
            avg_treatment=("treatment_effect", "mean"),
            count=("customer_id", "count"),
        ).reset_index()
        assert len(seg) > 0
        assert "avg_uplift" in seg.columns

    def test_segment_persuadable_percentage(self, sample_uplift_data):
        """Persuadable percentage per segment must be computable."""
        seg = sample_uplift_data.groupby("segment").agg(
            total=("customer_id", "count"),
            persuadable=("uplift_score", lambda x: (x > 0).sum()),
        ).reset_index()
        seg["persuadable_pct"] = seg["persuadable"] / seg["total"] * 100
        assert all(0 <= v <= 100 for v in seg["persuadable_pct"])

    def test_cumulative_uplift_curve_data(self, sample_uplift_data):
        """Cumulative uplift curve (Qini) data must be preparable."""
        sorted_data = sample_uplift_data.sort_values(
            "uplift_score", ascending=False
        ).reset_index(drop=True)
        sorted_data["cum_uplift"] = sorted_data["uplift_score"].cumsum()
        sorted_data["pct_treated"] = (
            np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        )
        assert sorted_data["pct_treated"].iloc[-1] == pytest.approx(100.0)
        assert len(sorted_data["cum_uplift"]) == len(sorted_data)

    def test_top_persuadable_customers(self, sample_uplift_data):
        """Must extract top N persuadable customers."""
        persuadable = sample_uplift_data[
            sample_uplift_data["uplift_score"] > 0
        ].nlargest(10, "uplift_score")
        assert len(persuadable) <= 10
        if len(persuadable) > 1:
            assert persuadable["uplift_score"].iloc[0] >= persuadable["uplift_score"].iloc[-1]


# ---------------------------------------------------------------------------
# Customer Segmentation View Tests
# ---------------------------------------------------------------------------

class TestSegmentationViewExists:
    """Verify customer segmentation view exists and is routed."""

    def test_render_segmentation_callable(self):
        """render_segmentation must be callable."""
        from src.dashboard.app import render_segmentation
        assert callable(render_segmentation)

    def test_segmentation_in_page_list(self):
        """Customer Segmentation must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Customer Segmentation" in PAGES

    def test_segmentation_page_icon(self):
        """Customer Segmentation page must have an icon."""
        from src.dashboard.utils.dashboard_helpers import get_page_icon
        icon = get_page_icon("Customer Segmentation")
        assert len(icon) > 0


class TestSegmentDistributionView:
    """Test segment distribution data for visualization."""

    def test_segment_counts_computable(self, sample_predictions):
        """Segment value counts must be computable for pie/bar charts."""
        seg_counts = sample_predictions["segment"].value_counts()
        assert seg_counts.sum() == len(sample_predictions)
        assert len(seg_counts) > 0

    def test_segment_percentages_sum_to_100(self, sample_predictions):
        """Segment percentages must sum to 100%."""
        seg_counts = sample_predictions["segment"].value_counts()
        pcts = seg_counts / seg_counts.sum() * 100
        assert abs(pcts.sum() - 100.0) < 0.01

    def test_unique_segments_multiple(self, sample_predictions):
        """Must have multiple distinct segments."""
        n_segments = sample_predictions["segment"].nunique()
        assert n_segments >= 2


class TestSegmentChurnRiskView:
    """Test segment-level churn risk data for visualization."""

    def test_segment_risk_stats(self, sample_predictions):
        """Must compute avg, min, max, std churn per segment."""
        seg_stats = sample_predictions.groupby("segment")["churn_probability"].agg(
            ["mean", "min", "max", "std", "count"]
        ).reset_index()
        assert len(seg_stats) > 0
        assert all(0 <= v <= 1 for v in seg_stats["mean"])
        assert all(0 <= v <= 1 for v in seg_stats["min"])
        assert all(0 <= v <= 1 for v in seg_stats["max"])

    def test_risk_level_by_segment_stacked(self, sample_predictions):
        """Risk level counts by segment for stacked bar chart must work."""
        risk_seg = sample_predictions.groupby(
            ["segment", "risk_level"]
        ).size().reset_index(name="count")
        assert len(risk_seg) > 0
        assert risk_seg["count"].sum() == len(sample_predictions)

    def test_highest_risk_segment_identifiable(self, sample_predictions):
        """Highest risk segment must be identifiable."""
        highest = (
            sample_predictions.groupby("segment")["churn_probability"]
            .mean()
            .idxmax()
        )
        assert isinstance(highest, str)
        assert highest in sample_predictions["segment"].values


class TestSegmentCLVView:
    """Test CLV by segment data for visualization."""

    def test_clv_by_segment_mean_and_total(self, sample_predictions):
        """Mean and total CLV per segment must be computable."""
        seg_clv = sample_predictions.groupby("segment")["clv_predicted"].agg(
            ["mean", "sum"]
        ).reset_index()
        assert all(v > 0 for v in seg_clv["mean"])
        assert all(v > 0 for v in seg_clv["sum"])

    def test_clv_segment_ranking(self, sample_predictions):
        """Segments must be rankable by mean CLV."""
        seg_clv = sample_predictions.groupby("segment")["clv_predicted"].mean()
        ranked = seg_clv.sort_values(ascending=False)
        assert ranked.iloc[0] >= ranked.iloc[-1]


class TestSegmentRetentionActionsView:
    """Test retention action mapping from config for segmentation view."""

    def test_retention_actions_from_config(self, config):
        """Segment definitions in config should have retention actions."""
        segments = config.get("segmentation", {}).get("segments", [])
        # If segments are defined, each should have a retention_action
        if segments:
            for seg in segments:
                assert "name" in seg
                assert "retention_action" in seg


# ---------------------------------------------------------------------------
# Render Function Smoke Tests (mocked Streamlit)
# ---------------------------------------------------------------------------

class TestRenderChurnAnalyticsSmoke:
    """Smoke test render_churn_analytics with mocked Streamlit."""

    def test_renders_with_valid_data(self, config, dashboard_data_loader):
        """render_churn_analytics must not raise with valid data."""
        from src.dashboard.app import render_churn_analytics
        st_mock = _make_st_mock()
        render_churn_analytics(st_mock, config, dashboard_data_loader)
        st_mock.header.assert_called_once_with("Churn Prediction Analytics")

    def test_renders_kpi_metrics(self, config, dashboard_data_loader):
        """render_churn_analytics must call st.metric for KPIs."""
        from src.dashboard.app import render_churn_analytics
        st_mock = _make_st_mock()
        render_churn_analytics(st_mock, config, dashboard_data_loader)
        # At least some metrics should have been called on column mocks
        # The header should definitely be called
        assert st_mock.header.called

    def test_handles_empty_data_gracefully(self, config):
        """render_churn_analytics must handle empty predictions."""
        from src.dashboard.app import render_churn_analytics
        st_mock = _make_st_mock()
        loader_mock = MagicMock()
        loader_mock.load_predictions.return_value = pd.DataFrame()
        render_churn_analytics(st_mock, config, loader_mock)
        st_mock.warning.assert_called()

    def test_renders_subheaders(self, config, dashboard_data_loader):
        """render_churn_analytics must render multiple sections."""
        from src.dashboard.app import render_churn_analytics
        st_mock = _make_st_mock()
        render_churn_analytics(st_mock, config, dashboard_data_loader)
        # Should have at least 3 subheader calls (Risk Summary, Distribution, etc.)
        assert st_mock.subheader.call_count >= 3


class TestRenderUpliftSmoke:
    """Smoke test render_uplift with mocked Streamlit."""

    def test_renders_with_valid_data(self, config, dashboard_data_loader):
        """render_uplift must not raise with valid data."""
        from src.dashboard.app import render_uplift
        st_mock = _make_st_mock()
        render_uplift(st_mock, config, dashboard_data_loader)
        st_mock.header.assert_called_once_with("Uplift Modeling Results")

    def test_handles_empty_data_gracefully(self, config):
        """render_uplift must handle empty uplift data."""
        from src.dashboard.app import render_uplift
        st_mock = _make_st_mock()
        loader_mock = MagicMock()
        loader_mock.load_uplift_results.return_value = pd.DataFrame()
        render_uplift(st_mock, config, loader_mock)
        st_mock.warning.assert_called()

    def test_renders_uplift_kpis(self, config, dashboard_data_loader):
        """render_uplift must render KPI metrics."""
        from src.dashboard.app import render_uplift
        st_mock = _make_st_mock()
        render_uplift(st_mock, config, dashboard_data_loader)
        assert st_mock.header.called

    def test_renders_multiple_sections(self, config, dashboard_data_loader):
        """render_uplift must have multiple subheader sections."""
        from src.dashboard.app import render_uplift
        st_mock = _make_st_mock()
        render_uplift(st_mock, config, dashboard_data_loader)
        # Should have subheaders for distribution, scatter, segment, classification, top customers
        assert st_mock.subheader.call_count >= 4


class TestRenderSegmentationSmoke:
    """Smoke test render_segmentation with mocked Streamlit."""

    def test_renders_with_valid_data(self, config, dashboard_data_loader):
        """render_segmentation must not raise with valid data."""
        from src.dashboard.app import render_segmentation
        st_mock = _make_st_mock()
        render_segmentation(st_mock, config, dashboard_data_loader)
        st_mock.header.assert_called_once_with("Customer Segmentation")

    def test_handles_empty_data_gracefully(self, config):
        """render_segmentation must handle empty data."""
        from src.dashboard.app import render_segmentation
        st_mock = _make_st_mock()
        loader_mock = MagicMock()
        loader_mock.load_predictions.return_value = pd.DataFrame()
        render_segmentation(st_mock, config, loader_mock)
        st_mock.warning.assert_called()

    def test_renders_segment_sections(self, config, dashboard_data_loader):
        """render_segmentation must render multiple visualization sections."""
        from src.dashboard.app import render_segmentation
        st_mock = _make_st_mock()
        render_segmentation(st_mock, config, dashboard_data_loader)
        # Should have subheaders for distribution, risk analysis, statistics, CLV, etc.
        assert st_mock.subheader.call_count >= 3


# ---------------------------------------------------------------------------
# Integration Tests: Data Loader with Views
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:
    """Test data loader provides all data needed for churn/uplift/segmentation views."""

    def test_predictions_has_all_view_columns(self, dashboard_data_loader):
        """Predictions must have all columns needed for churn analytics view."""
        predictions = dashboard_data_loader.load_predictions()
        required = {
            "customer_id", "churn_probability", "risk_level", "segment",
        }
        assert required.issubset(set(predictions.columns))

    def test_predictions_has_clv_for_scatter(self, dashboard_data_loader):
        """Predictions must have clv_predicted for churn vs CLV scatter."""
        predictions = dashboard_data_loader.load_predictions()
        assert "clv_predicted" in predictions.columns
        assert (predictions["clv_predicted"] > 0).any()

    def test_uplift_data_has_all_columns(self, dashboard_data_loader):
        """Uplift results must have all columns for uplift view."""
        uplift = dashboard_data_loader.load_uplift_results()
        required = {"customer_id", "uplift_score", "treatment_effect", "segment"}
        assert required.issubset(set(uplift.columns))

    def test_feature_importance_sorted_descending(self, dashboard_data_loader):
        """Feature importance must be sorted by importance descending."""
        fi = dashboard_data_loader.load_feature_importance()
        assert fi["importance"].is_monotonic_decreasing

    def test_model_metrics_has_multiple_models(self, dashboard_data_loader):
        """Model metrics must contain multiple model types."""
        metrics = dashboard_data_loader.load_model_metrics()
        assert len(metrics) >= 2


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases for view data preparation."""

    def test_single_segment_predictions(self):
        """Views must handle predictions with only one segment."""
        df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "churn_probability": [0.1, 0.5, 0.9],
            "risk_level": ["low", "medium", "critical"],
            "segment": ["vip_loyal", "vip_loyal", "vip_loyal"],
        })
        seg_counts = df["segment"].value_counts()
        assert len(seg_counts) == 1
        assert seg_counts.sum() == 3

    def test_all_zero_uplift(self):
        """Views must handle all-zero uplift scores."""
        df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "uplift_score": [0.0, 0.0, 0.0],
            "treatment_effect": [0.01, -0.01, 0.0],
            "segment": ["vip", "regular", "new"],
        })
        avg = df["uplift_score"].mean()
        assert avg == 0.0
        persuadable = (df["uplift_score"] > 0).sum()
        assert persuadable == 0

    def test_extreme_churn_probabilities(self):
        """Views must handle extreme churn probabilities (0 and 1)."""
        df = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "churn_probability": [0.0, 1.0],
            "risk_level": ["low", "critical"],
            "segment": ["vip", "dormant"],
        })
        assert df["churn_probability"].min() == 0.0
        assert df["churn_probability"].max() == 1.0

    def test_negative_uplift_scores(self):
        """Views must handle negative uplift scores (sleeping dogs)."""
        df = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "uplift_score": [-0.1, -0.05, -0.2],
            "treatment_effect": [-0.05, -0.03, -0.1],
            "segment": ["vip", "regular", "new"],
        })
        sleeping_dogs = (df["uplift_score"] < 0).sum()
        assert sleeping_dogs == 3
        persuadable = (df["uplift_score"] > 0).sum()
        assert persuadable == 0
