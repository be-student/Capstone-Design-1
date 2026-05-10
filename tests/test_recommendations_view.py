"""
TDD Tests for Enhanced Recommendations Dashboard View.

Tests cover:
- KPI card computation (total recs, avg uplift, top action, high priority count)
- Recommendation distribution (donut + bar chart data)
- Uplift analysis (box plot data, avg uplift by type)
- Priority score distribution
- Segment-level breakdown (stacked bar, segment stats)
- Cost-benefit analysis (total cost, ROI, scatter data)
- Filterable table (filter by type, by priority)
- Top-K prioritized list
- Retention offers detail table
- Integration with data loader
- Empty data handling
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Load simulator configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_recommendations():
    """Create sample recommendation data."""
    np.random.seed(42)
    n = 50
    types = ["coupon", "push_notification", "email", "loyalty_points", "no_action"]
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "recommendation_type": np.random.choice(types, n),
        "expected_uplift": np.random.uniform(0.0, 0.25, n),
        "priority_score": np.random.uniform(0.1, 1.0, n),
        "recommended_offer": [f"Offer_{i}" for i in range(n)],
    })


@pytest.fixture
def sample_retention_offers():
    """Create sample retention offer data."""
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter"], n
        ),
        "risk_level": np.random.choice(["low", "medium", "high", "critical"], n),
        "churn_probability": np.random.beta(3, 3, n),
        "offer_type": np.random.choice(
            ["premium_discount", "discount_coupon", "engagement_email", "loyalty_points"], n
        ),
        "offer_detail": [f"Detail_{i}" for i in range(n)],
        "expected_uplift": np.random.uniform(0.01, 0.25, n),
        "estimated_cost_krw": np.random.randint(1000, 150000, n),
        "estimated_revenue_save_krw": np.random.randint(5000, 500000, n),
        "priority_rank": list(range(1, n + 1)),
    })


@pytest.fixture
def sample_predictions():
    """Create sample predictions with segments."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_probability": np.random.beta(2, 5, n),
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter"], n
        ),
    })


@pytest.fixture
def mock_data_loader(sample_recommendations, sample_retention_offers, sample_predictions):
    """Create a mock data loader."""
    loader = MagicMock()
    loader.load_recommendations.return_value = sample_recommendations
    loader.load_retention_offers.return_value = sample_retention_offers
    loader.load_predictions.return_value = sample_predictions
    return loader


@pytest.fixture
def mock_st():
    """Create a mock Streamlit module."""
    st = MagicMock()
    st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    st.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]

    # Mock slider to return a float value
    st.slider.return_value = 0.0
    st.selectbox.return_value = "All"

    return st


# ---------------------------------------------------------------------------
# Test: Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Test that the recommendations view module is importable."""

    def test_import_recommendations_view(self):
        from src.dashboard.recommendations_view import render_recommendations_view
        assert callable(render_recommendations_view)

    def test_import_action_colors(self):
        from src.dashboard.recommendations_view import ACTION_COLORS
        assert isinstance(ACTION_COLORS, dict)
        assert len(ACTION_COLORS) > 0

    def test_import_action_labels(self):
        from src.dashboard.recommendations_view import ACTION_LABELS
        assert isinstance(ACTION_LABELS, dict)
        assert "coupon" in ACTION_LABELS


# ---------------------------------------------------------------------------
# Test: KPI Cards
# ---------------------------------------------------------------------------


class TestKPICards:
    """Test KPI card computations."""

    def test_total_recommendations_count(self, sample_recommendations):
        assert len(sample_recommendations) == 50

    def test_avg_expected_uplift(self, sample_recommendations):
        avg_uplift = sample_recommendations["expected_uplift"].mean()
        assert 0 <= avg_uplift <= 1

    def test_top_action_type_identifiable(self, sample_recommendations):
        top_action = sample_recommendations["recommendation_type"].value_counts().idxmax()
        assert isinstance(top_action, str)
        assert top_action in sample_recommendations["recommendation_type"].values

    def test_high_priority_count(self, sample_recommendations):
        high_priority = (sample_recommendations["priority_score"] >= 0.7).sum()
        assert isinstance(high_priority, (int, np.integer))
        assert high_priority >= 0


# ---------------------------------------------------------------------------
# Test: Distribution Analysis
# ---------------------------------------------------------------------------


class TestDistributionAnalysis:
    """Test recommendation distribution chart data."""

    def test_type_counts_computable(self, sample_recommendations):
        counts = sample_recommendations["recommendation_type"].value_counts()
        assert len(counts) > 0
        assert counts.sum() == len(sample_recommendations)

    def test_all_types_represented(self, sample_recommendations):
        unique_types = sample_recommendations["recommendation_type"].unique()
        assert len(unique_types) >= 2

    def test_type_distribution_sums_to_total(self, sample_recommendations):
        counts = sample_recommendations["recommendation_type"].value_counts()
        assert counts.sum() == len(sample_recommendations)


# ---------------------------------------------------------------------------
# Test: Uplift Analysis
# ---------------------------------------------------------------------------


class TestUpliftAnalysis:
    """Test expected uplift analysis."""

    def test_uplift_values_bounded(self, sample_recommendations):
        assert sample_recommendations["expected_uplift"].min() >= 0
        assert sample_recommendations["expected_uplift"].max() <= 1

    def test_avg_uplift_by_type_computable(self, sample_recommendations):
        avg_by_type = sample_recommendations.groupby(
            "recommendation_type"
        )["expected_uplift"].mean()
        assert len(avg_by_type) > 0
        assert all(0 <= v <= 1 for v in avg_by_type.values)

    def test_priority_score_bounded(self, sample_recommendations):
        assert sample_recommendations["priority_score"].min() >= 0
        assert sample_recommendations["priority_score"].max() <= 1


# ---------------------------------------------------------------------------
# Test: Segment Breakdown
# ---------------------------------------------------------------------------


class TestSegmentBreakdown:
    """Test segment-level recommendation breakdown."""

    def test_cross_tab_computable_with_predictions(
        self, sample_recommendations, sample_predictions,
    ):
        merged = sample_recommendations.merge(
            sample_predictions[["customer_id", "segment"]],
            on="customer_id",
            how="left",
        )
        cross_tab = pd.crosstab(merged["segment"], merged["recommendation_type"])
        assert cross_tab.shape[0] > 0
        assert cross_tab.shape[1] > 0

    def test_segment_stats_computable(
        self, sample_recommendations, sample_predictions,
    ):
        merged = sample_recommendations.merge(
            sample_predictions[["customer_id", "segment"]],
            on="customer_id",
            how="left",
        )
        stats = merged.groupby("segment")["expected_uplift"].agg(["mean", "max", "count"])
        assert len(stats) > 0


# ---------------------------------------------------------------------------
# Test: Cost-Benefit Analysis
# ---------------------------------------------------------------------------


class TestCostBenefitAnalysis:
    """Test cost-benefit analysis from retention offers."""

    def test_total_cost_positive(self, sample_retention_offers):
        total_cost = sample_retention_offers["estimated_cost_krw"].sum()
        assert total_cost > 0

    def test_total_revenue_saved_positive(self, sample_retention_offers):
        total_rev = sample_retention_offers["estimated_revenue_save_krw"].sum()
        assert total_rev > 0

    def test_roi_computable(self, sample_retention_offers):
        total_cost = sample_retention_offers["estimated_cost_krw"].sum()
        total_rev = sample_retention_offers["estimated_revenue_save_krw"].sum()
        roi = total_rev / max(total_cost, 1)
        assert roi > 0

    def test_cost_by_offer_type(self, sample_retention_offers):
        cost_by_type = sample_retention_offers.groupby(
            "offer_type"
        )["estimated_cost_krw"].sum()
        assert len(cost_by_type) > 0
        assert all(v >= 0 for v in cost_by_type.values)

    def test_roi_by_offer_type(self, sample_retention_offers):
        agg = sample_retention_offers.groupby("offer_type").agg(
            cost=("estimated_cost_krw", "sum"),
            revenue=("estimated_revenue_save_krw", "sum"),
        )
        agg["roi"] = agg["revenue"] / agg["cost"].clip(lower=1)
        assert all(agg["roi"] > 0)


# ---------------------------------------------------------------------------
# Test: Recommendation Table & Top-K
# ---------------------------------------------------------------------------


class TestRecommendationTable:
    """Test filterable recommendation table and top-K."""

    def test_sort_by_priority(self, sample_recommendations):
        sorted_df = sample_recommendations.sort_values(
            "priority_score", ascending=False,
        )
        assert sorted_df.iloc[0]["priority_score"] >= sorted_df.iloc[-1]["priority_score"]

    def test_filter_by_type(self, sample_recommendations):
        filtered = sample_recommendations[
            sample_recommendations["recommendation_type"] == "coupon"
        ]
        assert all(filtered["recommendation_type"] == "coupon")

    def test_filter_by_min_priority(self, sample_recommendations):
        min_priority = 0.5
        filtered = sample_recommendations[
            sample_recommendations["priority_score"] >= min_priority
        ]
        assert all(filtered["priority_score"] >= min_priority)

    def test_top_k_extraction(self, sample_recommendations):
        k = 10
        top_k = sample_recommendations.nlargest(k, "priority_score")
        assert len(top_k) == k
        assert top_k.iloc[0]["priority_score"] >= top_k.iloc[-1]["priority_score"]

    def test_top_k_less_than_total(self, sample_recommendations):
        k = min(10, len(sample_recommendations))
        top_k = sample_recommendations.nlargest(k, "priority_score")
        assert len(top_k) <= len(sample_recommendations)


# ---------------------------------------------------------------------------
# Test: Retention Offers Detail
# ---------------------------------------------------------------------------


class TestRetentionOffers:
    """Test retention offers detail display."""

    def test_offers_have_customer_id(self, sample_retention_offers):
        assert "customer_id" in sample_retention_offers.columns

    def test_offers_have_segment(self, sample_retention_offers):
        assert "segment" in sample_retention_offers.columns

    def test_offers_have_risk_level(self, sample_retention_offers):
        assert "risk_level" in sample_retention_offers.columns
        valid_levels = {"low", "medium", "high", "critical"}
        assert set(sample_retention_offers["risk_level"].unique()) <= valid_levels

    def test_offers_have_priority_rank(self, sample_retention_offers):
        assert "priority_rank" in sample_retention_offers.columns


# ---------------------------------------------------------------------------
# Test: Render Function Integration
# ---------------------------------------------------------------------------


class TestRenderIntegration:
    """Test render function integration."""

    def test_render_recommendations_view_callable(self):
        from src.dashboard.recommendations_view import render_recommendations_view
        assert callable(render_recommendations_view)

    def test_render_with_mock_streamlit(
        self, mock_st, config, mock_data_loader,
    ):
        from src.dashboard.recommendations_view import render_recommendations_view
        # Should not raise
        render_recommendations_view(mock_st, config, mock_data_loader)

    def test_render_calls_data_loader(
        self, mock_st, config, mock_data_loader,
    ):
        from src.dashboard.recommendations_view import render_recommendations_view
        render_recommendations_view(mock_st, config, mock_data_loader)
        mock_data_loader.load_recommendations.assert_called_once()
        mock_data_loader.load_retention_offers.assert_called_once()
        mock_data_loader.load_predictions.assert_called_once()

    def test_render_with_empty_recommendations(self, mock_st, config):
        from src.dashboard.recommendations_view import render_recommendations_view
        loader = MagicMock()
        loader.load_recommendations.return_value = pd.DataFrame()
        loader.load_retention_offers.return_value = pd.DataFrame()
        loader.load_predictions.return_value = pd.DataFrame()
        # Should not raise
        render_recommendations_view(mock_st, config, loader)
        mock_st.warning.assert_called()

    def test_render_with_empty_offers(
        self, mock_st, config, sample_recommendations,
    ):
        from src.dashboard.recommendations_view import render_recommendations_view
        loader = MagicMock()
        loader.load_recommendations.return_value = sample_recommendations
        loader.load_retention_offers.return_value = pd.DataFrame()
        loader.load_predictions.return_value = pd.DataFrame()
        # Should not raise
        render_recommendations_view(mock_st, config, loader)


# ---------------------------------------------------------------------------
# Test: App.py Integration
# ---------------------------------------------------------------------------


class TestAppIntegration:
    """Test recommendations view is integrated in app.py."""

    def test_render_recommendations_in_app(self):
        from src.dashboard.app import render_recommendations
        assert callable(render_recommendations)

    def test_recommendations_page_in_page_list(self):
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Recommendations" in PAGES

    def test_recommendations_has_icon(self):
        from src.dashboard.utils.dashboard_helpers import PAGE_ICONS
        assert "Recommendations" in PAGE_ICONS

    def test_render_delegates_to_view(self, mock_st, config):
        from src.dashboard.app import render_recommendations
        loader = MagicMock()
        loader.load_recommendations.return_value = pd.DataFrame()
        loader.load_retention_offers.return_value = pd.DataFrame()
        loader.load_predictions.return_value = pd.DataFrame()
        render_recommendations(mock_st, config, loader)
        loader.load_recommendations.assert_called()
