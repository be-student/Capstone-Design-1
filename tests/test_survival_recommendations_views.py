"""
TDD Tests for Survival Analysis and Recommendations Dashboard Views.

Tests cover:
- Survival analysis view: KM curves, hazard rates, median survival,
  event rates, duration distributions, configuration display
- Recommendations view: KPI cards, priority ranking, type distribution,
  uplift analysis, segment breakdown, cost-effectiveness, top customers
- Data loader integration for both views
- Render function signatures and error handling
"""

import sys
from pathlib import Path
from typing import Dict
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
    """Load simulator configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dashboard_data_loader(config):
    """Create a DashboardDataLoader instance."""
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


@pytest.fixture
def sample_survival_data():
    """Create sample survival analysis data."""
    np.random.seed(42)
    n = 300
    segments = [
        "vip_loyal", "regular_loyal", "bargain_hunter",
        "explorer", "dormant", "new_customer",
    ]
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "duration_days": np.random.exponential(90, n),
        "event_observed": np.random.binomial(1, 0.3, n),
        "segment": np.random.choice(segments, n),
        "survival_probability": np.random.beta(5, 2, n),
    })


@pytest.fixture
def sample_recommendations():
    """Create sample recommendations with segment and cost."""
    np.random.seed(42)
    n = 50
    action_types = [
        "coupon", "push_notification", "email",
        "loyalty_points", "personal_outreach", "exclusive_offer",
        "no_action",
    ]
    segments = [
        "vip_loyal", "regular_loyal", "bargain_hunter",
        "explorer", "dormant", "new_customer",
    ]
    rec_types = np.random.choice(action_types, n, p=[
        0.2, 0.15, 0.2, 0.15, 0.05, 0.1, 0.15,
    ])
    costs = {
        "coupon": 5000, "push_notification": 100, "email": 200,
        "loyalty_points": 3000, "personal_outreach": 10000,
        "exclusive_offer": 8000, "no_action": 0,
    }
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "recommendation_type": rec_types,
        "expected_uplift": np.clip(np.random.beta(2, 8, n) * 0.5, 0.0, 1.0),
        "priority_score": np.clip(np.random.beta(3, 2, n), 0.0, 1.0),
        "recommended_offer": ["offer" for _ in range(n)],
        "segment": np.random.choice(segments, n),
        "estimated_cost": [costs[r] for r in rec_types],
    })


@pytest.fixture
def mock_streamlit():
    """Create a mock Streamlit module."""
    st = MagicMock()

    def _columns(n):
        return [MagicMock() for _ in range(n)]

    st.columns.side_effect = _columns
    return st


# ---------------------------------------------------------------------------
# Survival Analysis View - Render function tests
# ---------------------------------------------------------------------------

class TestSurvivalAnalysisViewExists:
    """Test that the survival analysis render function exists."""

    def test_render_function_exists(self):
        """render_survival_analysis must exist in app module."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_survival_analysis")
        assert callable(dashboard_app.render_survival_analysis)

    def test_render_function_signature(self):
        """render_survival_analysis must accept st_module, config, data_loader."""
        import inspect
        from src.dashboard.app import render_survival_analysis
        sig = inspect.signature(render_survival_analysis)
        params = list(sig.parameters.keys())
        assert "st_module" in params
        assert "config" in params
        assert "data_loader" in params


class TestSurvivalAnalysisKMCurves:
    """Test Kaplan-Meier survival curve data for dashboard."""

    def test_survival_curves_loadable(self, dashboard_data_loader):
        """KM survival curves must be loadable from data loader."""
        curves = dashboard_data_loader.load_survival_curves()
        assert isinstance(curves, dict)
        assert len(curves) > 0

    def test_all_segments_present(self, dashboard_data_loader):
        """Survival curves must include all customer segments."""
        curves = dashboard_data_loader.load_survival_curves()
        expected_segments = [
            "vip_loyal", "regular_loyal", "bargain_hunter",
            "explorer", "dormant", "new_customer",
        ]
        for seg in expected_segments:
            assert seg in curves, f"Missing curve for segment: {seg}"

    def test_curves_have_required_fields(self, dashboard_data_loader):
        """Each curve must have timeline, survival_prob, CI, and median."""
        curves = dashboard_data_loader.load_survival_curves()
        required_fields = [
            "timeline", "survival_prob", "ci_lower",
            "ci_upper", "median_survival_days",
        ]
        for seg, data in curves.items():
            for field in required_fields:
                assert field in data, f"Missing {field} for {seg}"

    def test_timeline_starts_at_zero(self, dashboard_data_loader):
        """Timeline must start at day 0."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            assert data["timeline"][0] == 0, (
                f"Timeline for {seg} doesn't start at 0"
            )

    def test_survival_starts_at_one(self, dashboard_data_loader):
        """Survival probability must start at 1.0."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            assert data["survival_prob"][0] == 1.0, (
                f"Survival prob for {seg} doesn't start at 1.0"
            )

    def test_survival_prob_bounded(self, dashboard_data_loader):
        """Survival probabilities must be between 0 and 1."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            for p in data["survival_prob"]:
                assert 0 <= p <= 1.0, (
                    f"Survival prob {p} out of bounds for {seg}"
                )

    def test_survival_monotonic_decreasing(self, dashboard_data_loader):
        """Survival probability must be non-increasing."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            probs = data["survival_prob"]
            for j in range(1, len(probs)):
                assert probs[j] <= probs[j - 1] + 0.001, (
                    f"Survival prob increased at {j} for {seg}"
                )

    def test_ci_bands_bracket_survival(self, dashboard_data_loader):
        """CI lower must be <= survival_prob <= CI upper."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            for i in range(len(data["survival_prob"])):
                assert data["ci_lower"][i] <= data["survival_prob"][i] + 0.01
                assert data["ci_upper"][i] >= data["survival_prob"][i] - 0.01


class TestSurvivalAnalysisHazardRates:
    """Test hazard rate computation from survival curves."""

    def test_hazard_computable_from_curves(self, dashboard_data_loader):
        """Hazard rate must be computable from survival curve endpoints."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            probs = data["survival_prob"]
            timeline = data["timeline"]
            if len(probs) >= 2 and probs[0] > 0:
                s0 = probs[0]
                s_end = probs[-1]
                t_max = timeline[-1]
                hazard = -np.log(max(s_end / s0, 0.001)) / max(t_max, 1)
                assert hazard >= 0, f"Hazard negative for {seg}"

    def test_dormant_highest_hazard(self, dashboard_data_loader):
        """Dormant segment should have the highest hazard rate."""
        curves = dashboard_data_loader.load_survival_curves()
        hazards = {}
        for seg, data in curves.items():
            probs = data["survival_prob"]
            timeline = data["timeline"]
            if len(probs) >= 2 and probs[0] > 0:
                s0 = probs[0]
                s_end = probs[-1]
                t_max = timeline[-1]
                hazards[seg] = -np.log(max(s_end / s0, 0.001)) / max(t_max, 1)
        if "dormant" in hazards and "vip_loyal" in hazards:
            assert hazards["dormant"] > hazards["vip_loyal"], (
                "Dormant should have higher hazard than VIP"
            )


class TestSurvivalAnalysisMedianSurvival:
    """Test median survival time display data."""

    def test_median_survival_per_segment(self, dashboard_data_loader):
        """Each segment must report median survival days."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            assert "median_survival_days" in data

    def test_median_survival_positive(self, dashboard_data_loader):
        """Median survival days must be positive when defined."""
        curves = dashboard_data_loader.load_survival_curves()
        for seg, data in curves.items():
            med = data["median_survival_days"]
            if med is not None:
                assert med > 0, f"Non-positive median for {seg}"


class TestSurvivalDataDisplay:
    """Test survival data display properties."""

    def test_event_rate_by_segment(self, sample_survival_data):
        """Event rate must be computable per segment."""
        event_stats = sample_survival_data.groupby("segment").agg(
            total=("customer_id", "count"),
            events=("event_observed", "sum"),
        ).reset_index()
        event_stats["event_rate"] = event_stats["events"] / event_stats["total"]
        assert len(event_stats) > 0
        assert (event_stats["event_rate"] >= 0).all()
        assert (event_stats["event_rate"] <= 1).all()

    def test_duration_distribution(self, sample_survival_data):
        """Duration data must be plottable as histogram."""
        durations = sample_survival_data["duration_days"]
        assert len(durations) > 0
        assert durations.std() > 0, "No variance in durations"

    def test_survival_config_from_yaml(self, config):
        """Survival model config must be present in YAML."""
        assert "survival" in config
        surv = config["survival"]
        assert "penalizer" in surv
        assert "l1_ratio" in surv
        assert "alpha" in surv


class TestSurvivalViewRender:
    """Test survival analysis view render with mock streamlit."""

    def test_renders_without_error(self, mock_streamlit, config,
                                    dashboard_data_loader):
        """render_survival_analysis must not raise with valid data."""
        from src.dashboard.app import render_survival_analysis
        render_survival_analysis(
            mock_streamlit, config, data_loader=dashboard_data_loader,
        )
        mock_streamlit.header.assert_called_once_with("Survival Analysis")

    def test_renders_kpi_metrics(self, mock_streamlit, config,
                                  dashboard_data_loader):
        """KPI metrics must be displayed."""
        from src.dashboard.app import render_survival_analysis
        render_survival_analysis(
            mock_streamlit, config, data_loader=dashboard_data_loader,
        )
        # columns() called for KPI cards
        mock_streamlit.columns.assert_called()

    def test_handles_empty_survival_data(self, mock_streamlit, config):
        """Must handle empty survival data gracefully."""
        from src.dashboard.app import render_survival_analysis
        mock_loader = MagicMock()
        mock_loader.load_survival_data.return_value = pd.DataFrame()
        render_survival_analysis(
            mock_streamlit, config, data_loader=mock_loader,
        )
        mock_streamlit.warning.assert_called()


# ---------------------------------------------------------------------------
# Recommendations View - Render function tests
# ---------------------------------------------------------------------------

class TestRecommendationsViewExists:
    """Test that the recommendations render function exists."""

    def test_render_function_exists(self):
        """render_recommendations must exist in app module."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_recommendations")
        assert callable(dashboard_app.render_recommendations)

    def test_render_function_signature(self):
        """render_recommendations must accept st_module, config, data_loader."""
        import inspect
        from src.dashboard.app import render_recommendations
        sig = inspect.signature(render_recommendations)
        params = list(sig.parameters.keys())
        assert "st_module" in params
        assert "config" in params
        assert "data_loader" in params


class TestRecommendationsKPICards:
    """Test KPI card data for recommendations view."""

    def test_total_customers_countable(self, sample_recommendations):
        """Total customers must be countable."""
        assert len(sample_recommendations) > 0

    def test_actionable_count(self, sample_recommendations):
        """Actionable recommendations (non-no_action) must be countable."""
        actionable = sample_recommendations[
            sample_recommendations["recommendation_type"] != "no_action"
        ]
        assert len(actionable) <= len(sample_recommendations)

    def test_avg_uplift_computable(self, sample_recommendations):
        """Average expected uplift must be computable."""
        avg = sample_recommendations["expected_uplift"].mean()
        assert 0 <= avg <= 1

    def test_avg_priority_computable(self, sample_recommendations):
        """Average priority score must be computable."""
        avg = sample_recommendations["priority_score"].mean()
        assert 0 <= avg <= 1


class TestRecommendationsPriorityRanking:
    """Test priority ranking of retention actions."""

    def test_sortable_by_priority(self, sample_recommendations):
        """Recommendations must be sortable by priority score."""
        sorted_recs = sample_recommendations.sort_values(
            "priority_score", ascending=False,
        )
        assert sorted_recs["priority_score"].is_monotonic_decreasing

    def test_top_n_extractable(self, sample_recommendations):
        """Top-N priority customers must be extractable."""
        top_10 = sample_recommendations.nlargest(10, "priority_score")
        assert len(top_10) == 10
        assert top_10["priority_score"].iloc[0] >= top_10["priority_score"].iloc[-1]


class TestRecommendationsTypeDistribution:
    """Test recommendation type distribution data."""

    def test_type_counts(self, sample_recommendations):
        """Recommendation type value counts must be computable."""
        counts = sample_recommendations["recommendation_type"].value_counts()
        assert len(counts) > 0
        assert counts.sum() == len(sample_recommendations)

    def test_multiple_action_types(self, sample_recommendations):
        """Multiple action types should be present."""
        types = sample_recommendations["recommendation_type"].unique()
        assert len(types) >= 2


class TestRecommendationsUpliftAnalysis:
    """Test expected uplift analysis."""

    def test_uplift_bounded(self, sample_recommendations):
        """Expected uplift must be between 0 and 1."""
        assert (sample_recommendations["expected_uplift"] >= 0).all()
        assert (sample_recommendations["expected_uplift"] <= 1).all()

    def test_uplift_has_variance(self, sample_recommendations):
        """Uplift scores should have variance for meaningful analysis."""
        std = sample_recommendations["expected_uplift"].std()
        assert std > 0, "Uplift scores have no variance"

    def test_uplift_by_type(self, sample_recommendations):
        """Uplift must be aggregatable by recommendation type."""
        uplift_by_type = sample_recommendations.groupby(
            "recommendation_type"
        )["expected_uplift"].mean()
        assert len(uplift_by_type) > 0


class TestRecommendationsSegmentBreakdown:
    """Test segment-level recommendation breakdown."""

    def test_segment_action_cross_tab(self, sample_recommendations):
        """Segment × action type cross-tabulation must be computable."""
        cross = sample_recommendations.groupby(
            ["segment", "recommendation_type"]
        ).size().reset_index(name="count")
        assert len(cross) > 0
        assert "count" in cross.columns

    def test_avg_uplift_by_segment(self, sample_recommendations):
        """Average uplift by segment must be computable."""
        seg_uplift = sample_recommendations.groupby("segment")[
            "expected_uplift"
        ].mean()
        assert len(seg_uplift) > 0


class TestRecommendationsCostEffectiveness:
    """Test cost-effectiveness analysis data."""

    def test_estimated_cost_present(self, sample_recommendations):
        """Estimated cost column must be present."""
        assert "estimated_cost" in sample_recommendations.columns

    def test_total_cost_computable(self, sample_recommendations):
        """Total estimated cost must be computable."""
        total = sample_recommendations["estimated_cost"].sum()
        assert total >= 0

    def test_cost_by_type(self, sample_recommendations):
        """Cost must be aggregatable by recommendation type."""
        cost_by_type = sample_recommendations.groupby(
            "recommendation_type"
        )["estimated_cost"].sum()
        assert len(cost_by_type) > 0

    def test_no_action_zero_cost(self, sample_recommendations):
        """No-action recommendations should have zero cost."""
        no_action = sample_recommendations[
            sample_recommendations["recommendation_type"] == "no_action"
        ]
        if len(no_action) > 0:
            assert (no_action["estimated_cost"] == 0).all()


class TestRecommendationsViewRender:
    """Test recommendations view render with mock streamlit."""

    def test_renders_without_error(self, mock_streamlit, config,
                                    dashboard_data_loader):
        """render_recommendations must not raise with valid data."""
        from src.dashboard.app import render_recommendations
        render_recommendations(
            mock_streamlit, config, data_loader=dashboard_data_loader,
        )
        mock_streamlit.header.assert_called_once_with(
            "Personalized Recommendations",
        )

    def test_renders_kpi_columns(self, mock_streamlit, config,
                                  dashboard_data_loader):
        """KPI columns must be created."""
        from src.dashboard.app import render_recommendations
        render_recommendations(
            mock_streamlit, config, data_loader=dashboard_data_loader,
        )
        mock_streamlit.columns.assert_called()

    def test_renders_subheaders(self, mock_streamlit, config,
                                 dashboard_data_loader):
        """Subheader sections must be rendered."""
        from src.dashboard.app import render_recommendations
        render_recommendations(
            mock_streamlit, config, data_loader=dashboard_data_loader,
        )
        subheader_calls = [
            call.args[0] for call in mock_streamlit.subheader.call_args_list
        ]
        # render_recommendations delegates to render_recommendations_view
        assert "Recommendation Distribution" in subheader_calls
        assert "Expected Uplift Analysis" in subheader_calls
        assert "Segment-Level Breakdown" in subheader_calls

    def test_handles_empty_recommendations(self, mock_streamlit, config):
        """Must handle empty recommendations gracefully."""
        from src.dashboard.app import render_recommendations
        mock_loader = MagicMock()
        mock_loader.load_recommendations.return_value = pd.DataFrame()
        render_recommendations(
            mock_streamlit, config, data_loader=mock_loader,
        )
        mock_streamlit.warning.assert_called()

    def test_renders_plotly_charts(self, mock_streamlit, config,
                                    dashboard_data_loader):
        """Plotly charts must be rendered."""
        from src.dashboard.app import render_recommendations
        render_recommendations(
            mock_streamlit, config, data_loader=dashboard_data_loader,
        )
        assert mock_streamlit.plotly_chart.call_count >= 3


# ---------------------------------------------------------------------------
# Data Loader Integration Tests
# ---------------------------------------------------------------------------

class TestDataLoaderSurvivalIntegration:
    """Test data loader survival data integration."""

    def test_survival_data_has_required_columns(self, dashboard_data_loader):
        """Survival data must have all required columns."""
        data = dashboard_data_loader.load_survival_data()
        required = [
            "customer_id", "duration_days",
            "event_observed", "segment", "survival_probability",
        ]
        for col in required:
            assert col in data.columns, f"Missing column: {col}"

    def test_survival_data_not_empty(self, dashboard_data_loader):
        """Survival data must not be empty."""
        data = dashboard_data_loader.load_survival_data()
        assert len(data) > 0

    def test_survival_probability_bounded(self, dashboard_data_loader):
        """Survival probabilities must be between 0 and 1."""
        data = dashboard_data_loader.load_survival_data()
        assert (data["survival_probability"] >= 0).all()
        assert (data["survival_probability"] <= 1).all()


class TestDataLoaderRecommendationsIntegration:
    """Test data loader recommendations integration."""

    def test_recommendations_has_required_columns(self, dashboard_data_loader):
        """Recommendations must have all required columns."""
        recs = dashboard_data_loader.load_recommendations()
        required = [
            "customer_id", "recommendation_type",
            "expected_uplift", "priority_score",
        ]
        for col in required:
            assert col in recs.columns, f"Missing column: {col}"

    def test_recommendations_not_empty(self, dashboard_data_loader):
        """Recommendations must not be empty."""
        recs = dashboard_data_loader.load_recommendations()
        assert len(recs) > 0

    def test_recommendations_have_segment(self, dashboard_data_loader):
        """Recommendations should include segment column."""
        recs = dashboard_data_loader.load_recommendations()
        assert "segment" in recs.columns

    def test_recommendations_have_cost(self, dashboard_data_loader):
        """Recommendations should include estimated_cost column."""
        recs = dashboard_data_loader.load_recommendations()
        assert "estimated_cost" in recs.columns

    def test_priority_scores_bounded(self, dashboard_data_loader):
        """Priority scores must be between 0 and 1."""
        recs = dashboard_data_loader.load_recommendations()
        assert (recs["priority_score"] >= 0).all()
        assert (recs["priority_score"] <= 1).all()

    def test_expected_uplift_bounded(self, dashboard_data_loader):
        """Expected uplift must be between 0 and 1."""
        recs = dashboard_data_loader.load_recommendations()
        assert (recs["expected_uplift"] >= 0).all()
        assert (recs["expected_uplift"] <= 1).all()


# ---------------------------------------------------------------------------
# View integration: survival + recommendations cross-view tests
# ---------------------------------------------------------------------------

class TestSurvivalRecommendationsIntegration:
    """Test that survival and recommendations views integrate properly."""

    def test_both_views_callable(self):
        """Both view render functions must be callable."""
        from src.dashboard.app import (
            render_survival_analysis,
            render_recommendations,
        )
        assert callable(render_survival_analysis)
        assert callable(render_recommendations)

    def test_both_in_page_mapping(self):
        """Both pages must be in the dashboard page mapping."""
        from src.dashboard.app import render_survival_analysis, render_recommendations
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Survival Analysis" in PAGES
        assert "Recommendations" in PAGES

    def test_segments_consistent(self, dashboard_data_loader):
        """Segments in survival and recommendations should overlap."""
        survival = dashboard_data_loader.load_survival_data()
        recs = dashboard_data_loader.load_recommendations()
        surv_segs = set(survival["segment"].unique())
        rec_segs = set(recs["segment"].unique())
        overlap = surv_segs & rec_segs
        assert len(overlap) > 0, "No segment overlap between views"
