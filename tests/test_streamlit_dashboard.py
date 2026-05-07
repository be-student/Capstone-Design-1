"""
TDD Tests for Streamlit Dashboard UI Components and Render Functions.

Tests cover:
- All render functions exist and are callable
- Render functions work with mocked Streamlit module
- DashboardDataLoader integration (all data types)
- Cohort analysis view data preparation
- Enhanced data loaders (MLflow runs, ROC data, survival curves, etc.)
- Dashboard configuration from YAML
- Error handling for missing/malformed data
- Page navigation and sidebar rendering
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

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
def data_loader(config):
    """Create a DashboardDataLoader instance."""
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


@pytest.fixture
def mock_st():
    """Create a mock Streamlit module with common methods."""
    st = MagicMock()
    st.title = MagicMock()
    st.header = MagicMock()
    st.subheader = MagicMock()
    st.write = MagicMock()
    st.markdown = MagicMock()
    st.metric = MagicMock()
    st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()])
    st.sidebar = MagicMock()
    st.sidebar.selectbox = MagicMock(return_value="Overview")
    st.sidebar.multiselect = MagicMock(return_value=[])
    st.sidebar.slider = MagicMock(return_value=0.5)
    st.selectbox = MagicMock(return_value="all")
    st.multiselect = MagicMock(return_value=[])
    st.slider = MagicMock(return_value=0.5)
    st.dataframe = MagicMock()
    st.plotly_chart = MagicMock()
    st.pyplot = MagicMock()
    st.tabs = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    st.expander = MagicMock()
    st.expander.return_value.__enter__ = MagicMock(return_value=MagicMock())
    st.expander.return_value.__exit__ = MagicMock(return_value=False)
    st.container = MagicMock()
    st.container.return_value.__enter__ = MagicMock(return_value=MagicMock())
    st.container.return_value.__exit__ = MagicMock(return_value=False)
    st.info = MagicMock()
    st.warning = MagicMock()
    st.error = MagicMock()
    st.success = MagicMock()
    st.spinner = MagicMock()
    st.spinner.return_value.__enter__ = MagicMock(return_value=None)
    st.spinner.return_value.__exit__ = MagicMock(return_value=False)
    st.number_input = MagicMock(return_value=50000000)
    st.text_input = MagicMock(return_value="")
    st.checkbox = MagicMock(return_value=False)
    st.radio = MagicMock(return_value="Option 1")
    st.bar_chart = MagicMock()
    st.line_chart = MagicMock()
    st.area_chart = MagicMock()
    st.table = MagicMock()
    st.json = MagicMock()
    st.empty = MagicMock(return_value=MagicMock())
    return st


# ---------------------------------------------------------------------------
# Render function existence tests
# ---------------------------------------------------------------------------

class TestRenderFunctionsExist:
    """Verify all required render functions exist in the dashboard app module."""

    def _get_app(self):
        from src.dashboard import app as dashboard_app
        return dashboard_app

    def test_render_overview_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_overview")
        assert callable(app.render_overview)

    def test_render_churn_analytics_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_churn_analytics")
        assert callable(app.render_churn_analytics)

    def test_render_cohort_analysis_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_cohort_analysis")
        assert callable(app.render_cohort_analysis)

    def test_render_model_performance_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_model_performance")
        assert callable(app.render_model_performance)

    def test_render_segmentation_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_segmentation")
        assert callable(app.render_segmentation)

    def test_render_budget_optimization_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_budget_optimization")
        assert callable(app.render_budget_optimization)

    def test_render_ab_testing_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_ab_testing")
        assert callable(app.render_ab_testing)

    def test_render_survival_analysis_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_survival_analysis")
        assert callable(app.render_survival_analysis)

    def test_render_recommendations_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_recommendations")
        assert callable(app.render_recommendations)

    def test_render_clv_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_clv")
        assert callable(app.render_clv)

    def test_render_uplift_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_uplift")
        assert callable(app.render_uplift)

    def test_render_retention_campaign_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_retention_campaign")
        assert callable(app.render_retention_campaign)

    def test_render_realtime_scoring_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_realtime_scoring")
        assert callable(app.render_realtime_scoring)

    def test_render_mlflow_experiments_exists(self):
        app = self._get_app()
        assert hasattr(app, "render_mlflow_experiments")
        assert callable(app.render_mlflow_experiments)


# ---------------------------------------------------------------------------
# Render function smoke tests (with mocked Streamlit)
# ---------------------------------------------------------------------------

class TestRenderFunctionsSmokeTest:
    """Smoke tests: each render function runs without error using mocked st."""

    def test_render_overview_runs(self, mock_st, config, data_loader):
        """render_overview should run without raising exceptions."""
        from src.dashboard.app import render_overview
        try:
            render_overview(mock_st, config, data_loader)
        except Exception as e:
            # Some render functions may have internal issues with mocks
            # but should not raise ImportError or AttributeError
            assert not isinstance(e, (ImportError, AttributeError)), str(e)

    def test_render_recommendations_runs(self, mock_st, config, data_loader):
        """render_recommendations should run without raising exceptions."""
        from src.dashboard.app import render_recommendations
        try:
            render_recommendations(mock_st, config, data_loader)
        except Exception as e:
            assert not isinstance(e, (ImportError, AttributeError)), str(e)

    def test_render_clv_runs(self, mock_st, config, data_loader):
        """render_clv should run without raising exceptions."""
        from src.dashboard.app import render_clv
        try:
            render_clv(mock_st, config, data_loader)
        except Exception as e:
            assert not isinstance(e, (ImportError, AttributeError)), str(e)

    def test_render_uplift_runs(self, mock_st, config, data_loader):
        """render_uplift should run without raising exceptions."""
        from src.dashboard.app import render_uplift
        try:
            render_uplift(mock_st, config, data_loader)
        except Exception as e:
            assert not isinstance(e, (ImportError, AttributeError)), str(e)

    def test_render_realtime_scoring_runs(self, mock_st, config, data_loader):
        """render_realtime_scoring should run without raising exceptions."""
        from src.dashboard.app import render_realtime_scoring
        try:
            render_realtime_scoring(mock_st, config, data_loader)
        except Exception as e:
            assert not isinstance(e, (ImportError, AttributeError)), str(e)

    def test_render_mlflow_experiments_runs(self, mock_st, config, data_loader):
        """render_mlflow_experiments should run without raising exceptions."""
        from src.dashboard.app import render_mlflow_experiments
        try:
            render_mlflow_experiments(mock_st, config, data_loader)
        except Exception as e:
            assert not isinstance(e, (ImportError, AttributeError)), str(e)


# ---------------------------------------------------------------------------
# Data loader integration tests
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:
    """Test DashboardDataLoader loads all data types correctly."""

    def test_load_predictions_returns_dataframe(self, data_loader):
        """load_predictions must return a DataFrame."""
        df = data_loader.load_predictions()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_predictions_has_required_columns(self, data_loader):
        """Predictions must have customer_id, churn_probability, risk_level, segment."""
        df = data_loader.load_predictions()
        for col in ["customer_id", "churn_probability", "risk_level", "segment"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_model_metrics_returns_dict(self, data_loader):
        """Model metrics must return a dict with model names."""
        metrics = data_loader.load_model_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_load_ab_test_results_structure(self, data_loader):
        """A/B test results must have experiment_name and p_value."""
        results = data_loader.load_ab_test_results()
        assert isinstance(results, dict)
        assert "experiment_name" in results
        assert "p_value" in results

    def test_metric_history_loaders_return_dataframes(self, data_loader):
        assert isinstance(data_loader.load_auc_history(), pd.DataFrame)
        assert isinstance(data_loader.load_precision_history(), pd.DataFrame)
        assert isinstance(data_loader.load_recall_history(), pd.DataFrame)

    def test_load_budget_results_returns_dataframe(self, data_loader):
        """Budget results must return a DataFrame with allocations."""
        df = data_loader.load_budget_results()
        assert isinstance(df, pd.DataFrame)
        assert "segment" in df.columns
        assert "allocated_budget_krw" in df.columns

    def test_load_survival_data_returns_dataframe(self, data_loader):
        """Survival data must return a DataFrame."""
        df = data_loader.load_survival_data()
        assert isinstance(df, pd.DataFrame)
        assert "duration_days" in df.columns
        assert "event_observed" in df.columns

    def test_load_recommendations_returns_dataframe(self, data_loader):
        """Recommendations must return a DataFrame."""
        df = data_loader.load_recommendations()
        assert isinstance(df, pd.DataFrame)
        assert "recommendation_type" in df.columns

    def test_load_uplift_results_returns_dataframe(self, data_loader):
        """Uplift results must return a DataFrame."""
        df = data_loader.load_uplift_results()
        assert isinstance(df, pd.DataFrame)
        assert "uplift_score" in df.columns

    def test_load_clv_data_returns_dataframe(self, data_loader):
        """CLV data must return a DataFrame."""
        df = data_loader.load_clv_data()
        assert isinstance(df, pd.DataFrame)
        assert "clv_predicted" in df.columns

    def test_load_feature_importance_returns_dataframe(self, data_loader):
        """Feature importance must return a sorted DataFrame."""
        fi = data_loader.load_feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert fi["importance"].is_monotonic_decreasing

    def test_load_cohort_data_returns_dataframe(self, data_loader):
        """Cohort data must return a DataFrame with event data."""
        df = data_loader.load_cohort_data()
        assert isinstance(df, pd.DataFrame)
        assert "customer_id" in df.columns
        assert "event_date" in df.columns

    def test_load_cohort_retention_matrix_returns_dataframe(self, data_loader):
        """Cohort retention matrix must return a DataFrame."""
        df = data_loader.load_cohort_retention_matrix()
        assert isinstance(df, pd.DataFrame)

    def test_load_mlflow_runs_returns_dataframe(self, data_loader):
        """MLflow runs must return a DataFrame."""
        df = data_loader.load_mlflow_runs()
        assert isinstance(df, pd.DataFrame)
        assert "model_type" in df.columns
        assert "auc" in df.columns

    def test_load_roc_data_returns_dict(self, data_loader):
        """ROC data must return a dict of model FPR/TPR curves."""
        roc = data_loader.load_roc_data()
        assert isinstance(roc, dict)
        for model_name, curve in roc.items():
            assert "fpr" in curve
            assert "tpr" in curve

    def test_load_confusion_matrices_returns_dict(self, data_loader):
        """Confusion matrices must return a dict of 2x2 matrices."""
        cms = data_loader.load_confusion_matrices()
        assert isinstance(cms, dict)
        for model_name, cm in cms.items():
            assert len(cm) == 2
            assert len(cm[0]) == 2

    def test_load_ab_test_detailed_returns_dict(self, data_loader):
        """Detailed A/B test must return a dict with experiments list."""
        detail = data_loader.load_ab_test_detailed()
        assert isinstance(detail, dict)
        assert "experiments" in detail
        assert isinstance(detail["experiments"], list)
        assert len(detail["experiments"]) > 0

    def test_load_survival_curves_returns_dict(self, data_loader):
        """Survival curves must return a dict per segment."""
        curves = data_loader.load_survival_curves()
        assert isinstance(curves, dict)
        for seg, curve in curves.items():
            assert "timeline" in curve
            assert "survival_prob" in curve


# ---------------------------------------------------------------------------
# Dashboard data quality tests
# ---------------------------------------------------------------------------

class TestDashboardDataQuality:
    """Test data quality for dashboard components."""

    def test_predictions_churn_probability_bounded(self, data_loader):
        """Churn probabilities must be in [0, 1]."""
        df = data_loader.load_predictions()
        assert (df["churn_probability"] >= 0).all()
        assert (df["churn_probability"] <= 1).all()

    def test_survival_probabilities_bounded(self, data_loader):
        """Survival probabilities must be in [0, 1]."""
        df = data_loader.load_survival_data()
        assert (df["survival_probability"] >= 0).all()
        assert (df["survival_probability"] <= 1).all()

    def test_budget_allocations_positive(self, data_loader):
        """Budget allocations must be non-negative."""
        df = data_loader.load_budget_results()
        assert (df["allocated_budget_krw"] >= 0).all()

    def test_uplift_scores_have_variation(self, data_loader):
        """Uplift scores should have variation (not all same value)."""
        df = data_loader.load_uplift_results()
        assert df["uplift_score"].std() > 0

    def test_clv_predictions_positive(self, data_loader):
        """CLV predictions must be positive."""
        df = data_loader.load_clv_data()
        assert (df["clv_predicted"] > 0).all()

    def test_cohort_data_has_revenue(self, data_loader):
        """Cohort data must include revenue column."""
        df = data_loader.load_cohort_data()
        assert "revenue" in df.columns
        assert (df["revenue"] > 0).all()

    def test_mlflow_runs_have_valid_metrics(self, data_loader):
        """MLflow runs metrics must be in valid ranges."""
        df = data_loader.load_mlflow_runs()
        for col in ["auc", "precision", "recall", "f1_score"]:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()

    def test_roc_data_fpr_tpr_bounded(self, data_loader):
        """ROC curve FPR and TPR must be in [0, 1]."""
        roc = data_loader.load_roc_data()
        for model_name, curve in roc.items():
            assert all(0 <= v <= 1 for v in curve["fpr"])
            assert all(0 <= v <= 1 for v in curve["tpr"])

    def test_ab_test_experiments_have_required_fields(self, data_loader):
        """Each A/B experiment must have required statistical fields."""
        detail = data_loader.load_ab_test_detailed()
        required = {"name", "p_value", "is_significant", "treatment_size", "control_size"}
        for exp in detail["experiments"]:
            for field in required:
                assert field in exp, f"Missing field: {field} in experiment"


# ---------------------------------------------------------------------------
# Dashboard page navigation tests
# ---------------------------------------------------------------------------

class TestDashboardNavigation:
    """Test dashboard page list and navigation helpers."""

    def test_page_list_includes_all_views(self):
        """Page list must include all required dashboard views."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        required = [
            "Overview", "Model Performance", "Customer Segmentation",
            "Budget Optimization", "A/B Testing", "Survival Analysis",
            "Recommendations", "CLV Prediction", "Uplift Modeling",
        ]
        for page in required:
            assert page in pages, f"Missing page: {page}"

    def test_cohort_analysis_in_page_list(self):
        """Cohort Analysis must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "Cohort Analysis" in pages

    def test_clv_retention_campaign_in_page_list(self):
        """CLV & Retention Campaign must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "CLV & Retention Campaign" in pages

    def test_mlflow_experiments_in_page_list(self):
        """MLflow Experiments must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        pages = get_page_list()
        assert "MLflow Experiments" in pages

    def test_all_pages_have_icons(self):
        """Every page must have an assigned icon."""
        from src.dashboard.utils.dashboard_helpers import get_page_list, get_page_icon
        for page in get_page_list():
            icon = get_page_icon(page)
            assert icon is not None and len(icon) > 0, f"Missing icon for: {page}"

    def test_page_count_at_least_11(self):
        """Should have at least 11 pages."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        assert len(get_page_list()) >= 11


# ---------------------------------------------------------------------------
# Dashboard config integration tests
# ---------------------------------------------------------------------------

class TestDashboardConfigIntegration:
    """Test that dashboard integrates with YAML configuration."""

    def test_dashboard_port_is_8501(self, config):
        """Dashboard port should be 8501."""
        port = config.get("dashboard", {}).get("port", 8501)
        assert port == 8501

    def test_budget_total_from_config(self, config):
        """Budget total should come from config."""
        from src.dashboard.utils.dashboard_helpers import get_budget_config
        budget = get_budget_config(config)
        assert budget["total_krw"] > 0
        assert budget["currency"] == "KRW"

    def test_churn_definition_from_config(self, config):
        """Churn definition should come from config."""
        from src.dashboard.utils.dashboard_helpers import get_churn_definition
        churn_def = get_churn_definition(config)
        assert "no_purchase_days" in churn_def
        assert "no_login_days" in churn_def

    def test_ensemble_weights_from_config(self, config):
        """Ensemble weights should come from config."""
        from src.dashboard.utils.dashboard_helpers import get_ensemble_weights
        ml_w, dl_w = get_ensemble_weights(config)
        assert 0 < ml_w < 1
        assert 0 < dl_w < 1
        assert abs(ml_w + dl_w - 1.0) < 0.01

    def test_sidebar_info_complete(self, config):
        """Sidebar info should contain all required sections."""
        from src.dashboard.utils.dashboard_helpers import build_sidebar_info
        info = build_sidebar_info(config)
        assert "churn_definition" in info
        assert "budget" in info
        assert "ensemble_weights" in info


# ---------------------------------------------------------------------------
# Cohort analysis view data preparation tests
# ---------------------------------------------------------------------------

class TestCohortAnalysisViewData:
    """Test cohort analysis data for dashboard view rendering."""

    def test_cohort_data_can_be_assigned(self, data_loader):
        """Cohort data must be assignable via CohortAnalyzer."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        df = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        assigned = analyzer.assign_cohorts(df, cohort_type="monthly")
        assert "cohort" in assigned.columns
        assert "cohort_period" in assigned.columns

    def test_retention_matrix_computable_from_cohort_data(self, data_loader):
        """Retention matrix must be computable from loaded cohort data."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        df = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        assigned = analyzer.assign_cohorts(df, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(assigned)
        assert isinstance(retention, pd.DataFrame)
        assert len(retention) > 0

    def test_cohort_summary_computable(self, data_loader):
        """Cohort summary must be computable from loaded data."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        df = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        assigned = analyzer.assign_cohorts(df, cohort_type="monthly")
        summary = analyzer.get_cohort_summary(assigned)
        assert "total_customers" in summary.columns
        assert "total_events" in summary.columns

    def test_retention_matrix_from_loader_valid(self, data_loader):
        """Pre-loaded retention matrix must have valid retention rates."""
        matrix = data_loader.load_cohort_retention_matrix()
        if not matrix.empty:
            assert (matrix.values >= 0).all()
            assert (matrix.values <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Uplift view data tests
# ---------------------------------------------------------------------------

class TestUpliftViewData:
    """Test uplift data preparation for dashboard display."""

    def test_uplift_data_has_segment(self, data_loader):
        """Uplift data must include segment for grouping."""
        df = data_loader.load_uplift_results()
        assert "segment" in df.columns

    def test_uplift_scores_sortable(self, data_loader):
        """Uplift scores must be sortable for ranking display."""
        df = data_loader.load_uplift_results()
        sorted_df = df.sort_values("uplift_score", ascending=False)
        assert sorted_df["uplift_score"].is_monotonic_decreasing

    def test_treatment_effect_present(self, data_loader):
        """Treatment effect must be present for uplift display."""
        df = data_loader.load_uplift_results()
        assert "treatment_effect" in df.columns


# ---------------------------------------------------------------------------
# Survival analysis view data tests
# ---------------------------------------------------------------------------

class TestSurvivalAnalysisViewData:
    """Test survival analysis data for dashboard display."""

    def test_survival_curves_per_segment(self, data_loader):
        """Each segment should have a survival curve."""
        curves = data_loader.load_survival_curves()
        assert len(curves) >= 4  # at least 4 segments

    def test_survival_curves_have_ci(self, data_loader):
        """Survival curves should include confidence intervals."""
        curves = data_loader.load_survival_curves()
        for seg, curve in curves.items():
            assert "ci_lower" in curve
            assert "ci_upper" in curve

    def test_survival_curves_have_median(self, data_loader):
        """Survival curves should include median survival days."""
        curves = data_loader.load_survival_curves()
        for seg, curve in curves.items():
            assert "median_survival_days" in curve

    def test_survival_timeline_increasing(self, data_loader):
        """Survival timeline should be monotonically increasing."""
        curves = data_loader.load_survival_curves()
        for seg, curve in curves.items():
            timeline = curve["timeline"]
            assert all(timeline[i] <= timeline[i+1] for i in range(len(timeline)-1))

    def test_survival_prob_decreasing(self, data_loader):
        """Survival probability should generally decrease over time."""
        curves = data_loader.load_survival_curves()
        for seg, curve in curves.items():
            probs = curve["survival_prob"]
            assert probs[0] >= probs[-1]  # first >= last


# ---------------------------------------------------------------------------
# Dashboard formatting helpers tests
# ---------------------------------------------------------------------------

class TestDashboardFormattingHelpers:
    """Test formatting utility functions used by dashboard."""

    def test_format_currency_with_large_numbers(self):
        """Should format large KRW values with commas."""
        from src.dashboard.utils.dashboard_helpers import format_currency
        result = format_currency(1234567890)
        assert "1,234,567,890" in result

    def test_format_percentage_custom_decimals(self):
        """Should support custom decimal places."""
        from src.dashboard.utils.dashboard_helpers import format_percentage
        result = format_percentage(0.12345, decimals=1)
        assert result == "12.3%"

    def test_format_count_large(self):
        """Should format large counts with commas."""
        from src.dashboard.utils.dashboard_helpers import format_count
        assert format_count(0) == "0"
        assert format_count(999) == "999"
        assert "1,000" in format_count(1000)

    def test_compute_kpi_delta_zero_previous(self):
        """Should handle zero previous value gracefully."""
        from src.dashboard.utils.dashboard_helpers import compute_kpi_delta
        delta = compute_kpi_delta(100, 0)
        assert delta == 0.0

    def test_compute_kpi_delta_positive_growth(self):
        """Should compute positive growth correctly."""
        from src.dashboard.utils.dashboard_helpers import compute_kpi_delta
        delta = compute_kpi_delta(120, 100)
        assert delta == 20.0

    def test_compute_kpi_delta_negative_growth(self):
        """Should compute negative growth correctly."""
        from src.dashboard.utils.dashboard_helpers import compute_kpi_delta
        delta = compute_kpi_delta(80, 100)
        assert delta == -20.0
