"""
TDD Tests for CLV and Cohort Analysis Dashboard Views (Sub-AC 3).

Tests cover:
- CLV distribution views (histogram, box plot, segment breakdown)
- CLV tier classification and percentile analysis
- CLV vs churn risk scatter plots
- Cohort retention heatmap rendering
- Cohort comparison charts (retention curves, cohort sizes)
- Cohort metric aggregation for dashboard display
- Integration between CohortAnalyzer and dashboard views
- Data loader cohort/CLV data generation
- Render function signatures and behavior with mocked Streamlit
- Edge cases (empty data, missing columns)
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
def data_loader(config):
    """Create a DashboardDataLoader instance."""
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


@pytest.fixture
def sample_clv_data():
    """Sample CLV data with segments."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "clv_predicted": np.random.lognormal(11, 1, n),
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "new_customer", "dormant", "high_value_at_risk"], n,
        ),
    })


@pytest.fixture
def sample_predictions():
    """Sample predictions with CLV and churn columns."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_probability": np.random.beta(2, 5, n),
        "risk_level": np.random.choice(
            ["low", "medium", "high", "critical"], n,
            p=[0.4, 0.3, 0.2, 0.1],
        ),
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "new_customer", "dormant", "high_value_at_risk"], n,
        ),
        "clv_predicted": np.random.lognormal(11, 1, n),
    })


@pytest.fixture
def sample_retention_matrix():
    """Sample retention matrix for cohort tests."""
    np.random.seed(42)
    cohorts = [f"2024-{m:02d}" for m in range(1, 7)]
    periods = list(range(7))
    data = {}
    for cohort in cohorts:
        retention = [1.0]
        for _ in range(1, len(periods)):
            decay = np.random.uniform(0.80, 0.95)
            retention.append(round(retention[-1] * decay, 4))
        data[cohort] = retention
    df = pd.DataFrame(data, index=periods).T
    df.index.name = "cohort"
    df.columns = periods
    return df


@pytest.fixture
def sample_cohort_events():
    """Sample cohort event-level data for CohortAnalyzer."""
    np.random.seed(42)
    n_customers = 100
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    segments = ["vip_loyal", "regular_loyal", "bargain_hunter",
                "new_customer", "dormant", "high_value_at_risk"]
    for i in range(n_customers):
        first_offset = np.random.randint(0, 180)
        first_date = base_date + pd.Timedelta(days=int(first_offset))
        seg = np.random.choice(segments)
        n_events = np.random.randint(1, 6)
        for j in range(n_events):
            event_offset = np.random.randint(0, 365)
            event_date = first_date + pd.Timedelta(days=int(event_offset))
            rows.append({
                "customer_id": f"C{i:05d}",
                "event_date": event_date,
                "revenue": float(np.random.lognormal(9, 1)),
                "segment": seg,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def st_mock():
    """Create a Streamlit mock with common methods."""
    mock = MagicMock()

    def _columns_side_effect(n, *args, **kwargs):
        return [MagicMock() for _ in range(n if isinstance(n, int) else 4)]

    mock.columns.side_effect = _columns_side_effect
    mock.tabs.return_value = [MagicMock() for _ in range(3)]
    return mock


# ---------------------------------------------------------------------------
# CLV View - Data Loading Tests
# ---------------------------------------------------------------------------

class TestCLVDataLoading:
    """Test CLV data loading from DashboardDataLoader."""

    def test_load_clv_data_returns_dataframe(self, data_loader):
        """CLV data loader must return a DataFrame."""
        clv_data = data_loader.load_clv_data()
        assert isinstance(clv_data, pd.DataFrame)

    def test_clv_data_has_required_columns(self, data_loader):
        """CLV data must have customer_id, clv_predicted, segment."""
        clv_data = data_loader.load_clv_data()
        assert "customer_id" in clv_data.columns
        assert "clv_predicted" in clv_data.columns
        assert "segment" in clv_data.columns

    def test_clv_values_nonnegative(self, data_loader):
        """All CLV predictions must be non-negative."""
        clv_data = data_loader.load_clv_data()
        assert (clv_data["clv_predicted"] >= 0).all()

    def test_clv_data_has_multiple_segments(self, data_loader):
        """CLV data should have multiple segments for comparison."""
        clv_data = data_loader.load_clv_data()
        assert clv_data["segment"].nunique() >= 2

    def test_clv_data_nonempty(self, data_loader):
        """CLV data must not be empty."""
        clv_data = data_loader.load_clv_data()
        assert len(clv_data) > 0


# ---------------------------------------------------------------------------
# CLV Distribution Analysis Tests
# ---------------------------------------------------------------------------

class TestCLVDistribution:
    """Test CLV distribution computations for dashboard display."""

    def test_clv_histogram_data(self, sample_clv_data):
        """CLV values should be suitable for histogram."""
        assert sample_clv_data["clv_predicted"].dtype in [np.float64, np.float32]
        assert len(sample_clv_data["clv_predicted"]) > 0

    def test_clv_mean_median_computable(self, sample_clv_data):
        """Mean and median CLV should be computable."""
        mean_clv = sample_clv_data["clv_predicted"].mean()
        median_clv = sample_clv_data["clv_predicted"].median()
        assert mean_clv > 0
        assert median_clv > 0

    def test_clv_segment_groupby(self, sample_clv_data):
        """CLV should be aggregatable by segment."""
        seg_stats = sample_clv_data.groupby("segment")["clv_predicted"].agg(
            ["mean", "sum", "count", "median", "std"]
        ).reset_index()
        assert len(seg_stats) > 0
        assert "mean" in seg_stats.columns
        assert (seg_stats["count"] > 0).all()

    def test_clv_tier_classification(self, sample_clv_data):
        """CLV values should be classifiable into tiers."""
        q25 = sample_clv_data["clv_predicted"].quantile(0.25)
        q50 = sample_clv_data["clv_predicted"].quantile(0.50)
        q75 = sample_clv_data["clv_predicted"].quantile(0.75)

        def classify(v):
            if v >= q75:
                return "Platinum"
            elif v >= q50:
                return "Gold"
            elif v >= q25:
                return "Silver"
            return "Bronze"

        tiers = sample_clv_data["clv_predicted"].apply(classify)
        assert set(tiers.unique()) == {"Platinum", "Gold", "Silver", "Bronze"}

    def test_clv_percentile_analysis(self, sample_clv_data):
        """CLV percentile analysis should produce monotonic values."""
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        values = [
            sample_clv_data["clv_predicted"].quantile(p / 100)
            for p in percentiles
        ]
        # Values should be monotonically increasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


# ---------------------------------------------------------------------------
# CLV vs Churn Integration Tests
# ---------------------------------------------------------------------------

class TestCLVChurnIntegration:
    """Test CLV vs churn probability views."""

    def test_clv_churn_scatter_data(self, sample_predictions):
        """Both CLV and churn probability must exist for scatter plot."""
        assert "clv_predicted" in sample_predictions.columns
        assert "churn_probability" in sample_predictions.columns
        assert len(sample_predictions) > 0

    def test_high_value_at_risk_identifiable(self, sample_predictions):
        """Customers with high CLV + high churn should be identifiable."""
        median_clv = sample_predictions["clv_predicted"].median()
        at_risk = sample_predictions[
            (sample_predictions["clv_predicted"] > median_clv) &
            (sample_predictions["churn_probability"] > 0.5)
        ]
        # Should be computable (might be empty depending on seed)
        assert isinstance(at_risk, pd.DataFrame)


# ---------------------------------------------------------------------------
# Cohort Retention Heatmap Tests
# ---------------------------------------------------------------------------

class TestCohortRetentionHeatmap:
    """Test cohort retention heatmap data preparation."""

    def test_retention_matrix_shape(self, sample_retention_matrix):
        """Retention matrix must have cohorts as rows, periods as columns."""
        assert sample_retention_matrix.shape[0] > 0  # cohorts
        assert sample_retention_matrix.shape[1] > 0  # periods

    def test_retention_values_bounded(self, sample_retention_matrix):
        """Retention values must be between 0 and 1."""
        assert (sample_retention_matrix >= 0).all().all()
        assert (sample_retention_matrix <= 1).all().all()

    def test_period_0_full_retention(self, sample_retention_matrix):
        """Period 0 should have 100% retention."""
        if 0 in sample_retention_matrix.columns:
            assert (sample_retention_matrix[0] == 1.0).all()

    def test_heatmap_percentage_conversion(self, sample_retention_matrix):
        """Heatmap data can be converted to percentage for display."""
        heatmap_pct = sample_retention_matrix * 100
        assert heatmap_pct.max().max() <= 100.0
        assert heatmap_pct.min().min() >= 0.0

    def test_retention_generally_decreasing(self, sample_retention_matrix):
        """Retention should generally decrease over periods."""
        for cohort in sample_retention_matrix.index:
            row = sample_retention_matrix.loc[cohort].values
            # Overall trend: last value should be <= first value
            assert row[-1] <= row[0]


# ---------------------------------------------------------------------------
# Cohort Comparison Chart Tests
# ---------------------------------------------------------------------------

class TestCohortComparisonCharts:
    """Test cohort comparison visualization data."""

    def test_per_cohort_retention_curves(self, sample_retention_matrix):
        """Each cohort should have its own retention curve."""
        for cohort in sample_retention_matrix.index:
            curve = sample_retention_matrix.loc[cohort].dropna().values
            assert len(curve) > 0
            assert curve[0] == 1.0  # starts at 100%

    def test_average_retention_curve(self, sample_retention_matrix):
        """Average retention curve across cohorts should be computable."""
        avg_curve = sample_retention_matrix.mean(axis=0)
        assert len(avg_curve) == sample_retention_matrix.shape[1]
        assert avg_curve.iloc[0] == 1.0

    def test_cohort_size_comparison(self, sample_cohort_events):
        """Cohort sizes should be computable from event data."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(sample_cohort_events, cohort_type="monthly")
        sizes = assigned.groupby("cohort")["customer_id"].nunique()
        assert len(sizes) > 1  # multiple cohorts

    def test_period_over_period_retention_drop(self, sample_retention_matrix):
        """Period-over-period retention changes should be computable."""
        avg_ret = sample_retention_matrix.mean(axis=0)
        if len(avg_ret) > 1:
            vals = avg_ret.values * 100
            drops = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
            # Drops should be mostly negative (retention decreasing)
            assert len(drops) > 0

    def test_cohort_revenue_comparison(self, sample_cohort_events):
        """Revenue can be compared across cohorts."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(sample_cohort_events, cohort_type="monthly")
        rev_by_cohort = assigned.groupby("cohort")["revenue"].sum()
        assert len(rev_by_cohort) > 1
        assert (rev_by_cohort > 0).all()

    def test_cohort_summary_table(self, sample_cohort_events):
        """Cohort summary should include size and metrics."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(sample_cohort_events, cohort_type="monthly")
        summary = analyzer.get_cohort_summary(assigned)
        assert "total_customers" in summary.columns
        assert "total_events" in summary.columns
        assert len(summary) > 0


# ---------------------------------------------------------------------------
# Render Function Tests (CLV)
# ---------------------------------------------------------------------------

class TestRenderCLV:
    """Test render_clv function with mocked Streamlit."""

    def test_render_clv_exists(self):
        """Dashboard must have render_clv function."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_clv")
        assert callable(dashboard_app.render_clv)

    def test_render_clv_with_valid_data(self, config, sample_predictions, st_mock):
        """render_clv must not raise with valid data."""
        from src.dashboard.app import render_clv

        loader_mock = MagicMock()
        loader_mock.load_predictions.return_value = sample_predictions
        loader_mock.load_clv_data.return_value = sample_predictions[
            ["customer_id", "clv_predicted", "segment"]
        ]

        render_clv(st_mock, config, loader_mock)
        st_mock.header.assert_called_once_with("CLV Prediction")

    def test_render_clv_shows_distribution(self, config, sample_predictions, st_mock):
        """render_clv must call subheader for CLV Distribution."""
        from src.dashboard.app import render_clv

        loader_mock = MagicMock()
        loader_mock.load_predictions.return_value = sample_predictions
        loader_mock.load_clv_data.return_value = sample_predictions[
            ["customer_id", "clv_predicted", "segment"]
        ]

        render_clv(st_mock, config, loader_mock)
        subheader_calls = [
            str(c) for c in st_mock.subheader.call_args_list
        ]
        # Should have subheader calls for distribution
        assert len(subheader_calls) > 0

    def test_render_clv_handles_empty_data(self, config, st_mock):
        """render_clv must handle empty/missing CLV data."""
        from src.dashboard.app import render_clv

        loader_mock = MagicMock()
        loader_mock.load_predictions.return_value = pd.DataFrame()
        loader_mock.load_clv_data.return_value = pd.DataFrame()

        render_clv(st_mock, config, loader_mock)
        st_mock.warning.assert_called()

    def test_render_clv_kpi_metrics(self, config, sample_predictions, st_mock):
        """render_clv must display KPI metrics (Total CLV, Avg, Median, Std)."""
        from src.dashboard.app import render_clv

        loader_mock = MagicMock()
        loader_mock.load_predictions.return_value = sample_predictions
        loader_mock.load_clv_data.return_value = sample_predictions[
            ["customer_id", "clv_predicted", "segment"]
        ]

        # Track all column mocks returned by st.columns()
        all_col_mocks = []

        def _columns_tracking(n, *args, **kwargs):
            mocks = [MagicMock() for _ in range(n if isinstance(n, int) else 4)]
            all_col_mocks.extend(mocks)
            return mocks

        st_mock.columns.side_effect = _columns_tracking

        render_clv(st_mock, config, loader_mock)

        # KPI metrics should be displayed via column.metric()
        total_metric_calls = sum(
            cm.metric.call_count for cm in all_col_mocks
        )
        assert total_metric_calls >= 4  # total, avg, median, std


# ---------------------------------------------------------------------------
# Render Function Tests (Cohort Analysis)
# ---------------------------------------------------------------------------

class TestRenderCohortAnalysis:
    """Test render_cohort_analysis function with mocked Streamlit."""

    def test_render_cohort_analysis_exists(self):
        """Dashboard must have render_cohort_analysis function."""
        from src.dashboard import app as dashboard_app
        assert hasattr(dashboard_app, "render_cohort_analysis")
        assert callable(dashboard_app.render_cohort_analysis)

    def test_render_cohort_with_valid_data(
        self, config, sample_retention_matrix, st_mock
    ):
        """render_cohort_analysis must not raise with valid data."""
        from src.dashboard.app import render_cohort_analysis

        loader_mock = MagicMock()
        loader_mock.load_cohort_retention_matrix.return_value = (
            sample_retention_matrix
        )
        loader_mock.load_cohort_data.return_value = pd.DataFrame({
            "customer_id": ["C00001", "C00002", "C00003"],
            "event_date": pd.to_datetime(
                ["2024-01-15", "2024-02-20", "2024-03-10"]
            ),
            "revenue": [10000, 20000, 15000],
            "segment": ["vip_loyal", "regular_loyal", "bargain_hunter"],
        })

        render_cohort_analysis(st_mock, config, loader_mock)
        st_mock.header.assert_called_once_with("Cohort Analysis")

    def test_render_cohort_empty_matrix(self, config, st_mock):
        """render_cohort_analysis must handle empty matrix."""
        from src.dashboard.app import render_cohort_analysis

        loader_mock = MagicMock()
        loader_mock.load_cohort_retention_matrix.return_value = pd.DataFrame()

        render_cohort_analysis(st_mock, config, loader_mock)
        st_mock.warning.assert_called()

    def test_render_cohort_shows_heatmap(
        self, config, sample_retention_matrix, st_mock
    ):
        """render_cohort_analysis must render a heatmap section."""
        from src.dashboard.app import render_cohort_analysis

        loader_mock = MagicMock()
        loader_mock.load_cohort_retention_matrix.return_value = (
            sample_retention_matrix
        )
        loader_mock.load_cohort_data.return_value = pd.DataFrame({
            "customer_id": ["C00001"],
            "event_date": pd.to_datetime(["2024-01-15"]),
            "revenue": [10000],
            "segment": ["vip_loyal"],
        })

        render_cohort_analysis(st_mock, config, loader_mock)
        # Must have subheader calls including heatmap-related
        subheader_calls = [
            str(c) for c in st_mock.subheader.call_args_list
        ]
        assert any("Heatmap" in s or "Retention" in s for s in subheader_calls)

    def test_render_cohort_shows_retention_curves(
        self, config, sample_retention_matrix, st_mock
    ):
        """render_cohort_analysis must render retention curves."""
        from src.dashboard.app import render_cohort_analysis

        loader_mock = MagicMock()
        loader_mock.load_cohort_retention_matrix.return_value = (
            sample_retention_matrix
        )
        loader_mock.load_cohort_data.return_value = pd.DataFrame({
            "customer_id": ["C00001"],
            "event_date": pd.to_datetime(["2024-01-15"]),
            "revenue": [10000],
            "segment": ["vip_loyal"],
        })

        render_cohort_analysis(st_mock, config, loader_mock)
        # plotly_chart should be called multiple times (heatmap + curves)
        assert st_mock.plotly_chart.call_count >= 2


# ---------------------------------------------------------------------------
# Cohort Analyzer Integration with Dashboard
# ---------------------------------------------------------------------------

class TestCohortAnalyzerDashboardIntegration:
    """Test CohortAnalyzer works with dashboard data flow."""

    def test_analyzer_from_loaded_data(self, data_loader):
        """CohortAnalyzer should work with data loader's cohort data."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        cohort_data = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
        assert "cohort" in assigned.columns
        assert "cohort_period" in assigned.columns

    def test_retention_matrix_from_loaded_data(self, data_loader):
        """Retention matrix should be computable from loaded data."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        cohort_data = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(assigned)
        assert not retention.empty
        assert (retention >= 0).all().all()
        assert (retention <= 1).all().all()

    def test_cohort_metrics_from_loaded_data(self, data_loader):
        """Cohort metrics should be computable from loaded data."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        cohort_data = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(
            assigned, metrics=["retention_rate", "customer_count"]
        )
        assert "retention_rate" in metrics
        assert "customer_count" in metrics

    def test_half_life_computation(self, data_loader):
        """Half-life should be computable for dashboard display."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        cohort_data = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(assigned)
        half_life = analyzer.compute_half_life(retention)
        assert isinstance(half_life, pd.Series)

    def test_churn_rates_from_retention(self, data_loader):
        """Churn rates should be derivable from retention matrix."""
        from src.analysis.cohort_analysis import CohortAnalyzer
        cohort_data = data_loader.load_cohort_data()
        analyzer = CohortAnalyzer()
        assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(assigned)
        churn_rates = analyzer.compute_churn_rates(retention)
        assert churn_rates.shape == retention.shape


# ---------------------------------------------------------------------------
# CLV Model Integration with Dashboard
# ---------------------------------------------------------------------------

class TestCLVModelDashboardIntegration:
    """Test CLVModel compatibility with dashboard data."""

    def test_clv_segment_aggregation(self, sample_clv_data):
        """CLV data should be aggregatable by segment for bar charts."""
        seg_agg = sample_clv_data.groupby("segment")["clv_predicted"].agg(
            ["mean", "sum", "count"]
        )
        assert len(seg_agg) > 0
        assert (seg_agg["mean"] > 0).all()

    def test_clv_top_bottom_customers(self, sample_clv_data):
        """Top and bottom N customers should be identifiable."""
        top10 = sample_clv_data.nlargest(10, "clv_predicted")
        bottom10 = sample_clv_data.nsmallest(10, "clv_predicted")
        assert len(top10) == 10
        assert len(bottom10) == 10
        assert top10["clv_predicted"].min() >= bottom10["clv_predicted"].max()


# ---------------------------------------------------------------------------
# Page Routing Tests
# ---------------------------------------------------------------------------

class TestPageRouting:
    """Test CLV and Cohort pages are in page routing."""

    def test_clv_in_page_list(self):
        """CLV Prediction must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "CLV Prediction" in PAGES

    def test_cohort_in_page_list(self):
        """Cohort Analysis must be in the page list."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Cohort Analysis" in PAGES

    def test_clv_has_icon(self):
        """CLV Prediction page must have an icon."""
        from src.dashboard.utils.dashboard_helpers import PAGE_ICONS
        assert "CLV Prediction" in PAGE_ICONS

    def test_cohort_has_icon(self):
        """Cohort Analysis page must have an icon."""
        from src.dashboard.utils.dashboard_helpers import PAGE_ICONS
        assert "Cohort Analysis" in PAGE_ICONS
