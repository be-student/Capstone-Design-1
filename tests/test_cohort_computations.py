"""
TDD Tests for Cohort Analysis Module - Extended Computations.

Tests cover:
- Cumulative revenue computation
- Full analysis pipeline
- Retention curve properties (monotonicity, bounds)
- Churn rate edge cases
- Cohort filter interactions
- Visualization methods (heatmap, line plot, cohort sizes)
- Multi-cohort comparative analysis
- Config-driven behavior
- Empty/sparse data handling
- Integration with CohortAnalyzer config
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.cohort_analysis import CohortAnalyzer, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_month_events() -> pd.DataFrame:
    """Create event data spanning 6 months with known retention patterns."""
    np.random.seed(42)
    data = []

    # Jan cohort: 20 customers, 50% retain M2, 25% retain M3
    for cid in range(1, 21):
        # All active in Jan
        for _ in range(np.random.randint(1, 4)):
            day = np.random.randint(1, 28)
            data.append({
                "customer_id": f"C{cid:04d}",
                "event_date": pd.Timestamp(2024, 1, day),
                "revenue": np.random.uniform(5000, 50000),
                "segment": "vip" if cid <= 10 else "regular",
            })
        # 50% active in Feb
        if cid <= 10:
            for _ in range(np.random.randint(1, 3)):
                day = np.random.randint(1, 28)
                data.append({
                    "customer_id": f"C{cid:04d}",
                    "event_date": pd.Timestamp(2024, 2, day),
                    "revenue": np.random.uniform(5000, 50000),
                    "segment": "vip" if cid <= 10 else "regular",
                })
        # 25% active in Mar
        if cid <= 5:
            for _ in range(np.random.randint(1, 3)):
                day = np.random.randint(1, 28)
                data.append({
                    "customer_id": f"C{cid:04d}",
                    "event_date": pd.Timestamp(2024, 3, day),
                    "revenue": np.random.uniform(5000, 50000),
                    "segment": "vip",
                })

    # Feb cohort: 15 customers
    for cid in range(21, 36):
        for _ in range(np.random.randint(1, 4)):
            day = np.random.randint(1, 28)
            data.append({
                "customer_id": f"C{cid:04d}",
                "event_date": pd.Timestamp(2024, 2, day),
                "revenue": np.random.uniform(3000, 30000),
                "segment": "new_customer",
            })
        # 40% active in Mar
        if cid <= 27:
            for _ in range(np.random.randint(1, 2)):
                day = np.random.randint(1, 28)
                data.append({
                    "customer_id": f"C{cid:04d}",
                    "event_date": pd.Timestamp(2024, 3, day),
                    "revenue": np.random.uniform(3000, 30000),
                    "segment": "new_customer",
                })

    # Mar cohort: 10 customers (single month)
    for cid in range(36, 46):
        for _ in range(np.random.randint(1, 3)):
            day = np.random.randint(1, 28)
            data.append({
                "customer_id": f"C{cid:04d}",
                "event_date": pd.Timestamp(2024, 3, day),
                "revenue": np.random.uniform(2000, 20000),
                "segment": "bargain_hunter",
            })

    return pd.DataFrame(data)


@pytest.fixture
def analyzer() -> CohortAnalyzer:
    """CohortAnalyzer with min_cohort_size=1 for testing."""
    return CohortAnalyzer(config={"min_cohort_size": 1})


@pytest.fixture
def assigned_data(analyzer, multi_month_events):
    """Pre-assigned monthly cohort data."""
    return analyzer.assign_cohorts(multi_month_events, cohort_type="monthly")


@pytest.fixture
def retention_matrix(analyzer, assigned_data):
    """Pre-computed retention matrix."""
    return analyzer.compute_retention_matrix(assigned_data)


# ---------------------------------------------------------------------------
# Cumulative Revenue Tests
# ---------------------------------------------------------------------------

class TestCumulativeRevenue:
    """Tests for compute_cumulative_revenue method."""

    def test_returns_dataframe(self, analyzer, assigned_data):
        """Should return a DataFrame."""
        cum_rev = analyzer.compute_cumulative_revenue(assigned_data)
        assert isinstance(cum_rev, pd.DataFrame)

    def test_cumulative_is_monotonically_increasing(self, analyzer, assigned_data):
        """Cumulative revenue should be monotonically increasing per cohort."""
        cum_rev = analyzer.compute_cumulative_revenue(assigned_data)
        for cohort_label in cum_rev.index:
            row = cum_rev.loc[cohort_label].dropna().values
            # Each value should be >= previous
            for i in range(1, len(row)):
                assert row[i] >= row[i-1], (
                    f"Cumulative revenue decreased for {cohort_label} at period {i}"
                )

    def test_cumulative_period_zero_equals_period_revenue(self, analyzer, assigned_data):
        """Period 0 cumulative should equal period 0 revenue."""
        metrics = analyzer.compute_cohort_metrics(assigned_data, metrics=["revenue"])
        cum_rev = analyzer.compute_cumulative_revenue(assigned_data)
        if "revenue" in metrics and 0 in metrics["revenue"].columns and 0 in cum_rev.columns:
            common = metrics["revenue"].index.intersection(cum_rev.index)
            for cohort in common:
                np.testing.assert_almost_equal(
                    cum_rev.loc[cohort, 0],
                    metrics["revenue"].loc[cohort, 0],
                    decimal=2,
                )

    def test_cumulative_all_values_non_negative(self, analyzer, assigned_data):
        """All cumulative revenue values should be >= 0."""
        cum_rev = analyzer.compute_cumulative_revenue(assigned_data)
        assert (cum_rev.values >= 0).all()

    def test_missing_revenue_raises(self, analyzer):
        """Should raise ValueError if revenue column missing."""
        data = pd.DataFrame({
            "customer_id": ["C1"],
            "event_date": [pd.Timestamp("2024-01-01")],
        })
        cohort_data = analyzer.assign_cohorts(data, cohort_type="monthly")
        with pytest.raises(ValueError, match="Revenue column"):
            analyzer.compute_cumulative_revenue(cohort_data)

    def test_max_periods_limits_columns(self, analyzer, assigned_data):
        """max_periods should limit columns in cumulative revenue."""
        cum_rev = analyzer.compute_cumulative_revenue(assigned_data, max_periods=1)
        assert all(c <= 1 for c in cum_rev.columns)


# ---------------------------------------------------------------------------
# Full Analysis Pipeline Tests
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    """Tests for the full_analysis convenience method."""

    def test_returns_dict(self, analyzer, multi_month_events):
        """Should return a dict with all expected keys."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result, dict)

    def test_has_all_required_keys(self, analyzer, multi_month_events):
        """Result should contain all expected analysis outputs."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        expected_keys = {
            "cohort_data", "retention_matrix", "retention_curves",
            "avg_retention_curve", "churn_rates", "milestone_retention", "half_life",
            "summary", "metrics",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_milestone_retention_is_dataframe(self, analyzer, multi_month_events):
        """full_analysis should expose milestone retention table."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result["milestone_retention"], pd.DataFrame)
        assert "M1" in result["milestone_retention"].columns

    def test_cohort_data_has_assignments(self, analyzer, multi_month_events):
        """cohort_data should have cohort and cohort_period columns."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        df = result["cohort_data"]
        assert "cohort" in df.columns
        assert "cohort_period" in df.columns

    def test_retention_matrix_is_dataframe(self, analyzer, multi_month_events):
        """retention_matrix should be a DataFrame."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result["retention_matrix"], pd.DataFrame)

    def test_retention_curves_is_dict(self, analyzer, multi_month_events):
        """retention_curves should be a dict of numpy arrays."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        curves = result["retention_curves"]
        assert isinstance(curves, dict)
        for curve in curves.values():
            assert isinstance(curve, np.ndarray)

    def test_avg_retention_curve_is_array(self, analyzer, multi_month_events):
        """avg_retention_curve should be a numpy array."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result["avg_retention_curve"], np.ndarray)

    def test_churn_rates_is_dataframe(self, analyzer, multi_month_events):
        """churn_rates should be a DataFrame."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result["churn_rates"], pd.DataFrame)

    def test_half_life_is_series(self, analyzer, multi_month_events):
        """half_life should be a pandas Series."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result["half_life"], pd.Series)

    def test_summary_is_dataframe(self, analyzer, multi_month_events):
        """summary should be a DataFrame."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result["summary"], pd.DataFrame)

    def test_metrics_is_dict(self, analyzer, multi_month_events):
        """metrics should be a dict of DataFrames."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="monthly")
        assert isinstance(result["metrics"], dict)

    def test_full_analysis_weekly(self, analyzer, multi_month_events):
        """full_analysis should work with weekly cohort type."""
        result = analyzer.full_analysis(multi_month_events, cohort_type="weekly")
        assert len(result["cohort_data"]) > 0
        assert "-W" in result["cohort_data"]["cohort"].iloc[0]


# ---------------------------------------------------------------------------
# Retention Curve Properties Tests
# ---------------------------------------------------------------------------

class TestRetentionCurveProperties:
    """Test mathematical properties of retention curves."""

    def test_all_curves_start_at_one(self, analyzer, retention_matrix):
        """Every retention curve must start at 1.0."""
        curves = analyzer.get_retention_curves(retention_matrix)
        for label, curve in curves.items():
            assert curve[0] == pytest.approx(1.0), (
                f"Curve for {label} starts at {curve[0]}, not 1.0"
            )

    def test_all_values_bounded_0_to_1(self, analyzer, retention_matrix):
        """All retention values must be in [0, 1]."""
        curves = analyzer.get_retention_curves(retention_matrix)
        for label, curve in curves.items():
            assert (curve >= 0).all(), f"Negative retention for {label}"
            assert (curve <= 1.0 + 1e-9).all(), f"Retention > 1 for {label}"

    def test_average_curve_between_min_and_max(self, analyzer, retention_matrix):
        """Average curve should be between min and max per-cohort curves."""
        curves = analyzer.get_retention_curves(retention_matrix)
        avg = analyzer.get_average_retention_curve(retention_matrix)

        if len(curves) > 1:
            min_len = min(len(c) for c in curves.values())
            for i in range(min_len):
                vals = [c[i] for c in curves.values() if len(c) > i]
                assert avg[i] >= min(vals) - 1e-9
                assert avg[i] <= max(vals) + 1e-9


# ---------------------------------------------------------------------------
# Churn Rate Edge Cases
# ---------------------------------------------------------------------------

class TestChurnRateEdgeCases:
    """Edge case tests for churn rate computation."""

    def test_constant_retention_gives_zero_churn(self):
        """Constant retention should give zero churn (except period 0)."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        matrix = pd.DataFrame(
            [[1.0, 1.0, 1.0]],
            index=["2024-01"],
            columns=[0, 1, 2],
        )
        churn = analyzer.compute_churn_rates(matrix)
        np.testing.assert_array_almost_equal(churn.values, 0.0)

    def test_complete_churn_gives_one(self):
        """Dropping to zero should give churn rate of 1.0."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        matrix = pd.DataFrame(
            [[1.0, 0.0]],
            index=["2024-01"],
            columns=[0, 1],
        )
        churn = analyzer.compute_churn_rates(matrix)
        assert churn.loc["2024-01", 0] == 0.0
        assert churn.loc["2024-01", 1] == pytest.approx(1.0)

    def test_churn_matches_retention_complement(self, analyzer, assigned_data):
        """churn(t) = 1 - retention(t) / retention(t-1), consistent computation."""
        matrix = analyzer.compute_retention_matrix(assigned_data)
        churn = analyzer.compute_churn_rates(matrix)
        for cohort in matrix.index:
            cols = sorted(matrix.columns)
            for i in range(1, len(cols)):
                t_curr = cols[i]
                t_prev = cols[i-1]
                ret_curr = matrix.loc[cohort, t_curr]
                ret_prev = matrix.loc[cohort, t_prev]
                if ret_prev > 0:
                    expected = 1.0 - ret_curr / ret_prev
                    np.testing.assert_almost_equal(
                        churn.loc[cohort, t_curr], expected, decimal=5,
                    )


# ---------------------------------------------------------------------------
# Visualization Tests
# ---------------------------------------------------------------------------

class TestVisualizationMethods:
    """Test visualization methods produce valid matplotlib Figures."""

    def test_retention_heatmap_returns_figure(self, analyzer, retention_matrix):
        """plot_retention_heatmap should return a matplotlib Figure."""
        fig = analyzer.plot_retention_heatmap(retention_matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_retention_heatmap_saves_to_file(self, analyzer, retention_matrix, tmp_path):
        """plot_retention_heatmap should save when save_path provided."""
        save_path = str(tmp_path / "heatmap.png")
        fig = analyzer.plot_retention_heatmap(retention_matrix, save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_retention_lines_returns_figure(self, analyzer, retention_matrix):
        """plot_retention_lines should return a matplotlib Figure."""
        fig = analyzer.plot_retention_lines(retention_matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_retention_lines_with_average(self, analyzer, retention_matrix):
        """plot_retention_lines should include average line when show_average=True."""
        fig = analyzer.plot_retention_lines(retention_matrix, show_average=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_retention_lines_without_average(self, analyzer, retention_matrix):
        """plot_retention_lines should work without average line."""
        fig = analyzer.plot_retention_lines(retention_matrix, show_average=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_retention_lines_saves_to_file(self, analyzer, retention_matrix, tmp_path):
        """plot_retention_lines should save when save_path provided."""
        save_path = str(tmp_path / "lines.png")
        fig = analyzer.plot_retention_lines(retention_matrix, save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_cohort_sizes_returns_figure(self, analyzer, assigned_data):
        """plot_cohort_sizes should return a matplotlib Figure."""
        fig = analyzer.plot_cohort_sizes(assigned_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cohort_sizes_saves_to_file(self, analyzer, assigned_data, tmp_path):
        """plot_cohort_sizes should save when save_path provided."""
        save_path = str(tmp_path / "sizes.png")
        fig = analyzer.plot_cohort_sizes(assigned_data, save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_cohort_sizes_custom_color(self, analyzer, assigned_data):
        """plot_cohort_sizes should accept custom bar color."""
        fig = analyzer.plot_cohort_sizes(assigned_data, color="#ff5733")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-Cohort Comparative Tests
# ---------------------------------------------------------------------------

class TestMultiCohortComparison:
    """Tests for comparing multiple cohorts."""

    def test_jan_cohort_largest(self, analyzer, assigned_data):
        """January cohort should be the largest (20 customers)."""
        summary = analyzer.get_cohort_summary(assigned_data)
        jan = summary[summary["cohort"] == "2024-01"]
        assert len(jan) == 1
        assert jan["total_customers"].iloc[0] == 20

    def test_feb_cohort_size(self, analyzer, assigned_data):
        """February cohort should have 15 customers."""
        summary = analyzer.get_cohort_summary(assigned_data)
        feb = summary[summary["cohort"] == "2024-02"]
        assert len(feb) == 1
        assert feb["total_customers"].iloc[0] == 15

    def test_mar_cohort_size(self, analyzer, assigned_data):
        """March cohort should have 10 customers."""
        summary = analyzer.get_cohort_summary(assigned_data)
        mar = summary[summary["cohort"] == "2024-03"]
        assert len(mar) == 1
        assert mar["total_customers"].iloc[0] == 10

    def test_three_monthly_cohorts(self, analyzer, assigned_data):
        """Should have exactly 3 monthly cohorts."""
        cohorts = assigned_data["cohort"].unique()
        assert len(cohorts) == 3

    def test_jan_retention_decreases(self, analyzer, retention_matrix):
        """Jan cohort retention should decrease over periods."""
        if "2024-01" in retention_matrix.index:
            jan_retention = retention_matrix.loc["2024-01"].dropna().values
            if len(jan_retention) > 1:
                assert jan_retention[0] >= jan_retention[-1]


# ---------------------------------------------------------------------------
# Config-Driven Behavior Tests
# ---------------------------------------------------------------------------

class TestConfigDrivenBehavior:
    """Test that CohortAnalyzer respects configuration."""

    def test_custom_date_column(self):
        """Should use custom date column name from config."""
        config = {"date_column": "purchase_date", "customer_column": "user_id", "min_cohort_size": 1}
        analyzer = CohortAnalyzer(config=config)
        data = pd.DataFrame({
            "user_id": ["U1", "U1", "U2"],
            "purchase_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15"]),
            "revenue": [100, 200, 150],
        })
        result = analyzer.assign_cohorts(data, cohort_type="monthly")
        assert "cohort" in result.columns

    def test_custom_customer_column(self):
        """Should use custom customer column name from config."""
        config = {"customer_column": "user_id", "min_cohort_size": 1}
        analyzer = CohortAnalyzer(config=config)
        data = pd.DataFrame({
            "user_id": ["U1", "U2"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
        })
        result = analyzer.assign_cohorts(data, cohort_type="monthly")
        assert len(result) == 2

    def test_custom_revenue_column(self):
        """Should use custom revenue column name from config."""
        config = {"revenue_column": "amount", "min_cohort_size": 1}
        analyzer = CohortAnalyzer(config=config)
        data = pd.DataFrame({
            "customer_id": ["C1", "C1"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "amount": [100, 200],
        })
        cohort_data = analyzer.assign_cohorts(data, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(cohort_data, metrics=["revenue"])
        assert "revenue" in metrics

    def test_min_cohort_size_filtering(self, multi_month_events):
        """min_cohort_size should filter out small cohorts from retention matrix."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 16})
        cohort_data = analyzer.assign_cohorts(multi_month_events, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(cohort_data)
        # Only Jan cohort (20) should survive
        assert len(retention) == 1

    def test_periods_config_limits_columns(self, multi_month_events):
        """periods config should limit retention matrix columns."""
        analyzer = CohortAnalyzer(config={"periods": 1, "min_cohort_size": 1})
        cohort_data = analyzer.assign_cohorts(multi_month_events, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(cohort_data)
        assert all(c <= 1 for c in retention.columns)


# ---------------------------------------------------------------------------
# Sparse / Empty Data Edge Cases
# ---------------------------------------------------------------------------

class TestSparseAndEmptyData:
    """Edge cases with sparse or minimal data."""

    def test_single_event_single_customer(self):
        """Should handle a single event from a single customer."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        data = pd.DataFrame({
            "customer_id": ["C1"],
            "event_date": [pd.Timestamp("2024-06-15")],
            "revenue": [10000],
        })
        result = analyzer.assign_cohorts(data, cohort_type="monthly")
        assert len(result) == 1
        assert result["cohort_period"].iloc[0] == 0

    def test_all_same_month(self):
        """Should handle all events in the same month."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        data = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "event_date": [pd.Timestamp("2024-01-05"),
                           pd.Timestamp("2024-01-15"),
                           pd.Timestamp("2024-01-25")],
            "revenue": [100, 200, 300],
        })
        result = analyzer.assign_cohorts(data, cohort_type="monthly")
        assert (result["cohort_period"] == 0).all()
        retention = analyzer.compute_retention_matrix(result)
        assert len(retention) == 1
        if 0 in retention.columns:
            assert retention.iloc[0, 0] == pytest.approx(1.0)

    def test_single_customer_multiple_months(self):
        """Single customer across multiple months should create one cohort."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        data = pd.DataFrame({
            "customer_id": ["C1", "C1", "C1"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "revenue": [100, 200, 300],
        })
        result = analyzer.assign_cohorts(data, cohort_type="monthly")
        assert result["cohort"].nunique() == 1
        assert result["cohort_period"].max() == 2

    def test_avg_order_value_metric(self):
        """avg_order_value metric should work correctly."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        data = pd.DataFrame({
            "customer_id": ["C1", "C1", "C2", "C2"],
            "event_date": pd.to_datetime([
                "2024-01-01", "2024-01-15", "2024-01-05", "2024-02-05"
            ]),
            "revenue": [100, 200, 300, 400],
        })
        cohort_data = analyzer.assign_cohorts(data, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(
            cohort_data, metrics=["avg_order_value"]
        )
        assert "avg_order_value" in metrics
        aov = metrics["avg_order_value"]
        assert (aov.values >= 0).all()


# ---------------------------------------------------------------------------
# Integration with Dashboard Data Loader
# ---------------------------------------------------------------------------

class TestCohortDashboardIntegration:
    """Test cohort analysis integration with dashboard data pipeline."""

    def test_cohort_data_from_loader_is_analyzable(self):
        """Cohort data from DashboardDataLoader should work with CohortAnalyzer."""
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "simulator_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader(config)
        cohort_data = loader.load_cohort_data()

        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
        assert "cohort" in assigned.columns

        retention = analyzer.compute_retention_matrix(assigned)
        assert isinstance(retention, pd.DataFrame)
        assert len(retention) > 0

    def test_full_analysis_on_loader_data(self):
        """Full analysis pipeline should work on dashboard loader data."""
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "simulator_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        from src.dashboard.data_loader import DashboardDataLoader
        loader = DashboardDataLoader(config)
        cohort_data = loader.load_cohort_data()

        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        result = analyzer.full_analysis(cohort_data, cohort_type="monthly")

        assert "retention_matrix" in result
        assert "retention_curves" in result
        assert "summary" in result
        assert len(result["summary"]) > 0
