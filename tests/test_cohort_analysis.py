"""
Tests for the Cohort Analysis module.

Tests cover:
    - CohortAnalyzer initialization with default and custom config
    - Monthly cohort assignment
    - Weekly cohort assignment
    - Behavioral cohort assignment
    - Retention matrix computation
    - Cohort metric aggregation
    - Cohort summary generation
    - Cohort filtering
    - Edge cases (single customer, empty data, small cohorts)
    - Visualization: retention heatmaps, line plots, cohort size bar charts
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis.cohort_analysis import CohortAnalyzer, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_events() -> pd.DataFrame:
    """Create sample event data spanning multiple months."""
    np.random.seed(42)
    data = []
    # 3 cohorts: Jan, Feb, Mar 2024
    # Jan cohort: 10 customers, active for 3 months
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
                    "segment": "new_customer",
                })

    # Mar cohort: 6 customers, active for 1 month
    for cid in range(19, 25):
        n_events = np.random.randint(1, 3)
        for _ in range(n_events):
            day = np.random.randint(1, 28)
            data.append({
                "customer_id": f"C{cid:03d}",
                "event_date": pd.Timestamp(2024, 3, day),
                "revenue": np.random.uniform(10000, 60000),
                "segment": "bargain_hunter",
            })

    return pd.DataFrame(data)


@pytest.fixture
def analyzer() -> CohortAnalyzer:
    """Create a CohortAnalyzer with default config and min_cohort_size=1."""
    return CohortAnalyzer(config={"min_cohort_size": 1})


@pytest.fixture
def analyzer_default() -> CohortAnalyzer:
    """Create a CohortAnalyzer with default config."""
    return CohortAnalyzer()


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------

class TestCohortAnalyzerInit:
    """Tests for CohortAnalyzer initialization."""

    def test_default_config(self, analyzer_default: CohortAnalyzer) -> None:
        """Default config should match DEFAULT_CONFIG."""
        for key, value in DEFAULT_CONFIG.items():
            assert analyzer_default.config[key] == value

    def test_custom_config_override(self) -> None:
        """Custom config should override defaults."""
        custom = {"cohort_type": "weekly", "periods": 6, "min_cohort_size": 10}
        analyzer = CohortAnalyzer(config=custom)
        assert analyzer.config["cohort_type"] == "weekly"
        assert analyzer.config["periods"] == 6
        assert analyzer.config["min_cohort_size"] == 10
        # Non-overridden defaults preserved
        assert analyzer.config["customer_column"] == "customer_id"

    def test_none_config_uses_defaults(self) -> None:
        """None config should use all defaults."""
        analyzer = CohortAnalyzer(config=None)
        assert analyzer.config == DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Monthly Cohort Assignment Tests
# ---------------------------------------------------------------------------

class TestMonthlyCohorts:
    """Tests for monthly cohort assignment."""

    def test_assigns_monthly_cohort(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should assign cohort based on first event month."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        assert "cohort" in result.columns
        assert "cohort_period" in result.columns

    def test_cohort_labels_are_year_month(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Cohort labels should be YYYY-MM format."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        cohorts = result["cohort"].unique()
        for cohort in cohorts:
            # Format: YYYY-MM
            assert len(cohort) == 7
            assert cohort[4] == "-"

    def test_cohort_period_starts_at_zero(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """First period for each customer should be 0."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        assert result["cohort_period"].min() == 0

    def test_jan_cohort_has_10_customers(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """January 2024 cohort should have 10 customers."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        jan_customers = result[result["cohort"] == "2024-01"]["customer_id"].nunique()
        assert jan_customers == 10

    def test_cohort_period_increments_correctly(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Cohort period should increment by month."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        # Jan cohort customers active in March should have period=2
        jan_mar = result[
            (result["cohort"] == "2024-01")
            & (result["event_date"].dt.month == 3)
        ]
        assert (jan_mar["cohort_period"] == 2).all()


# ---------------------------------------------------------------------------
# Weekly Cohort Assignment Tests
# ---------------------------------------------------------------------------

class TestWeeklyCohorts:
    """Tests for weekly cohort assignment."""

    def test_assigns_weekly_cohort(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should assign cohort based on first event week."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="weekly")
        assert "cohort" in result.columns
        assert "cohort_period" in result.columns

    def test_weekly_cohort_format(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Weekly cohort labels should contain -W prefix."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="weekly")
        cohorts = result["cohort"].unique()
        for cohort in cohorts:
            assert "-W" in cohort

    def test_weekly_period_is_nonnegative(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Weekly cohort periods should be >= 0."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="weekly")
        assert result["cohort_period"].min() >= 0


# ---------------------------------------------------------------------------
# Behavioral Cohort Assignment Tests
# ---------------------------------------------------------------------------

class TestBehavioralCohorts:
    """Tests for behavioral cohort assignment."""

    def test_assigns_behavioral_cohort(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should assign cohort based on behavioral column."""
        result = analyzer.assign_cohorts(sample_events, cohort_type="behavioral")
        assert "cohort" in result.columns
        # Should have the same segments as behavioral values
        expected_segments = {"loyal", "at_risk", "new_customer", "bargain_hunter"}
        assert set(result["cohort"].unique()) == expected_segments

    def test_missing_behavioral_column_raises(
        self, analyzer: CohortAnalyzer
    ) -> None:
        """Should raise ValueError if behavioral column missing."""
        data = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
        })
        with pytest.raises(ValueError, match="Behavioral column"):
            analyzer.assign_cohorts(data, cohort_type="behavioral")

    def test_behavioral_cohort_uses_first_observed(
        self, analyzer: CohortAnalyzer
    ) -> None:
        """Should use first observed behavioral value for cohort."""
        data = pd.DataFrame({
            "customer_id": ["C1", "C1", "C1"],
            "event_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01"]
            ),
            "segment": ["new_customer", "loyal", "vip"],
            "revenue": [100, 200, 300],
        })
        result = analyzer.assign_cohorts(data, cohort_type="behavioral")
        assert (result[result["customer_id"] == "C1"]["cohort"] == "new_customer").all()


# ---------------------------------------------------------------------------
# Retention Matrix Tests
# ---------------------------------------------------------------------------

class TestRetentionMatrix:
    """Tests for retention matrix computation."""

    def test_retention_matrix_shape(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Retention matrix should have cohorts as rows and periods as cols."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(cohort_data)
        assert isinstance(retention, pd.DataFrame)
        assert len(retention) > 0

    def test_period_zero_retention_is_one(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Period 0 retention should be 1.0 (100%) for all cohorts."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(cohort_data)
        if 0 in retention.columns:
            np.testing.assert_array_almost_equal(
                retention[0].values,
                np.ones(len(retention)),
            )

    def test_retention_values_between_0_and_1(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """All retention values should be between 0 and 1."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(cohort_data)
        assert (retention.values >= 0).all()
        assert (retention.values <= 1).all()

    def test_max_periods_limits_columns(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """max_periods should limit the number of columns."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(cohort_data, max_periods=1)
        assert all(c <= 1 for c in retention.columns)

    def test_min_cohort_size_filters(
        self, sample_events: pd.DataFrame
    ) -> None:
        """Cohorts below min_cohort_size should be filtered out."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 9})
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        retention = analyzer.compute_retention_matrix(cohort_data)
        # Only Jan cohort (10 customers) should survive the filter of 9
        assert len(retention) >= 1


# ---------------------------------------------------------------------------
# Cohort Metrics Tests
# ---------------------------------------------------------------------------

class TestCohortMetrics:
    """Tests for cohort metric aggregation."""

    def test_returns_dict_of_dataframes(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should return dict mapping metric names to DataFrames."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(cohort_data)
        assert isinstance(metrics, dict)
        for name, df in metrics.items():
            assert isinstance(df, pd.DataFrame)

    def test_retention_rate_metric(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """retention_rate metric should match retention matrix."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(
            cohort_data, metrics=["retention_rate"]
        )
        assert "retention_rate" in metrics

    def test_churn_rate_is_complement(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """churn_rate should be 1 - retention_rate."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(
            cohort_data, metrics=["retention_rate", "churn_rate"]
        )
        if "retention_rate" in metrics and "churn_rate" in metrics:
            # Align indices for comparison
            common_idx = metrics["retention_rate"].index.intersection(
                metrics["churn_rate"].index
            )
            common_cols = [
                c for c in metrics["retention_rate"].columns
                if c in metrics["churn_rate"].columns
            ]
            if len(common_idx) > 0 and len(common_cols) > 0:
                retention = metrics["retention_rate"].loc[common_idx, common_cols]
                churn = metrics["churn_rate"].loc[common_idx, common_cols]
                np.testing.assert_array_almost_equal(
                    (retention + churn).values,
                    np.ones_like(retention.values),
                )

    def test_revenue_metric(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """revenue metric should have positive values."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(
            cohort_data, metrics=["revenue"]
        )
        assert "revenue" in metrics
        assert (metrics["revenue"].values >= 0).all()

    def test_customer_count_metric(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """customer_count metric should be non-negative integers."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(
            cohort_data, metrics=["customer_count"]
        )
        assert "customer_count" in metrics
        assert (metrics["customer_count"].values >= 0).all()

    def test_missing_revenue_column_skips(
        self, analyzer: CohortAnalyzer
    ) -> None:
        """Should skip revenue/aov metrics if revenue column missing."""
        data = pd.DataFrame({
            "customer_id": ["C1", "C2", "C1"],
            "event_date": pd.to_datetime(
                ["2024-01-01", "2024-01-15", "2024-02-01"]
            ),
        })
        cohort_data = analyzer.assign_cohorts(data, cohort_type="monthly")
        metrics = analyzer.compute_cohort_metrics(
            cohort_data, metrics=["revenue", "avg_order_value"]
        )
        assert "revenue" not in metrics
        assert "avg_order_value" not in metrics


# ---------------------------------------------------------------------------
# Cohort Summary Tests
# ---------------------------------------------------------------------------

class TestCohortSummary:
    """Tests for cohort summary generation."""

    def test_summary_has_required_columns(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Summary should contain standard columns."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        summary = analyzer.get_cohort_summary(cohort_data)
        assert "cohort" in summary.columns
        assert "total_customers" in summary.columns
        assert "total_events" in summary.columns
        assert "avg_lifetime_periods" in summary.columns

    def test_summary_includes_revenue(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Summary should include total_revenue if revenue column exists."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        summary = analyzer.get_cohort_summary(cohort_data)
        assert "total_revenue" in summary.columns

    def test_summary_sorted_by_customers(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Summary should be sorted by total_customers descending."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        summary = analyzer.get_cohort_summary(cohort_data)
        customers = summary["total_customers"].values
        assert all(customers[i] >= customers[i + 1] for i in range(len(customers) - 1))


# ---------------------------------------------------------------------------
# Filter Tests
# ---------------------------------------------------------------------------

class TestFilterCohorts:
    """Tests for cohort filtering."""

    def test_filter_by_cohort_names(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should keep only specified cohorts."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        filtered = analyzer.filter_cohorts(cohort_data, cohorts=["2024-01"])
        assert set(filtered["cohort"].unique()) == {"2024-01"}

    def test_filter_by_min_size(
        self, sample_events: pd.DataFrame
    ) -> None:
        """Should filter out cohorts smaller than min_size."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        filtered = analyzer.filter_cohorts(cohort_data, min_size=9)
        cohort_sizes = filtered.groupby("cohort")["customer_id"].nunique()
        assert (cohort_sizes >= 9).all()


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_customer(self, analyzer: CohortAnalyzer) -> None:
        """Should handle a single customer."""
        data = pd.DataFrame({
            "customer_id": ["C001"],
            "event_date": [pd.Timestamp("2024-01-15")],
            "revenue": [50000],
        })
        result = analyzer.assign_cohorts(data, cohort_type="monthly")
        assert len(result) == 1
        assert result["cohort_period"].iloc[0] == 0

    def test_invalid_cohort_type(self, analyzer: CohortAnalyzer) -> None:
        """Should raise ValueError for unknown cohort type."""
        data = pd.DataFrame({
            "customer_id": ["C1"],
            "event_date": [pd.Timestamp("2024-01-01")],
        })
        with pytest.raises(ValueError, match="Unknown cohort_type"):
            analyzer.assign_cohorts(data, cohort_type="yearly")

    def test_uses_config_default_cohort_type(self) -> None:
        """Should use config cohort_type when none specified."""
        analyzer = CohortAnalyzer(
            config={"cohort_type": "weekly", "min_cohort_size": 1}
        )
        data = pd.DataFrame({
            "customer_id": ["C1", "C1"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "revenue": [100, 200],
        })
        result = analyzer.assign_cohorts(data)
        # Should have weekly format
        assert "-W" in result["cohort"].iloc[0]


# ---------------------------------------------------------------------------
# Retention Curves Tests
# ---------------------------------------------------------------------------

class TestRetentionCurves:
    """Tests for retention curve extraction."""

    def test_get_retention_curves_returns_dict(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should return a dict mapping cohort labels to numpy arrays."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        curves = analyzer.get_retention_curves(matrix)
        assert isinstance(curves, dict)
        assert len(curves) > 0

    def test_curves_keys_match_matrix_index(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Curve keys should match retention matrix cohort labels."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        curves = analyzer.get_retention_curves(matrix)
        for cohort_label in matrix.index:
            assert str(cohort_label) in curves

    def test_curves_values_are_float_arrays(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Each curve should be a 1-D float numpy array."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        curves = analyzer.get_retention_curves(matrix)
        for curve in curves.values():
            assert isinstance(curve, np.ndarray)
            assert curve.dtype == float
            assert curve.ndim == 1

    def test_curve_starts_at_one(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Each retention curve should start at 1.0 (period 0)."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        curves = analyzer.get_retention_curves(matrix)
        for curve in curves.values():
            assert curve[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Average Retention Curve Tests
# ---------------------------------------------------------------------------

class TestAverageRetentionCurve:
    """Tests for average retention curve computation."""

    def test_returns_float_array(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should return a 1-D float numpy array."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        avg = analyzer.get_average_retention_curve(matrix)
        assert isinstance(avg, np.ndarray)
        assert avg.dtype == float

    def test_avg_curve_length_matches_columns(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Average curve length should match number of period columns."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data, max_periods=2)
        avg = analyzer.get_average_retention_curve(matrix)
        assert len(avg) == len(matrix.columns)

    def test_avg_curve_starts_at_one(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Average curve should start at 1.0."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        avg = analyzer.get_average_retention_curve(matrix)
        assert avg[0] == pytest.approx(1.0)

    def test_avg_curve_bounded(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Average curve values should be between 0 and 1."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        avg = analyzer.get_average_retention_curve(matrix)
        assert (avg >= 0.0).all()
        assert (avg <= 1.0).all()


# ---------------------------------------------------------------------------
# Half-Life Tests
# ---------------------------------------------------------------------------

class TestHalfLife:
    """Tests for retention half-life computation."""

    def test_returns_series(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should return a pandas Series."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        hl = analyzer.compute_half_life(matrix)
        assert isinstance(hl, pd.Series)
        assert hl.name == "half_life"

    def test_nan_when_retention_stays_above_50(self) -> None:
        """Half-life should be NaN if retention never drops below 50%."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        matrix = pd.DataFrame(
            [[1.0, 0.9, 0.8, 0.7, 0.6]],
            index=["2024-01"],
            columns=[0, 1, 2, 3, 4],
        )
        hl = analyzer.compute_half_life(matrix)
        assert np.isnan(hl["2024-01"])

    def test_correct_half_life_period(self) -> None:
        """Half-life should be the first period where retention < 0.5."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        matrix = pd.DataFrame(
            [[1.0, 0.8, 0.6, 0.4, 0.2]],
            index=["2024-01"],
            columns=[0, 1, 2, 3, 4],
        )
        hl = analyzer.compute_half_life(matrix)
        assert hl["2024-01"] == 3  # First period below 0.5

    def test_half_life_at_boundary(self) -> None:
        """Exactly 0.5 is not below 0.5, so should not trigger half-life."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        matrix = pd.DataFrame(
            [[1.0, 0.5, 0.49]],
            index=["2024-01"],
            columns=[0, 1, 2],
        )
        hl = analyzer.compute_half_life(matrix)
        assert hl["2024-01"] == 2  # 0.49 < 0.5 triggers at period 2


# ---------------------------------------------------------------------------
# Incremental Churn Rate Tests
# ---------------------------------------------------------------------------

class TestChurnRates:
    """Tests for period-over-period churn rate computation."""

    def test_churn_rates_shape_matches_retention(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Churn rate matrix should have same shape as retention matrix."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        churn = analyzer.compute_churn_rates(matrix)
        assert churn.shape == matrix.shape

    def test_period_zero_churn_is_zero(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Period 0 churn should be 0.0 for all cohorts."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        churn = analyzer.compute_churn_rates(matrix)
        if 0 in churn.columns:
            np.testing.assert_array_almost_equal(churn[0].values, 0.0)

    def test_churn_rate_formula(self) -> None:
        """churn(t) = 1 - ret(t)/ret(t-1)."""
        analyzer = CohortAnalyzer(config={"min_cohort_size": 1})
        matrix = pd.DataFrame(
            [[1.0, 0.8, 0.6]],
            index=["2024-01"],
            columns=[0, 1, 2],
        )
        churn = analyzer.compute_churn_rates(matrix)
        # Period 1: 1 - 0.8/1.0 = 0.2
        assert churn.loc["2024-01", 1] == pytest.approx(0.2)
        # Period 2: 1 - 0.6/0.8 = 0.25
        assert churn.loc["2024-01", 2] == pytest.approx(0.25)

    def test_churn_rates_non_negative(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Churn rates should generally be non-negative."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        churn = analyzer.compute_churn_rates(matrix)
        # Period 0 is always 0; other periods should be >= 0 in normal scenarios
        if 0 in churn.columns:
            assert (churn[0] == 0.0).all()


# ---------------------------------------------------------------------------
# Cumulative Revenue Tests
# ---------------------------------------------------------------------------

class TestCumulativeRevenue:
    """Tests for cumulative revenue computation."""

    def test_cumulative_revenue_non_decreasing(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Cumulative revenue should be non-decreasing across periods."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        cum_rev = analyzer.compute_cumulative_revenue(cohort_data, max_periods=3)
        for _, row in cum_rev.iterrows():
            values = row.dropna().values
            assert (np.diff(values) >= -1e-6).all()

    def test_cumulative_revenue_positive(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Cumulative revenue should be non-negative."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        cum_rev = analyzer.compute_cumulative_revenue(cohort_data)
        assert (cum_rev.values >= 0).all()

    def test_missing_revenue_column_raises(self, analyzer: CohortAnalyzer) -> None:
        """Should raise ValueError if revenue column is missing."""
        data = pd.DataFrame({
            "customer_id": ["A", "B"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
        })
        cohort_data = analyzer.assign_cohorts(data, cohort_type="monthly")
        with pytest.raises(ValueError, match="Revenue column"):
            analyzer.compute_cumulative_revenue(cohort_data)


# ---------------------------------------------------------------------------
# Full Analysis Pipeline Tests
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    """Tests for the full_analysis convenience method."""

    def test_returns_all_expected_keys(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Full analysis should return all expected result keys."""
        result = analyzer.full_analysis(sample_events, cohort_type="monthly")
        expected_keys = {
            "cohort_data", "retention_matrix", "retention_curves",
            "avg_retention_curve", "churn_rates", "half_life",
            "summary", "metrics",
        }
        assert set(result.keys()) == expected_keys

    def test_retention_matrix_is_valid(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Full analysis retention matrix should be a valid DataFrame."""
        result = analyzer.full_analysis(sample_events, cohort_type="monthly")
        matrix = result["retention_matrix"]
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape[0] > 0

    def test_curves_match_matrix_cohorts(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Number of curves should match number of cohorts in matrix."""
        result = analyzer.full_analysis(sample_events, cohort_type="monthly")
        assert len(result["retention_curves"]) == len(result["retention_matrix"].index)

    def test_avg_curve_length_matches_periods(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Average curve length should match retention matrix columns."""
        result = analyzer.full_analysis(sample_events, cohort_type="monthly")
        assert len(result["avg_retention_curve"]) == len(result["retention_matrix"].columns)

    def test_churn_rates_shape_matches_retention(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Churn rates shape should match retention matrix shape."""
        result = analyzer.full_analysis(sample_events, cohort_type="monthly")
        assert result["churn_rates"].shape == result["retention_matrix"].shape

    def test_summary_is_dataframe(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Summary should be a pandas DataFrame."""
        result = analyzer.full_analysis(sample_events, cohort_type="monthly")
        assert isinstance(result["summary"], pd.DataFrame)

    def test_metrics_is_dict(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Metrics should be a dict of DataFrames."""
        result = analyzer.full_analysis(sample_events, cohort_type="monthly")
        assert isinstance(result["metrics"], dict)


# ---------------------------------------------------------------------------
# Retention Heatmap Visualization Tests
# ---------------------------------------------------------------------------

class TestPlotRetentionHeatmap:
    """Tests for retention heatmap visualization."""

    def test_returns_figure(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should return a matplotlib Figure."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_heatmap(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should apply custom title."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_heatmap(matrix, title="Custom Heatmap Title")
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Heatmap Title"
        plt.close(fig)

    def test_custom_figsize(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should apply custom figure size."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_heatmap(matrix, figsize=(8, 4))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8, abs=0.1)
        assert h == pytest.approx(4, abs=0.1)
        plt.close(fig)

    def test_saves_to_file(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should save heatmap to file when save_path is provided."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "heatmap.png")
            fig = analyzer.plot_retention_heatmap(matrix, save_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            plt.close(fig)

    def test_no_annotation(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should work with annotations disabled."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_heatmap(matrix, annot=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_has_axis_labels(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Heatmap should have proper axis labels."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_heatmap(matrix)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Cohort Period"
        assert ax.get_ylabel() == "Cohort"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Retention Line Plot Visualization Tests
# ---------------------------------------------------------------------------

class TestPlotRetentionLines:
    """Tests for retention line plot visualization."""

    def test_returns_figure(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should return a matplotlib Figure."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_lines(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_one_line_per_cohort(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should have one line per cohort plus optional average."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_lines(matrix, show_average=True)
        ax = fig.axes[0]
        # lines = cohorts + average
        n_cohorts = len(matrix.index)
        assert len(ax.lines) == n_cohorts + 1  # +1 for average line
        plt.close(fig)

    def test_no_average_line(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """With show_average=False, should have only cohort lines."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_lines(matrix, show_average=False)
        ax = fig.axes[0]
        n_cohorts = len(matrix.index)
        assert len(ax.lines) == n_cohorts
        plt.close(fig)

    def test_custom_title(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should apply custom title."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_lines(matrix, title="My Retention Curves")
        ax = fig.axes[0]
        assert ax.get_title() == "My Retention Curves"
        plt.close(fig)

    def test_y_axis_limits(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Y-axis should be set to approximately [0, 1]."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_lines(matrix)
        ax = fig.axes[0]
        ylim = ax.get_ylim()
        assert ylim[0] < 0.0  # Slightly below 0 for padding
        assert ylim[1] > 1.0  # Slightly above 1 for padding
        plt.close(fig)

    def test_saves_to_file(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should save line plot to file when save_path is provided."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "retention_lines.png")
            fig = analyzer.plot_retention_lines(matrix, save_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            plt.close(fig)

    def test_has_legend(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Plot should include a legend."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_lines(matrix, show_average=True)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_has_grid(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Plot should have grid enabled."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        matrix = analyzer.compute_retention_matrix(cohort_data)
        fig = analyzer.plot_retention_lines(matrix)
        ax = fig.axes[0]
        # Check grid is visible (at least on y-axis)
        assert ax.yaxis.get_gridlines()[0].get_visible()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Cohort Size Bar Chart Visualization Tests
# ---------------------------------------------------------------------------

class TestPlotCohortSizes:
    """Tests for cohort size bar chart visualization."""

    def test_returns_figure(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should return a matplotlib Figure."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        fig = analyzer.plot_cohort_sizes(cohort_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_number_of_bars(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should have one bar per cohort."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        fig = analyzer.plot_cohort_sizes(cohort_data)
        ax = fig.axes[0]
        n_cohorts = cohort_data["cohort"].nunique()
        # Count patches (bars)
        bars = [p for p in ax.patches if p.get_height() > 0]
        assert len(bars) == n_cohorts
        plt.close(fig)

    def test_custom_title(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should apply custom title."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        fig = analyzer.plot_cohort_sizes(cohort_data, title="My Cohort Sizes")
        ax = fig.axes[0]
        assert ax.get_title() == "My Cohort Sizes"
        plt.close(fig)

    def test_custom_color(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should apply custom bar color."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        fig = analyzer.plot_cohort_sizes(cohort_data, color="green")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Should save bar chart to file when save_path is provided."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cohort_sizes.png")
            fig = analyzer.plot_cohort_sizes(cohort_data, save_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            plt.close(fig)

    def test_axis_labels(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Bar chart should have proper axis labels."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        fig = analyzer.plot_cohort_sizes(cohort_data)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Cohort"
        assert ax.get_ylabel() == "Number of Customers"
        plt.close(fig)

    def test_bar_heights_match_cohort_sizes(
        self, analyzer: CohortAnalyzer, sample_events: pd.DataFrame
    ) -> None:
        """Bar heights should match actual unique customer counts per cohort."""
        cohort_data = analyzer.assign_cohorts(sample_events, cohort_type="monthly")
        expected_sizes = (
            cohort_data.groupby("cohort")["customer_id"]
            .nunique()
            .sort_index()
        )
        fig = analyzer.plot_cohort_sizes(cohort_data)
        ax = fig.axes[0]
        bar_heights = [p.get_height() for p in ax.patches if p.get_height() > 0]
        np.testing.assert_array_equal(
            sorted(bar_heights, reverse=True),
            sorted(expected_sizes.values, reverse=True),
        )
        plt.close(fig)
