"""
TDD tests for KS (Kolmogorov-Smirnov) and Chi-Square drift detection module.

Tests cover:
- KS statistic computation for numerical features
- Chi-square test for categorical features
- KSDriftDetector class (fit/detect API)
- Threshold-based alerting (no_drift / warning / drift)
- Edge cases (identical distributions, single-value, empty categories)
- Integration with config
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.ks_drift import (
    compute_ks_statistic,
    compute_chi_square,
    KSDriftAlert,
    KSDriftReport,
    KSDriftDetector,
)


# =========================================================================
# KS statistic for numerical features
# =========================================================================

class TestComputeKSStatistic:
    """Test the KS statistic computation for numerical features."""

    def test_identical_distributions_ks_zero(self):
        """KS statistic should be ~0 for identical distributions."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 10000)
        production = reference.copy()
        stat, p_value = compute_ks_statistic(reference, production)
        assert stat == pytest.approx(0.0, abs=1e-6)
        assert p_value > 0.99

    def test_same_distribution_high_p_value(self):
        """Samples from same distribution should have high p-value."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 5000)
        production = rng.normal(0, 1, 5000)
        stat, p_value = compute_ks_statistic(reference, production)
        assert p_value > 0.05  # Not significant at 5%

    def test_different_distributions_low_p_value(self):
        """Samples from different distributions should have low p-value."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 5000)
        production = rng.normal(3, 2, 5000)
        stat, p_value = compute_ks_statistic(reference, production)
        assert stat > 0.1
        assert p_value < 0.05  # Significant drift

    def test_ks_statistic_bounded_zero_one(self):
        """KS statistic should be between 0 and 1."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 1000)
        production = rng.normal(5, 3, 1000)
        stat, p_value = compute_ks_statistic(reference, production)
        assert 0.0 <= stat <= 1.0
        assert 0.0 <= p_value <= 1.0

    def test_ks_returns_tuple(self):
        """compute_ks_statistic should return (statistic, p_value) tuple."""
        rng = np.random.RandomState(42)
        result = compute_ks_statistic(
            rng.normal(0, 1, 100),
            rng.normal(0, 1, 100),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ks_small_shift(self):
        """Small shift should produce small KS statistic."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 10000)
        production = rng.normal(0.1, 1, 10000)
        stat, p_value = compute_ks_statistic(reference, production)
        assert stat < 0.1

    def test_ks_large_shift(self):
        """Large shift should produce large KS statistic."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 5000)
        production = rng.normal(5, 0.5, 5000)
        stat, _ = compute_ks_statistic(reference, production)
        assert stat > 0.5

    def test_ks_with_numpy_arrays(self):
        """Should work with numpy arrays."""
        stat, p_value = compute_ks_statistic(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        )
        assert isinstance(stat, float)
        assert isinstance(p_value, float)

    def test_ks_with_lists(self):
        """Should work with Python lists."""
        stat, p_value = compute_ks_statistic(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5, 4.5, 5.5],
        )
        assert isinstance(stat, float)
        assert isinstance(p_value, float)


# =========================================================================
# Chi-square test for categorical features
# =========================================================================

class TestComputeChiSquare:
    """Test chi-square test for categorical features."""

    def test_identical_distributions_high_p_value(self):
        """Identical category distributions should have high p-value."""
        rng = np.random.RandomState(42)
        categories = ["A", "B", "C", "D"]
        reference = rng.choice(categories, size=5000, p=[0.4, 0.3, 0.2, 0.1])
        production = rng.choice(categories, size=5000, p=[0.4, 0.3, 0.2, 0.1])
        stat, p_value = compute_chi_square(reference, production)
        assert p_value > 0.05

    def test_different_distributions_low_p_value(self):
        """Very different category distributions should have low p-value."""
        rng = np.random.RandomState(42)
        categories = ["A", "B", "C", "D"]
        reference = rng.choice(categories, size=5000, p=[0.7, 0.1, 0.1, 0.1])
        production = rng.choice(categories, size=5000, p=[0.1, 0.1, 0.1, 0.7])
        stat, p_value = compute_chi_square(reference, production)
        assert p_value < 0.05
        assert stat > 0.0

    def test_chi_square_returns_tuple(self):
        """compute_chi_square should return (statistic, p_value) tuple."""
        result = compute_chi_square(
            np.array(["A", "B", "A", "C"]),
            np.array(["A", "B", "B", "C"]),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_chi_square_statistic_non_negative(self):
        """Chi-square statistic should always be non-negative."""
        rng = np.random.RandomState(42)
        ref = rng.choice(["X", "Y", "Z"], 1000)
        prod = rng.choice(["X", "Y", "Z"], 1000)
        stat, p_value = compute_chi_square(ref, prod)
        assert stat >= 0.0

    def test_chi_square_handles_new_categories(self):
        """Should handle categories present in production but not in reference."""
        reference = np.array(["A", "B", "A", "B", "A"] * 100)
        production = np.array(["A", "B", "C", "A", "B"] * 100)
        stat, p_value = compute_chi_square(reference, production)
        assert isinstance(stat, float)
        assert isinstance(p_value, float)

    def test_chi_square_handles_missing_categories(self):
        """Should handle categories in reference missing from production."""
        reference = np.array(["A", "B", "C"] * 100)
        production = np.array(["A", "B", "A", "B"] * 100)
        stat, p_value = compute_chi_square(reference, production)
        assert isinstance(stat, float)
        assert stat > 0.0

    def test_chi_square_with_pandas_series(self):
        """Should work with pandas Series."""
        ref = pd.Series(["cat", "dog", "cat", "bird"] * 50)
        prod = pd.Series(["cat", "dog", "bird", "bird"] * 50)
        stat, p_value = compute_chi_square(ref, prod)
        assert isinstance(stat, float)


# =========================================================================
# KSDriftAlert
# =========================================================================

class TestKSDriftAlert:
    """Test KS drift alert classification."""

    def test_no_drift_alert(self):
        """High p-value should indicate no drift."""
        alert = KSDriftAlert(
            statistic=0.01,
            p_value=0.95,
            feature_name="feat_a",
            test_type="ks",
        )
        assert alert.level == "no_drift"
        assert alert.is_drifted is False

    def test_warning_alert(self):
        """Moderate p-value should indicate warning."""
        alert = KSDriftAlert(
            statistic=0.05,
            p_value=0.03,
            feature_name="feat_b",
            test_type="ks",
            warning_threshold=0.05,
            drift_threshold=0.01,
        )
        assert alert.level == "warning"
        assert alert.is_drifted is False

    def test_drift_alert(self):
        """Very low p-value should indicate drift."""
        alert = KSDriftAlert(
            statistic=0.3,
            p_value=0.001,
            feature_name="feat_c",
            test_type="ks",
            warning_threshold=0.05,
            drift_threshold=0.01,
        )
        assert alert.level == "drift"
        assert alert.is_drifted is True

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        alert = KSDriftAlert(
            statistic=0.15,
            p_value=0.02,
            feature_name="feat_x",
            test_type="chi_square",
        )
        d = alert.to_dict()
        assert "statistic" in d
        assert "p_value" in d
        assert "feature_name" in d
        assert "test_type" in d
        assert "level" in d
        assert "is_drifted" in d

    def test_default_thresholds(self):
        """Default thresholds: warning=0.05, drift=0.01."""
        alert = KSDriftAlert(
            statistic=0.1,
            p_value=0.03,
            feature_name="f1",
            test_type="ks",
        )
        assert alert.warning_threshold == 0.05
        assert alert.drift_threshold == 0.01

    def test_boundary_warning(self):
        """p_value exactly at warning threshold should be warning."""
        alert = KSDriftAlert(
            statistic=0.1,
            p_value=0.05,
            feature_name="f1",
            test_type="ks",
            warning_threshold=0.05,
            drift_threshold=0.01,
        )
        # p_value <= warning_threshold → warning
        assert alert.level == "warning"

    def test_boundary_drift(self):
        """p_value exactly at drift threshold should be drift."""
        alert = KSDriftAlert(
            statistic=0.1,
            p_value=0.01,
            feature_name="f1",
            test_type="ks",
            warning_threshold=0.05,
            drift_threshold=0.01,
        )
        assert alert.level == "drift"


# =========================================================================
# KSDriftDetector class
# =========================================================================

class TestKSDriftDetector:
    """Test the KSDriftDetector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample reference and production DataFrames."""
        rng = np.random.RandomState(42)
        n_ref = 5000
        n_prod = 3000
        reference = pd.DataFrame({
            "num_a": rng.normal(0, 1, n_ref),
            "num_b": rng.exponential(2, n_ref),
            "cat_a": rng.choice(["X", "Y", "Z"], n_ref, p=[0.5, 0.3, 0.2]),
            "cat_b": rng.choice(["low", "mid", "high"], n_ref),
        })
        production_no_drift = pd.DataFrame({
            "num_a": rng.normal(0, 1, n_prod),
            "num_b": rng.exponential(2, n_prod),
            "cat_a": rng.choice(["X", "Y", "Z"], n_prod, p=[0.5, 0.3, 0.2]),
            "cat_b": rng.choice(["low", "mid", "high"], n_prod),
        })
        production_drift = pd.DataFrame({
            "num_a": rng.normal(3, 2, n_prod),         # Drifted
            "num_b": rng.exponential(10, n_prod),       # Drifted
            "cat_a": rng.choice(["X", "Y", "Z"], n_prod, p=[0.1, 0.1, 0.8]),  # Drifted
            "cat_b": rng.choice(["low", "mid", "high"], n_prod),  # No drift
        })
        return reference, production_no_drift, production_drift

    def test_initialization(self):
        """KSDriftDetector should initialize with parameters."""
        detector = KSDriftDetector(
            numerical_features=["a", "b"],
            categorical_features=["c"],
        )
        assert detector.numerical_features == ["a", "b"]
        assert detector.categorical_features == ["c"]

    def test_from_config(self):
        """Should create from config dict."""
        config = {
            "numerical_features": ["f1", "f2"],
            "categorical_features": ["c1"],
            "warning_threshold": 0.10,
            "drift_threshold": 0.05,
        }
        detector = KSDriftDetector.from_config(config)
        assert detector.numerical_features == ["f1", "f2"]
        assert detector.warning_threshold == 0.10

    def test_fit_stores_reference(self, sample_data):
        """fit() should store reference data."""
        reference, _, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a", "num_b"],
            categorical_features=["cat_a", "cat_b"],
        )
        detector.fit(reference)
        assert detector.reference_data is not None

    def test_detect_returns_report(self, sample_data):
        """detect() should return a KSDriftReport."""
        reference, production, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a", "num_b"],
            categorical_features=["cat_a", "cat_b"],
        )
        detector.fit(reference)
        report = detector.detect(production)
        assert isinstance(report, KSDriftReport)

    def test_report_contains_all_features(self, sample_data):
        """Report should contain results for all features."""
        reference, production, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a", "num_b"],
            categorical_features=["cat_a", "cat_b"],
        )
        detector.fit(reference)
        report = detector.detect(production)
        assert "num_a" in report.feature_alerts
        assert "num_b" in report.feature_alerts
        assert "cat_a" in report.feature_alerts
        assert "cat_b" in report.feature_alerts

    def test_numerical_uses_ks_test(self, sample_data):
        """Numerical features should use KS test."""
        reference, production, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a"],
            categorical_features=[],
        )
        detector.fit(reference)
        report = detector.detect(production)
        assert report.feature_alerts["num_a"].test_type == "ks"

    def test_categorical_uses_chi_square(self, sample_data):
        """Categorical features should use chi-square test."""
        reference, production, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=[],
            categorical_features=["cat_a"],
        )
        detector.fit(reference)
        report = detector.detect(production)
        assert report.feature_alerts["cat_a"].test_type == "chi_square"

    def test_detects_numerical_drift(self, sample_data):
        """Should detect drift in shifted numerical features."""
        reference, _, production_drift = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a", "num_b"],
            categorical_features=[],
            drift_threshold=0.01,
        )
        detector.fit(reference)
        report = detector.detect(production_drift)
        assert report.feature_alerts["num_a"].is_drifted is True
        assert report.feature_alerts["num_b"].is_drifted is True

    def test_detects_categorical_drift(self, sample_data):
        """Should detect drift in shifted categorical features."""
        reference, _, production_drift = sample_data
        detector = KSDriftDetector(
            numerical_features=[],
            categorical_features=["cat_a"],
            drift_threshold=0.01,
        )
        detector.fit(reference)
        report = detector.detect(production_drift)
        assert report.feature_alerts["cat_a"].is_drifted is True

    def test_no_false_positive_on_stable(self, sample_data):
        """Should not flag drift for stable features."""
        reference, production_stable, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a"],
            categorical_features=["cat_b"],
            drift_threshold=0.01,
        )
        detector.fit(reference)
        report = detector.detect(production_stable)
        # Stable features should not be flagged as drifted
        assert report.feature_alerts["num_a"].is_drifted is False

    def test_detect_without_fit_raises(self, sample_data):
        """detect() without fit() should raise RuntimeError."""
        _, production, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a"],
            categorical_features=[],
        )
        with pytest.raises(RuntimeError, match="fit"):
            detector.detect(production)

    def test_report_summary(self, sample_data):
        """Report summary should contain expected keys."""
        reference, _, production_drift = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a", "num_b"],
            categorical_features=["cat_a", "cat_b"],
        )
        detector.fit(reference)
        report = detector.detect(production_drift)
        summary = report.summary()
        assert "total_features" in summary
        assert "drifted_features" in summary
        assert "numerical_drifted" in summary
        assert "categorical_drifted" in summary
        assert summary["total_features"] == 4

    def test_report_to_dict(self, sample_data):
        """to_dict() should return JSON-serializable dict."""
        reference, production, _ = sample_data
        detector = KSDriftDetector(
            numerical_features=["num_a"],
            categorical_features=["cat_a"],
        )
        detector.fit(reference)
        report = detector.detect(production)
        result = report.to_dict()
        assert isinstance(result, dict)
        import json
        json.dumps(result)  # Should not raise

    def test_auto_detect_feature_types(self, sample_data):
        """Should auto-detect feature types if not specified."""
        reference, production, _ = sample_data
        detector = KSDriftDetector.auto_detect(reference)
        assert "num_a" in detector.numerical_features
        assert "num_b" in detector.numerical_features
        assert "cat_a" in detector.categorical_features
        assert "cat_b" in detector.categorical_features
