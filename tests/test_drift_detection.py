"""
TDD tests for PSI (Population Stability Index) drift detection module.

Tests cover:
- PSI calculation between reference and production distributions
- Multiple binning strategies (quantile, equal-width, custom)
- Threshold-based alerting (green/yellow/red)
- Feature-level drift reports
- Edge cases (empty data, identical distributions, single feature)
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detection import (
    DriftDetector,
    DriftAlert,
    DriftReport,
    calculate_psi,
    compute_bins,
)


class TestCalculatePSI:
    """Test the core PSI calculation function."""

    def test_identical_distributions_psi_zero(self):
        """PSI should be ~0 for identical distributions."""
        reference = np.random.RandomState(42).normal(0, 1, 10000)
        production = reference.copy()
        psi = calculate_psi(reference, production, n_bins=10)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_slightly_shifted_distribution_low_psi(self):
        """Slightly shifted distribution should have low PSI (< 0.1)."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 10000)
        production = rng.normal(0.1, 1, 10000)
        psi = calculate_psi(reference, production, n_bins=10)
        assert 0.0 < psi < 0.1

    def test_large_shift_high_psi(self):
        """Large distribution shift should produce high PSI (> 0.25)."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 10000)
        production = rng.normal(3, 2, 10000)
        psi = calculate_psi(reference, production, n_bins=10)
        assert psi > 0.25

    def test_psi_is_non_negative(self):
        """PSI should always be non-negative."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 5000)
        production = rng.normal(1, 1.5, 5000)
        psi = calculate_psi(reference, production, n_bins=10)
        assert psi >= 0.0

    def test_psi_symmetric_approximate(self):
        """PSI(A, B) should approximately equal PSI(B, A) for similar sizes."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 10000)
        b = rng.normal(0.5, 1.2, 10000)
        psi_ab = calculate_psi(a, b, n_bins=10)
        psi_ba = calculate_psi(b, a, n_bins=10)
        # PSI is not perfectly symmetric but should be close
        assert abs(psi_ab - psi_ba) < 0.05

    def test_psi_with_custom_bins(self):
        """PSI should work with explicitly provided bin edges."""
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 5000)
        production = rng.normal(0, 1, 5000)
        bin_edges = np.array([-3, -2, -1, 0, 1, 2, 3])
        psi = calculate_psi(reference, production, bin_edges=bin_edges)
        assert psi >= 0.0
        assert psi < 0.1  # Same distribution, should be low

    def test_psi_handles_zeros_with_epsilon(self):
        """PSI should handle empty bins gracefully using epsilon smoothing."""
        reference = np.array([1.0] * 100 + [5.0] * 100)
        production = np.array([1.0] * 200)  # All in one bin
        psi = calculate_psi(reference, production, n_bins=5)
        assert np.isfinite(psi)
        assert psi > 0.0


class TestComputeBins:
    """Test binning strategies."""

    def test_quantile_binning(self):
        """Quantile binning should produce n_bins+1 edges."""
        data = np.random.RandomState(42).normal(0, 1, 1000)
        edges = compute_bins(data, strategy="quantile", n_bins=10)
        assert len(edges) == 11  # n_bins + 1
        assert edges[0] <= data.min()
        assert edges[-1] >= data.max()

    def test_equal_width_binning(self):
        """Equal-width binning should produce evenly spaced edges."""
        data = np.random.RandomState(42).uniform(0, 100, 1000)
        edges = compute_bins(data, strategy="equal_width", n_bins=5)
        assert len(edges) == 6
        widths = np.diff(edges)
        assert np.allclose(widths, widths[0], atol=1e-6)

    def test_invalid_strategy_raises(self):
        """Invalid binning strategy should raise ValueError."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="strategy"):
            compute_bins(data, strategy="invalid_strategy", n_bins=5)


class TestDriftAlert:
    """Test drift alert thresholds."""

    def test_green_alert(self):
        """PSI < 0.1 should be GREEN (no significant drift)."""
        alert = DriftAlert(psi_value=0.05)
        assert alert.level == "green"
        assert alert.is_drifted is False

    def test_yellow_alert(self):
        """0.1 <= PSI < 0.25 should be YELLOW (moderate drift)."""
        alert = DriftAlert(psi_value=0.15)
        assert alert.level == "yellow"
        assert alert.is_drifted is False

    def test_red_alert(self):
        """PSI >= 0.25 should be RED (significant drift)."""
        alert = DriftAlert(psi_value=0.30)
        assert alert.level == "red"
        assert alert.is_drifted is True

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        alert = DriftAlert(psi_value=0.08, yellow_threshold=0.05, red_threshold=0.15)
        assert alert.level == "yellow"

    def test_boundary_yellow(self):
        """PSI exactly at yellow threshold should be yellow."""
        alert = DriftAlert(psi_value=0.1)
        assert alert.level == "yellow"

    def test_boundary_red(self):
        """PSI exactly at red threshold should be red."""
        alert = DriftAlert(psi_value=0.25)
        assert alert.level == "red"


class TestDriftDetector:
    """Test the main DriftDetector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample reference and production DataFrames."""
        rng = np.random.RandomState(42)
        n_ref = 5000
        n_prod = 3000
        reference = pd.DataFrame({
            "feature_a": rng.normal(0, 1, n_ref),
            "feature_b": rng.exponential(2, n_ref),
            "feature_c": rng.uniform(0, 10, n_ref),
        })
        production = pd.DataFrame({
            "feature_a": rng.normal(0, 1, n_prod),       # No drift
            "feature_b": rng.exponential(4, n_prod),       # Drift
            "feature_c": rng.uniform(0, 10, n_prod),       # No drift
        })
        return reference, production

    def test_detector_initialization(self):
        """DriftDetector should initialize with config parameters."""
        detector = DriftDetector(
            n_bins=10,
            binning_strategy="quantile",
            yellow_threshold=0.1,
            red_threshold=0.25,
        )
        assert detector.n_bins == 10
        assert detector.binning_strategy == "quantile"

    def test_detector_from_config(self):
        """DriftDetector should be creatable from a config dict."""
        config = {
            "n_bins": 15,
            "binning_strategy": "equal_width",
            "yellow_threshold": 0.08,
            "red_threshold": 0.20,
        }
        detector = DriftDetector.from_config(config)
        assert detector.n_bins == 15
        assert detector.binning_strategy == "equal_width"

    def test_fit_stores_reference(self, sample_data):
        """fit() should store reference distribution statistics."""
        reference, _ = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        assert detector.reference_data is not None
        assert set(detector.feature_names) == {"feature_a", "feature_b", "feature_c"}

    def test_detect_returns_report(self, sample_data):
        """detect() should return a DriftReport object."""
        reference, production = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        report = detector.detect(production)
        assert isinstance(report, DriftReport)

    def test_report_contains_all_features(self, sample_data):
        """DriftReport should contain PSI values for all features."""
        reference, production = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        report = detector.detect(production)
        assert set(report.feature_psi.keys()) == {"feature_a", "feature_b", "feature_c"}

    def test_drifted_feature_detected(self, sample_data):
        """feature_b with shifted exponential should be flagged as drifted."""
        reference, production = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        report = detector.detect(production)
        # feature_b has large shift (exp(2) -> exp(4))
        assert report.feature_alerts["feature_b"].level in ("yellow", "red")
        assert report.feature_psi["feature_b"] > report.feature_psi["feature_a"]

    def test_stable_feature_not_flagged(self, sample_data):
        """feature_a and feature_c with same distribution should be green."""
        reference, production = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        report = detector.detect(production)
        assert report.feature_alerts["feature_a"].level == "green"
        assert report.feature_alerts["feature_c"].level == "green"

    def test_report_summary(self, sample_data):
        """Report should provide summary with drifted feature count."""
        reference, production = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        report = detector.detect(production)
        summary = report.summary()
        assert "total_features" in summary
        assert "drifted_features" in summary
        assert "max_psi_feature" in summary
        assert summary["total_features"] == 3

    def test_detect_subset_features(self, sample_data):
        """detect() should work on a subset of features."""
        reference, production = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        report = detector.detect(production, features=["feature_a"])
        assert set(report.feature_psi.keys()) == {"feature_a"}

    def test_detect_without_fit_raises(self, sample_data):
        """detect() without prior fit() should raise RuntimeError."""
        _, production = sample_data
        detector = DriftDetector(n_bins=10)
        with pytest.raises(RuntimeError, match="fit"):
            detector.detect(production)

    def test_to_dict_serializable(self, sample_data):
        """DriftReport.to_dict() should return JSON-serializable dict."""
        reference, production = sample_data
        detector = DriftDetector(n_bins=10)
        detector.fit(reference)
        report = detector.detect(production)
        result = report.to_dict()
        assert isinstance(result, dict)
        assert "feature_psi" in result
        assert "alerts" in result
        import json
        json.dumps(result)  # Should not raise
