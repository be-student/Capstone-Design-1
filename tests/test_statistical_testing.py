"""
TDD Tests for Statistical Testing Framework.

Tests cover:
- T-test (Welch's and Student's)
- Chi-square test of independence
- Mann-Whitney U test (non-parametric)
- Z-test for proportions
- Multiple comparison corrections (Bonferroni, FDR, Holm-Bonferroni)
- Multi-metric testing pipeline
- Edge cases and integration with ABTestFramework
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ab_testing import (
    ABTestFramework,
    MultipleComparisonCorrection,
    PowerAnalysis,
    StatisticalTestSuite,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def suite():
    """Create a StatisticalTestSuite with default alpha."""
    return StatisticalTestSuite(alpha=0.05)


@pytest.fixture
def continuous_groups():
    """Two continuous samples with a known difference."""
    np.random.seed(42)
    a = np.random.normal(loc=105, scale=10, size=500)
    b = np.random.normal(loc=100, scale=10, size=500)
    return a, b


@pytest.fixture
def no_diff_continuous():
    """Two continuous samples with no real difference."""
    np.random.seed(99)
    a = np.random.normal(loc=100, scale=10, size=500)
    b = np.random.normal(loc=100, scale=10, size=500)
    return a, b


@pytest.fixture
def binary_groups():
    """Two binary samples with different proportions."""
    np.random.seed(42)
    a = np.random.binomial(1, 0.30, size=1000)
    b = np.random.binomial(1, 0.20, size=1000)
    return a, b


@pytest.fixture
def experiment_data():
    """Multi-metric experiment DataFrame."""
    np.random.seed(42)
    n = 2000
    group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
    return pd.DataFrame({
        "group": group,
        "revenue": np.where(
            group == "treatment",
            np.random.normal(110, 15, n),
            np.random.normal(100, 15, n),
        ),
        "churned": np.where(
            group == "treatment",
            np.random.binomial(1, 0.15, n),
            np.random.binomial(1, 0.25, n),
        ),
        "converted": np.where(
            group == "treatment",
            np.random.binomial(1, 0.12, n),
            np.random.binomial(1, 0.10, n),
        ),
        "session_time": np.where(
            group == "treatment",
            np.random.exponential(20, n),
            np.random.exponential(18, n),
        ),
    })


# ---------------------------------------------------------------------------
# StatisticalTestSuite instantiation
# ---------------------------------------------------------------------------


class TestSuiteInstantiation:
    def test_default_alpha(self):
        s = StatisticalTestSuite()
        assert s.alpha == 0.05

    def test_custom_alpha(self):
        s = StatisticalTestSuite(alpha=0.01)
        assert s.alpha == 0.01

    def test_has_correction(self, suite):
        assert hasattr(suite, "correction")
        assert isinstance(suite.correction, MultipleComparisonCorrection)


# ---------------------------------------------------------------------------
# T-test
# ---------------------------------------------------------------------------


class TestTTest:
    def test_returns_required_keys(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.t_test(a, b)
        for key in ["test_statistic", "p_value", "is_significant",
                     "test_used", "effect_size", "mean_a", "mean_b"]:
            assert key in result, f"Missing key: {key}"

    def test_detects_significant_difference(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.t_test(a, b)
        assert result["is_significant"] is True
        assert result["p_value"] < 0.05

    def test_no_difference_not_significant(self, suite, no_diff_continuous):
        a, b = no_diff_continuous
        result = suite.t_test(a, b)
        assert result["p_value"] > 0.01

    def test_welch_default(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.t_test(a, b)
        assert result["test_used"] == "welch_t_test"

    def test_student_option(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.t_test(a, b, equal_var=True)
        assert result["test_used"] == "student_t_test"

    def test_effect_size_positive_when_a_greater(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.t_test(a, b)
        assert result["effect_size"] > 0

    def test_custom_alpha(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.t_test(a, b, alpha=0.001)
        assert result["alpha"] == 0.001

    def test_accepts_lists(self, suite):
        result = suite.t_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert "p_value" in result

    def test_accepts_pandas_series(self, suite):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([6.0, 7.0, 8.0, 9.0, 10.0])
        result = suite.t_test(a, b)
        assert "p_value" in result


# ---------------------------------------------------------------------------
# Chi-square test
# ---------------------------------------------------------------------------


class TestChiSquareTest:
    def test_returns_required_keys(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.chi_square_test(a, b)
        for key in ["test_statistic", "p_value", "is_significant",
                     "test_used", "dof", "effect_size",
                     "proportion_a", "proportion_b"]:
            assert key in result, f"Missing key: {key}"

    def test_detects_proportion_difference(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.chi_square_test(a, b)
        assert result["is_significant"] is True
        assert result["test_used"] == "chi_square"

    def test_dof_is_one_for_2x2(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.chi_square_test(a, b)
        assert result["dof"] == 1

    def test_contingency_table_shape(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.chi_square_test(a, b)
        ct = result["contingency_table"]
        assert len(ct) == 2
        assert len(ct[0]) == 2

    def test_no_difference_high_p(self, suite):
        np.random.seed(77)
        a = np.random.binomial(1, 0.20, size=500)
        b = np.random.binomial(1, 0.20, size=500)
        result = suite.chi_square_test(a, b)
        assert result["p_value"] > 0.01

    def test_effect_size_cramers_v(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.chi_square_test(a, b)
        assert 0 <= result["effect_size"] <= 1


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------


class TestMannWhitneyUTest:
    def test_returns_required_keys(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.mann_whitney_u_test(a, b)
        for key in ["test_statistic", "p_value", "is_significant",
                     "test_used", "effect_size", "median_a", "median_b"]:
            assert key in result, f"Missing key: {key}"

    def test_detects_shift(self, suite, continuous_groups):
        a, b = continuous_groups
        result = suite.mann_whitney_u_test(a, b)
        assert result["is_significant"] is True
        assert result["test_used"] == "mann_whitney_u"

    def test_no_difference_high_p(self, suite, no_diff_continuous):
        a, b = no_diff_continuous
        result = suite.mann_whitney_u_test(a, b)
        assert result["p_value"] > 0.01

    def test_alternative_less(self, suite):
        np.random.seed(42)
        a = np.random.normal(90, 5, 200)
        b = np.random.normal(100, 5, 200)
        result = suite.mann_whitney_u_test(a, b, alternative="less")
        assert result["alternative"] == "less"
        assert result["p_value"] < 0.05

    def test_alternative_greater(self, suite):
        np.random.seed(42)
        a = np.random.normal(110, 5, 200)
        b = np.random.normal(100, 5, 200)
        result = suite.mann_whitney_u_test(a, b, alternative="greater")
        assert result["alternative"] == "greater"
        assert result["p_value"] < 0.05

    def test_handles_non_normal_data(self, suite):
        np.random.seed(42)
        a = np.random.exponential(5, 300) + 2
        b = np.random.exponential(3, 300)
        result = suite.mann_whitney_u_test(a, b)
        assert "p_value" in result


# ---------------------------------------------------------------------------
# Z-test for proportions
# ---------------------------------------------------------------------------


class TestZTestProportions:
    def test_returns_required_keys(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.z_test_proportions(a, b)
        for key in ["test_statistic", "p_value", "is_significant",
                     "test_used", "effect_size",
                     "proportion_a", "proportion_b", "proportion_diff"]:
            assert key in result, f"Missing key: {key}"

    def test_detects_proportion_difference(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.z_test_proportions(a, b)
        assert result["is_significant"] is True

    def test_cohens_h_sign(self, suite, binary_groups):
        a, b = binary_groups
        result = suite.z_test_proportions(a, b)
        # a has higher proportion => positive Cohen's h
        if result["proportion_a"] > result["proportion_b"]:
            assert result["effect_size"] > 0

    def test_equal_proportions(self, suite):
        np.random.seed(42)
        a = np.random.binomial(1, 0.25, 500)
        b = np.random.binomial(1, 0.25, 500)
        result = suite.z_test_proportions(a, b)
        assert result["p_value"] > 0.01

    def test_zero_se_handled(self, suite):
        """All zeros should not crash."""
        a = np.zeros(100, dtype=int)
        b = np.zeros(100, dtype=int)
        result = suite.z_test_proportions(a, b)
        assert result["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Multiple Comparison Corrections
# ---------------------------------------------------------------------------


class TestBonferroniCorrection:
    def test_adjusts_p_values(self):
        p_values = [0.01, 0.04, 0.20]
        result = MultipleComparisonCorrection.bonferroni(p_values)
        assert len(result["adjusted_p_values"]) == 3
        # 0.01 * 3 = 0.03
        assert abs(result["adjusted_p_values"][0] - 0.03) < 1e-10
        # 0.04 * 3 = 0.12
        assert abs(result["adjusted_p_values"][1] - 0.12) < 1e-10

    def test_caps_at_one(self):
        p_values = [0.5, 0.9]
        result = MultipleComparisonCorrection.bonferroni(p_values)
        assert all(p <= 1.0 for p in result["adjusted_p_values"])

    def test_rejected_count(self):
        p_values = [0.001, 0.01, 0.02, 0.5]
        result = MultipleComparisonCorrection.bonferroni(p_values, alpha=0.05)
        # 0.001*4=0.004 < 0.05, 0.01*4=0.04 < 0.05, 0.02*4=0.08 >= 0.05
        assert result["n_rejected"] == 2

    def test_empty_input(self):
        result = MultipleComparisonCorrection.bonferroni([])
        assert result["adjusted_p_values"] == []
        assert result["n_rejected"] == 0

    def test_method_label(self):
        result = MultipleComparisonCorrection.bonferroni([0.05])
        assert result["method"] == "bonferroni"


class TestFDRCorrection:
    def test_adjusts_p_values(self):
        p_values = [0.001, 0.01, 0.04, 0.50]
        result = MultipleComparisonCorrection.fdr_bh(p_values)
        assert len(result["adjusted_p_values"]) == 4
        # FDR is less conservative than Bonferroni
        bonf = MultipleComparisonCorrection.bonferroni(p_values)
        assert result["n_rejected"] >= bonf["n_rejected"]

    def test_monotonicity(self):
        """Adjusted p-values in sorted order should be monotonically non-decreasing."""
        p_values = [0.03, 0.001, 0.15, 0.005, 0.90]
        result = MultipleComparisonCorrection.fdr_bh(p_values)
        sorted_indices = np.argsort(p_values)
        adj_sorted = [result["adjusted_p_values"][i] for i in sorted_indices]
        for i in range(1, len(adj_sorted)):
            assert adj_sorted[i] >= adj_sorted[i - 1] - 1e-10

    def test_caps_at_one(self):
        p_values = [0.8, 0.9]
        result = MultipleComparisonCorrection.fdr_bh(p_values)
        assert all(p <= 1.0 for p in result["adjusted_p_values"])

    def test_empty_input(self):
        result = MultipleComparisonCorrection.fdr_bh([])
        assert result["n_rejected"] == 0

    def test_method_label(self):
        result = MultipleComparisonCorrection.fdr_bh([0.05])
        assert result["method"] == "fdr_bh"

    def test_single_p_value(self):
        result = MultipleComparisonCorrection.fdr_bh([0.03], alpha=0.05)
        assert abs(result["adjusted_p_values"][0] - 0.03) < 1e-10
        assert result["rejected"][0] == True


class TestHolmBonferroniCorrection:
    def test_adjusts_p_values(self):
        p_values = [0.001, 0.01, 0.04, 0.50]
        result = MultipleComparisonCorrection.holm_bonferroni(p_values)
        assert len(result["adjusted_p_values"]) == 4

    def test_more_powerful_than_bonferroni(self):
        """Holm should reject at least as many as Bonferroni."""
        p_values = [0.001, 0.008, 0.015, 0.04, 0.50]
        holm = MultipleComparisonCorrection.holm_bonferroni(p_values)
        bonf = MultipleComparisonCorrection.bonferroni(p_values)
        assert holm["n_rejected"] >= bonf["n_rejected"]

    def test_caps_at_one(self):
        p_values = [0.7, 0.8, 0.9]
        result = MultipleComparisonCorrection.holm_bonferroni(p_values)
        assert all(p <= 1.0 for p in result["adjusted_p_values"])

    def test_empty_input(self):
        result = MultipleComparisonCorrection.holm_bonferroni([])
        assert result["n_rejected"] == 0

    def test_method_label(self):
        result = MultipleComparisonCorrection.holm_bonferroni([0.05])
        assert result["method"] == "holm_bonferroni"


# ---------------------------------------------------------------------------
# Multi-metric testing
# ---------------------------------------------------------------------------


class TestMultiMetricTesting:
    def test_returns_individual_results(self, suite, experiment_data):
        result = suite.run_multiple_tests(
            data=experiment_data,
            metrics=["revenue", "churned", "converted"],
        )
        assert "individual_results" in result
        assert len(result["individual_results"]) == 3

    def test_returns_correction(self, suite, experiment_data):
        result = suite.run_multiple_tests(
            data=experiment_data,
            metrics=["revenue", "churned"],
            correction_method="bonferroni",
        )
        assert "correction" in result
        assert result["correction"]["method"] == "bonferroni"

    def test_fdr_correction(self, suite, experiment_data):
        result = suite.run_multiple_tests(
            data=experiment_data,
            metrics=["revenue", "churned"],
            correction_method="fdr_bh",
        )
        assert result["correction"]["method"] == "fdr_bh"

    def test_holm_correction(self, suite, experiment_data):
        result = suite.run_multiple_tests(
            data=experiment_data,
            metrics=["revenue", "churned"],
            correction_method="holm_bonferroni",
        )
        assert result["correction"]["method"] == "holm_bonferroni"

    def test_individual_results_have_adjusted_p(self, suite, experiment_data):
        result = suite.run_multiple_tests(
            data=experiment_data,
            metrics=["revenue", "churned"],
        )
        for metric in ["revenue", "churned"]:
            assert "adjusted_p_value" in result["individual_results"][metric]
            assert "is_significant_corrected" in result["individual_results"][metric]

    def test_auto_selects_test_type(self, suite, experiment_data):
        result = suite.run_multiple_tests(
            data=experiment_data,
            metrics=["revenue", "churned"],
        )
        # Revenue is continuous => t-test
        assert result["individual_results"]["revenue"]["test_used"] in (
            "welch_t_test", "student_t_test"
        )
        # Churned is binary => chi-square
        assert result["individual_results"]["churned"]["test_used"] == "chi_square"

    def test_metrics_list_returned(self, suite, experiment_data):
        metrics = ["revenue", "session_time"]
        result = suite.run_multiple_tests(
            data=experiment_data,
            metrics=metrics,
        )
        assert result["metrics"] == metrics


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_ab_test_framework_accessible(self):
        assert ABTestFramework is not None

    def test_power_analysis_accessible(self):
        assert PowerAnalysis is not None
