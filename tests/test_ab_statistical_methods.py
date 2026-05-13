"""
TDD Tests for A/B Testing Statistical Methods.

Focused on verifying correctness of statistical computations:
- Power analysis: sample size formulas, MDE computation, power curves
- Effect size: Cohen's d/h calculations, direction consistency
- Confidence intervals: coverage properties, width vs sample size
- Statistical test properties: Type I error control, sensitivity
- Integration: A/B framework + StatisticalTestSuite + ExperimentManager
- Edge cases for statistical methods
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def framework(config):
    """ABTestFramework instance."""
    from src.models.ab_testing import ABTestFramework

    return ABTestFramework(config)


@pytest.fixture
def suite():
    """StatisticalTestSuite instance."""
    from src.analysis.ab_testing import StatisticalTestSuite

    return StatisticalTestSuite(alpha=0.05)


@pytest.fixture
def correction():
    """MultipleComparisonCorrection class."""
    from src.analysis.ab_testing import MultipleComparisonCorrection

    return MultipleComparisonCorrection


# ---------------------------------------------------------------------------
# Power Analysis: Mathematical Correctness
# ---------------------------------------------------------------------------


class TestPowerAnalysisMath:
    """Verify mathematical correctness of power analysis formulas."""

    def test_sample_size_increases_with_smaller_mde(self):
        """Smaller MDE should require larger sample size."""
        from src.models.ab_testing import PowerAnalysis

        n_large_mde = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.80
        )
        n_small_mde = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=0.02, alpha=0.05, power=0.80
        )
        assert n_small_mde > n_large_mde

    def test_sample_size_increases_with_higher_power(self):
        """Higher power should require larger sample size."""
        from src.models.ab_testing import PowerAnalysis

        n_80 = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.80
        )
        n_95 = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.95
        )
        assert n_95 > n_80

    def test_sample_size_increases_with_smaller_alpha(self):
        """Smaller alpha (more stringent) should require larger sample size."""
        from src.models.ab_testing import PowerAnalysis

        n_05 = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.80
        )
        n_01 = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=0.05, alpha=0.01, power=0.80
        )
        assert n_01 > n_05

    def test_sample_size_zero_mde_raises(self):
        """MDE of zero should raise ValueError."""
        from src.models.ab_testing import PowerAnalysis

        with pytest.raises(ValueError):
            PowerAnalysis.required_sample_size(
                baseline_rate=0.20, mde=0, alpha=0.05, power=0.80
            )

    def test_sample_size_positive(self):
        """Sample size must always be positive."""
        from src.models.ab_testing import PowerAnalysis

        n = PowerAnalysis.required_sample_size(
            baseline_rate=0.50, mde=0.10, alpha=0.05, power=0.80
        )
        assert n > 0
        assert isinstance(n, int)

    def test_power_roundtrip_consistency(self):
        """Computing power with sample size from required_sample_size
        should return approximately the target power."""
        from src.models.ab_testing import PowerAnalysis

        target_power = 0.80
        n = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=0.05, alpha=0.05, power=target_power
        )
        achieved_power = PowerAnalysis.compute_power(
            n=n, baseline_rate=0.20, mde=0.05, alpha=0.05
        )
        assert abs(achieved_power - target_power) < 0.02

    def test_mde_roundtrip_consistency(self):
        """MDE computed from sample size should approximately match original."""
        from src.models.ab_testing import PowerAnalysis

        original_mde = 0.05
        n = PowerAnalysis.required_sample_size(
            baseline_rate=0.20, mde=original_mde, alpha=0.05, power=0.80
        )
        computed_mde = PowerAnalysis.minimum_detectable_effect(
            n=n, baseline_rate=0.20, alpha=0.05, power=0.80
        )
        assert abs(computed_mde - original_mde) < 0.005

    def test_compute_power_bounds(self):
        """Power must be between 0 and 1."""
        from src.models.ab_testing import PowerAnalysis

        power = PowerAnalysis.compute_power(
            n=1000, baseline_rate=0.20, mde=0.05, alpha=0.05
        )
        assert 0 <= power <= 1

    def test_compute_power_increases_with_n(self):
        """Power should increase with sample size."""
        from src.models.ab_testing import PowerAnalysis

        powers = []
        for n in [100, 500, 1000, 5000]:
            p = PowerAnalysis.compute_power(
                n=n, baseline_rate=0.20, mde=0.05, alpha=0.05
            )
            powers.append(p)
        # Should be monotonically non-decreasing
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1] - 1e-6

    def test_compute_power_one_sided_vs_two_sided(self):
        """One-sided test should have higher power than two-sided."""
        from src.models.ab_testing import PowerAnalysis

        power_two = PowerAnalysis.compute_power(
            n=500, baseline_rate=0.20, mde=0.05, alpha=0.05, two_sided=True
        )
        power_one = PowerAnalysis.compute_power(
            n=500, baseline_rate=0.20, mde=0.05, alpha=0.05, two_sided=False
        )
        assert power_one >= power_two

    def test_mde_positive_sample_size_raises(self):
        """Negative or zero sample size should raise."""
        from src.models.ab_testing import PowerAnalysis

        with pytest.raises(ValueError):
            PowerAnalysis.minimum_detectable_effect(
                n=0, baseline_rate=0.20, alpha=0.05, power=0.80
            )

    def test_mde_invalid_baseline_raises(self):
        """Baseline rate outside (0,1) should raise."""
        from src.models.ab_testing import PowerAnalysis

        with pytest.raises(ValueError):
            PowerAnalysis.minimum_detectable_effect(
                n=1000, baseline_rate=0.0, alpha=0.05, power=0.80
            )
        with pytest.raises(ValueError):
            PowerAnalysis.minimum_detectable_effect(
                n=1000, baseline_rate=1.0, alpha=0.05, power=0.80
            )

    def test_mde_decreases_with_larger_n(self):
        """Larger sample should detect smaller effects."""
        from src.models.ab_testing import PowerAnalysis

        mde_small = PowerAnalysis.minimum_detectable_effect(
            n=500, baseline_rate=0.20, alpha=0.05, power=0.80
        )
        mde_large = PowerAnalysis.minimum_detectable_effect(
            n=5000, baseline_rate=0.20, alpha=0.05, power=0.80
        )
        assert mde_large < mde_small


# ---------------------------------------------------------------------------
# Confidence Interval Properties
# ---------------------------------------------------------------------------


class TestConfidenceIntervalProperties:
    """Test mathematical properties of confidence intervals."""

    def test_ci_contains_observed_effect(self, framework):
        """CI should contain the observed effect size."""
        np.random.seed(42)
        n = 2000
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "metric": np.concatenate(
                    [
                        np.random.normal(105, 10, n // 2),
                        np.random.normal(100, 10, n // 2),
                    ]
                ),
            }
        )
        result = framework.analyze(data, metric="metric")
        ci = result["confidence_interval"]
        assert ci[0] <= result["effect_size"] <= ci[1]

    def test_ci_narrows_with_larger_sample(self, framework):
        """CI width should decrease with larger sample size."""
        np.random.seed(42)
        widths = []
        for n in [200, 2000, 20000]:
            data = pd.DataFrame(
                {
                    "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                    "metric": np.concatenate(
                        [
                            np.random.normal(105, 10, n // 2),
                            np.random.normal(100, 10, n // 2),
                        ]
                    ),
                }
            )
            result = framework.analyze(data, metric="metric")
            ci = result["confidence_interval"]
            widths.append(ci[1] - ci[0])
        # CI widths should decrease
        assert widths[0] > widths[1] > widths[2]

    def test_ci_wider_for_higher_variance(self, framework):
        """Higher variance data should produce wider CIs."""
        np.random.seed(42)
        n = 2000
        data_low_var = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "metric": np.concatenate(
                    [
                        np.random.normal(105, 5, n // 2),
                        np.random.normal(100, 5, n // 2),
                    ]
                ),
            }
        )
        data_high_var = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "metric": np.concatenate(
                    [
                        np.random.normal(105, 50, n // 2),
                        np.random.normal(100, 50, n // 2),
                    ]
                ),
            }
        )
        ci_low = framework.analyze(data_low_var, metric="metric")[
            "confidence_interval"
        ]
        ci_high = framework.analyze(data_high_var, metric="metric")[
            "confidence_interval"
        ]
        width_low = ci_low[1] - ci_low[0]
        width_high = ci_high[1] - ci_high[0]
        assert width_high > width_low

    def test_ci_lower_le_upper(self, framework):
        """Lower bound must always be <= upper bound."""
        np.random.seed(42)
        n = 500
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "metric": np.random.normal(100, 10, n),
            }
        )
        result = framework.analyze(data, metric="metric")
        ci = result["confidence_interval"]
        assert ci[0] <= ci[1]


# ---------------------------------------------------------------------------
# Type I Error Control
# ---------------------------------------------------------------------------


class TestTypeIErrorControl:
    """Test that statistical tests control Type I error rate."""

    def test_z_test_type_i_error_rate(self, framework):
        """Under the null (no effect), p < 0.05 should happen ~5% of the time."""
        np.random.seed(42)
        n = 1000
        rejections = 0
        n_simulations = 100

        for i in range(n_simulations):
            rng = np.random.RandomState(i)
            data = pd.DataFrame(
                {
                    "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                    "converted": rng.binomial(1, 0.20, n),
                }
            )
            result = framework.compute_significance(data, metric="converted")
            if result["p_value"] < 0.05:
                rejections += 1

        # False positive rate should be around 5%, allow up to 12%
        fp_rate = rejections / n_simulations
        assert fp_rate < 0.12, f"Type I error rate {fp_rate:.2f} too high"

    def test_t_test_type_i_error_rate(self, framework):
        """Under null, t-test should reject ~5% of the time."""
        n = 1000
        rejections = 0
        n_simulations = 100

        for i in range(n_simulations):
            rng = np.random.RandomState(i + 1000)
            data = pd.DataFrame(
                {
                    "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                    "revenue": rng.normal(100, 10, n),
                }
            )
            result = framework.compute_significance(data, metric="revenue")
            if result["p_value"] < 0.05:
                rejections += 1

        fp_rate = rejections / n_simulations
        assert fp_rate < 0.12, f"Type I error rate {fp_rate:.2f} too high"


# ---------------------------------------------------------------------------
# Statistical Test Sensitivity
# ---------------------------------------------------------------------------


class TestStatisticalTestSensitivity:
    """Test that tests detect real effects with adequate power."""

    def test_detects_large_binary_effect(self, framework):
        """Large binary effect should be detected (p < 0.05)."""
        np.random.seed(42)
        n = 2000
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "converted": np.concatenate(
                    [
                        np.random.binomial(1, 0.30, n // 2),
                        np.random.binomial(1, 0.10, n // 2),
                    ]
                ),
            }
        )
        result = framework.compute_significance(data, metric="converted")
        assert result["is_significant"] is True
        assert result["p_value"] < 0.001

    def test_detects_large_continuous_effect(self, framework):
        """Large continuous effect should be detected."""
        np.random.seed(42)
        n = 2000
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "revenue": np.concatenate(
                    [
                        np.random.normal(120, 10, n // 2),
                        np.random.normal(100, 10, n // 2),
                    ]
                ),
            }
        )
        result = framework.compute_significance(data, metric="revenue")
        assert result["is_significant"] is True

    def test_small_effect_not_detected_with_small_sample(self, framework):
        """Small effect with small sample should not be detected."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "revenue": np.concatenate(
                    [
                        np.random.normal(101, 20, n // 2),
                        np.random.normal(100, 20, n // 2),
                    ]
                ),
            }
        )
        result = framework.compute_significance(data, metric="revenue")
        # Very small effect + small sample => shouldn't be significant
        assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# StatisticalTestSuite: Effect Size Calculations
# ---------------------------------------------------------------------------


class TestEffectSizeCalculations:
    """Test effect size calculations in StatisticalTestSuite."""

    def test_cohens_d_direction(self, suite):
        """Cohen's d should be positive when group a > group b."""
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0] * 20)
        b = np.array([5.0, 6.0, 7.0, 8.0, 9.0] * 20)
        result = suite.t_test(a, b)
        assert result["effect_size"] > 0

    def test_cohens_d_magnitude(self, suite):
        """Large mean difference with small variance should give large d."""
        np.random.seed(42)
        a = np.random.normal(110, 2, 200)
        b = np.random.normal(100, 2, 200)
        result = suite.t_test(a, b)
        # d = (110-100)/2 = 5.0, very large effect
        assert abs(result["effect_size"]) > 2.0

    def test_chi_square_effect_size_cramers_v_range(self, suite):
        """Cramér's V should be in [0, 1]."""
        np.random.seed(42)
        a = np.random.binomial(1, 0.40, 500)
        b = np.random.binomial(1, 0.20, 500)
        result = suite.chi_square_test(a, b)
        assert 0 <= result["effect_size"] <= 1

    def test_z_test_cohens_h_direction(self, suite):
        """Cohen's h should reflect proportion difference direction."""
        np.random.seed(42)
        a = np.random.binomial(1, 0.40, 500)
        b = np.random.binomial(1, 0.20, 500)
        result = suite.z_test_proportions(a, b)
        # a has higher proportion, so h should be positive
        assert result["effect_size"] > 0

    def test_mann_whitney_effect_size_range(self, suite):
        """Mann-Whitney effect size (rank-biserial r) should be in [-1, 1]."""
        np.random.seed(42)
        a = np.random.normal(110, 10, 200)
        b = np.random.normal(100, 10, 200)
        result = suite.mann_whitney_u_test(a, b)
        assert -1 <= result["effect_size"] <= 1


# ---------------------------------------------------------------------------
# Multiple Comparison Corrections: Mathematical Properties
# ---------------------------------------------------------------------------


class TestMultipleComparisonMath:
    """Test mathematical properties of correction methods."""

    def test_bonferroni_adjusted_ge_raw(self, correction):
        """Bonferroni adjusted p-values should be >= raw p-values."""
        p_values = [0.001, 0.01, 0.04, 0.20, 0.50]
        result = correction.bonferroni(p_values)
        for raw, adj in zip(p_values, result["adjusted_p_values"]):
            assert adj >= raw

    def test_fdr_adjusted_ge_raw(self, correction):
        """FDR adjusted p-values should be >= raw p-values."""
        p_values = [0.001, 0.01, 0.04, 0.20, 0.50]
        result = correction.fdr_bh(p_values)
        for raw, adj in zip(p_values, result["adjusted_p_values"]):
            assert adj >= raw - 1e-10

    def test_holm_adjusted_ge_raw(self, correction):
        """Holm adjusted p-values should be >= raw p-values."""
        p_values = [0.001, 0.01, 0.04, 0.20, 0.50]
        result = correction.holm_bonferroni(p_values)
        for raw, adj in zip(p_values, result["adjusted_p_values"]):
            assert adj >= raw - 1e-10

    def test_bonferroni_most_conservative(self, correction):
        """Bonferroni should reject fewest or equal hypotheses."""
        p_values = [0.001, 0.005, 0.01, 0.03, 0.045]
        bonf = correction.bonferroni(p_values)
        holm = correction.holm_bonferroni(p_values)
        fdr = correction.fdr_bh(p_values)
        assert bonf["n_rejected"] <= holm["n_rejected"]
        assert bonf["n_rejected"] <= fdr["n_rejected"]

    def test_holm_at_least_as_powerful_as_bonferroni(self, correction):
        """Holm should reject at least as many as Bonferroni."""
        p_values = [0.001, 0.008, 0.012, 0.04, 0.50]
        bonf = correction.bonferroni(p_values)
        holm = correction.holm_bonferroni(p_values)
        assert holm["n_rejected"] >= bonf["n_rejected"]

    def test_single_test_no_adjustment(self, correction):
        """Single test should not change p-value for Bonferroni."""
        p_values = [0.03]
        result = correction.bonferroni(p_values)
        assert abs(result["adjusted_p_values"][0] - 0.03) < 1e-10

    def test_all_significant_stays_significant(self, correction):
        """Very small p-values should remain significant after correction."""
        p_values = [0.0001, 0.0002, 0.0003]
        for method in [
            correction.bonferroni,
            correction.fdr_bh,
            correction.holm_bonferroni,
        ]:
            result = method(p_values, alpha=0.05)
            assert result["n_rejected"] == 3

    def test_all_nonsignificant_stays_nonsignificant(self, correction):
        """Large p-values should remain non-significant after correction."""
        p_values = [0.50, 0.60, 0.70]
        for method in [
            correction.bonferroni,
            correction.fdr_bh,
            correction.holm_bonferroni,
        ]:
            result = method(p_values, alpha=0.05)
            assert result["n_rejected"] == 0


# ---------------------------------------------------------------------------
# Cross-Module Integration: Framework + Suite
# ---------------------------------------------------------------------------


class TestABTestingIntegration:
    """Integration tests between ABTestFramework and StatisticalTestSuite."""

    def test_framework_and_suite_agree_on_binary(self, framework, suite):
        """Framework and suite should give consistent p-values for binary data."""
        np.random.seed(42)
        n = 2000
        group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
        converted = np.where(
            group == "treatment",
            np.random.binomial(1, 0.25, n),
            np.random.binomial(1, 0.20, n),
        )
        data = pd.DataFrame({"group": group, "converted": converted})

        fw_result = framework.compute_significance(data, metric="converted")
        suite_result = suite.z_test_proportions(
            converted[: n // 2], converted[n // 2 :]
        )

        # Both should agree on significance direction
        assert fw_result["is_significant"] == suite_result["is_significant"]

    def test_framework_and_suite_agree_on_continuous(self, framework, suite):
        """Framework and suite should agree for continuous data."""
        np.random.seed(42)
        n = 2000
        group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
        revenue = np.where(
            group == "treatment",
            np.random.normal(110, 15, n),
            np.random.normal(100, 15, n),
        )
        data = pd.DataFrame({"group": group, "revenue": revenue})

        fw_result = framework.compute_significance(data, metric="revenue")
        suite_result = suite.t_test(revenue[: n // 2], revenue[n // 2 :])

        # P-values should be very close (same underlying test)
        assert abs(fw_result["p_value"] - suite_result["p_value"]) < 0.01

    def test_multi_metric_pipeline(self, suite):
        """Multi-metric testing pipeline should work end-to-end."""
        np.random.seed(42)
        n = 2000
        group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
        data = pd.DataFrame(
            {
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
            }
        )
        result = suite.run_multiple_tests(
            data=data,
            metrics=["revenue", "churned"],
            correction_method="bonferroni",
        )

        assert len(result["individual_results"]) == 2
        assert result["correction"]["method"] == "bonferroni"
        # Both metrics have real effects, should both be significant
        for metric in ["revenue", "churned"]:
            assert "adjusted_p_value" in result["individual_results"][metric]

    def test_experiment_manager_uses_framework(self, config):
        """ExperimentManager should use ABTestFramework internally."""
        from src.analysis.ab_testing import ExperimentManager

        mgr = ExperimentManager(config=config, alpha=0.05)
        from src.models.ab_testing import ABTestFramework

        assert isinstance(mgr.framework, ABTestFramework)

    def test_experiment_lifecycle_with_stats(self, config):
        """Full experiment lifecycle: create → record → analyze → complete."""
        from src.analysis.ab_testing import ExperimentManager

        np.random.seed(42)
        n = 2000
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "churned": np.concatenate(
                    [
                        np.random.binomial(1, 0.15, n // 2),
                        np.random.binomial(1, 0.25, n // 2),
                    ]
                ),
            }
        )

        mgr = ExperimentManager(config=config, alpha=0.05)

        # Create
        exp_id = mgr.create_experiment(
            name="stats_lifecycle",
            metrics=["churned"],
            baseline_rate=0.25,
            mde=0.05,
        )

        # Record
        result = mgr.record_result(exp_id, data, metric="churned")
        assert "p_value" in result

        # Complete
        summary = mgr.complete_experiment(exp_id)
        assert summary["experiment"]["status"] == "completed"
        assert summary["power_analysis"]["required_sample_size"] is not None


# ---------------------------------------------------------------------------
# ABTestFramework: Analyze Method Properties
# ---------------------------------------------------------------------------


class TestAnalyzeMethodProperties:
    """Test properties of the analyze method output."""

    def test_analyze_returns_all_required_keys(self, framework):
        """Analyze must return a complete set of result keys."""
        np.random.seed(42)
        n = 1000
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "metric": np.random.normal(100, 10, n),
            }
        )
        result = framework.analyze(data, metric="metric")
        required_keys = {
            "effect_size",
            "confidence_interval",
            "treatment_mean",
            "control_mean",
            "treatment_size",
            "control_size",
            "p_value",
            "test_statistic",
            "is_significant",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_group_sizes_match_input(self, framework):
        """Reported group sizes should match actual data."""
        np.random.seed(42)
        n_treat, n_ctrl = 600, 400
        data = pd.DataFrame(
            {
                "group": ["treatment"] * n_treat + ["control"] * n_ctrl,
                "metric": np.random.normal(100, 10, n_treat + n_ctrl),
            }
        )
        result = framework.analyze(data, metric="metric")
        assert result["treatment_size"] == n_treat
        assert result["control_size"] == n_ctrl

    def test_effect_size_equals_mean_difference(self, framework):
        """Effect size should equal treatment_mean - control_mean."""
        np.random.seed(42)
        n = 1000
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "metric": np.concatenate(
                    [
                        np.random.normal(110, 10, n // 2),
                        np.random.normal(100, 10, n // 2),
                    ]
                ),
            }
        )
        result = framework.analyze(data, metric="metric")
        expected = result["treatment_mean"] - result["control_mean"]
        assert abs(result["effect_size"] - expected) < 1e-10

    def test_summary_relative_lift_correct(self, framework):
        """Relative lift should be (treatment - control) / |control|."""
        np.random.seed(42)
        n = 2000
        data = pd.DataFrame(
            {
                "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
                "metric": np.concatenate(
                    [
                        np.random.normal(120, 10, n // 2),
                        np.random.normal(100, 10, n // 2),
                    ]
                ),
            }
        )
        summary = framework.get_summary(data, metric="metric")
        expected_lift = (
            summary["treatment_mean"] - summary["control_mean"]
        ) / abs(summary["control_mean"])
        assert abs(summary["relative_lift"] - expected_lift) < 1e-8


# ---------------------------------------------------------------------------
# Group Assignment Statistical Properties
# ---------------------------------------------------------------------------


class TestGroupAssignmentStatistics:
    """Test statistical properties of group assignment."""

    def test_assignment_ratio_converges(self, config):
        """Assignment ratio should converge to configured ratio for large n."""
        from src.models.ab_testing import ABTestFramework

        fw = ABTestFramework(config)
        ids = [f"C{i}" for i in range(50000)]
        assignments = fw.assign_groups(ids)
        actual_ratio = (assignments["group"] == "treatment").mean()
        expected_ratio = config["treatment"]["treatment_ratio"]
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_multi_variant_uniformity(self, config):
        """Multi-variant groups should be approximately uniform."""
        from src.models.ab_testing import ABTestFramework

        fw = ABTestFramework(config)
        ids = [f"C{i}" for i in range(30000)]
        assignments = fw.assign_groups(ids, n_variants=5)
        counts = assignments["group"].value_counts()
        expected = 30000 / 5
        for count in counts.values:
            assert abs(count - expected) / expected < 0.05
