"""
TDD Tests for A/B Testing Framework Module.

Tests cover:
- A/B test experiment creation and configuration
- Treatment/control group assignment (50/50 split from config)
- Randomized group assignment with seed reproducibility
- Statistical significance testing (chi-squared, t-test, z-test)
- Sample size validation (minimum group size from config)
- Metric computation (conversion rate, retention rate, revenue lift)
- Experiment result summary and reporting
- Multi-variant (A/B/n) test support
- Sequential testing / early stopping
- Effect size and confidence interval estimation
- Integration with uplift modeling results
- Experiment persistence (save/load)
- Configurable parameters from YAML
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

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
def sample_experiment_data():
    """Create synthetic experiment data with treatment/control outcomes.

    Simulates customer-level data for an A/B test where treatment group
    received a retention intervention and control did not.
    """
    np.random.seed(42)
    n = 4000

    # Group assignment: 50% treatment, 50% control
    group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
    np.random.shuffle(group)

    # Baseline churn probability
    base_churn = np.random.beta(2, 5, n)

    # Treatment effect: treatment group has slightly lower churn
    treatment_effect = np.where(group == "treatment", -0.05, 0.0)
    actual_churn = np.clip(base_churn + treatment_effect, 0, 1)
    churned = np.random.binomial(1, actual_churn)

    # Revenue (higher for retained customers)
    revenue = np.where(
        churned == 0,
        np.random.lognormal(10, 1, n),
        np.random.lognormal(8, 1, n) * 0.1,
    )

    # Conversion events
    converted = np.random.binomial(1, np.where(group == "treatment", 0.15, 0.10))

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "group": group,
        "churned": churned,
        "revenue": revenue,
        "converted": converted,
        "churn_prob": base_churn,
        "days_active": np.random.poisson(30, n),
    })

    return df


@pytest.fixture
def ab_test_framework(config):
    """Create an A/B testing framework instance."""
    from src.models.ab_testing import ABTestFramework
    return ABTestFramework(config)


# ---------------------------------------------------------------------------
# Framework interface tests
# ---------------------------------------------------------------------------

class TestABTestFrameworkInterface:
    """Test A/B testing framework instantiation and interface."""

    def test_instantiation(self, ab_test_framework):
        """A/B test framework must be instantiable from config."""
        assert ab_test_framework is not None

    def test_has_create_experiment_method(self, ab_test_framework):
        """Must implement experiment creation."""
        assert hasattr(ab_test_framework, "create_experiment")
        assert callable(ab_test_framework.create_experiment)

    def test_has_assign_groups_method(self, ab_test_framework):
        """Must implement group assignment."""
        assert hasattr(ab_test_framework, "assign_groups")
        assert callable(ab_test_framework.assign_groups)

    def test_has_analyze_method(self, ab_test_framework):
        """Must implement experiment analysis."""
        assert hasattr(ab_test_framework, "analyze")
        assert callable(ab_test_framework.analyze)

    def test_has_compute_significance_method(self, ab_test_framework):
        """Must implement statistical significance testing."""
        assert hasattr(ab_test_framework, "compute_significance")
        assert callable(ab_test_framework.compute_significance)

    def test_has_get_summary_method(self, ab_test_framework):
        """Must implement experiment summary reporting."""
        assert hasattr(ab_test_framework, "get_summary")
        assert callable(ab_test_framework.get_summary)

    def test_reads_treatment_ratio_from_config(self, ab_test_framework, config):
        """Framework must read treatment ratio from config."""
        expected_ratio = config["treatment"]["treatment_ratio"]
        assert ab_test_framework.treatment_ratio == expected_ratio


# ---------------------------------------------------------------------------
# Group assignment tests
# ---------------------------------------------------------------------------

class TestGroupAssignment:
    """Test treatment/control group assignment."""

    def test_assigns_all_customers(self, ab_test_framework,
                                    sample_experiment_data):
        """All customers must be assigned to a group."""
        assignments = ab_test_framework.assign_groups(
            customer_ids=sample_experiment_data["customer_id"].tolist(),
        )

        assert len(assignments) == len(sample_experiment_data), (
            f"Expected {len(sample_experiment_data)} assignments, "
            f"got {len(assignments)}"
        )

    def test_only_valid_groups(self, ab_test_framework,
                                sample_experiment_data):
        """Only 'treatment' and 'control' group labels should be assigned."""
        assignments = ab_test_framework.assign_groups(
            customer_ids=sample_experiment_data["customer_id"].tolist(),
        )

        unique_groups = set(assignments["group"].unique())
        assert unique_groups <= {"treatment", "control"}, (
            f"Unexpected groups: {unique_groups - {'treatment', 'control'}}"
        )

    def test_approximate_split_ratio(self, ab_test_framework, config,
                                      sample_experiment_data):
        """Group sizes should approximately match the configured ratio."""
        expected_ratio = config["treatment"]["treatment_ratio"]
        assignments = ab_test_framework.assign_groups(
            customer_ids=sample_experiment_data["customer_id"].tolist(),
        )

        treatment_count = (assignments["group"] == "treatment").sum()
        actual_ratio = treatment_count / len(assignments)

        assert abs(actual_ratio - expected_ratio) < 0.05, (
            f"Treatment ratio {actual_ratio:.3f} deviates from "
            f"expected {expected_ratio:.3f}"
        )

    def test_no_duplicate_assignments(self, ab_test_framework,
                                       sample_experiment_data):
        """Each customer must be assigned to exactly one group."""
        assignments = ab_test_framework.assign_groups(
            customer_ids=sample_experiment_data["customer_id"].tolist(),
        )

        assert assignments["customer_id"].nunique() == len(assignments), (
            "Duplicate customer assignments found"
        )

    def test_minimum_group_size(self, ab_test_framework, config):
        """Each group must meet minimum size requirements."""
        min_size = config["treatment"]["min_group_size"]
        n_customers = max(min_size * 3, 30000)
        customer_ids = [f"C{i:06d}" for i in range(n_customers)]

        assignments = ab_test_framework.assign_groups(
            customer_ids=customer_ids,
        )

        treatment_size = (assignments["group"] == "treatment").sum()
        control_size = (assignments["group"] == "control").sum()

        assert treatment_size >= min_size, (
            f"Treatment group size {treatment_size} < minimum {min_size}"
        )
        assert control_size >= min_size, (
            f"Control group size {control_size} < minimum {min_size}"
        )


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

class TestStatisticalSignificance:
    """Test statistical significance computation."""

    def test_significance_returns_p_value(self, ab_test_framework,
                                           sample_experiment_data):
        """compute_significance must return a p-value."""
        result = ab_test_framework.compute_significance(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "p_value" in result, "Result must include p_value"
        assert 0 <= result["p_value"] <= 1, (
            f"p-value {result['p_value']} out of [0, 1] range"
        )

    def test_significance_returns_test_statistic(self, ab_test_framework,
                                                   sample_experiment_data):
        """compute_significance must return a test statistic."""
        result = ab_test_framework.compute_significance(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "test_statistic" in result, (
            "Result must include test_statistic"
        )
        assert isinstance(result["test_statistic"], (int, float,
                                                       np.floating))

    def test_significance_returns_significant_flag(self, ab_test_framework,
                                                     sample_experiment_data):
        """compute_significance must indicate statistical significance."""
        result = ab_test_framework.compute_significance(
            data=sample_experiment_data,
            metric="churned",
            alpha=0.05,
        )

        assert "is_significant" in result, (
            "Result must include is_significant flag"
        )
        assert isinstance(result["is_significant"], bool)

    def test_significance_with_continuous_metric(self, ab_test_framework,
                                                   sample_experiment_data):
        """Must handle continuous metrics (e.g., revenue) using t-test."""
        result = ab_test_framework.compute_significance(
            data=sample_experiment_data,
            metric="revenue",
        )

        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_significance_with_binary_metric(self, ab_test_framework,
                                               sample_experiment_data):
        """Must handle binary metrics (e.g., churned) using z-test/chi2."""
        result = ab_test_framework.compute_significance(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_no_effect_high_p_value(self, ab_test_framework):
        """When there is no true effect, p-value should tend to be high."""
        np.random.seed(123)
        n = 2000
        data = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "group": ["treatment"] * (n // 2) + ["control"] * (n // 2),
            "metric": np.random.normal(100, 10, n),  # No difference
        })

        result = ab_test_framework.compute_significance(
            data=data,
            metric="metric",
        )

        # With no real effect, p-value should typically be > 0.01
        assert result["p_value"] > 0.001, (
            "p-value suspiciously low for a null effect"
        )


# ---------------------------------------------------------------------------
# Effect size and confidence interval tests
# ---------------------------------------------------------------------------

class TestEffectSize:
    """Test effect size and confidence interval estimation."""

    def test_effect_size_returned(self, ab_test_framework,
                                   sample_experiment_data):
        """Analysis must return effect size."""
        result = ab_test_framework.analyze(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "effect_size" in result, "Result must include effect_size"

    def test_confidence_interval_returned(self, ab_test_framework,
                                            sample_experiment_data):
        """Analysis must return confidence interval."""
        result = ab_test_framework.analyze(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "confidence_interval" in result, (
            "Result must include confidence_interval"
        )
        ci = result["confidence_interval"]
        assert len(ci) == 2, "CI must be a (lower, upper) tuple/list"
        assert ci[0] <= ci[1], "CI lower bound must be <= upper bound"

    def test_effect_size_sign_matches_direction(self, ab_test_framework,
                                                  sample_experiment_data):
        """Effect size sign should match treatment vs control difference."""
        result = ab_test_framework.analyze(
            data=sample_experiment_data,
            metric="churned",
        )

        treatment_mean = sample_experiment_data[
            sample_experiment_data["group"] == "treatment"
        ]["churned"].mean()
        control_mean = sample_experiment_data[
            sample_experiment_data["group"] == "control"
        ]["churned"].mean()

        observed_diff = treatment_mean - control_mean
        if abs(observed_diff) > 0.001:
            assert np.sign(result["effect_size"]) == np.sign(observed_diff), (
                "Effect size direction doesn't match observed difference"
            )


# ---------------------------------------------------------------------------
# Experiment summary tests
# ---------------------------------------------------------------------------

class TestExperimentSummary:
    """Test experiment summary / reporting."""

    def test_summary_includes_group_sizes(self, ab_test_framework,
                                            sample_experiment_data):
        """Summary must report group sizes."""
        summary = ab_test_framework.get_summary(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "treatment_size" in summary
        assert "control_size" in summary
        assert summary["treatment_size"] > 0
        assert summary["control_size"] > 0

    def test_summary_includes_group_means(self, ab_test_framework,
                                            sample_experiment_data):
        """Summary must report group means for the metric."""
        summary = ab_test_framework.get_summary(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "treatment_mean" in summary
        assert "control_mean" in summary

    def test_summary_includes_lift(self, ab_test_framework,
                                     sample_experiment_data):
        """Summary must report relative lift (treatment vs control)."""
        summary = ab_test_framework.get_summary(
            data=sample_experiment_data,
            metric="churned",
        )

        assert "relative_lift" in summary, (
            "Summary must include relative_lift"
        )

    def test_summary_returns_dict(self, ab_test_framework,
                                    sample_experiment_data):
        """Summary must return a dictionary."""
        summary = ab_test_framework.get_summary(
            data=sample_experiment_data,
            metric="churned",
        )

        assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# Multi-variant tests
# ---------------------------------------------------------------------------

class TestMultiVariantExperiment:
    """Test A/B/n (multi-variant) experiment support."""

    def test_assign_multiple_variants(self, ab_test_framework):
        """Must support assigning more than two groups."""
        customer_ids = [f"C{i:05d}" for i in range(6000)]
        assignments = ab_test_framework.assign_groups(
            customer_ids=customer_ids,
            n_variants=3,
        )

        unique_groups = assignments["group"].nunique()
        assert unique_groups == 3, (
            f"Expected 3 variant groups, got {unique_groups}"
        )

    def test_multi_variant_balanced(self, ab_test_framework):
        """Multi-variant groups should be approximately balanced."""
        customer_ids = [f"C{i:05d}" for i in range(9000)]
        assignments = ab_test_framework.assign_groups(
            customer_ids=customer_ids,
            n_variants=3,
        )

        group_sizes = assignments["group"].value_counts()
        expected_size = len(customer_ids) / 3

        for grp, size in group_sizes.items():
            assert abs(size - expected_size) / expected_size < 0.1, (
                f"Group '{grp}' has {size} members, expected ~{expected_size:.0f}"
            )


# ---------------------------------------------------------------------------
# Experiment creation tests
# ---------------------------------------------------------------------------

class TestExperimentCreation:
    """Test experiment creation and management."""

    def test_create_experiment_returns_id(self, ab_test_framework):
        """create_experiment must return an experiment identifier."""
        exp_id = ab_test_framework.create_experiment(
            name="retention_coupon_test",
            description="Test coupon effectiveness on retention",
        )

        assert exp_id is not None
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0

    def test_create_experiment_with_metrics(self, ab_test_framework):
        """Experiment must accept a list of metrics to track."""
        exp_id = ab_test_framework.create_experiment(
            name="push_notification_test",
            description="Test push notification impact",
            metrics=["churned", "revenue", "converted"],
        )

        assert exp_id is not None


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestABTestReproducibility:
    """Test A/B testing reproducibility with same seed."""

    def test_same_seed_same_assignments(self, config):
        """Same seed must produce identical group assignments."""
        from src.models.ab_testing import ABTestFramework

        customer_ids = [f"C{i:05d}" for i in range(2000)]

        fw1 = ABTestFramework(config)
        assignments1 = fw1.assign_groups(customer_ids=customer_ids)

        fw2 = ABTestFramework(config)
        assignments2 = fw2.assign_groups(customer_ids=customer_ids)

        pd.testing.assert_frame_equal(
            assignments1.sort_values("customer_id").reset_index(drop=True),
            assignments2.sort_values("customer_id").reset_index(drop=True),
        )

    def test_same_seed_same_analysis(self, config, sample_experiment_data):
        """Same seed must produce identical analysis results."""
        from src.models.ab_testing import ABTestFramework

        fw1 = ABTestFramework(config)
        result1 = fw1.analyze(data=sample_experiment_data, metric="churned")

        fw2 = ABTestFramework(config)
        result2 = fw2.analyze(data=sample_experiment_data, metric="churned")

        assert abs(result1["effect_size"] - result2["effect_size"]) < 1e-10


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestABTestPersistence:
    """Test A/B testing framework save/load functionality."""

    def test_save_experiment(self, ab_test_framework, tmp_path):
        """Experiment state must be saveable."""
        ab_test_framework.create_experiment(
            name="test_experiment",
            description="Persistence test",
        )

        save_path = tmp_path / "ab_test_state"
        ab_test_framework.save(str(save_path))

        saved_files = list(tmp_path.glob("ab_test_state*"))
        assert len(saved_files) > 0, "No experiment state saved"

    def test_load_experiment(self, ab_test_framework, tmp_path):
        """Saved experiment must be loadable."""
        from src.models.ab_testing import ABTestFramework

        ab_test_framework.create_experiment(
            name="test_experiment",
            description="Persistence test",
        )

        save_path = tmp_path / "ab_test_state"
        ab_test_framework.save(str(save_path))

        loaded = ABTestFramework.load(str(save_path))
        assert loaded is not None
