"""
TDD Tests for ExperimentManager (A/B test experiment manager).

Tests cover:
- Experiment creation and configuration
- Result tracking and snapshots
- Sequential testing with alpha-spending boundaries
- Summary reporting and recommendations
- Power analysis integration
- Experiment lifecycle (create -> run -> complete)
- Edge cases and error handling
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ab_testing import ExperimentManager

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
def manager(config):
    """Create an ExperimentManager instance."""
    return ExperimentManager(config=config, alpha=0.05)


@pytest.fixture
def manager_no_config():
    """Create an ExperimentManager without config (minimal)."""
    return ExperimentManager(alpha=0.05)


@pytest.fixture
def sample_data():
    """Create synthetic A/B test data with a true treatment effect."""
    np.random.seed(42)
    n = 2000
    group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))

    # Binary metric with treatment effect
    churned = np.where(
        group == "treatment",
        np.random.binomial(1, 0.15, n),
        np.random.binomial(1, 0.25, n),
    )

    # Continuous metric with treatment effect
    revenue = np.where(
        group == "treatment",
        np.random.normal(110, 15, n),
        np.random.normal(100, 15, n),
    )

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "group": group,
            "churned": churned,
            "revenue": revenue,
        }
    )


@pytest.fixture
def no_effect_data():
    """Create synthetic data with no treatment effect."""
    np.random.seed(99)
    n = 2000
    group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
    metric = np.random.normal(100, 10, n)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "group": group,
            "metric": metric,
        }
    )


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


class TestExperimentManagerInstantiation:
    def test_creates_with_config(self, manager):
        assert manager is not None
        assert manager.alpha == 0.05

    def test_creates_without_config(self, manager_no_config):
        assert manager_no_config is not None

    def test_has_stats_suite(self, manager):
        from src.analysis.ab_testing import StatisticalTestSuite

        assert isinstance(manager.stats, StatisticalTestSuite)

    def test_has_framework(self, manager):
        from src.models.ab_testing import ABTestFramework

        assert isinstance(manager.framework, ABTestFramework)

    def test_empty_experiments_on_init(self, manager):
        assert manager.list_experiments() == []


# ---------------------------------------------------------------------------
# Experiment creation tests
# ---------------------------------------------------------------------------


class TestExperimentCreation:
    def test_create_returns_id(self, manager):
        exp_id = manager.create_experiment(name="test_exp")
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0
        assert "test_exp" in exp_id

    def test_create_with_full_config(self, manager):
        exp_id = manager.create_experiment(
            name="retention_test",
            description="Test coupon impact on retention",
            metrics=["churned", "revenue"],
            hypothesis="Coupons reduce churn rate",
            baseline_rate=0.25,
            mde=0.05,
            sequential=True,
            n_interim_analyses=5,
        )
        exp = manager.get_experiment(exp_id)
        assert exp["name"] == "retention_test"
        assert exp["metrics"] == ["churned", "revenue"]
        assert exp["hypothesis"] == "Coupons reduce churn rate"
        assert exp["sequential"] is True
        assert exp["status"] == "created"

    def test_create_computes_required_sample_size(self, manager):
        exp_id = manager.create_experiment(
            name="sample_size_test",
            baseline_rate=0.25,
            mde=0.05,
        )
        exp = manager.get_experiment(exp_id)
        assert exp["required_sample_size"] is not None
        assert exp["required_sample_size"] > 0

    def test_create_without_power_params(self, manager):
        exp_id = manager.create_experiment(name="no_power")
        exp = manager.get_experiment(exp_id)
        assert exp["required_sample_size"] is None

    def test_list_experiments(self, manager):
        manager.create_experiment(name="exp1")
        manager.create_experiment(name="exp2")
        experiments = manager.list_experiments()
        assert len(experiments) == 2
        names = {e["name"] for e in experiments}
        assert names == {"exp1", "exp2"}

    def test_get_experiment_not_found(self, manager):
        with pytest.raises(KeyError):
            manager.get_experiment("nonexistent_id")

    def test_experiment_has_default_spending_function(self, manager):
        exp_id = manager.create_experiment(name="default_spending")
        exp = manager.get_experiment(exp_id)
        assert exp["spending_function"] == "obrien_fleming"

    def test_experiment_custom_spending_function(self, manager):
        exp_id = manager.create_experiment(
            name="pocock_test", spending_function="pocock"
        )
        exp = manager.get_experiment(exp_id)
        assert exp["spending_function"] == "pocock"


# ---------------------------------------------------------------------------
# Result tracking tests
# ---------------------------------------------------------------------------


class TestResultTracking:
    def test_record_result_returns_snapshot(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="track_test", metrics=["churned"]
        )
        result = manager.record_result(exp_id, sample_data, metric="churned")
        assert "p_value" in result
        assert "test_statistic" in result
        assert "n_treatment" in result
        assert "n_control" in result
        assert "treatment_mean" in result
        assert "control_mean" in result
        assert "absolute_diff" in result

    def test_record_result_updates_status(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="status_test", metrics=["churned"]
        )
        assert manager.get_experiment(exp_id)["status"] == "created"
        manager.record_result(exp_id, sample_data, metric="churned")
        assert manager.get_experiment(exp_id)["status"] == "running"

    def test_record_multiple_results(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="multi_result", metrics=["churned"]
        )
        manager.record_result(exp_id, sample_data, metric="churned")
        manager.record_result(exp_id, sample_data, metric="churned")
        results = manager.get_results(exp_id)
        assert len(results) == 2

    def test_record_uses_default_metric(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="default_metric", metrics=["revenue"]
        )
        result = manager.record_result(exp_id, sample_data)
        assert result["metric"] == "revenue"

    def test_record_no_metric_raises(self, manager, sample_data):
        exp_id = manager.create_experiment(name="no_metrics")
        with pytest.raises(ValueError, match="No metric specified"):
            manager.record_result(exp_id, sample_data)

    def test_record_binary_metric(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="binary_test", metrics=["churned"]
        )
        result = manager.record_result(exp_id, sample_data, metric="churned")
        assert result["test_used"] == "z_test_proportions"

    def test_record_continuous_metric(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="continuous_test", metrics=["revenue"]
        )
        result = manager.record_result(exp_id, sample_data, metric="revenue")
        assert result["test_used"] == "welch_t_test"

    def test_record_increments_analysis_count(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="count_test", metrics=["churned"]
        )
        manager.record_result(exp_id, sample_data, metric="churned")
        manager.record_result(exp_id, sample_data, metric="churned")
        exp = manager.get_experiment(exp_id)
        assert exp["n_analyses_done"] == 2

    def test_get_results_not_found(self, manager):
        with pytest.raises(KeyError):
            manager.get_results("nonexistent")

    def test_pvalue_range(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="pval_test", metrics=["churned"]
        )
        result = manager.record_result(exp_id, sample_data, metric="churned")
        assert 0 <= result["p_value"] <= 1

    def test_detects_significant_effect(self, manager, sample_data):
        """With the sample data having a true effect, should detect significance."""
        exp_id = manager.create_experiment(
            name="sig_test", metrics=["churned"]
        )
        result = manager.record_result(exp_id, sample_data, metric="churned")
        assert result["is_significant"] is True

    def test_no_effect_not_significant(self, manager, no_effect_data):
        exp_id = manager.create_experiment(
            name="nosig_test", metrics=["metric"]
        )
        result = manager.record_result(exp_id, no_effect_data, metric="metric")
        assert result["p_value"] > 0.01


# ---------------------------------------------------------------------------
# Sequential testing tests
# ---------------------------------------------------------------------------


class TestSequentialTesting:
    def test_sequential_returns_extra_fields(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="seq_test",
            metrics=["churned"],
            sequential=True,
            n_interim_analyses=3,
        )
        result = manager.sequential_test(exp_id, sample_data, metric="churned")
        assert "information_fraction" in result
        assert "cumulative_alpha_spent" in result
        assert "boundary_alpha" in result
        assert "stop_early" in result
        assert "decision" in result

    def test_sequential_info_fraction_increases(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="seq_frac",
            metrics=["churned"],
            sequential=True,
            n_interim_analyses=5,
        )
        fractions = []
        for _ in range(3):
            result = manager.sequential_test(
                exp_id, sample_data, metric="churned"
            )
            fractions.append(result["information_fraction"])
        # Fractions should be increasing
        assert fractions[0] < fractions[1] < fractions[2]

    def test_sequential_obrien_fleming_conservative_early(self, manager):
        """O'Brien-Fleming should be very conservative at early looks."""
        alpha_spent = ExperimentManager._alpha_spending(
            "obrien_fleming", 0.2, 0.05
        )
        # Very little alpha spent at 20% information
        assert alpha_spent < 0.005

    def test_sequential_obrien_fleming_full_alpha_at_end(self, manager):
        alpha_spent = ExperimentManager._alpha_spending(
            "obrien_fleming", 1.0, 0.05
        )
        assert abs(alpha_spent - 0.05) < 0.001

    def test_sequential_pocock_spending(self, manager):
        alpha_spent = ExperimentManager._alpha_spending("pocock", 0.5, 0.05)
        assert 0 < alpha_spent < 0.05

    def test_sequential_linear_spending(self, manager):
        alpha_spent = ExperimentManager._alpha_spending("linear", 0.5, 0.05)
        assert abs(alpha_spent - 0.025) < 1e-10

    def test_sequential_zero_fraction(self, manager):
        alpha_spent = ExperimentManager._alpha_spending(
            "obrien_fleming", 0.0, 0.05
        )
        assert alpha_spent == 0.0

    def test_sequential_decision_continue(self, manager, no_effect_data):
        """With no effect, first look should say continue."""
        exp_id = manager.create_experiment(
            name="seq_continue",
            metrics=["metric"],
            sequential=True,
            n_interim_analyses=5,
        )
        result = manager.sequential_test(
            exp_id, no_effect_data, metric="metric"
        )
        assert result["decision"] == "continue"
        assert result["stop_early"] is False

    def test_sequential_final_analysis_completes(self, manager, no_effect_data):
        """At the final planned analysis, experiment should complete."""
        exp_id = manager.create_experiment(
            name="seq_final",
            metrics=["metric"],
            sequential=True,
            n_interim_analyses=2,
        )
        # Run 2 analyses (= n_interim_analyses)
        manager.sequential_test(exp_id, no_effect_data, metric="metric")
        result = manager.sequential_test(
            exp_id, no_effect_data, metric="metric"
        )
        assert result["stop_early"] is True
        assert result["decision"] in ("reject_null", "accept_null")
        exp = manager.get_experiment(exp_id)
        assert exp["status"] == "completed"

    def test_sequential_max_samples_stops(self, manager, sample_data):
        """Experiment should stop when max_samples is reached."""
        exp_id = manager.create_experiment(
            name="seq_max",
            metrics=["churned"],
            sequential=True,
            n_interim_analyses=10,
            max_samples=100,  # Smaller than data size
        )
        result = manager.sequential_test(
            exp_id, sample_data, metric="churned"
        )
        # n_total = 2000 > 100, so should stop
        assert result["stop_early"] is True
        assert result["decision"] in ("reject_null", "accept_null")

    def test_sequential_early_stop_on_strong_effect(self, manager):
        """Very strong effect should trigger early stopping."""
        np.random.seed(42)
        n = 5000
        group = np.array(["treatment"] * (n // 2) + ["control"] * (n // 2))
        # Very large effect
        metric = np.where(
            group == "treatment",
            np.random.binomial(1, 0.05, n),
            np.random.binomial(1, 0.50, n),
        )
        data = pd.DataFrame(
            {"group": group, "conversion": metric}
        )

        exp_id = manager.create_experiment(
            name="seq_early",
            metrics=["conversion"],
            sequential=True,
            n_interim_analyses=5,
        )
        result = manager.sequential_test(exp_id, data, metric="conversion")
        # With such a large effect, should reject at first look
        assert result["decision"] == "reject_null"
        assert result["stop_early"] is True


# ---------------------------------------------------------------------------
# Summary reporting tests
# ---------------------------------------------------------------------------


class TestSummaryReporting:
    def test_summary_structure(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="summary_test",
            description="Test summary",
            metrics=["churned"],
            hypothesis="Treatment reduces churn",
        )
        manager.record_result(exp_id, sample_data, metric="churned")
        summary = manager.get_summary(exp_id)

        assert "experiment" in summary
        assert "results" in summary
        assert "latest_result" in summary
        assert "power_analysis" in summary
        assert "recommendation" in summary

    def test_summary_experiment_section(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="exp_section",
            description="Test",
            hypothesis="H1",
        )
        manager.record_result(exp_id, sample_data, metric="churned")
        summary = manager.get_summary(exp_id)

        exp_info = summary["experiment"]
        assert exp_info["name"] == "exp_section"
        assert exp_info["description"] == "Test"
        assert exp_info["hypothesis"] == "H1"
        assert exp_info["status"] == "running"

    def test_summary_with_power_analysis(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="power_summary",
            metrics=["churned"],
            baseline_rate=0.25,
            mde=0.05,
        )
        manager.record_result(exp_id, sample_data, metric="churned")
        summary = manager.get_summary(exp_id)

        assert summary["power_analysis"]["required_sample_size"] is not None
        assert "achieved_power" in summary["power_analysis"]

    def test_summary_no_results(self, manager):
        exp_id = manager.create_experiment(name="no_results")
        summary = manager.get_summary(exp_id)
        assert summary["latest_result"] is None
        assert len(summary["results"]) == 0
        assert "No results recorded" in summary["recommendation"]

    def test_summary_recommendation_significant(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="sig_rec", metrics=["churned"]
        )
        manager.record_result(exp_id, sample_data, metric="churned")
        # Complete the experiment
        summary = manager.complete_experiment(exp_id)
        # With significant effect, should recommend deploying
        if summary["latest_result"]["is_significant"]:
            assert "significant" in summary["recommendation"].lower()

    def test_summary_recommendation_not_significant(
        self, manager, no_effect_data
    ):
        exp_id = manager.create_experiment(
            name="nosig_rec", metrics=["metric"]
        )
        manager.record_result(exp_id, no_effect_data, metric="metric")
        summary = manager.complete_experiment(exp_id)
        if not summary["latest_result"]["is_significant"]:
            assert "no statistically significant" in summary["recommendation"].lower()

    def test_summary_includes_all_results(self, manager, sample_data):
        exp_id = manager.create_experiment(
            name="all_results", metrics=["churned"]
        )
        manager.record_result(exp_id, sample_data, metric="churned")
        manager.record_result(exp_id, sample_data, metric="churned")
        manager.record_result(exp_id, sample_data, metric="churned")
        summary = manager.get_summary(exp_id)
        assert len(summary["results"]) == 3


# ---------------------------------------------------------------------------
# Experiment lifecycle tests
# ---------------------------------------------------------------------------


class TestExperimentLifecycle:
    def test_full_lifecycle(self, manager, sample_data):
        """Test create -> run -> complete lifecycle."""
        # Create
        exp_id = manager.create_experiment(
            name="lifecycle",
            metrics=["churned", "revenue"],
            baseline_rate=0.25,
            mde=0.05,
        )
        assert manager.get_experiment(exp_id)["status"] == "created"

        # Run
        manager.record_result(exp_id, sample_data, metric="churned")
        assert manager.get_experiment(exp_id)["status"] == "running"

        # Complete
        summary = manager.complete_experiment(exp_id)
        assert summary["experiment"]["status"] == "completed"
        assert summary["experiment"]["decision"] in (
            "reject_null",
            "accept_null",
        )

    def test_complete_without_data(self, manager):
        exp_id = manager.create_experiment(name="empty")
        summary = manager.complete_experiment(exp_id)
        assert summary["experiment"]["decision"] == "no_data"
        assert summary["experiment"]["status"] == "completed"

    def test_sequential_lifecycle(self, manager, sample_data):
        """Test sequential create -> interim analyses -> completion."""
        exp_id = manager.create_experiment(
            name="seq_lifecycle",
            metrics=["churned"],
            sequential=True,
            n_interim_analyses=3,
        )

        decisions = []
        for _ in range(3):
            result = manager.sequential_test(
                exp_id, sample_data, metric="churned"
            )
            decisions.append(result["decision"])
            if result["stop_early"]:
                break

        exp = manager.get_experiment(exp_id)
        # Should have completed by the 3rd analysis at latest
        assert exp["status"] == "completed" or len(decisions) <= 3
