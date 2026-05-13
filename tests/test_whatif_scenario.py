"""
TDD Tests for What-If Scenario Analysis in BudgetOptimizer.

Tests cover:
- Scenario simulation with parameter variations (cost, uplift, churn, CLV multipliers)
- Parameter variation / sensitivity analysis (sweeping single parameters)
- Comparative analysis of allocation strategies (LP vs proportional vs uniform)
- Multi-scenario comparison via compare_budget_scenarios
- Budget constraint satisfaction under all scenarios
- Monotonicity properties (higher budget -> more retained value)
- Edge cases (zero budget, extreme multipliers, all sleeping dogs)
- Reproducibility with same seed
"""

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
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_data():
    """Create synthetic customer data for scenario analysis."""
    np.random.seed(42)
    n = 300

    churn_prob = np.random.beta(2, 5, n)
    clv = np.random.lognormal(10, 1, n)
    uplift = np.random.randn(n) * 0.1
    cost_per_action = np.random.choice([5000, 10000, 20000, 50000], size=n)
    expected_retention_lift = np.clip(uplift + 0.05, 0, 0.5)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_prob": churn_prob,
        "clv": clv,
        "uplift_score": uplift,
        "cost_per_action": cost_per_action,
        "expected_retention_lift": expected_retention_lift,
    })


@pytest.fixture
def optimizer(config):
    """Create a BudgetOptimizer instance."""
    from src.models.budget_optimizer import BudgetOptimizer
    return BudgetOptimizer(config)


# ---------------------------------------------------------------------------
# simulate_scenario interface tests
# ---------------------------------------------------------------------------

class TestSimulateScenarioInterface:
    """Test simulate_scenario method exists and has correct interface."""

    def test_has_simulate_scenario(self, optimizer):
        assert hasattr(optimizer, "simulate_scenario")
        assert callable(optimizer.simulate_scenario)

    def test_has_vary_parameter(self, optimizer):
        assert hasattr(optimizer, "vary_parameter")
        assert callable(optimizer.vary_parameter)

    def test_has_compare_strategies(self, optimizer):
        assert hasattr(optimizer, "compare_strategies")
        assert callable(optimizer.compare_strategies)

    def test_has_compare_budget_scenarios(self, optimizer):
        assert hasattr(optimizer, "compare_budget_scenarios")
        assert callable(optimizer.compare_budget_scenarios)


# ---------------------------------------------------------------------------
# Scenario simulation tests
# ---------------------------------------------------------------------------

class TestSimulateScenario:
    """Test running individual what-if scenarios."""

    def test_returns_dict(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="baseline",
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="test",
        )
        required = [
            "scenario_name", "parameters", "total_budget",
            "total_allocated", "retained_value", "roi",
            "customers_treated", "allocation",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_scenario_name_preserved(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="my_scenario",
        )
        assert result["scenario_name"] == "my_scenario"

    def test_parameters_dict_recorded(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="test",
            total_budget=10_000_000,
            cost_multiplier=1.5,
            uplift_multiplier=0.8,
            churn_multiplier=1.2,
            clv_multiplier=0.9,
        )
        params = result["parameters"]
        assert params["total_budget"] == 10_000_000
        assert params["cost_multiplier"] == 1.5
        assert params["uplift_multiplier"] == 0.8
        assert params["churn_multiplier"] == 1.2
        assert params["clv_multiplier"] == 0.9

    def test_custom_budget(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="low_budget",
            total_budget=5_000_000,
        )
        assert result["total_budget"] == 5_000_000
        assert result["total_allocated"] <= 5_000_000 * 1.001

    def test_allocation_is_dataframe(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(data=sample_data)
        assert isinstance(result["allocation"], pd.DataFrame)
        assert "customer_id" in result["allocation"].columns
        assert "allocated_budget" in result["allocation"].columns

    def test_roi_is_numeric(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(data=sample_data)
        assert isinstance(result["roi"], (int, float, np.floating))

    def test_customers_treated_is_int(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(data=sample_data)
        assert isinstance(result["customers_treated"], (int, np.integer))
        assert result["customers_treated"] >= 0

    def test_retained_value_non_negative(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(data=sample_data)
        assert result["retained_value"] >= 0


# ---------------------------------------------------------------------------
# Parameter variation tests
# ---------------------------------------------------------------------------

class TestParameterVariation:
    """Test cost, uplift, churn, and CLV multipliers."""

    def test_cost_multiplier_effect(self, optimizer, sample_data):
        """Higher cost -> lower efficiency (fewer treated or less value)."""
        r_base = optimizer.simulate_scenario(
            data=sample_data, scenario_name="base", cost_multiplier=1.0,
            total_budget=5_000_000,
        )
        r_high = optimizer.simulate_scenario(
            data=sample_data, scenario_name="high_cost", cost_multiplier=3.0,
            total_budget=5_000_000,
        )
        # With same budget but tripled costs, retained value should be lower
        assert r_high["retained_value"] <= r_base["retained_value"] * 1.05

    def test_uplift_multiplier_effect(self, optimizer, sample_data):
        """Higher uplift multiplier -> higher retained value."""
        r_low = optimizer.simulate_scenario(
            data=sample_data, scenario_name="low_uplift",
            uplift_multiplier=0.5, total_budget=10_000_000,
        )
        r_high = optimizer.simulate_scenario(
            data=sample_data, scenario_name="high_uplift",
            uplift_multiplier=2.0, total_budget=10_000_000,
        )
        assert r_high["retained_value"] >= r_low["retained_value"] * 0.95

    def test_churn_multiplier_clamped(self, optimizer, sample_data):
        """Churn multiplier should clamp probabilities to [0, 1]."""
        result = optimizer.simulate_scenario(
            data=sample_data, scenario_name="extreme_churn",
            churn_multiplier=10.0,
        )
        # Should not error and should produce valid results
        assert result["total_allocated"] >= 0

    def test_clv_multiplier_effect(self, optimizer, sample_data):
        """Higher CLV multiplier -> higher retained value."""
        r_low = optimizer.simulate_scenario(
            data=sample_data, scenario_name="low_clv",
            clv_multiplier=0.5, total_budget=10_000_000,
        )
        r_high = optimizer.simulate_scenario(
            data=sample_data, scenario_name="high_clv",
            clv_multiplier=2.0, total_budget=10_000_000,
        )
        assert r_high["retained_value"] >= r_low["retained_value"] * 0.95

    def test_zero_uplift_multiplier(self, optimizer, sample_data):
        """Zero uplift -> zero retained value."""
        result = optimizer.simulate_scenario(
            data=sample_data, scenario_name="no_uplift",
            uplift_multiplier=0.0,
        )
        assert result["retained_value"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# vary_parameter sensitivity analysis tests
# ---------------------------------------------------------------------------

class TestVaryParameter:
    """Test vary_parameter sensitivity analysis."""

    def test_returns_dataframe(self, optimizer, sample_data):
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="budget",
            values=[5_000_000, 10_000_000, 20_000_000],
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_rows(self, optimizer, sample_data):
        values = [5_000_000, 10_000_000, 30_000_000, 50_000_000]
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="budget",
            values=values,
        )
        assert len(result) == len(values)

    def test_has_expected_columns(self, optimizer, sample_data):
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="budget",
            values=[10_000_000],
        )
        expected_cols = [
            "parameter_value", "total_allocated",
            "retained_value", "roi", "customers_treated",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_budget_sweep_monotonic_retained_value(self, optimizer, sample_data):
        """Retained value should be non-decreasing as budget increases."""
        budgets = [1_000_000, 5_000_000, 10_000_000, 30_000_000]
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="budget",
            values=budgets,
        )
        rvs = result["retained_value"].values
        for i in range(1, len(rvs)):
            assert rvs[i] >= rvs[i - 1] * 0.95, (
                f"Retained value decreased: {rvs[i]:.0f} < {rvs[i-1]:.0f}"
            )

    def test_cost_multiplier_sweep(self, optimizer, sample_data):
        """Sweeping cost multiplier should work."""
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="cost_multiplier",
            values=[0.5, 1.0, 2.0],
            base_budget=10_000_000,
        )
        assert len(result) == 3
        assert all(result["total_allocated"] >= 0)

    def test_uplift_multiplier_sweep(self, optimizer, sample_data):
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="uplift_multiplier",
            values=[0.5, 1.0, 1.5, 2.0],
            base_budget=10_000_000,
        )
        assert len(result) == 4

    def test_churn_multiplier_sweep(self, optimizer, sample_data):
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="churn_multiplier",
            values=[0.5, 1.0, 2.0],
            base_budget=10_000_000,
        )
        assert len(result) == 3

    def test_clv_multiplier_sweep(self, optimizer, sample_data):
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="clv_multiplier",
            values=[0.5, 1.0, 2.0],
            base_budget=10_000_000,
        )
        assert len(result) == 3

    def test_invalid_parameter_raises(self, optimizer, sample_data):
        with pytest.raises(ValueError, match="Unknown parameter"):
            optimizer.vary_parameter(
                data=sample_data,
                parameter="invalid_param",
                values=[1.0],
            )

    def test_parameter_values_recorded(self, optimizer, sample_data):
        values = [5_000_000, 15_000_000, 25_000_000]
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="budget",
            values=values,
        )
        np.testing.assert_array_equal(
            result["parameter_value"].values, values,
        )


# ---------------------------------------------------------------------------
# Strategy comparison tests
# ---------------------------------------------------------------------------

class TestCompareStrategies:
    """Test comparative analysis of allocation strategies."""

    def test_returns_dataframe(self, optimizer, sample_data):
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=10_000_000,
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_three_strategies(self, optimizer, sample_data):
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=10_000_000,
        )
        assert len(result) == 3
        assert set(result["strategy"].values) == {"lp", "proportional", "uniform"}

    def test_has_expected_columns(self, optimizer, sample_data):
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=10_000_000,
        )
        expected = ["strategy", "total_allocated", "retained_value",
                     "roi", "customers_treated"]
        for col in expected:
            assert col in result.columns

    def test_all_strategies_produce_positive_retained_value(self, optimizer, sample_data):
        """All strategies should produce positive retained value."""
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=5_000_000,
        )
        for _, row in result.iterrows():
            assert row["retained_value"] > 0, (
                f"Strategy {row['strategy']} should produce positive retained value"
            )

    def test_strategies_differ_in_allocation(self, optimizer, sample_data):
        """Different strategies should produce different allocation patterns."""
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=5_000_000,
        )
        # At least two strategies should have different customers_treated
        treated_counts = result["customers_treated"].values
        assert len(set(treated_counts)) >= 2, (
            "Strategies should differ in number of customers treated"
        )

    def test_all_strategies_non_negative_allocation(self, optimizer, sample_data):
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=10_000_000,
        )
        assert (result["total_allocated"] >= 0).all()

    def test_all_strategies_non_negative_roi(self, optimizer, sample_data):
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=10_000_000,
        )
        assert (result["roi"] >= 0).all()

    def test_all_strategies_respect_budget(self, optimizer, sample_data):
        budget = 10_000_000
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=budget,
        )
        assert (result["total_allocated"] <= budget * 1.01).all()

    def test_uses_default_budget(self, optimizer, sample_data, config):
        """compare_strategies should use config budget when none specified."""
        result = optimizer.compare_strategies(data=sample_data)
        assert len(result) == 3
        # All allocations should be within config budget
        default_budget = config["budget"]["total_krw"]
        assert (result["total_allocated"] <= default_budget * 1.01).all()


# ---------------------------------------------------------------------------
# Multi-scenario comparison tests
# ---------------------------------------------------------------------------

class TestCompareBudgetScenarios:
    """Test compare_budget_scenarios convenience method."""

    def test_returns_dataframe(self, optimizer, sample_data):
        scenarios = [
            {"scenario_name": "low", "total_budget": 5_000_000},
            {"scenario_name": "high", "total_budget": 30_000_000},
        ]
        result = optimizer.compare_budget_scenarios(
            data=sample_data,
            scenarios=scenarios,
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_rows(self, optimizer, sample_data):
        scenarios = [
            {"scenario_name": "a", "total_budget": 5_000_000},
            {"scenario_name": "b", "total_budget": 10_000_000},
            {"scenario_name": "c", "total_budget": 20_000_000},
        ]
        result = optimizer.compare_budget_scenarios(
            data=sample_data,
            scenarios=scenarios,
        )
        assert len(result) == 3

    def test_scenario_names_preserved(self, optimizer, sample_data):
        scenarios = [
            {"scenario_name": "alpha"},
            {"scenario_name": "beta"},
        ]
        result = optimizer.compare_budget_scenarios(
            data=sample_data,
            scenarios=scenarios,
        )
        assert set(result["scenario_name"].values) == {"alpha", "beta"}

    def test_has_metric_columns(self, optimizer, sample_data):
        scenarios = [{"scenario_name": "test"}]
        result = optimizer.compare_budget_scenarios(
            data=sample_data,
            scenarios=scenarios,
        )
        for col in ["scenario_name", "total_budget", "total_allocated",
                     "retained_value", "roi", "customers_treated"]:
            assert col in result.columns

    def test_mixed_parameter_scenarios(self, optimizer, sample_data):
        """Handle scenarios with different multiplier combinations."""
        scenarios = [
            {"scenario_name": "baseline",
             "total_budget": 20_000_000},
            {"scenario_name": "cheap_interventions",
             "total_budget": 20_000_000,
             "cost_multiplier": 0.5},
            {"scenario_name": "high_churn_high_clv",
             "total_budget": 20_000_000,
             "churn_multiplier": 1.5,
             "clv_multiplier": 1.5},
        ]
        result = optimizer.compare_budget_scenarios(
            data=sample_data,
            scenarios=scenarios,
        )
        assert len(result) == 3
        # All retained values should be non-negative
        assert (result["retained_value"] >= 0).all()

    def test_default_scenario_name(self, optimizer, sample_data):
        """Scenarios without explicit names should get auto-generated names."""
        scenarios = [
            {"total_budget": 10_000_000},
            {"total_budget": 20_000_000},
        ]
        result = optimizer.compare_budget_scenarios(
            data=sample_data,
            scenarios=scenarios,
        )
        assert "scenario_0" in result["scenario_name"].values
        assert "scenario_1" in result["scenario_name"].values


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestWhatIfEdgeCases:
    """Test edge cases for what-if analysis."""

    def test_zero_budget_scenario(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="zero",
            total_budget=0,
        )
        assert result["total_allocated"] == 0
        assert result["customers_treated"] == 0

    def test_very_large_budget(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="unlimited",
            total_budget=1_000_000_000_000,
        )
        assert result["total_allocated"] > 0

    def test_zero_clv_multiplier(self, optimizer, sample_data):
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="no_clv",
            clv_multiplier=0.0,
        )
        assert result["retained_value"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_churn_multiplier(self, optimizer, sample_data):
        """Zero churn -> zero priority -> zero allocation."""
        result = optimizer.simulate_scenario(
            data=sample_data,
            scenario_name="no_churn",
            churn_multiplier=0.0,
        )
        assert result["retained_value"] == pytest.approx(0.0, abs=1e-6)

    def test_compare_strategies_zero_budget(self, optimizer, sample_data):
        result = optimizer.compare_strategies(
            data=sample_data,
            total_budget=0,
        )
        assert len(result) == 3
        assert (result["total_allocated"] == 0).all()

    def test_vary_parameter_single_value(self, optimizer, sample_data):
        result = optimizer.vary_parameter(
            data=sample_data,
            parameter="budget",
            values=[10_000_000],
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestWhatIfReproducibility:
    """Test reproducibility of what-if analysis."""

    def test_same_seed_same_scenario(self, config, sample_data):
        from src.models.budget_optimizer import BudgetOptimizer

        opt1 = BudgetOptimizer(config)
        r1 = opt1.simulate_scenario(
            data=sample_data,
            scenario_name="test",
            total_budget=20_000_000,
            cost_multiplier=1.5,
        )

        opt2 = BudgetOptimizer(config)
        r2 = opt2.simulate_scenario(
            data=sample_data,
            scenario_name="test",
            total_budget=20_000_000,
            cost_multiplier=1.5,
        )

        assert r1["retained_value"] == pytest.approx(r2["retained_value"], rel=1e-6)
        assert r1["customers_treated"] == r2["customers_treated"]

    def test_same_seed_same_strategy_comparison(self, config, sample_data):
        from src.models.budget_optimizer import BudgetOptimizer

        opt1 = BudgetOptimizer(config)
        s1 = opt1.compare_strategies(data=sample_data, total_budget=10_000_000)

        opt2 = BudgetOptimizer(config)
        s2 = opt2.compare_strategies(data=sample_data, total_budget=10_000_000)

        np.testing.assert_array_almost_equal(
            s1["retained_value"].values,
            s2["retained_value"].values,
            decimal=2,
        )
