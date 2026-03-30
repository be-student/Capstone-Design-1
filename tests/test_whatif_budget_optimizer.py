"""
TDD Tests for What-If Scenario Analysis in LPBudgetOptimizer.

Tests cover:
- WhatIfScenario dataclass and scenario definition
- simulate_budget_change: run optimization under modified parameters
- compare_scenarios: baseline vs alternatives side-by-side
- Impact projections: churn rate changes, revenue impact, customers treated
- Budget sweep with LP optimizer
- Churn rate impact projections
- Revenue impact projections
- Edge cases (zero budget, empty data, extreme multipliers)
- Reproducibility
- Integration with existing LPBudgetOptimizer.solve()
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
def sample_data():
    """Create synthetic customer data for scenario testing."""
    np.random.seed(42)
    n = 200

    churn_prob = np.random.beta(2, 5, n)
    clv = np.random.lognormal(10, 1, n)
    uplift = np.random.randn(n) * 0.1
    cost_per_action = np.random.choice([5000, 10000, 20000], size=n)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_prob": churn_prob,
        "clv": clv,
        "uplift_score": uplift,
        "cost_per_action": cost_per_action,
    })


@pytest.fixture
def channels():
    """Channel constraints for multi-channel tests."""
    from src.optimization.budget_optimizer import ChannelConstraint
    return [
        ChannelConstraint(name="email", min_budget=0, max_budget=20_000_000,
                          cost_per_action=5000, expected_roi_multiplier=1.0),
        ChannelConstraint(name="sms", min_budget=0, max_budget=15_000_000,
                          cost_per_action=2000, expected_roi_multiplier=0.8),
    ]


@pytest.fixture
def optimizer(config, channels):
    """Create an LPBudgetOptimizer with channels."""
    from src.optimization.budget_optimizer import LPBudgetOptimizer
    return LPBudgetOptimizer(config, channels=channels)


@pytest.fixture
def optimizer_default(config):
    """Create an LPBudgetOptimizer with default single channel."""
    from src.optimization.budget_optimizer import LPBudgetOptimizer
    return LPBudgetOptimizer(config)


# ---------------------------------------------------------------------------
# WhatIfScenario dataclass tests
# ---------------------------------------------------------------------------

class TestWhatIfScenarioDefinition:
    """Test scenario definition structures."""

    def test_whatif_scenario_import(self):
        """WhatIfScenario must be importable."""
        from src.optimization.budget_optimizer import WhatIfScenario
        assert WhatIfScenario is not None

    def test_whatif_scenario_creation(self):
        """WhatIfScenario can be created with name and budget."""
        from src.optimization.budget_optimizer import WhatIfScenario
        s = WhatIfScenario(name="baseline", total_budget=50_000_000)
        assert s.name == "baseline"
        assert s.total_budget == 50_000_000

    def test_whatif_scenario_defaults(self):
        """WhatIfScenario has sensible defaults for multipliers."""
        from src.optimization.budget_optimizer import WhatIfScenario
        s = WhatIfScenario(name="test")
        assert s.cost_multiplier == 1.0
        assert s.uplift_multiplier == 1.0
        assert s.total_budget is None  # means use default

    def test_whatif_scenario_with_multipliers(self):
        """WhatIfScenario accepts cost and uplift multipliers."""
        from src.optimization.budget_optimizer import WhatIfScenario
        s = WhatIfScenario(
            name="expensive",
            total_budget=30_000_000,
            cost_multiplier=2.0,
            uplift_multiplier=0.5,
        )
        assert s.cost_multiplier == 2.0
        assert s.uplift_multiplier == 0.5


# ---------------------------------------------------------------------------
# simulate_budget_change tests
# ---------------------------------------------------------------------------

class TestSimulateBudgetChange:
    """Test the simulate_budget_change method."""

    def test_method_exists(self, optimizer):
        """LPBudgetOptimizer must have simulate_budget_change method."""
        assert hasattr(optimizer, "simulate_budget_change")
        assert callable(optimizer.simulate_budget_change)

    def test_returns_dict(self, optimizer, sample_data):
        """simulate_budget_change must return a dict."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="test",
        )
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, optimizer, sample_data):
        """Result must contain required projection keys."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="baseline",
        )
        required_keys = [
            "scenario_name",
            "total_budget",
            "total_allocated",
            "retained_value",
            "roi",
            "customers_treated",
            "projected_churn_rate",
            "projected_revenue_impact",
            "status",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_custom_budget(self, optimizer, sample_data):
        """Must accept a custom budget parameter."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="custom",
            total_budget=10_000_000,
        )
        assert result["total_budget"] == 10_000_000
        assert result["total_allocated"] <= 10_000_000 * 1.001

    def test_cost_multiplier(self, optimizer, sample_data):
        """Must accept a cost multiplier to scale costs."""
        r1 = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="base",
            total_budget=20_000_000,
            cost_multiplier=1.0,
        )
        r2 = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="expensive",
            total_budget=20_000_000,
            cost_multiplier=2.0,
        )
        assert r1["scenario_name"] == "base"
        assert r2["scenario_name"] == "expensive"

    def test_uplift_multiplier(self, optimizer, sample_data):
        """Must accept an uplift multiplier to scale uplift scores."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="boosted",
            uplift_multiplier=1.5,
        )
        assert result["scenario_name"] == "boosted"

    def test_projected_churn_rate_bounded(self, optimizer, sample_data):
        """Projected churn rate must be between 0 and 1."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="test",
        )
        assert 0.0 <= result["projected_churn_rate"] <= 1.0

    def test_projected_revenue_impact_is_numeric(self, optimizer, sample_data):
        """Projected revenue impact must be numeric."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="test",
        )
        assert isinstance(result["projected_revenue_impact"], (int, float, np.floating))

    def test_customers_treated_is_integer(self, optimizer, sample_data):
        """customers_treated must be a non-negative integer."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="test",
        )
        assert isinstance(result["customers_treated"], (int, np.integer))
        assert result["customers_treated"] >= 0


# ---------------------------------------------------------------------------
# compare_scenarios tests
# ---------------------------------------------------------------------------

class TestCompareScenarios:
    """Test scenario comparison."""

    def test_method_exists(self, optimizer):
        """LPBudgetOptimizer must have compare_scenarios method."""
        assert hasattr(optimizer, "compare_scenarios")
        assert callable(optimizer.compare_scenarios)

    def test_returns_dataframe(self, optimizer, sample_data):
        """compare_scenarios must return a DataFrame."""
        from src.optimization.budget_optimizer import WhatIfScenario
        scenarios = [
            WhatIfScenario(name="low", total_budget=10_000_000),
            WhatIfScenario(name="high", total_budget=50_000_000),
        ]
        result = optimizer.compare_scenarios(data=sample_data, scenarios=scenarios)
        assert isinstance(result, pd.DataFrame)

    def test_includes_all_scenarios(self, optimizer, sample_data):
        """Comparison must include all scenario names."""
        from src.optimization.budget_optimizer import WhatIfScenario
        scenarios = [
            WhatIfScenario(name="baseline", total_budget=50_000_000),
            WhatIfScenario(name="cut", total_budget=25_000_000),
            WhatIfScenario(name="boost", total_budget=75_000_000),
        ]
        result = optimizer.compare_scenarios(data=sample_data, scenarios=scenarios)
        assert set(result["scenario_name"].values) == {"baseline", "cut", "boost"}

    def test_has_metric_columns(self, optimizer, sample_data):
        """Comparison DataFrame must have key metric columns."""
        from src.optimization.budget_optimizer import WhatIfScenario
        scenarios = [
            WhatIfScenario(name="A", total_budget=20_000_000),
            WhatIfScenario(name="B", total_budget=40_000_000),
        ]
        result = optimizer.compare_scenarios(data=sample_data, scenarios=scenarios)
        expected_cols = [
            "scenario_name", "total_budget", "total_allocated",
            "retained_value", "roi", "customers_treated",
            "projected_churn_rate", "projected_revenue_impact",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_higher_budget_higher_or_equal_retained_value(self, optimizer, sample_data):
        """Higher budget should yield higher or equal retained value."""
        from src.optimization.budget_optimizer import WhatIfScenario
        scenarios = [
            WhatIfScenario(name="low", total_budget=5_000_000),
            WhatIfScenario(name="high", total_budget=50_000_000),
        ]
        result = optimizer.compare_scenarios(data=sample_data, scenarios=scenarios)
        low_rv = result[result["scenario_name"] == "low"]["retained_value"].values[0]
        high_rv = result[result["scenario_name"] == "high"]["retained_value"].values[0]
        assert high_rv >= low_rv * 0.95

    def test_compare_with_multipliers(self, optimizer, sample_data):
        """Must handle scenarios with varying cost and uplift multipliers."""
        from src.optimization.budget_optimizer import WhatIfScenario
        scenarios = [
            WhatIfScenario(name="baseline", total_budget=50_000_000),
            WhatIfScenario(name="cheap", total_budget=50_000_000,
                           cost_multiplier=0.5, uplift_multiplier=1.5),
        ]
        result = optimizer.compare_scenarios(data=sample_data, scenarios=scenarios)
        assert len(result) == 2

    def test_churn_rate_projection_varies_with_budget(self, optimizer, sample_data):
        """Projected churn rate should decrease (or stay) with more budget."""
        from src.optimization.budget_optimizer import WhatIfScenario
        scenarios = [
            WhatIfScenario(name="minimal", total_budget=1_000_000),
            WhatIfScenario(name="generous", total_budget=50_000_000),
        ]
        result = optimizer.compare_scenarios(data=sample_data, scenarios=scenarios)
        min_churn = result[result["scenario_name"] == "minimal"]["projected_churn_rate"].values[0]
        gen_churn = result[result["scenario_name"] == "generous"]["projected_churn_rate"].values[0]
        # More budget should reduce churn (or at least not increase it significantly)
        assert gen_churn <= min_churn * 1.05


# ---------------------------------------------------------------------------
# Impact projection tests
# ---------------------------------------------------------------------------

class TestImpactProjections:
    """Test churn rate and revenue impact projections."""

    def test_baseline_churn_rate_matches_data(self, optimizer, sample_data):
        """With zero budget, projected churn should match raw churn rate."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="no_intervention",
            total_budget=0,
        )
        expected_churn = float(sample_data["churn_prob"].mean())
        assert result["projected_churn_rate"] == pytest.approx(expected_churn, abs=0.01)

    def test_revenue_impact_positive_with_budget(self, optimizer, sample_data):
        """Revenue impact should be positive when budget is allocated."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="funded",
            total_budget=30_000_000,
        )
        assert result["projected_revenue_impact"] >= 0

    def test_zero_budget_zero_revenue_impact(self, optimizer, sample_data):
        """Zero budget should yield zero revenue impact."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="zero",
            total_budget=0,
        )
        assert result["projected_revenue_impact"] == pytest.approx(0.0, abs=1e-6)

    def test_revenue_impact_increases_with_budget(self, optimizer, sample_data):
        """Revenue impact should be non-decreasing with more budget."""
        r_low = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="low",
            total_budget=5_000_000,
        )
        r_high = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="high",
            total_budget=50_000_000,
        )
        assert r_high["projected_revenue_impact"] >= r_low["projected_revenue_impact"] * 0.95


# ---------------------------------------------------------------------------
# Budget sweep tests
# ---------------------------------------------------------------------------

class TestBudgetSweep:
    """Test budget sweep functionality on LP optimizer."""

    def test_method_exists(self, optimizer):
        """LPBudgetOptimizer must have run_budget_sweep method."""
        assert hasattr(optimizer, "run_budget_sweep")
        assert callable(optimizer.run_budget_sweep)

    def test_returns_dataframe(self, optimizer, sample_data):
        """Budget sweep must return a DataFrame."""
        result = optimizer.run_budget_sweep(
            data=sample_data,
            budget_levels=[10_000_000, 30_000_000, 50_000_000],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_has_expected_columns(self, optimizer, sample_data):
        """Sweep result must include key columns."""
        result = optimizer.run_budget_sweep(
            data=sample_data,
            budget_levels=[10_000_000, 50_000_000],
        )
        for col in ["total_budget", "total_allocated", "retained_value",
                     "roi", "projected_churn_rate"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_monotonic_allocation(self, optimizer, sample_data):
        """Total allocation should be non-decreasing with budget."""
        levels = [5_000_000, 15_000_000, 30_000_000, 50_000_000]
        result = optimizer.run_budget_sweep(
            data=sample_data,
            budget_levels=levels,
        )
        allocs = result["total_allocated"].values
        for i in range(1, len(allocs)):
            assert allocs[i] >= allocs[i - 1] * 0.99


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestWhatIfEdgeCases:
    """Test edge cases for what-if analysis on LP optimizer."""

    def test_zero_budget(self, optimizer, sample_data):
        """Zero budget scenario must produce zero allocation."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="zero",
            total_budget=0,
        )
        assert result["total_allocated"] == 0
        assert result["customers_treated"] == 0

    def test_empty_data(self, optimizer):
        """Empty DataFrame should not crash."""
        empty = pd.DataFrame(columns=[
            "customer_id", "churn_prob", "clv", "uplift_score", "cost_per_action",
        ])
        result = optimizer.simulate_budget_change(
            data=empty,
            scenario_name="empty",
            total_budget=50_000_000,
        )
        assert result["total_allocated"] == 0
        assert result["customers_treated"] == 0

    def test_very_large_budget(self, optimizer, sample_data):
        """Very large budget should not cause errors."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="huge",
            total_budget=1_000_000_000_000,
        )
        assert result["total_allocated"] > 0
        assert result["status"] in ("optimal", "empty")

    def test_zero_uplift_multiplier(self, optimizer, sample_data):
        """Zero uplift multiplier should yield zero retained value."""
        result = optimizer.simulate_budget_change(
            data=sample_data,
            scenario_name="no_uplift",
            uplift_multiplier=0.0,
        )
        assert result["retained_value"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestWhatIfReproducibility:
    """Test reproducibility of what-if analysis."""

    def test_same_config_same_results(self, config, channels, sample_data):
        """Same config must produce identical scenario results."""
        from src.optimization.budget_optimizer import LPBudgetOptimizer

        a1 = LPBudgetOptimizer(config, channels=channels)
        r1 = a1.simulate_budget_change(
            data=sample_data,
            scenario_name="test",
            total_budget=30_000_000,
        )

        a2 = LPBudgetOptimizer(config, channels=channels)
        r2 = a2.simulate_budget_change(
            data=sample_data,
            scenario_name="test",
            total_budget=30_000_000,
        )

        assert r1["retained_value"] == pytest.approx(r2["retained_value"], rel=1e-6)
        assert r1["customers_treated"] == r2["customers_treated"]
        assert r1["projected_churn_rate"] == pytest.approx(
            r2["projected_churn_rate"], rel=1e-6
        )
