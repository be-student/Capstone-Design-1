"""
TDD Tests for What-If Scenario Analysis Module.

Tests cover:
- Scenario definition with varying budget, cost, and uplift parameters
- Running multiple scenarios and collecting results
- Side-by-side comparison of optimization outcomes
- Scenario metrics (total spend, retained value, ROI, customers treated)
- Default scenario generation from config
- Custom scenario overrides
- Reproducibility with same seed
- Edge cases (zero budget, extreme multipliers)
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
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_customer_data():
    """Create synthetic customer data for scenario analysis."""
    np.random.seed(42)
    n = 500

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
        "segment": np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "explorer", "dormant", "new_customer"],
            size=n,
        ),
    })


@pytest.fixture
def analyzer(config):
    """Create a WhatIfAnalyzer instance."""
    from src.models.whatif_analysis import WhatIfAnalyzer
    return WhatIfAnalyzer(config)


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------

class TestWhatIfAnalyzerInterface:
    """Test analyzer instantiation and interface."""

    def test_instantiation(self, analyzer):
        """WhatIfAnalyzer must be instantiable from config."""
        assert analyzer is not None

    def test_has_run_scenario_method(self, analyzer):
        """Must have a run_scenario method."""
        assert hasattr(analyzer, "run_scenario")
        assert callable(analyzer.run_scenario)

    def test_has_compare_scenarios_method(self, analyzer):
        """Must have a compare_scenarios method."""
        assert hasattr(analyzer, "compare_scenarios")
        assert callable(analyzer.compare_scenarios)

    def test_has_run_budget_sweep_method(self, analyzer):
        """Must have a method to sweep across budget levels."""
        assert hasattr(analyzer, "run_budget_sweep")
        assert callable(analyzer.run_budget_sweep)

    def test_reads_default_budget(self, analyzer, config):
        """Must read default budget from config."""
        expected = config["budget"]["total_krw"]
        assert analyzer.default_budget == expected


# ---------------------------------------------------------------------------
# Single scenario tests
# ---------------------------------------------------------------------------

class TestSingleScenario:
    """Test running a single what-if scenario."""

    def test_run_scenario_returns_dict(self, analyzer, sample_customer_data):
        """run_scenario must return a dict with scenario results."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="baseline",
        )
        assert isinstance(result, dict)

    def test_scenario_has_required_keys(self, analyzer, sample_customer_data):
        """Scenario result must contain required metrics."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="baseline",
        )
        required_keys = [
            "scenario_name",
            "total_budget",
            "total_allocated",
            "retained_value",
            "roi",
            "customers_treated",
            "allocation",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_scenario_with_custom_budget(self, analyzer, sample_customer_data):
        """Must accept a custom budget override."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="low_budget",
            total_budget=10_000_000,
        )
        assert result["total_budget"] == 10_000_000
        assert result["total_allocated"] <= 10_000_000 * 1.001

    def test_scenario_with_cost_multiplier(self, analyzer, sample_customer_data):
        """Must accept a cost multiplier to scale cost_per_action."""
        result_base = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="base_cost",
            cost_multiplier=1.0,
        )
        result_high = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="high_cost",
            cost_multiplier=2.0,
        )
        # With same budget but doubled costs, fewer customers should be treated
        # or each gets less relative allocation
        assert result_base["scenario_name"] == "base_cost"
        assert result_high["scenario_name"] == "high_cost"

    def test_scenario_with_uplift_multiplier(self, analyzer, sample_customer_data):
        """Must accept an uplift multiplier to scale uplift_score."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="high_uplift",
            uplift_multiplier=1.5,
        )
        assert result["scenario_name"] == "high_uplift"

    def test_roi_is_numeric(self, analyzer, sample_customer_data):
        """ROI metric must be numeric."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="test",
        )
        assert isinstance(result["roi"], (int, float, np.floating))

    def test_customers_treated_count(self, analyzer, sample_customer_data):
        """customers_treated must be a non-negative integer."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="test",
        )
        assert isinstance(result["customers_treated"], (int, np.integer))
        assert result["customers_treated"] >= 0


# ---------------------------------------------------------------------------
# Multi-scenario comparison tests
# ---------------------------------------------------------------------------

class TestScenarioComparison:
    """Test comparing multiple scenarios side by side."""

    def test_compare_returns_dataframe(self, analyzer, sample_customer_data):
        """compare_scenarios must return a DataFrame."""
        scenarios = [
            {"scenario_name": "low", "total_budget": 10_000_000},
            {"scenario_name": "mid", "total_budget": 30_000_000},
            {"scenario_name": "high", "total_budget": 50_000_000},
        ]
        comparison = analyzer.compare_scenarios(
            data=sample_customer_data,
            scenarios=scenarios,
        )
        assert isinstance(comparison, pd.DataFrame)

    def test_compare_has_all_scenarios(self, analyzer, sample_customer_data):
        """Comparison DataFrame must include all scenario names."""
        scenarios = [
            {"scenario_name": "low", "total_budget": 10_000_000},
            {"scenario_name": "high", "total_budget": 50_000_000},
        ]
        comparison = analyzer.compare_scenarios(
            data=sample_customer_data,
            scenarios=scenarios,
        )
        assert set(comparison["scenario_name"].values) == {"low", "high"}

    def test_compare_has_metric_columns(self, analyzer, sample_customer_data):
        """Comparison must include key metric columns."""
        scenarios = [
            {"scenario_name": "A", "total_budget": 20_000_000},
            {"scenario_name": "B", "total_budget": 40_000_000},
        ]
        comparison = analyzer.compare_scenarios(
            data=sample_customer_data,
            scenarios=scenarios,
        )
        expected_cols = [
            "scenario_name", "total_budget", "total_allocated",
            "retained_value", "roi", "customers_treated",
        ]
        for col in expected_cols:
            assert col in comparison.columns, f"Missing column: {col}"

    def test_higher_budget_higher_retained_value(self, analyzer,
                                                  sample_customer_data):
        """Higher budget should yield higher or equal retained value."""
        scenarios = [
            {"scenario_name": "low", "total_budget": 10_000_000},
            {"scenario_name": "high", "total_budget": 50_000_000},
        ]
        comparison = analyzer.compare_scenarios(
            data=sample_customer_data,
            scenarios=scenarios,
        )
        low_rv = comparison[
            comparison["scenario_name"] == "low"
        ]["retained_value"].values[0]
        high_rv = comparison[
            comparison["scenario_name"] == "high"
        ]["retained_value"].values[0]
        assert high_rv >= low_rv * 0.95

    def test_compare_with_cost_and_uplift_variations(self, analyzer,
                                                      sample_customer_data):
        """Must handle scenarios with varying cost and uplift multipliers."""
        scenarios = [
            {"scenario_name": "baseline",
             "total_budget": 50_000_000,
             "cost_multiplier": 1.0,
             "uplift_multiplier": 1.0},
            {"scenario_name": "cheap_high_uplift",
             "total_budget": 50_000_000,
             "cost_multiplier": 0.5,
             "uplift_multiplier": 1.5},
            {"scenario_name": "expensive_low_uplift",
             "total_budget": 50_000_000,
             "cost_multiplier": 2.0,
             "uplift_multiplier": 0.5},
        ]
        comparison = analyzer.compare_scenarios(
            data=sample_customer_data,
            scenarios=scenarios,
        )
        assert len(comparison) == 3


# ---------------------------------------------------------------------------
# Budget sweep tests
# ---------------------------------------------------------------------------

class TestBudgetSweep:
    """Test budget sweep functionality."""

    def test_budget_sweep_returns_dataframe(self, analyzer,
                                            sample_customer_data):
        """Budget sweep must return a DataFrame with results per level."""
        result = analyzer.run_budget_sweep(
            data=sample_customer_data,
            budget_levels=[10_000_000, 30_000_000, 50_000_000],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_budget_sweep_monotonic_allocation(self, analyzer,
                                               sample_customer_data):
        """Total allocation should be non-decreasing with budget."""
        levels = [5_000_000, 15_000_000, 30_000_000, 50_000_000]
        result = analyzer.run_budget_sweep(
            data=sample_customer_data,
            budget_levels=levels,
        )
        allocations = result["total_allocated"].values
        for i in range(1, len(allocations)):
            assert allocations[i] >= allocations[i - 1] * 0.99

    def test_budget_sweep_has_roi_column(self, analyzer,
                                         sample_customer_data):
        """Budget sweep result must include ROI column."""
        result = analyzer.run_budget_sweep(
            data=sample_customer_data,
            budget_levels=[10_000_000, 50_000_000],
        )
        assert "roi" in result.columns


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestWhatIfEdgeCases:
    """Test edge cases for what-if analysis."""

    def test_zero_budget_scenario(self, analyzer, sample_customer_data):
        """Zero budget scenario must produce zero allocation."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="zero",
            total_budget=0,
        )
        assert result["total_allocated"] == 0
        assert result["customers_treated"] == 0

    def test_very_large_budget(self, analyzer, sample_customer_data):
        """Very large budget should not cause errors."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="unlimited",
            total_budget=1_000_000_000_000,
        )
        assert result["total_allocated"] > 0

    def test_zero_uplift_multiplier(self, analyzer, sample_customer_data):
        """Zero uplift multiplier should yield zero retained value."""
        result = analyzer.run_scenario(
            data=sample_customer_data,
            scenario_name="no_uplift",
            uplift_multiplier=0.0,
        )
        assert result["retained_value"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestWhatIfReproducibility:
    """Test reproducibility of scenario analysis."""

    def test_same_config_same_results(self, config, sample_customer_data):
        """Same config must produce identical scenario results."""
        from src.models.whatif_analysis import WhatIfAnalyzer

        a1 = WhatIfAnalyzer(config)
        r1 = a1.run_scenario(
            data=sample_customer_data,
            scenario_name="test",
            total_budget=30_000_000,
        )

        a2 = WhatIfAnalyzer(config)
        r2 = a2.run_scenario(
            data=sample_customer_data,
            scenario_name="test",
            total_budget=30_000_000,
        )

        assert r1["retained_value"] == pytest.approx(
            r2["retained_value"], rel=1e-6
        )
        assert r1["customers_treated"] == r2["customers_treated"]
