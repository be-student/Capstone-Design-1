"""
TDD Tests for Budget Optimization Logic.

Focused on verifying correctness of optimization algorithms:
- LP formulation: objective function, constraint correctness
- Optimality conditions: KKT-like properties for LP solutions
- Allocation mathematical properties: monotonicity, convexity
- Multi-channel optimization logic
- ROI computation correctness
- Budget allocation fairness and efficiency
- Integration: BudgetOptimizer + LPBudgetOptimizer + CostConfig
- Stress tests for edge cases in optimization
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
def budget_optimizer(config):
    """BudgetOptimizer from src/models."""
    from src.models.budget_optimizer import BudgetOptimizer

    return BudgetOptimizer(config)


@pytest.fixture
def lp_optimizer(config):
    """LPBudgetOptimizer from src/optimization."""
    from src.optimization.budget_optimizer import LPBudgetOptimizer

    return LPBudgetOptimizer(config)


@pytest.fixture
def sample_data():
    """Synthetic customer data for optimization tests."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "churn_prob": np.random.beta(2, 5, n),
            "clv": np.random.lognormal(10, 1, n),
            "uplift_score": np.random.randn(n) * 0.1,
            "cost_per_action": np.random.choice([5000, 10000, 20000, 50000], size=n),
            "expected_retention_lift": np.clip(
                np.random.randn(n) * 0.1 + 0.05, 0, 0.5
            ),
        }
    )


@pytest.fixture
def controlled_data():
    """Small controlled dataset for precise verification."""
    return pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E"],
            "churn_prob": [0.9, 0.7, 0.5, 0.3, 0.1],
            "clv": [100000, 80000, 60000, 40000, 20000],
            "uplift_score": [0.4, 0.3, 0.2, 0.1, -0.1],
            "cost_per_action": [10000, 10000, 10000, 10000, 10000],
            "expected_retention_lift": [0.35, 0.25, 0.15, 0.05, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# LP Formulation Correctness
# ---------------------------------------------------------------------------


class TestLPFormulationCorrectness:
    """Test that the LP is correctly formulated."""

    def test_objective_maximizes_retained_value(self, lp_optimizer, controlled_data):
        """LP should allocate budget to maximise retained CLV."""
        result = lp_optimizer.solve(controlled_data, total_budget=20000)
        alloc = lp_optimizer.get_customer_allocations(result)
        merged = alloc.merge(controlled_data, on="customer_id")

        # Compute priority: uplift * clv * churn_prob
        merged["priority"] = np.maximum(merged["uplift_score"], 0) * merged["clv"] * merged["churn_prob"]

        # Customer A has highest priority (0.4 * 100000 * 0.9 = 36000)
        # With tight budget, A should get most allocation
        a_budget = merged[merged["customer_id"] == "A"]["allocated_budget"].values[0]
        assert a_budget > 0, "Highest-priority customer should receive budget"

    def test_sleeping_dogs_excluded_from_objective(self, lp_optimizer, controlled_data):
        """Customers with negative uplift should get zero (priority clipped)."""
        result = lp_optimizer.solve(controlled_data, total_budget=20000)
        alloc = lp_optimizer.get_customer_allocations(result)
        merged = alloc.merge(controlled_data, on="customer_id")

        sleeping = merged[merged["uplift_score"] < 0]
        assert sleeping["allocated_budget"].sum() < 1.0

    def test_budget_fully_utilized_when_beneficial(self, lp_optimizer, sample_data):
        """When there are beneficial customers, LP should use most of the budget."""
        budget = 1_000_000
        result = lp_optimizer.solve(sample_data, total_budget=budget)
        # With 300 customers and positive uplift customers, should use most budget
        utilization = result.total_allocated / budget
        assert utilization > 0.80, f"Budget utilization {utilization:.2%} too low"

    def test_per_customer_cap_respected(self, lp_optimizer, controlled_data):
        """Each customer's allocation should not exceed cost_per_action."""
        result = lp_optimizer.solve(controlled_data, total_budget=100000)
        alloc = lp_optimizer.get_customer_allocations(result)
        merged = alloc.merge(controlled_data, on="customer_id")

        violations = merged[merged["allocated_budget"] > merged["cost_per_action"] * 1.001]
        assert len(violations) == 0, f"Cap violations: {violations}"


# ---------------------------------------------------------------------------
# Optimality Properties
# ---------------------------------------------------------------------------


class TestOptimalityProperties:
    """Test that LP solutions satisfy optimality properties."""

    def test_lp_dominates_uniform_allocation(self, budget_optimizer, sample_data):
        """LP allocation should achieve higher ROI than uniform allocation."""
        budget = 5_000_000
        lp_result = budget_optimizer.optimize(sample_data, total_budget=budget)
        lp_roi = budget_optimizer.compute_roi(lp_result, sample_data)

        # Uniform allocation
        n = len(sample_data)
        uniform = pd.DataFrame(
            {
                "customer_id": sample_data["customer_id"],
                "allocated_budget": np.full(n, budget / n),
            }
        )
        uniform_roi = budget_optimizer.compute_roi(uniform, sample_data)

        assert lp_roi >= uniform_roi * 0.99

    def test_lp_dominates_random_allocation(self, budget_optimizer, sample_data):
        """LP allocation should achieve higher ROI than random allocation."""
        budget = 5_000_000
        lp_result = budget_optimizer.optimize(sample_data, total_budget=budget)
        lp_roi = budget_optimizer.compute_roi(lp_result, sample_data)

        # Random allocation
        rng = np.random.RandomState(123)
        n = len(sample_data)
        random_weights = rng.dirichlet(np.ones(n))
        random_alloc = pd.DataFrame(
            {
                "customer_id": sample_data["customer_id"],
                "allocated_budget": random_weights * budget,
            }
        )
        random_roi = budget_optimizer.compute_roi(random_alloc, sample_data)

        assert lp_roi >= random_roi * 0.95

    def test_adding_budget_never_decreases_objective(self, lp_optimizer, sample_data):
        """More budget should yield equal or higher objective value."""
        budgets = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
        objectives = []
        for b in budgets:
            result = lp_optimizer.solve(sample_data, total_budget=b)
            objectives.append(result.objective_value)

        for i in range(1, len(objectives)):
            assert objectives[i] >= objectives[i - 1] * 0.99, (
                f"Objective decreased: {objectives[i]:.0f} < {objectives[i-1]:.0f} "
                f"at budget {budgets[i]}"
            )

    def test_diminishing_marginal_returns(self, lp_optimizer, sample_data):
        """Marginal value of additional budget should be non-increasing."""
        budgets = [1_000_000, 2_000_000, 4_000_000, 8_000_000]
        objectives = []
        for b in budgets:
            result = lp_optimizer.solve(sample_data, total_budget=b)
            objectives.append(result.objective_value)

        marginals = []
        for i in range(1, len(objectives)):
            marginal = (objectives[i] - objectives[i - 1]) / (budgets[i] - budgets[i - 1])
            marginals.append(marginal)

        # Marginal returns should be non-increasing (allow small tolerance)
        for i in range(1, len(marginals)):
            assert marginals[i] <= marginals[i - 1] * 1.05, (
                f"Marginal return increased: {marginals[i]:.6f} > {marginals[i-1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Allocation Monotonicity
# ---------------------------------------------------------------------------


class TestAllocationMonotonicity:
    """Test monotonicity properties of budget allocations."""

    def test_total_allocated_monotone_in_budget(self, budget_optimizer, sample_data):
        """Total allocated should be non-decreasing with budget."""
        budgets = [1_000_000, 5_000_000, 10_000_000, 50_000_000]
        prev_total = 0
        for b in budgets:
            result = budget_optimizer.optimize(sample_data, total_budget=b)
            total = result["allocated_budget"].sum()
            assert total >= prev_total * 0.99
            prev_total = total

    def test_treated_customers_monotone_in_budget(self, budget_optimizer, sample_data):
        """Number of treated customers should be non-decreasing with budget."""
        budgets = [1_000_000, 5_000_000, 10_000_000, 50_000_000]
        prev_treated = 0
        for b in budgets:
            result = budget_optimizer.optimize(sample_data, total_budget=b)
            treated = (result["allocated_budget"] > 0).sum()
            assert treated >= prev_treated * 0.95
            prev_treated = treated

    def test_lp_allocation_monotone_in_budget(self, lp_optimizer, sample_data):
        """LP total allocated should be monotone in budget."""
        budgets = [500_000, 2_000_000, 5_000_000]
        prev = 0
        for b in budgets:
            result = lp_optimizer.solve(sample_data, total_budget=b)
            assert result.total_allocated >= prev * 0.99
            prev = result.total_allocated


# ---------------------------------------------------------------------------
# Constraint Satisfaction
# ---------------------------------------------------------------------------


class TestConstraintSatisfaction:
    """Rigorous constraint satisfaction tests."""

    def test_budget_constraint_exact(self, budget_optimizer, sample_data):
        """Total allocation must not exceed budget (strict)."""
        for budget in [100_000, 1_000_000, 10_000_000, 50_000_000]:
            result = budget_optimizer.optimize(sample_data, total_budget=budget)
            total = result["allocated_budget"].sum()
            assert total <= budget * 1.001, (
                f"Budget {budget}: allocated {total:.0f} exceeds budget"
            )

    def test_non_negativity_constraint(self, budget_optimizer, sample_data):
        """All allocations must be >= 0."""
        result = budget_optimizer.optimize(sample_data, total_budget=5_000_000)
        assert (result["allocated_budget"] >= -1e-6).all()

    def test_lp_budget_constraint_strict(self, lp_optimizer, sample_data):
        """LP total allocation must not exceed budget."""
        for budget in [500_000, 2_000_000, 10_000_000]:
            result = lp_optimizer.solve(sample_data, total_budget=budget)
            assert result.total_allocated <= budget * 1.001

    def test_lp_non_negativity(self, lp_optimizer, sample_data):
        """LP allocations must be non-negative."""
        result = lp_optimizer.solve(sample_data, total_budget=5_000_000)
        assert (result.allocations["allocated_budget"] >= -1e-8).all()

    def test_multi_channel_total_constraint(self, config, sample_data):
        """Multi-channel LP must respect total budget."""
        from src.optimization.budget_optimizer import LPBudgetOptimizer, ChannelConstraint

        channels = [
            ChannelConstraint(name="email", cost_per_action=5000),
            ChannelConstraint(name="sms", cost_per_action=2000),
        ]
        opt = LPBudgetOptimizer(config, channels=channels)
        budget = 3_000_000
        result = opt.solve(sample_data, total_budget=budget)
        assert result.total_allocated <= budget * 1.001

    def test_multi_channel_per_channel_min(self, config, sample_data):
        """Multi-channel LP must respect per-channel min budget."""
        from src.optimization.budget_optimizer import LPBudgetOptimizer, ChannelConstraint

        channels = [
            ChannelConstraint(
                name="required", cost_per_action=5000, min_budget=500_000
            ),
        ]
        opt = LPBudgetOptimizer(config, channels=channels)
        result = opt.solve(sample_data, total_budget=5_000_000)
        if result.status == "optimal":
            summary = opt.get_channel_summary(result)
            assert summary.get("required", 0) >= 500_000 * 0.999

    def test_multi_channel_per_channel_max(self, config, sample_data):
        """Multi-channel LP must respect per-channel max budget."""
        from src.optimization.budget_optimizer import LPBudgetOptimizer, ChannelConstraint

        channels = [
            ChannelConstraint(
                name="capped", cost_per_action=5000, max_budget=1_000_000
            ),
        ]
        opt = LPBudgetOptimizer(config, channels=channels)
        result = opt.solve(sample_data, total_budget=10_000_000)
        if result.status == "optimal":
            summary = opt.get_channel_summary(result)
            assert summary.get("capped", 0) <= 1_000_000 * 1.001


# ---------------------------------------------------------------------------
# ROI Computation Correctness
# ---------------------------------------------------------------------------


class TestROIComputationCorrectness:
    """Test ROI computation logic."""

    def test_roi_non_negative(self, budget_optimizer, sample_data):
        """ROI should be non-negative for valid data."""
        result = budget_optimizer.optimize(sample_data, total_budget=5_000_000)
        roi = budget_optimizer.compute_roi(result, sample_data)
        assert roi >= 0

    def test_roi_zero_for_zero_budget(self, budget_optimizer, sample_data):
        """Zero budget should yield zero ROI."""
        result = budget_optimizer.optimize(sample_data, total_budget=0)
        roi = budget_optimizer.compute_roi(result, sample_data)
        assert roi == 0.0

    def test_roi_zero_for_sleeping_dogs_only(self, budget_optimizer):
        """All sleeping-dog customers should yield zero ROI."""
        data = pd.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "churn_prob": [0.5, 0.6, 0.7],
                "clv": [100000, 200000, 300000],
                "uplift_score": [-0.2, -0.3, -0.1],
                "cost_per_action": [10000, 10000, 10000],
                "expected_retention_lift": [0.0, 0.0, 0.0],
            }
        )
        result = budget_optimizer.optimize(data, total_budget=50000)
        roi = budget_optimizer.compute_roi(result, data)
        assert roi == 0.0

    def test_roi_increases_with_budget(self, budget_optimizer, sample_data):
        """ROI (total retained value) should be non-decreasing with budget."""
        prev_roi = 0
        for budget in [1_000_000, 5_000_000, 10_000_000, 50_000_000]:
            result = budget_optimizer.optimize(sample_data, total_budget=budget)
            roi = budget_optimizer.compute_roi(result, sample_data)
            assert roi >= prev_roi * 0.95
            prev_roi = roi

    def test_lp_expected_value_positive(self, lp_optimizer, sample_data):
        """LP expected value should be positive for valid data."""
        result = lp_optimizer.solve(sample_data, total_budget=5_000_000)
        ev = lp_optimizer.compute_expected_value(result, sample_data)
        assert ev > 0

    def test_lp_expected_value_zero_for_zero_budget(self, lp_optimizer, sample_data):
        """LP expected value should be zero for zero budget."""
        result = lp_optimizer.solve(sample_data, total_budget=0)
        ev = lp_optimizer.compute_expected_value(result, sample_data)
        assert ev == 0.0


# ---------------------------------------------------------------------------
# Priority-Based Allocation Logic
# ---------------------------------------------------------------------------


class TestPriorityBasedAllocation:
    """Test that allocation follows priority ordering."""

    def test_highest_priority_gets_budget_first(self, budget_optimizer, controlled_data):
        """With tight budget, highest priority customer gets budget first."""
        # Budget for only 1 customer
        result = budget_optimizer.optimize(controlled_data, total_budget=10000)
        merged = result.merge(controlled_data, on="customer_id")

        # Priority = max(uplift, 0) * clv * churn_prob
        merged["priority"] = (
            np.maximum(merged["uplift_score"], 0)
            * merged["clv"]
            * merged["churn_prob"]
        )
        top_customer = merged.loc[merged["priority"].idxmax()]
        assert top_customer["allocated_budget"] > 0

    def test_priority_ordering_preserved(self, budget_optimizer, controlled_data):
        """Customers with higher priority should get >= budget than lower."""
        result = budget_optimizer.optimize(controlled_data, total_budget=30000)
        merged = result.merge(controlled_data, on="customer_id")

        merged["priority"] = (
            np.maximum(merged["uplift_score"], 0)
            * merged["clv"]
            * merged["churn_prob"]
        )
        merged = merged.sort_values("priority", ascending=False)

        # Top customer should get most budget
        budgets = merged["allocated_budget"].values
        assert budgets[0] >= budgets[1]

    def test_zero_clv_gets_zero(self, budget_optimizer):
        """Customer with zero CLV should receive zero allocation."""
        data = pd.DataFrame(
            {
                "customer_id": ["zero_clv", "has_clv"],
                "churn_prob": [0.8, 0.8],
                "clv": [0.0, 100000],
                "uplift_score": [0.3, 0.3],
                "cost_per_action": [10000, 10000],
                "expected_retention_lift": [0.2, 0.2],
            }
        )
        result = budget_optimizer.optimize(data, total_budget=50000)
        merged = result.merge(data[["customer_id", "clv"]], on="customer_id")
        zero_alloc = merged[merged["clv"] == 0]["allocated_budget"].values[0]
        assert zero_alloc == 0.0

    def test_zero_churn_gets_zero(self, budget_optimizer):
        """Customer with zero churn probability should receive zero allocation."""
        data = pd.DataFrame(
            {
                "customer_id": ["no_churn", "has_churn"],
                "churn_prob": [0.0, 0.8],
                "clv": [100000, 100000],
                "uplift_score": [0.3, 0.3],
                "cost_per_action": [10000, 10000],
                "expected_retention_lift": [0.2, 0.2],
            }
        )
        result = budget_optimizer.optimize(data, total_budget=50000)
        merged = result.merge(data[["customer_id", "churn_prob"]], on="customer_id")
        no_churn = merged[merged["churn_prob"] == 0]["allocated_budget"].values[0]
        assert no_churn == 0.0


# ---------------------------------------------------------------------------
# CostConfig + LP Integration
# ---------------------------------------------------------------------------


class TestCostConfigLPIntegration:
    """Test integration between CostConfig and LP optimizer."""

    def test_cost_config_channels_produce_valid_lp(self, config, sample_data):
        """CostConfig channels should feed into LP optimizer correctly."""
        from src.optimization.budget_optimizer import CostConfig, LPBudgetOptimizer

        cc = CostConfig.from_config(config)
        channels = cc.get_channel_constraints()
        opt = LPBudgetOptimizer(config, channels=channels)
        result = opt.solve(sample_data)
        assert result.status == "optimal"

    def test_npv_adjustment_reduces_value(self, config, sample_data):
        """NPV adjustment with positive discount rate should reduce value."""
        from src.optimization.budget_optimizer import CostConfig, run_optimization

        result = run_optimization(data=sample_data, config=config)
        # NPV adjusted should be less than raw expected value
        if result["expected_value"] > 0:
            assert result["npv_adjusted_value"] <= result["expected_value"]

    def test_higher_discount_lower_npv(self, sample_data, config):
        """Higher discount rate should produce lower NPV-adjusted value."""
        from src.optimization.budget_optimizer import run_optimization

        result_low = run_optimization(data=sample_data, config=config)

        config_high = dict(config)
        config_high["optimization"] = dict(config.get("optimization", {}))
        config_high["optimization"]["discount_rate"] = 0.50
        result_high = run_optimization(data=sample_data, config=config_high)

        assert result_high["npv_adjusted_value"] < result_low["npv_adjusted_value"]


# ---------------------------------------------------------------------------
# What-If Scenario Logic
# ---------------------------------------------------------------------------


class TestWhatIfScenarioLogic:
    """Test what-if scenario analysis logic."""

    def test_budget_sweep_returns_ordered_results(self, config, sample_data):
        """Budget sweep should return results for all budget levels."""
        from src.optimization.budget_optimizer import run_whatif

        levels = [5_000_000, 10_000_000, 25_000_000, 50_000_000]
        result = run_whatif(data=sample_data, config=config, budget_levels=levels)
        sweep = result["budget_sweep"]
        assert len(sweep) == len(levels)

    def test_scenario_comparison_has_all_scenarios(self, config, sample_data):
        """Scenario comparison should include all defined scenarios."""
        from src.optimization.budget_optimizer import run_whatif, WhatIfScenario

        scenarios = [
            WhatIfScenario(name="baseline"),
            WhatIfScenario(name="double", total_budget=100_000_000),
        ]
        result = run_whatif(data=sample_data, config=config, scenarios=scenarios)
        comp = result["scenario_comparison"]
        names = set(comp["scenario_name"])
        assert names == {"baseline", "double"}

    def test_higher_budget_scenario_higher_value(self, config, sample_data):
        """Scenario with higher budget should retain more value."""
        from src.optimization.budget_optimizer import run_whatif, WhatIfScenario

        scenarios = [
            WhatIfScenario(name="low", total_budget=10_000_000),
            WhatIfScenario(name="high", total_budget=50_000_000),
        ]
        result = run_whatif(data=sample_data, config=config, scenarios=scenarios)
        comp = result["scenario_comparison"]
        low_val = comp[comp["scenario_name"] == "low"]["retained_value"].values[0]
        high_val = comp[comp["scenario_name"] == "high"]["retained_value"].values[0]
        assert high_val >= low_val * 0.95


# ---------------------------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------------------------


class TestValidationLogic:
    """Test input validation for budget optimization."""

    def test_missing_required_columns_raises(self):
        """Missing columns should raise BudgetValidationError."""
        from src.optimization.budget_optimizer import (
            validate_dataframe,
            BudgetValidationError,
        )

        df = pd.DataFrame({"customer_id": ["A"], "clv": [100]})
        with pytest.raises(BudgetValidationError, match="Missing required"):
            validate_dataframe(df)

    def test_nan_in_numeric_raises(self):
        """NaN values in numeric columns should raise."""
        from src.optimization.budget_optimizer import (
            validate_dataframe,
            BudgetValidationError,
        )

        df = pd.DataFrame(
            {
                "customer_id": ["A"],
                "clv": [np.nan],
                "uplift_score": [0.1],
                "churn_prob": [0.5],
                "cost_per_action": [1000],
            }
        )
        with pytest.raises(BudgetValidationError, match="NaN"):
            validate_dataframe(df)

    def test_negative_budget_raises(self):
        """Negative budget should raise."""
        from src.optimization.budget_optimizer import (
            validate_budget,
            BudgetValidationError,
        )

        with pytest.raises(BudgetValidationError):
            validate_budget(-1)

    def test_infinite_budget_raises(self):
        """Infinite budget should raise."""
        from src.optimization.budget_optimizer import (
            validate_budget,
            BudgetValidationError,
        )

        with pytest.raises(BudgetValidationError):
            validate_budget(float("inf"))

    def test_none_data_raises(self):
        """None data should raise."""
        from src.optimization.budget_optimizer import (
            validate_dataframe,
            BudgetValidationError,
        )

        with pytest.raises(BudgetValidationError, match="cannot be None"):
            validate_dataframe(None)


# ---------------------------------------------------------------------------
# Edge Cases: Stress Tests
# ---------------------------------------------------------------------------


class TestOptimizationEdgeCases:
    """Stress-test edge cases for optimization."""

    def test_single_customer_gets_budget(self, budget_optimizer):
        """Single positive-uplift customer should get budget."""
        data = pd.DataFrame(
            {
                "customer_id": ["solo"],
                "churn_prob": [0.8],
                "clv": [100000],
                "uplift_score": [0.3],
                "cost_per_action": [10000],
                "expected_retention_lift": [0.2],
            }
        )
        result = budget_optimizer.optimize(data, total_budget=50000)
        assert result["allocated_budget"].values[0] > 0

    def test_large_customer_count(self, budget_optimizer):
        """Optimizer should handle many customers efficiently."""
        np.random.seed(42)
        n = 10000
        data = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(n)],
                "churn_prob": np.random.beta(2, 5, n),
                "clv": np.random.lognormal(10, 1, n),
                "uplift_score": np.random.randn(n) * 0.1,
                "cost_per_action": np.random.choice([5000, 10000], size=n),
                "expected_retention_lift": np.random.uniform(0, 0.3, n),
            }
        )
        result = budget_optimizer.optimize(data, total_budget=50_000_000)
        assert len(result) == n
        assert result["allocated_budget"].sum() <= 50_000_000 * 1.001

    def test_budget_larger_than_total_cost(self, budget_optimizer, controlled_data):
        """When budget exceeds total possible cost, allocation <= budget."""
        total_cost = controlled_data["cost_per_action"].sum()
        large_budget = total_cost * 10
        result = budget_optimizer.optimize(
            controlled_data, total_budget=large_budget
        )
        # Total allocation must not exceed the provided budget
        assert result["allocated_budget"].sum() <= large_budget * 1.001
        # Non-negative allocations
        assert (result["allocated_budget"] >= 0).all()

    def test_all_identical_customers(self, budget_optimizer):
        """Identical customers should receive approximately equal allocation."""
        n = 10
        data = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(n)],
                "churn_prob": [0.5] * n,
                "clv": [100000] * n,
                "uplift_score": [0.2] * n,
                "cost_per_action": [10000] * n,
                "expected_retention_lift": [0.1] * n,
            }
        )
        # Budget enough for all
        result = budget_optimizer.optimize(data, total_budget=100000)
        allocations = result["allocated_budget"].values
        # All should be equal (or at least very close)
        assert allocations.std() / allocations.mean() < 0.1

    def test_lp_infeasible_min_constraints(self, config, sample_data):
        """Infeasible constraints should be detected."""
        from src.optimization.budget_optimizer import LPBudgetOptimizer, ChannelConstraint

        channels = [
            ChannelConstraint(name="ch1", min_budget=6_000_000),
            ChannelConstraint(name="ch2", min_budget=6_000_000),
        ]
        opt = LPBudgetOptimizer(config, channels=channels)
        result = opt.solve(sample_data, total_budget=1_000_000)
        assert result.status in ("infeasible", "failed", "empty")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestOptimizationReproducibility:
    """Test that optimization is deterministic."""

    def test_budget_optimizer_deterministic(self, config, sample_data):
        """Same input should produce identical output."""
        from src.models.budget_optimizer import BudgetOptimizer

        opt1 = BudgetOptimizer(config)
        opt2 = BudgetOptimizer(config)
        r1 = opt1.optimize(sample_data, total_budget=5_000_000)
        r2 = opt2.optimize(sample_data, total_budget=5_000_000)
        np.testing.assert_array_almost_equal(
            r1["allocated_budget"].values, r2["allocated_budget"].values, decimal=2
        )

    def test_lp_optimizer_deterministic(self, config, sample_data):
        """LP optimizer should produce identical results for same input."""
        from src.optimization.budget_optimizer import LPBudgetOptimizer

        opt1 = LPBudgetOptimizer(config)
        opt2 = LPBudgetOptimizer(config)
        r1 = opt1.solve(sample_data, total_budget=5_000_000)
        r2 = opt2.solve(sample_data, total_budget=5_000_000)
        assert abs(r1.objective_value - r2.objective_value) < 1.0

    def test_run_optimization_deterministic(self, config, sample_data):
        """run_optimization should produce identical results."""
        from src.optimization.budget_optimizer import run_optimization

        r1 = run_optimization(data=sample_data, config=config)
        r2 = run_optimization(data=sample_data, config=config)
        assert abs(r1["expected_value"] - r2["expected_value"]) < 1.0


# ---------------------------------------------------------------------------
# Persistence Round-Trip
# ---------------------------------------------------------------------------


class TestOptimizationPersistence:
    """Test that optimizers can be saved and loaded correctly."""

    def test_budget_optimizer_save_load(self, budget_optimizer, sample_data, tmp_path):
        """BudgetOptimizer should produce same results after save/load."""
        from src.models.budget_optimizer import BudgetOptimizer

        budget = 5_000_000
        r_orig = budget_optimizer.optimize(sample_data, total_budget=budget)

        path = tmp_path / "bo_state"
        budget_optimizer.save(str(path))
        loaded = BudgetOptimizer.load(str(path))
        r_loaded = loaded.optimize(sample_data, total_budget=budget)

        np.testing.assert_array_almost_equal(
            r_orig["allocated_budget"].values,
            r_loaded["allocated_budget"].values,
            decimal=2,
        )

    def test_lp_optimizer_save_load(self, lp_optimizer, sample_data, tmp_path):
        """LPBudgetOptimizer should save and load correctly."""
        from src.optimization.budget_optimizer import LPBudgetOptimizer

        lp_optimizer.solve(sample_data, total_budget=5_000_000)
        path = tmp_path / "lp_state"
        lp_optimizer.save(str(path))

        loaded = LPBudgetOptimizer.load(str(path))
        assert loaded.total_budget == lp_optimizer.total_budget
        assert loaded.seed == lp_optimizer.seed
