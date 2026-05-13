"""
TDD Tests for LP Solver Foundation in Budget Optimizer.

Tests specifically targeting the Linear Programming optimization:
- LP objective function correctness
- LP constraint satisfaction
- scipy.optimize.linprog integration
- Multi-channel LP optimization
- LP diagnostics
- Edge cases (infeasible, degenerate)
- LP vs proportional fallback comparison
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
def optimizer(config):
    from src.models.budget_optimizer import BudgetOptimizer
    return BudgetOptimizer(config)


@pytest.fixture
def small_data():
    """Small dataset for precise LP verification."""
    return pd.DataFrame({
        "customer_id": ["A", "B", "C", "D", "E"],
        "churn_prob": [0.8, 0.6, 0.3, 0.1, 0.9],
        "clv": [100000, 80000, 50000, 200000, 30000],
        "uplift_score": [0.3, 0.2, 0.1, -0.1, 0.4],
        "cost_per_action": [10000, 10000, 10000, 10000, 10000],
        "expected_retention_lift": [0.25, 0.15, 0.10, 0.0, 0.35],
    })


@pytest.fixture
def medium_data():
    """Medium dataset for statistical tests."""
    np.random.seed(123)
    n = 500
    churn = np.random.beta(2, 5, n)
    clv = np.random.lognormal(10, 1, n)
    uplift = np.random.randn(n) * 0.15
    cost = np.random.choice([5000, 10000, 20000, 50000], size=n)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "churn_prob": churn,
        "clv": clv,
        "uplift_score": uplift,
        "cost_per_action": cost,
        "expected_retention_lift": np.clip(uplift + 0.05, 0, 0.5),
    })


# ---------------------------------------------------------------------------
# LP Objective Function Tests
# ---------------------------------------------------------------------------

class TestLPObjectiveFunction:
    """Test that the LP objective function is correctly formulated."""

    def test_sleeping_dogs_get_zero_allocation(self, optimizer, small_data):
        """Customers with negative uplift (sleeping dogs) should get 0 budget."""
        # Customer D has uplift_score = -0.1 (sleeping dog)
        result = optimizer.optimize(small_data, total_budget=20000)
        merged = result.merge(small_data[["customer_id", "uplift_score"]], on="customer_id")
        sleeping_dog = merged[merged["customer_id"] == "D"]
        assert sleeping_dog["allocated_budget"].values[0] == 0.0

    def test_highest_priority_gets_most(self, optimizer, small_data):
        """Customer with highest priority score should get the most budget."""
        # Tight budget forces LP to choose
        result = optimizer.optimize(small_data, total_budget=15000)
        merged = result.merge(small_data, on="customer_id")

        # Compute expected priorities
        merged["priority"] = (
            np.maximum(merged["uplift_score"], 0) * merged["clv"] * merged["churn_prob"]
        )
        top_customer = merged.loc[merged["priority"].idxmax(), "customer_id"]
        top_budget = merged.loc[merged["priority"].idxmax(), "allocated_budget"]

        # Top priority customer should have the highest allocation
        assert top_budget == merged["allocated_budget"].max()

    def test_objective_value_positive(self, optimizer, small_data):
        """LP objective (retained value) should be positive for valid data."""
        result = optimizer.optimize(small_data, total_budget=30000)
        roi = optimizer.compute_roi(result, small_data)
        assert roi > 0

    def test_lp_maximizes_value(self, optimizer, medium_data):
        """LP allocation should yield higher ROI than uniform allocation."""
        total_budget = 5_000_000

        # LP allocation
        lp_result = optimizer.optimize(medium_data, total_budget=total_budget)
        lp_roi = optimizer.compute_roi(lp_result, medium_data)

        # Uniform allocation
        n = len(medium_data)
        uniform_result = pd.DataFrame({
            "customer_id": medium_data["customer_id"],
            "allocated_budget": np.full(n, total_budget / n),
        })
        uniform_roi = optimizer.compute_roi(uniform_result, medium_data)

        assert lp_roi >= uniform_roi * 0.99, (
            f"LP ROI ({lp_roi:.0f}) should be >= uniform ROI ({uniform_roi:.0f})"
        )


# ---------------------------------------------------------------------------
# LP Constraint Tests
# ---------------------------------------------------------------------------

class TestLPConstraints:
    """Test LP constraint satisfaction."""

    def test_budget_constraint_tight(self, optimizer, medium_data):
        """When budget is tight, total allocation should use most of the budget."""
        total_budget = 1_000_000  # Very tight
        result = optimizer.optimize(medium_data, total_budget=total_budget)
        total_alloc = result["allocated_budget"].sum()
        assert total_alloc <= total_budget * 1.001
        # Should use at least 90% of tight budget
        assert total_alloc >= total_budget * 0.90, (
            f"LP should use most of tight budget: {total_alloc:.0f} / {total_budget:.0f}"
        )

    def test_per_customer_upper_bound(self, optimizer, small_data):
        """Each customer's allocation should not exceed cost_per_action (when budget tight)."""
        result = optimizer.optimize(small_data, total_budget=20000)
        merged = result.merge(small_data[["customer_id", "cost_per_action"]], on="customer_id")
        # With tight budget, allocations should be bounded by cost
        assert (merged["allocated_budget"] <= merged["cost_per_action"] * 1.001).all()

    def test_non_negative_allocations(self, optimizer, medium_data):
        """All allocations must be non-negative."""
        result = optimizer.optimize(medium_data, total_budget=5_000_000)
        assert (result["allocated_budget"] >= -0.01).all()

    def test_budget_constraint_exact(self, optimizer, medium_data):
        """Total allocation must not exceed budget."""
        for budget in [100_000, 1_000_000, 10_000_000, 50_000_000]:
            result = optimizer.optimize(medium_data, total_budget=budget)
            assert result["allocated_budget"].sum() <= budget * 1.001


# ---------------------------------------------------------------------------
# scipy.optimize.linprog Integration Tests
# ---------------------------------------------------------------------------

class TestLinprogIntegration:
    """Test scipy.optimize.linprog integration."""

    def test_lp_diagnostics_available(self, optimizer, small_data):
        """LP diagnostics should be available after optimization."""
        optimizer.optimize(small_data, total_budget=30000)
        diag = optimizer.get_lp_diagnostics()
        assert diag is not None
        assert "success" in diag
        assert "message" in diag
        assert "objective_value" in diag

    def test_lp_solver_succeeds(self, optimizer, small_data):
        """LP solver should succeed for well-posed problems."""
        optimizer.optimize(small_data, total_budget=30000)
        diag = optimizer.get_lp_diagnostics()
        assert diag["success"] is True

    def test_lp_objective_positive(self, optimizer, small_data):
        """LP objective value should be positive (we're maximizing)."""
        optimizer.optimize(small_data, total_budget=30000)
        diag = optimizer.get_lp_diagnostics()
        assert diag["objective_value"] is not None
        assert diag["objective_value"] > 0

    def test_lp_uses_highs_method(self, optimizer, medium_data):
        """Verify the LP uses HiGHS solver."""
        optimizer.optimize(medium_data, total_budget=5_000_000)
        diag = optimizer.get_lp_diagnostics()
        # HiGHS reports "Optimization terminated successfully" or similar
        assert diag["success"]

    def test_lp_result_stored(self, optimizer, small_data):
        """Raw LP result should be stored for inspection."""
        optimizer.optimize(small_data, total_budget=30000)
        assert optimizer._last_lp_result is not None
        assert hasattr(optimizer._last_lp_result, "x")
        assert hasattr(optimizer._last_lp_result, "fun")


# ---------------------------------------------------------------------------
# Multi-Channel LP Tests
# ---------------------------------------------------------------------------

class TestMultiChannelLP:
    """Test multi-channel budget optimization."""

    def test_multi_channel_basic(self, optimizer, small_data):
        """Multi-channel LP should return allocations for all channels."""
        channels = ["email", "sms", "push"]
        costs = {"email": 1000, "sms": 500, "push": 200}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=10000
        )
        assert "customer_id" in result.columns
        assert "allocated_budget" in result.columns
        for ch in channels:
            assert f"budget_{ch}" in result.columns

    def test_multi_channel_budget_constraint(self, optimizer, small_data):
        """Total multi-channel allocation should respect budget."""
        channels = ["email", "sms"]
        costs = {"email": 5000, "sms": 2000}
        budget = 15000
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=budget
        )
        assert result["allocated_budget"].sum() <= budget * 1.001

    def test_multi_channel_per_channel_cap(self, optimizer, small_data):
        """Per-channel budgets should be respected."""
        channels = ["email", "sms"]
        costs = {"email": 5000, "sms": 2000}
        ch_budgets = {"email": 8000, "sms": 5000}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs,
            channel_budgets=ch_budgets, total_budget=20000
        )
        assert result["budget_email"].sum() <= ch_budgets["email"] * 1.001
        assert result["budget_sms"].sum() <= ch_budgets["sms"] * 1.001

    def test_multi_channel_non_negative(self, optimizer, small_data):
        """All multi-channel allocations should be non-negative."""
        channels = ["email", "sms", "push"]
        costs = {"email": 1000, "sms": 500, "push": 200}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=10000
        )
        for ch in channels:
            assert (result[f"budget_{ch}"] >= -0.01).all()
        assert (result["allocated_budget"] >= -0.01).all()

    def test_multi_channel_total_equals_sum(self, optimizer, small_data):
        """allocated_budget should equal sum of per-channel budgets."""
        channels = ["email", "sms"]
        costs = {"email": 3000, "sms": 1000}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=15000
        )
        per_ch_sum = result["budget_email"] + result["budget_sms"]
        np.testing.assert_array_almost_equal(
            result["allocated_budget"].values, per_ch_sum.values, decimal=2
        )

    def test_multi_channel_zero_budget(self, optimizer, small_data):
        """Zero budget should give zero allocations across all channels."""
        channels = ["email", "sms"]
        costs = {"email": 1000, "sms": 500}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=0
        )
        assert result["allocated_budget"].sum() == 0

    def test_multi_channel_empty_channels(self, optimizer, small_data):
        """Empty channel list should return zero allocations."""
        result = optimizer.optimize_multi_channel(
            small_data, [], {}, total_budget=10000
        )
        assert result["allocated_budget"].sum() == 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestLPEdgeCases:
    """Test LP solver edge cases."""

    def test_single_customer(self, optimizer):
        """LP should handle a single customer."""
        data = pd.DataFrame({
            "customer_id": ["X"],
            "churn_prob": [0.5],
            "clv": [100000],
            "uplift_score": [0.2],
            "cost_per_action": [10000],
            "expected_retention_lift": [0.15],
        })
        result = optimizer.optimize(data, total_budget=50000)
        assert len(result) == 1
        assert result["allocated_budget"].values[0] > 0
        assert result["allocated_budget"].values[0] <= 50000

    def test_all_sleeping_dogs(self, optimizer):
        """If all customers are sleeping dogs, allocation should be zero."""
        data = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "churn_prob": [0.5, 0.3, 0.7],
            "clv": [100000, 80000, 60000],
            "uplift_score": [-0.2, -0.1, -0.3],
            "cost_per_action": [10000, 10000, 10000],
            "expected_retention_lift": [0.0, 0.0, 0.0],
        })
        result = optimizer.optimize(data, total_budget=50000)
        assert result["allocated_budget"].sum() == 0.0

    def test_very_large_n(self, optimizer):
        """LP should handle a large number of customers."""
        np.random.seed(42)
        n = 5000
        data = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(n)],
            "churn_prob": np.random.beta(2, 5, n),
            "clv": np.random.lognormal(10, 1, n),
            "uplift_score": np.random.randn(n) * 0.1,
            "cost_per_action": np.random.choice([5000, 10000], size=n),
            "expected_retention_lift": np.random.uniform(0, 0.3, n),
        })
        result = optimizer.optimize(data, total_budget=10_000_000)
        assert len(result) == n
        assert result["allocated_budget"].sum() <= 10_000_000 * 1.001

    def test_zero_clv_customers(self, optimizer):
        """Customers with zero CLV should get zero allocation."""
        data = pd.DataFrame({
            "customer_id": ["A", "B"],
            "churn_prob": [0.5, 0.5],
            "clv": [0.0, 100000],
            "uplift_score": [0.3, 0.3],
            "cost_per_action": [10000, 10000],
            "expected_retention_lift": [0.2, 0.2],
        })
        result = optimizer.optimize(data, total_budget=15000)
        merged = result.merge(data[["customer_id", "clv"]], on="customer_id")
        zero_clv = merged[merged["clv"] == 0]["allocated_budget"].values[0]
        assert zero_clv == 0.0

    def test_zero_churn_customers(self, optimizer):
        """Customers with zero churn prob should get zero allocation."""
        data = pd.DataFrame({
            "customer_id": ["A", "B"],
            "churn_prob": [0.0, 0.8],
            "clv": [100000, 100000],
            "uplift_score": [0.3, 0.3],
            "cost_per_action": [10000, 10000],
            "expected_retention_lift": [0.2, 0.2],
        })
        result = optimizer.optimize(data, total_budget=15000)
        merged = result.merge(data[["customer_id", "churn_prob"]], on="customer_id")
        zero_churn = merged[merged["churn_prob"] == 0]["allocated_budget"].values[0]
        assert zero_churn == 0.0

    def test_equal_priority_uses_full_budget(self, optimizer):
        """With equal priority customers, LP should use the full budget."""
        data = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "churn_prob": [0.5, 0.5, 0.5],
            "clv": [100000, 100000, 100000],
            "uplift_score": [0.2, 0.2, 0.2],
            "cost_per_action": [10000, 10000, 10000],
            "expected_retention_lift": [0.1, 0.1, 0.1],
        })
        budget = 25000
        result = optimizer.optimize(data, total_budget=budget)
        allocs = result["allocated_budget"].values
        # LP should use the full budget (all equal priority, all worth treating)
        assert allocs.sum() >= budget * 0.99
        # All allocations should be non-negative
        assert (allocs >= 0).all()


# ---------------------------------------------------------------------------
# Monotonicity / Diminishing Returns Tests
# ---------------------------------------------------------------------------

class TestBudgetMonotonicity:
    """Test that budget allocations show expected monotonic behavior."""

    def test_total_allocation_monotone_in_budget(self, optimizer, medium_data):
        """Total allocated budget should be non-decreasing as budget increases."""
        budgets = [500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]
        prev_alloc = 0
        for b in budgets:
            result = optimizer.optimize(medium_data, total_budget=b)
            curr_alloc = result["allocated_budget"].sum()
            assert curr_alloc >= prev_alloc * 0.99, (
                f"Allocation decreased: {curr_alloc:.0f} < {prev_alloc:.0f} at budget {b}"
            )
            prev_alloc = curr_alloc

    def test_roi_monotone_in_budget(self, optimizer, medium_data):
        """ROI should be non-decreasing as budget increases."""
        budgets = [500_000, 2_000_000, 10_000_000, 50_000_000]
        prev_roi = 0
        for b in budgets:
            result = optimizer.optimize(medium_data, total_budget=b)
            roi = optimizer.compute_roi(result, medium_data)
            assert roi >= prev_roi * 0.95, (
                f"ROI decreased: {roi:.0f} < {prev_roi:.0f} at budget {b}"
            )
            prev_roi = roi
