"""
TDD Tests for Budget Optimization Module.

Tests cover:
- Budget optimizer instantiation and interface
- Constrained optimization (total budget <= 50,000,000 KRW)
- Per-customer budget allocation using churn prob, CLV, and uplift
- ROI-maximizing allocation strategy
- Budget constraint satisfaction (non-negative, within total)
- Channel-level budget distribution
- Sensitivity analysis across budget levels
- Integration with CLV and uplift scores
- Model save/load functionality
- Reproducibility with same random seed
- Configurable budget from YAML
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
def sample_optimization_data():
    """Create synthetic customer data with churn, CLV, and uplift scores.

    Simulates the input a budget optimizer receives after upstream models
    have produced their predictions.
    """
    np.random.seed(42)
    n = 2000

    # Churn probabilities (0-1)
    churn_prob = np.random.beta(2, 5, n)

    # CLV predictions (KRW, non-negative)
    clv = np.random.lognormal(10, 1, n)

    # Uplift scores (can be negative for sleeping dogs)
    uplift = np.random.randn(n) * 0.1

    # Cost per intervention per customer (KRW)
    cost_per_action = np.random.choice([5000, 10000, 20000, 50000], size=n)

    # Expected retention probability increase from treatment
    expected_retention_lift = np.clip(uplift + 0.05, 0, 0.5)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_prob": churn_prob,
        "clv": clv,
        "uplift_score": uplift,
        "cost_per_action": cost_per_action,
        "expected_retention_lift": expected_retention_lift,
    })

    # Add segment labels for channel-level testing
    df["segment"] = np.random.choice(
        ["vip_loyal", "regular_loyal", "bargain_hunter",
         "explorer", "dormant", "new_customer"],
        size=n,
    )

    return df


@pytest.fixture
def budget_optimizer(config):
    """Create a budget optimizer instance."""
    from src.models.budget_optimizer import BudgetOptimizer
    return BudgetOptimizer(config)


# ---------------------------------------------------------------------------
# Optimizer interface tests
# ---------------------------------------------------------------------------

class TestBudgetOptimizerInterface:
    """Test budget optimizer instantiation and interface."""

    def test_instantiation(self, budget_optimizer):
        """Budget optimizer must be instantiable from config."""
        assert budget_optimizer is not None

    def test_has_optimize_method(self, budget_optimizer):
        """Budget optimizer must implement an optimize method."""
        assert hasattr(budget_optimizer, "optimize")
        assert callable(budget_optimizer.optimize)

    def test_has_allocate_method(self, budget_optimizer):
        """Budget optimizer must implement per-customer allocation."""
        assert hasattr(budget_optimizer, "allocate")
        assert callable(budget_optimizer.allocate)

    def test_has_compute_roi_method(self, budget_optimizer):
        """Budget optimizer must compute expected ROI."""
        assert hasattr(budget_optimizer, "compute_roi")
        assert callable(budget_optimizer.compute_roi)

    def test_reads_budget_from_config(self, budget_optimizer, config):
        """Budget optimizer must read total budget from config."""
        expected_budget = config["budget"]["total_krw"]
        assert budget_optimizer.total_budget == expected_budget


# ---------------------------------------------------------------------------
# Budget constraint tests
# ---------------------------------------------------------------------------

class TestBudgetConstraints:
    """Test that budget constraints are respected."""

    def test_total_allocation_within_budget(self, budget_optimizer,
                                            sample_optimization_data, config):
        """Total allocated budget must not exceed the configured total."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        assert result["allocated_budget"].sum() <= total_budget * 1.001, (
            f"Allocated {result['allocated_budget'].sum():,.0f} KRW "
            f"exceeds budget {total_budget:,.0f} KRW"
        )

    def test_allocations_non_negative(self, budget_optimizer,
                                       sample_optimization_data, config):
        """Individual allocations must be non-negative."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        assert (result["allocated_budget"] >= 0).all(), (
            "Negative budget allocations found"
        )

    def test_covers_all_customers(self, budget_optimizer,
                                   sample_optimization_data, config):
        """Result must include all input customers."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        assert len(result) == len(sample_optimization_data), (
            f"Result has {len(result)} rows, expected "
            f"{len(sample_optimization_data)}"
        )

    def test_zero_budget_allocates_nothing(self, budget_optimizer,
                                            sample_optimization_data):
        """Zero total budget must result in zero allocations."""
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=0,
        )

        assert result["allocated_budget"].sum() == 0, (
            "Non-zero allocations with zero budget"
        )

    def test_custom_budget_override(self, budget_optimizer,
                                     sample_optimization_data):
        """Must accept a custom budget that overrides config default."""
        custom_budget = 10_000_000
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=custom_budget,
        )

        assert result["allocated_budget"].sum() <= custom_budget * 1.001


# ---------------------------------------------------------------------------
# ROI-based allocation tests
# ---------------------------------------------------------------------------

class TestROIOptimization:
    """Test that allocation maximizes expected ROI."""

    def test_high_value_customers_get_more(self, budget_optimizer,
                                            sample_optimization_data, config):
        """Customers with higher expected value should receive more budget."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        merged = result.merge(
            sample_optimization_data[["customer_id", "clv"]],
            on="customer_id",
        )
        median_clv = merged["clv"].median()
        high_clv_avg = merged[merged["clv"] >= median_clv][
            "allocated_budget"
        ].mean()
        low_clv_avg = merged[merged["clv"] < median_clv][
            "allocated_budget"
        ].mean()

        assert high_clv_avg >= low_clv_avg, (
            f"High-CLV avg budget ({high_clv_avg:,.0f}) should be >= "
            f"low-CLV avg budget ({low_clv_avg:,.0f})"
        )

    def test_positive_uplift_preferred(self, budget_optimizer,
                                        sample_optimization_data, config):
        """Customers with positive uplift should be prioritized."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        merged = result.merge(
            sample_optimization_data[["customer_id", "uplift_score"]],
            on="customer_id",
        )
        pos_uplift_avg = merged[merged["uplift_score"] > 0][
            "allocated_budget"
        ].mean()
        neg_uplift_avg = merged[merged["uplift_score"] <= 0][
            "allocated_budget"
        ].mean()

        assert pos_uplift_avg >= neg_uplift_avg, (
            "Positive-uplift customers should receive more budget on average"
        )

    def test_sleeping_dogs_get_minimal_budget(self, budget_optimizer,
                                               sample_optimization_data,
                                               config):
        """Customers with strongly negative uplift should get little/no budget."""
        total_budget = config["budget"]["total_krw"]

        # Force some extreme sleeping dogs
        data = sample_optimization_data.copy()
        bottom_10 = data["uplift_score"].quantile(0.10)
        sleeping_mask = data["uplift_score"] <= bottom_10

        result = budget_optimizer.optimize(
            data=data,
            total_budget=total_budget,
        )

        merged = result.merge(
            data[["customer_id", "uplift_score"]],
            on="customer_id",
        )
        sleeping_budget = merged[
            merged["uplift_score"] <= bottom_10
        ]["allocated_budget"].mean()
        overall_avg = merged["allocated_budget"].mean()

        assert sleeping_budget <= overall_avg, (
            "Sleeping-dog customers should receive below-average budget"
        )

    def test_compute_roi_returns_numeric(self, budget_optimizer,
                                          sample_optimization_data, config):
        """compute_roi must return a numeric ROI value."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        roi = budget_optimizer.compute_roi(
            allocation=result,
            data=sample_optimization_data,
        )
        assert isinstance(roi, (int, float, np.floating)), (
            f"ROI must be numeric, got {type(roi)}"
        )

    def test_higher_budget_higher_or_equal_retained_value(
        self, budget_optimizer, sample_optimization_data
    ):
        """Increasing budget should retain at least as much value."""
        result_low = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=10_000_000,
        )
        result_high = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=50_000_000,
        )

        roi_low = budget_optimizer.compute_roi(
            allocation=result_low,
            data=sample_optimization_data,
        )
        roi_high = budget_optimizer.compute_roi(
            allocation=result_high,
            data=sample_optimization_data,
        )

        # Higher budget should yield at least the same absolute retained value
        # (diminishing returns possible, but absolute value should not decrease)
        assert roi_high >= roi_low * 0.95, (
            "Higher budget should not dramatically reduce retained value"
        )


# ---------------------------------------------------------------------------
# Channel-level allocation tests
# ---------------------------------------------------------------------------

class TestChannelAllocation:
    """Test budget distribution across channels/segments."""

    def test_allocate_returns_per_customer_df(self, budget_optimizer,
                                               sample_optimization_data,
                                               config):
        """allocate must return a DataFrame with customer_id and budget."""
        total_budget = config["budget"]["total_krw"]
        allocation = budget_optimizer.allocate(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        assert isinstance(allocation, pd.DataFrame)
        assert "customer_id" in allocation.columns
        assert "allocated_budget" in allocation.columns

    def test_segment_budget_summary(self, budget_optimizer,
                                     sample_optimization_data, config):
        """Must be able to summarize budget by customer segment."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        merged = result.merge(
            sample_optimization_data[["customer_id", "segment"]],
            on="customer_id",
        )
        segment_totals = merged.groupby("segment")["allocated_budget"].sum()

        assert len(segment_totals) > 0, "Segment budget summary is empty"
        assert segment_totals.sum() <= total_budget * 1.001


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------

class TestBudgetSensitivity:
    """Test budget sensitivity analysis."""

    def test_sensitivity_across_budget_levels(self, budget_optimizer,
                                               sample_optimization_data):
        """Optimizer should handle different budget levels gracefully."""
        budgets = [5_000_000, 10_000_000, 25_000_000, 50_000_000]
        previous_total = 0

        for budget in budgets:
            result = budget_optimizer.optimize(
                data=sample_optimization_data,
                total_budget=budget,
            )
            current_total = result["allocated_budget"].sum()

            assert current_total <= budget * 1.001, (
                f"Allocation {current_total:,.0f} exceeds budget {budget:,.0f}"
            )
            assert current_total >= previous_total * 0.99, (
                "Allocation should not decrease with increasing budget"
            )
            previous_total = current_total


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------

class TestBudgetOutputFormat:
    """Test output format and content of optimization results."""

    def test_output_has_customer_id(self, budget_optimizer,
                                     sample_optimization_data, config):
        """Output must include customer_id column."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )
        assert "customer_id" in result.columns

    def test_output_has_allocated_budget(self, budget_optimizer,
                                          sample_optimization_data, config):
        """Output must include allocated_budget column."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )
        assert "allocated_budget" in result.columns

    def test_no_nan_in_allocation(self, budget_optimizer,
                                   sample_optimization_data, config):
        """Allocated budget must not contain NaN."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )
        assert not result["allocated_budget"].isna().any(), (
            "NaN values found in budget allocation"
        )

    def test_no_duplicate_customers(self, budget_optimizer,
                                     sample_optimization_data, config):
        """Each customer should appear exactly once."""
        total_budget = config["budget"]["total_krw"]
        result = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )
        assert result["customer_id"].nunique() == len(result), (
            "Duplicate customers in optimization result"
        )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestBudgetReproducibility:
    """Test budget optimizer reproducibility with same seed."""

    def test_same_seed_same_allocation(self, config, sample_optimization_data):
        """Same seed must produce identical budget allocations."""
        from src.models.budget_optimizer import BudgetOptimizer

        total_budget = config["budget"]["total_krw"]

        opt1 = BudgetOptimizer(config)
        result1 = opt1.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        opt2 = BudgetOptimizer(config)
        result2 = opt2.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        np.testing.assert_array_almost_equal(
            result1["allocated_budget"].values,
            result2["allocated_budget"].values,
            decimal=2,
        )


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestBudgetOptimizerPersistence:
    """Test budget optimizer save/load functionality."""

    def test_save_optimizer(self, budget_optimizer, sample_optimization_data,
                            tmp_path, config):
        """Budget optimizer state must be saveable."""
        total_budget = config["budget"]["total_krw"]
        budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        save_path = tmp_path / "budget_optimizer"
        budget_optimizer.save(str(save_path))

        saved_files = list(tmp_path.glob("budget_optimizer*"))
        assert len(saved_files) > 0, "No optimizer state saved"

    def test_load_optimizer(self, budget_optimizer, sample_optimization_data,
                            tmp_path, config):
        """Saved optimizer must be loadable and produce same results."""
        from src.models.budget_optimizer import BudgetOptimizer

        total_budget = config["budget"]["total_krw"]
        result_original = budget_optimizer.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        save_path = tmp_path / "budget_optimizer"
        budget_optimizer.save(str(save_path))

        loaded = BudgetOptimizer.load(str(save_path))
        result_loaded = loaded.optimize(
            data=sample_optimization_data,
            total_budget=total_budget,
        )

        np.testing.assert_array_almost_equal(
            result_original["allocated_budget"].values,
            result_loaded["allocated_budget"].values,
            decimal=2,
        )
