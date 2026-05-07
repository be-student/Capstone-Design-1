"""
TDD Tests for LP Budget Optimizer (src/optimization/budget_optimizer.py).

Tests cover:
- LPBudgetOptimizer instantiation and interface
- Single-channel LP solving with total budget constraint
- Multi-channel LP with per-channel min/max constraints
- Objective function maximises retained CLV
- Constraint satisfaction (total budget, per-channel bounds, non-negativity)
- High-value customer prioritisation
- Sleeping-dog (negative uplift) exclusion
- Channel summary and customer aggregation
- Expected value computation
- Edge cases (zero budget, empty data, infeasible constraints)
- Persistence (save/load)
- Reproducibility
- Integration with existing BudgetOptimizer schema
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
    """Synthetic customer data matching existing BudgetOptimizer schema."""
    np.random.seed(42)
    n = 200

    churn_prob = np.random.beta(2, 5, n)
    clv = np.random.lognormal(10, 1, n)
    uplift = np.random.randn(n) * 0.1
    cost_per_action = np.random.choice([5000, 10000, 20000, 50000], size=n)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_prob": churn_prob,
        "clv": clv,
        "uplift_score": uplift,
        "cost_per_action": cost_per_action,
    })


@pytest.fixture
def channels():
    """Multi-channel definitions with min/max constraints."""
    from src.optimization.budget_optimizer import ChannelConstraint
    return [
        ChannelConstraint(
            name="email",
            min_budget=100_000,
            max_budget=2_000_000,
            cost_per_action=5_000,
            expected_roi_multiplier=1.0,
        ),
        ChannelConstraint(
            name="coupon",
            min_budget=200_000,
            max_budget=3_000_000,
            cost_per_action=10_000,
            expected_roi_multiplier=1.2,
        ),
        ChannelConstraint(
            name="push_notification",
            min_budget=50_000,
            max_budget=1_000_000,
            cost_per_action=2_000,
            expected_roi_multiplier=0.8,
        ),
    ]


@pytest.fixture
def optimizer(config):
    """Single-channel LP optimizer."""
    from src.optimization.budget_optimizer import LPBudgetOptimizer
    return LPBudgetOptimizer(config)


@pytest.fixture
def multi_optimizer(config, channels):
    """Multi-channel LP optimizer."""
    from src.optimization.budget_optimizer import LPBudgetOptimizer
    return LPBudgetOptimizer(config, channels=channels)


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestLPBudgetOptimizerInterface:
    """Test instantiation and public API surface."""

    def test_instantiation(self, optimizer):
        assert optimizer is not None

    def test_has_solve_method(self, optimizer):
        assert hasattr(optimizer, "solve")
        assert callable(optimizer.solve)

    def test_has_get_customer_allocations(self, optimizer):
        assert hasattr(optimizer, "get_customer_allocations")

    def test_has_get_channel_summary(self, optimizer):
        assert hasattr(optimizer, "get_channel_summary")

    def test_has_compute_expected_value(self, optimizer):
        assert hasattr(optimizer, "compute_expected_value")

    def test_reads_budget_from_config(self, optimizer, config):
        expected = config["budget"]["total_krw"]
        assert optimizer.total_budget == expected

    def test_default_single_channel(self, optimizer):
        assert len(optimizer.channels) == 1
        assert optimizer.channels[0].name == "default"

    def test_multi_channel_setup(self, multi_optimizer, channels):
        assert len(multi_optimizer.channels) == len(channels)
        names = {ch.name for ch in multi_optimizer.channels}
        assert names == {"email", "coupon", "push_notification"}


# ---------------------------------------------------------------------------
# Single-channel solve tests
# ---------------------------------------------------------------------------

class TestSingleChannelSolve:
    """Test LP solver with a single default channel."""

    def test_solve_returns_optimization_result(self, optimizer, sample_data):
        from src.optimization.budget_optimizer import OptimizationResult
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        assert isinstance(result, OptimizationResult)

    def test_solve_status_optimal(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        assert result.status == "optimal"

    def test_total_allocation_within_budget(self, optimizer, sample_data):
        budget = 5_000_000
        result = optimizer.solve(sample_data, total_budget=budget)
        assert result.total_allocated <= budget * 1.001

    def test_allocations_non_negative(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        assert (result.allocations["allocated_budget"] >= -1e-8).all()

    def test_objective_value_positive(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        assert result.objective_value >= 0

    def test_all_customers_represented(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        agg = optimizer.get_customer_allocations(result)
        assert len(agg) == len(sample_data)

    def test_customer_allocations_sum_matches(self, optimizer, sample_data):
        budget = 5_000_000
        result = optimizer.solve(sample_data, total_budget=budget)
        agg = optimizer.get_customer_allocations(result)
        assert agg["allocated_budget"].sum() <= budget * 1.001

    def test_summarize_result_metrics_contains_expected_revenue(
        self, optimizer, sample_data,
    ):
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        metrics = optimizer.summarize_result_metrics(result, sample_data)
        assert "expected_revenue_saved" in metrics
        assert metrics["expected_revenue_saved"] == pytest.approx(
            metrics["retained_value"]
        )

    def test_default_budget_sweep_uses_200_percent_cap(
        self, optimizer, sample_data,
    ):
        sweep = optimizer.run_budget_sweep(sample_data)
        assert sweep["total_budget"].tolist() == [
            optimizer.total_budget * 0.5,
            optimizer.total_budget,
            optimizer.total_budget * 2.0,
        ]


# ---------------------------------------------------------------------------
# Multi-channel constraint tests
# ---------------------------------------------------------------------------

class TestMultiChannelConstraints:
    """Test per-channel min/max constraints are respected."""

    def test_multi_channel_solve_optimal(self, multi_optimizer, sample_data):
        result = multi_optimizer.solve(sample_data, total_budget=5_000_000)
        assert result.status == "optimal"

    def test_channel_min_constraints(self, multi_optimizer, sample_data, channels):
        result = multi_optimizer.solve(sample_data, total_budget=5_000_000)
        summary = multi_optimizer.get_channel_summary(result)
        for ch in channels:
            if ch.min_budget > 0 and ch.name in summary:
                assert summary[ch.name] >= ch.min_budget * 0.999, (
                    f"Channel {ch.name}: {summary[ch.name]:.0f} < min {ch.min_budget:.0f}"
                )

    def test_channel_max_constraints(self, multi_optimizer, sample_data, channels):
        result = multi_optimizer.solve(sample_data, total_budget=5_000_000)
        summary = multi_optimizer.get_channel_summary(result)
        for ch in channels:
            if ch.max_budget is not None and ch.name in summary:
                assert summary[ch.name] <= ch.max_budget * 1.001, (
                    f"Channel {ch.name}: {summary[ch.name]:.0f} > max {ch.max_budget:.0f}"
                )

    def test_total_budget_respected_multi_channel(self, multi_optimizer, sample_data):
        budget = 5_000_000
        result = multi_optimizer.solve(sample_data, total_budget=budget)
        assert result.total_allocated <= budget * 1.001

    def test_channel_summary_has_all_channels(self, multi_optimizer, sample_data, channels):
        result = multi_optimizer.solve(sample_data, total_budget=5_000_000)
        summary = multi_optimizer.get_channel_summary(result)
        for ch in channels:
            assert ch.name in summary


# ---------------------------------------------------------------------------
# Objective quality tests
# ---------------------------------------------------------------------------

class TestObjectiveQuality:
    """Test that LP prioritises high-value customers correctly."""

    def test_high_clv_customers_get_more(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=2_000_000)
        agg = optimizer.get_customer_allocations(result)
        merged = agg.merge(sample_data[["customer_id", "clv"]], on="customer_id")
        median_clv = merged["clv"].median()
        high_avg = merged[merged["clv"] >= median_clv]["allocated_budget"].mean()
        low_avg = merged[merged["clv"] < median_clv]["allocated_budget"].mean()
        assert high_avg >= low_avg

    def test_positive_uplift_preferred(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=2_000_000)
        agg = optimizer.get_customer_allocations(result)
        merged = agg.merge(
            sample_data[["customer_id", "uplift_score"]], on="customer_id",
        )
        pos_avg = merged[merged["uplift_score"] > 0]["allocated_budget"].mean()
        neg_avg = merged[merged["uplift_score"] <= 0]["allocated_budget"].mean()
        assert pos_avg >= neg_avg

    def test_sleeping_dogs_get_zero(self, optimizer, sample_data):
        """Negative-uplift customers should get zero from LP (priority clipped to 0)."""
        result = optimizer.solve(sample_data, total_budget=2_000_000)
        agg = optimizer.get_customer_allocations(result)
        merged = agg.merge(
            sample_data[["customer_id", "uplift_score"]], on="customer_id",
        )
        neg_total = merged[merged["uplift_score"] < 0]["allocated_budget"].sum()
        # All negative uplift customers should have 0 budget (priority clipped)
        assert neg_total < 1.0  # tolerance for numerical noise

    def test_higher_budget_higher_objective(self, optimizer, sample_data):
        """More budget should yield equal or higher objective value."""
        r_low = optimizer.solve(sample_data, total_budget=1_000_000)
        r_high = optimizer.solve(sample_data, total_budget=5_000_000)
        assert r_high.objective_value >= r_low.objective_value * 0.99

    def test_roi_multiplier_affects_channel_allocation(
        self, config, sample_data
    ):
        """Channel with higher ROI multiplier should get more budget."""
        from src.optimization.budget_optimizer import (
            LPBudgetOptimizer, ChannelConstraint,
        )
        ch_low = ChannelConstraint(
            name="low_roi", cost_per_action=10_000,
            expected_roi_multiplier=0.5,
        )
        ch_high = ChannelConstraint(
            name="high_roi", cost_per_action=10_000,
            expected_roi_multiplier=2.0,
        )
        opt = LPBudgetOptimizer(config, channels=[ch_low, ch_high])
        result = opt.solve(sample_data, total_budget=2_000_000)
        summary = opt.get_channel_summary(result)
        assert summary.get("high_roi", 0) >= summary.get("low_roi", 0)


# ---------------------------------------------------------------------------
# Expected value computation
# ---------------------------------------------------------------------------

class TestExpectedValue:
    """Test compute_expected_value method."""

    def test_returns_float(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        ev = optimizer.compute_expected_value(result, sample_data)
        assert isinstance(ev, float)

    def test_non_negative(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        ev = optimizer.compute_expected_value(result, sample_data)
        assert ev >= 0

    def test_zero_budget_zero_value(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=0)
        ev = optimizer.compute_expected_value(result, sample_data)
        assert ev == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_budget(self, optimizer, sample_data):
        result = optimizer.solve(sample_data, total_budget=0)
        assert result.total_allocated == 0.0

    def test_empty_data(self, optimizer):
        empty = pd.DataFrame(columns=[
            "customer_id", "clv", "uplift_score", "churn_prob", "cost_per_action",
        ])
        result = optimizer.solve(empty, total_budget=5_000_000)
        assert result.total_allocated == 0.0
        assert result.status in ("empty", "optimal")

    def test_single_customer(self, optimizer):
        data = pd.DataFrame({
            "customer_id": ["C00001"],
            "clv": [100_000.0],
            "uplift_score": [0.2],
            "churn_prob": [0.5],
            "cost_per_action": [10_000.0],
        })
        result = optimizer.solve(data, total_budget=50_000)
        assert result.status == "optimal"
        assert result.total_allocated <= 50_000 * 1.001

    def test_infeasible_min_constraints(self, config, sample_data):
        """Min constraints exceeding total budget should be infeasible."""
        from src.optimization.budget_optimizer import (
            LPBudgetOptimizer, ChannelConstraint,
        )
        channels = [
            ChannelConstraint(name="ch1", min_budget=3_000_000),
            ChannelConstraint(name="ch2", min_budget=3_000_000),
        ]
        opt = LPBudgetOptimizer(config, channels=channels)
        result = opt.solve(sample_data, total_budget=1_000_000)
        # Should detect infeasibility
        assert result.status in ("infeasible", "failed", "empty")

    def test_all_negative_uplift(self, optimizer):
        """All sleeping dogs should result in zero allocation."""
        data = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(10)],
            "clv": [50_000.0] * 10,
            "uplift_score": [-0.1] * 10,
            "churn_prob": [0.5] * 10,
            "cost_per_action": [10_000.0] * 10,
        })
        result = optimizer.solve(data, total_budget=100_000)
        # All priorities are clipped to 0, so no allocation
        assert result.total_allocated < 1.0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestLPBudgetOptimizerPersistence:
    """Test save/load functionality."""

    def test_save_creates_files(self, optimizer, sample_data, tmp_path):
        optimizer.solve(sample_data, total_budget=5_000_000)
        save_path = tmp_path / "lp_optimizer"
        optimizer.save(str(save_path))

        assert (tmp_path / "lp_optimizer.json").exists()
        assert (tmp_path / "lp_optimizer_allocations.csv").exists()

    def test_load_restores_config(self, optimizer, sample_data, tmp_path):
        from src.optimization.budget_optimizer import LPBudgetOptimizer
        optimizer.solve(sample_data, total_budget=5_000_000)
        save_path = tmp_path / "lp_optimizer"
        optimizer.save(str(save_path))

        loaded = LPBudgetOptimizer.load(str(save_path))
        assert loaded.total_budget == optimizer.total_budget
        assert loaded.seed == optimizer.seed

    def test_load_restores_channels(self, multi_optimizer, sample_data,
                                     tmp_path, channels):
        from src.optimization.budget_optimizer import LPBudgetOptimizer
        multi_optimizer.solve(sample_data, total_budget=5_000_000)
        save_path = tmp_path / "lp_multi"
        multi_optimizer.save(str(save_path))

        loaded = LPBudgetOptimizer.load(str(save_path))
        assert len(loaded.channels) == len(channels)
        loaded_names = {ch.name for ch in loaded.channels}
        expected_names = {ch.name for ch in channels}
        assert loaded_names == expected_names

    def test_loaded_optimizer_can_solve(self, optimizer, sample_data, tmp_path):
        from src.optimization.budget_optimizer import LPBudgetOptimizer
        optimizer.solve(sample_data, total_budget=5_000_000)
        save_path = tmp_path / "lp_opt"
        optimizer.save(str(save_path))

        loaded = LPBudgetOptimizer.load(str(save_path))
        result = loaded.solve(sample_data, total_budget=5_000_000)
        assert result.status == "optimal"


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestReproducibility:
    """LP is deterministic; same input should give same output."""

    def test_same_input_same_result(self, config, sample_data):
        from src.optimization.budget_optimizer import LPBudgetOptimizer
        opt1 = LPBudgetOptimizer(config)
        opt2 = LPBudgetOptimizer(config)

        r1 = opt1.solve(sample_data, total_budget=5_000_000)
        r2 = opt2.solve(sample_data, total_budget=5_000_000)

        agg1 = opt1.get_customer_allocations(r1).sort_values("customer_id")
        agg2 = opt2.get_customer_allocations(r2).sort_values("customer_id")

        np.testing.assert_array_almost_equal(
            agg1["allocated_budget"].values,
            agg2["allocated_budget"].values,
            decimal=2,
        )

    def test_objective_values_match(self, config, sample_data):
        from src.optimization.budget_optimizer import LPBudgetOptimizer
        opt1 = LPBudgetOptimizer(config)
        opt2 = LPBudgetOptimizer(config)

        r1 = opt1.solve(sample_data, total_budget=5_000_000)
        r2 = opt2.solve(sample_data, total_budget=5_000_000)

        assert abs(r1.objective_value - r2.objective_value) < 1.0


# ---------------------------------------------------------------------------
# Integration with existing BudgetOptimizer schema
# ---------------------------------------------------------------------------

class TestSchemaCompatibility:
    """Ensure LP optimizer works with the same data schema as
    src.models.budget_optimizer.BudgetOptimizer."""

    def test_accepts_same_dataframe_schema(self, optimizer):
        """LP optimizer should accept the same DataFrame columns."""
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "churn_prob": np.random.beta(2, 5, n),
            "clv": np.random.lognormal(10, 1, n),
            "uplift_score": np.random.randn(n) * 0.1,
            "cost_per_action": np.random.choice([5000, 10000, 20000], size=n),
            "expected_retention_lift": np.random.uniform(0, 0.3, n),
            "segment": np.random.choice(["vip", "regular", "dormant"], size=n),
        })
        result = optimizer.solve(data, total_budget=1_000_000)
        assert result.status == "optimal"

    def test_customer_allocation_df_compatible(self, optimizer, sample_data):
        """get_customer_allocations output has same columns as existing optimizer."""
        result = optimizer.solve(sample_data, total_budget=5_000_000)
        agg = optimizer.get_customer_allocations(result)
        assert "customer_id" in agg.columns
        assert "allocated_budget" in agg.columns
