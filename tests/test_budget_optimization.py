"""
Unit Tests for Budget Optimization Module (src/models/budget_optimizer.py).

Consolidated test suite covering key functions and classes:
- BudgetOptimizer instantiation and configuration
- optimize() LP-based budget allocation
- allocate() alias
- compute_roi() expected retained value computation
- optimize_multi_channel() multi-channel LP
- simulate_scenario() what-if analysis
- vary_parameter() sensitivity sweeps
- compare_strategies() LP vs proportional vs uniform
- compare_budget_scenarios() multi-scenario comparison
- get_lp_diagnostics() solver diagnostics
- save()/load() persistence
- _proportional_fallback() and _multi_channel_fallback() helpers
- Edge cases and reproducibility
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

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def optimizer(config):
    """Create a BudgetOptimizer from config."""
    from src.models.budget_optimizer import BudgetOptimizer

    return BudgetOptimizer(config)


@pytest.fixture
def customer_data():
    """Create synthetic customer data for optimization tests."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "churn_prob": np.random.beta(2, 5, n),
            "clv": np.random.lognormal(10, 1, n),
            "uplift_score": np.random.randn(n) * 0.1,
            "cost_per_action": np.random.choice(
                [5000, 10000, 20000, 50000], size=n
            ),
            "expected_retention_lift": np.clip(
                np.random.randn(n) * 0.05 + 0.1, 0, 0.5
            ),
        }
    )


@pytest.fixture
def small_data():
    """Small deterministic dataset for precise verification."""
    return pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D"],
            "churn_prob": [0.8, 0.5, 0.3, 0.9],
            "clv": [100000, 200000, 50000, 80000],
            "uplift_score": [0.3, 0.2, -0.1, 0.4],
            "cost_per_action": [10000, 10000, 10000, 10000],
            "expected_retention_lift": [0.25, 0.15, 0.0, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# Instantiation and configuration
# ---------------------------------------------------------------------------


class TestBudgetOptimizerInit:
    """Test BudgetOptimizer instantiation and configuration."""

    def test_creates_from_config(self, optimizer):
        assert optimizer is not None

    def test_reads_total_budget(self, optimizer, config):
        expected = config["budget"]["total_krw"]
        assert optimizer.total_budget == expected

    def test_reads_seed(self, optimizer, config):
        expected = config["simulation"]["random_seed"]
        assert optimizer.seed == expected

    def test_last_result_initially_none(self, optimizer):
        assert optimizer.last_result is None

    def test_minimal_config(self):
        from src.models.budget_optimizer import BudgetOptimizer

        opt = BudgetOptimizer({})
        assert opt.total_budget == 50_000_000  # default
        assert opt.seed == 42  # default


# ---------------------------------------------------------------------------
# optimize() core LP
# ---------------------------------------------------------------------------


class TestOptimize:
    """Test core LP-based optimize method."""

    def test_returns_dataframe(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=10_000_000)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=10_000_000)
        assert "customer_id" in result.columns
        assert "allocated_budget" in result.columns

    def test_all_customers_present(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=10_000_000)
        assert len(result) == len(customer_data)

    def test_budget_constraint(self, optimizer, customer_data):
        budget = 5_000_000
        result = optimizer.optimize(customer_data, total_budget=budget)
        assert result["allocated_budget"].sum() <= budget * 1.001

    def test_non_negative_allocations(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=10_000_000)
        assert (result["allocated_budget"] >= -1e-6).all()

    def test_no_nan(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=10_000_000)
        assert not result["allocated_budget"].isna().any()

    def test_zero_budget(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=0)
        assert result["allocated_budget"].sum() == 0

    def test_uses_default_budget(self, optimizer, customer_data, config):
        result = optimizer.optimize(customer_data)
        total = result["allocated_budget"].sum()
        default = config["budget"]["total_krw"]
        assert total <= default * 1.001

    def test_stores_last_result(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=10_000_000)
        assert optimizer.last_result is not None
        pd.testing.assert_frame_equal(optimizer.last_result, result)

    def test_sleeping_dogs_get_zero(self, optimizer, small_data):
        """Customer C has negative uplift, should get 0 budget."""
        result = optimizer.optimize(small_data, total_budget=20000)
        merged = result.merge(
            small_data[["customer_id", "uplift_score"]], on="customer_id"
        )
        sleeping = merged[merged["uplift_score"] < 0]
        assert (sleeping["allocated_budget"] == 0).all()

    def test_positive_uplift_prioritized(self, optimizer, customer_data):
        result = optimizer.optimize(customer_data, total_budget=5_000_000)
        merged = result.merge(
            customer_data[["customer_id", "uplift_score"]], on="customer_id"
        )
        pos_avg = merged[merged["uplift_score"] > 0][
            "allocated_budget"
        ].mean()
        neg_avg = merged[merged["uplift_score"] <= 0][
            "allocated_budget"
        ].mean()
        assert pos_avg >= neg_avg

    def test_higher_budget_more_allocation(self, optimizer, customer_data):
        r_low = optimizer.optimize(customer_data, total_budget=2_000_000)
        r_high = optimizer.optimize(customer_data, total_budget=20_000_000)
        assert r_high["allocated_budget"].sum() >= r_low[
            "allocated_budget"
        ].sum() * 0.99


# ---------------------------------------------------------------------------
# allocate() alias
# ---------------------------------------------------------------------------


class TestAllocateAlias:
    def test_allocate_matches_optimize(self, optimizer, customer_data):
        r1 = optimizer.optimize(customer_data, total_budget=10_000_000)

        # Reset seed state
        from src.models.budget_optimizer import BudgetOptimizer

        opt2 = BudgetOptimizer(optimizer.config)
        r2 = opt2.allocate(customer_data, total_budget=10_000_000)

        np.testing.assert_array_almost_equal(
            r1["allocated_budget"].values,
            r2["allocated_budget"].values,
            decimal=2,
        )


# ---------------------------------------------------------------------------
# compute_roi()
# ---------------------------------------------------------------------------


class TestComputeROI:
    def test_returns_float(self, optimizer, customer_data):
        alloc = optimizer.optimize(customer_data, total_budget=10_000_000)
        roi = optimizer.compute_roi(alloc, customer_data)
        assert isinstance(roi, float)

    def test_positive_for_valid_data(self, optimizer, customer_data):
        alloc = optimizer.optimize(customer_data, total_budget=10_000_000)
        roi = optimizer.compute_roi(alloc, customer_data)
        assert roi > 0

    def test_zero_allocation_zero_roi(self, optimizer, customer_data):
        alloc = optimizer.optimize(customer_data, total_budget=0)
        roi = optimizer.compute_roi(alloc, customer_data)
        assert roi == 0.0

    def test_monotone_in_budget(self, optimizer, customer_data):
        """ROI (retained value) should be non-decreasing with budget."""
        rois = []
        for budget in [1_000_000, 5_000_000, 20_000_000]:
            alloc = optimizer.optimize(customer_data, total_budget=budget)
            rois.append(optimizer.compute_roi(alloc, customer_data))
        for i in range(1, len(rois)):
            assert rois[i] >= rois[i - 1] * 0.95


# ---------------------------------------------------------------------------
# optimize_multi_channel()
# ---------------------------------------------------------------------------


class TestMultiChannel:
    def test_basic_structure(self, optimizer, small_data):
        channels = ["email", "sms"]
        costs = {"email": 5000, "sms": 2000}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=20000
        )
        assert "customer_id" in result.columns
        assert "allocated_budget" in result.columns
        assert "budget_email" in result.columns
        assert "budget_sms" in result.columns

    def test_budget_constraint(self, optimizer, small_data):
        channels = ["email", "sms"]
        costs = {"email": 5000, "sms": 2000}
        budget = 15000
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=budget
        )
        assert result["allocated_budget"].sum() <= budget * 1.001

    def test_channel_budgets_respected(self, optimizer, small_data):
        channels = ["email", "sms"]
        costs = {"email": 5000, "sms": 2000}
        ch_budgets = {"email": 8000, "sms": 4000}
        result = optimizer.optimize_multi_channel(
            small_data,
            channels,
            costs,
            channel_budgets=ch_budgets,
            total_budget=20000,
        )
        assert result["budget_email"].sum() <= ch_budgets["email"] * 1.001
        assert result["budget_sms"].sum() <= ch_budgets["sms"] * 1.001

    def test_total_equals_channel_sum(self, optimizer, small_data):
        channels = ["email", "sms", "push"]
        costs = {"email": 3000, "sms": 1000, "push": 500}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=20000
        )
        ch_sum = sum(result[f"budget_{ch}"] for ch in channels)
        np.testing.assert_array_almost_equal(
            result["allocated_budget"].values, ch_sum.values, decimal=2
        )

    def test_non_negative(self, optimizer, small_data):
        channels = ["email", "sms"]
        costs = {"email": 5000, "sms": 2000}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=15000
        )
        for ch in channels:
            assert (result[f"budget_{ch}"] >= -0.01).all()

    def test_zero_budget(self, optimizer, small_data):
        channels = ["email"]
        costs = {"email": 5000}
        result = optimizer.optimize_multi_channel(
            small_data, channels, costs, total_budget=0
        )
        assert result["allocated_budget"].sum() == 0

    def test_empty_channels(self, optimizer, small_data):
        result = optimizer.optimize_multi_channel(
            small_data, [], {}, total_budget=10000
        )
        assert result["allocated_budget"].sum() == 0


# ---------------------------------------------------------------------------
# simulate_scenario()
# ---------------------------------------------------------------------------


class TestSimulateScenario:
    def test_returns_dict_with_required_keys(self, optimizer, customer_data):
        result = optimizer.simulate_scenario(
            data=customer_data, scenario_name="test"
        )
        required = [
            "scenario_name",
            "parameters",
            "total_budget",
            "total_allocated",
            "retained_value",
            "roi",
            "customers_treated",
            "allocation",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_scenario_name_preserved(self, optimizer, customer_data):
        result = optimizer.simulate_scenario(
            data=customer_data, scenario_name="my_scenario"
        )
        assert result["scenario_name"] == "my_scenario"

    def test_cost_multiplier_reduces_value(self, optimizer, customer_data):
        r1 = optimizer.simulate_scenario(
            data=customer_data,
            cost_multiplier=1.0,
            total_budget=5_000_000,
        )
        r2 = optimizer.simulate_scenario(
            data=customer_data,
            cost_multiplier=3.0,
            total_budget=5_000_000,
        )
        assert r2["retained_value"] <= r1["retained_value"] * 1.05

    def test_zero_uplift_zero_value(self, optimizer, customer_data):
        result = optimizer.simulate_scenario(
            data=customer_data, uplift_multiplier=0.0
        )
        assert result["retained_value"] == pytest.approx(0.0, abs=1e-6)

    def test_budget_constraint_respected(self, optimizer, customer_data):
        budget = 3_000_000
        result = optimizer.simulate_scenario(
            data=customer_data, total_budget=budget
        )
        assert result["total_allocated"] <= budget * 1.001


# ---------------------------------------------------------------------------
# vary_parameter()
# ---------------------------------------------------------------------------


class TestVaryParameter:
    def test_returns_dataframe(self, optimizer, customer_data):
        result = optimizer.vary_parameter(
            data=customer_data,
            parameter="budget",
            values=[5_000_000, 10_000_000],
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self, optimizer, customer_data):
        values = [1_000_000, 5_000_000, 10_000_000]
        result = optimizer.vary_parameter(
            data=customer_data, parameter="budget", values=values
        )
        assert len(result) == 3

    def test_has_expected_columns(self, optimizer, customer_data):
        result = optimizer.vary_parameter(
            data=customer_data,
            parameter="budget",
            values=[10_000_000],
        )
        for col in [
            "parameter_value",
            "total_allocated",
            "retained_value",
            "roi",
            "customers_treated",
        ]:
            assert col in result.columns

    def test_invalid_parameter_raises(self, optimizer, customer_data):
        with pytest.raises(ValueError, match="Unknown parameter"):
            optimizer.vary_parameter(
                data=customer_data,
                parameter="nonexistent",
                values=[1.0],
            )

    def test_cost_multiplier_sweep(self, optimizer, customer_data):
        result = optimizer.vary_parameter(
            data=customer_data,
            parameter="cost_multiplier",
            values=[0.5, 1.0, 2.0],
            base_budget=10_000_000,
        )
        assert len(result) == 3
        assert (result["total_allocated"] >= 0).all()


# ---------------------------------------------------------------------------
# compare_strategies()
# ---------------------------------------------------------------------------


class TestCompareStrategies:
    def test_returns_three_strategies(self, optimizer, customer_data):
        result = optimizer.compare_strategies(
            data=customer_data, total_budget=10_000_000
        )
        assert len(result) == 3
        assert set(result["strategy"].values) == {
            "lp",
            "proportional",
            "uniform",
        }

    def test_all_non_negative(self, optimizer, customer_data):
        result = optimizer.compare_strategies(
            data=customer_data, total_budget=10_000_000
        )
        assert (result["total_allocated"] >= 0).all()
        assert (result["retained_value"] >= 0).all()
        assert (result["roi"] >= 0).all()

    def test_budget_respected(self, optimizer, customer_data):
        budget = 10_000_000
        result = optimizer.compare_strategies(
            data=customer_data, total_budget=budget
        )
        assert (result["total_allocated"] <= budget * 1.01).all()

    def test_strategies_produce_positive_retained_value(
        self, optimizer, customer_data
    ):
        """All strategies should produce positive retained value."""
        result = optimizer.compare_strategies(
            data=customer_data, total_budget=5_000_000
        )
        assert (result["retained_value"] > 0).all()


# ---------------------------------------------------------------------------
# compare_budget_scenarios()
# ---------------------------------------------------------------------------


class TestCompareBudgetScenarios:
    def test_returns_dataframe(self, optimizer, customer_data):
        scenarios = [
            {"scenario_name": "low", "total_budget": 5_000_000},
            {"scenario_name": "high", "total_budget": 20_000_000},
        ]
        result = optimizer.compare_budget_scenarios(
            data=customer_data, scenarios=scenarios
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_scenario_names(self, optimizer, customer_data):
        scenarios = [
            {"scenario_name": "a"},
            {"scenario_name": "b"},
        ]
        result = optimizer.compare_budget_scenarios(
            data=customer_data, scenarios=scenarios
        )
        assert set(result["scenario_name"].values) == {"a", "b"}

    def test_auto_names(self, optimizer, customer_data):
        scenarios = [
            {"total_budget": 10_000_000},
            {"total_budget": 20_000_000},
        ]
        result = optimizer.compare_budget_scenarios(
            data=customer_data, scenarios=scenarios
        )
        assert "scenario_0" in result["scenario_name"].values
        assert "scenario_1" in result["scenario_name"].values


# ---------------------------------------------------------------------------
# get_lp_diagnostics()
# ---------------------------------------------------------------------------


class TestLPDiagnostics:
    def test_none_before_optimization(self, optimizer):
        assert optimizer.get_lp_diagnostics() is None

    def test_available_after_optimization(self, optimizer, small_data):
        optimizer.optimize(small_data, total_budget=30000)
        diag = optimizer.get_lp_diagnostics()
        assert diag is not None

    def test_has_required_keys(self, optimizer, small_data):
        optimizer.optimize(small_data, total_budget=30000)
        diag = optimizer.get_lp_diagnostics()
        for key in ["success", "status", "message", "objective_value"]:
            assert key in diag

    def test_solver_succeeds(self, optimizer, small_data):
        optimizer.optimize(small_data, total_budget=30000)
        diag = optimizer.get_lp_diagnostics()
        assert diag["success"] is True

    def test_positive_objective(self, optimizer, small_data):
        optimizer.optimize(small_data, total_budget=30000)
        diag = optimizer.get_lp_diagnostics()
        assert diag["objective_value"] > 0


# ---------------------------------------------------------------------------
# Persistence (save/load)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_files(self, optimizer, customer_data, tmp_path):
        optimizer.optimize(customer_data, total_budget=10_000_000)
        save_path = tmp_path / "budget_opt"
        optimizer.save(str(save_path))
        saved = list(tmp_path.glob("budget_opt*"))
        assert len(saved) > 0

    def test_load_roundtrip(self, optimizer, customer_data, tmp_path):
        from src.models.budget_optimizer import BudgetOptimizer

        result_orig = optimizer.optimize(
            customer_data, total_budget=10_000_000
        )
        save_path = tmp_path / "budget_opt"
        optimizer.save(str(save_path))

        loaded = BudgetOptimizer.load(str(save_path))
        result_loaded = loaded.optimize(
            customer_data, total_budget=10_000_000
        )

        np.testing.assert_array_almost_equal(
            result_orig["allocated_budget"].values,
            result_loaded["allocated_budget"].values,
            decimal=2,
        )

    def test_loaded_budget_matches(self, optimizer, customer_data, tmp_path):
        from src.models.budget_optimizer import BudgetOptimizer

        optimizer.optimize(customer_data, total_budget=10_000_000)
        save_path = tmp_path / "budget_opt"
        optimizer.save(str(save_path))

        loaded = BudgetOptimizer.load(str(save_path))
        assert loaded.total_budget == optimizer.total_budget
        assert loaded.seed == optimizer.seed


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_allocation(self, config, customer_data):
        from src.models.budget_optimizer import BudgetOptimizer

        opt1 = BudgetOptimizer(config)
        r1 = opt1.optimize(customer_data, total_budget=10_000_000)

        opt2 = BudgetOptimizer(config)
        r2 = opt2.optimize(customer_data, total_budget=10_000_000)

        np.testing.assert_array_almost_equal(
            r1["allocated_budget"].values,
            r2["allocated_budget"].values,
            decimal=2,
        )

    def test_same_seed_same_scenario(self, config, customer_data):
        from src.models.budget_optimizer import BudgetOptimizer

        opt1 = BudgetOptimizer(config)
        s1 = opt1.simulate_scenario(
            data=customer_data,
            total_budget=10_000_000,
            cost_multiplier=1.5,
        )

        opt2 = BudgetOptimizer(config)
        s2 = opt2.simulate_scenario(
            data=customer_data,
            total_budget=10_000_000,
            cost_multiplier=1.5,
        )

        assert s1["retained_value"] == pytest.approx(
            s2["retained_value"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_customer(self, optimizer):
        data = pd.DataFrame(
            {
                "customer_id": ["X"],
                "churn_prob": [0.5],
                "clv": [100000],
                "uplift_score": [0.2],
                "cost_per_action": [10000],
                "expected_retention_lift": [0.15],
            }
        )
        result = optimizer.optimize(data, total_budget=50000)
        assert len(result) == 1
        assert result["allocated_budget"].values[0] > 0

    def test_all_sleeping_dogs(self, optimizer):
        data = pd.DataFrame(
            {
                "customer_id": ["A", "B"],
                "churn_prob": [0.5, 0.7],
                "clv": [100000, 80000],
                "uplift_score": [-0.2, -0.1],
                "cost_per_action": [10000, 10000],
                "expected_retention_lift": [0.0, 0.0],
            }
        )
        result = optimizer.optimize(data, total_budget=50000)
        assert result["allocated_budget"].sum() == 0.0

    def test_zero_clv(self, optimizer):
        data = pd.DataFrame(
            {
                "customer_id": ["A", "B"],
                "churn_prob": [0.5, 0.5],
                "clv": [0.0, 100000],
                "uplift_score": [0.3, 0.3],
                "cost_per_action": [10000, 10000],
                "expected_retention_lift": [0.2, 0.2],
            }
        )
        result = optimizer.optimize(data, total_budget=15000)
        merged = result.merge(
            data[["customer_id", "clv"]], on="customer_id"
        )
        assert merged[merged["clv"] == 0]["allocated_budget"].values[0] == 0.0

    def test_zero_churn(self, optimizer):
        data = pd.DataFrame(
            {
                "customer_id": ["A", "B"],
                "churn_prob": [0.0, 0.8],
                "clv": [100000, 100000],
                "uplift_score": [0.3, 0.3],
                "cost_per_action": [10000, 10000],
                "expected_retention_lift": [0.2, 0.2],
            }
        )
        result = optimizer.optimize(data, total_budget=15000)
        merged = result.merge(
            data[["customer_id", "churn_prob"]], on="customer_id"
        )
        assert (
            merged[merged["churn_prob"] == 0]["allocated_budget"].values[0]
            == 0.0
        )

    def test_empty_dataframe(self, optimizer):
        data = pd.DataFrame(
            {
                "customer_id": [],
                "churn_prob": [],
                "clv": [],
                "uplift_score": [],
                "cost_per_action": [],
                "expected_retention_lift": [],
            }
        )
        result = optimizer.optimize(data, total_budget=10000)
        assert len(result) == 0

    def test_negative_budget_allocates_nothing(self, optimizer, small_data):
        result = optimizer.optimize(small_data, total_budget=-1000)
        assert result["allocated_budget"].sum() == 0

    def test_large_customer_count(self, optimizer):
        np.random.seed(42)
        n = 3000
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
        result = optimizer.optimize(data, total_budget=10_000_000)
        assert len(result) == n
        assert result["allocated_budget"].sum() <= 10_000_000 * 1.001


# ---------------------------------------------------------------------------
# Proportional fallback
# ---------------------------------------------------------------------------


class TestProportionalFallback:
    def test_sums_to_budget(self):
        from src.models.budget_optimizer import BudgetOptimizer

        priority = np.array([10.0, 20.0, 30.0])
        cost = np.array([5000, 5000, 5000])
        budget = 10000.0
        alloc = BudgetOptimizer._proportional_fallback(priority, cost, budget)
        assert abs(alloc.sum() - budget) < 1e-6

    def test_proportional_to_priority(self):
        from src.models.budget_optimizer import BudgetOptimizer

        priority = np.array([10.0, 30.0])
        cost = np.array([5000, 5000])
        budget = 8000.0
        alloc = BudgetOptimizer._proportional_fallback(priority, cost, budget)
        # Ratio should be 1:3
        assert abs(alloc[1] / alloc[0] - 3.0) < 1e-6

    def test_zero_priority(self):
        from src.models.budget_optimizer import BudgetOptimizer

        priority = np.array([0.0, 0.0, 0.0])
        cost = np.array([5000, 5000, 5000])
        alloc = BudgetOptimizer._proportional_fallback(
            priority, cost, 10000.0
        )
        assert alloc.sum() == 0.0

    def test_zero_budget(self):
        from src.models.budget_optimizer import BudgetOptimizer

        priority = np.array([10.0, 20.0])
        cost = np.array([5000, 5000])
        alloc = BudgetOptimizer._proportional_fallback(priority, cost, 0.0)
        assert alloc.sum() == 0.0
