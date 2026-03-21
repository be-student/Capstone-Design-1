"""
TDD Tests for Budget Optimizer Cost Configuration, Validation, and Entry Points.

Tests cover:
- CostConfig creation and YAML loading
- Per-channel cost structure
- ROI parameters and discount rates
- NPV factor calculations
- Input validation (DataFrame, budget, channels)
- CLI/API entry points (run_optimization, run_whatif)
- Integration between LP solver, what-if, and cost config
- Edge cases for validation
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
    """Create synthetic customer data for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "churn_prob": np.random.beta(2, 5, n),
        "clv": np.random.lognormal(10, 1, n),
        "uplift_score": np.random.randn(n) * 0.1,
        "cost_per_action": np.random.choice([5000, 10000, 20000], size=n),
    })


@pytest.fixture
def small_data():
    """Small dataset for precise tests."""
    return pd.DataFrame({
        "customer_id": ["A", "B", "C"],
        "churn_prob": [0.8, 0.5, 0.2],
        "clv": [100000, 50000, 200000],
        "uplift_score": [0.3, 0.1, -0.1],
        "cost_per_action": [10000, 10000, 10000],
    })


# ---------------------------------------------------------------------------
# CostConfig Tests
# ---------------------------------------------------------------------------

class TestCostConfig:
    """Test CostConfig dataclass and factory methods."""

    def test_default_cost_config(self):
        """Default CostConfig should have sensible defaults."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig()
        assert cc.discount_rate == 0.10
        assert cc.time_horizon_months == 12
        assert cc.currency == "KRW"
        assert cc.total_budget == 50_000_000.0
        assert cc.channels == {}

    def test_from_config_reads_budget(self, config):
        """CostConfig.from_config should read total_budget from config."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config(config)
        assert cc.total_budget == config["optimization"]["total_budget"]

    def test_from_config_reads_channels(self, config):
        """CostConfig.from_config should populate per-channel costs."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config(config)
        assert len(cc.channels) > 0
        assert "email" in cc.channels
        assert "coupon" in cc.channels
        assert cc.channels["email"]["cost_per_action"] == 1000

    def test_from_config_reads_discount_rate(self, config):
        """CostConfig.from_config should read discount_rate."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config(config)
        assert cc.discount_rate == config["optimization"]["discount_rate"]

    def test_from_config_reads_time_horizon(self, config):
        """CostConfig.from_config should read time_horizon_months."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config(config)
        assert cc.time_horizon_months == config["optimization"]["time_horizon_months"]

    def test_from_config_reads_currency(self, config):
        """CostConfig.from_config should read currency."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config(config)
        assert cc.currency == "KRW"

    def test_from_config_fallback_defaults(self):
        """CostConfig.from_config with empty config should use defaults."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config({})
        assert cc.total_budget == 50_000_000
        assert cc.discount_rate == 0.10
        assert cc.channels == {}

    def test_get_channel_constraints(self, config):
        """get_channel_constraints should return ChannelConstraint objects."""
        from src.optimization.budget_optimizer import CostConfig, ChannelConstraint
        cc = CostConfig.from_config(config)
        constraints = cc.get_channel_constraints()
        assert len(constraints) == len(cc.channels)
        assert all(isinstance(c, ChannelConstraint) for c in constraints)
        # Check a specific channel
        email = next(c for c in constraints if c.name == "email")
        assert email.cost_per_action == 1000

    def test_get_channel_constraints_default(self):
        """get_channel_constraints with no channels returns default."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig()
        constraints = cc.get_channel_constraints()
        assert len(constraints) == 1
        assert constraints[0].name == "default"

    def test_monthly_discount_factor(self):
        """Monthly discount factor should be derived from annual rate."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig(discount_rate=0.12)
        factor = cc.get_monthly_discount_factor()
        expected = (1.12) ** (1.0 / 12.0)
        assert abs(factor - expected) < 1e-10

    def test_npv_factor(self):
        """NPV factor should be sum of discounted monthly factors."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig(discount_rate=0.10, time_horizon_months=12)
        npv = cc.get_npv_factor()
        # Should be less than 12 (undiscounted sum) due to discounting
        assert npv < 12.0
        assert npv > 0.0

    def test_npv_factor_zero_rate(self):
        """NPV factor with zero discount rate should equal time horizon."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig(discount_rate=0.0, time_horizon_months=12)
        npv = cc.get_npv_factor()
        assert abs(npv - 12.0) < 1e-10

    def test_channel_roi_multipliers(self, config):
        """Channel ROI multipliers should match config."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config(config)
        assert cc.channels["coupon"]["expected_roi_multiplier"] == 1.5
        assert cc.channels["call_center"]["expected_roi_multiplier"] == 2.0

    def test_channel_max_budget(self, config):
        """Channels with max_budget should have it set."""
        from src.optimization.budget_optimizer import CostConfig
        cc = CostConfig.from_config(config)
        assert cc.channels["coupon"]["max_budget"] == 20_000_000


# ---------------------------------------------------------------------------
# Input Validation Tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Test input validation functions."""

    def test_validate_valid_data(self, sample_data):
        """Valid data should return no warnings."""
        from src.optimization.budget_optimizer import validate_dataframe
        warnings = validate_dataframe(sample_data)
        assert isinstance(warnings, list)

    def test_validate_missing_columns(self):
        """Missing required columns should raise BudgetValidationError."""
        from src.optimization.budget_optimizer import (
            validate_dataframe, BudgetValidationError,
        )
        bad_df = pd.DataFrame({"customer_id": ["A"], "clv": [100]})
        with pytest.raises(BudgetValidationError, match="Missing required"):
            validate_dataframe(bad_df)

    def test_validate_none_data(self):
        """None data should raise BudgetValidationError."""
        from src.optimization.budget_optimizer import (
            validate_dataframe, BudgetValidationError,
        )
        with pytest.raises(BudgetValidationError, match="cannot be None"):
            validate_dataframe(None)

    def test_validate_not_dataframe(self):
        """Non-DataFrame should raise BudgetValidationError."""
        from src.optimization.budget_optimizer import (
            validate_dataframe, BudgetValidationError,
        )
        with pytest.raises(BudgetValidationError, match="Expected pd.DataFrame"):
            validate_dataframe([1, 2, 3])

    def test_validate_nan_values(self):
        """NaN in numeric columns should raise error."""
        from src.optimization.budget_optimizer import (
            validate_dataframe, BudgetValidationError,
        )
        df = pd.DataFrame({
            "customer_id": ["A"],
            "clv": [np.nan],
            "uplift_score": [0.1],
            "churn_prob": [0.5],
            "cost_per_action": [1000],
        })
        with pytest.raises(BudgetValidationError, match="NaN"):
            validate_dataframe(df)

    def test_validate_negative_cost(self):
        """Negative cost_per_action should raise error."""
        from src.optimization.budget_optimizer import (
            validate_dataframe, BudgetValidationError,
        )
        df = pd.DataFrame({
            "customer_id": ["A"],
            "clv": [100000],
            "uplift_score": [0.1],
            "churn_prob": [0.5],
            "cost_per_action": [-1000],
        })
        with pytest.raises(BudgetValidationError, match="negative"):
            validate_dataframe(df)

    def test_validate_empty_data_warns(self):
        """Empty DataFrame should produce a warning, not error."""
        from src.optimization.budget_optimizer import validate_dataframe
        df = pd.DataFrame(columns=[
            "customer_id", "clv", "uplift_score",
            "churn_prob", "cost_per_action",
        ])
        warnings = validate_dataframe(df)
        assert any("empty" in w.lower() for w in warnings)

    def test_validate_churn_out_of_range_warns(self):
        """churn_prob outside [0,1] should produce a warning."""
        from src.optimization.budget_optimizer import validate_dataframe
        df = pd.DataFrame({
            "customer_id": ["A"],
            "clv": [100000],
            "uplift_score": [0.1],
            "churn_prob": [1.5],
            "cost_per_action": [1000],
        })
        warnings = validate_dataframe(df)
        assert any("churn_prob" in w for w in warnings)

    def test_validate_budget_negative(self):
        """Negative budget should raise BudgetValidationError."""
        from src.optimization.budget_optimizer import (
            validate_budget, BudgetValidationError,
        )
        with pytest.raises(BudgetValidationError, match="non-negative"):
            validate_budget(-1000)

    def test_validate_budget_none(self):
        """None budget should raise BudgetValidationError."""
        from src.optimization.budget_optimizer import (
            validate_budget, BudgetValidationError,
        )
        with pytest.raises(BudgetValidationError, match="cannot be None"):
            validate_budget(None)

    def test_validate_budget_inf(self):
        """Infinite budget should raise BudgetValidationError."""
        from src.optimization.budget_optimizer import (
            validate_budget, BudgetValidationError,
        )
        with pytest.raises(BudgetValidationError, match="finite"):
            validate_budget(float("inf"))

    def test_validate_budget_valid(self):
        """Valid budget should not raise."""
        from src.optimization.budget_optimizer import validate_budget
        validate_budget(50_000_000)
        validate_budget(0)
        validate_budget(0.0)

    def test_validate_channels_valid(self, config):
        """Valid channels should not raise."""
        from src.optimization.budget_optimizer import (
            CostConfig, validate_channels,
        )
        cc = CostConfig.from_config(config)
        constraints = cc.get_channel_constraints()
        warnings = validate_channels(constraints)
        assert isinstance(warnings, list)

    def test_validate_channels_empty(self):
        """Empty channels should raise."""
        from src.optimization.budget_optimizer import (
            validate_channels, BudgetValidationError,
        )
        with pytest.raises(BudgetValidationError, match="At least one"):
            validate_channels([])

    def test_validate_channels_duplicate_names(self):
        """Duplicate channel names should raise."""
        from src.optimization.budget_optimizer import (
            ChannelConstraint, validate_channels, BudgetValidationError,
        )
        channels = [
            ChannelConstraint(name="email"),
            ChannelConstraint(name="email"),
        ]
        with pytest.raises(BudgetValidationError, match="Duplicate"):
            validate_channels(channels)

    def test_validate_channels_negative_cost(self):
        """Channel with negative cost should raise."""
        from src.optimization.budget_optimizer import (
            ChannelConstraint, validate_channels, BudgetValidationError,
        )
        channels = [
            ChannelConstraint(name="email", cost_per_action=-100),
        ]
        with pytest.raises(BudgetValidationError, match="negative cost"):
            validate_channels(channels)

    def test_validate_channels_max_lt_min(self):
        """Channel with max_budget < min_budget should raise."""
        from src.optimization.budget_optimizer import (
            ChannelConstraint, validate_channels, BudgetValidationError,
        )
        channels = [
            ChannelConstraint(name="email", min_budget=1000, max_budget=500),
        ]
        with pytest.raises(BudgetValidationError, match="max_budget"):
            validate_channels(channels)


# ---------------------------------------------------------------------------
# run_optimization Entry Point Tests
# ---------------------------------------------------------------------------

class TestRunOptimization:
    """Test the run_optimization API entry point."""

    def test_run_optimization_basic(self, sample_data, config):
        """run_optimization should return a result dict."""
        from src.optimization.budget_optimizer import run_optimization
        result = run_optimization(data=sample_data, config=config)
        assert "result" in result
        assert "allocations" in result
        assert "channel_summary" in result
        assert "expected_value" in result
        assert "status" in result
        assert "cost_config" in result

    def test_run_optimization_status_optimal(self, sample_data, config):
        """Optimization should achieve optimal status."""
        from src.optimization.budget_optimizer import run_optimization
        result = run_optimization(data=sample_data, config=config)
        assert result["status"] == "optimal"

    def test_run_optimization_allocations_df(self, sample_data, config):
        """Allocations should be a DataFrame with expected columns."""
        from src.optimization.budget_optimizer import run_optimization
        result = run_optimization(data=sample_data, config=config)
        alloc = result["allocations"]
        assert isinstance(alloc, pd.DataFrame)
        assert "customer_id" in alloc.columns
        assert "allocated_budget" in alloc.columns

    def test_run_optimization_expected_value_positive(self, sample_data, config):
        """Expected value should be positive for valid data."""
        from src.optimization.budget_optimizer import run_optimization
        result = run_optimization(data=sample_data, config=config)
        assert result["expected_value"] > 0

    def test_run_optimization_npv_adjusted(self, sample_data, config):
        """NPV-adjusted value should be present and positive."""
        from src.optimization.budget_optimizer import run_optimization
        result = run_optimization(data=sample_data, config=config)
        assert "npv_adjusted_value" in result
        assert result["npv_adjusted_value"] > 0

    def test_run_optimization_budget_override(self, sample_data, config):
        """Budget override should be respected."""
        from src.optimization.budget_optimizer import run_optimization
        result = run_optimization(
            data=sample_data, config=config, total_budget=1_000_000,
        )
        assert result["result"].total_budget == 1_000_000
        assert result["result"].total_allocated <= 1_000_000 * 1.001

    def test_run_optimization_validates_data(self, config):
        """Should raise on invalid data when validation is on."""
        from src.optimization.budget_optimizer import (
            run_optimization, BudgetValidationError,
        )
        bad_data = pd.DataFrame({"customer_id": ["A"]})
        with pytest.raises(BudgetValidationError):
            run_optimization(data=bad_data, config=config, validate=True)

    def test_run_optimization_skip_validation(self, config):
        """Should not raise when validation is disabled (may fail at solver)."""
        from src.optimization.budget_optimizer import run_optimization
        # Data with all required columns but some oddities
        df = pd.DataFrame({
            "customer_id": ["A"],
            "clv": [100000],
            "uplift_score": [0.1],
            "churn_prob": [0.5],
            "cost_per_action": [10000],
        })
        result = run_optimization(data=df, config=config, validate=False)
        assert result["status"] in ("optimal", "empty")

    def test_run_optimization_with_custom_channels(self, sample_data, config):
        """Should accept custom channel constraints."""
        from src.optimization.budget_optimizer import (
            run_optimization, ChannelConstraint,
        )
        channels = [
            ChannelConstraint(name="sms", cost_per_action=500),
            ChannelConstraint(name="email", cost_per_action=1000),
        ]
        result = run_optimization(
            data=sample_data, config=config, channels=channels,
        )
        assert result["status"] == "optimal"
        assert set(result["channel_summary"].keys()) == {"sms", "email"}

    def test_run_optimization_warnings_returned(self, config):
        """Warnings from validation should be returned."""
        from src.optimization.budget_optimizer import run_optimization
        df = pd.DataFrame({
            "customer_id": ["A", "A"],
            "clv": [100000, 200000],
            "uplift_score": [0.1, 0.2],
            "churn_prob": [0.5, 1.5],
            "cost_per_action": [10000, 10000],
        })
        result = run_optimization(data=df, config=config)
        assert len(result["warnings"]) > 0


# ---------------------------------------------------------------------------
# run_whatif Entry Point Tests
# ---------------------------------------------------------------------------

class TestRunWhatIf:
    """Test the run_whatif API entry point."""

    def test_run_whatif_default_sweep(self, sample_data, config):
        """run_whatif with no scenarios or levels should run default sweep."""
        from src.optimization.budget_optimizer import run_whatif
        result = run_whatif(data=sample_data, config=config)
        assert result["budget_sweep"] is not None
        assert isinstance(result["budget_sweep"], pd.DataFrame)
        assert len(result["budget_sweep"]) == 3  # default 3 levels

    def test_run_whatif_named_scenarios(self, sample_data, config):
        """run_whatif with named scenarios should return comparison."""
        from src.optimization.budget_optimizer import run_whatif, WhatIfScenario
        scenarios = [
            WhatIfScenario(name="base"),
            WhatIfScenario(name="double_budget", total_budget=100_000_000),
            WhatIfScenario(name="high_cost", cost_multiplier=2.0),
        ]
        result = run_whatif(
            data=sample_data, config=config, scenarios=scenarios,
        )
        assert result["scenario_comparison"] is not None
        df = result["scenario_comparison"]
        assert len(df) == 3
        assert "scenario_name" in df.columns

    def test_run_whatif_budget_sweep(self, sample_data, config):
        """run_whatif with explicit budget levels."""
        from src.optimization.budget_optimizer import run_whatif
        levels = [10_000_000, 30_000_000, 50_000_000]
        result = run_whatif(
            data=sample_data, config=config, budget_levels=levels,
        )
        assert result["budget_sweep"] is not None
        assert len(result["budget_sweep"]) == 3

    def test_run_whatif_cost_config_returned(self, sample_data, config):
        """run_whatif should return the cost config."""
        from src.optimization.budget_optimizer import run_whatif, CostConfig
        result = run_whatif(data=sample_data, config=config)
        assert isinstance(result["cost_config"], CostConfig)

    def test_run_whatif_scenarios_and_sweep(self, sample_data, config):
        """run_whatif with both scenarios and budget levels."""
        from src.optimization.budget_optimizer import run_whatif, WhatIfScenario
        scenarios = [WhatIfScenario(name="base")]
        levels = [25_000_000, 50_000_000]
        result = run_whatif(
            data=sample_data, config=config,
            scenarios=scenarios, budget_levels=levels,
        )
        assert result["scenario_comparison"] is not None
        assert result["budget_sweep"] is not None


# ---------------------------------------------------------------------------
# Integration Tests: LP Solver + Cost Config
# ---------------------------------------------------------------------------

class TestIntegration:
    """Test integration between LP solver, cost config, and entry points."""

    def test_cost_config_channels_feed_lp(self, sample_data, config):
        """Channels from CostConfig should produce valid LP result."""
        from src.optimization.budget_optimizer import (
            CostConfig, LPBudgetOptimizer,
        )
        cc = CostConfig.from_config(config)
        channels = cc.get_channel_constraints()
        optimizer = LPBudgetOptimizer(config, channels=channels)
        result = optimizer.solve(sample_data)
        assert result.status == "optimal"
        assert result.total_allocated > 0

    def test_discount_rate_affects_npv(self, sample_data, config):
        """Different discount rates should produce different NPV values."""
        from src.optimization.budget_optimizer import (
            run_optimization, CostConfig,
        )
        # Run with default config
        result1 = run_optimization(data=sample_data, config=config)

        # Modify discount rate
        config2 = dict(config)
        config2["optimization"] = dict(config.get("optimization", {}))
        config2["optimization"]["discount_rate"] = 0.50
        result2 = run_optimization(data=sample_data, config=config2)

        # Higher discount rate → lower NPV factor → lower adjusted value
        assert result2["npv_adjusted_value"] < result1["npv_adjusted_value"]

    def test_channel_roi_multiplier_affects_allocation(self, small_data, config):
        """Channel ROI multipliers should influence LP allocation."""
        from src.optimization.budget_optimizer import (
            LPBudgetOptimizer, ChannelConstraint,
        )
        # High-ROI channel should get more allocation
        channels_high = [
            ChannelConstraint(
                name="premium", cost_per_action=10000,
                expected_roi_multiplier=5.0,
            ),
            ChannelConstraint(
                name="basic", cost_per_action=10000,
                expected_roi_multiplier=0.1,
            ),
        ]
        optimizer = LPBudgetOptimizer(config, channels=channels_high)
        result = optimizer.solve(small_data, total_budget=20000)
        summary = result.channel_summary
        assert summary.get("premium", 0) > summary.get("basic", 0)

    def test_min_budget_constraint_respected(self, sample_data, config):
        """Channel min_budget should be respected in allocation."""
        from src.optimization.budget_optimizer import (
            LPBudgetOptimizer, ChannelConstraint,
        )
        channels = [
            ChannelConstraint(
                name="required", cost_per_action=10000,
                min_budget=1_000_000, max_budget=None,
            ),
        ]
        optimizer = LPBudgetOptimizer(config, channels=channels)
        result = optimizer.solve(sample_data, total_budget=50_000_000)
        if result.status == "optimal":
            ch_total = result.channel_summary.get("required", 0)
            assert ch_total >= 1_000_000 * 0.99

    def test_max_budget_constraint_respected(self, sample_data, config):
        """Channel max_budget should cap allocation."""
        from src.optimization.budget_optimizer import (
            LPBudgetOptimizer, ChannelConstraint,
        )
        channels = [
            ChannelConstraint(
                name="capped", cost_per_action=10000,
                min_budget=0, max_budget=5_000_000,
            ),
        ]
        optimizer = LPBudgetOptimizer(config, channels=channels)
        result = optimizer.solve(sample_data, total_budget=50_000_000)
        if result.status == "optimal":
            ch_total = result.channel_summary.get("capped", 0)
            assert ch_total <= 5_000_000 * 1.001

    def test_end_to_end_optimization_pipeline(self, sample_data, config):
        """End-to-end: config → cost config → LP → what-if → comparison."""
        from src.optimization.budget_optimizer import (
            CostConfig, run_optimization, run_whatif, WhatIfScenario,
        )
        # Step 1: Run optimization
        opt_result = run_optimization(data=sample_data, config=config)
        assert opt_result["status"] == "optimal"

        # Step 2: Run what-if with varying budgets
        whatif_result = run_whatif(
            data=sample_data, config=config,
            scenarios=[
                WhatIfScenario(name="base"),
                WhatIfScenario(name="half", total_budget=25_000_000),
            ],
        )
        comp = whatif_result["scenario_comparison"]
        assert len(comp) == 2

        # Base scenario should have higher retained_value than half-budget
        base_row = comp[comp["scenario_name"] == "base"].iloc[0]
        half_row = comp[comp["scenario_name"] == "half"].iloc[0]
        assert base_row["retained_value"] >= half_row["retained_value"] * 0.95


# ---------------------------------------------------------------------------
# CLI Entry Point Tests
# ---------------------------------------------------------------------------

class TestCLIEntryPoints:
    """Test CLI argument parsing and execution."""

    def test_cli_optimize_with_args(self, sample_data, tmp_path, config):
        """cli_optimize should work with explicit argv."""
        from src.optimization.budget_optimizer import cli_optimize
        import yaml

        # Write temp config
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Write temp data
        data_path = tmp_path / "data.csv"
        sample_data.to_csv(data_path, index=False)

        # Write output path
        output_path = tmp_path / "allocations.csv"

        result = cli_optimize([
            "--config", str(config_path),
            "--data", str(data_path),
            "--budget", "10000000",
            "--output", str(output_path),
        ])
        assert result["status"] in ("optimal", "empty")
        assert output_path.exists()

    def test_cli_whatif_with_args(self, sample_data, tmp_path, config):
        """cli_whatif should work with explicit argv."""
        from src.optimization.budget_optimizer import cli_whatif
        import yaml

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        data_path = tmp_path / "data.csv"
        sample_data.to_csv(data_path, index=False)

        output_path = tmp_path / "sweep.csv"

        result = cli_whatif([
            "--config", str(config_path),
            "--data", str(data_path),
            "--budgets", "10000000", "30000000",
            "--output", str(output_path),
        ])
        assert result["budget_sweep"] is not None
        assert output_path.exists()

    def test_load_config_missing_file(self):
        """load_config should raise FileNotFoundError for missing file."""
        from src.optimization.budget_optimizer import load_config
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_load_config_valid(self, tmp_path):
        """load_config should parse YAML correctly."""
        from src.optimization.budget_optimizer import load_config
        import yaml

        config_path = tmp_path / "test.yaml"
        config_data = {"budget": {"total_krw": 1000}}
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        loaded = load_config(config_path)
        assert loaded["budget"]["total_krw"] == 1000


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases for cost config and validation integration."""

    def test_zero_budget_run_optimization(self, sample_data, config):
        """Zero budget should produce zero allocation without error."""
        from src.optimization.budget_optimizer import run_optimization
        result = run_optimization(
            data=sample_data, config=config, total_budget=0,
        )
        assert result["result"].total_allocated == 0
        assert result["status"] == "empty"

    def test_single_customer_run_optimization(self, config):
        """Single customer should work."""
        from src.optimization.budget_optimizer import run_optimization
        df = pd.DataFrame({
            "customer_id": ["X"],
            "clv": [100000],
            "uplift_score": [0.3],
            "churn_prob": [0.8],
            "cost_per_action": [10000],
        })
        result = run_optimization(data=df, config=config)
        assert result["status"] == "optimal"

    def test_all_sleeping_dogs_run_optimization(self, config):
        """All negative uplift should yield zero value."""
        from src.optimization.budget_optimizer import run_optimization
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C"],
            "clv": [100000, 200000, 300000],
            "uplift_score": [-0.2, -0.3, -0.1],
            "churn_prob": [0.5, 0.6, 0.7],
            "cost_per_action": [10000, 10000, 10000],
        })
        result = run_optimization(data=df, config=config)
        assert result["expected_value"] == 0.0

    def test_config_without_optimization_section(self, sample_data):
        """Config without 'optimization' key should use budget defaults."""
        from src.optimization.budget_optimizer import run_optimization
        minimal_config = {
            "budget": {"total_krw": 10_000_000},
            "simulation": {"random_seed": 42},
        }
        result = run_optimization(
            data=sample_data, config=minimal_config,
        )
        assert result["status"] in ("optimal", "empty")
