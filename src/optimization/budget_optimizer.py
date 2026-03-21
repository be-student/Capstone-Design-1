"""
LP-Based Budget Optimizer for Multi-Channel Retention Campaigns.

Uses scipy.optimize.linprog to solve a linear programming problem that
allocates a total retention budget optimally across channels/interventions.

**Objective function** (maximise retained CLV):
    max  sum_c  sum_i  (uplift_ci * clv_ci * x_ci)

Where:
    c = channel index, i = customer index within channel c,
    x_ci >= 0 is the budget allocated to customer i via channel c.

Since linprog minimises, we negate the objective coefficients.

**Constraints**:
    1. Total budget:      sum_c sum_i x_ci  <= total_budget
    2. Per-channel min:   sum_i x_ci        >= channel_min_c   (for each c)
    3. Per-channel max:   sum_i x_ci        <= channel_max_c   (for each c)
    4. Per-customer max:  x_ci              <= cost_per_action  (for each c,i)
    5. Non-negativity:    x_ci              >= 0

Integrates with the existing BudgetOptimizer in src/models/budget_optimizer.py
by accepting the same customer-level DataFrame schema and config YAML.

Provides:
- CostConfig: Configurable cost structure with per-channel costs, ROI params,
  discount rates, loaded from YAML config.
- Input validation: Strict schema/value checks before solving.
- CLI/API entry points: ``run_optimization`` and ``run_whatif`` tie the LP
  solver and what-if components together for programmatic or CLI use.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog, OptimizeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = frozenset([
    "customer_id", "clv", "uplift_score", "churn_prob", "cost_per_action",
])


class BudgetValidationError(ValueError):
    """Raised when input data or config fails validation."""


def validate_dataframe(data: pd.DataFrame) -> List[str]:
    """Validate the customer DataFrame for budget optimization.

    Checks:
    - Required columns are present.
    - No NaN in required numeric columns.
    - Numeric columns have expected value ranges.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data to validate.

    Returns
    -------
    list of str
        List of warning messages (empty if clean).

    Raises
    ------
    BudgetValidationError
        If required columns are missing or data is fundamentally invalid.
    """
    warnings: List[str] = []

    if data is None:
        raise BudgetValidationError("Data cannot be None")

    if not isinstance(data, pd.DataFrame):
        raise BudgetValidationError(
            f"Expected pd.DataFrame, got {type(data).__name__}"
        )

    missing = _REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise BudgetValidationError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(data) == 0:
        warnings.append("DataFrame is empty — solver will return zero allocation")
        return warnings

    # Check NaN in numeric columns
    numeric_cols = ["clv", "uplift_score", "churn_prob", "cost_per_action"]
    for col in numeric_cols:
        n_nan = int(data[col].isna().sum())
        if n_nan > 0:
            raise BudgetValidationError(
                f"Column '{col}' has {n_nan} NaN value(s)"
            )

    # Range checks
    if (data["churn_prob"] < 0).any() or (data["churn_prob"] > 1).any():
        warnings.append("churn_prob contains values outside [0, 1]")
    if (data["clv"] < 0).any():
        warnings.append("clv contains negative values")
    if (data["cost_per_action"] < 0).any():
        raise BudgetValidationError("cost_per_action contains negative values")

    # Duplicate customer_id
    n_dup = int(data["customer_id"].duplicated().sum())
    if n_dup > 0:
        warnings.append(f"customer_id has {n_dup} duplicate(s)")

    return warnings


def validate_budget(total_budget: float) -> None:
    """Validate total budget parameter.

    Raises
    ------
    BudgetValidationError
        If total_budget is not a finite non-negative number.
    """
    if total_budget is None:
        raise BudgetValidationError("total_budget cannot be None")
    if not isinstance(total_budget, (int, float)):
        raise BudgetValidationError(
            f"total_budget must be numeric, got {type(total_budget).__name__}"
        )
    if not np.isfinite(total_budget):
        raise BudgetValidationError("total_budget must be finite")
    if total_budget < 0:
        raise BudgetValidationError(
            f"total_budget must be non-negative, got {total_budget}"
        )


def validate_channels(channels: List["ChannelConstraint"]) -> List[str]:
    """Validate channel constraints.

    Returns
    -------
    list of str
        Warnings (empty if clean).

    Raises
    ------
    BudgetValidationError
        On structural issues.
    """
    warnings: List[str] = []
    if not channels:
        raise BudgetValidationError("At least one channel is required")
    names = [ch.name for ch in channels]
    if len(names) != len(set(names)):
        raise BudgetValidationError("Duplicate channel names detected")
    for ch in channels:
        if ch.cost_per_action < 0:
            raise BudgetValidationError(
                f"Channel '{ch.name}' has negative cost_per_action"
            )
        if ch.min_budget < 0:
            raise BudgetValidationError(
                f"Channel '{ch.name}' has negative min_budget"
            )
        if ch.max_budget is not None and ch.max_budget < ch.min_budget:
            raise BudgetValidationError(
                f"Channel '{ch.name}': max_budget ({ch.max_budget}) < "
                f"min_budget ({ch.min_budget})"
            )
        if ch.expected_roi_multiplier < 0:
            warnings.append(
                f"Channel '{ch.name}' has negative ROI multiplier "
                f"({ch.expected_roi_multiplier})"
            )
    return warnings


# ---------------------------------------------------------------------------
# Configurable cost structure
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Configurable cost structure for budget optimization.

    Centralizes per-channel costs, ROI parameters, and discount rates
    loaded from YAML configuration.

    Attributes
    ----------
    channels : dict
        Mapping of channel name -> dict with keys:
        ``cost_per_action``, ``expected_roi_multiplier``,
        ``min_budget``, ``max_budget``.
    discount_rate : float
        Annual discount rate for NPV calculations (e.g. 0.10 = 10%).
    time_horizon_months : int
        Number of months for forward-looking value calculations.
    currency : str
        Currency code (default "KRW").
    total_budget : float
        Default total budget constraint.
    """

    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    discount_rate: float = 0.10
    time_horizon_months: int = 12
    currency: str = "KRW"
    total_budget: float = 50_000_000.0

    def get_channel_constraints(self) -> List["ChannelConstraint"]:
        """Convert channel config dicts to ChannelConstraint objects.

        Returns
        -------
        list of ChannelConstraint
        """
        if not self.channels:
            return [ChannelConstraint(name="default")]

        result = []
        for name, params in self.channels.items():
            result.append(ChannelConstraint(
                name=name,
                cost_per_action=params.get("cost_per_action", 10_000.0),
                expected_roi_multiplier=params.get("expected_roi_multiplier", 1.0),
                min_budget=params.get("min_budget", 0.0),
                max_budget=params.get("max_budget"),
            ))
        return result

    def get_monthly_discount_factor(self) -> float:
        """Return monthly discount factor derived from annual rate.

        Returns
        -------
        float
            (1 + annual_rate)^(1/12) - derived monthly factor.
        """
        return (1 + self.discount_rate) ** (1.0 / 12.0)

    def get_npv_factor(self) -> float:
        """Return the NPV factor for ``time_horizon_months``.

        Returns
        -------
        float
            Sum of discounted monthly factors.
        """
        monthly = self.get_monthly_discount_factor()
        if monthly <= 1.0:
            return float(self.time_horizon_months)
        return float(sum(
            1.0 / (monthly ** m)
            for m in range(self.time_horizon_months)
        ))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CostConfig":
        """Create CostConfig from a parsed YAML config dict.

        Reads from ``config["optimization"]`` if present, otherwise
        falls back to ``config["budget"]`` and sensible defaults.

        Parameters
        ----------
        config : dict
            Full simulator_config.yaml parsed dict.

        Returns
        -------
        CostConfig
        """
        opt = config.get("optimization", {})
        budget_section = config.get("budget", {})

        total_budget = opt.get(
            "total_budget",
            budget_section.get("total_krw", 50_000_000),
        )
        currency = opt.get(
            "currency",
            budget_section.get("currency", "KRW"),
        )

        # Channel definitions from config
        channels_raw = opt.get("channels", {})
        channels = {}
        for ch_name, ch_params in channels_raw.items():
            channels[ch_name] = {
                "cost_per_action": ch_params.get("cost_per_action", 10_000.0),
                "expected_roi_multiplier": ch_params.get(
                    "expected_roi_multiplier", 1.0
                ),
                "min_budget": ch_params.get("min_budget", 0.0),
                "max_budget": ch_params.get("max_budget"),
            }

        return cls(
            channels=channels,
            discount_rate=opt.get("discount_rate", 0.10),
            time_horizon_months=opt.get("time_horizon_months", 12),
            currency=currency,
            total_budget=total_budget,
        )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class WhatIfScenario:
    """Definition of a what-if scenario for budget simulation.

    Attributes:
        name: Human-readable label for the scenario.
        total_budget: Budget constraint (KRW). None means use optimizer default.
        cost_multiplier: Multiplier applied to cost_per_action (1.0 = no change).
        uplift_multiplier: Multiplier applied to uplift_score (1.0 = no change).
    """

    name: str
    total_budget: Optional[float] = None
    cost_multiplier: float = 1.0
    uplift_multiplier: float = 1.0


@dataclass
class ChannelConstraint:
    """Budget constraints for a single channel/intervention type.

    Attributes:
        name: Channel identifier (e.g. "email", "coupon", "push_notification").
        min_budget: Minimum budget to allocate to this channel (KRW).
        max_budget: Maximum budget for this channel (KRW).  None = no cap.
        cost_per_action: Default per-customer cost for this channel (KRW).
            Overridden by per-customer cost_per_action if present in data.
        expected_roi_multiplier: Multiplier on the base uplift score when
            this channel is used (1.0 = neutral).
    """

    name: str
    min_budget: float = 0.0
    max_budget: Optional[float] = None
    cost_per_action: float = 10_000.0
    expected_roi_multiplier: float = 1.0


@dataclass
class OptimizationResult:
    """Container for LP solver output.

    Attributes:
        allocations: DataFrame with columns [customer_id, channel,
            allocated_budget].
        total_allocated: Sum of all allocated budgets (KRW).
        total_budget: The budget constraint passed to the solver.
        objective_value: Optimal objective value (expected retained CLV).
        channel_summary: Per-channel aggregated budget.
        status: Solver status string ("optimal", "infeasible", etc.).
        solver_message: Raw message from scipy solver.
    """

    allocations: pd.DataFrame
    total_allocated: float
    total_budget: float
    objective_value: float
    channel_summary: Dict[str, float]
    status: str
    solver_message: str


# ---------------------------------------------------------------------------
# LP Budget Optimizer
# ---------------------------------------------------------------------------


class LPBudgetOptimizer:
    """Linear-programming based multi-channel budget optimizer.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration (same schema as simulator_config.yaml).
    channels : list of ChannelConstraint, optional
        Channel definitions with min/max constraints.  If not provided,
        a single default channel is used.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        channels: Optional[List[ChannelConstraint]] = None,
    ):
        self.config = config
        self.total_budget = config.get("budget", {}).get("total_krw", 50_000_000)
        self.seed = config.get("simulation", {}).get("random_seed", 42)

        # Channel definitions
        if channels is not None:
            self.channels = list(channels)
        else:
            self.channels = [
                ChannelConstraint(name="default", min_budget=0.0, max_budget=None),
            ]

        self.last_result: Optional[OptimizationResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        data: pd.DataFrame,
        total_budget: Optional[float] = None,
        channels: Optional[List[ChannelConstraint]] = None,
    ) -> OptimizationResult:
        """Solve the LP to optimally allocate budget across channels and customers.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns: customer_id, clv, uplift_score, churn_prob,
            cost_per_action.  Optionally: channel (pre-assigned channel name).
        total_budget : float, optional
            Total budget constraint (KRW).  Defaults to config value.
        channels : list of ChannelConstraint, optional
            Override instance-level channel definitions for this solve call.

        Returns
        -------
        OptimizationResult
            Solved allocation with per-customer, per-channel budgets.
        """
        if total_budget is None:
            total_budget = self.total_budget
        if channels is not None:
            active_channels = list(channels)
        else:
            active_channels = self.channels

        n_customers = len(data)

        # Edge case: no customers or zero budget
        if n_customers == 0 or total_budget <= 0:
            return self._empty_result(data, total_budget)

        n_channels = len(active_channels)
        n_vars = n_customers * n_channels  # decision variable per (customer, channel)

        # ------------------------------------------------------------------
        # Build objective: maximise sum uplift_ci * clv_i * churn_i * roi_mult_c * x_ci
        # linprog minimises, so negate.
        # ------------------------------------------------------------------
        clv = data["clv"].values.astype(np.float64)
        uplift = data["uplift_score"].values.astype(np.float64)
        churn_prob = data["churn_prob"].values.astype(np.float64)
        cost_per_action = data["cost_per_action"].values.astype(np.float64)

        # Priority per customer (base, before channel multiplier)
        base_priority = np.maximum(uplift * clv * churn_prob, 0.0)

        # Objective coefficients: negate for minimisation
        c_obj = np.zeros(n_vars, dtype=np.float64)
        for ch_idx, ch in enumerate(active_channels):
            start = ch_idx * n_customers
            end = start + n_customers
            c_obj[start:end] = -base_priority * ch.expected_roi_multiplier

        # ------------------------------------------------------------------
        # Variable bounds: 0 <= x_ci <= cost_per_action_i (per channel)
        # ------------------------------------------------------------------
        bounds: List[Tuple[float, float]] = []
        for ch_idx, ch in enumerate(active_channels):
            ch_cost = ch.cost_per_action
            for i in range(n_customers):
                upper = min(cost_per_action[i], ch_cost) if ch_cost > 0 else cost_per_action[i]
                bounds.append((0.0, float(upper)))

        # ------------------------------------------------------------------
        # Inequality constraints (A_ub @ x <= b_ub)
        # ------------------------------------------------------------------
        A_ub_rows: List[np.ndarray] = []
        b_ub_vals: List[float] = []

        # 1. Total budget constraint: sum of all x_ci <= total_budget
        row_total = np.ones(n_vars, dtype=np.float64)
        A_ub_rows.append(row_total)
        b_ub_vals.append(float(total_budget))

        # 2. Per-channel max constraints: sum_i x_ci <= channel_max_c
        for ch_idx, ch in enumerate(active_channels):
            if ch.max_budget is not None:
                row = np.zeros(n_vars, dtype=np.float64)
                start = ch_idx * n_customers
                row[start: start + n_customers] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(float(ch.max_budget))

        # 3. Per-channel min constraints: sum_i x_ci >= channel_min_c
        #    Rewrite as: -sum_i x_ci <= -channel_min_c
        for ch_idx, ch in enumerate(active_channels):
            if ch.min_budget > 0:
                row = np.zeros(n_vars, dtype=np.float64)
                start = ch_idx * n_customers
                row[start: start + n_customers] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(-float(ch.min_budget))

        A_ub = np.array(A_ub_rows, dtype=np.float64) if A_ub_rows else None
        b_ub = np.array(b_ub_vals, dtype=np.float64) if b_ub_vals else None

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        try:
            result: OptimizeResult = linprog(
                c=c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
                options={"disp": False},
            )
        except Exception as exc:
            logger.error("LP solver failed: %s", exc)
            return self._empty_result(data, total_budget, status="error",
                                       message=str(exc))

        if not result.success:
            logger.warning("LP solver did not find optimal: %s", result.message)
            return self._empty_result(
                data, total_budget,
                status="infeasible" if result.status == 2 else "failed",
                message=result.message,
            )

        # ------------------------------------------------------------------
        # Parse solution
        # ------------------------------------------------------------------
        x_opt = result.x
        customer_ids = data["customer_id"].values

        records: List[Dict[str, Any]] = []
        channel_totals: Dict[str, float] = {}

        for ch_idx, ch in enumerate(active_channels):
            start = ch_idx * n_customers
            allocs = x_opt[start: start + n_customers]
            ch_total = float(allocs.sum())
            channel_totals[ch.name] = ch_total

            for i in range(n_customers):
                if allocs[i] > 1e-8:
                    records.append({
                        "customer_id": customer_ids[i],
                        "channel": ch.name,
                        "allocated_budget": float(allocs[i]),
                    })

        # Include zero-allocation customers (aggregate across channels)
        allocated_ids = {r["customer_id"] for r in records}
        for cid in customer_ids:
            if cid not in allocated_ids:
                records.append({
                    "customer_id": cid,
                    "channel": active_channels[0].name,
                    "allocated_budget": 0.0,
                })

        alloc_df = pd.DataFrame(records)

        opt_result = OptimizationResult(
            allocations=alloc_df,
            total_allocated=float(x_opt.sum()),
            total_budget=total_budget,
            objective_value=float(-result.fun),  # negate back
            channel_summary=channel_totals,
            status="optimal",
            solver_message=result.message,
        )
        self.last_result = opt_result
        return opt_result

    def get_customer_allocations(
        self,
        result: Optional[OptimizationResult] = None,
    ) -> pd.DataFrame:
        """Aggregate allocations per customer across all channels.

        Parameters
        ----------
        result : OptimizationResult, optional
            If None, uses last_result.

        Returns
        -------
        pd.DataFrame
            Columns: customer_id, allocated_budget (summed across channels).
        """
        if result is None:
            result = self.last_result
        if result is None:
            raise RuntimeError("No optimisation result available. Call solve() first.")

        agg = (
            result.allocations
            .groupby("customer_id", as_index=False)["allocated_budget"]
            .sum()
        )
        return agg

    def get_channel_summary(
        self,
        result: Optional[OptimizationResult] = None,
    ) -> Dict[str, float]:
        """Return per-channel budget totals from the last solve.

        Parameters
        ----------
        result : OptimizationResult, optional
            If None, uses last_result.

        Returns
        -------
        dict
            Mapping channel_name -> total_budget_allocated.
        """
        if result is None:
            result = self.last_result
        if result is None:
            raise RuntimeError("No optimisation result available. Call solve() first.")
        return dict(result.channel_summary)

    def compute_expected_value(
        self,
        result: OptimizationResult,
        data: pd.DataFrame,
    ) -> float:
        """Compute expected retained CLV from an allocation.

        Parameters
        ----------
        result : OptimizationResult
            Solver output.
        data : pd.DataFrame
            Original customer data with clv, uplift_score, churn_prob,
            cost_per_action columns.

        Returns
        -------
        float
            Expected retained value (KRW).
        """
        agg = self.get_customer_allocations(result)
        merged = agg.merge(data, on="customer_id", how="left")

        cost = merged["cost_per_action"].values.astype(np.float64)
        alloc = merged["allocated_budget"].values.astype(np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(cost > 0, np.minimum(alloc / cost, 1.0), 0.0)

        clv = merged["clv"].values.astype(np.float64)
        uplift = merged["uplift_score"].values.astype(np.float64)
        churn_prob = merged["churn_prob"].values.astype(np.float64)

        retained = np.sum(fraction * np.maximum(uplift, 0) * clv * churn_prob)
        return float(retained)

    # ------------------------------------------------------------------
    # What-If Scenario Analysis
    # ------------------------------------------------------------------

    def simulate_budget_change(
        self,
        data: pd.DataFrame,
        scenario_name: str,
        total_budget: Optional[float] = None,
        cost_multiplier: float = 1.0,
        uplift_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Simulate a budget change and return impact projections.

        Runs the LP solver under modified parameters and computes projected
        churn rates and revenue impact compared to a no-intervention baseline.

        Parameters
        ----------
        data : pd.DataFrame
            Customer data with customer_id, clv, uplift_score, churn_prob,
            cost_per_action columns.
        scenario_name : str
            Human-readable label for this scenario.
        total_budget : float, optional
            Budget constraint (KRW). Defaults to optimizer's configured budget.
        cost_multiplier : float
            Multiplier applied to cost_per_action (1.0 = no change).
        uplift_multiplier : float
            Multiplier applied to uplift_score (1.0 = no change).

        Returns
        -------
        dict
            Scenario result with keys: scenario_name, total_budget,
            total_allocated, retained_value, roi, customers_treated,
            projected_churn_rate, projected_revenue_impact, status.
        """
        if total_budget is None:
            total_budget = self.total_budget

        n = len(data)

        # Edge case: empty data or zero budget
        if n == 0 or total_budget <= 0:
            baseline_churn = float(data["churn_prob"].mean()) if n > 0 else 0.0
            return {
                "scenario_name": scenario_name,
                "total_budget": total_budget,
                "total_allocated": 0.0,
                "retained_value": 0.0,
                "roi": 0.0,
                "customers_treated": 0,
                "projected_churn_rate": baseline_churn,
                "projected_revenue_impact": 0.0,
                "status": "empty",
            }

        # Apply multipliers to create scenario-specific data
        scenario_data = data.copy()
        scenario_data["cost_per_action"] = (
            scenario_data["cost_per_action"] * cost_multiplier
        )
        scenario_data["uplift_score"] = (
            scenario_data["uplift_score"] * uplift_multiplier
        )

        # Run LP optimization
        opt_result = self.solve(
            data=scenario_data,
            total_budget=total_budget,
        )

        # Compute metrics from optimization result
        total_allocated = float(opt_result.total_allocated)
        retained_value = float(opt_result.objective_value)

        # Count customers with non-trivial allocation
        alloc_df = opt_result.allocations
        customer_allocs = (
            alloc_df.groupby("customer_id", as_index=False)["allocated_budget"]
            .sum()
        )
        customers_treated = int((customer_allocs["allocated_budget"] > 1e-8).sum())

        # ROI: retained value per unit spent
        roi = retained_value / total_allocated if total_allocated > 0 else 0.0

        # Project churn rate impact
        projected_churn_rate = self._project_churn_rate(
            data=scenario_data,
            customer_allocs=customer_allocs,
        )

        # Project revenue impact (additional revenue retained vs no intervention)
        projected_revenue_impact = self._project_revenue_impact(
            data=scenario_data,
            customer_allocs=customer_allocs,
        )

        return {
            "scenario_name": scenario_name,
            "total_budget": total_budget,
            "total_allocated": total_allocated,
            "retained_value": retained_value,
            "roi": roi,
            "customers_treated": customers_treated,
            "projected_churn_rate": projected_churn_rate,
            "projected_revenue_impact": projected_revenue_impact,
            "status": opt_result.status,
        }

    def compare_scenarios(
        self,
        data: pd.DataFrame,
        scenarios: Sequence["WhatIfScenario"],
    ) -> pd.DataFrame:
        """Run and compare multiple what-if scenarios side by side.

        Parameters
        ----------
        data : pd.DataFrame
            Customer data (same for all scenarios).
        scenarios : sequence of WhatIfScenario
            Scenario definitions to simulate.

        Returns
        -------
        pd.DataFrame
            One row per scenario with metric columns.
        """
        rows = []
        for scenario in scenarios:
            result = self.simulate_budget_change(
                data=data,
                scenario_name=scenario.name,
                total_budget=scenario.total_budget,
                cost_multiplier=scenario.cost_multiplier,
                uplift_multiplier=scenario.uplift_multiplier,
            )
            rows.append(result)

        return pd.DataFrame(rows)

    def run_budget_sweep(
        self,
        data: pd.DataFrame,
        budget_levels: List[float],
        cost_multiplier: float = 1.0,
        uplift_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        """Sweep across multiple budget levels and collect metrics.

        Parameters
        ----------
        data : pd.DataFrame
            Customer data.
        budget_levels : list of float
            Budget values to test.
        cost_multiplier : float
            Applied to all scenarios.
        uplift_multiplier : float
            Applied to all scenarios.

        Returns
        -------
        pd.DataFrame
            One row per budget level with metric columns.
        """
        scenarios = [
            WhatIfScenario(
                name=f"budget_{int(b / 1_000_000)}M",
                total_budget=b,
                cost_multiplier=cost_multiplier,
                uplift_multiplier=uplift_multiplier,
            )
            for b in budget_levels
        ]
        return self.compare_scenarios(data=data, scenarios=scenarios)

    # ------------------------------------------------------------------
    # Impact projection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_churn_rate(
        data: pd.DataFrame,
        customer_allocs: pd.DataFrame,
    ) -> float:
        """Project the churn rate after intervention.

        For each customer, the intervention reduces their churn probability
        proportionally to the fraction of cost_per_action allocated and their
        uplift score.

        Parameters
        ----------
        data : pd.DataFrame
            Scenario-adjusted customer data.
        customer_allocs : pd.DataFrame
            Per-customer allocated budget (customer_id, allocated_budget).

        Returns
        -------
        float
            Projected population churn rate in [0, 1].
        """
        if len(data) == 0:
            return 0.0

        merged = customer_allocs.merge(data, on="customer_id", how="right")

        churn_prob = merged["churn_prob"].values.astype(np.float64)
        cost = merged["cost_per_action"].values.astype(np.float64)
        uplift = merged["uplift_score"].values.astype(np.float64)

        alloc = merged["allocated_budget"].values.astype(np.float64)
        alloc = np.nan_to_num(alloc, nan=0.0)

        # Fraction of treatment delivered
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(cost > 0, np.minimum(alloc / cost, 1.0), 0.0)

        # Reduce churn_prob by fraction * max(uplift, 0)
        churn_reduction = fraction * np.maximum(uplift, 0.0)
        projected_churn = np.maximum(churn_prob - churn_reduction, 0.0)

        return float(np.mean(projected_churn))

    @staticmethod
    def _project_revenue_impact(
        data: pd.DataFrame,
        customer_allocs: pd.DataFrame,
    ) -> float:
        """Project additional revenue retained vs no intervention.

        Revenue impact = sum over customers of
            fraction_i * max(uplift_i, 0) * clv_i * churn_prob_i

        This is the expected CLV saved by the intervention.

        Parameters
        ----------
        data : pd.DataFrame
            Scenario-adjusted customer data.
        customer_allocs : pd.DataFrame
            Per-customer allocated budget.

        Returns
        -------
        float
            Projected additional revenue retained (KRW).
        """
        if len(data) == 0:
            return 0.0

        merged = customer_allocs.merge(data, on="customer_id", how="right")

        clv = merged["clv"].values.astype(np.float64)
        churn_prob = merged["churn_prob"].values.astype(np.float64)
        uplift = merged["uplift_score"].values.astype(np.float64)
        cost = merged["cost_per_action"].values.astype(np.float64)

        alloc = merged["allocated_budget"].values.astype(np.float64)
        alloc = np.nan_to_num(alloc, nan=0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(cost > 0, np.minimum(alloc / cost, 1.0), 0.0)

        revenue_impact = np.sum(fraction * np.maximum(uplift, 0.0) * clv * churn_prob)
        return float(revenue_impact)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save optimizer configuration and last result to disk.

        Parameters
        ----------
        path : str
            Base path (without extension). Creates ``<path>.json`` and
            optionally ``<path>_allocations.csv``.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state: Dict[str, Any] = {
            "total_budget": self.total_budget,
            "seed": self.seed,
            "channels": [
                {
                    "name": ch.name,
                    "min_budget": ch.min_budget,
                    "max_budget": ch.max_budget,
                    "cost_per_action": ch.cost_per_action,
                    "expected_roi_multiplier": ch.expected_roi_multiplier,
                }
                for ch in self.channels
            ],
        }
        if self.last_result is not None:
            state["last_status"] = self.last_result.status
            state["last_objective"] = self.last_result.objective_value
            state["last_total_allocated"] = self.last_result.total_allocated
            state["channel_summary"] = self.last_result.channel_summary

        with open(f"{path}.json", "w") as f:
            json.dump(state, f, indent=2)

        if self.last_result is not None:
            self.last_result.allocations.to_csv(
                f"{path}_allocations.csv", index=False,
            )

        logger.info("LPBudgetOptimizer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LPBudgetOptimizer":
        """Load optimizer from saved state.

        Parameters
        ----------
        path : str
            Base path used in ``save()`` (without extension).

        Returns
        -------
        LPBudgetOptimizer
            Restored instance (without last_result data — call solve() again).
        """
        with open(f"{path}.json", "r") as f:
            state = json.load(f)

        channels = [
            ChannelConstraint(
                name=ch["name"],
                min_budget=ch.get("min_budget", 0.0),
                max_budget=ch.get("max_budget"),
                cost_per_action=ch.get("cost_per_action", 10_000.0),
                expected_roi_multiplier=ch.get("expected_roi_multiplier", 1.0),
            )
            for ch in state.get("channels", [])
        ]

        config = {
            "budget": {"total_krw": state["total_budget"]},
            "simulation": {"random_seed": state["seed"]},
        }
        instance = cls(config, channels=channels if channels else None)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        data: pd.DataFrame,
        total_budget: float,
        status: str = "empty",
        message: str = "No allocation (empty input or zero budget)",
    ) -> OptimizationResult:
        """Return an empty OptimizationResult."""
        n = len(data)
        if n > 0:
            alloc_df = pd.DataFrame({
                "customer_id": data["customer_id"].values,
                "channel": "default",
                "allocated_budget": np.zeros(n),
            })
        else:
            alloc_df = pd.DataFrame(
                columns=["customer_id", "channel", "allocated_budget"],
            )

        return OptimizationResult(
            allocations=alloc_df,
            total_allocated=0.0,
            total_budget=total_budget,
            objective_value=0.0,
            channel_summary={},
            status=status,
            solver_message=message,
        )


# ---------------------------------------------------------------------------
# CLI / API Entry Points
# ---------------------------------------------------------------------------


def run_optimization(
    data: pd.DataFrame,
    config: Dict[str, Any],
    total_budget: Optional[float] = None,
    channels: Optional[List[ChannelConstraint]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """High-level entry point for budget optimization.

    Ties the LP solver, cost configuration, and validation together
    into a single call suitable for CLI scripts or API handlers.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with required columns: customer_id, clv,
        uplift_score, churn_prob, cost_per_action.
    config : dict
        Parsed YAML configuration dict.
    total_budget : float, optional
        Override total budget.  If None, uses config value.
    channels : list of ChannelConstraint, optional
        Override channel definitions.  If None, derived from config
        or uses a default single channel.
    validate : bool
        Whether to run input validation (default True).

    Returns
    -------
    dict
        Keys: ``result`` (OptimizationResult), ``allocations`` (DataFrame),
        ``channel_summary`` (dict), ``expected_value`` (float),
        ``warnings`` (list of str), ``cost_config`` (CostConfig).
    """
    warnings: List[str] = []

    # Build cost config
    cost_config = CostConfig.from_config(config)

    if total_budget is None:
        total_budget = cost_config.total_budget

    # Resolve channels
    if channels is None:
        channels = cost_config.get_channel_constraints()

    # Validation
    if validate:
        validate_budget(total_budget)
        warnings.extend(validate_dataframe(data))
        warnings.extend(validate_channels(channels))

    # Create optimizer and solve
    optimizer = LPBudgetOptimizer(config, channels=channels)
    opt_result = optimizer.solve(data, total_budget=total_budget)

    # Compute expected value
    expected_value = optimizer.compute_expected_value(opt_result, data)

    # Apply NPV discount if configured
    npv_factor = cost_config.get_npv_factor()
    npv_adjusted_value = expected_value * npv_factor / cost_config.time_horizon_months

    return {
        "result": opt_result,
        "allocations": optimizer.get_customer_allocations(opt_result),
        "channel_summary": opt_result.channel_summary,
        "expected_value": expected_value,
        "npv_adjusted_value": npv_adjusted_value,
        "warnings": warnings,
        "cost_config": cost_config,
        "status": opt_result.status,
    }


def run_whatif(
    data: pd.DataFrame,
    config: Dict[str, Any],
    scenarios: Optional[Sequence[WhatIfScenario]] = None,
    budget_levels: Optional[List[float]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """High-level entry point for what-if scenario analysis.

    Runs either named scenarios or a budget sweep (or both) and
    returns comparison DataFrames.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with required columns.
    config : dict
        Parsed YAML configuration.
    scenarios : sequence of WhatIfScenario, optional
        Named scenarios to compare.
    budget_levels : list of float, optional
        Budget levels for a sweep analysis.
    validate : bool
        Whether to run input validation.

    Returns
    -------
    dict
        Keys: ``scenario_comparison`` (DataFrame or None),
        ``budget_sweep`` (DataFrame or None),
        ``warnings`` (list of str), ``cost_config`` (CostConfig).
    """
    warnings: List[str] = []

    cost_config = CostConfig.from_config(config)

    if validate:
        validate_dataframe(data)

    optimizer = LPBudgetOptimizer(config)

    scenario_comparison = None
    budget_sweep = None

    if scenarios:
        scenario_comparison = optimizer.compare_scenarios(
            data=data, scenarios=scenarios,
        )

    if budget_levels:
        budget_sweep = optimizer.run_budget_sweep(
            data=data, budget_levels=budget_levels,
        )

    if scenarios is None and budget_levels is None:
        # Default: run a 3-level budget sweep
        base = cost_config.total_budget
        default_levels = [
            base * 0.5,
            base,
            base * 1.5,
        ]
        budget_sweep = optimizer.run_budget_sweep(
            data=data, budget_levels=default_levels,
        )

    return {
        "scenario_comparison": scenario_comparison,
        "budget_sweep": budget_sweep,
        "warnings": warnings,
        "cost_config": cost_config,
    }


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration from file path.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def cli_optimize(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point for running budget optimization.

    Parses command-line arguments and runs optimization. Can be used
    as a script entry point or called programmatically.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  If None, reads from sys.argv[1:].

    Returns
    -------
    dict
        Optimization results (same as ``run_optimization``).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Budget Optimization for Retention Campaigns",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to customer data CSV file",
    )
    parser.add_argument(
        "--budget", type=float, default=None,
        help="Total budget override (KRW)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save allocation CSV",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip input validation",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    data = pd.read_csv(args.data)

    result = run_optimization(
        data=data,
        config=config,
        total_budget=args.budget,
        validate=not args.no_validate,
    )

    if args.output:
        result["allocations"].to_csv(args.output, index=False)
        logger.info("Allocations saved to %s", args.output)

    return result


def cli_whatif(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point for what-if scenario analysis.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    dict
        What-if analysis results (same as ``run_whatif``).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="What-If Scenario Analysis for Budget Optimization",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to customer data CSV file",
    )
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=None,
        help="Budget levels for sweep analysis (KRW)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save comparison CSV",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    data = pd.read_csv(args.data)

    result = run_whatif(
        data=data,
        config=config,
        budget_levels=args.budgets,
    )

    if args.output:
        sweep = result.get("budget_sweep")
        if sweep is not None:
            sweep.to_csv(args.output, index=False)
            logger.info("Budget sweep saved to %s", args.output)

    return result
