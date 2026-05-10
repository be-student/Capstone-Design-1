"""
Budget Optimization Engine for Customer Retention.

Uses Linear Programming (scipy.optimize.linprog) to allocate retention
campaign budget optimally across customers/segments, maximizing expected
retained CLV subject to a total budget constraint.

The objective function maximizes:
    sum_i (uplift_score_i * clv_i * churn_prob_i * x_i)
where x_i in [0, 1] is the fraction of cost_per_action allocated to customer i.

Subject to:
    sum_i (cost_per_action_i * x_i) <= total_budget
    0 <= x_i <= 1  for all i

Since linprog *minimizes*, we negate the objective coefficients.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog, LinearConstraint

logger = logging.getLogger(__name__)

DEFAULT_WHATIF_MULTIPLIERS = (0.5, 1.0, 2.0)


class BudgetOptimizer:
    """LP-based budget optimizer for retention campaigns.

    Uses scipy.optimize.linprog to solve the constrained budget allocation
    problem. The LP maximises expected retained value (uplift * CLV * churn_prob)
    subject to a total budget constraint and per-customer upper bounds.

    Attributes:
        total_budget: Default total budget from config (KRW).
        seed: Random seed for reproducibility.
        last_result: Cached result from last optimization run.
        channels: List of channel names for multi-channel optimization.
    """

    def __init__(self, config: Dict):
        """Initialize optimizer from YAML config.

        Args:
            config: Parsed simulator_config.yaml dict.
        """
        self.total_budget = config.get("budget", {}).get("total_krw", 50_000_000)
        self.seed = config.get("simulation", {}).get("random_seed", 42)
        self.config = config
        self.last_result: Optional[pd.DataFrame] = None
        self._last_params: Optional[Dict] = None
        self._last_lp_result = None  # Store raw LP result for inspection

    # ------------------------------------------------------------------
    # Core LP optimisation
    # ------------------------------------------------------------------
    def optimize(
        self,
        data: pd.DataFrame,
        total_budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Optimize budget allocation across customers using LP.

        Solves:
            maximize  sum_i  c_i * x_i
            s.t.      sum_i  cost_i * x_i <= total_budget
                      0 <= x_i <= 1   for all i

        where c_i = max(uplift_i, 0) * clv_i * churn_prob_i  (priority score)
        and the allocated budget for customer i is cost_i * x_i.

        Args:
            data: DataFrame with columns: customer_id, churn_prob, clv,
                  uplift_score, cost_per_action, expected_retention_lift.
            total_budget: Total budget constraint (KRW). If None, uses config default.

        Returns:
            DataFrame with columns: customer_id, allocated_budget.
        """
        if total_budget is None:
            total_budget = self.total_budget

        n = len(data)

        if total_budget <= 0 or n == 0:
            result = pd.DataFrame({
                "customer_id": data["customer_id"].values if n > 0 else [],
                "allocated_budget": np.zeros(n),
            })
            self.last_result = result
            return result

        np.random.seed(self.seed)

        # Extract arrays
        clv = data["clv"].values.astype(float)
        uplift = data["uplift_score"].values.astype(float)
        churn_prob = data["churn_prob"].values.astype(float)
        cost_per_action = data["cost_per_action"].values.astype(float)

        # Solve LP
        allocated = self._solve_lp(
            clv=clv,
            uplift=uplift,
            churn_prob=churn_prob,
            cost_per_action=cost_per_action,
            total_budget=total_budget,
        )

        # Store params for save/load
        self._last_params = {
            "total_budget": total_budget,
            "seed": self.seed,
        }

        output = pd.DataFrame({
            "customer_id": data["customer_id"].values,
            "allocated_budget": allocated,
        })
        self.last_result = output
        return output

    def _solve_lp(
        self,
        clv: np.ndarray,
        uplift: np.ndarray,
        churn_prob: np.ndarray,
        cost_per_action: np.ndarray,
        total_budget: float,
    ) -> np.ndarray:
        """Solve the budget allocation LP using scipy.optimize.linprog.

        Decision variables: x_i ≥ 0 — budget allocated to customer i.
        Objective: maximise  Σ  (priority_i / cost_i) * x_i
                 = minimise  Σ -(priority_i / cost_i) * x_i   (linprog minimises)

        where priority_i = max(uplift_i, 0) * clv_i * churn_prob_i

        Constraints:
            Σ x_i  ≤  total_budget                     (budget cap)
            0 ≤ x_i ≤ cost_per_action_i   for all i    (variable bounds)

        This formulation directly allocates budget amounts. The objective
        weights by priority/cost so that each unit of budget goes to the
        customers with the highest expected value per KRW spent.

        Args:
            clv: Customer lifetime values.
            uplift: Uplift scores (may be negative for sleeping dogs).
            churn_prob: Churn probabilities.
            cost_per_action: Per-customer intervention cost.
            total_budget: Total budget constraint.

        Returns:
            Per-customer allocated budget array.
        """
        n = len(clv)

        # Priority score: max(uplift, 0) * clv * churn_prob
        # Negative uplift customers (sleeping dogs) get zero priority
        priority = np.maximum(uplift, 0.0) * clv * churn_prob

        # Efficiency: value per unit cost
        safe_cost = np.where(cost_per_action > 0, cost_per_action, 1.0)
        efficiency = priority / safe_cost

        # Objective coefficients (negate because linprog minimises)
        c = -efficiency

        # Budget constraint: Σ x_i ≤ total_budget
        A_ub = np.ones((1, n))
        b_ub = np.array([total_budget])

        # Variable bounds: 0 ≤ x_i ≤ cost_per_action_i. This keeps the
        # assignment aligned with the requirement's binary Action_i decision:
        # each customer can receive at most one configured intervention.
        positive_priority = priority > 0
        upper = np.where(positive_priority, cost_per_action, 0.0)
        bounds = [(0.0, max(float(u), 0.0)) for u in upper]

        # Solve LP using HiGHS solver (default in modern scipy)
        try:
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
                options={"presolve": True, "time_limit": 60.0},
            )
            self._last_lp_result = result

            if result.success:
                allocated = np.maximum(result.x, 0.0)
                allocated = self._assign_residual_high_value_budget(
                    allocated=allocated,
                    clv=clv,
                    uplift=uplift,
                    churn_prob=churn_prob,
                    cost_per_action=cost_per_action,
                    total_budget=total_budget,
                )
                logger.info(
                    "LP solved successfully: status=%s, total_allocated=%.0f KRW",
                    result.message,
                    allocated.sum(),
                )
                return allocated
            else:
                logger.warning(
                    "LP solver did not converge (%s), falling back to "
                    "proportional allocation.",
                    result.message,
                )
                return self._proportional_fallback(priority, cost_per_action, total_budget)

        except Exception as exc:
            logger.error("LP solver error: %s. Falling back to proportional.", exc)
            return self._proportional_fallback(priority, cost_per_action, total_budget)

    def _assign_residual_high_value_budget(
        self,
        allocated: np.ndarray,
        clv: np.ndarray,
        uplift: np.ndarray,
        churn_prob: np.ndarray,
        cost_per_action: np.ndarray,
        total_budget: float,
    ) -> np.ndarray:
        """Use leftover budget for high-value customers after positive uplift is funded."""
        leftover = float(total_budget - allocated.sum())
        if leftover <= 1e-6 or len(allocated) == 0:
            return allocated
        if not (uplift > 0).any():
            return allocated

        clv_cut = float(np.median(clv))
        uplift_floor = float(np.quantile(uplift, 0.10))
        remaining_capacity = np.maximum(cost_per_action - allocated, 0.0)
        candidates = np.where(
            (remaining_capacity > 0)
            & (clv >= clv_cut)
            & (churn_prob > 0)
            & (uplift >= uplift_floor)
        )[0]
        if len(candidates) == 0:
            return allocated

        ordered = candidates[np.argsort(-clv[candidates])]
        adjusted = allocated.copy()
        for idx in ordered:
            if leftover <= 1e-6:
                break
            add = min(float(remaining_capacity[idx]), leftover)
            adjusted[idx] += add
            leftover -= add
        return adjusted

    # ------------------------------------------------------------------
    # Multi-channel LP optimisation
    # ------------------------------------------------------------------
    def optimize_multi_channel(
        self,
        data: pd.DataFrame,
        channels: List[str],
        channel_costs: Dict[str, float],
        channel_budgets: Optional[Dict[str, float]] = None,
        total_budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Optimize budget allocation across multiple channels using LP.

        Decision variables: x_{i,c} ∈ [0, 1] for customer i, channel c.

        Objective: maximise Σ_i Σ_c  priority_i * effectiveness_{i,c} * x_{i,c}
        Constraints:
            Σ_i Σ_c  channel_cost_c * x_{i,c}  ≤  total_budget   (global budget)
            Σ_i  channel_cost_c * x_{i,c}  ≤  channel_budget_c   (per-channel)
            Σ_c  x_{i,c}  ≤  1   for each customer i             (no double-treat)
            0 ≤ x_{i,c} ≤ 1

        Args:
            data: Customer data with customer_id, churn_prob, clv, uplift_score, cost_per_action.
            channels: List of channel names (e.g. ["email", "sms", "push"]).
            channel_costs: Cost per action for each channel.
            channel_budgets: Optional per-channel budget caps.
            total_budget: Global budget constraint.

        Returns:
            DataFrame with customer_id and allocated_budget per channel.
        """
        if total_budget is None:
            total_budget = self.total_budget

        n = len(data)
        k = len(channels)

        if total_budget <= 0 or n == 0 or k == 0:
            result_dict = {"customer_id": data["customer_id"].values if n > 0 else []}
            for ch in channels:
                result_dict[f"budget_{ch}"] = np.zeros(n)
            result_dict["allocated_budget"] = np.zeros(n)
            return pd.DataFrame(result_dict)

        np.random.seed(self.seed)

        clv = data["clv"].values.astype(float)
        uplift = data["uplift_score"].values.astype(float)
        churn_prob = data["churn_prob"].values.astype(float)

        # Base priority
        priority = np.maximum(uplift, 0.0) * clv * churn_prob

        # Decision variables: x has shape (n * k,) — flattened [customer, channel]
        # Index: i * k + c  →  customer i, channel c

        # Objective: -priority_i for each (i, c) pair
        c_obj = np.zeros(n * k)
        for c_idx, ch in enumerate(channels):
            for i in range(n):
                c_obj[i * k + c_idx] = -priority[i]

        # --- Inequality constraints ---
        A_ub_rows = []
        b_ub_vals = []

        # 1) Global budget: Σ_{i,c} channel_cost_c * x_{i,c} ≤ total_budget
        row_global = np.zeros(n * k)
        for c_idx, ch in enumerate(channels):
            cost_c = channel_costs[ch]
            for i in range(n):
                row_global[i * k + c_idx] = cost_c
        A_ub_rows.append(row_global)
        b_ub_vals.append(total_budget)

        # 2) Per-channel budgets (if provided)
        if channel_budgets:
            for c_idx, ch in enumerate(channels):
                if ch in channel_budgets:
                    row_ch = np.zeros(n * k)
                    cost_c = channel_costs[ch]
                    for i in range(n):
                        row_ch[i * k + c_idx] = cost_c
                    A_ub_rows.append(row_ch)
                    b_ub_vals.append(channel_budgets[ch])

        # 3) Per-customer: Σ_c x_{i,c} ≤ 1  (no over-treatment)
        for i in range(n):
            row_cust = np.zeros(n * k)
            for c_idx in range(k):
                row_cust[i * k + c_idx] = 1.0
            A_ub_rows.append(row_cust)
            b_ub_vals.append(1.0)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)

        bounds = [(0.0, 1.0)] * (n * k)

        try:
            result = linprog(
                c=c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
                options={"presolve": True, "time_limit": 120.0},
            )

            if not result.success:
                logger.warning("Multi-channel LP did not converge: %s", result.message)
                # Fallback: single-channel proportional
                return self._multi_channel_fallback(data, channels, channel_costs, total_budget)

            x_opt = result.x.reshape(n, k) if result.x is not None else np.zeros((n, k))

        except Exception as exc:
            logger.error("Multi-channel LP error: %s", exc)
            return self._multi_channel_fallback(data, channels, channel_costs, total_budget)

        # Build output DataFrame
        result_dict = {"customer_id": data["customer_id"].values}
        total_alloc = np.zeros(n)
        for c_idx, ch in enumerate(channels):
            cost_c = channel_costs[ch]
            ch_alloc = x_opt[:, c_idx] * cost_c
            result_dict[f"budget_{ch}"] = ch_alloc
            total_alloc += ch_alloc
        result_dict["allocated_budget"] = total_alloc

        return pd.DataFrame(result_dict)

    # ------------------------------------------------------------------
    # What-If Scenario Analysis
    # ------------------------------------------------------------------
    def enrich_allocation_metrics(
        self,
        allocation: pd.DataFrame,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Attach dashboard-friendly business metrics to an allocation.

        Returns retained value, expected revenue saved, and ROI from the
        current allocation. ``expected_revenue_saved`` aliases the retained
        value so downstream consumers can render business-facing wording
        without re-computing it.
        """
        total_allocated = float(allocation["allocated_budget"].sum())
        retained_value = self.compute_roi(allocation=allocation, data=data)
        expected_revenue_saved = retained_value
        roi = (
            expected_revenue_saved / total_allocated
            if total_allocated > 0
            else 0.0
        )
        customers_treated = int((allocation["allocated_budget"] > 0).sum())
        return {
            "total_allocated": total_allocated,
            "retained_value": retained_value,
            "expected_revenue_saved": expected_revenue_saved,
            "roi": roi,
            "customers_treated": customers_treated,
        }

    def simulate_scenario(
        self,
        data: pd.DataFrame,
        scenario_name: str = "default",
        total_budget: Optional[float] = None,
        cost_multiplier: float = 1.0,
        uplift_multiplier: float = 1.0,
        churn_multiplier: float = 1.0,
        clv_multiplier: float = 1.0,
    ) -> Dict:
        """Simulate a what-if scenario with parameter variations.

        Applies multipliers to key input columns before running the LP
        optimizer, enabling exploration of hypothetical situations (e.g.
        "what if intervention costs doubled?" or "what if uplift improved
        by 50%?").

        Args:
            data: Customer DataFrame with churn_prob, clv, uplift_score,
                  cost_per_action columns.
            scenario_name: Human-readable label for this scenario.
            total_budget: Budget constraint (KRW). Defaults to config value.
            cost_multiplier: Multiplier applied to cost_per_action.
            uplift_multiplier: Multiplier applied to uplift_score.
            churn_multiplier: Multiplier applied to churn_prob (clamped to [0, 1]).
            clv_multiplier: Multiplier applied to clv.

        Returns:
            Dict with scenario_name, parameters, total_budget, total_allocated,
            retained_value, roi, customers_treated, allocation DataFrame,
            and efficiency (retained value per KRW spent).
        """
        if total_budget is None:
            total_budget = self.total_budget

        # Apply multipliers to create scenario-specific data
        scenario_data = data.copy()
        scenario_data["cost_per_action"] = (
            scenario_data["cost_per_action"] * cost_multiplier
        )
        scenario_data["uplift_score"] = (
            scenario_data["uplift_score"] * uplift_multiplier
        )
        scenario_data["churn_prob"] = np.clip(
            scenario_data["churn_prob"] * churn_multiplier, 0.0, 1.0
        )
        scenario_data["clv"] = scenario_data["clv"] * clv_multiplier

        # Run optimization
        allocation = self.optimize(
            data=scenario_data,
            total_budget=total_budget,
        )

        metrics = self.enrich_allocation_metrics(
            allocation=allocation,
            data=scenario_data,
        )

        return {
            "scenario_name": scenario_name,
            "parameters": {
                "total_budget": total_budget,
                "cost_multiplier": cost_multiplier,
                "uplift_multiplier": uplift_multiplier,
                "churn_multiplier": churn_multiplier,
                "clv_multiplier": clv_multiplier,
            },
            "total_budget": total_budget,
            **metrics,
            "allocation": allocation,
        }

    def vary_parameter(
        self,
        data: pd.DataFrame,
        parameter: str,
        values: List[float],
        base_budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run scenarios varying a single parameter across a range of values.

        Performs sensitivity analysis by holding all other parameters fixed
        and sweeping one parameter (budget, cost_multiplier, uplift_multiplier,
        churn_multiplier, or clv_multiplier) through the given values.

        Args:
            data: Customer DataFrame.
            parameter: One of 'budget', 'cost_multiplier', 'uplift_multiplier',
                       'churn_multiplier', 'clv_multiplier'.
            values: List of values to sweep for the parameter.
            base_budget: Base budget to use (ignored when parameter='budget').

        Returns:
            DataFrame with one row per parameter value, columns:
            parameter_value, total_allocated, retained_value, roi,
            customers_treated.

        Raises:
            ValueError: If parameter is not a recognised name.
        """
        valid_params = {
            "budget", "cost_multiplier", "uplift_multiplier",
            "churn_multiplier", "clv_multiplier",
        }
        if parameter not in valid_params:
            raise ValueError(
                f"Unknown parameter '{parameter}'. Must be one of {valid_params}"
            )

        rows: List[Dict] = []
        for val in values:
            kwargs: Dict = {
                "data": data,
                "scenario_name": f"{parameter}={val}",
            }

            if parameter == "budget":
                kwargs["total_budget"] = val
            else:
                kwargs["total_budget"] = base_budget
                kwargs[parameter] = val

            result = self.simulate_scenario(**kwargs)
            rows.append({
                "parameter_value": val,
                "total_allocated": result["total_allocated"],
                "retained_value": result["retained_value"],
                "roi": result["roi"],
                "customers_treated": result["customers_treated"],
            })

        return pd.DataFrame(rows)

    def compare_strategies(
        self,
        data: pd.DataFrame,
        total_budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compare different budget allocation strategies on the same data.

        Runs three strategies side by side:
        - **lp**: LP-optimised allocation (the default optimizer).
        - **proportional**: Priority-weighted proportional allocation.
        - **uniform**: Equal allocation across all eligible customers.

        Args:
            data: Customer DataFrame with required columns.
            total_budget: Budget constraint (KRW). Defaults to config value.

        Returns:
            DataFrame with one row per strategy and metric columns:
            strategy, total_allocated, retained_value, roi, customers_treated.
        """
        if total_budget is None:
            total_budget = self.total_budget

        n = len(data)
        clv = data["clv"].values.astype(float)
        uplift = data["uplift_score"].values.astype(float)
        churn_prob = data["churn_prob"].values.astype(float)
        cost_per_action = data["cost_per_action"].values.astype(float)
        priority = np.maximum(uplift, 0.0) * clv * churn_prob

        strategies: List[Dict] = []

        # 1) LP strategy
        lp_alloc = self.optimize(data=data, total_budget=total_budget)
        lp_rv = self.compute_roi(lp_alloc, data)
        lp_total = float(lp_alloc["allocated_budget"].sum())
        strategies.append({
            "strategy": "lp",
            "total_allocated": lp_total,
            "retained_value": lp_rv,
            "roi": lp_rv / lp_total if lp_total > 0 else 0.0,
            "customers_treated": int((lp_alloc["allocated_budget"] > 0).sum()),
        })

        # 2) Proportional strategy
        prop_alloc_arr = self._proportional_fallback(
            priority, cost_per_action, total_budget,
        )
        prop_df = pd.DataFrame({
            "customer_id": data["customer_id"].values,
            "allocated_budget": prop_alloc_arr,
        })
        prop_rv = self.compute_roi(prop_df, data)
        prop_total = float(prop_alloc_arr.sum())
        strategies.append({
            "strategy": "proportional",
            "total_allocated": prop_total,
            "retained_value": prop_rv,
            "roi": prop_rv / prop_total if prop_total > 0 else 0.0,
            "customers_treated": int((prop_alloc_arr > 0).sum()),
        })

        # 3) Uniform strategy — split budget equally among eligible customers
        eligible_mask = priority > 0
        n_eligible = int(eligible_mask.sum())
        uniform_arr = np.zeros(n)
        if n_eligible > 0 and total_budget > 0:
            uniform_arr[eligible_mask] = total_budget / n_eligible
        uniform_df = pd.DataFrame({
            "customer_id": data["customer_id"].values,
            "allocated_budget": uniform_arr,
        })
        uniform_rv = self.compute_roi(uniform_df, data)
        uniform_total = float(uniform_arr.sum())
        strategies.append({
            "strategy": "uniform",
            "total_allocated": uniform_total,
            "retained_value": uniform_rv,
            "roi": uniform_rv / uniform_total if uniform_total > 0 else 0.0,
            "customers_treated": n_eligible,
        })

        return pd.DataFrame(strategies)

    def compare_budget_scenarios(
        self,
        data: pd.DataFrame,
        scenarios: List[Dict],
    ) -> pd.DataFrame:
        """Run and compare multiple budget allocation scenarios.

        A convenience method that accepts a list of scenario definitions
        (each a dict of keyword arguments to ``simulate_scenario``) and
        returns a comparison DataFrame.

        Args:
            data: Customer DataFrame.
            scenarios: List of dicts, each with optional keys:
                scenario_name, total_budget, cost_multiplier,
                uplift_multiplier, churn_multiplier, clv_multiplier.

        Returns:
            DataFrame with one row per scenario and metric columns.
        """
        rows: List[Dict] = []
        for i, params in enumerate(scenarios):
            name = params.get("scenario_name", f"scenario_{i}")
            result = self.simulate_scenario(
                data=data,
                scenario_name=name,
                total_budget=params.get("total_budget"),
                cost_multiplier=params.get("cost_multiplier", 1.0),
                uplift_multiplier=params.get("uplift_multiplier", 1.0),
                churn_multiplier=params.get("churn_multiplier", 1.0),
                clv_multiplier=params.get("clv_multiplier", 1.0),
            )
            rows.append({
                "scenario_name": name,
                "total_budget": result["total_budget"],
                "total_allocated": result["total_allocated"],
                "retained_value": result["retained_value"],
                "roi": result["roi"],
                "customers_treated": result["customers_treated"],
            })

        return pd.DataFrame(rows)

    def run_budget_sweep(
        self,
        data: pd.DataFrame,
        budget_levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Run the standard budget what-if sweep.

        Defaults to 50/100/200% of the configured budget to match the
        dashboard and requirements document.
        """
        if budget_levels is None:
            budget_levels = [
                self.total_budget * multiplier
                for multiplier in DEFAULT_WHATIF_MULTIPLIERS
            ]
        return self.vary_parameter(
            data=data,
            parameter="budget",
            values=budget_levels,
        )

    # ------------------------------------------------------------------
    # Alias
    # ------------------------------------------------------------------
    def allocate(
        self,
        data: pd.DataFrame,
        total_budget: Optional[float] = None,
    ) -> pd.DataFrame:
        """Per-customer budget allocation (alias for optimize).

        Args:
            data: Customer data with scoring columns.
            total_budget: Budget constraint in KRW.

        Returns:
            DataFrame with customer_id and allocated_budget.
        """
        return self.optimize(data=data, total_budget=total_budget)

    # ------------------------------------------------------------------
    # ROI computation
    # ------------------------------------------------------------------
    def compute_roi(
        self,
        allocation: pd.DataFrame,
        data: pd.DataFrame,
    ) -> float:
        """Compute expected absolute retained value of an allocation.

        retained_value = sum_i(fraction_i * max(uplift_i, 0) * clv_i * churn_i)

        Higher budget should yield monotonically non-decreasing retained value
        (with diminishing returns), making this suitable for budget comparison.

        Args:
            allocation: DataFrame with customer_id, allocated_budget.
            data: Original customer data with clv, uplift_score, cost_per_action.

        Returns:
            Expected absolute retained value as a float (KRW).
        """
        merged = allocation.merge(data, on="customer_id", how="left")

        cost = merged["cost_per_action"].values.astype(float)
        alloc = merged["allocated_budget"].values.astype(float)

        # Fraction of treatment applied
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(cost > 0, alloc / cost, 0.0)
        fraction = np.clip(fraction, 0.0, 1.0)

        clv = merged["clv"].values.astype(float)
        uplift = merged["uplift_score"].values.astype(float)
        churn_prob = merged["churn_prob"].values.astype(float)

        # Expected retained value = fraction * uplift * clv * churn_prob
        retained_value = np.sum(fraction * np.maximum(uplift, 0) * clv * churn_prob)

        return float(retained_value)

    # ------------------------------------------------------------------
    # LP diagnostics
    # ------------------------------------------------------------------
    def get_lp_diagnostics(self) -> Optional[Dict]:
        """Return diagnostics from the last LP solve.

        Returns:
            Dict with solver status, objective value, and summary, or None.
        """
        if self._last_lp_result is None:
            return None
        res = self._last_lp_result
        return {
            "success": res.success,
            "status": res.status,
            "message": res.message,
            "objective_value": float(-res.fun) if res.fun is not None else None,
            "n_iterations": getattr(res, "nit", None),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save optimizer state to disk.

        Args:
            path: Base path for saved files (without extension).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "total_budget": self.total_budget,
            "seed": self.seed,
            "config_budget": self.config.get("budget", {}),
        }
        if self._last_params is not None:
            state["last_params"] = self._last_params

        state_file = f"{path}.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        if self.last_result is not None:
            self.last_result.to_csv(f"{path}_allocation.csv", index=False)

        logger.info("Budget optimizer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BudgetOptimizer":
        """Load optimizer from saved state.

        Args:
            path: Base path used in save() (without extension).

        Returns:
            Restored BudgetOptimizer instance.
        """
        state_file = f"{path}.json"
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        config = {
            "budget": state.get("config_budget", {"total_krw": state["total_budget"]}),
            "simulation": {"random_seed": state["seed"]},
        }
        optimizer = cls(config)

        allocation_file = f"{path}_allocation.csv"
        if Path(allocation_file).exists():
            optimizer.last_result = pd.read_csv(allocation_file)

        return optimizer

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _proportional_fallback(
        priority: np.ndarray,
        cost: np.ndarray,
        total_budget: float,
    ) -> np.ndarray:
        """Allocate budget proportionally when LP fails.

        Args:
            priority: Non-negative priority scores per customer.
            cost: Cost per action per customer.
            total_budget: Total budget constraint.

        Returns:
            Allocated budget per customer.
        """
        n = len(priority)
        allocated = np.zeros(n)
        total_score = priority.sum()

        if total_score <= 0 or total_budget <= 0:
            return allocated

        # Pure proportional allocation based on priority scores
        allocated = (priority / total_score) * total_budget

        return allocated

    @staticmethod
    def _multi_channel_fallback(
        data: pd.DataFrame,
        channels: List[str],
        channel_costs: Dict[str, float],
        total_budget: float,
    ) -> pd.DataFrame:
        """Fallback: split budget equally across channels, then proportionally."""
        n = len(data)
        k = len(channels)
        per_channel_budget = total_budget / max(k, 1)

        clv = data["clv"].values.astype(float)
        uplift = data["uplift_score"].values.astype(float)
        churn_prob = data["churn_prob"].values.astype(float)
        priority = np.maximum(uplift, 0.0) * clv * churn_prob
        total_score = priority.sum()

        result_dict = {"customer_id": data["customer_id"].values}
        total_alloc = np.zeros(n)
        for ch in channels:
            if total_score > 0:
                ch_alloc = (priority / total_score) * per_channel_budget
            else:
                ch_alloc = np.zeros(n)
            result_dict[f"budget_{ch}"] = ch_alloc
            total_alloc += ch_alloc
        result_dict["allocated_budget"] = total_alloc

        return pd.DataFrame(result_dict)
