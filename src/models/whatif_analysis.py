"""
What-If Scenario Analysis for Budget Optimization.

Provides functionality to vary budget constraints, cost parameters, and
uplift assumptions, then compare multiple optimization outcomes side by side.

This module wraps the BudgetOptimizer and enables:
- Running individual scenarios with custom parameters
- Comparing multiple scenarios in a single DataFrame
- Budget sweep analysis across different budget levels
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.budget_optimizer import BudgetOptimizer

logger = logging.getLogger(__name__)


class WhatIfAnalyzer:
    """What-if scenario analyzer for retention budget optimization.

    Allows running optimization under different assumptions (budget,
    cost multiplier, uplift multiplier) and comparing results side by side.

    Attributes:
        default_budget: Default total budget from config (KRW).
        config: Full configuration dictionary.
    """

    def __init__(self, config: Dict):
        """Initialize analyzer from YAML config.

        Args:
            config: Parsed simulator_config.yaml dict.
        """
        self.config = config
        self.default_budget = config.get("budget", {}).get(
            "total_krw", 50_000_000
        )
        self.seed = config.get("simulation", {}).get("random_seed", 42)

    def run_scenario(
        self,
        data: pd.DataFrame,
        scenario_name: str,
        total_budget: Optional[float] = None,
        cost_multiplier: float = 1.0,
        uplift_multiplier: float = 1.0,
    ) -> Dict:
        """Run a single what-if scenario.

        Args:
            data: Customer DataFrame with churn_prob, clv, uplift_score,
                  cost_per_action columns.
            scenario_name: Human-readable label for this scenario.
            total_budget: Budget constraint (KRW). Defaults to config value.
            cost_multiplier: Multiplier applied to cost_per_action.
            uplift_multiplier: Multiplier applied to uplift_score.

        Returns:
            Dict with scenario_name, total_budget, total_allocated,
            retained_value, roi, customers_treated, allocation.
        """
        if total_budget is None:
            total_budget = self.default_budget

        # Apply multipliers to create scenario-specific data
        scenario_data = data.copy()
        scenario_data["cost_per_action"] = (
            scenario_data["cost_per_action"] * cost_multiplier
        )
        scenario_data["uplift_score"] = (
            scenario_data["uplift_score"] * uplift_multiplier
        )

        # Run optimization
        optimizer = BudgetOptimizer(self.config)
        allocation = optimizer.optimize(
            data=scenario_data,
            total_budget=total_budget,
        )

        # Compute metrics
        total_allocated = float(allocation["allocated_budget"].sum())
        retained_value = optimizer.compute_roi(
            allocation=allocation,
            data=scenario_data,
        )

        # Count customers with non-trivial allocation
        customers_treated = int(
            (allocation["allocated_budget"] > 0).sum()
        )

        # ROI: retained value per unit spent
        if total_allocated > 0:
            roi = retained_value / total_allocated
        else:
            roi = 0.0

        return {
            "scenario_name": scenario_name,
            "total_budget": total_budget,
            "total_allocated": total_allocated,
            "retained_value": retained_value,
            "roi": roi,
            "customers_treated": customers_treated,
            "allocation": allocation,
        }

    def compare_scenarios(
        self,
        data: pd.DataFrame,
        scenarios: List[Dict],
    ) -> pd.DataFrame:
        """Run and compare multiple scenarios side by side.

        Args:
            data: Customer DataFrame (same for all scenarios).
            scenarios: List of dicts, each with keys:
                - scenario_name (str, required)
                - total_budget (float, optional)
                - cost_multiplier (float, optional, default 1.0)
                - uplift_multiplier (float, optional, default 1.0)

        Returns:
            DataFrame with one row per scenario and metric columns.
        """
        results = []
        for scenario_params in scenarios:
            name = scenario_params["scenario_name"]
            result = self.run_scenario(
                data=data,
                scenario_name=name,
                total_budget=scenario_params.get("total_budget"),
                cost_multiplier=scenario_params.get("cost_multiplier", 1.0),
                uplift_multiplier=scenario_params.get(
                    "uplift_multiplier", 1.0
                ),
            )
            # Exclude allocation DataFrame from comparison table
            row = {k: v for k, v in result.items() if k != "allocation"}
            results.append(row)

        comparison = pd.DataFrame(results)
        return comparison

    def run_budget_sweep(
        self,
        data: pd.DataFrame,
        budget_levels: List[float],
        cost_multiplier: float = 1.0,
        uplift_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        """Sweep across multiple budget levels and collect metrics.

        Args:
            data: Customer DataFrame.
            budget_levels: List of budget values to test.
            cost_multiplier: Applied to all scenarios.
            uplift_multiplier: Applied to all scenarios.

        Returns:
            DataFrame with one row per budget level.
        """
        scenarios = [
            {
                "scenario_name": f"budget_{int(b / 1_000_000)}M",
                "total_budget": b,
                "cost_multiplier": cost_multiplier,
                "uplift_multiplier": uplift_multiplier,
            }
            for b in budget_levels
        ]
        return self.compare_scenarios(data=data, scenarios=scenarios)
