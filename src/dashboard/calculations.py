"""
Pure calculation helpers for dashboard views.

This module keeps statistical and budget computation logic out of the
Streamlit app module so that app.py only orchestrates rendering.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def _compute_power_analysis(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_enrollment: int = 100,
) -> Dict[str, Any]:
    """Compute required sample size using normal approximation."""
    from scipy import stats

    p1 = baseline_rate
    p2 = baseline_rate - mde
    p2 = max(p2, 0.001)

    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p1 - p2) ** 2

    n_per_group = int(np.ceil(numerator / denominator))
    total = n_per_group * 2
    duration = int(np.ceil(total / max(daily_enrollment, 1)))

    return {
        "sample_size_per_group": n_per_group,
        "total_participants": total,
        "estimated_duration_days": duration,
    }


def _compute_power_curve(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    max_n: int = 5000,
    steps: int = 50,
) -> pd.DataFrame:
    """Compute power at different sample sizes."""
    from scipy import stats

    p1 = baseline_rate
    p2 = max(baseline_rate - mde, 0.001)
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    sample_sizes = np.linspace(10, max(max_n, 100), steps).astype(int)
    powers = []
    for n in sample_sizes:
        se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        if se > 0:
            z = abs(p1 - p2) / se - z_alpha
            pw = float(stats.norm.cdf(z))
        else:
            pw = 1.0
        powers.append(min(pw, 1.0))

    return pd.DataFrame({"n": sample_sizes, "power": powers})


def _compute_mde_sensitivity(
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """Compute sample sizes for different MDE values."""
    mde_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    rows = []
    for m in mde_values:
        if m >= baseline_rate:
            continue
        result = _compute_power_analysis(
            baseline_rate=baseline_rate,
            mde=m,
            alpha=alpha,
            power=power,
        )
        rows.append({
            "MDE": m,
            "Sample Size (per group)": result["sample_size_per_group"],
            "Total Participants": result["total_participants"],
        })
    return pd.DataFrame(rows)


def _compute_multiple_comparison_corrections(
    p_values: list,
    experiment_names: list,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply multiple comparison correction methods."""
    n = len(p_values)
    bonferroni = [min(p * n, 1.0) for p in p_values]

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    holm = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        holm[orig_idx] = min(p * (n - rank), 1.0)

    bh = [0.0] * n
    indexed_rev = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    min_val = 1.0
    for rank_rev, (orig_idx, p) in enumerate(indexed_rev):
        actual_rank = n - rank_rev
        adjusted = min(p * n / actual_rank, 1.0)
        min_val = min(min_val, adjusted)
        bh[orig_idx] = min_val

    rows = []
    for i in range(n):
        rows.append({
            "Experiment": experiment_names[i],
            "Raw p-value": p_values[i],
            "Bonferroni": bonferroni[i],
            "Holm-Bonferroni": holm[i],
            "BH (FDR)": bh[i],
            "Significant (Bonferroni)": "Yes" if bonferroni[i] < alpha else "No",
            "Significant (BH)": "Yes" if bh[i] < alpha else "No",
        })
    return pd.DataFrame(rows)


def _build_channel_allocation_data(
    budget_results: pd.DataFrame,
    channel_config: Dict,
    total_budget: float,
) -> pd.DataFrame:
    """Build channel-level allocation data from config."""
    _ = budget_results
    rows = []
    total_weight = sum(
        ch.get("roi_multiplier", 1.0)
        for ch in channel_config.values()
    )

    for ch_name, ch_conf in channel_config.items():
        cost_per_action = ch_conf.get("cost_per_action", 1000)
        roi_mult = ch_conf.get("roi_multiplier", 1.0)
        weight = roi_mult / max(total_weight, 0.01)
        alloc = int(total_budget * weight)
        expected_actions = int(alloc / max(cost_per_action, 1))

        rows.append({
            "channel": ch_name,
            "cost_per_action": cost_per_action,
            "roi_multiplier": roi_mult,
            "allocated_budget": alloc,
            "expected_actions": expected_actions,
        })

    return pd.DataFrame(rows)


def _build_whatif_scenarios(
    default_budget: float,
    current_budget: float,
    cost_multiplier: float,
    uplift_multiplier: float,
) -> list:
    """Build what-if scenarios for budget optimization comparison."""
    return [
        {
            "name": "Baseline",
            "budget": default_budget,
            "cost_mult": 1.0,
            "uplift_mult": 1.0,
        },
        {
            "name": "Current Selection",
            "budget": current_budget,
            "cost_mult": cost_multiplier,
            "uplift_mult": uplift_multiplier,
        },
        {
            "name": "Conservative (-30%)",
            "budget": default_budget * 0.7,
            "cost_mult": 1.0,
            "uplift_mult": 0.8,
        },
        {
            "name": "Aggressive (+50%)",
            "budget": default_budget * 1.5,
            "cost_mult": 1.0,
            "uplift_mult": 1.2,
        },
        {
            "name": "Cost Reduction",
            "budget": default_budget,
            "cost_mult": 0.7,
            "uplift_mult": 1.0,
        },
    ]


def _compute_scenario_comparison(
    budget_results: pd.DataFrame,
    baseline_total: float,
    scenarios: list,
) -> pd.DataFrame:
    """Compute comparison metrics for each budget what-if scenario."""
    rows = []
    for sc in scenarios:
        budget = sc["budget"]
        cost_m = sc["cost_mult"]
        uplift_m = sc["uplift_mult"]

        if baseline_total > 0:
            scale = budget / baseline_total
        else:
            scale = 1.0

        alloc = (budget_results["allocated_budget_krw"] * scale).sum()
        retained = int(
            (budget_results["expected_retained"] * scale * uplift_m).sum()
        )
        rev = int(
            (budget_results["expected_revenue_saved_krw"]
             * scale * uplift_m).sum()
        )
        avg_roi = float(
            (budget_results["roi"] * uplift_m / max(cost_m, 0.01)).mean()
        )

        rows.append({
            "Scenario": sc["name"],
            "Budget (KRW)": budget,
            "Total Allocated": alloc,
            "Expected Retained": retained,
            "Revenue Saved": rev,
            "Avg ROI": avg_roi,
        })

    return pd.DataFrame(rows)


def _compute_budget_sweep(
    budget_results: pd.DataFrame,
    baseline_total: float,
    min_budget: float,
    max_budget: float,
    steps: int,
    cost_multiplier: float = 1.0,
    uplift_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Compute a budget sweep analysis."""
    _ = cost_multiplier
    budgets = np.linspace(min_budget, max_budget, steps)
    rows = []
    for b in budgets:
        if baseline_total > 0:
            scale = b / baseline_total
        else:
            scale = 1.0

        retained = int(
            (budget_results["expected_retained"]
             * scale * uplift_multiplier).sum()
        )
        rev = int(
            (budget_results["expected_revenue_saved_krw"]
             * scale * uplift_multiplier).sum()
        )
        rows.append({
            "Budget": b,
            "Retained": retained,
            "Revenue Saved": rev,
        })

    return pd.DataFrame(rows)
