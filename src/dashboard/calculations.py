"""
Pure calculation helpers for dashboard views.

This module keeps statistical and budget computation logic out of the
Streamlit app module so that app.py only orchestrates rendering.
"""

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


# =========================================================================
# ROI calculation helpers
# =========================================================================
#
# The iter9 audit (a5) flagged that Pages 05/09/12 each compute "Overall
# ROI" against a different denominator (LP budget envelope, cost-actually-
# spent, segment mean) and report 3.5x / 9.0x / 3.8x for the same campaign.
# These helpers are the calculation-layer counterparts to the formatter in
# ``dashboard_helpers.compute_overall_roi`` so every page reaches into the
# same definition. Format conversion (display string, tooltip) stays in
# the helpers module; raw numeric ROI lives here.


def roi_budget_envelope(
    revenue_saved: float,
    total_budget: float,
) -> Optional[float]:
    """ROI scoped to the full LP budget envelope.

    Args:
        revenue_saved: Expected revenue retained by the campaign (KRW).
        total_budget: Allocated budget for the campaign — denominator
            includes unspent budget. Use this on Page 12 (Retention
            Campaign) where the headline is supposed to reflect the
            entire envelope decision, not just the issued offers.

    Returns:
        ``revenue_saved / total_budget`` as a float, or ``None`` when the
        denominator is zero / missing.
    """
    if total_budget is None or total_budget == 0:
        return None
    if revenue_saved is None:
        return None
    try:
        if pd.isna(total_budget) or pd.isna(revenue_saved):
            return None
    except (TypeError, ValueError):
        pass
    return float(revenue_saved) / float(total_budget)


def roi_treated_only(
    revenue_saved: float,
    cost_spent: float,
) -> Optional[float]:
    """ROI scoped to the customers actually treated (spent only).

    Args:
        revenue_saved: Revenue retained from the treated subset (KRW).
        cost_spent: Sum of issued offer costs (excludes unspent budget).
            Use this on Page 09 where the campaign-cost KPI is the
            offer total (e.g. 1,211,055 KRW), not the budget envelope.

    Returns:
        ``revenue_saved / cost_spent`` as a float, or ``None`` when the
        denominator is zero / missing.
    """
    if cost_spent is None or cost_spent == 0:
        return None
    if revenue_saved is None:
        return None
    try:
        if pd.isna(cost_spent) or pd.isna(revenue_saved):
            return None
    except (TypeError, ValueError):
        pass
    return float(revenue_saved) / float(cost_spent)


def roi_segment_average(segment_rois: Iterable[float]) -> Optional[float]:
    """Mean of per-segment ROI values, ignoring excluded (zero-cost) segments.

    Segments with cost == 0 (e.g. ``sleeping_dog``, ``high_value_lost_cause``
    on Page 12) carry ROI = 0.0 by construction; mixing them into the mean
    drags the headline down without reflecting campaign performance. This
    helper drops zero / NaN entries before averaging.

    Args:
        segment_rois: Iterable of per-segment ROI floats.

    Returns:
        Arithmetic mean of non-zero, non-NaN segment ROIs, or ``None`` when
        no qualifying segments are present.
    """
    if segment_rois is None:
        return None
    vals = []
    for v in segment_rois:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN
            continue
        if f == 0.0:
            continue
        vals.append(f)
    if not vals:
        return None
    return sum(vals) / len(vals)


def customers_retained_int(value: Any) -> Optional[int]:
    """Floor a model-derived "customers retained" expectation to a whole int.

    Closes the Page 12 ``Customers Retained = 122.29548658078494`` audit
    finding at the calculation layer: any caller that downstream renders
    a count KPI should pass through this helper so a 14-decimal float can
    never leak into the display layer even if the formatter is bypassed.

    Args:
        value: Numeric value (float or int). ``None``/``NaN`` returns
            ``None`` so the caller can render an em-dash.

    Returns:
        Integer count, or ``None`` if the input is missing.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return int(f)


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


# =========================================================================
# Real-only KPI rendering helpers (iter13)
# =========================================================================


def safe_real_metric(value, fallback_indicator: str = "—") -> str:
    """Last-line-of-defense renderer for ``st.metric()`` values.

    Used by KPI cards across the dashboard to ensure that ``None`` / NaN /
    +/-infinity never leak into the headline number tiles. When the value
    is unusable a typographic em-dash (or the caller-supplied indicator)
    is returned instead, signalling "no real data" to the operator without
    silently rendering a misleading zero.

    Args:
        value: The metric value about to be passed to ``st.metric(value=)``.
            Numeric types are returned as-is when finite; ``None`` / NaN /
            inf are converted to the fallback indicator. Non-numeric values
            (e.g., pre-formatted strings) are returned as-is.
        fallback_indicator: String returned when ``value`` is unusable.
            Defaults to em-dash (``"—"``).

    Returns:
        Either the original value (when safe) or the fallback indicator
        string. Always safe to pass directly to ``st.metric``.
    """
    import math
    if value is None:
        return fallback_indicator
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return fallback_indicator
    return value
