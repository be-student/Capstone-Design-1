"""
Cohort Analysis Module for E-Commerce Churn Prediction.

Provides time-based and behavior-based cohort assignment, retention matrix
computation, and cohort-level metric aggregation for understanding customer
lifecycle patterns.

Cohort Types:
    1. Monthly Acquisition Cohort - Groups by first purchase/signup month
    2. Weekly Acquisition Cohort - Groups by first purchase/signup week
    3. Behavioral Cohort - Groups by initial behavior patterns (e.g., channel, segment)

Key Metrics:
    - Retention Rate: % of cohort active in each subsequent period
    - Revenue per Cohort: Aggregated monetary value over time
    - Churn Rate per Cohort: Proportion churned by period
    - Average Order Value per Cohort: Mean transaction value over time

Usage:
    analyzer = CohortAnalyzer(config=cohort_config)
    cohorts = analyzer.assign_cohorts(events_df, cohort_type="monthly")
    retention = analyzer.compute_retention_matrix(cohorts)
    metrics = analyzer.compute_cohort_metrics(cohorts)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# Default cohort analysis configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "cohort_type": "monthly",           # "monthly", "weekly", or "behavioral"
    "periods": 12,                      # Number of periods to track
    "min_cohort_size": 5,               # Minimum customers per cohort
    "behavioral_column": "segment",     # Column for behavioral cohort grouping
    "date_column": "event_date",        # Column containing event timestamps
    "customer_column": "customer_id",   # Column containing customer identifier
    "revenue_column": "revenue",        # Column containing revenue/monetary value
    "metrics": [                        # Metrics to compute
        "retention_rate",
        "revenue",
        "avg_order_value",
        "churn_rate",
        "customer_count",
    ],
}


class CohortAnalyzer:
    """Cohort analysis for customer lifecycle and retention tracking.

    Supports monthly, weekly, and behavioral cohort assignment with
    configurable retention matrix computation and metric aggregation.

    Attributes:
        config: Cohort analysis configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the cohort analyzer.

        Args:
            config: Cohort analysis configuration. If None, uses defaults.
                Expected keys: cohort_type, periods, min_cohort_size,
                behavioral_column, date_column, customer_column,
                revenue_column, metrics.
        """
        self.config: Dict[str, Any] = {**DEFAULT_CONFIG}
        if config:
            self.config.update(config)

    # ------------------------------------------------------------------
    # Cohort Assignment
    # ------------------------------------------------------------------

    def assign_cohorts(
        self,
        data: pd.DataFrame,
        cohort_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Assign each customer to a cohort based on their first activity.

        Args:
            data: DataFrame with at least customer_column and date_column.
                For behavioral cohorts, also requires behavioral_column.
            cohort_type: Override cohort type ("monthly", "weekly", or
                "behavioral"). If None, uses config default.

        Returns:
            DataFrame with original columns plus:
                - cohort: The assigned cohort label
                - cohort_period: The period index relative to cohort start
        """
        cohort_type = cohort_type or self.config["cohort_type"]
        customer_col = self.config["customer_column"]
        date_col = self.config["date_column"]

        result = data.copy()

        # Ensure date column is datetime
        if date_col in result.columns:
            result[date_col] = pd.to_datetime(result[date_col])

        if cohort_type == "monthly":
            result = self._assign_monthly_cohorts(result, customer_col, date_col)
        elif cohort_type == "weekly":
            result = self._assign_weekly_cohorts(result, customer_col, date_col)
        elif cohort_type == "behavioral":
            result = self._assign_behavioral_cohorts(result, customer_col)
        else:
            raise ValueError(
                f"Unknown cohort_type: {cohort_type}. "
                "Expected 'monthly', 'weekly', or 'behavioral'."
            )

        return result

    def _assign_monthly_cohorts(
        self,
        data: pd.DataFrame,
        customer_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Assign cohorts by the month of each customer's first event.

        Args:
            data: DataFrame with customer and date columns.
            customer_col: Name of customer identifier column.
            date_col: Name of date column.

        Returns:
            DataFrame with cohort and cohort_period columns added.
        """
        # Find each customer's first event date
        first_event = (
            data.groupby(customer_col)[date_col]
            .min()
            .reset_index()
            .rename(columns={date_col: "first_event_date"})
        )

        result = data.merge(first_event, on=customer_col, how="left")

        # Cohort = year-month of first event
        result["cohort"] = result["first_event_date"].dt.to_period("M").astype(str)

        # Period index = months since cohort start
        result["cohort_period"] = (
            result[date_col].dt.to_period("M").astype("int64")
            - result["first_event_date"].dt.to_period("M").astype("int64")
        )

        result = result.drop(columns=["first_event_date"])
        return result

    def _assign_weekly_cohorts(
        self,
        data: pd.DataFrame,
        customer_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Assign cohorts by the week of each customer's first event.

        Args:
            data: DataFrame with customer and date columns.
            customer_col: Name of customer identifier column.
            date_col: Name of date column.

        Returns:
            DataFrame with cohort and cohort_period columns added.
        """
        first_event = (
            data.groupby(customer_col)[date_col]
            .min()
            .reset_index()
            .rename(columns={date_col: "first_event_date"})
        )

        result = data.merge(first_event, on=customer_col, how="left")

        # Cohort = year-week of first event (ISO week)
        result["cohort"] = (
            result["first_event_date"].dt.isocalendar().year.astype(str)
            + "-W"
            + result["first_event_date"]
            .dt.isocalendar()
            .week.astype(str)
            .str.zfill(2)
        )

        # Period index = weeks since cohort start
        result["cohort_period"] = (
            (result[date_col] - result["first_event_date"]).dt.days // 7
        )

        result = result.drop(columns=["first_event_date"])
        return result

    def _assign_behavioral_cohorts(
        self,
        data: pd.DataFrame,
        customer_col: str,
    ) -> pd.DataFrame:
        """Assign cohorts based on a behavioral/categorical column.

        Uses the first observed value of the behavioral column for each
        customer as their cohort assignment.

        Args:
            data: DataFrame with customer and behavioral columns.
            customer_col: Name of customer identifier column.

        Returns:
            DataFrame with cohort column added. cohort_period is set to 0
            for behavioral cohorts (no temporal ordering).
        """
        behavioral_col = self.config["behavioral_column"]
        date_col = self.config["date_column"]

        if behavioral_col not in data.columns:
            raise ValueError(
                f"Behavioral column '{behavioral_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        result = data.copy()

        # Use first observed behavioral value per customer
        if date_col in result.columns:
            sorted_data = result.sort_values(date_col)
            first_behavior = (
                sorted_data.groupby(customer_col)[behavioral_col]
                .first()
                .reset_index()
                .rename(columns={behavioral_col: "cohort"})
            )
        else:
            first_behavior = (
                result.groupby(customer_col)[behavioral_col]
                .first()
                .reset_index()
                .rename(columns={behavioral_col: "cohort"})
            )

        result = result.merge(first_behavior, on=customer_col, how="left")

        # For behavioral cohorts, compute period from date if available
        if date_col in result.columns:
            first_event = (
                result.groupby(customer_col)[date_col]
                .min()
                .reset_index()
                .rename(columns={date_col: "first_event_date"})
            )
            result = result.merge(first_event, on=customer_col, how="left")
            result["cohort_period"] = (
                result[date_col].dt.to_period("M").astype("int64")
                - result["first_event_date"].dt.to_period("M").astype("int64")
            )
            result = result.drop(columns=["first_event_date"])
        else:
            result["cohort_period"] = 0

        return result

    # ------------------------------------------------------------------
    # Retention Matrix
    # ------------------------------------------------------------------

    def compute_retention_matrix(
        self,
        cohort_data: pd.DataFrame,
        max_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute a retention matrix from cohort-assigned data.

        The retention matrix shows the percentage of each cohort's customers
        that remain active in each subsequent period.

        Args:
            cohort_data: DataFrame with cohort and cohort_period columns
                (output of assign_cohorts).
            max_periods: Maximum number of periods to include. If None,
                uses config['periods'].

        Returns:
            DataFrame with cohorts as rows, period indices as columns,
            and retention rates (0.0 to 1.0) as values.
        """
        max_periods = max_periods or self.config["periods"]
        customer_col = self.config["customer_column"]
        min_size = self.config["min_cohort_size"]

        # Count unique customers per cohort per period
        cohort_counts = (
            cohort_data.groupby(["cohort", "cohort_period"])[customer_col]
            .nunique()
            .reset_index()
            .rename(columns={customer_col: "customers"})
        )

        # Pivot to matrix form
        retention_pivot = cohort_counts.pivot_table(
            index="cohort",
            columns="cohort_period",
            values="customers",
            aggfunc="sum",
            fill_value=0,
        )

        # Filter to max_periods
        valid_cols = [c for c in retention_pivot.columns if c <= max_periods]
        retention_pivot = retention_pivot[sorted(valid_cols)]

        # Filter out small cohorts
        if 0 in retention_pivot.columns:
            retention_pivot = retention_pivot[
                retention_pivot[0] >= min_size
            ]

        # Convert to retention rate (divide by period 0 count)
        if 0 in retention_pivot.columns:
            base_counts = retention_pivot[0]
            retention_matrix = retention_pivot.div(base_counts, axis=0)
        else:
            retention_matrix = retention_pivot

        return retention_matrix

    # ------------------------------------------------------------------
    # Cohort Metrics Aggregation
    # ------------------------------------------------------------------

    def compute_cohort_metrics(
        self,
        cohort_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute multiple metrics aggregated by cohort and period.

        Args:
            cohort_data: DataFrame with cohort, cohort_period, and
                relevant value columns (output of assign_cohorts).
            metrics: List of metric names to compute. If None, uses
                config['metrics']. Supported metrics:
                - "retention_rate": Customer retention per period
                - "revenue": Total revenue per cohort per period
                - "avg_order_value": Mean revenue per cohort per period
                - "churn_rate": 1 - retention_rate
                - "customer_count": Unique customers per cohort per period

        Returns:
            Dictionary mapping metric names to DataFrames with cohorts
            as rows and periods as columns.
        """
        metrics = metrics or self.config.get("metrics", [])
        customer_col = self.config["customer_column"]
        revenue_col = self.config["revenue_column"]
        max_periods = self.config["periods"]
        min_size = self.config["min_cohort_size"]

        results: Dict[str, pd.DataFrame] = {}

        for metric in metrics:
            if metric == "retention_rate":
                results["retention_rate"] = self.compute_retention_matrix(
                    cohort_data, max_periods=max_periods
                )

            elif metric == "churn_rate":
                retention = self.compute_retention_matrix(
                    cohort_data, max_periods=max_periods
                )
                results["churn_rate"] = 1.0 - retention

            elif metric == "revenue":
                if revenue_col not in cohort_data.columns:
                    continue
                rev_agg = (
                    cohort_data.groupby(["cohort", "cohort_period"])[revenue_col]
                    .sum()
                    .reset_index()
                )
                rev_pivot = rev_agg.pivot_table(
                    index="cohort",
                    columns="cohort_period",
                    values=revenue_col,
                    aggfunc="sum",
                    fill_value=0,
                )
                valid_cols = [c for c in rev_pivot.columns if c <= max_periods]
                results["revenue"] = rev_pivot[sorted(valid_cols)]

            elif metric == "avg_order_value":
                if revenue_col not in cohort_data.columns:
                    continue
                aov_agg = (
                    cohort_data.groupby(["cohort", "cohort_period"])[revenue_col]
                    .mean()
                    .reset_index()
                )
                aov_pivot = aov_agg.pivot_table(
                    index="cohort",
                    columns="cohort_period",
                    values=revenue_col,
                    aggfunc="mean",
                    fill_value=0,
                )
                valid_cols = [c for c in aov_pivot.columns if c <= max_periods]
                results["avg_order_value"] = aov_pivot[sorted(valid_cols)]

            elif metric == "customer_count":
                count_agg = (
                    cohort_data.groupby(["cohort", "cohort_period"])[customer_col]
                    .nunique()
                    .reset_index()
                    .rename(columns={customer_col: "count"})
                )
                count_pivot = count_agg.pivot_table(
                    index="cohort",
                    columns="cohort_period",
                    values="count",
                    aggfunc="sum",
                    fill_value=0,
                )
                valid_cols = [c for c in count_pivot.columns if c <= max_periods]
                results["customer_count"] = count_pivot[sorted(valid_cols)]

        return results

    # ------------------------------------------------------------------
    # Summary & Utility Methods
    # ------------------------------------------------------------------

    def get_cohort_summary(
        self,
        cohort_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate a summary of each cohort.

        Args:
            cohort_data: DataFrame with cohort assignments.

        Returns:
            DataFrame with one row per cohort containing:
                - cohort: Cohort label
                - total_customers: Unique customer count
                - total_events: Number of events
                - avg_lifetime_periods: Mean number of active periods
                - total_revenue: Sum of revenue (if available)
        """
        customer_col = self.config["customer_column"]
        revenue_col = self.config["revenue_column"]

        agg_dict: Dict[str, Any] = {
            customer_col: "nunique",
            "cohort_period": ["count", "max"],
        }

        summary = cohort_data.groupby("cohort").agg(agg_dict)
        summary.columns = [
            "total_customers",
            "total_events",
            "max_period",
        ]

        # Average lifetime periods per customer
        lifetime = (
            cohort_data.groupby(["cohort", customer_col])["cohort_period"]
            .max()
            .reset_index()
            .groupby("cohort")["cohort_period"]
            .mean()
        )
        summary["avg_lifetime_periods"] = lifetime

        # Revenue if available
        if revenue_col in cohort_data.columns:
            rev = cohort_data.groupby("cohort")[revenue_col].sum()
            summary["total_revenue"] = rev

        summary = summary.reset_index()
        summary = summary.sort_values("total_customers", ascending=False)
        summary = summary.reset_index(drop=True)

        return summary

    def filter_cohorts(
        self,
        cohort_data: pd.DataFrame,
        cohorts: Optional[List[str]] = None,
        min_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Filter cohort data by cohort names or minimum size.

        Args:
            cohort_data: DataFrame with cohort assignments.
            cohorts: List of cohort labels to keep. If None, keeps all.
            min_size: Minimum unique customers per cohort. If None, uses
                config['min_cohort_size'].

        Returns:
            Filtered DataFrame.
        """
        customer_col = self.config["customer_column"]
        min_size = min_size if min_size is not None else self.config["min_cohort_size"]

        result = cohort_data.copy()

        if cohorts is not None:
            result = result[result["cohort"].isin(cohorts)]

        # Filter by minimum size
        cohort_sizes = result.groupby("cohort")[customer_col].nunique()
        valid_cohorts = cohort_sizes[cohort_sizes >= min_size].index
        result = result[result["cohort"].isin(valid_cohorts)]

        return result

    # ------------------------------------------------------------------
    # Retention Curves
    # ------------------------------------------------------------------

    def get_retention_curves(
        self,
        retention_matrix: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Extract per-cohort retention curves from a retention matrix.

        Each curve is an array of retention rates over successive periods,
        starting from period 0 (always 1.0) through the last tracked period.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``,
                with cohort labels as index and period offsets as columns.

        Returns:
            Dictionary mapping cohort label strings to 1-D numpy arrays
            of retention rates.
        """
        curves: Dict[str, np.ndarray] = {}
        for cohort_label in retention_matrix.index:
            row = retention_matrix.loc[cohort_label].dropna()
            curves[str(cohort_label)] = row.values.astype(float)
        return curves

    def get_average_retention_curve(
        self,
        retention_matrix: pd.DataFrame,
    ) -> np.ndarray:
        """Compute the average retention curve across all cohorts.

        For each period offset, averages the retention rate across all
        cohorts that have data for that period.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.

        Returns:
            1-D numpy array of average retention rates per period.
        """
        return retention_matrix.mean(axis=0, skipna=True).values.astype(float)

    def compute_half_life(
        self,
        retention_matrix: pd.DataFrame,
    ) -> pd.Series:
        """Compute the retention half-life for each cohort.

        Half-life is the first period offset where retention drops below 50%.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.

        Returns:
            Series indexed by cohort label with half-life period values.
            ``NaN`` if retention never drops below 50%.
        """
        curves = self.get_retention_curves(retention_matrix)
        half_lives: Dict[str, float] = {}
        for cohort_label, curve in curves.items():
            below_50 = np.where(curve < 0.5)[0]
            if len(below_50) > 0:
                half_lives[cohort_label] = int(below_50[0])
            else:
                half_lives[cohort_label] = np.nan
        return pd.Series(half_lives, name="half_life")

    def compute_churn_rates(
        self,
        retention_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute period-over-period churn rates from a retention matrix.

        Churn rate at period *t* is defined as:
            ``churn(t) = 1 - retention(t) / retention(t-1)``

        This gives the *incremental* churn between consecutive periods,
        not just ``1 - retention``.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.

        Returns:
            DataFrame with same shape as ``retention_matrix``, where each
            cell contains the incremental churn rate for that
            (cohort, period) pair. Period 0 is always 0.0.
        """
        shifted = retention_matrix.shift(1, axis=1)
        # Avoid division by zero
        churn = 1.0 - retention_matrix.div(shifted.replace(0, np.nan))
        # Period 0 churn is 0 by definition
        if 0 in churn.columns:
            churn[0] = 0.0
        return churn

    def compute_cumulative_revenue(
        self,
        cohort_data: pd.DataFrame,
        max_periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute cumulative revenue per cohort over time.

        Args:
            cohort_data: DataFrame with cohort, cohort_period, and
                revenue columns.
            max_periods: Maximum periods to include. Defaults to
                config['periods'].

        Returns:
            DataFrame with cohorts as rows, periods as columns, and
            cumulative revenue as values.
        """
        max_periods = max_periods or self.config["periods"]
        revenue_col = self.config["revenue_column"]

        if revenue_col not in cohort_data.columns:
            raise ValueError(
                f"Revenue column '{revenue_col}' not found in data. "
                f"Available columns: {list(cohort_data.columns)}"
            )

        rev_agg = (
            cohort_data.groupby(["cohort", "cohort_period"])[revenue_col]
            .sum()
            .reset_index()
        )

        rev_pivot = rev_agg.pivot_table(
            index="cohort",
            columns="cohort_period",
            values=revenue_col,
            aggfunc="sum",
            fill_value=0,
        )

        valid_cols = sorted(c for c in rev_pivot.columns if c <= max_periods)
        rev_pivot = rev_pivot[valid_cols]

        # Cumulative sum across periods
        return rev_pivot.cumsum(axis=1)

    # ------------------------------------------------------------------
    # Visualization Methods
    # ------------------------------------------------------------------

    def plot_retention_heatmap(
        self,
        retention_matrix: pd.DataFrame,
        title: str = "Cohort Retention Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "YlOrRd",
        annot: bool = True,
        fmt: str = ".0%",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a heatmap of the retention matrix.

        Displays retention rates as a color-coded heatmap with cohorts
        on the y-axis and periods on the x-axis.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.
            title: Plot title.
            figsize: Figure size as (width, height).
            cmap: Matplotlib/seaborn colormap name.
            annot: Whether to annotate cells with retention values.
            fmt: Format string for annotations (e.g., ".0%" for percentages).
            save_path: If provided, saves the figure to this file path.

        Returns:
            The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            retention_matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cohort Period", fontsize=12)
        ax.set_ylabel("Cohort", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Retention heatmap saved to %s", save_path)

        return fig

    def plot_retention_lines(
        self,
        retention_matrix: pd.DataFrame,
        title: str = "Cohort Retention Curves",
        figsize: Tuple[int, int] = (12, 6),
        show_average: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot retention curves as line plots, one line per cohort.

        Args:
            retention_matrix: DataFrame from ``compute_retention_matrix()``.
            title: Plot title.
            figsize: Figure size as (width, height).
            show_average: If True, overlay the average retention curve
                as a dashed black line.
            save_path: If provided, saves the figure to this file path.

        Returns:
            The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        curves = self.get_retention_curves(retention_matrix)
        periods_all = sorted(retention_matrix.columns)

        for cohort_label, curve in curves.items():
            periods = periods_all[: len(curve)]
            ax.plot(periods, curve, marker="o", markersize=4, label=cohort_label)

        if show_average and len(curves) > 0:
            avg_curve = self.get_average_retention_curve(retention_matrix)
            avg_periods = periods_all[: len(avg_curve)]
            ax.plot(
                avg_periods,
                avg_curve,
                color="black",
                linewidth=2.5,
                linestyle="--",
                marker="s",
                markersize=5,
                label="Average",
                zorder=10,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cohort Period", fontsize=12)
        ax.set_ylabel("Retention Rate", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Cohort", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Retention line plot saved to %s", save_path)

        return fig

    def plot_cohort_sizes(
        self,
        cohort_data: pd.DataFrame,
        title: str = "Cohort Sizes",
        figsize: Tuple[int, int] = (10, 6),
        color: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a bar chart of cohort sizes (unique customers per cohort).

        Args:
            cohort_data: DataFrame with cohort assignments
                (output of ``assign_cohorts()``).
            title: Plot title.
            figsize: Figure size as (width, height).
            color: Bar color. If None, uses the default seaborn palette.
            save_path: If provided, saves the figure to this file path.

        Returns:
            The matplotlib Figure object.
        """
        customer_col = self.config["customer_column"]

        cohort_sizes = (
            cohort_data.groupby("cohort")[customer_col]
            .nunique()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=figsize)

        bar_color = color or sns.color_palette()[0]
        ax.bar(
            range(len(cohort_sizes)),
            cohort_sizes.values,
            color=bar_color,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xticks(range(len(cohort_sizes)))
        ax.set_xticklabels(cohort_sizes.index, rotation=45, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cohort", fontsize=12)
        ax.set_ylabel("Number of Customers", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        # Add value labels on top of bars
        for i, (_, val) in enumerate(cohort_sizes.items()):
            ax.text(
                i,
                val + max(cohort_sizes.values) * 0.01,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Cohort size bar chart saved to %s", save_path)

        return fig

    def full_analysis(
        self,
        data: pd.DataFrame,
        cohort_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a complete cohort analysis pipeline.

        Convenience method that chains cohort assignment, retention matrix
        computation, retention curves, churn rates, summary, and half-life.

        Args:
            data: Raw event-level DataFrame.
            cohort_type: Cohort type override.

        Returns:
            Dictionary with keys:
                - ``cohort_data``: DataFrame with cohort assignments
                - ``retention_matrix``: Retention rate matrix
                - ``retention_curves``: Per-cohort retention curves
                - ``avg_retention_curve``: Average curve across cohorts
                - ``churn_rates``: Period-over-period churn rates
                - ``half_life``: Half-life per cohort
                - ``summary``: Cohort summary table
                - ``metrics``: Dict of all configured metric matrices
        """
        cohort_data = self.assign_cohorts(data, cohort_type=cohort_type)
        cohort_data = self.filter_cohorts(cohort_data)

        retention_matrix = self.compute_retention_matrix(cohort_data)
        retention_curves = self.get_retention_curves(retention_matrix)
        avg_curve = self.get_average_retention_curve(retention_matrix)
        churn_rates = self.compute_churn_rates(retention_matrix)
        half_life = self.compute_half_life(retention_matrix)
        summary = self.get_cohort_summary(cohort_data)
        metrics = self.compute_cohort_metrics(cohort_data)

        return {
            "cohort_data": cohort_data,
            "retention_matrix": retention_matrix,
            "retention_curves": retention_curves,
            "avg_retention_curve": avg_curve,
            "churn_rates": churn_rates,
            "half_life": half_life,
            "summary": summary,
            "metrics": metrics,
        }
