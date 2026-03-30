"""
Enhanced Recommendations Dashboard View.

Provides a comprehensive view of personalized retention recommendations
including KPI summary, distribution analysis, segment breakdowns,
cost-benefit analysis, and filterable recommendation tables.

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Action type display names and colors
ACTION_COLORS = {
    "coupon": "#2ecc71",
    "push_notification": "#3498db",
    "email_campaign": "#9b59b6",
    "email": "#9b59b6",
    "loyalty_points": "#f39c12",
    "personal_outreach": "#e74c3c",
    "exclusive_offer": "#1abc9c",
    "no_action": "#95a5a6",
}

ACTION_LABELS = {
    "coupon": "Coupon / Discount",
    "push_notification": "Push Notification",
    "email_campaign": "Email Campaign",
    "email": "Email Campaign",
    "loyalty_points": "Loyalty Points",
    "personal_outreach": "Personal Outreach",
    "exclusive_offer": "Exclusive Offer",
    "no_action": "No Action",
}


def render_recommendations_view(st_module, config: Dict, data_loader=None):
    """Render enhanced personalized recommendations page.

    Shows:
    - KPI summary cards (total recs, avg uplift, top action, coverage)
    - Recommendation type distribution (donut chart)
    - Expected uplift by recommendation type (box + bar)
    - Priority score distribution (histogram)
    - Segment-level recommendation breakdown (stacked bar)
    - Cost-benefit analysis with retention offers
    - Filterable recommendation table
    - Top-K prioritized recommendations

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Personalized Recommendations")
    st.markdown(
        "AI-driven retention recommendations based on churn risk, "
        "CLV, uplift scores, and customer segment affinity."
    )

    if data_loader is None:
        from src.dashboard.app import get_data_loader
        data_loader = get_data_loader(config)

    # Load data
    recs = data_loader.load_recommendations()
    retention_offers = data_loader.load_retention_offers()
    predictions = data_loader.load_predictions()

    if recs.empty:
        st.warning("No recommendations available.")
        return

    # ==================================================================
    # Section 1: KPI Summary Cards
    # ==================================================================
    _render_kpi_cards(st, recs, retention_offers)

    # ==================================================================
    # Section 2: Recommendation Distribution
    # ==================================================================
    st.markdown("---")
    _render_distribution_section(st, recs)

    # ==================================================================
    # Section 3: Uplift Analysis
    # ==================================================================
    st.markdown("---")
    _render_uplift_analysis(st, recs)

    # ==================================================================
    # Section 4: Segment Breakdown
    # ==================================================================
    st.markdown("---")
    _render_segment_breakdown(st, recs, predictions)

    # ==================================================================
    # Section 5: Cost-Benefit Analysis
    # ==================================================================
    if not retention_offers.empty:
        st.markdown("---")
        _render_cost_benefit_analysis(st, config, retention_offers)

    # ==================================================================
    # Section 6: Filterable Recommendation Table
    # ==================================================================
    st.markdown("---")
    _render_recommendation_table(st, recs, retention_offers)


# =========================================================================
# Internal rendering helpers
# =========================================================================


def _render_kpi_cards(st, recs: pd.DataFrame, offers: pd.DataFrame):
    """Render KPI summary cards for recommendations."""
    kc1, kc2, kc3, kc4 = st.columns(4)

    total_recs = len(recs)
    kc1.metric("Total Recommendations", f"{total_recs:,}")

    if "expected_uplift" in recs.columns:
        avg_uplift = recs["expected_uplift"].mean()
        kc2.metric("Avg Expected Uplift", f"{avg_uplift:.2%}")
    else:
        kc2.metric("Avg Expected Uplift", "N/A")

    if "recommendation_type" in recs.columns:
        top_action = recs["recommendation_type"].value_counts().idxmax()
        top_label = ACTION_LABELS.get(top_action, top_action)
        kc3.metric("Top Action Type", top_label)
    else:
        kc3.metric("Top Action Type", "N/A")

    if "priority_score" in recs.columns:
        high_priority = (recs["priority_score"] >= 0.7).sum()
        kc4.metric("High Priority", f"{high_priority:,}")
    elif not offers.empty and "priority_rank" in offers.columns:
        kc4.metric("Offers Generated", f"{len(offers):,}")
    else:
        kc4.metric("High Priority", "N/A")


def _render_distribution_section(st, recs: pd.DataFrame):
    """Render recommendation type distribution with donut and bar charts."""
    st.subheader("Recommendation Distribution")

    if "recommendation_type" not in recs.columns:
        st.info("No recommendation type data available.")
        return

    type_counts = recs["recommendation_type"].value_counts().reset_index()
    type_counts.columns = ["recommendation_type", "count"]

    col_donut, col_bar = st.columns(2)

    with col_donut:
        colors = [
            ACTION_COLORS.get(t, "#95a5a6")
            for t in type_counts["recommendation_type"]
        ]
        fig_donut = go.Figure(data=[go.Pie(
            labels=type_counts["recommendation_type"],
            values=type_counts["count"],
            hole=0.4,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>"
                          "Count: %{value}<br>"
                          "Share: %{percent}<extra></extra>",
        )])
        fig_donut.update_layout(
            title="Recommendation Type Distribution",
            showlegend=True,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            type_counts,
            x="recommendation_type",
            y="count",
            color="recommendation_type",
            color_discrete_map=ACTION_COLORS,
            title="Recommendations by Type",
            labels={
                "recommendation_type": "Action Type",
                "count": "Number of Recommendations",
            },
            text="count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


def _render_uplift_analysis(st, recs: pd.DataFrame):
    """Render expected uplift analysis by recommendation type."""
    st.subheader("Expected Uplift Analysis")

    if "expected_uplift" not in recs.columns:
        st.info("No uplift data available for recommendations.")
        return

    if "recommendation_type" not in recs.columns:
        # Simple histogram
        fig = px.histogram(
            recs, x="expected_uplift", nbins=30,
            title="Expected Uplift Distribution",
            color_discrete_sequence=["#3498db"],
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    col_box, col_avg = st.columns(2)

    with col_box:
        fig_box = px.box(
            recs,
            x="recommendation_type",
            y="expected_uplift",
            color="recommendation_type",
            color_discrete_map=ACTION_COLORS,
            title="Uplift Distribution by Action Type",
            labels={
                "recommendation_type": "Action Type",
                "expected_uplift": "Expected Uplift",
            },
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_avg:
        avg_uplift = (
            recs.groupby("recommendation_type")["expected_uplift"]
            .mean()
            .reset_index()
        )
        avg_uplift.columns = ["recommendation_type", "avg_uplift"]
        avg_uplift = avg_uplift.sort_values("avg_uplift", ascending=False)

        fig_avg = px.bar(
            avg_uplift,
            x="recommendation_type",
            y="avg_uplift",
            color="recommendation_type",
            color_discrete_map=ACTION_COLORS,
            title="Average Expected Uplift by Action",
            labels={
                "recommendation_type": "Action Type",
                "avg_uplift": "Average Expected Uplift",
            },
            text=avg_uplift["avg_uplift"].apply(lambda v: f"{v:.2%}"),
        )
        fig_avg.update_traces(textposition="outside")
        fig_avg.update_layout(showlegend=False)
        st.plotly_chart(fig_avg, use_container_width=True)

    # Priority score distribution
    if "priority_score" in recs.columns:
        st.markdown("#### Priority Score Distribution")
        fig_priority = px.histogram(
            recs, x="priority_score", nbins=30,
            color_discrete_sequence=["#e67e22"],
            title="Priority Score Distribution",
            labels={"priority_score": "Priority Score"},
        )
        mean_priority = recs["priority_score"].mean()
        fig_priority.add_vline(
            x=mean_priority, line_dash="dash", line_color="red",
            annotation_text=f"Mean: {mean_priority:.2f}",
        )
        st.plotly_chart(fig_priority, use_container_width=True)


def _render_segment_breakdown(
    st, recs: pd.DataFrame, predictions: pd.DataFrame,
):
    """Render segment-level recommendation breakdown."""
    st.subheader("Segment-Level Breakdown")

    # Try to merge segment info from predictions if not in recs
    if "segment" not in recs.columns and not predictions.empty:
        if "customer_id" in recs.columns and "customer_id" in predictions.columns:
            merged = recs.merge(
                predictions[["customer_id", "segment"]],
                on="customer_id",
                how="left",
            )
        else:
            merged = recs.copy()
    else:
        merged = recs.copy()

    if "segment" not in merged.columns or "recommendation_type" not in merged.columns:
        st.info("Segment or recommendation type data not available.")
        return

    # Stacked bar: action types per segment
    cross_tab = pd.crosstab(
        merged["segment"], merged["recommendation_type"],
    ).reset_index()
    action_cols = [c for c in cross_tab.columns if c != "segment"]

    fig_stacked = go.Figure()
    for action in action_cols:
        color = ACTION_COLORS.get(action, "#95a5a6")
        fig_stacked.add_trace(go.Bar(
            name=action,
            x=cross_tab["segment"],
            y=cross_tab[action],
            marker_color=color,
        ))
    fig_stacked.update_layout(
        title="Recommendation Types by Segment",
        xaxis_title="Segment",
        yaxis_title="Count",
        barmode="stack",
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Segment summary stats
    if "expected_uplift" in merged.columns:
        seg_stats = merged.groupby("segment").agg(
            count=("customer_id", "count") if "customer_id" in merged.columns
            else ("expected_uplift", "count"),
            avg_uplift=("expected_uplift", "mean"),
            max_uplift=("expected_uplift", "max"),
        ).reset_index()
        seg_stats["avg_uplift"] = seg_stats["avg_uplift"].round(4)
        seg_stats["max_uplift"] = seg_stats["max_uplift"].round(4)
        st.markdown("#### Segment Summary")
        st.dataframe(seg_stats, use_container_width=True)


def _render_cost_benefit_analysis(
    st, config: Dict, offers: pd.DataFrame,
):
    """Render cost-benefit analysis for retention offers."""
    st.subheader("Cost-Benefit Analysis")

    currency = config.get("budget", {}).get("currency", "KRW")

    if offers.empty:
        st.info("No retention offer data available.")
        return

    # KPI cards for offers
    oc1, oc2, oc3, oc4 = st.columns(4)

    total_cost = offers["estimated_cost_krw"].sum() if "estimated_cost_krw" in offers.columns else 0
    total_revenue = offers["estimated_revenue_save_krw"].sum() if "estimated_revenue_save_krw" in offers.columns else 0
    overall_roi = total_revenue / max(total_cost, 1)
    avg_uplift = offers["expected_uplift"].mean() if "expected_uplift" in offers.columns else 0

    oc1.metric("Total Campaign Cost", f"{total_cost:,.0f} {currency}")
    oc2.metric("Est. Revenue Saved", f"{total_revenue:,.0f} {currency}")
    oc3.metric("Overall ROI", f"{overall_roi:.1f}x")
    oc4.metric("Avg Expected Uplift", f"{avg_uplift:.2%}")

    col_cost, col_roi = st.columns(2)

    with col_cost:
        if "offer_type" in offers.columns and "estimated_cost_krw" in offers.columns:
            cost_by_type = (
                offers.groupby("offer_type")["estimated_cost_krw"]
                .sum()
                .reset_index()
            )
            fig_cost = px.bar(
                cost_by_type,
                x="offer_type",
                y="estimated_cost_krw",
                title=f"Total Cost by Offer Type ({currency})",
                labels={
                    "offer_type": "Offer Type",
                    "estimated_cost_krw": f"Cost ({currency})",
                },
                color="offer_type",
                text=cost_by_type["estimated_cost_krw"].apply(
                    lambda v: f"{v:,.0f}"
                ),
            )
            fig_cost.update_traces(textposition="outside")
            fig_cost.update_layout(showlegend=False)
            st.plotly_chart(fig_cost, use_container_width=True)

    with col_roi:
        if all(c in offers.columns for c in ["offer_type", "estimated_cost_krw", "estimated_revenue_save_krw"]):
            roi_by_type = offers.groupby("offer_type").agg(
                total_cost=("estimated_cost_krw", "sum"),
                total_revenue=("estimated_revenue_save_krw", "sum"),
            ).reset_index()
            roi_by_type["roi"] = (
                roi_by_type["total_revenue"] / roi_by_type["total_cost"].clip(lower=1)
            )
            fig_roi = px.bar(
                roi_by_type.sort_values("roi", ascending=False),
                x="offer_type",
                y="roi",
                title="ROI by Offer Type",
                labels={
                    "offer_type": "Offer Type",
                    "roi": "ROI (x)",
                },
                color="offer_type",
                text=roi_by_type.sort_values("roi", ascending=False)["roi"].apply(
                    lambda v: f"{v:.1f}x"
                ),
            )
            fig_roi.update_traces(textposition="outside")
            fig_roi.update_layout(showlegend=False)
            st.plotly_chart(fig_roi, use_container_width=True)

    # Scatter: cost vs revenue saved, colored by risk
    if all(c in offers.columns for c in ["estimated_cost_krw", "estimated_revenue_save_krw"]):
        color_col = "risk_level" if "risk_level" in offers.columns else None
        fig_scatter = px.scatter(
            offers,
            x="estimated_cost_krw",
            y="estimated_revenue_save_krw",
            color=color_col,
            size="expected_uplift" if "expected_uplift" in offers.columns else None,
            title="Cost vs Revenue Saved per Customer",
            labels={
                "estimated_cost_krw": f"Estimated Cost ({currency})",
                "estimated_revenue_save_krw": f"Est. Revenue Saved ({currency})",
            },
            hover_data=["customer_id"] if "customer_id" in offers.columns else None,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


def _render_recommendation_table(
    st, recs: pd.DataFrame, offers: pd.DataFrame,
):
    """Render filterable recommendation table and top-K list."""
    st.subheader("Recommendation Details")

    # Filters
    filter_col1, filter_col2 = st.columns(2)

    display_df = recs.copy()

    with filter_col1:
        if "recommendation_type" in recs.columns:
            all_types = ["All"] + sorted(recs["recommendation_type"].unique().tolist())
            selected_type = st.selectbox("Filter by Action Type", all_types)
            if selected_type != "All":
                display_df = display_df[
                    display_df["recommendation_type"] == selected_type
                ]

    with filter_col2:
        if "priority_score" in recs.columns:
            min_priority = float(st.slider(
                "Minimum Priority Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
            ))
            display_df = display_df[
                display_df["priority_score"] >= min_priority
            ]

    # Sort by priority
    if "priority_score" in display_df.columns:
        display_df = display_df.sort_values("priority_score", ascending=False)

    st.markdown(f"**Showing {len(display_df)} recommendations**")
    st.dataframe(display_df, use_container_width=True)

    # Top-K prioritized list
    if "priority_score" in recs.columns:
        st.markdown("#### Top Priority Recommendations")
        top_k = min(10, len(recs))
        top_recs = recs.nlargest(top_k, "priority_score")
        st.dataframe(top_recs, use_container_width=True)

    # Retention offers detail table
    if not offers.empty:
        st.markdown("#### Detailed Retention Offers")
        display_cols = [
            c for c in [
                "customer_id", "segment", "risk_level", "churn_probability",
                "offer_type", "offer_detail", "expected_uplift",
                "estimated_cost_krw", "estimated_revenue_save_krw",
                "priority_rank",
            ]
            if c in offers.columns
        ]
        st.dataframe(
            offers[display_cols].head(20),
            use_container_width=True,
        )
