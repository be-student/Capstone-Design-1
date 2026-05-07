"""
Streamlit Dashboard Application for Churn Prediction System.

Provides interactive views for:
- Churn prediction overview with KPI cards
- Model performance comparison (ML, DL, Ensemble)
- Customer segmentation visualization
- Budget optimization with what-if scenarios
- A/B testing results
- Survival analysis curves
- Personalized recommendations
- CLV prediction distribution
- Uplift modeling results
- Real-time scoring status
- MLflow experiment comparison

All configurable parameters are loaded from config/simulator_config.yaml.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

from src.dashboard.monitoring_view import (
    render_model_monitoring as render_monitoring_view,
)
from src.dashboard.recommendations_view import render_recommendations_view
from src.dashboard.system_health_view import render_system_health
from src.dashboard.utils.dashboard_helpers import (
    PAGES,
    classify_risk,
    format_count,
    format_currency,
    format_percentage,
    get_app_title,
    get_budget_config,
    get_churn_definition,
    get_color_palette,
    get_ensemble_weights,
    get_page_icon,
    get_page_list,
    get_risk_color,
    get_segment_colors,
    build_sidebar_info,
    validate_predictions,
    safe_get_column,
    compute_kpi_delta,
)

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


def load_config() -> Dict[str, Any]:
    """Load YAML configuration.

    Returns:
        Parsed configuration dictionary.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_data_loader(config: Dict[str, Any]):
    """Create a DashboardDataLoader instance.

    Args:
        config: Parsed YAML configuration.

    Returns:
        DashboardDataLoader instance.
    """
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


# =========================================================================
# Page render functions
# =========================================================================

def render_overview(st_module, config: Dict, data_loader=None):
    """Render churn prediction overview page.

    Shows KPI cards, churn probability distribution, risk levels,
    segment churn rates, feature importance chart, and individual
    customer lookup with detailed risk profile.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Churn Prediction Overview")

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()

    if predictions.empty:
        st.warning("No prediction data available.")
        return

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(predictions)
    avg_churn = predictions["churn_probability"].mean()
    high_risk = (predictions["churn_probability"] > 0.5).sum()
    total_clv = predictions.get("clv_predicted", pd.Series([0])).sum()

    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Avg Churn Prob", f"{avg_churn:.2%}")
    col3.metric("High Risk", f"{high_risk:,}")
    col4.metric("Total CLV", f"{total_clv:,.0f} KRW")

    # Churn probability distribution
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(
        predictions, x="churn_probability", nbins=30,
        title="Distribution of Churn Probabilities",
        labels={"churn_probability": "Churn Probability"},
        color_discrete_sequence=["#3498db"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk level distribution
    st.subheader("Risk Level Distribution")
    risk_counts = predictions["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig2 = px.pie(
        risk_counts, values="Count", names="Risk Level",
        title="Customer Risk Levels",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Segment churn rates
    st.subheader("Average Churn Probability by Segment")
    seg_rates = predictions.groupby("segment")[
        "churn_probability"
    ].mean().reset_index()
    seg_rates.columns = ["Segment", "Avg Churn Prob"]
    fig3 = px.bar(
        seg_rates, x="Segment", y="Avg Churn Prob",
        title="Churn Rate by Customer Segment",
        color="Avg Churn Prob",
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------------------------------------------
    # Feature importance chart
    # -----------------------------------------------------------------
    st.subheader("Feature Importance")
    feature_importance = data_loader.load_feature_importance()
    if not feature_importance.empty:
        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)
        fig_fi = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {top_n} Feature Importance Scores",
            labels={"importance": "Importance", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment overview table
    # -----------------------------------------------------------------
    st.subheader("Customer Segment Overview")
    seg_summary = predictions.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_churn=("churn_probability", "mean"),
        high_risk_count=("churn_probability", lambda x: (x > 0.5).sum()),
        avg_clv=("clv_predicted", "mean")
        if "clv_predicted" in predictions.columns
        else ("churn_probability", "count"),
    ).reset_index()
    seg_summary.columns = [
        "Segment", "Customers", "Avg Churn Prob",
        "High Risk Count", "Avg CLV",
    ]
    st.dataframe(
        seg_summary.style.format({
            "Avg Churn Prob": "{:.2%}",
            "Avg CLV": "{:,.0f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Individual customer lookup
    # -----------------------------------------------------------------
    st.subheader("Individual Customer Lookup")
    customer_ids = sorted(predictions["customer_id"].unique().tolist())
    selected_id = st.selectbox(
        "Select Customer ID",
        options=customer_ids,
        key="customer_lookup_select",
    )

    if selected_id:
        customer_row = predictions[
            predictions["customer_id"] == selected_id
        ]
        if not customer_row.empty:
            row = customer_row.iloc[0]
            st.markdown(f"### Customer: {selected_id}")

            lc1, lc2, lc3 = st.columns(3)
            churn_prob = row["churn_probability"]
            risk = row.get("risk_level", classify_risk(churn_prob))
            segment = row.get("segment", "unknown")

            lc1.metric("Churn Probability", f"{churn_prob:.2%}")
            lc2.metric("Risk Level", risk.upper())
            lc3.metric("Segment", segment)

            # Additional details row
            lc4, lc5, lc6 = st.columns(3)
            clv = row.get("clv_predicted", 0)
            action = row.get("recommended_action", "N/A")
            days_purchase = row.get("days_since_last_purchase", 0)

            lc4.metric("Predicted CLV", f"{clv:,.0f} KRW")
            lc5.metric("Recommended Action", str(action))
            lc6.metric("Days Since Purchase", f"{days_purchase:.0f}")

            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                title={"text": "Churn Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": get_risk_color(risk)},
                    "steps": [
                        {"range": [0, 25], "color": "#d5f5e3"},
                        {"range": [25, 50], "color": "#fdebd0"},
                        {"range": [50, 75], "color": "#fadbd8"},
                        {"range": [75, 100], "color": "#f5b7b1"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning(f"Customer {selected_id} not found.")


# Note: render_churn_analytics and render_cohort_analysis are defined
# below (after render_retention_campaign) with comprehensive
# implementations that include all visualization components.


def render_model_performance(st_module, config: Dict, data_loader=None):
    """Render model performance comparison page.

    Shows MLflow experiment metrics, model comparison charts (bar, radar,
    ROC curves), confusion matrices, and ensemble configuration details.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Model Performance")

    if data_loader is None:
        data_loader = get_data_loader(config)

    metrics = data_loader.load_model_metrics()

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    ml_auc = metrics.get("ml_model", {}).get("auc", 0)
    dl_auc = metrics.get("dl_model", {}).get("auc", 0)
    ens_auc = metrics.get("ensemble", {}).get("auc", 0)
    best_model = max(metrics.items(), key=lambda x: x[1].get("auc", 0))

    kc1.metric("ML Model AUC", f"{ml_auc:.4f}")
    kc2.metric("DL Model AUC", f"{dl_auc:.4f}")
    kc3.metric("Ensemble AUC", f"{ens_auc:.4f}")
    kc4.metric("Best Model", best_model[0])

    # AUC threshold indicator
    threshold = 0.78
    if ens_auc >= threshold:
        st.success(
            f"Ensemble AUC: {ens_auc:.4f} (>= {threshold} threshold)"
        )
    else:
        st.error(
            f"Ensemble AUC: {ens_auc:.4f} (< {threshold} threshold)"
        )

    # -----------------------------------------------------------------
    # Performance Comparison Table
    # -----------------------------------------------------------------
    st.subheader("Performance Comparison")
    df = pd.DataFrame(metrics).T
    df.index.name = "Model"
    st.dataframe(
        df.style.highlight_max(axis=0).format("{:.4f}"),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Metrics Comparison Bar Chart
    # -----------------------------------------------------------------
    st.subheader("Metrics Comparison Chart")
    models = list(metrics.keys())
    metric_names = ["auc", "precision", "recall", "f1_score", "accuracy"]
    model_colors = {
        "ml_model": "#3498db",
        "dl_model": "#e67e22",
        "ensemble": "#2ecc71",
    }
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Bar(
            name=m,
            x=metric_names,
            y=[metrics[m].get(mn, 0) for mn in metric_names],
            marker_color=model_colors.get(m, "#95a5a6"),
        ))
    fig.update_layout(
        barmode="group", title="Model Metrics Comparison",
        yaxis_title="Score", yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # ROC Curves
    # -----------------------------------------------------------------
    st.subheader("ROC Curves")
    roc_data = data_loader.load_roc_data()
    fig_roc = go.Figure()
    for model_name, curve in roc_data.items():
        model_auc_val = metrics.get(model_name, {}).get("auc", 0)
        fig_roc.add_trace(go.Scatter(
            x=curve["fpr"],
            y=curve["tpr"],
            mode="lines",
            name=f"{model_name} (AUC={model_auc_val:.3f})",
            line=dict(
                color=model_colors.get(model_name, "#95a5a6"), width=2,
            ),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random (AUC=0.5)",
        line=dict(color="gray", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        title="ROC Curves - Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # -----------------------------------------------------------------
    # Confusion Matrices
    # -----------------------------------------------------------------
    st.subheader("Confusion Matrices")
    cm_data = data_loader.load_confusion_matrices()
    cm_cols = st.columns(len(cm_data))
    for idx, (model_name, matrix) in enumerate(cm_data.items()):
        with cm_cols[idx]:
            cm = np.array(matrix)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ))
            fig_cm.update_layout(
                title=f"{model_name}",
                height=300,
                xaxis_title="Predicted",
                yaxis_title="Actual",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            total = tn + fp + fn + tp
            st.caption(
                f"Acc: {(tn + tp) / total:.2%} | "
                f"Prec: {tp / max(tp + fp, 1):.2%} | "
                f"Rec: {tp / max(tp + fn, 1):.2%}"
            )

    # -----------------------------------------------------------------
    # Radar Chart - Model Comparison
    # -----------------------------------------------------------------
    st.subheader("Model Capability Radar")
    radar_metrics = ["auc", "precision", "recall", "f1_score", "accuracy"]
    fig_radar = go.Figure()
    for m in models:
        values = [metrics[m].get(mn, 0) for mn in radar_metrics]
        values.append(values[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            name=m,
            opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Performance Radar",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # -----------------------------------------------------------------
    # MLflow Experiment Runs
    # -----------------------------------------------------------------
    st.subheader("MLflow Experiment Runs")
    mlflow_runs = data_loader.load_mlflow_runs()
    if not mlflow_runs.empty:
        st.dataframe(
            mlflow_runs.style.highlight_max(
                subset=["auc", "precision", "recall", "f1_score"],
                axis=0,
            ).format({
                "auc": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1_score": "{:.4f}",
                "accuracy": "{:.4f}",
                "training_time_s": "{:.1f}",
            }),
            use_container_width=True,
        )

        # Training time comparison
        fig_time = px.bar(
            mlflow_runs, x="model_type", y="training_time_s",
            title="Training Time by Model Type",
            labels={
                "model_type": "Model Type",
                "training_time_s": "Training Time (seconds)",
            },
            color="model_type",
            text="training_time_s",
        )
        fig_time.update_traces(
            texttemplate="%{text:.1f}s", textposition="outside",
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # AUC vs Training Time scatter
        fig_tradeoff = px.scatter(
            mlflow_runs, x="training_time_s", y="auc",
            size="params_epochs",
            color="model_type",
            title="AUC vs Training Time Trade-off",
            labels={
                "training_time_s": "Training Time (s)",
                "auc": "AUC",
            },
            hover_data=["params_lr", "params_epochs"],
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)
    else:
        st.info("No MLflow run data available.")

    # -----------------------------------------------------------------
    # Ensemble Configuration
    # -----------------------------------------------------------------
    st.subheader("Ensemble Configuration")
    ml_w = config.get("pipeline", {}).get("ensemble_weight_ml", 0.6)
    dl_w = config.get("pipeline", {}).get("ensemble_weight_dl", 0.4)

    col_ens1, col_ens2 = st.columns(2)
    with col_ens1:
        st.info(f"ML Weight: {ml_w} | DL Weight: {dl_w}")
        fig_weights = px.pie(
            values=[ml_w, dl_w],
            names=["ML Model", "DL Model"],
            title="Ensemble Weight Distribution",
            color_discrete_sequence=["#3498db", "#e67e22"],
        )
        st.plotly_chart(fig_weights, use_container_width=True)

    with col_ens2:
        improvement_data = pd.DataFrame({
            "Metric": metric_names,
            "ML": [
                metrics.get("ml_model", {}).get(m, 0) for m in metric_names
            ],
            "DL": [
                metrics.get("dl_model", {}).get(m, 0) for m in metric_names
            ],
            "Ensemble": [
                metrics.get("ensemble", {}).get(m, 0)
                for m in metric_names
            ],
        })
        improvement_data["Ensemble Gain vs ML"] = (
            improvement_data["Ensemble"] - improvement_data["ML"]
        ).round(4)
        improvement_data["Ensemble Gain vs DL"] = (
            improvement_data["Ensemble"] - improvement_data["DL"]
        ).round(4)
        st.markdown("**Ensemble Improvement Over Individual Models**")
        st.dataframe(
            improvement_data.style.format({
                "ML": "{:.4f}", "DL": "{:.4f}",
                "Ensemble": "{:.4f}",
                "Ensemble Gain vs ML": "{:+.4f}",
                "Ensemble Gain vs DL": "{:+.4f}",
            }),
            use_container_width=True,
        )


def render_segmentation(st_module, config: Dict, data_loader=None):
    """Render customer segmentation visualization page.

    Shows segment distribution (pie and bar), per-segment churn risk
    heatmap, segment statistics table, retention action mapping from
    config, and per-segment CLV comparison.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Customer Segmentation")

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()

    if predictions.empty:
        st.warning("No segmentation data available.")
        return

    # KPI summary per segment
    n_segments = predictions["segment"].nunique()
    total_cust = len(predictions)
    highest_risk_seg = (
        predictions.groupby("segment")["churn_probability"]
        .mean()
        .idxmax()
    )
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Total Segments", n_segments)
    kc2.metric("Total Customers", f"{total_cust:,}")
    kc3.metric("Highest Risk Segment", highest_risk_seg)

    # Segment distribution - pie
    st.subheader("Segment Distribution")
    seg_counts = predictions["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig = px.pie(
            seg_counts, values="Count", names="Segment",
            title="Customer Segment Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            seg_counts, x="Segment", y="Count",
            title="Customers per Segment",
            color="Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Segment risk heatmap
    st.subheader("Segment Churn Risk Analysis")
    seg_stats = predictions.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_churn=("churn_probability", "mean"),
        min_churn=("churn_probability", "min"),
        max_churn=("churn_probability", "max"),
        std_churn=("churn_probability", "std"),
    ).reset_index()
    seg_stats.columns = [
        "Segment", "Count", "Avg Churn", "Min Churn",
        "Max Churn", "Std Churn",
    ]

    fig_heat = px.bar(
        seg_stats.sort_values("Avg Churn", ascending=True),
        x="Avg Churn",
        y="Segment",
        orientation="h",
        title="Average Churn Probability by Segment",
        color="Avg Churn",
        color_continuous_scale="RdYlGn_r",
        text="Count",
    )
    fig_heat.update_traces(textposition="outside")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Detailed segment statistics table
    st.subheader("Segment Statistics")
    st.dataframe(
        seg_stats.style.format({
            "Avg Churn": "{:.2%}",
            "Min Churn": "{:.2%}",
            "Max Churn": "{:.2%}",
            "Std Churn": "{:.4f}",
        }),
        use_container_width=True,
    )

    # CLV by segment (if available)
    if "clv_predicted" in predictions.columns:
        st.subheader("CLV by Segment")
        seg_clv = predictions.groupby("segment")["clv_predicted"].agg(
            ["mean", "sum"]
        ).reset_index()
        seg_clv.columns = ["Segment", "Mean CLV", "Total CLV"]
        fig_clv = px.bar(
            seg_clv, x="Segment", y="Mean CLV",
            title="Average CLV by Segment",
            color="Segment",
            text="Mean CLV",
        )
        fig_clv.update_traces(
            texttemplate="%{text:,.0f}", textposition="outside",
        )
        st.plotly_chart(fig_clv, use_container_width=True)

    # Risk level distribution within each segment
    st.subheader("Risk Level Distribution by Segment")
    if "risk_level" in predictions.columns:
        risk_seg = predictions.groupby(
            ["segment", "risk_level"]
        ).size().reset_index(name="count")
        fig_risk_seg = px.bar(
            risk_seg,
            x="segment",
            y="count",
            color="risk_level",
            title="Risk Level Distribution within Segments",
            barmode="stack",
            color_discrete_map={
                "low": "#2ecc71",
                "medium": "#f39c12",
                "high": "#e67e22",
                "critical": "#e74c3c",
            },
        )
        st.plotly_chart(fig_risk_seg, use_container_width=True)

    # Segment details from config
    seg_config = config.get("segmentation", {}).get("segments", [])
    if seg_config:
        st.subheader("Segment Definitions & Retention Actions")
        seg_df = pd.DataFrame([
            {
                "Name": s.get("name", ""),
                "Korean": s.get("name_kr", ""),
                "Retention Action": s.get("retention_action", ""),
            }
            for s in seg_config
        ])
        st.dataframe(seg_df, use_container_width=True)


def render_budget_optimization(st_module, config: Dict, data_loader=None):
    """Render budget optimization page with interactive controls.

    Features:
    - Interactive budget constraint slider
    - Cost and uplift multiplier inputs for what-if scenarios
    - Allocation results table and charts
    - Scenario comparison across what-if parameters
    - ROI visualization per segment

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Budget Optimization")

    if data_loader is None:
        data_loader = get_data_loader(config)

    default_budget = config.get("budget", {}).get("total_krw", 50_000_000)
    currency = config.get("budget", {}).get("currency", "KRW")

    # -----------------------------------------------------------------
    # Interactive inputs for budget constraints
    # -----------------------------------------------------------------
    st.subheader("Budget Constraints & Scenario Parameters")

    col1, col2 = st.columns(2)
    with col1:
        total_budget = st.slider(
            "Total Budget (KRW)",
            min_value=10_000_000,
            max_value=200_000_000,
            value=default_budget,
            step=5_000_000,
            format="%d",
            key="budget_slider",
        )
        st.caption(f"Default: {default_budget:,.0f} {currency}")

    with col2:
        cost_multiplier = st.slider(
            "Cost Multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="cost_mult",
            help="Adjust campaign cost assumptions (1.0 = baseline)",
        )
        uplift_multiplier = st.slider(
            "Uplift Multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="uplift_mult",
            help="Adjust uplift effectiveness assumptions (1.0 = baseline)",
        )

    # -----------------------------------------------------------------
    # Load budget results (baseline)
    # -----------------------------------------------------------------
    budget_results = data_loader.load_budget_results()

    if budget_results.empty:
        st.warning("No budget optimization data available.")
        return

    # Scale allocations proportionally to the selected budget
    baseline_total = budget_results["allocated_budget_krw"].sum()
    if baseline_total > 0:
        scale = total_budget / baseline_total
    else:
        scale = 1.0

    display_results = budget_results.copy()
    display_results["allocated_budget_krw"] = (
        display_results["allocated_budget_krw"] * scale
    ).astype(int)
    display_results["expected_revenue_saved_krw"] = (
        display_results["expected_revenue_saved_krw"]
        * scale * uplift_multiplier
    ).astype(int)
    display_results["expected_retained"] = (
        display_results["expected_retained"]
        * scale * uplift_multiplier
    ).astype(int)
    # Adjust ROI by uplift/cost ratio
    if cost_multiplier > 0:
        display_results["roi"] = (
            display_results["roi"] * uplift_multiplier / cost_multiplier
        ).round(2)

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    st.subheader("Allocation Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_alloc = display_results["allocated_budget_krw"].sum()
    total_retained = display_results["expected_retained"].sum()
    total_rev_saved = display_results["expected_revenue_saved_krw"].sum()
    avg_roi = display_results["roi"].mean()

    kpi1.metric("Total Allocated", f"{total_alloc:,.0f} {currency}")
    kpi2.metric("Expected Retained", f"{total_retained:,}")
    kpi3.metric("Revenue Saved", f"{total_rev_saved:,.0f} {currency}")
    kpi4.metric("Avg ROI", f"{avg_roi:.1f}x")

    # -----------------------------------------------------------------
    # Allocation results table
    # -----------------------------------------------------------------
    st.subheader("Budget Allocation by Segment")
    st.dataframe(
        display_results.style.format({
            "allocated_budget_krw": "{:,.0f}",
            "expected_revenue_saved_krw": "{:,.0f}",
            "roi": "{:.2f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Allocation visualization - bar chart
    # -----------------------------------------------------------------
    st.subheader("Allocation Distribution")
    fig_alloc = px.bar(
        display_results,
        x="segment",
        y="allocated_budget_krw",
        title=f"Budget Allocation by Segment (Total: {total_alloc:,.0f} {currency})",
        labels={
            "segment": "Customer Segment",
            "allocated_budget_krw": f"Allocated Budget ({currency})",
        },
        color="roi",
        color_continuous_scale="Viridis",
        text="allocated_budget_krw",
    )
    fig_alloc.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_alloc, use_container_width=True)

    # ROI by segment
    st.subheader("ROI by Segment")
    fig_roi = px.bar(
        display_results,
        x="segment",
        y="roi",
        title="Expected ROI by Customer Segment",
        labels={"segment": "Segment", "roi": "ROI (x)"},
        color="segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # Pie chart of allocation proportions
    st.subheader("Allocation Proportions")
    fig_pie = px.pie(
        display_results,
        values="allocated_budget_krw",
        names="segment",
        title="Budget Share by Segment",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------------------------------------------
    # Multi-Channel Budget Allocation
    # -----------------------------------------------------------------
    st.subheader("Channel-Level Cost Breakdown")
    channel_config = config.get("budget", {}).get("channels", {})
    if channel_config:
        channel_data = _build_channel_allocation_data(
            budget_results=display_results,
            channel_config=channel_config,
            total_budget=total_budget,
        )
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            fig_channel_bar = px.bar(
                channel_data,
                x="channel",
                y="allocated_budget",
                color="channel",
                title="Budget Allocation by Channel",
                labels={
                    "channel": "Channel",
                    "allocated_budget": f"Allocated ({currency})",
                },
                text="allocated_budget",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_channel_bar.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_channel_bar, use_container_width=True)

        with ch_col2:
            fig_channel_roi = px.bar(
                channel_data,
                x="channel",
                y="roi_multiplier",
                color="channel",
                title="ROI Multiplier by Channel",
                labels={
                    "channel": "Channel",
                    "roi_multiplier": "ROI Multiplier",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_channel_roi, use_container_width=True)

        # Channel cost per action table
        st.markdown("**Channel Cost & ROI Details**")
        st.dataframe(
            channel_data.style.format({
                "cost_per_action": "{:,.0f}",
                "allocated_budget": "{:,.0f}",
                "roi_multiplier": "{:.1f}x",
                "expected_actions": "{:,.0f}",
            }),
            use_container_width=True,
        )

        # Efficiency frontier
        st.markdown("**Channel Efficiency Frontier**")
        fig_frontier = px.scatter(
            channel_data,
            x="cost_per_action",
            y="roi_multiplier",
            size="allocated_budget",
            color="channel",
            title="Efficiency Frontier: Cost vs ROI",
            labels={
                "cost_per_action": f"Cost per Action ({currency})",
                "roi_multiplier": "ROI Multiplier",
            },
            text="channel",
        )
        fig_frontier.update_traces(textposition="top center")
        st.plotly_chart(fig_frontier, use_container_width=True)
    else:
        st.info(
            "Channel configuration not found in config. "
            "Add budget.channels to simulator_config.yaml for "
            "multi-channel allocation views."
        )

    # -----------------------------------------------------------------
    # What-If Scenario Comparison
    # -----------------------------------------------------------------
    st.subheader("What-If Scenario Comparison")
    st.markdown(
        "Compare budget optimization outcomes across different budget levels "
        "and parameter assumptions."
    )

    # Define scenarios
    scenarios = _build_whatif_scenarios(
        default_budget=default_budget,
        current_budget=total_budget,
        cost_multiplier=cost_multiplier,
        uplift_multiplier=uplift_multiplier,
    )

    comparison_df = _compute_scenario_comparison(
        budget_results=budget_results,
        baseline_total=baseline_total,
        scenarios=scenarios,
    )

    st.dataframe(
        comparison_df.style.format({
            "Budget (KRW)": "{:,.0f}",
            "Total Allocated": "{:,.0f}",
            "Expected Retained": "{:,.0f}",
            "Revenue Saved": "{:,.0f}",
            "Avg ROI": "{:.2f}",
        }),
        use_container_width=True,
    )

    # Comparison bar chart
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="Total Allocated",
        x=comparison_df["Scenario"],
        y=comparison_df["Total Allocated"],
        yaxis="y",
    ))
    fig_compare.add_trace(go.Scatter(
        name="Avg ROI",
        x=comparison_df["Scenario"],
        y=comparison_df["Avg ROI"],
        yaxis="y2",
        mode="lines+markers",
        marker=dict(size=10, color="red"),
        line=dict(width=2, color="red"),
    ))
    fig_compare.update_layout(
        title="Scenario Comparison: Allocation vs ROI",
        yaxis=dict(title=f"Total Allocated ({currency})", side="left"),
        yaxis2=dict(
            title="Avg ROI (x)", side="right",
            overlaying="y", showgrid=False,
        ),
        barmode="group",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Retained customers comparison
    fig_retained = px.bar(
        comparison_df,
        x="Scenario",
        y="Expected Retained",
        title="Expected Retained Customers by Scenario",
        color="Scenario",
        text="Expected Retained",
    )
    fig_retained.update_traces(
        texttemplate="%{text:,.0f}", textposition="outside",
    )
    st.plotly_chart(fig_retained, use_container_width=True)

    # Budget sweep chart
    st.subheader("Budget Sweep Analysis")
    sweep_df = _compute_budget_sweep(
        budget_results=budget_results,
        baseline_total=baseline_total,
        min_budget=10_000_000,
        max_budget=200_000_000,
        steps=20,
        cost_multiplier=cost_multiplier,
        uplift_multiplier=uplift_multiplier,
    )

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["Budget"],
        y=sweep_df["Retained"],
        name="Expected Retained",
        mode="lines+markers",
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["Budget"],
        y=sweep_df["Revenue Saved"],
        name="Revenue Saved",
        yaxis="y2",
        mode="lines+markers",
    ))
    fig_sweep.update_layout(
        title="Budget Sweep: Retained Customers & Revenue Saved",
        xaxis_title=f"Budget ({currency})",
        yaxis=dict(title="Retained Customers", side="left"),
        yaxis2=dict(
            title=f"Revenue Saved ({currency})", side="right",
            overlaying="y", showgrid=False,
        ),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)


def render_ab_testing(st_module, config: Dict, data_loader=None):
    """Render A/B testing results page with detailed statistical analysis.

    Shows multiple experiment results, statistical significance indicators,
    confidence intervals, effect sizes, power analysis, and comparison
    charts across all A/B test experiments.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("A/B Testing Results")

    if data_loader is None:
        data_loader = get_data_loader(config)

    # Load detailed multi-experiment results
    detailed = data_loader.load_ab_test_detailed()
    experiments = detailed.get("experiments", [])
    summary = detailed.get("summary", {})

    # Also load basic results for backward compatibility
    results = data_loader.load_ab_test_results()

    # -----------------------------------------------------------------
    # Summary KPI cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric(
        "Total Experiments",
        summary.get("total_experiments", len(experiments)),
    )
    kc2.metric(
        "Significant Results",
        summary.get("significant_count", 0),
    )
    kc3.metric(
        "Best Experiment",
        summary.get("best_experiment", "N/A"),
    )
    kc4.metric(
        "Avg Lift",
        f"{summary.get('avg_lift', 0):.1%}",
    )

    # -----------------------------------------------------------------
    # Per-experiment detailed results
    # -----------------------------------------------------------------
    for i, exp in enumerate(experiments):
        exp_name = exp.get("name", f"Experiment {i + 1}")
        is_sig = exp.get("is_significant", False)
        sig_icon = "✅" if is_sig else "⚠️"

        st.subheader(f"{sig_icon} Experiment: {exp_name}")

        # Metrics row
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        ec1.metric(
            "Treatment Churn",
            f"{exp.get('treatment_churn_rate', 0):.2%}",
        )
        ec2.metric(
            "Control Churn",
            f"{exp.get('control_churn_rate', 0):.2%}",
        )
        ec3.metric("Lift", f"{exp.get('lift', 0):.1%}")
        ec4.metric("p-value", f"{exp.get('p_value', 1.0):.4f}")
        ec5.metric("Power", f"{exp.get('power', 0):.2%}")

        # Significance indicator
        p_val = exp.get("p_value", 1.0)
        alpha = exp.get("alpha", 0.05)
        if is_sig:
            st.success(
                f"Statistically Significant (p={p_val:.4f} < α={alpha})"
            )
        else:
            st.warning(
                f"Not Significant (p={p_val:.4f} >= α={alpha})"
            )

        # Charts for this experiment
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Churn rate comparison with CI
            ci = exp.get("confidence_interval", [0, 0])
            effect = exp.get("absolute_effect", 0)
            fig_rate = go.Figure()
            fig_rate.add_trace(go.Bar(
                name="Treatment",
                x=["Churn Rate"],
                y=[exp.get("treatment_churn_rate", 0)],
                marker_color="#2ecc71",
                width=0.3,
            ))
            fig_rate.add_trace(go.Bar(
                name="Control",
                x=["Churn Rate"],
                y=[exp.get("control_churn_rate", 0)],
                marker_color="#e74c3c",
                width=0.3,
            ))
            fig_rate.update_layout(
                title=f"Churn Rate: {exp_name}",
                barmode="group",
                yaxis_title="Churn Rate",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_rate, use_container_width=True)

        with col_chart2:
            # Confidence interval visualization
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                x=[effect],
                y=[exp_name],
                mode="markers",
                marker=dict(size=12, color="#3498db"),
                name="Effect Size",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[ci[1] - effect] if len(ci) > 1 else [0],
                    arrayminus=[effect - ci[0]] if len(ci) > 0 else [0],
                    color="#3498db",
                    thickness=2,
                    width=6,
                ),
            ))
            fig_ci.add_vline(
                x=0, line_dash="dash", line_color="red",
                annotation_text="No Effect",
            )
            fig_ci.update_layout(
                title=f"Effect Size & 95% CI",
                xaxis_title="Absolute Effect (Churn Reduction)",
                xaxis_tickformat=".0%",
                height=250,
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        # Statistical details expander
        with st.expander(f"Statistical Details - {exp_name}"):
            detail_cols = st.columns(3)
            detail_cols[0].markdown(f"""
**Sample Sizes**
- Treatment: {exp.get('treatment_size', 0):,}
- Control: {exp.get('control_size', 0):,}
- Total: {exp.get('treatment_size', 0) + exp.get('control_size', 0):,}
            """)
            detail_cols[1].markdown(f"""
**Effect Measures**
- Absolute Effect: {exp.get('absolute_effect', 0):.4f}
- Relative Lift: {exp.get('lift', 0):.1%}
- Cohen's h: {exp.get('effect_size_cohens_h', 0):.4f}
            """)
            detail_cols[2].markdown(f"""
**Test Configuration**
- Test Type: {exp.get('test_type', 'N/A')}
- Significance Level (α): {exp.get('alpha', 0.05)}
- Statistical Power: {exp.get('power', 0):.2%}
- Duration: {exp.get('duration_days', 0)} days
            """)

        st.markdown("---")

    # -----------------------------------------------------------------
    # Cross-experiment comparison
    # -----------------------------------------------------------------
    if len(experiments) > 1:
        st.subheader("Cross-Experiment Comparison")

        exp_comparison = pd.DataFrame([
            {
                "Experiment": e.get("name", ""),
                "Treatment Churn": e.get("treatment_churn_rate", 0),
                "Control Churn": e.get("control_churn_rate", 0),
                "Lift": e.get("lift", 0),
                "p-value": e.get("p_value", 1.0),
                "Power": e.get("power", 0),
                "Significant": "Yes" if e.get("is_significant") else "No",
                "Cohen's h": e.get("effect_size_cohens_h", 0),
            }
            for e in experiments
        ])
        st.dataframe(
            exp_comparison.style.format({
                "Treatment Churn": "{:.2%}",
                "Control Churn": "{:.2%}",
                "Lift": "{:.1%}",
                "p-value": "{:.4f}",
                "Power": "{:.2%}",
                "Cohen's h": "{:.4f}",
            }),
            use_container_width=True,
        )

        # Lift comparison chart
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            fig_lift = px.bar(
                exp_comparison,
                x="Experiment", y="Lift",
                title="Relative Lift by Experiment",
                color="Significant",
                color_discrete_map={
                    "Yes": "#2ecc71", "No": "#e74c3c",
                },
                text="Lift",
            )
            fig_lift.update_traces(texttemplate="%{text:.1%}")
            fig_lift.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_lift, use_container_width=True)

        with col_comp2:
            # Power vs p-value scatter
            fig_power = px.scatter(
                exp_comparison,
                x="p-value", y="Power",
                size="Cohen's h",
                color="Significant",
                text="Experiment",
                title="Statistical Power vs p-value",
                color_discrete_map={
                    "Yes": "#2ecc71", "No": "#e74c3c",
                },
            )
            fig_power.add_vline(
                x=0.05, line_dash="dash", line_color="gray",
                annotation_text="α=0.05",
            )
            fig_power.add_hline(
                y=0.80, line_dash="dash", line_color="gray",
                annotation_text="80% Power",
            )
            fig_power.update_layout(
                xaxis_tickformat=".3f",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_power, use_container_width=True)

    # -----------------------------------------------------------------
    # Power Analysis / Sample Size Calculator
    # -----------------------------------------------------------------
    st.subheader("Power Analysis & Sample Size Calculator")
    st.markdown(
        "Estimate required sample sizes and statistical power for "
        "planning future A/B experiments."
    )

    pa_col1, pa_col2, pa_col3 = st.columns(3)
    with pa_col1:
        baseline_rate = st.slider(
            "Baseline Churn Rate",
            min_value=0.01,
            max_value=0.50,
            value=0.20,
            step=0.01,
            key="pa_baseline",
            help="Expected churn rate without treatment",
        )
    with pa_col2:
        mde = st.slider(
            "Minimum Detectable Effect (MDE)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="pa_mde",
            help="Smallest effect size you want to detect",
        )
    with pa_col3:
        pa_alpha = st.selectbox(
            "Significance Level (α)",
            options=[0.01, 0.05, 0.10],
            index=1,
            key="pa_alpha",
        )
        pa_power_target = st.selectbox(
            "Target Power (1-β)",
            options=[0.80, 0.85, 0.90, 0.95],
            index=0,
            key="pa_power_target",
        )

    # Compute sample size using normal approximation for two proportions
    pa_results = _compute_power_analysis(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=pa_alpha,
        power=pa_power_target,
    )

    pa_kpi1, pa_kpi2, pa_kpi3 = st.columns(3)
    pa_kpi1.metric(
        "Required Sample Size (per group)",
        f"{pa_results['sample_size_per_group']:,}",
    )
    pa_kpi2.metric(
        "Total Participants Needed",
        f"{pa_results['total_participants']:,}",
    )
    pa_kpi3.metric(
        "Expected Duration (days)",
        f"{pa_results['estimated_duration_days']}",
        help="Based on 100 new enrollments per day",
    )

    # Power curve chart
    st.markdown("**Power Curve: Sample Size vs Statistical Power**")
    power_curve = _compute_power_curve(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=pa_alpha,
        max_n=pa_results["sample_size_per_group"] * 3,
    )
    fig_pcurve = go.Figure()
    fig_pcurve.add_trace(go.Scatter(
        x=power_curve["n"],
        y=power_curve["power"],
        mode="lines",
        name="Power",
        line=dict(color="#3498db", width=2),
    ))
    fig_pcurve.add_hline(
        y=pa_power_target,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target: {pa_power_target:.0%}",
    )
    fig_pcurve.add_vline(
        x=pa_results["sample_size_per_group"],
        line_dash="dash",
        line_color="green",
        annotation_text=f"n={pa_results['sample_size_per_group']:,}",
    )
    fig_pcurve.update_layout(
        title="Power vs Sample Size",
        xaxis_title="Sample Size per Group",
        yaxis_title="Statistical Power",
        yaxis_tickformat=".0%",
        height=350,
    )
    st.plotly_chart(fig_pcurve, use_container_width=True)

    # MDE sensitivity table
    st.markdown("**MDE Sensitivity Analysis**")
    mde_table = _compute_mde_sensitivity(
        baseline_rate=baseline_rate,
        alpha=pa_alpha,
        power=pa_power_target,
    )
    st.dataframe(
        mde_table.style.format({
            "MDE": "{:.1%}",
            "Sample Size (per group)": "{:,.0f}",
            "Total Participants": "{:,.0f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Multiple Comparison Correction Summary
    # -----------------------------------------------------------------
    if len(experiments) > 1:
        st.subheader("Multiple Comparison Correction")
        st.markdown(
            "When running multiple experiments simultaneously, p-values "
            "should be corrected to control the family-wise error rate."
        )
        p_values = [e.get("p_value", 1.0) for e in experiments]
        exp_names = [e.get("name", f"Exp {i+1}") for i, e in enumerate(experiments)]
        correction_df = _compute_multiple_comparison_corrections(
            p_values=p_values,
            experiment_names=exp_names,
            alpha=0.05,
        )
        st.dataframe(
            correction_df.style.format({
                "Raw p-value": "{:.4f}",
                "Bonferroni": "{:.4f}",
                "Holm-Bonferroni": "{:.4f}",
                "BH (FDR)": "{:.4f}",
            }),
            use_container_width=True,
        )


def render_survival_analysis(st_module, config: Dict, data_loader=None):
    """Render survival analysis page with Kaplan-Meier curves.

    Shows KPI summary, Kaplan-Meier survival curves per segment with
    confidence intervals, median survival times, hazard rate comparison,
    event rate analysis, and duration distributions.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Survival Analysis")

    if data_loader is None:
        data_loader = get_data_loader(config)

    survival = data_loader.load_survival_data()

    if survival.empty:
        st.warning("No survival analysis data available.")
        return

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_cust = len(survival)
    event_count = survival["event_observed"].sum()
    event_rate = event_count / total_cust if total_cust > 0 else 0
    median_duration = survival["duration_days"].median()

    kc1.metric("Total Customers", f"{total_cust:,}")
    kc2.metric("Events Observed (Churn)", f"{int(event_count):,}")
    kc3.metric("Event Rate", f"{event_rate:.2%}")
    kc4.metric("Median Duration", f"{median_duration:.0f} days")

    # -----------------------------------------------------------------
    # Kaplan-Meier Survival Curves by Segment
    # -----------------------------------------------------------------
    st.subheader("Kaplan-Meier Survival Curves by Segment")
    surv_curves = data_loader.load_survival_curves()

    segment_colors = {
        "vip_loyal": "#2ecc71",
        "regular_loyal": "#3498db",
        "bargain_hunter": "#e67e22",
        "explorer": "#9b59b6",
        "dormant": "#e74c3c",
        "new_customer": "#f39c12",
    }

    fig_km = go.Figure()
    for seg_name, curve_data in surv_curves.items():
        timeline = curve_data.get("timeline", [])
        surv_prob = curve_data.get("survival_prob", [])
        ci_lower = curve_data.get("ci_lower", [])
        ci_upper = curve_data.get("ci_upper", [])
        color = segment_colors.get(seg_name, "#95a5a6")

        # Main survival curve
        fig_km.add_trace(go.Scatter(
            x=timeline, y=surv_prob,
            mode="lines",
            name=seg_name,
            line=dict(color=color, width=2),
        ))
        # Confidence interval band
        if ci_lower and ci_upper:
            fig_km.add_trace(go.Scatter(
                x=timeline + timeline[::-1],
                y=ci_upper + ci_lower[::-1],
                fill="toself",
                fillcolor=color.replace(")", ",0.1)").replace(
                    "#", "rgba("
                ) if color.startswith("rgba") else f"rgba(150,150,150,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                name=f"{seg_name} CI",
            ))

    # Median survival reference line
    fig_km.add_hline(
        y=0.5, line_dash="dash", line_color="gray",
        annotation_text="50% Survival (Median)",
    )
    fig_km.update_layout(
        title="Kaplan-Meier Survival Curves by Customer Segment",
        xaxis_title="Days Since First Purchase",
        yaxis_title="Survival Probability",
        yaxis_range=[0, 1.05],
        hovermode="x unified",
    )
    st.plotly_chart(fig_km, use_container_width=True)

    # -----------------------------------------------------------------
    # Median Survival Times Table
    # -----------------------------------------------------------------
    st.subheader("Median Survival Time by Segment")
    median_data = []
    for seg_name, curve_data in surv_curves.items():
        median_surv = curve_data.get("median_survival_days")
        surv_prob_list = curve_data.get("survival_prob", [])
        final_surv = surv_prob_list[-1] if surv_prob_list else 0
        median_data.append({
            "Segment": seg_name,
            "Median Survival (days)": (
                median_surv if median_surv else ">360"
            ),
            "Final Survival Prob": final_surv,
            "Risk Level": (
                "Low" if final_surv > 0.7
                else "Medium" if final_surv > 0.5
                else "High" if final_surv > 0.3
                else "Critical"
            ),
        })
    median_df = pd.DataFrame(median_data)
    st.dataframe(median_df, use_container_width=True)

    # -----------------------------------------------------------------
    # Survival Probability by Segment (bar chart)
    # -----------------------------------------------------------------
    st.subheader("Average Survival Probability by Segment")
    seg_surv = survival.groupby("segment")[
        "survival_probability"
    ].mean().reset_index()
    seg_surv.columns = ["Segment", "Avg Survival Prob"]
    seg_surv = seg_surv.sort_values("Avg Survival Prob", ascending=True)
    fig = px.bar(
        seg_surv, x="Avg Survival Prob", y="Segment",
        orientation="h",
        title="Average Survival Probability by Segment",
        color="Avg Survival Prob",
        color_continuous_scale="RdYlGn",
        text="Avg Survival Prob",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # Hazard Rate Comparison
    # -----------------------------------------------------------------
    st.subheader("Estimated Hazard Rate by Segment")
    hazard_data = []
    for seg_name, curve_data in surv_curves.items():
        surv_prob_list = curve_data.get("survival_prob", [1.0, 0.5])
        timeline = curve_data.get("timeline", [0, 360])
        if len(surv_prob_list) >= 2 and surv_prob_list[0] > 0:
            # Approximate hazard rate from survival curve
            s0 = surv_prob_list[0]
            s_end = surv_prob_list[-1]
            t_max = timeline[-1] if timeline else 360
            hazard = -np.log(max(s_end / s0, 0.001)) / max(t_max, 1)
            hazard_data.append({
                "Segment": seg_name,
                "Hazard Rate (per day)": round(hazard, 6),
                "Hazard Rate (annualized)": round(hazard * 365, 4),
            })

    if hazard_data:
        hazard_df = pd.DataFrame(hazard_data)
        col_hz1, col_hz2 = st.columns(2)
        with col_hz1:
            fig_hz = px.bar(
                hazard_df, x="Segment", y="Hazard Rate (per day)",
                title="Daily Hazard Rate by Segment",
                color="Segment",
                text="Hazard Rate (per day)",
            )
            fig_hz.update_traces(
                texttemplate="%{text:.5f}", textposition="outside",
            )
            st.plotly_chart(fig_hz, use_container_width=True)

        with col_hz2:
            st.dataframe(
                hazard_df.style.format({
                    "Hazard Rate (per day)": "{:.6f}",
                    "Hazard Rate (annualized)": "{:.4f}",
                }),
                use_container_width=True,
            )

    # -----------------------------------------------------------------
    # Event Rate by Segment
    # -----------------------------------------------------------------
    st.subheader("Event Rate by Segment")
    event_stats = survival.groupby("segment").agg(
        total=("customer_id", "count"),
        events=("event_observed", "sum"),
        avg_duration=("duration_days", "mean"),
        median_duration=("duration_days", "median"),
    ).reset_index()
    event_stats["event_rate"] = event_stats["events"] / event_stats["total"]
    event_stats.columns = [
        "Segment", "Total", "Events", "Avg Duration",
        "Median Duration", "Event Rate",
    ]

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        fig_ev = px.bar(
            event_stats, x="Segment", y="Event Rate",
            title="Churn Event Rate by Segment",
            color="Event Rate",
            color_continuous_scale="RdYlGn_r",
            text="Event Rate",
        )
        fig_ev.update_traces(
            texttemplate="%{text:.1%}", textposition="outside",
        )
        fig_ev.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_ev, use_container_width=True)

    with col_ev2:
        st.dataframe(
            event_stats.style.format({
                "Avg Duration": "{:.1f}",
                "Median Duration": "{:.1f}",
                "Event Rate": "{:.2%}",
            }),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Duration distribution
    # -----------------------------------------------------------------
    st.subheader("Customer Lifetime Duration Distribution")
    fig2 = px.histogram(
        survival, x="duration_days", nbins=30,
        title="Distribution of Customer Durations",
        color="segment",
        barmode="overlay",
        opacity=0.7,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Duration box plot by segment
    st.subheader("Duration Distribution by Segment")
    fig_box = px.box(
        survival, x="segment", y="duration_days",
        color="segment",
        title="Customer Duration Distribution by Segment",
        labels={
            "segment": "Segment",
            "duration_days": "Duration (days)",
        },
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # Config info
    # -----------------------------------------------------------------
    surv_config = config.get("survival", {})
    if surv_config:
        st.markdown("---")
        st.subheader("Survival Model Configuration")
        st.json({
            "penalizer": surv_config.get("penalizer", "N/A"),
            "l1_ratio": surv_config.get("l1_ratio", "N/A"),
            "alpha": surv_config.get("alpha", "N/A"),
        })


def render_recommendations(st_module, config: Dict, data_loader=None):
    """Render personalized recommendations page.

    Delegates to the enhanced recommendations view module which provides
    KPI cards, distribution analysis, uplift analysis, segment breakdown,
    cost-benefit analysis, and filterable tables.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    if data_loader is None:
        data_loader = get_data_loader(config)

    render_recommendations_view(st_module, config, data_loader)


def _render_recommendations_legacy(st_module, config: Dict, data_loader=None):
    """Legacy recommendations renderer (kept for backward compatibility).

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Personalized Recommendations")

    if data_loader is None:
        data_loader = get_data_loader(config)

    recs = data_loader.load_recommendations()

    if recs.empty:
        st.warning("No recommendations available.")
        return

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_customers = len(recs)
    actionable = recs[
        recs["recommendation_type"] != "no_action"
    ] if "recommendation_type" in recs.columns else recs
    actionable_count = len(actionable)
    avg_uplift = recs["expected_uplift"].mean() if "expected_uplift" in recs.columns else 0
    avg_priority = recs["priority_score"].mean() if "priority_score" in recs.columns else 0

    kc1.metric("Total Customers", f"{total_customers:,}")
    kc2.metric("Actionable Recommendations", f"{actionable_count:,}")
    kc3.metric("Avg Expected Uplift", f"{avg_uplift:.2%}")
    kc4.metric("Avg Priority Score", f"{avg_priority:.2f}")

    # -----------------------------------------------------------------
    # Priority-ranked recommendations table
    # -----------------------------------------------------------------
    st.subheader("Priority-Ranked Retention Actions")
    recs_sorted = recs.sort_values("priority_score", ascending=False)
    st.dataframe(recs_sorted, use_container_width=True)

    # -----------------------------------------------------------------
    # Recommendation type distribution
    # -----------------------------------------------------------------
    st.subheader("Recommendation Type Distribution")
    col_rt1, col_rt2 = st.columns(2)
    with col_rt1:
        fig_pie = px.pie(
            recs, names="recommendation_type",
            title="Action Type Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_rt2:
        type_counts = recs["recommendation_type"].value_counts().reset_index()
        type_counts.columns = ["Action Type", "Count"]
        fig_bar_types = px.bar(
            type_counts, x="Action Type", y="Count",
            title="Recommendation Counts by Type",
            color="Action Type",
            text="Count",
        )
        fig_bar_types.update_traces(textposition="outside")
        st.plotly_chart(fig_bar_types, use_container_width=True)

    # -----------------------------------------------------------------
    # Expected Uplift Analysis
    # -----------------------------------------------------------------
    st.subheader("Expected Uplift by Customer")
    fig_uplift = px.bar(
        recs_sorted, x="customer_id", y="expected_uplift",
        color="recommendation_type",
        title="Expected Retention Uplift per Customer",
        labels={
            "expected_uplift": "Expected Uplift",
            "customer_id": "Customer ID",
        },
    )
    st.plotly_chart(fig_uplift, use_container_width=True)

    # Uplift distribution
    st.subheader("Uplift Score Distribution")
    fig_uplift_hist = px.histogram(
        recs, x="expected_uplift", nbins=20,
        title="Distribution of Expected Uplift Scores",
        color="recommendation_type",
        barmode="overlay",
        opacity=0.7,
    )
    st.plotly_chart(fig_uplift_hist, use_container_width=True)

    # -----------------------------------------------------------------
    # Priority vs Uplift scatter
    # -----------------------------------------------------------------
    st.subheader("Priority Score vs Expected Uplift")
    fig_scatter = px.scatter(
        recs, x="priority_score", y="expected_uplift",
        color="recommendation_type",
        size="priority_score",
        title="Priority Score vs Expected Uplift",
        labels={
            "priority_score": "Priority Score",
            "expected_uplift": "Expected Uplift",
        },
        hover_data=["customer_id"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment-level action breakdown (if segment column exists)
    # -----------------------------------------------------------------
    if "segment" in recs.columns:
        st.subheader("Retention Actions by Segment")
        seg_action = recs.groupby(
            ["segment", "recommendation_type"]
        ).size().reset_index(name="count")
        fig_seg = px.bar(
            seg_action, x="segment", y="count",
            color="recommendation_type",
            title="Recommended Actions by Customer Segment",
            barmode="group",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        # Average uplift by segment
        seg_uplift = recs.groupby("segment")[
            "expected_uplift"
        ].mean().reset_index()
        seg_uplift.columns = ["Segment", "Avg Expected Uplift"]
        seg_uplift = seg_uplift.sort_values(
            "Avg Expected Uplift", ascending=True,
        )
        fig_seg_uplift = px.bar(
            seg_uplift, x="Avg Expected Uplift", y="Segment",
            orientation="h",
            title="Average Expected Uplift by Segment",
            color="Avg Expected Uplift",
            color_continuous_scale="Viridis",
            text="Avg Expected Uplift",
        )
        fig_seg_uplift.update_traces(
            texttemplate="%{text:.2%}", textposition="outside",
        )
        st.plotly_chart(fig_seg_uplift, use_container_width=True)

    # -----------------------------------------------------------------
    # Cost-effectiveness analysis (if estimated_cost column exists)
    # -----------------------------------------------------------------
    if "estimated_cost" in recs.columns:
        st.subheader("Cost-Effectiveness Analysis")
        total_cost = recs["estimated_cost"].sum()
        st.metric("Total Estimated Cost", format_currency(total_cost, "KRW"))

        cost_by_type = recs.groupby("recommendation_type").agg(
            total_cost=("estimated_cost", "sum"),
            avg_cost=("estimated_cost", "mean"),
            count=("customer_id", "count"),
            avg_uplift=("expected_uplift", "mean"),
        ).reset_index()
        cost_by_type["cost_per_uplift_pct"] = (
            cost_by_type["avg_cost"]
            / cost_by_type["avg_uplift"].clip(lower=0.001)
        )
        st.dataframe(cost_by_type, use_container_width=True)

    # -----------------------------------------------------------------
    # Top priority customers
    # -----------------------------------------------------------------
    st.subheader("Top Priority Customers for Retention")
    top_n = min(10, len(recs_sorted))
    top_customers = recs_sorted.head(top_n)
    st.dataframe(top_customers, use_container_width=True)

    # -----------------------------------------------------------------
    # Recommendation configuration
    # -----------------------------------------------------------------
    rec_config = config.get("recommendations", {})
    if rec_config:
        st.markdown("---")
        st.subheader("Recommendation Engine Configuration")
        st.json(rec_config)


def render_clv(st_module, config: Dict, data_loader=None):
    """Render CLV prediction page with distributions and segment analysis.

    Shows CLV distribution (histogram + box plot), segment-level CLV
    breakdown, CLV-vs-churn scatter, CLV percentile analysis, top/bottom
    customer tables, and CLV tier classification.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("CLV Prediction")

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()
    clv_data = data_loader.load_clv_data()

    if predictions.empty or "clv_predicted" not in predictions.columns:
        st.warning("No CLV data available.")
        return

    currency = config.get("budget", {}).get("currency", "KRW")

    # -----------------------------------------------------------------
    # KPI summary cards
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_clv = predictions["clv_predicted"].sum()
    avg_clv = predictions["clv_predicted"].mean()
    median_clv = predictions["clv_predicted"].median()
    std_clv = predictions["clv_predicted"].std()

    kc1.metric("Total CLV", f"{total_clv:,.0f} {currency}")
    kc2.metric("Average CLV", f"{avg_clv:,.0f} {currency}")
    kc3.metric("Median CLV", f"{median_clv:,.0f} {currency}")
    kc4.metric("CLV Std Dev", f"{std_clv:,.0f} {currency}")

    # -----------------------------------------------------------------
    # CLV distribution - histogram + box plot side by side
    # -----------------------------------------------------------------
    st.subheader("CLV Distribution")
    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            predictions, x="clv_predicted", nbins=50,
            title="Customer Lifetime Value Distribution",
            labels={"clv_predicted": f"Predicted CLV ({currency})"},
            color_discrete_sequence=["#3498db"],
        )
        fig_hist.add_vline(
            x=avg_clv, line_dash="dash", line_color="red",
            annotation_text=f"Mean: {avg_clv:,.0f}",
        )
        fig_hist.add_vline(
            x=median_clv, line_dash="dot", line_color="green",
            annotation_text=f"Median: {median_clv:,.0f}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        fig_box = px.box(
            predictions, y="clv_predicted", x="segment",
            title="CLV Distribution by Segment",
            labels={"clv_predicted": f"CLV ({currency})", "segment": "Segment"},
            color="segment",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # CLV by segment - detailed breakdown
    # -----------------------------------------------------------------
    st.subheader("CLV by Segment")
    seg_clv = predictions.groupby("segment")["clv_predicted"].agg(
        ["mean", "sum", "count", "median", "std"]
    ).reset_index()
    seg_clv.columns = [
        "Segment", "Mean CLV", "Total CLV", "Count", "Median CLV", "Std CLV",
    ]
    seg_clv = seg_clv.sort_values("Mean CLV", ascending=False)

    col_bar, col_total = st.columns(2)
    with col_bar:
        fig_mean = px.bar(
            seg_clv, x="Segment", y="Mean CLV",
            title="Average CLV by Segment",
            color="Segment",
            text="Mean CLV",
        )
        fig_mean.update_traces(
            texttemplate="%{text:,.0f}", textposition="outside",
        )
        st.plotly_chart(fig_mean, use_container_width=True)

    with col_total:
        fig_total = px.bar(
            seg_clv, x="Segment", y="Total CLV",
            title="Total CLV by Segment",
            color="Segment",
            text="Total CLV",
        )
        fig_total.update_traces(
            texttemplate="%{text:,.0f}", textposition="outside",
        )
        st.plotly_chart(fig_total, use_container_width=True)

    # Segment CLV statistics table
    st.dataframe(
        seg_clv.style.format({
            "Mean CLV": "{:,.0f}",
            "Total CLV": "{:,.0f}",
            "Median CLV": "{:,.0f}",
            "Std CLV": "{:,.0f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # CLV vs Churn Probability scatter
    # -----------------------------------------------------------------
    st.subheader("CLV vs Churn Risk")
    fig_scatter = px.scatter(
        predictions, x="churn_probability", y="clv_predicted",
        color="segment",
        title="CLV vs Churn Probability (High CLV + High Churn = Priority)",
        labels={
            "churn_probability": "Churn Probability",
            "clv_predicted": f"Predicted CLV ({currency})",
        },
        hover_data=["customer_id"],
        opacity=0.7,
    )
    # Add quadrant lines
    fig_scatter.add_hline(
        y=median_clv, line_dash="dash", line_color="gray",
        annotation_text="Median CLV",
    )
    fig_scatter.add_vline(
        x=0.5, line_dash="dash", line_color="gray",
        annotation_text="Churn Threshold",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------
    # CLV Tier Classification
    # -----------------------------------------------------------------
    st.subheader("CLV Tier Classification")
    q25 = predictions["clv_predicted"].quantile(0.25)
    q50 = predictions["clv_predicted"].quantile(0.50)
    q75 = predictions["clv_predicted"].quantile(0.75)

    def _classify_clv_tier(v):
        if v >= q75:
            return "Platinum"
        elif v >= q50:
            return "Gold"
        elif v >= q25:
            return "Silver"
        return "Bronze"

    predictions_copy = predictions.copy()
    predictions_copy["clv_tier"] = predictions_copy["clv_predicted"].apply(
        _classify_clv_tier
    )

    tier_counts = predictions_copy["clv_tier"].value_counts().reset_index()
    tier_counts.columns = ["CLV Tier", "Count"]
    tier_order = ["Platinum", "Gold", "Silver", "Bronze"]
    tier_counts["CLV Tier"] = pd.Categorical(
        tier_counts["CLV Tier"], categories=tier_order, ordered=True,
    )
    tier_counts = tier_counts.sort_values("CLV Tier")

    col_pie, col_stats = st.columns(2)
    with col_pie:
        fig_tier = px.pie(
            tier_counts, values="Count", names="CLV Tier",
            title="CLV Tier Distribution",
            color="CLV Tier",
            color_discrete_map={
                "Platinum": "#e5e4e2",
                "Gold": "#ffd700",
                "Silver": "#c0c0c0",
                "Bronze": "#cd7f32",
            },
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_stats:
        tier_stats = predictions_copy.groupby("clv_tier").agg(
            count=("customer_id", "count"),
            avg_clv=("clv_predicted", "mean"),
            avg_churn=("churn_probability", "mean"),
        ).reset_index()
        tier_stats.columns = ["CLV Tier", "Customers", "Avg CLV", "Avg Churn"]
        st.dataframe(
            tier_stats.style.format({
                "Avg CLV": "{:,.0f}",
                "Avg Churn": "{:.2%}",
            }),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Top & bottom customers
    # -----------------------------------------------------------------
    st.subheader("Top 10 Customers by CLV")
    top_cols = ["customer_id", "clv_predicted", "segment", "churn_probability"]
    available_cols = [c for c in top_cols if c in predictions.columns]
    top10 = predictions.nlargest(10, "clv_predicted")[available_cols]
    st.dataframe(
        top10.style.format({"clv_predicted": "{:,.0f}"}),
        use_container_width=True,
    )

    st.subheader("Bottom 10 Customers by CLV")
    bottom10 = predictions.nsmallest(10, "clv_predicted")[available_cols]
    st.dataframe(
        bottom10.style.format({"clv_predicted": "{:,.0f}"}),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # CLV Percentile Analysis
    # -----------------------------------------------------------------
    st.subheader("CLV Percentile Analysis")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = [
        predictions["clv_predicted"].quantile(p / 100) for p in percentiles
    ]
    pct_df = pd.DataFrame({
        "Percentile": [f"P{p}" for p in percentiles],
        f"CLV ({currency})": pct_values,
    })
    fig_pct = px.bar(
        pct_df, x="Percentile", y=f"CLV ({currency})",
        title="CLV by Percentile",
        color_discrete_sequence=["#2ecc71"],
        text=f"CLV ({currency})",
    )
    fig_pct.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_pct, use_container_width=True)


def render_uplift(st_module, config: Dict, data_loader=None):
    """Render uplift modeling results page.

    Shows uplift score distributions, treatment effect analysis,
    segment-level uplift breakdown, persuadable/sleeping dog
    classification, and uplift-based targeting recommendations.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Uplift Modeling Results")

    if data_loader is None:
        data_loader = get_data_loader(config)

    uplift = data_loader.load_uplift_results()

    if uplift.empty:
        st.warning("No uplift data available.")
        return

    # -----------------------------------------------------------------
    # KPI summary
    # -----------------------------------------------------------------
    kc1, kc2, kc3, kc4 = st.columns(4)
    avg_uplift = uplift["uplift_score"].mean()
    avg_treatment = uplift["treatment_effect"].mean()
    persuadable = (uplift["uplift_score"] > 0).sum()
    sleeping_dogs = (uplift["uplift_score"] < 0).sum()

    kc1.metric("Avg Uplift Score", f"{avg_uplift:.4f}")
    kc2.metric("Avg Treatment Effect", f"{avg_treatment:.4f}")
    kc3.metric("Persuadable Customers", f"{persuadable:,}")
    kc4.metric("Sleeping Dogs", f"{sleeping_dogs:,}")

    # -----------------------------------------------------------------
    # Uplift distribution with treatment effect
    # -----------------------------------------------------------------
    st.subheader("Uplift Score Distribution")
    col_up, col_te = st.columns(2)

    with col_up:
        fig = px.histogram(
            uplift, x="uplift_score", nbins=30,
            title="Distribution of Uplift Scores",
            color_discrete_sequence=["#e67e22"],
        )
        fig.add_vline(
            x=0, line_dash="dash", line_color="red",
            annotation_text="Zero (No Effect)",
        )
        fig.add_vline(
            x=avg_uplift, line_dash="dot", line_color="blue",
            annotation_text=f"Mean: {avg_uplift:.4f}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_te:
        fig_te = px.histogram(
            uplift, x="treatment_effect", nbins=30,
            title="Distribution of Treatment Effects",
            color_discrete_sequence=["#9b59b6"],
        )
        fig_te.add_vline(
            x=0, line_dash="dash", line_color="red",
            annotation_text="Zero",
        )
        st.plotly_chart(fig_te, use_container_width=True)

    # -----------------------------------------------------------------
    # Uplift vs Treatment Effect scatter
    # -----------------------------------------------------------------
    st.subheader("Uplift Score vs Treatment Effect")
    fig_scatter = px.scatter(
        uplift, x="uplift_score", y="treatment_effect",
        color="segment",
        title="Uplift Score vs Treatment Effect by Segment",
        labels={
            "uplift_score": "Uplift Score",
            "treatment_effect": "Treatment Effect",
        },
        opacity=0.7,
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------
    # Uplift by segment - detailed
    # -----------------------------------------------------------------
    st.subheader("Uplift by Segment")
    seg_uplift = uplift.groupby("segment").agg(
        avg_uplift=("uplift_score", "mean"),
        avg_treatment=("treatment_effect", "mean"),
        count=("customer_id", "count"),
        persuadable=("uplift_score", lambda x: (x > 0).sum()),
    ).reset_index()
    seg_uplift.columns = [
        "Segment", "Avg Uplift", "Avg Treatment Effect",
        "Count", "Persuadable",
    ]
    seg_uplift["Persuadable %"] = (
        seg_uplift["Persuadable"] / seg_uplift["Count"] * 100
    ).round(1)

    fig_seg = px.bar(
        seg_uplift, x="Segment", y="Avg Uplift",
        title="Average Uplift Score by Segment",
        color="Segment",
        text="Avg Uplift",
    )
    fig_seg.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig_seg, use_container_width=True)

    st.dataframe(
        seg_uplift.style.format({
            "Avg Uplift": "{:.4f}",
            "Avg Treatment Effect": "{:.4f}",
            "Persuadable %": "{:.1f}%",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Customer classification (Persuadable / Sure Things / etc.)
    # -----------------------------------------------------------------
    st.subheader("Customer Response Classification")

    def _classify_customer(row):
        if row["uplift_score"] > 0 and row["treatment_effect"] > 0:
            return "Persuadable"
        elif row["uplift_score"] <= 0 and row["treatment_effect"] > 0:
            return "Sure Thing"
        elif row["uplift_score"] <= 0 and row["treatment_effect"] <= 0:
            return "Lost Cause"
        return "Sleeping Dog"

    uplift_classified = uplift.copy()
    uplift_classified["response_class"] = uplift_classified.apply(
        _classify_customer, axis=1,
    )

    class_counts = uplift_classified[
        "response_class"
    ].value_counts().reset_index()
    class_counts.columns = ["Response Class", "Count"]

    col_cpie, col_cbar = st.columns(2)
    with col_cpie:
        fig_class = px.pie(
            class_counts, values="Count", names="Response Class",
            title="Customer Response Classification",
            color="Response Class",
            color_discrete_map={
                "Persuadable": "#2ecc71",
                "Sure Thing": "#3498db",
                "Lost Cause": "#95a5a6",
                "Sleeping Dog": "#e74c3c",
            },
        )
        st.plotly_chart(fig_class, use_container_width=True)

    with col_cbar:
        # Classification by segment
        class_seg = uplift_classified.groupby(
            ["segment", "response_class"]
        ).size().reset_index(name="count")
        fig_class_seg = px.bar(
            class_seg, x="segment", y="count",
            color="response_class",
            title="Response Classification by Segment",
            barmode="stack",
            color_discrete_map={
                "Persuadable": "#2ecc71",
                "Sure Thing": "#3498db",
                "Lost Cause": "#95a5a6",
                "Sleeping Dog": "#e74c3c",
            },
        )
        st.plotly_chart(fig_class_seg, use_container_width=True)

    # -----------------------------------------------------------------
    # Top persuadable customers
    # -----------------------------------------------------------------
    st.subheader("Top 10 Persuadable Customers")
    persuadable_df = uplift[uplift["uplift_score"] > 0].nlargest(
        10, "uplift_score",
    )
    st.dataframe(persuadable_df, use_container_width=True)


def render_retention_campaign(st_module, config: Dict, data_loader=None):
    """Render CLV & Retention Campaign view.

    Combines CLV distributions, uplift modeling results, budget
    optimization outcomes, and campaign ROI metrics in a single
    integrated view for retention strategy planning.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("CLV & Retention Campaign")

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()
    uplift_data = data_loader.load_uplift_results()
    budget_data = data_loader.load_budget_results()
    clv_data = data_loader.load_clv_data()

    currency = config.get("budget", {}).get("currency", "KRW")
    default_budget = config.get("budget", {}).get("total_krw", 50_000_000)

    # =================================================================
    # Section 1: CLV Distribution Summary
    # =================================================================
    st.subheader("1. Customer Lifetime Value Overview")

    if not predictions.empty and "clv_predicted" in predictions.columns:
        total_clv = predictions["clv_predicted"].sum()
        avg_clv = predictions["clv_predicted"].mean()
        at_risk_clv = predictions[
            predictions["churn_probability"] > 0.5
        ]["clv_predicted"].sum()
        at_risk_pct = (
            at_risk_clv / total_clv * 100 if total_clv > 0 else 0
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total CLV", f"{total_clv:,.0f} {currency}")
        mc2.metric("Avg CLV", f"{avg_clv:,.0f} {currency}")
        mc3.metric("At-Risk CLV", f"{at_risk_clv:,.0f} {currency}")
        mc4.metric("At-Risk CLV %", f"{at_risk_pct:.1f}%")

        # CLV distribution by risk level
        col_clv1, col_clv2 = st.columns(2)
        with col_clv1:
            fig_clv_hist = px.histogram(
                predictions, x="clv_predicted", color="risk_level",
                nbins=40,
                title="CLV Distribution by Risk Level",
                labels={"clv_predicted": f"CLV ({currency})"},
                color_discrete_map={
                    "low": "#2ecc71", "medium": "#f39c12",
                    "high": "#e67e22", "critical": "#e74c3c",
                },
                barmode="overlay",
                opacity=0.7,
            )
            st.plotly_chart(fig_clv_hist, use_container_width=True)

        with col_clv2:
            # CLV vs churn - segment bubble chart
            seg_summary = predictions.groupby("segment").agg(
                avg_clv=("clv_predicted", "mean"),
                avg_churn=("churn_probability", "mean"),
                count=("customer_id", "count"),
                total_clv=("clv_predicted", "sum"),
            ).reset_index()

            fig_bubble = px.scatter(
                seg_summary, x="avg_churn", y="avg_clv",
                size="count", color="segment",
                title="Segment CLV vs Churn Risk (size = customers)",
                labels={
                    "avg_churn": "Avg Churn Probability",
                    "avg_clv": f"Avg CLV ({currency})",
                },
                hover_data=["total_clv", "count"],
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.warning("No CLV prediction data available.")

    # =================================================================
    # Section 2: Uplift Modeling Results
    # =================================================================
    st.subheader("2. Uplift Modeling & Treatment Effectiveness")

    if not uplift_data.empty:
        uc1, uc2, uc3 = st.columns(3)
        avg_uplift = uplift_data["uplift_score"].mean()
        max_uplift = uplift_data["uplift_score"].max()
        treatable = (uplift_data["uplift_score"] > 0).sum()
        treatable_pct = treatable / len(uplift_data) * 100

        uc1.metric("Avg Uplift", f"{avg_uplift:.4f}")
        uc2.metric("Max Uplift", f"{max_uplift:.4f}")
        uc3.metric(
            "Treatable Customers",
            f"{treatable:,} ({treatable_pct:.1f}%)",
        )

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            # Uplift score by segment
            seg_up = uplift_data.groupby("segment").agg(
                avg_uplift=("uplift_score", "mean"),
                avg_treatment=("treatment_effect", "mean"),
            ).reset_index()

            fig_up_seg = go.Figure()
            fig_up_seg.add_trace(go.Bar(
                name="Avg Uplift",
                x=seg_up["segment"],
                y=seg_up["avg_uplift"],
                marker_color="#e67e22",
            ))
            fig_up_seg.add_trace(go.Bar(
                name="Avg Treatment Effect",
                x=seg_up["segment"],
                y=seg_up["avg_treatment"],
                marker_color="#9b59b6",
            ))
            fig_up_seg.update_layout(
                title="Uplift & Treatment Effect by Segment",
                barmode="group",
                yaxis_title="Score",
            )
            st.plotly_chart(fig_up_seg, use_container_width=True)

        with col_u2:
            # Uplift distribution with classification
            uplift_copy = uplift_data.copy()
            uplift_copy["category"] = np.where(
                uplift_copy["uplift_score"] > 0,
                "Persuadable",
                "Do Not Treat",
            )
            fig_up_dist = px.histogram(
                uplift_copy, x="uplift_score",
                color="category",
                nbins=30,
                title="Uplift Score Distribution",
                labels={"uplift_score": "Uplift Score"},
                color_discrete_map={
                    "Persuadable": "#2ecc71",
                    "Do Not Treat": "#e74c3c",
                },
                barmode="overlay",
                opacity=0.7,
            )
            fig_up_dist.add_vline(
                x=0, line_dash="dash", line_color="black",
            )
            st.plotly_chart(fig_up_dist, use_container_width=True)

        # Cumulative uplift curve (Qini-like)
        st.markdown("**Cumulative Uplift Curve**")
        sorted_uplift = uplift_data.sort_values(
            "uplift_score", ascending=False,
        ).reset_index(drop=True)
        sorted_uplift["cum_uplift"] = sorted_uplift[
            "uplift_score"
        ].cumsum()
        sorted_uplift["pct_treated"] = (
            np.arange(1, len(sorted_uplift) + 1) / len(sorted_uplift) * 100
        )

        fig_qini = px.line(
            sorted_uplift, x="pct_treated", y="cum_uplift",
            title="Cumulative Uplift Curve (Qini-style)",
            labels={
                "pct_treated": "% Customers Treated",
                "cum_uplift": "Cumulative Uplift",
            },
        )
        fig_qini.add_hline(
            y=0, line_dash="dash", line_color="gray",
        )
        st.plotly_chart(fig_qini, use_container_width=True)
    else:
        st.warning("No uplift modeling data available.")

    # =================================================================
    # Section 3: Budget Optimization Outcomes
    # =================================================================
    st.subheader("3. Budget Optimization Outcomes")

    if not budget_data.empty:
        total_allocated = budget_data["allocated_budget_krw"].sum()
        total_rev_saved = budget_data["expected_revenue_saved_krw"].sum()
        total_retained = budget_data["expected_retained"].sum()
        overall_roi = (
            total_rev_saved / total_allocated if total_allocated > 0 else 0
        )

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric(
            "Budget Allocated",
            f"{total_allocated:,.0f} {currency}",
        )
        bc2.metric(
            "Revenue Saved",
            f"{total_rev_saved:,.0f} {currency}",
        )
        bc3.metric("Customers Retained", f"{total_retained:,}")
        bc4.metric("Overall ROI", f"{overall_roi:.1f}x")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_budget_alloc = px.bar(
                budget_data, x="segment",
                y="allocated_budget_krw",
                title="Budget Allocation by Segment",
                color="roi",
                color_continuous_scale="Viridis",
                text="allocated_budget_krw",
                labels={
                    "segment": "Segment",
                    "allocated_budget_krw": f"Budget ({currency})",
                },
            )
            fig_budget_alloc.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_budget_alloc, use_container_width=True)

        with col_b2:
            fig_revenue = px.bar(
                budget_data, x="segment",
                y="expected_revenue_saved_krw",
                title="Expected Revenue Saved by Segment",
                color="segment",
                text="expected_revenue_saved_krw",
                labels={
                    "segment": "Segment",
                    "expected_revenue_saved_krw": f"Revenue ({currency})",
                },
            )
            fig_revenue.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

        # Budget efficiency scatter
        fig_eff = px.scatter(
            budget_data, x="allocated_budget_krw",
            y="expected_revenue_saved_krw",
            size="expected_retained",
            color="segment",
            title="Budget Efficiency: Spend vs Revenue Saved",
            labels={
                "allocated_budget_krw": f"Budget Spent ({currency})",
                "expected_revenue_saved_krw": f"Revenue Saved ({currency})",
            },
            hover_data=["roi"],
        )
        # Add break-even line (1x ROI)
        max_val = max(
            budget_data["allocated_budget_krw"].max(),
            budget_data["expected_revenue_saved_krw"].max(),
        )
        fig_eff.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Break-even (1x ROI)",
            line=dict(dash="dash", color="gray"),
        ))
        st.plotly_chart(fig_eff, use_container_width=True)

        # Allocation table
        st.dataframe(
            budget_data.style.format({
                "allocated_budget_krw": "{:,.0f}",
                "expected_revenue_saved_krw": "{:,.0f}",
                "roi": "{:.2f}x",
            }),
            use_container_width=True,
        )
    else:
        st.warning("No budget optimization data available.")

    # =================================================================
    # Section 4: Campaign ROI Metrics
    # =================================================================
    st.subheader("4. Campaign ROI Metrics")

    if not budget_data.empty:
        # ROI comparison by segment
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            fig_roi_bar = px.bar(
                budget_data.sort_values("roi", ascending=True),
                x="roi", y="segment",
                orientation="h",
                title="ROI by Segment (sorted)",
                labels={"roi": "ROI (x)", "segment": "Segment"},
                color="roi",
                color_continuous_scale="RdYlGn",
                text="roi",
            )
            fig_roi_bar.update_traces(
                texttemplate="%{text:.1f}x", textposition="outside",
            )
            st.plotly_chart(fig_roi_bar, use_container_width=True)

        with col_r2:
            # Cost per retained customer
            budget_copy = budget_data.copy()
            budget_copy["cost_per_retained"] = np.where(
                budget_copy["expected_retained"] > 0,
                budget_copy["allocated_budget_krw"]
                / budget_copy["expected_retained"],
                0,
            )
            fig_cpr = px.bar(
                budget_copy, x="segment", y="cost_per_retained",
                title="Cost per Retained Customer",
                labels={
                    "segment": "Segment",
                    "cost_per_retained": f"Cost per Retention ({currency})",
                },
                color="segment",
                text="cost_per_retained",
            )
            fig_cpr.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside",
            )
            st.plotly_chart(fig_cpr, use_container_width=True)

        # ROI summary metrics
        st.markdown("**Campaign ROI Summary**")
        roi_summary = pd.DataFrame({
            "Metric": [
                "Total Campaign Budget",
                "Total Revenue Saved",
                "Net Revenue Impact",
                "Overall ROI",
                "Total Customers Retained",
                "Avg Cost per Retention",
                "Avg Revenue per Retained Customer",
                "Highest ROI Segment",
                "Lowest ROI Segment",
            ],
            "Value": [
                f"{total_allocated:,.0f} {currency}",
                f"{total_rev_saved:,.0f} {currency}",
                f"{total_rev_saved - total_allocated:,.0f} {currency}",
                f"{overall_roi:.2f}x",
                f"{total_retained:,}",
                f"{total_allocated / max(total_retained, 1):,.0f} {currency}",
                f"{total_rev_saved / max(total_retained, 1):,.0f} {currency}",
                budget_data.loc[budget_data["roi"].idxmax(), "segment"],
                budget_data.loc[budget_data["roi"].idxmin(), "segment"],
            ],
        })
        st.dataframe(roi_summary, use_container_width=True)

        # ROI waterfall chart
        st.markdown("**Revenue Waterfall**")
        waterfall_data = budget_data.sort_values(
            "expected_revenue_saved_krw", ascending=False,
        )
        fig_waterfall = go.Figure(go.Waterfall(
            name="Revenue Impact",
            orientation="v",
            x=waterfall_data["segment"].tolist() + ["Total"],
            y=waterfall_data[
                "expected_revenue_saved_krw"
            ].tolist() + [0],
            measure=["relative"] * len(waterfall_data) + ["total"],
            text=[
                f"{v:,.0f}" for v in waterfall_data[
                    "expected_revenue_saved_krw"
                ]
            ] + [f"{total_rev_saved:,.0f}"],
            textposition="outside",
            connector=dict(line=dict(color="#3498db")),
        ))
        fig_waterfall.update_layout(
            title="Revenue Saved Waterfall by Segment",
            yaxis_title=f"Revenue ({currency})",
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Campaign effectiveness radar
        if len(budget_data) >= 3:
            st.markdown("**Segment Campaign Effectiveness Radar**")
            # Normalize metrics for radar
            max_roi = budget_data["roi"].max()
            max_retained = budget_data["expected_retained"].max()
            max_rev = budget_data["expected_revenue_saved_krw"].max()

            fig_radar = go.Figure()
            for _, row in budget_data.iterrows():
                norm_roi = (
                    row["roi"] / max_roi * 100 if max_roi > 0 else 0
                )
                norm_retained = (
                    row["expected_retained"] / max_retained * 100
                    if max_retained > 0 else 0
                )
                norm_rev = (
                    row["expected_revenue_saved_krw"] / max_rev * 100
                    if max_rev > 0 else 0
                )

                fig_radar.add_trace(go.Scatterpolar(
                    r=[norm_roi, norm_retained, norm_rev, norm_roi],
                    theta=["ROI", "Retention", "Revenue", "ROI"],
                    fill="toself",
                    name=row["segment"],
                    opacity=0.5,
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Campaign Effectiveness by Segment",
            )
            st.plotly_chart(fig_radar, use_container_width=True)


def render_churn_analytics(st_module, config: Dict, data_loader=None):
    """Render churn prediction analytics page.

    Shows detailed model predictions with churn risk scores,
    feature importance analysis, risk score distributions,
    segment-level analytics, and predictive insights.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Churn Prediction Analytics")

    if data_loader is None:
        data_loader = get_data_loader(config)

    predictions = data_loader.load_predictions()
    feature_importance = data_loader.load_feature_importance()
    model_metrics = data_loader.load_model_metrics()

    if predictions.empty:
        st.warning("No prediction data available.")
        return

    # -----------------------------------------------------------------
    # KPI Summary Row
    # -----------------------------------------------------------------
    st.subheader("Churn Risk Summary")
    k1, k2, k3, k4, k5 = st.columns(5)

    total = len(predictions)
    avg_churn = predictions["churn_probability"].mean()
    median_churn = predictions["churn_probability"].median()
    high_risk = (predictions["churn_probability"] > 0.5).sum()
    critical_risk = (predictions["churn_probability"] > 0.75).sum()

    k1.metric("Total Customers", f"{total:,}")
    k2.metric("Avg Churn Prob", f"{avg_churn:.2%}")
    k3.metric("Median Churn Prob", f"{median_churn:.2%}")
    k4.metric("High Risk (>50%)", f"{high_risk:,}")
    k5.metric("Critical (>75%)", f"{critical_risk:,}")

    # -----------------------------------------------------------------
    # Churn Risk Score Distribution - detailed histogram with thresholds
    # -----------------------------------------------------------------
    st.subheader("Churn Risk Score Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=predictions["churn_probability"],
        nbinsx=50,
        name="Churn Probability",
        marker_color="#3498db",
    ))
    # Add threshold lines
    for thresh, color, label in [
        (0.25, "#2ecc71", "Low/Medium"),
        (0.50, "#f39c12", "Medium/High"),
        (0.75, "#e74c3c", "High/Critical"),
    ]:
        fig_dist.add_vline(
            x=thresh, line_dash="dash", line_color=color,
            annotation_text=label,
        )
    fig_dist.update_layout(
        title="Distribution of Churn Risk Scores with Threshold Boundaries",
        xaxis_title="Churn Probability",
        yaxis_title="Customer Count",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # -----------------------------------------------------------------
    # Risk Level Breakdown
    # -----------------------------------------------------------------
    st.subheader("Risk Level Breakdown")
    col_pie, col_table = st.columns(2)

    risk_counts = predictions["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_counts["Percentage"] = (
        risk_counts["Count"] / risk_counts["Count"].sum() * 100
    ).round(1)

    with col_pie:
        fig_risk = px.pie(
            risk_counts, values="Count", names="Risk Level",
            title="Customer Risk Level Distribution",
            color="Risk Level",
            color_discrete_map={
                "low": "#2ecc71", "medium": "#f39c12",
                "high": "#e67e22", "critical": "#e74c3c",
            },
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_table:
        st.dataframe(
            risk_counts.style.format({"Percentage": "{:.1f}%"}),
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Churn probability density by segment
    # -----------------------------------------------------------------
    st.subheader("Churn Probability Density by Segment")
    fig_density = px.histogram(
        predictions, x="churn_probability", color="segment",
        nbins=40,
        title="Churn Probability Distribution by Segment",
        labels={"churn_probability": "Churn Probability"},
        barmode="overlay",
        opacity=0.6,
    )
    st.plotly_chart(fig_density, use_container_width=True)

    # -----------------------------------------------------------------
    # Churn probability by risk level - box plot
    # -----------------------------------------------------------------
    if "risk_level" in predictions.columns:
        st.subheader("Churn Probability by Risk Level")
        fig_box = px.box(
            predictions, x="risk_level", y="churn_probability",
            color="risk_level",
            title="Churn Probability Distribution by Risk Level",
            labels={
                "risk_level": "Risk Level",
                "churn_probability": "Churn Probability",
            },
            color_discrete_map={
                "low": "#2ecc71", "medium": "#f39c12",
                "high": "#e67e22", "critical": "#e74c3c",
            },
            category_orders={
                "risk_level": ["low", "medium", "high", "critical"],
            },
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment x Risk cross-tabulation heatmap
    # -----------------------------------------------------------------
    if "risk_level" in predictions.columns:
        st.subheader("Segment x Risk Level Cross-Tabulation")
        cross_tab = pd.crosstab(
            predictions["segment"], predictions["risk_level"],
            normalize="index",
        )
        ordered = [
            c for c in ["low", "medium", "high", "critical"]
            if c in cross_tab.columns
        ]
        cross_tab = cross_tab[ordered]

        fig_heat = px.imshow(
            cross_tab,
            title="Proportion of Risk Levels within Each Segment",
            labels=dict(x="Risk Level", y="Segment", color="Proportion"),
            color_continuous_scale="RdYlGn_r",
            text_auto=".2f",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # -----------------------------------------------------------------
    # Churn drivers correlation (numeric columns)
    # -----------------------------------------------------------------
    numeric_cols = predictions.select_dtypes(
        include=[np.number],
    ).columns.tolist()
    if len(numeric_cols) >= 3:
        st.subheader("Churn Drivers Correlation")
        corr = predictions[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            zmin=-1, zmax=1,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # -----------------------------------------------------------------
    # Feature Importance Analysis
    # -----------------------------------------------------------------
    st.subheader("Feature Importance Analysis")
    if not feature_importance.empty:
        col_bar, col_cum = st.columns(2)

        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)

        with col_bar:
            fig_fi = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top {top_n} Churn Prediction Features",
                labels={"importance": "Importance Score", "feature": "Feature"},
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        with col_cum:
            # Cumulative importance
            cum_importance = top_features["importance"].cumsum()
            total_importance = feature_importance["importance"].sum()
            cum_pct = (cum_importance / total_importance * 100).values

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=list(range(1, top_n + 1)),
                y=cum_pct,
                mode="lines+markers",
                name="Cumulative Importance %",
                fill="tozeroy",
                fillcolor="rgba(52, 152, 219, 0.2)",
                line=dict(color="#3498db"),
            ))
            fig_cum.add_hline(
                y=80, line_dash="dash", line_color="red",
                annotation_text="80% threshold",
            )
            fig_cum.update_layout(
                title="Cumulative Feature Importance",
                xaxis_title="Number of Features",
                yaxis_title="Cumulative Importance (%)",
            )
            st.plotly_chart(fig_cum, use_container_width=True)

    # -----------------------------------------------------------------
    # Segment-level Churn Analysis
    # -----------------------------------------------------------------
    st.subheader("Segment-Level Churn Risk Analysis")
    seg_analysis = predictions.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_churn=("churn_probability", "mean"),
        median_churn=("churn_probability", "median"),
        std_churn=("churn_probability", "std"),
        high_risk_count=("churn_probability", lambda x: (x > 0.5).sum()),
        critical_count=("churn_probability", lambda x: (x > 0.75).sum()),
    ).reset_index()
    seg_analysis["high_risk_pct"] = (
        seg_analysis["high_risk_count"] / seg_analysis["customer_count"] * 100
    ).round(1)

    # Segment comparison chart
    fig_seg = px.bar(
        seg_analysis.sort_values("avg_churn", ascending=True),
        x="avg_churn",
        y="segment",
        orientation="h",
        title="Average Churn Risk by Segment",
        color="avg_churn",
        color_continuous_scale="RdYlGn_r",
        text="customer_count",
        hover_data=["high_risk_count", "critical_count"],
    )
    fig_seg.update_traces(texttemplate="n=%{text}", textposition="outside")
    st.plotly_chart(fig_seg, use_container_width=True)

    # Detailed segment table
    st.dataframe(
        seg_analysis.style.format({
            "avg_churn": "{:.2%}",
            "median_churn": "{:.2%}",
            "std_churn": "{:.4f}",
            "high_risk_pct": "{:.1f}%",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Model Performance Summary
    # -----------------------------------------------------------------
    st.subheader("Model Performance Summary")
    if model_metrics:
        m1, m2, m3 = st.columns(3)
        for col, (name, metrics) in zip(
            [m1, m2, m3], model_metrics.items()
        ):
            with col:
                st.markdown(f"**{name}**")
                st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")

    # -----------------------------------------------------------------
    # Churn vs CLV Scatter
    # -----------------------------------------------------------------
    if "clv_predicted" in predictions.columns:
        st.subheader("Churn Risk vs Customer Lifetime Value")
        fig_scatter = px.scatter(
            predictions,
            x="churn_probability",
            y="clv_predicted",
            color="risk_level",
            title="Churn Probability vs Predicted CLV",
            labels={
                "churn_probability": "Churn Probability",
                "clv_predicted": "Predicted CLV (KRW)",
            },
            color_discrete_map={
                "low": "#2ecc71", "medium": "#f39c12",
                "high": "#e67e22", "critical": "#e74c3c",
            },
            hover_data=["customer_id", "segment"],
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # At-risk revenue
        at_risk_clv = predictions[
            predictions["churn_probability"] > 0.5
        ]["clv_predicted"].sum()
        total_clv = predictions["clv_predicted"].sum()
        st.warning(
            f"At-Risk Revenue (churn prob > 50%): "
            f"{at_risk_clv:,.0f} KRW "
            f"({at_risk_clv / total_clv * 100:.1f}% of total CLV)"
        )

    # -----------------------------------------------------------------
    # High Risk Customer Table
    # -----------------------------------------------------------------
    st.subheader("High Risk Customers")
    risk_threshold = st.slider(
        "Churn probability threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        key="churn_analytics_threshold",
    )
    high_risk_df = predictions[
        predictions["churn_probability"] >= risk_threshold
    ].sort_values("churn_probability", ascending=False)

    st.markdown(
        f"**{len(high_risk_df)}** customers above "
        f"threshold ({risk_threshold:.0%})"
    )
    st.dataframe(
        high_risk_df.head(50).style.format({
            "churn_probability": "{:.2%}",
            "clv_predicted": "{:,.0f}",
            "days_since_last_purchase": "{:.0f}",
            "days_since_last_login": "{:.0f}",
        }),
        use_container_width=True,
    )


def render_cohort_analysis(st_module, config: Dict, data_loader=None):
    """Render cohort analysis visualization page.

    Shows cohort retention heatmap, retention curves, cohort size
    distribution, average retention curve, and cohort metrics.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Cohort Analysis")

    if data_loader is None:
        data_loader = get_data_loader(config)

    # Load retention matrix
    retention_matrix = data_loader.load_cohort_retention_matrix()

    if retention_matrix.empty:
        st.warning("No cohort analysis data available.")
        return

    # -----------------------------------------------------------------
    # KPI Summary
    # -----------------------------------------------------------------
    st.subheader("Cohort Overview")
    c1, c2, c3, c4 = st.columns(4)

    n_cohorts = len(retention_matrix)
    n_periods = len(retention_matrix.columns)

    # Average period-1 retention
    if 1 in retention_matrix.columns:
        avg_p1_retention = retention_matrix[1].mean()
    elif len(retention_matrix.columns) > 1:
        avg_p1_retention = retention_matrix.iloc[:, 1].mean()
    else:
        avg_p1_retention = 0.0

    # Average final-period retention
    last_col = retention_matrix.columns[-1]
    avg_final_retention = retention_matrix[last_col].mean()

    c1.metric("Total Cohorts", n_cohorts)
    c2.metric("Periods Tracked", n_periods)
    c3.metric("Avg Period-1 Retention", f"{avg_p1_retention:.1%}")
    c4.metric("Avg Final Retention", f"{avg_final_retention:.1%}")

    # -----------------------------------------------------------------
    # Retention Heatmap
    # -----------------------------------------------------------------
    st.subheader("Retention Heatmap")
    # Convert to percentage for display
    heatmap_data = retention_matrix * 100

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"Period {c}" for c in heatmap_data.columns],
        y=heatmap_data.index.tolist(),
        colorscale="RdYlGn",
        text=np.round(heatmap_data.values, 1),
        texttemplate="%{text:.1f}%",
        textfont={"size": 10},
        hovertemplate=(
            "Cohort: %{y}<br>"
            "%{x}<br>"
            "Retention: %{z:.1f}%<extra></extra>"
        ),
    ))
    fig_heatmap.update_layout(
        title="Customer Retention by Cohort (%)",
        xaxis_title="Period",
        yaxis_title="Cohort",
        height=max(300, n_cohorts * 50 + 100),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # -----------------------------------------------------------------
    # Retention Curves (line chart per cohort)
    # -----------------------------------------------------------------
    st.subheader("Retention Curves by Cohort")
    fig_lines = go.Figure()
    for cohort_label in retention_matrix.index:
        values = retention_matrix.loc[cohort_label].dropna()
        fig_lines.add_trace(go.Scatter(
            x=[int(c) for c in values.index],
            y=values.values * 100,
            mode="lines+markers",
            name=str(cohort_label),
        ))
    fig_lines.update_layout(
        title="Retention Rate Over Time by Cohort",
        xaxis_title="Period",
        yaxis_title="Retention Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_lines, use_container_width=True)

    # -----------------------------------------------------------------
    # Average Retention Curve
    # -----------------------------------------------------------------
    st.subheader("Average Retention Curve")
    avg_retention = retention_matrix.mean(axis=0)

    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(
        x=[int(c) for c in avg_retention.index],
        y=avg_retention.values * 100,
        mode="lines+markers",
        name="Average Retention",
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.2)",
        line=dict(color="#2ecc71", width=3),
    ))
    fig_avg.update_layout(
        title="Average Retention Rate Across All Cohorts",
        xaxis_title="Period",
        yaxis_title="Retention Rate (%)",
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # -----------------------------------------------------------------
    # Cohort Size Distribution
    # -----------------------------------------------------------------
    if 0 in retention_matrix.columns:
        st.subheader("Cohort Sizes")
        # Period 0 retention is always 1.0, so we need raw cohort data
        # Show relative sizes from the data loader
        cohort_data = data_loader.load_cohort_data()
        if not cohort_data.empty and "customer_id" in cohort_data.columns:
            cohort_data["event_date"] = pd.to_datetime(
                cohort_data["event_date"]
            )
            first_event = cohort_data.groupby("customer_id")[
                "event_date"
            ].min().reset_index()
            first_event["cohort"] = (
                first_event["event_date"].dt.to_period("M").astype(str)
            )
            cohort_sizes = (
                first_event["cohort"].value_counts()
                .sort_index().reset_index()
            )
            cohort_sizes.columns = ["Cohort", "Customers"]

            fig_sizes = px.bar(
                cohort_sizes, x="Cohort", y="Customers",
                title="New Customers per Cohort",
                color="Customers",
                color_continuous_scale="Blues",
                text="Customers",
            )
            fig_sizes.update_traces(textposition="outside")
            st.plotly_chart(fig_sizes, use_container_width=True)

    # -----------------------------------------------------------------
    # Period-over-Period Retention Drop
    # -----------------------------------------------------------------
    st.subheader("Period-over-Period Retention Change")
    avg_ret = retention_matrix.mean(axis=0)
    if len(avg_ret) > 1:
        period_labels = [int(c) for c in avg_ret.index]
        retention_vals = avg_ret.values * 100
        drops = [0] + [
            retention_vals[i] - retention_vals[i - 1]
            for i in range(1, len(retention_vals))
        ]
        fig_drops = go.Figure()
        fig_drops.add_trace(go.Bar(
            x=[f"Period {p}" for p in period_labels],
            y=drops,
            marker_color=[
                "#e74c3c" if d < 0 else "#2ecc71" for d in drops
            ],
            text=[f"{d:+.1f}%" for d in drops],
            textposition="outside",
        ))
        fig_drops.update_layout(
            title="Retention Change Between Periods (Average)",
            xaxis_title="Period",
            yaxis_title="Change in Retention (%)",
        )
        st.plotly_chart(fig_drops, use_container_width=True)

    # -----------------------------------------------------------------
    # Retention data table
    # -----------------------------------------------------------------
    st.subheader("Retention Matrix (Raw Data)")
    display_matrix = (retention_matrix * 100).round(1)
    display_matrix.columns = [f"Period {c}" for c in display_matrix.columns]
    st.dataframe(display_matrix, use_container_width=True)


def render_realtime_scoring(st_module, config: Dict, data_loader=None):
    """Render real-time scoring & recommendations page.

    Displays three main sections:
    1. Live Scoring Status — Redis connection health, stream metrics,
       consumer throughput, and latency monitoring.
    2. Personalized Retention Offers — prioritized customer-level
       recommendations with offer details, expected uplift, cost/ROI.
    3. Model Monitoring Dashboard — drift detection history (PSI/KS),
       alert timeline, scoring distribution trends.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module

    if data_loader is None:
        data_loader = get_data_loader(config)

    st.header("Real-Time Scoring & Recommendations")
    st.markdown(
        "Live scoring status, personalized retention offers, "
        "and model monitoring dashboards."
    )

    # Create three tabs for the main sections
    tab_scoring, tab_offers, tab_monitoring = st.tabs([
        "Live Scoring Status",
        "Retention Offer Recommendations",
        "Model Monitoring",
    ])

    # ==================================================================
    # TAB 1: Live Scoring Status
    # ==================================================================
    with tab_scoring:
        _render_scoring_status_tab(st, config, data_loader)

    # ==================================================================
    # TAB 2: Personalized Retention Offer Recommendations
    # ==================================================================
    with tab_offers:
        _render_retention_offers_tab(st, config, data_loader)

    # ==================================================================
    # TAB 3: Model Monitoring Dashboard
    # ==================================================================
    with tab_monitoring:
        _render_monitoring_tab(st, config, data_loader)


def _render_scoring_status_tab(st_module, config: Dict, data_loader):
    """Render the live scoring status tab.

    Shows Redis connection health, stream metrics, throughput chart,
    latency distribution, recent scoring history, and scoring
    distribution over time.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: DashboardDataLoader instance.
    """
    st = st_module
    redis_config = config.get("redis", {})

    st.subheader("Service Health")

    # Redis connection status
    redis_status = "Unavailable"
    redis_healthy = False
    stream_len_req = 0
    stream_len_resp = 0

    try:
        import redis as redis_lib
        r = redis_lib.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            socket_connect_timeout=2,
        )
        r.ping()
        redis_healthy = True
        redis_status = "Connected"
        req_stream = redis_config.get("stream_name", "scoring_requests")
        resp_stream = redis_config.get("response_stream", "scoring_responses")
        stream_len_req = r.xlen(req_stream) if r.exists(req_stream) else 0
        stream_len_resp = r.xlen(resp_stream) if r.exists(resp_stream) else 0
    except Exception:
        redis_status = "Unavailable"

    # KPI row for service health
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if redis_healthy:
            st.success("Redis: Connected")
        else:
            st.error("Redis: Unavailable")
    with col2:
        st.metric("Request Stream", format_count(stream_len_req))
    with col3:
        st.metric("Response Stream", format_count(stream_len_resp))
    with col4:
        st.metric("Consumer Group",
                   redis_config.get("consumer_group", "scoring_consumers"))

    # Redis configuration details
    with st.expander("Redis Configuration Details"):
        st.json({
            "host": redis_config.get("host", "localhost"),
            "port": redis_config.get("port", 6379),
            "db": redis_config.get("db", 0),
            "stream_name": redis_config.get("stream_name", "scoring_requests"),
            "response_stream": redis_config.get(
                "response_stream", "scoring_responses"
            ),
            "consumer_group": redis_config.get(
                "consumer_group", "scoring_consumers"
            ),
            "consumer_batch_size": redis_config.get("consumer_batch_size", 10),
            "stream_maxlen": redis_config.get("stream_maxlen", 10000),
            "cache_ttl_seconds": redis_config.get("cache_ttl_seconds", 3600),
        })

    st.markdown("---")

    # Throughput & Latency Charts
    st.subheader("Scoring Throughput & Latency")
    throughput_df = data_loader.load_scoring_throughput()

    if not throughput_df.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            fig_throughput = go.Figure()
            fig_throughput.add_trace(go.Scatter(
                x=throughput_df["timestamp"],
                y=throughput_df["requests_per_minute"],
                mode="lines+markers",
                name="Requests/min",
                line=dict(color="#3498db", width=2),
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.1)",
            ))
            fig_throughput.update_layout(
                title="Scoring Requests per Minute",
                xaxis_title="Time",
                yaxis_title="Requests/min",
                height=350,
            )
            st.plotly_chart(fig_throughput, use_container_width=True)

        with col_right:
            fig_latency = go.Figure()
            fig_latency.add_trace(go.Scatter(
                x=throughput_df["timestamp"],
                y=throughput_df["avg_latency_ms"],
                mode="lines+markers",
                name="Avg Latency (ms)",
                line=dict(color="#e67e22", width=2),
            ))
            # Add error rate on secondary axis
            fig_latency.add_trace(go.Scatter(
                x=throughput_df["timestamp"],
                y=throughput_df["error_rate"] * 100,
                mode="lines",
                name="Error Rate (%)",
                line=dict(color="#e74c3c", width=1, dash="dot"),
                yaxis="y2",
            ))
            fig_latency.update_layout(
                title="Response Latency & Error Rate",
                xaxis_title="Time",
                yaxis_title="Latency (ms)",
                yaxis2=dict(
                    title="Error Rate (%)",
                    overlaying="y",
                    side="right",
                    range=[0, 5],
                ),
                height=350,
            )
            st.plotly_chart(fig_latency, use_container_width=True)

    st.markdown("---")

    # Recent Scoring History
    st.subheader("Recent Scoring History")
    scoring_history = data_loader.load_scoring_history()

    if not scoring_history.empty:
        # Summary KPIs from scoring history
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Scores", format_count(len(scoring_history)))
        with kpi2:
            avg_prob = scoring_history["churn_probability"].mean()
            st.metric("Avg Churn Prob", format_percentage(avg_prob))
        with kpi3:
            high_risk = (
                scoring_history["risk_level"].isin(["high", "critical"]).sum()
            )
            st.metric("High/Critical Risk",
                       format_count(high_risk))
        with kpi4:
            model_counts = scoring_history["model_type"].value_counts()
            top_model = model_counts.index[0] if len(model_counts) > 0 else "N/A"
            st.metric("Primary Model", top_model)

        # Scoring distribution chart
        col_hist, col_risk = st.columns(2)
        with col_hist:
            fig_dist = px.histogram(
                scoring_history, x="churn_probability",
                nbins=30,
                color_discrete_sequence=["#3498db"],
                title="Churn Probability Distribution (Recent Scores)",
            )
            fig_dist.update_layout(height=300)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_risk:
            risk_counts = scoring_history["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            color_map = get_risk_color("low")  # just for reference
            fig_risk = px.bar(
                risk_counts, x="risk_level", y="count",
                color="risk_level",
                color_discrete_map={
                    "low": "#2ecc71", "medium": "#f39c12",
                    "high": "#e67e22", "critical": "#e74c3c",
                },
                title="Risk Level Distribution",
            )
            fig_risk.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_risk, use_container_width=True)

        # Recent scores table
        with st.expander("Detailed Scoring Log (Latest 50)", expanded=False):
            display_df = scoring_history.sort_values(
                "scored_at", ascending=False
            ).head(50)
            st.dataframe(display_df, use_container_width=True)


def _render_retention_offers_tab(st_module, config: Dict, data_loader):
    """Render the personalized retention offers tab.

    Shows prioritized customer-level retention offers with details,
    expected uplift, cost analysis, and ROI breakdown by segment.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: DashboardDataLoader instance.
    """
    st = st_module

    st.subheader("Personalized Retention Offer Recommendations")
    st.markdown(
        "AI-driven retention offers optimized per customer based on "
        "churn risk, segment, CLV, and expected uplift."
    )

    offers = data_loader.load_retention_offers()
    recs = data_loader.load_recommendations()

    if offers.empty:
        st.warning("No retention offers available.")
        return

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["critical", "high", "medium", "low"],
            default=["critical", "high", "medium"],
        )
    with col_f2:
        all_segments = sorted(offers["segment"].unique().tolist())
        seg_filter = st.multiselect(
            "Filter by Segment",
            options=all_segments,
            default=all_segments,
        )
    with col_f3:
        all_offer_types = sorted(offers["offer_type"].unique().tolist())
        offer_filter = st.multiselect(
            "Filter by Offer Type",
            options=all_offer_types,
            default=all_offer_types,
        )

    # Apply filters
    filtered = offers[
        offers["risk_level"].isin(risk_filter)
        & offers["segment"].isin(seg_filter)
        & offers["offer_type"].isin(offer_filter)
    ].copy()

    if filtered.empty:
        st.info("No offers match the selected filters.")
        return

    # Sort by priority
    filtered = filtered.sort_values("priority_rank")

    # KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Offers", format_count(len(filtered)))
    with kpi2:
        total_cost = filtered["estimated_cost_krw"].sum()
        st.metric("Total Cost", format_currency(total_cost, "KRW"))
    with kpi3:
        total_revenue = filtered["estimated_revenue_save_krw"].sum()
        st.metric("Expected Revenue Saved",
                   format_currency(total_revenue, "KRW"))
    with kpi4:
        roi = (total_revenue / max(total_cost, 1)) - 1
        st.metric("Expected ROI", f"{roi:.1f}x")

    st.markdown("---")

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        # Offer type distribution
        offer_counts = filtered["offer_type"].value_counts().reset_index()
        offer_counts.columns = ["offer_type", "count"]
        fig_offers = px.pie(
            offer_counts, names="offer_type", values="count",
            title="Offer Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_offers.update_layout(height=350)
        st.plotly_chart(fig_offers, use_container_width=True)

    with col_right:
        # Expected uplift by segment
        seg_uplift = filtered.groupby("segment").agg(
            avg_uplift=("expected_uplift", "mean"),
            count=("customer_id", "count"),
        ).reset_index()
        fig_uplift = px.bar(
            seg_uplift, x="segment", y="avg_uplift",
            color="segment",
            text="count",
            title="Average Expected Uplift by Segment",
            labels={"avg_uplift": "Avg Expected Uplift", "count": "# Customers"},
        )
        fig_uplift.update_layout(height=350, showlegend=False)
        fig_uplift.update_traces(textposition="outside")
        st.plotly_chart(fig_uplift, use_container_width=True)

    # Cost vs Revenue Saved by segment
    seg_cost = filtered.groupby("segment").agg(
        total_cost=("estimated_cost_krw", "sum"),
        total_revenue_save=("estimated_revenue_save_krw", "sum"),
    ).reset_index()
    seg_cost["roi"] = (
        seg_cost["total_revenue_save"] / seg_cost["total_cost"].clip(lower=1)
    ).round(1)

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=seg_cost["segment"],
        y=seg_cost["total_cost"],
        name="Estimated Cost (KRW)",
        marker_color="#e74c3c",
    ))
    fig_cost.add_trace(go.Bar(
        x=seg_cost["segment"],
        y=seg_cost["total_revenue_save"],
        name="Expected Revenue Saved (KRW)",
        marker_color="#2ecc71",
    ))
    fig_cost.update_layout(
        title="Cost vs Expected Revenue Saved by Segment",
        barmode="group",
        xaxis_title="Segment",
        yaxis_title="Amount (KRW)",
        height=350,
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Scatter: churn probability vs expected uplift
    fig_scatter = px.scatter(
        filtered, x="churn_probability", y="expected_uplift",
        color="risk_level",
        size="estimated_revenue_save_krw",
        hover_data=["customer_id", "segment", "offer_type", "offer_detail"],
        color_discrete_map={
            "low": "#2ecc71", "medium": "#f39c12",
            "high": "#e67e22", "critical": "#e74c3c",
        },
        title="Churn Probability vs Expected Uplift",
        labels={
            "churn_probability": "Churn Probability",
            "expected_uplift": "Expected Uplift",
        },
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed offers table
    st.subheader("Detailed Offer Recommendations")
    st.dataframe(
        filtered[[
            "priority_rank", "customer_id", "segment", "risk_level",
            "churn_probability", "offer_type", "offer_detail",
            "expected_uplift", "estimated_cost_krw",
            "estimated_revenue_save_krw",
        ]],
        use_container_width=True,
    )

    # Individual customer lookup from recommendations
    if not recs.empty:
        st.markdown("---")
        st.subheader("Quick Recommendation Lookup")
        customer_options = recs["customer_id"].tolist()
        selected_cust = st.selectbox(
            "Select Customer", customer_options, key="rec_lookup"
        )
        cust_rec = recs[recs["customer_id"] == selected_cust]
        if not cust_rec.empty:
            row = cust_rec.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Recommendation", row.get("recommendation_type", "N/A"))
            with c2:
                st.metric("Expected Uplift",
                           format_percentage(row.get("expected_uplift", 0)))
            with c3:
                st.metric("Priority Score",
                           f"{row.get('priority_score', 0):.2f}")
            st.info(f"**Recommended Offer:** {row.get('recommended_offer', 'N/A')}")


def _render_monitoring_tab(st_module, config: Dict, data_loader):
    """Render the model monitoring dashboard tab.

    Shows drift detection history (PSI/KS), alert timeline,
    feature drift trends, and scoring quality metrics.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: DashboardDataLoader instance.
    """
    st = st_module

    st.subheader("Model Monitoring Dashboard")
    st.markdown(
        "Track model drift (PSI & KS), alert history, and scoring "
        "quality over time to ensure reliable predictions."
    )

    drift_history = data_loader.load_drift_history()
    scoring_history = data_loader.load_scoring_history()

    # Monitoring config from YAML
    mon_config = config.get("monitoring", {})
    drift_config = config.get("drift_detection", {})

    # Alert summary KPIs
    if not drift_history.empty:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            total_checks = len(drift_history)
            st.metric("Total Drift Checks", format_count(total_checks))
        with kpi2:
            red_alerts = (drift_history["alert_level"] == "red").sum()
            st.metric("Red Alerts", format_count(int(red_alerts)))
        with kpi3:
            yellow_alerts = (drift_history["alert_level"] == "yellow").sum()
            st.metric("Yellow Warnings", format_count(int(yellow_alerts)))
        with kpi4:
            latest = drift_history.iloc[-1]
            st.metric("Latest Alert Level",
                       latest["alert_level"].upper())

        st.markdown("---")

        # Drift timeline chart
        st.subheader("Drift Alert Timeline")
        alert_color_map = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}
        drift_history["color"] = drift_history["alert_level"].map(alert_color_map)

        fig_timeline = go.Figure()
        for level, color in alert_color_map.items():
            mask = drift_history["alert_level"] == level
            subset = drift_history[mask]
            if not subset.empty:
                fig_timeline.add_trace(go.Scatter(
                    x=subset["timestamp"],
                    y=subset["num_drifted_features"],
                    mode="markers",
                    name=level.capitalize(),
                    marker=dict(color=color, size=10),
                ))
        fig_timeline.update_layout(
            title="Drift Alerts Over Time",
            xaxis_title="Date",
            yaxis_title="# Drifted Features",
            height=350,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # PSI and KS trend charts
        col_psi, col_ks = st.columns(2)

        with col_psi:
            fig_psi = go.Figure()
            fig_psi.add_trace(go.Scatter(
                x=drift_history["timestamp"],
                y=drift_history["psi_mean"],
                mode="lines+markers",
                name="Mean PSI",
                line=dict(color="#9b59b6", width=2),
            ))
            # PSI thresholds
            psi_warn = drift_config.get("warning_threshold", 0.1)
            psi_alert = drift_config.get("alert_threshold", 0.2)
            fig_psi.add_hline(
                y=psi_warn, line_dash="dash",
                line_color="#f39c12",
                annotation_text=f"Warning ({psi_warn})",
            )
            fig_psi.add_hline(
                y=psi_alert, line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Alert ({psi_alert})",
            )
            fig_psi.update_layout(
                title="PSI Trend (Population Stability Index)",
                xaxis_title="Date",
                yaxis_title="Mean PSI",
                height=300,
            )
            st.plotly_chart(fig_psi, use_container_width=True)

        with col_ks:
            fig_ks = go.Figure()
            fig_ks.add_trace(go.Scatter(
                x=drift_history["timestamp"],
                y=drift_history["ks_mean"],
                mode="lines+markers",
                name="Mean KS Statistic",
                line=dict(color="#1abc9c", width=2),
            ))
            ks_config = config.get("ks_drift_detection", {})
            ks_warn = ks_config.get("warning_threshold", 0.05)
            ks_drift = ks_config.get("drift_threshold", 0.01)
            fig_ks.add_hline(
                y=ks_warn, line_dash="dash",
                line_color="#f39c12",
                annotation_text=f"Warning ({ks_warn})",
            )
            fig_ks.update_layout(
                title="KS Statistic Trend (Kolmogorov-Smirnov)",
                xaxis_title="Date",
                yaxis_title="Mean KS Statistic",
                height=300,
            )
            st.plotly_chart(fig_ks, use_container_width=True)

        # Drift history table
        with st.expander("Drift Detection History (Full)", expanded=False):
            st.dataframe(drift_history, use_container_width=True)
    else:
        st.info("No drift detection history available yet.")

    st.markdown("---")

    # Scoring quality over time
    st.subheader("Scoring Quality Metrics")
    if not scoring_history.empty and "scored_at" in scoring_history.columns:
        scoring_history["scored_at_dt"] = pd.to_datetime(
            scoring_history["scored_at"], errors="coerce"
        )
        scoring_history["score_hour"] = (
            scoring_history["scored_at_dt"].dt.floor("h")
        )

        hourly_stats = scoring_history.groupby("score_hour").agg(
            count=("churn_probability", "count"),
            mean_prob=("churn_probability", "mean"),
            std_prob=("churn_probability", "std"),
        ).reset_index()

        col_vol, col_prob = st.columns(2)

        with col_vol:
            fig_vol = px.bar(
                hourly_stats, x="score_hour", y="count",
                title="Scoring Volume Over Time",
                labels={"count": "# Scores", "score_hour": "Time"},
                color_discrete_sequence=["#3498db"],
            )
            fig_vol.update_layout(height=300)
            st.plotly_chart(fig_vol, use_container_width=True)

        with col_prob:
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=hourly_stats["score_hour"],
                y=hourly_stats["mean_prob"],
                mode="lines+markers",
                name="Mean Churn Prob",
                line=dict(color="#e67e22", width=2),
            ))
            if "std_prob" in hourly_stats.columns:
                upper = hourly_stats["mean_prob"] + hourly_stats["std_prob"].fillna(0)
                lower = (hourly_stats["mean_prob"] - hourly_stats["std_prob"].fillna(0)).clip(0)
                fig_prob.add_trace(go.Scatter(
                    x=hourly_stats["score_hour"],
                    y=upper,
                    mode="lines",
                    name="Upper Bound (+1 std)",
                    line=dict(width=0),
                    showlegend=False,
                ))
                fig_prob.add_trace(go.Scatter(
                    x=hourly_stats["score_hour"],
                    y=lower,
                    mode="lines",
                    name="Lower Bound (-1 std)",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(230,126,34,0.15)",
                    showlegend=False,
                ))
            fig_prob.update_layout(
                title="Mean Churn Probability Over Time",
                xaxis_title="Time",
                yaxis_title="Mean Churn Probability",
                height=300,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Model type usage breakdown
        model_usage = scoring_history["model_type"].value_counts().reset_index()
        model_usage.columns = ["model_type", "count"]
        fig_model = px.pie(
            model_usage, names="model_type", values="count",
            title="Model Type Usage in Recent Scoring",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_model.update_layout(height=300)
        st.plotly_chart(fig_model, use_container_width=True)
    else:
        st.info("No scoring history available yet.")

    # Monitoring configuration summary
    with st.expander("Monitoring Configuration"):
        st.json({
            "drift_detection": {
                "method": drift_config.get("method", "psi"),
                "num_bins": drift_config.get("num_bins", 10),
                "warning_threshold": drift_config.get("warning_threshold", 0.1),
                "alert_threshold": drift_config.get("alert_threshold", 0.2),
            },
            "ks_drift_detection": config.get("ks_drift_detection", {}),
            "monitoring": {
                "alert_on_yellow": mon_config.get("alert_on_yellow", False),
                "alert_on_red": mon_config.get("alert_on_red", True),
                "log_to_mlflow": mon_config.get("log_to_mlflow", True),
            },
        })


def render_model_monitoring(st_module, config: Dict, data_loader=None):
    """Render standalone model monitoring page.

    Provides drift detection history (PSI/KS), alert timeline, and
    scoring quality metrics. Delegates to _render_monitoring_tab.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    return render_monitoring_view(st_module, config, data_loader)


def render_mlflow_experiments(st_module, config: Dict, data_loader=None):
    """Render MLflow experiment comparison page.

    Shows MLflow configuration, experiment run history, metric comparison
    charts, hyperparameter analysis, model registry status, and training
    efficiency analysis from logged experiment data.

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("MLflow Experiments")

    if data_loader is None:
        data_loader = get_data_loader(config)

    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "N/A")
    experiment_name = mlflow_config.get(
        "experiment_name", "churn_prediction",
    )

    # -----------------------------------------------------------------
    # MLflow Configuration
    # -----------------------------------------------------------------
    st.subheader("MLflow Configuration")
    st.json({
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "log_models": mlflow_config.get("log_models", True),
        "log_artifacts": mlflow_config.get("log_artifacts", True),
    })

    # -----------------------------------------------------------------
    # Try loading live experiments from MLflow server
    # -----------------------------------------------------------------
    mlflow_connected = False
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()
        if experiments:
            mlflow_connected = True
            st.success("Connected to MLflow tracking server")
            exp_data = [
                {
                    "Name": e.name,
                    "ID": e.experiment_id,
                    "Lifecycle": e.lifecycle_stage,
                }
                for e in experiments
            ]
            st.dataframe(
                pd.DataFrame(exp_data), use_container_width=True,
            )
        else:
            st.info("No live experiments found on MLflow server.")
    except Exception:
        st.info(
            "MLflow tracking server not available. "
            "Showing cached experiment data from artifacts."
        )

    # -----------------------------------------------------------------
    # Experiment Run History (from artifacts or MLflow)
    # -----------------------------------------------------------------
    st.subheader("Experiment Run History")
    mlflow_runs = data_loader.load_mlflow_runs()

    if mlflow_runs.empty:
        st.warning("No experiment run data available.")
        return

    # KPI cards from runs
    kc1, kc2, kc3, kc4 = st.columns(4)
    total_runs = len(mlflow_runs)
    best_run = mlflow_runs.loc[mlflow_runs["auc"].idxmax()]
    avg_auc = mlflow_runs["auc"].mean()
    total_train_time = mlflow_runs["training_time_s"].sum()

    kc1.metric("Total Runs", total_runs)
    kc2.metric("Best AUC", f"{best_run['auc']:.4f}")
    kc3.metric("Best Model", best_run["model_type"])
    kc4.metric(
        "Total Training Time",
        f"{total_train_time:.0f}s",
    )

    # Runs table
    st.dataframe(
        mlflow_runs.style.highlight_max(
            subset=["auc", "precision", "recall", "f1_score"],
            axis=0,
        ).format({
            "auc": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1_score": "{:.4f}",
            "accuracy": "{:.4f}",
            "training_time_s": "{:.1f}",
            "params_lr": "{:.4f}",
        }),
        use_container_width=True,
    )

    # -----------------------------------------------------------------
    # Metric Comparison Charts
    # -----------------------------------------------------------------
    st.subheader("Metric Comparison Across Runs")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_auc = px.bar(
            mlflow_runs.sort_values("auc", ascending=False),
            x="model_type", y="auc",
            title="AUC by Model Type",
            color="model_type",
            text="auc",
        )
        fig_auc.update_traces(
            texttemplate="%{text:.4f}", textposition="outside",
        )
        fig_auc.add_hline(
            y=0.78, line_dash="dash", line_color="red",
            annotation_text="Threshold (0.78)",
        )
        fig_auc.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_auc, use_container_width=True)

    with col_m2:
        # Multi-metric grouped bar
        metric_cols = ["auc", "precision", "recall", "f1_score"]
        fig_multi = go.Figure()
        for mt in mlflow_runs["model_type"]:
            row = mlflow_runs[mlflow_runs["model_type"] == mt].iloc[0]
            fig_multi.add_trace(go.Bar(
                name=mt,
                x=metric_cols,
                y=[row[c] for c in metric_cols],
            ))
        fig_multi.update_layout(
            title="All Metrics by Model",
            barmode="group",
            yaxis_title="Score",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    # -----------------------------------------------------------------
    # Hyperparameter Analysis
    # -----------------------------------------------------------------
    st.subheader("Hyperparameter Analysis")
    if "params_lr" in mlflow_runs.columns:
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            fig_lr = px.scatter(
                mlflow_runs, x="params_lr", y="auc",
                color="model_type",
                size="params_epochs",
                title="Learning Rate vs AUC",
                labels={
                    "params_lr": "Learning Rate",
                    "auc": "AUC",
                },
                hover_data=["model_type", "params_epochs"],
            )
            fig_lr.update_xaxes(type="log")
            st.plotly_chart(fig_lr, use_container_width=True)

        with col_hp2:
            fig_epochs = px.scatter(
                mlflow_runs, x="params_epochs", y="auc",
                color="model_type",
                size="training_time_s",
                title="Epochs vs AUC (size = training time)",
                labels={
                    "params_epochs": "Epochs",
                    "auc": "AUC",
                },
            )
            st.plotly_chart(fig_epochs, use_container_width=True)

    # -----------------------------------------------------------------
    # Training Efficiency Analysis
    # -----------------------------------------------------------------
    st.subheader("Training Efficiency")
    col_te1, col_te2 = st.columns(2)

    with col_te1:
        fig_eff = px.scatter(
            mlflow_runs, x="training_time_s", y="auc",
            color="model_type",
            title="AUC vs Training Time",
            labels={
                "training_time_s": "Training Time (seconds)",
                "auc": "AUC",
            },
            trendline="ols",
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with col_te2:
        # AUC per second of training
        runs_copy = mlflow_runs.copy()
        runs_copy["auc_per_second"] = (
            runs_copy["auc"] / runs_copy["training_time_s"].clip(lower=0.1)
        )
        fig_aps = px.bar(
            runs_copy.sort_values("auc_per_second", ascending=False),
            x="model_type", y="auc_per_second",
            title="AUC per Training Second (Efficiency)",
            color="model_type",
            text="auc_per_second",
        )
        fig_aps.update_traces(
            texttemplate="%{text:.4f}", textposition="outside",
        )
        st.plotly_chart(fig_aps, use_container_width=True)

    # -----------------------------------------------------------------
    # Model Performance Radar (MLflow runs)
    # -----------------------------------------------------------------
    st.subheader("Model Performance Radar (MLflow Runs)")
    radar_metrics = ["auc", "precision", "recall", "f1_score", "accuracy"]
    fig_radar = go.Figure()
    for _, row in mlflow_runs.iterrows():
        values = [row[m] for m in radar_metrics]
        values.append(values[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            name=row["model_type"],
            opacity=0.5,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="MLflow Run Performance Comparison",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # -----------------------------------------------------------------
    # Run timeline
    # -----------------------------------------------------------------
    if "timestamp" in mlflow_runs.columns:
        st.subheader("Experiment Timeline")
        runs_timeline = mlflow_runs.copy()
        runs_timeline["timestamp"] = pd.to_datetime(
            runs_timeline["timestamp"], errors="coerce",
        )
        if not runs_timeline["timestamp"].isna().all():
            fig_timeline = px.scatter(
                runs_timeline, x="timestamp", y="auc",
                color="model_type",
                size="training_time_s",
                title="Model Performance Over Time",
                labels={
                    "timestamp": "Run Date",
                    "auc": "AUC",
                },
                hover_data=["model_type", "params_lr"],
            )
            fig_timeline.add_hline(
                y=0.78, line_dash="dash", line_color="red",
                annotation_text="Threshold",
            )
            st.plotly_chart(fig_timeline, use_container_width=True)


# =========================================================================
# Helper functions for A/B testing power analysis
# =========================================================================


def _compute_power_analysis(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_enrollment: int = 100,
) -> Dict[str, Any]:
    """Compute required sample size using normal approximation.

    Args:
        baseline_rate: Expected baseline churn rate.
        mde: Minimum detectable effect size.
        alpha: Significance level.
        power: Target statistical power.
        daily_enrollment: Expected enrollments per day.

    Returns:
        Dict with sample_size_per_group, total_participants,
        estimated_duration_days.
    """
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
    """Compute power at different sample sizes.

    Args:
        baseline_rate: Expected baseline churn rate.
        mde: Minimum detectable effect.
        alpha: Significance level.
        max_n: Maximum sample size to consider.
        steps: Number of points.

    Returns:
        DataFrame with n and power columns.
    """
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
    """Compute sample sizes for different MDE values.

    Args:
        baseline_rate: Baseline churn rate.
        alpha: Significance level.
        power: Target power.

    Returns:
        DataFrame with MDE and sample size columns.
    """
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
    """Apply multiple comparison correction methods.

    Args:
        p_values: List of raw p-values.
        experiment_names: Names for each experiment.
        alpha: Family-wise significance level.

    Returns:
        DataFrame with corrected p-values and significance decisions.
    """
    n = len(p_values)
    # Bonferroni
    bonferroni = [min(p * n, 1.0) for p in p_values]

    # Holm-Bonferroni
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    holm = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        holm[orig_idx] = min(p * (n - rank), 1.0)

    # Benjamini-Hochberg
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


# =========================================================================
# Helper functions for multi-channel budget allocation
# =========================================================================


def _build_channel_allocation_data(
    budget_results: pd.DataFrame,
    channel_config: Dict,
    total_budget: float,
) -> pd.DataFrame:
    """Build channel-level allocation data from config.

    Args:
        budget_results: Budget results by segment.
        channel_config: Channel configuration dict from YAML.
        total_budget: Total budget amount.

    Returns:
        DataFrame with channel, cost_per_action, roi_multiplier,
        allocated_budget, expected_actions.
    """
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


# =========================================================================
# Helper functions for budget optimization what-if analysis
# =========================================================================

def _build_whatif_scenarios(
    default_budget: float,
    current_budget: float,
    cost_multiplier: float,
    uplift_multiplier: float,
) -> list:
    """Build a list of what-if scenarios for comparison.

    Args:
        default_budget: Default budget from config.
        current_budget: Currently selected budget.
        cost_multiplier: Current cost multiplier.
        uplift_multiplier: Current uplift multiplier.

    Returns:
        List of scenario dicts with name, budget, cost_mult, uplift_mult.
    """
    scenarios = [
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
    return scenarios


def _compute_scenario_comparison(
    budget_results: pd.DataFrame,
    baseline_total: float,
    scenarios: list,
) -> pd.DataFrame:
    """Compute comparison metrics for each what-if scenario.

    Args:
        budget_results: Baseline budget allocation results.
        baseline_total: Sum of baseline allocations.
        scenarios: List of scenario parameter dicts.

    Returns:
        DataFrame with one row per scenario.
    """
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
    """Compute a budget sweep analysis.

    Args:
        budget_results: Baseline budget allocation results.
        baseline_total: Sum of baseline allocations.
        min_budget: Minimum budget for sweep.
        max_budget: Maximum budget for sweep.
        steps: Number of budget levels to test.
        cost_multiplier: Cost adjustment multiplier.
        uplift_multiplier: Uplift adjustment multiplier.

    Returns:
        DataFrame with Budget, Retained, Revenue Saved columns.
    """
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
# Main Streamlit app entry point
# =========================================================================

def main():
    """Main Streamlit application entry point."""
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit not installed. pip install streamlit")
        return

    st.set_page_config(
        page_title=get_app_title(),
        page_icon=get_page_icon("Overview"),
        layout="wide",
    )

    config = load_config()

    # Sidebar navigation with icons
    st.sidebar.title("Navigation")
    pages = get_page_list()
    page_labels = [f"{get_page_icon(p)} {p}" for p in pages]
    selected_label = st.sidebar.radio("Select Page", page_labels)
    page = pages[page_labels.index(selected_label)]

    data_loader = get_data_loader(config)

    # Sidebar info from config helpers
    sidebar_info = build_sidebar_info(config)
    churn_def = sidebar_info["churn_definition"]
    budget_info = sidebar_info["budget"]
    ew = sidebar_info["ensemble_weights"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Churn Definition**")
    st.sidebar.markdown(
        f"- No purchase: {churn_def['no_purchase_days']} days\n"
        f"- No login: {churn_def['no_login_days']} days\n"
        f"- Operator: {churn_def['operator']}"
    )
    st.sidebar.markdown("**Budget**")
    st.sidebar.markdown(
        f"- Total: {format_currency(budget_info['total_krw'], budget_info['currency'])}"
    )
    st.sidebar.markdown("**Ensemble Weights**")
    st.sidebar.markdown(
        f"- ML: {ew['ml']} | DL: {ew['dl']}"
    )

    # Manual refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Route to page
    page_map = {
        "Overview": render_overview,
        "Churn Analytics": render_churn_analytics,
        "Model Performance": render_model_performance,
        "Customer Segmentation": render_segmentation,
        "Cohort Analysis": render_cohort_analysis,
        "Budget Optimization": render_budget_optimization,
        "A/B Testing": render_ab_testing,
        "Survival Analysis": render_survival_analysis,
        "Model Monitoring": render_model_monitoring,
        "Recommendations": render_recommendations,
        "CLV Prediction": render_clv,
        "Uplift Modeling": render_uplift,
        "CLV & Retention Campaign": render_retention_campaign,
        "Real-Time Scoring": render_realtime_scoring,
        "MLflow Experiments": render_mlflow_experiments,
        "System Health": render_system_health,
    }

    render_fn = page_map.get(page, render_overview)
    render_fn(st, config, data_loader)


if __name__ == "__main__":
    main()
