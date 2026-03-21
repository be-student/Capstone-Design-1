"""
Shared Dashboard Utility Functions.

Provides formatting, validation, configuration extraction, chart helpers,
risk classification, and page routing utilities used across all Streamlit
dashboard pages.

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# =========================================================================
# Page definitions
# =========================================================================

PAGES = [
    "Overview",
    "Churn Analytics",
    "Model Performance",
    "Customer Segmentation",
    "Cohort Analysis",
    "Budget Optimization",
    "A/B Testing",
    "Survival Analysis",
    "Model Monitoring",
    "Recommendations",
    "CLV Prediction",
    "Uplift Modeling",
    "CLV & Retention Campaign",
    "Real-Time Scoring",
    "MLflow Experiments",
    "System Health",
]

PAGE_ICONS = {
    "Overview": "\U0001f4ca",              # bar chart
    "Churn Analytics": "\U0001f50d",       # magnifying glass
    "Model Performance": "\U0001f3af",     # target
    "Customer Segmentation": "\U0001f465", # people
    "Cohort Analysis": "\U0001f4c5",       # calendar
    "Budget Optimization": "\U0001f4b0",   # money bag
    "A/B Testing": "\U0001f9ea",           # test tube
    "Survival Analysis": "\U0001f4c8",     # chart increasing
    "Model Monitoring": "\U0001f6e1",      # shield
    "Recommendations": "\U0001f4e9",       # envelope
    "CLV Prediction": "\U0001f4b5",        # dollar bill
    "Uplift Modeling": "\U0001f4c8",       # chart increasing
    "CLV & Retention Campaign": "\U0001f3af",  # target
    "Real-Time Scoring": "\u26a1",         # lightning
    "MLflow Experiments": "\U0001f52c",    # microscope
    "System Health": "\U0001f3e5",          # hospital
}

# Default color palette
DEFAULT_PALETTE = [
    "#2ecc71", "#3498db", "#9b59b6", "#e67e22",
    "#e74c3c", "#1abc9c", "#f39c12", "#2c3e50",
    "#16a085", "#8e44ad",
]

RISK_COLORS = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e67e22",
    "critical": "#e74c3c",
}

# Required columns for churn prediction DataFrame
REQUIRED_PREDICTION_COLUMNS = [
    "customer_id",
    "churn_probability",
    "risk_level",
    "segment",
]

APP_TITLE = "Churn Prediction & Retention Dashboard"


# =========================================================================
# Formatting helpers
# =========================================================================

def format_currency(value: float, currency: str = "KRW") -> str:
    """Format a numeric value as currency with commas.

    Args:
        value: Numeric value to format.
        currency: Currency code (default: KRW).

    Returns:
        Formatted currency string, e.g. '50,000,000 KRW'.
    """
    return f"{value:,.0f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage string.

    Args:
        value: Decimal value (e.g. 0.1234).
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string, e.g. '12.34%'.
    """
    return f"{value * 100:.{decimals}f}%"


def format_count(value: int) -> str:
    """Format an integer count with comma separators.

    Args:
        value: Integer value to format.

    Returns:
        Comma-separated string, e.g. '1,234,567'.
    """
    return f"{value:,}"


# =========================================================================
# Risk classification
# =========================================================================

def classify_risk(
    probability: float,
    thresholds: Tuple[float, float, float] = (0.25, 0.5, 0.75),
) -> str:
    """Classify churn probability into risk level.

    Args:
        probability: Churn probability between 0 and 1.
        thresholds: Tuple of (low_max, medium_max, high_max) boundaries.
            Values <= thresholds[0] are 'low',
            <= thresholds[1] are 'medium',
            <= thresholds[2] are 'high',
            above are 'critical'.

    Returns:
        Risk level string: 'low', 'medium', 'high', or 'critical'.
    """
    low_max, med_max, high_max = thresholds
    if probability <= low_max:
        return "low"
    elif probability <= med_max:
        return "medium"
    elif probability <= high_max:
        return "high"
    else:
        return "critical"


def get_risk_color(risk_level: str) -> str:
    """Get the color code for a risk level.

    Args:
        risk_level: One of 'low', 'medium', 'high', 'critical'.

    Returns:
        Hex color string.
    """
    return RISK_COLORS.get(risk_level, "#95a5a6")


# =========================================================================
# Data validation
# =========================================================================

def validate_predictions(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate a churn predictions DataFrame.

    Checks for required columns and non-empty data.

    Args:
        df: Predictions DataFrame to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if df is None or df.empty:
        errors.append("Predictions DataFrame is empty or None.")
        return False, errors

    missing = set(REQUIRED_PREDICTION_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    return len(errors) == 0, errors


def safe_get_column(
    df: pd.DataFrame,
    column: str,
    default: Any = 0,
) -> pd.Series:
    """Safely get a column from a DataFrame with a default.

    Args:
        df: Source DataFrame.
        column: Column name to retrieve.
        default: Default value if column is missing.

    Returns:
        Series with column values or filled with default.
    """
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


# =========================================================================
# Chart helpers
# =========================================================================

def get_color_palette() -> List[str]:
    """Get the default color palette for charts.

    Returns:
        List of hex color strings.
    """
    return list(DEFAULT_PALETTE)


def get_segment_colors(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract segment-to-color mapping from configuration.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict mapping segment name to hex color.
    """
    segments = config.get("segmentation", {}).get("segments", [])
    colors = {}
    for i, seg in enumerate(segments):
        name = seg.get("name", f"segment_{i}")
        color = seg.get("color", DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
        colors[name] = color
    return colors


def compute_kpi_delta(
    current: float,
    previous: float,
) -> float:
    """Compute percentage delta between current and previous KPI values.

    Args:
        current: Current period value.
        previous: Previous period value.

    Returns:
        Percentage change (e.g. 25.0 means +25%).
    """
    if previous == 0:
        return 0.0
    return round((current - previous) / previous * 100, 1)


# =========================================================================
# Page routing
# =========================================================================

def get_page_list() -> List[str]:
    """Get the ordered list of dashboard pages.

    Returns:
        List of page name strings (11 pages).
    """
    return list(PAGES)


def get_page_icon(page_name: str) -> str:
    """Get the icon for a dashboard page.

    Args:
        page_name: Name of the page.

    Returns:
        Emoji/icon string for the page.
    """
    return PAGE_ICONS.get(page_name, "\U0001f4cb")  # default: clipboard


# =========================================================================
# Config extraction helpers
# =========================================================================

def get_churn_definition(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract churn definition parameters from config.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict with no_purchase_days, no_login_days, operator.
    """
    churn_def = config.get("churn_definition", {})
    return {
        "no_purchase_days": churn_def.get("no_purchase_days", 30),
        "no_login_days": churn_def.get("no_login_days", 60),
        "operator": churn_def.get("operator", "OR"),
    }


def get_ensemble_weights(
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """Extract ensemble model weights from config.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Tuple of (ml_weight, dl_weight).
    """
    pipeline = config.get("pipeline", {})
    ml_w = pipeline.get("ensemble_weight_ml", 0.6)
    dl_w = pipeline.get("ensemble_weight_dl", 0.4)
    return ml_w, dl_w


def get_budget_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract budget configuration.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict with total_krw, currency keys.
    """
    budget = config.get("budget", {})
    return {
        "total_krw": budget.get("total_krw", 50_000_000),
        "currency": budget.get("currency", "KRW"),
    }


# =========================================================================
# Sidebar helpers
# =========================================================================

def build_sidebar_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build sidebar information dictionary from config.

    Collects churn definition, budget, and ensemble weight info
    for display in the Streamlit sidebar.

    Args:
        config: Parsed YAML configuration dictionary.

    Returns:
        Dict with churn_definition, budget, ensemble_weights keys.
    """
    return {
        "churn_definition": get_churn_definition(config),
        "budget": get_budget_config(config),
        "ensemble_weights": {
            "ml": get_ensemble_weights(config)[0],
            "dl": get_ensemble_weights(config)[1],
        },
    }


def get_app_title() -> str:
    """Get the dashboard application title.

    Returns:
        Application title string.
    """
    return APP_TITLE
