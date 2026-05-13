"""
Model Monitoring Service.

Orchestrates drift detection by running PSI and KS checks on incoming
scoring data, logging results to MLflow, and triggering alerts when
drift exceeds configurable thresholds.

This service combines:
- PSI (Population Stability Index) drift detection for numerical features
- KS (Kolmogorov-Smirnov) test for numerical features
- Model performance degradation detection for AUC/Precision/Recall
- Configurable alert thresholds and callback mechanism
- MLflow integration for drift metric logging
- History tracking for monitoring over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

from src.monitoring.drift_detection import DriftDetector, DriftReport
from src.monitoring.ks_drift import KSDriftDetector, KSDriftReport

logger = logging.getLogger(__name__)

PERFORMANCE_METRICS = ("auc", "precision", "recall")
DEFAULT_PERFORMANCE_DROP_THRESHOLDS = {
    "auc": 0.03,
    "precision": 0.05,
    "recall": 0.05,
}


def serialize_monitoring_report(report: Any) -> Dict[str, Any]:
    """Normalize drift report objects or payloads for JSON/report consumers."""
    if report is None:
        return {}
    if hasattr(report, "to_dict"):
        return report.to_dict()
    if isinstance(report, dict):
        normalized = dict(report)
        alerts = normalized.get("feature_alerts") or normalized.get("alerts") or {}
        normalized["feature_alerts"] = alerts
        normalized["alerts"] = alerts
        normalized.setdefault("summary", {})
        return normalized
    raise TypeError(f"Unsupported report type: {type(report).__name__}")


def _performance_thresholds_from_config(
    config_or_thresholds: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """Read performance degradation thresholds from config or metric mapping."""
    thresholds = dict(DEFAULT_PERFORMANCE_DROP_THRESHOLDS)
    if not config_or_thresholds:
        return thresholds

    candidates: List[Dict[str, Any]] = []
    if any(metric in config_or_thresholds for metric in PERFORMANCE_METRICS):
        candidates.append(config_or_thresholds)

    monitoring_cfg = config_or_thresholds.get("monitoring", {})
    if isinstance(monitoring_cfg, dict):
        for key in [
            "performance_thresholds",
            "performance_degradation_thresholds",
        ]:
            if isinstance(monitoring_cfg.get(key), dict):
                candidates.append(monitoring_cfg[key])
        perf_cfg = monitoring_cfg.get("performance_degradation", {})
        if isinstance(perf_cfg, dict):
            candidates.append(perf_cfg.get("thresholds", perf_cfg))

    for key in [
        "performance_thresholds",
        "performance_degradation_thresholds",
        "performance_degradation",
    ]:
        section = config_or_thresholds.get(key, {})
        if isinstance(section, dict):
            candidates.append(section.get("thresholds", section))

    for candidate in candidates:
        for metric in PERFORMANCE_METRICS:
            if metric in candidate:
                thresholds[metric] = float(candidate[metric])
    return thresholds


@dataclass
class PerformanceMetricAlert:
    """Threshold comparison for one model performance metric."""

    metric: str
    current: float
    baseline: float
    drop: float
    threshold: float
    status: str
    current_timestamp: Optional[str] = None
    baseline_timestamp: Optional[str] = None

    @property
    def is_degraded(self) -> bool:
        """Return True when the metric drop crosses its alert threshold."""
        return self.status == "degraded"

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable metric alert payload."""
        return {
            "metric": self.metric,
            "current": self.current,
            "baseline": self.baseline,
            "drop": self.drop,
            "threshold": self.threshold,
            "status": self.status,
            "is_degraded": self.is_degraded,
            "current_timestamp": self.current_timestamp,
            "baseline_timestamp": self.baseline_timestamp,
        }


def evaluate_performance_degradation(
    performance_history: Optional[pd.DataFrame],
    thresholds: Optional[Dict[str, Any]] = None,
    model_type: Optional[str] = None,
    metrics: tuple[str, ...] = PERFORMANCE_METRICS,
    baseline_strategy: str = "best",
) -> Dict[str, Any]:
    """Compare latest AUC/Precision/Recall against historical baselines.

    Args:
        performance_history: Time series with timestamp, model/model_type,
            auc/auc_roc, precision, and recall columns.
        thresholds: Metric drop thresholds. Defaults are AUC 0.03,
            Precision 0.05, and Recall 0.05.
        model_type: Optional model filter. When omitted, the ensemble model is
            preferred if present, otherwise the latest model in the history.
        metrics: Metrics to evaluate.
        baseline_strategy: ``"best"`` compares with the best previous value;
            ``"first"`` compares with the first historical value.

    Returns:
        Dict containing metric-level current/baseline/drop/threshold/status
        fields plus aggregate performance_degradation status.
    """
    thresholds = _performance_thresholds_from_config(thresholds)
    empty = {
        "enabled": True,
        "status": "insufficient_history",
        "alert_level": AlertLevel.GREEN.value,
        "performance_degradation": False,
        "model_type": model_type,
        "baseline_strategy": baseline_strategy,
        "thresholds": thresholds,
        "metrics": {},
        "degraded_metrics": [],
    }
    if performance_history is None or performance_history.empty:
        return empty

    history = performance_history.copy()
    if "model_type" not in history.columns and "model" in history.columns:
        history = history.rename(columns={"model": "model_type"})
    if "auc" not in history.columns and "auc_roc" in history.columns:
        history["auc"] = history["auc_roc"]

    if "model_type" in history.columns:
        available_models = [str(m) for m in history["model_type"].dropna().unique()]
        if model_type is None:
            if "ensemble" in available_models:
                model_type = "ensemble"
            else:
                latest_idx = _latest_history_index(history)
                model_type = str(history.loc[latest_idx, "model_type"])
        history = history[history["model_type"].astype(str) == str(model_type)]
    else:
        model_type = model_type or "all"

    if len(history) < 2:
        empty["model_type"] = model_type
        return empty

    history = _sort_performance_history(history)
    current_row = history.iloc[-1]
    previous = history.iloc[:-1]
    current_timestamp = _string_value(current_row.get("timestamp"))

    metric_alerts: Dict[str, Dict[str, Any]] = {}
    degraded_metrics: List[str] = []
    warning_metrics: List[str] = []

    for metric in metrics:
        if metric not in history.columns:
            continue
        current = _safe_float(current_row.get(metric))
        if current is None:
            continue

        previous_metric = pd.to_numeric(previous[metric], errors="coerce")
        previous_metric = previous_metric.dropna()
        if previous_metric.empty:
            continue

        if baseline_strategy == "first":
            baseline_idx = previous_metric.index[0]
        else:
            baseline_idx = previous_metric.idxmax()
        baseline = _safe_float(history.loc[baseline_idx, metric])
        if baseline is None:
            continue

        drop = max(0.0, baseline - current)
        threshold = float(thresholds.get(metric, 0.0))
        if drop >= threshold:
            status = "degraded"
            degraded_metrics.append(metric)
        elif drop >= threshold * 0.5 and drop > 0:
            status = "warning"
            warning_metrics.append(metric)
        else:
            status = "ok"

        alert = PerformanceMetricAlert(
            metric=metric,
            current=current,
            baseline=baseline,
            drop=drop,
            threshold=threshold,
            status=status,
            current_timestamp=current_timestamp,
            baseline_timestamp=_string_value(
                history.loc[baseline_idx].get("timestamp")
            ),
        )
        metric_alerts[metric] = alert.to_dict()

    if degraded_metrics:
        overall_status = "degraded"
        alert_level = AlertLevel.RED.value
    elif warning_metrics:
        overall_status = "warning"
        alert_level = AlertLevel.YELLOW.value
    else:
        overall_status = "ok" if metric_alerts else "insufficient_history"
        alert_level = AlertLevel.GREEN.value

    return {
        "enabled": True,
        "status": overall_status,
        "alert_level": alert_level,
        "performance_degradation": bool(degraded_metrics),
        "model_type": model_type,
        "baseline_strategy": baseline_strategy,
        "thresholds": thresholds,
        "metrics": metric_alerts,
        "degraded_metrics": degraded_metrics,
        "warning_metrics": warning_metrics,
        "current_timestamp": current_timestamp,
    }


def _sort_performance_history(history: pd.DataFrame) -> pd.DataFrame:
    """Return performance history ordered by timestamp when available."""
    ordered = history.copy()
    ordered["_row_order"] = range(len(ordered))
    if "timestamp" in ordered.columns:
        ordered["_timestamp_sort"] = pd.to_datetime(
            ordered["timestamp"], errors="coerce", utc=True,
        )
        ordered = ordered.sort_values(
            ["_timestamp_sort", "_row_order"],
            na_position="first",
        )
        return ordered.drop(columns=["_timestamp_sort", "_row_order"])
    return ordered.sort_values("_row_order").drop(columns=["_row_order"])


def _latest_history_index(history: pd.DataFrame) -> Any:
    """Return index of the latest row in a performance history frame."""
    if "timestamp" not in history.columns:
        return history.index[-1]
    sortable = pd.to_datetime(history["timestamp"], errors="coerce", utc=True)
    if sortable.notna().any():
        return sortable.idxmax()
    return history.index[-1]


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to finite float, returning None when unavailable."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _string_value(value: Any) -> Optional[str]:
    """Convert non-null scalar values to strings for JSON payloads."""
    if value is None or pd.isna(value):
        return None
    return str(value)


# ---------------------------------------------------------------------------
# Drift history (append-only) export helpers
# ---------------------------------------------------------------------------

DRIFT_HISTORY_COLUMNS = [
    "timestamp",
    "feature_name",
    "psi",
    "ks_stat",
    "ks_pvalue",
    "alert_level",
    "threshold_psi_yellow",
    "threshold_psi_red",
    "threshold_ks",
    "num_drifted_features",
    "psi_mean",
    "ks_mean",
    "is_initial_check",
]


def append_drift_history(
    history_path: Union[str, Path],
    monitoring_report: Dict[str, Any],
    psi_yellow_threshold: float = 0.10,
    psi_red_threshold: float = 0.25,
    ks_threshold: float = 0.01,
) -> pd.DataFrame:
    """Append per-feature drift rows to ``drift_history.csv``.

    Each call writes one row per drift-checked feature (plus a synthetic
    ``__overall__`` summary row). The file is created on first invocation;
    subsequent invocations append while keeping the schema stable. The
    ``is_initial_check`` flag is True only for the very first run so the
    dashboard can use ``drift_trend_guard`` to suppress trend views until
    at least two checks exist.

    Args:
        history_path: Destination CSV path.
        monitoring_report: The dict written to ``monitoring_report.json``
            (must contain ``timestamp``, ``overall_alert_level``, ``psi``
            and ``ks`` sections).
        psi_yellow_threshold: PSI alert threshold (yellow).
        psi_red_threshold: PSI alert threshold (red).
        ks_threshold: KS test p-value alert threshold (drift).

    Returns:
        The newly appended rows as a DataFrame.
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = monitoring_report.get(
        "timestamp", datetime.now(timezone.utc).isoformat()
    )
    overall_level = str(
        monitoring_report.get("overall_alert_level", "green")
    ).lower()
    drifted_features = set(
        str(f) for f in monitoring_report.get("drifted_features", []) or []
    )

    # Index PSI / KS alerts by feature for easy joining
    psi_section = monitoring_report.get("psi", {}) or {}
    ks_section = monitoring_report.get("ks", {}) or {}

    psi_alerts: Dict[str, Dict[str, Any]] = {}
    for row in psi_section.get("alerts", []) or []:
        feat = str(row.get("feature", ""))
        if feat:
            psi_alerts[feat] = row

    ks_alerts: Dict[str, Dict[str, Any]] = {}
    for row in ks_section.get("alerts", []) or []:
        feat = str(row.get("feature", ""))
        if feat:
            ks_alerts[feat] = row

    features = sorted(set(psi_alerts) | set(ks_alerts))
    rows: List[Dict[str, Any]] = []
    psi_values: List[float] = []
    ks_values: List[float] = []

    for feat in features:
        psi_row = psi_alerts.get(feat, {})
        ks_row = ks_alerts.get(feat, {})
        psi_val = _safe_float(psi_row.get("psi_value", psi_row.get("psi"))) or 0.0
        ks_stat = (
            _safe_float(ks_row.get("statistic", ks_row.get("ks_stat"))) or 0.0
        )
        ks_pvalue = _safe_float(ks_row.get("p_value", ks_row.get("ks_pvalue")))
        level = "green"
        if feat in drifted_features:
            if psi_val >= psi_red_threshold:
                level = "red"
            elif psi_val >= psi_yellow_threshold:
                level = "yellow"
            else:
                level = "yellow"
        elif str(psi_row.get("level", "")).lower() in {"red", "yellow"}:
            level = str(psi_row.get("level")).lower()
        psi_values.append(psi_val)
        ks_values.append(ks_stat)
        rows.append({
            "timestamp": timestamp,
            "feature_name": feat,
            "psi": round(psi_val, 6),
            "ks_stat": round(ks_stat, 6),
            "ks_pvalue": round(ks_pvalue, 6) if ks_pvalue is not None else None,
            "alert_level": level,
            "threshold_psi_yellow": float(psi_yellow_threshold),
            "threshold_psi_red": float(psi_red_threshold),
            "threshold_ks": float(ks_threshold),
            "num_drifted_features": int(len(drifted_features)),
            "psi_mean": None,
            "ks_mean": None,
            "is_initial_check": False,
        })

    # Always emit a synthetic overall summary row so single-feature runs
    # still produce at least one drift_history line for the dashboard.
    psi_mean = float(np.mean(psi_values)) if psi_values else 0.0
    ks_mean = float(np.mean(ks_values)) if ks_values else 0.0
    rows.append({
        "timestamp": timestamp,
        "feature_name": "__overall__",
        "psi": round(psi_mean, 6),
        "ks_stat": round(ks_mean, 6),
        "ks_pvalue": None,
        "alert_level": overall_level,
        "threshold_psi_yellow": float(psi_yellow_threshold),
        "threshold_psi_red": float(psi_red_threshold),
        "threshold_ks": float(ks_threshold),
        "num_drifted_features": int(len(drifted_features)),
        "psi_mean": round(psi_mean, 6),
        "ks_mean": round(ks_mean, 6),
        "is_initial_check": False,
    })

    new_rows = pd.DataFrame(rows, columns=DRIFT_HISTORY_COLUMNS)

    is_initial = not history_path.exists()
    if is_initial:
        new_rows["is_initial_check"] = True
        new_rows.to_csv(history_path, index=False)
    else:
        try:
            existing = pd.read_csv(history_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined.to_csv(history_path, index=False)
        except Exception as exc:
            logger.warning(
                "drift_history.csv could not be read for append (%s); "
                "rewriting from scratch.",
                exc,
            )
            new_rows.to_csv(history_path, index=False)

    return new_rows


class AlertLevel(Enum):
    """Alert severity levels for drift detection.

    Attributes:
        GREEN: No significant drift detected.
        YELLOW: Moderate drift — monitor closely.
        RED: Significant drift — action required.
    """

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class MonitoringResult:
    """Result of a single monitoring check.

    Attributes:
        psi_report: PSI drift report as a dict.
        ks_report: KS drift report as a dict.
        overall_alert_level: Highest alert level across all checks.
        drifted_features: List of feature names that exceeded drift thresholds.
        timestamp: ISO-format timestamp of the check.
    """

    psi_report: Dict[str, Any]
    ks_report: Dict[str, Any]
    overall_alert_level: AlertLevel
    drifted_features: List[str]
    timestamp: str
    performance_alerts: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_drift(self) -> bool:
        """Return True if any features are drifted."""
        return len(self.drifted_features) > 0

    @property
    def has_performance_degradation(self) -> bool:
        """Return True when AUC/Precision/Recall degradation was detected."""
        return bool(self.performance_alerts.get("performance_degradation"))

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict representation."""
        return {
            "psi_report": serialize_monitoring_report(self.psi_report),
            "ks_report": serialize_monitoring_report(self.ks_report),
            "overall_alert_level": self.overall_alert_level.value,
            "drifted_features": self.drifted_features,
            "timestamp": self.timestamp,
            "has_drift": self.has_drift,
            "performance_alerts": self.performance_alerts,
            "performance_degradation": self.has_performance_degradation,
        }


class ModelMonitoringService:
    """Orchestrates drift detection, MLflow logging, and alerting.

    Combines PSI and KS drift detectors to provide comprehensive
    monitoring of feature distributions in production scoring data.

    Usage::

        service = ModelMonitoringService(config)
        service.fit(train_df)
        service.register_alert_callback(my_alert_handler)
        result = service.check(production_df)

    Args:
        config: Configuration dictionary with sections for
            drift_detection, ks_drift_detection, mlflow, and monitoring.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the monitoring service from config.

        Args:
            config: Configuration dictionary.
        """
        self.config = config

        # PSI detector config
        psi_cfg = config.get("drift_detection", {})
        self.psi_detector = DriftDetector.from_config(psi_cfg)

        # KS detector config — will be configured with feature lists on fit()
        self._ks_cfg = config.get("ks_drift_detection", {})
        self.ks_detector: Optional[KSDriftDetector] = KSDriftDetector(
            numerical_features=[],
            categorical_features=[],
            warning_threshold=self._ks_cfg.get("warning_threshold", 0.05),
            drift_threshold=self._ks_cfg.get("drift_threshold", 0.01),
        )

        # Monitoring settings
        mon_cfg = config.get("monitoring", {})
        self.alert_on_yellow = mon_cfg.get("alert_on_yellow", False)
        self.alert_on_red = mon_cfg.get("alert_on_red", True)
        self.log_to_mlflow = mon_cfg.get("log_to_mlflow", True)
        self.performance_thresholds = _performance_thresholds_from_config(
            config
        )

        # MLflow config
        self._mlflow_cfg = config.get("mlflow", {})

        # State
        self.is_fitted = False
        self._alert_callbacks: List[Callable[[MonitoringResult], None]] = []
        self.history: List[MonitoringResult] = []

    def fit(self, reference: pd.DataFrame) -> "ModelMonitoringService":
        """Fit detectors with reference (training) data.

        Stores reference distributions for both PSI and KS detectors.

        Args:
            reference: DataFrame of reference feature values.
                All columns should be numeric for PSI detection.

        Returns:
            self (for method chaining).
        """
        # Fit PSI detector with numeric columns
        numeric_cols = [
            col for col in reference.columns
            if pd.api.types.is_numeric_dtype(reference[col])
        ]
        if numeric_cols:
            self.psi_detector.fit(reference[numeric_cols])

        # Configure and fit KS detector with auto-detected feature types
        self.ks_detector = KSDriftDetector.auto_detect(
            reference,
            warning_threshold=self._ks_cfg.get("warning_threshold", 0.05),
            drift_threshold=self._ks_cfg.get("drift_threshold", 0.01),
        )

        self.is_fitted = True
        return self

    def check(
        self,
        production: pd.DataFrame,
        performance_history: Optional[pd.DataFrame] = None,
    ) -> MonitoringResult:
        """Run drift and optional performance degradation detection.

        Executes both PSI and KS checks, determines overall alert level,
        logs results to MLflow (if enabled), and triggers alert callbacks.

        Args:
            production: DataFrame of production (scoring) feature values.
            performance_history: Optional model performance time series used
                to compare latest AUC/Precision/Recall against baselines.

        Returns:
            MonitoringResult with detailed drift analysis.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ModelMonitoringService has not been fit. "
                "Call fit() with reference data first."
            )

        timestamp = datetime.now(timezone.utc).isoformat()

        # Run PSI detection
        psi_features = [
            col for col in self.psi_detector.feature_names
            if col in production.columns
        ]
        psi_report_obj: Optional[DriftReport] = None
        psi_report_dict: Dict[str, Any] = {}
        if psi_features:
            psi_report_obj = self.psi_detector.detect(
                production, features=psi_features
            )
            psi_report_dict = serialize_monitoring_report(psi_report_obj)

        # Run KS detection
        ks_report_obj: Optional[KSDriftReport] = None
        ks_report_dict: Dict[str, Any] = {}
        if self.ks_detector and self.ks_detector.reference_data is not None:
            ks_report_obj = self.ks_detector.detect(production)
            ks_report_dict = serialize_monitoring_report(ks_report_obj)

        # Determine drifted features and overall alert level
        drifted_features = []
        max_alert = AlertLevel.GREEN

        # Check PSI alerts
        if psi_report_obj:
            for feat_name, alert in psi_report_obj.feature_alerts.items():
                if alert.is_drifted:
                    if feat_name not in drifted_features:
                        drifted_features.append(feat_name)
                if alert.level == "red":
                    max_alert = AlertLevel.RED
                elif alert.level == "yellow" and max_alert != AlertLevel.RED:
                    max_alert = AlertLevel.YELLOW

        # Check KS alerts
        if ks_report_obj:
            for feat_name, alert in ks_report_obj.feature_alerts.items():
                if alert.is_drifted:
                    if feat_name not in drifted_features:
                        drifted_features.append(feat_name)
                if alert.level == "drift":
                    max_alert = AlertLevel.RED
                elif (alert.level == "warning"
                      and max_alert != AlertLevel.RED):
                    max_alert = AlertLevel.YELLOW

        performance_alerts: Dict[str, Any] = {}
        if performance_history is not None:
            performance_alerts = evaluate_performance_degradation(
                performance_history,
                thresholds=self.performance_thresholds,
            )
            if performance_alerts.get("alert_level") == AlertLevel.RED.value:
                max_alert = AlertLevel.RED
            elif (
                performance_alerts.get("alert_level") == AlertLevel.YELLOW.value
                and max_alert != AlertLevel.RED
            ):
                max_alert = AlertLevel.YELLOW

        # Build result
        result = MonitoringResult(
            psi_report=psi_report_dict,
            ks_report=ks_report_dict,
            overall_alert_level=max_alert,
            drifted_features=drifted_features,
            timestamp=timestamp,
            performance_alerts=performance_alerts,
        )

        # Log to MLflow
        if self.log_to_mlflow:
            self._log_to_mlflow(result, psi_report_obj, ks_report_obj)

        # Trigger alerts
        self._trigger_alerts(result)

        # Track history
        self.history.append(result)

        return result

    def register_alert_callback(
        self, callback: Callable[[MonitoringResult], None]
    ) -> None:
        """Register a callback to be invoked when drift alerts are triggered.

        Args:
            callback: Callable that receives a MonitoringResult.
        """
        self._alert_callbacks.append(callback)

    def get_history(self) -> List[MonitoringResult]:
        """Return the history of monitoring results.

        Returns:
            List of MonitoringResult from previous checks.
        """
        return list(self.history)

    def _log_to_mlflow(
        self,
        result: MonitoringResult,
        psi_report: Optional[DriftReport],
        ks_report: Optional[KSDriftReport],
    ) -> None:
        """Log drift metrics and tags to MLflow.

        Args:
            result: The monitoring result to log.
            psi_report: PSI drift report object (or None).
            ks_report: KS drift report object (or None).
        """
        try:
            tracking_uri = self._mlflow_cfg.get(
                "tracking_uri", "sqlite:///mlflow/mlflow.db"
            )
            mlflow.set_tracking_uri(tracking_uri)

            experiment_name = self._mlflow_cfg.get(
                "experiment_name", "churn_prediction"
            )
            mlflow.set_experiment(f"{experiment_name}_monitoring")

            mlflow.start_run(run_name=f"drift_check_{result.timestamp}")

            # Log PSI metrics
            if psi_report:
                for feat_name, psi_val in psi_report.feature_psi.items():
                    mlflow.log_metric(f"psi_{feat_name}", psi_val)

            # Log KS metrics
            if ks_report:
                for feat_name, alert in ks_report.feature_alerts.items():
                    mlflow.log_metric(
                        f"ks_stat_{feat_name}", alert.statistic
                    )
                    mlflow.log_metric(
                        f"ks_pvalue_{feat_name}", alert.p_value
                    )

            # Log summary metrics
            mlflow.log_metric(
                "num_drifted_features", len(result.drifted_features)
            )
            for metric, alert in result.performance_alerts.get(
                "metrics", {}
            ).items():
                for field_name in ["current", "baseline", "drop", "threshold"]:
                    value = _safe_float(alert.get(field_name))
                    if value is not None:
                        mlflow.log_metric(
                            f"performance_{metric}_{field_name}", value
                        )

            # Log tags
            mlflow.set_tag("drift_alert_level", result.overall_alert_level.value)
            mlflow.set_tag("has_drift", str(result.has_drift))
            mlflow.set_tag(
                "performance_degradation",
                str(result.has_performance_degradation),
            )
            degraded_metrics = result.performance_alerts.get(
                "degraded_metrics", []
            )
            if degraded_metrics:
                mlflow.set_tag(
                    "performance_degraded_metrics",
                    ",".join(degraded_metrics),
                )
            if result.drifted_features:
                mlflow.set_tag(
                    "drifted_features", ",".join(result.drifted_features)
                )

            mlflow.end_run()

        except Exception as e:
            logger.warning(f"Failed to log drift results to MLflow: {e}")
            # Ensure run is ended even on failure
            try:
                mlflow.end_run()
            except Exception:
                pass

    def _trigger_alerts(self, result: MonitoringResult) -> None:
        """Trigger registered alert callbacks based on alert level.

        Args:
            result: The monitoring result to evaluate.
        """
        should_alert = False

        if result.overall_alert_level == AlertLevel.RED and self.alert_on_red:
            should_alert = True
        elif (result.overall_alert_level == AlertLevel.YELLOW
              and self.alert_on_yellow):
            should_alert = True

        if should_alert:
            for callback in self._alert_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {e}")
