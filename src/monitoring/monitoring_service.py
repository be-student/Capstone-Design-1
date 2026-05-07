"""
Model Monitoring Service.

Orchestrates drift detection by running PSI and KS checks on incoming
scoring data, logging results to MLflow, and triggering alerts when
drift exceeds configurable thresholds.

This service combines:
- PSI (Population Stability Index) drift detection for numerical features
- KS (Kolmogorov-Smirnov) test for numerical features
- Configurable alert thresholds and callback mechanism
- MLflow integration for drift metric logging
- History tracking for monitoring over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from src.monitoring.drift_detection import DriftDetector, DriftReport
from src.monitoring.ks_drift import KSDriftDetector, KSDriftReport

logger = logging.getLogger(__name__)


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

    @property
    def has_drift(self) -> bool:
        """Return True if any features are drifted."""
        return len(self.drifted_features) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict representation."""
        return {
            "psi_report": serialize_monitoring_report(self.psi_report),
            "ks_report": serialize_monitoring_report(self.ks_report),
            "overall_alert_level": self.overall_alert_level.value,
            "drifted_features": self.drifted_features,
            "timestamp": self.timestamp,
            "has_drift": self.has_drift,
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

    def check(self, production: pd.DataFrame) -> MonitoringResult:
        """Run drift detection on production data.

        Executes both PSI and KS checks, determines overall alert level,
        logs results to MLflow (if enabled), and triggers alert callbacks.

        Args:
            production: DataFrame of production (scoring) feature values.

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

        # Build result
        result = MonitoringResult(
            psi_report=psi_report_dict,
            ks_report=ks_report_dict,
            overall_alert_level=max_alert,
            drifted_features=drifted_features,
            timestamp=timestamp,
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

            # Log tags
            mlflow.set_tag("drift_alert_level", result.overall_alert_level.value)
            mlflow.set_tag("has_drift", str(result.has_drift))
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
