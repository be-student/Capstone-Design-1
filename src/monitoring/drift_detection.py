"""
PSI (Population Stability Index) drift detection module.

Provides functions and classes to detect data drift between training
(reference) and production feature distributions using PSI with
configurable binning strategies and threshold-based alerting.

PSI thresholds (default):
    - GREEN  : PSI < 0.10  → No significant drift
    - YELLOW : 0.10 <= PSI < 0.25  → Moderate drift, monitor closely
    - RED    : PSI >= 0.25  → Significant drift, action required
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# Default thresholds
DEFAULT_YELLOW_THRESHOLD = 0.10
DEFAULT_RED_THRESHOLD = 0.25
DEFAULT_N_BINS = 10
DEFAULT_EPSILON = 1e-6


def compute_bins(
    data: np.ndarray,
    strategy: str = "quantile",
    n_bins: int = DEFAULT_N_BINS,
) -> np.ndarray:
    """Compute bin edges for a 1-D array.

    Args:
        data: 1-D numeric array to compute bins for.
        strategy: Binning strategy — ``"quantile"`` or ``"equal_width"``.
        n_bins: Number of bins.

    Returns:
        Array of ``n_bins + 1`` bin edges covering the full data range.

    Raises:
        ValueError: If ``strategy`` is not a supported value.
    """
    if strategy == "quantile":
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(data, percentiles)
        # Ensure first/last edges cover full range
        edges[0] = min(edges[0], np.min(data)) - 1e-10
        edges[-1] = max(edges[-1], np.max(data)) + 1e-10
        # Remove duplicates while preserving bin count
        edges = np.unique(edges)
        if len(edges) < n_bins + 1:
            # Fall back to equal width if quantiles produce duplicates
            edges = np.linspace(np.min(data) - 1e-10, np.max(data) + 1e-10, n_bins + 1)
        return edges
    elif strategy == "equal_width":
        return np.linspace(np.min(data), np.max(data), n_bins + 1)
    else:
        raise ValueError(
            f"Unknown binning strategy '{strategy}'. "
            f"Supported strategies: 'quantile', 'equal_width'."
        )


def calculate_psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    bin_edges: Optional[np.ndarray] = None,
    binning_strategy: str = "quantile",
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Calculate Population Stability Index between two distributions.

    PSI = Σ (p_i - q_i) * ln(p_i / q_i)

    where p_i and q_i are the proportions in each bin for production
    and reference distributions respectively.

    Args:
        reference: 1-D array of reference (training) values.
        production: 1-D array of production (scoring) values.
        n_bins: Number of bins (ignored when ``bin_edges`` is provided).
        bin_edges: Explicit bin edges. If ``None``, edges are computed
            from ``reference`` using ``binning_strategy``.
        binning_strategy: Strategy for computing bins from reference data.
        epsilon: Small value added to prevent division by zero / log(0).

    Returns:
        Non-negative PSI value.
    """
    reference = np.asarray(reference, dtype=np.float64)
    production = np.asarray(production, dtype=np.float64)

    if bin_edges is None:
        bin_edges = compute_bins(reference, strategy=binning_strategy, n_bins=n_bins)

    # Compute proportions per bin
    ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(np.float64)
    prod_counts = np.histogram(production, bins=bin_edges)[0].astype(np.float64)

    # Normalize to proportions
    ref_props = ref_counts / ref_counts.sum()
    prod_props = prod_counts / prod_counts.sum()

    # Apply epsilon smoothing to avoid log(0) and division by zero
    ref_props = np.clip(ref_props, epsilon, None)
    prod_props = np.clip(prod_props, epsilon, None)

    # Re-normalize after clipping
    ref_props = ref_props / ref_props.sum()
    prod_props = prod_props / prod_props.sum()

    # PSI formula
    psi = np.sum((prod_props - ref_props) * np.log(prod_props / ref_props))
    return float(psi)


@dataclass
class DriftAlert:
    """Represents a drift alert for a single feature or overall metric.

    Attributes:
        psi_value: Computed PSI value.
        yellow_threshold: PSI threshold for yellow (moderate drift) alert.
        red_threshold: PSI threshold for red (significant drift) alert.
        level: Alert level — ``"green"``, ``"yellow"``, or ``"red"``.
        is_drifted: ``True`` if PSI >= red_threshold (significant drift).
    """

    psi_value: float
    yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD
    red_threshold: float = DEFAULT_RED_THRESHOLD
    level: str = field(init=False)
    is_drifted: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.psi_value >= self.red_threshold:
            self.level = "red"
            self.is_drifted = True
        elif self.psi_value >= self.yellow_threshold:
            self.level = "yellow"
            self.is_drifted = False
        else:
            self.level = "green"
            self.is_drifted = False

    def to_dict(self) -> Dict:
        """Return JSON-serializable dict."""
        return {
            "psi_value": self.psi_value,
            "level": self.level,
            "is_drifted": self.is_drifted,
            "yellow_threshold": self.yellow_threshold,
            "red_threshold": self.red_threshold,
        }


@dataclass
class DriftReport:
    """Aggregated drift report across multiple features.

    Attributes:
        feature_psi: Mapping of feature name → PSI value.
        feature_alerts: Mapping of feature name → DriftAlert.
    """

    feature_psi: Dict[str, float]
    feature_alerts: Dict[str, DriftAlert]

    def summary(self) -> Dict:
        """Return high-level summary of drift analysis.

        Returns:
            Dict with keys: total_features, drifted_features,
            yellow_features, green_features, max_psi_feature, max_psi_value.
        """
        drifted = [f for f, a in self.feature_alerts.items() if a.is_drifted]
        yellow = [f for f, a in self.feature_alerts.items() if a.level == "yellow"]
        green = [f for f, a in self.feature_alerts.items() if a.level == "green"]
        max_feat = max(self.feature_psi, key=self.feature_psi.get)
        return {
            "total_features": len(self.feature_psi),
            "drifted_features": len(drifted),
            "yellow_features": len(yellow),
            "green_features": len(green),
            "max_psi_feature": max_feat,
            "max_psi_value": self.feature_psi[max_feat],
            "drifted_feature_names": drifted,
        }

    def to_dict(self) -> Dict:
        """Return JSON-serializable representation of the full report."""
        return {
            "feature_psi": {k: float(v) for k, v in self.feature_psi.items()},
            "feature_alerts": {
                k: v.to_dict() for k, v in self.feature_alerts.items()
            },
            "alerts": {k: v.to_dict() for k, v in self.feature_alerts.items()},
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "DriftReport":
        """Reconstruct a DriftReport from serialized payload."""
        alerts_payload = payload.get("feature_alerts") or payload.get("alerts") or {}
        feature_alerts = {
            name: DriftAlert(
                psi_value=alert.get("psi_value", 0.0),
                yellow_threshold=alert.get(
                    "yellow_threshold", DEFAULT_YELLOW_THRESHOLD
                ),
                red_threshold=alert.get("red_threshold", DEFAULT_RED_THRESHOLD),
            )
            for name, alert in alerts_payload.items()
        }
        return cls(
            feature_psi={
                name: float(value)
                for name, value in payload.get("feature_psi", {}).items()
            },
            feature_alerts=feature_alerts,
        )


class DriftDetector:
    """Detect feature drift using Population Stability Index (PSI).

    Typical usage::

        detector = DriftDetector(n_bins=10, binning_strategy="quantile")
        detector.fit(train_df)
        report = detector.detect(production_df)
        print(report.summary())

    Args:
        n_bins: Number of histogram bins for PSI calculation.
        binning_strategy: ``"quantile"`` or ``"equal_width"``.
        yellow_threshold: PSI threshold for moderate drift alert.
        red_threshold: PSI threshold for significant drift alert.
        epsilon: Smoothing constant for zero-bin handling.
    """

    def __init__(
        self,
        n_bins: int = DEFAULT_N_BINS,
        binning_strategy: str = "quantile",
        yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD,
        red_threshold: float = DEFAULT_RED_THRESHOLD,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.yellow_threshold = yellow_threshold
        self.red_threshold = red_threshold
        self.epsilon = epsilon
        self.reference_data: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self._bin_edges: Dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: Dict) -> "DriftDetector":
        """Create a DriftDetector from a configuration dictionary.

        Args:
            config: Dict with optional keys: n_bins, binning_strategy,
                yellow_threshold, red_threshold, epsilon.

        Returns:
            Configured DriftDetector instance.
        """
        return cls(
            n_bins=config.get("n_bins", DEFAULT_N_BINS),
            binning_strategy=config.get("binning_strategy", "quantile"),
            yellow_threshold=config.get("yellow_threshold", DEFAULT_YELLOW_THRESHOLD),
            red_threshold=config.get("red_threshold", DEFAULT_RED_THRESHOLD),
            epsilon=config.get("epsilon", DEFAULT_EPSILON),
        )

    def fit(self, reference: pd.DataFrame) -> "DriftDetector":
        """Store reference distribution and pre-compute bin edges.

        Args:
            reference: DataFrame of reference (training) feature values.
                All columns must be numeric.

        Returns:
            self (for method chaining).
        """
        self.reference_data = reference.copy()
        self.feature_names = list(reference.columns)
        self._bin_edges = {}
        for col in self.feature_names:
            self._bin_edges[col] = compute_bins(
                reference[col].values,
                strategy=self.binning_strategy,
                n_bins=self.n_bins,
            )
        return self

    def detect(
        self,
        production: pd.DataFrame,
        features: Optional[Sequence[str]] = None,
    ) -> DriftReport:
        """Compute PSI for each feature and return a drift report.

        Args:
            production: DataFrame of production feature values.
            features: Optional subset of feature names to evaluate.
                If ``None``, all fitted features are evaluated.

        Returns:
            DriftReport with per-feature PSI values and alerts.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if self.reference_data is None:
            raise RuntimeError(
                "DriftDetector has not been fit. Call fit() with reference data first."
            )

        eval_features = features if features is not None else self.feature_names
        feature_psi: Dict[str, float] = {}
        feature_alerts: Dict[str, DriftAlert] = {}

        for col in eval_features:
            psi_val = calculate_psi(
                reference=self.reference_data[col].values,
                production=production[col].values,
                bin_edges=self._bin_edges[col],
                epsilon=self.epsilon,
            )
            feature_psi[col] = psi_val
            feature_alerts[col] = DriftAlert(
                psi_value=psi_val,
                yellow_threshold=self.yellow_threshold,
                red_threshold=self.red_threshold,
            )

        return DriftReport(
            feature_psi=feature_psi,
            feature_alerts=feature_alerts,
        )
