"""
KS (Kolmogorov-Smirnov) and Chi-Square drift detection module.

Provides functions and classes to detect data drift between reference
(training) and production feature distributions using:
- KS two-sample test for numerical features
- Chi-square test for categorical features

P-value thresholds (default):
    - no_drift  : p_value > warning_threshold (0.05)
    - warning   : drift_threshold < p_value <= warning_threshold
    - drift     : p_value <= drift_threshold (0.01)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# Default thresholds (p-value based)
DEFAULT_WARNING_THRESHOLD = 0.05
DEFAULT_DRIFT_THRESHOLD = 0.01


def compute_ks_statistic(
    reference: Union[np.ndarray, list],
    production: Union[np.ndarray, list],
) -> Tuple[float, float]:
    """Compute the two-sample Kolmogorov-Smirnov statistic.

    Uses scipy.stats.ks_2samp to compare two distributions.

    Args:
        reference: 1-D array of reference (training) values.
        production: 1-D array of production (scoring) values.

    Returns:
        Tuple of (ks_statistic, p_value).
    """
    reference = np.asarray(reference, dtype=np.float64)
    production = np.asarray(production, dtype=np.float64)
    stat, p_value = stats.ks_2samp(reference, production)
    return float(stat), float(p_value)


def compute_chi_square(
    reference: Union[np.ndarray, pd.Series, list],
    production: Union[np.ndarray, pd.Series, list],
) -> Tuple[float, float]:
    """Compute chi-square test for categorical feature drift.

    Builds frequency tables from reference and production samples,
    aligns categories (union of both), and runs a chi-square test
    comparing observed (production) vs expected (reference) proportions.

    Args:
        reference: 1-D array of categorical reference values.
        production: 1-D array of categorical production values.

    Returns:
        Tuple of (chi2_statistic, p_value).
    """
    ref_series = pd.Series(reference)
    prod_series = pd.Series(production)

    # Count frequencies
    ref_counts = ref_series.value_counts()
    prod_counts = prod_series.value_counts()

    # Union of all categories
    all_categories = sorted(set(ref_counts.index) | set(prod_counts.index))

    # Align to same categories, fill missing with 0
    ref_aligned = np.array([ref_counts.get(c, 0) for c in all_categories], dtype=np.float64)
    prod_aligned = np.array([prod_counts.get(c, 0) for c in all_categories], dtype=np.float64)

    # Compute expected counts from reference proportions scaled to production size
    ref_proportions = ref_aligned / ref_aligned.sum()
    expected = ref_proportions * prod_aligned.sum()

    # Avoid zero expected values (add small epsilon)
    epsilon = 1e-10
    expected = np.clip(expected, epsilon, None)

    # Chi-square test
    chi2_stat, p_value = stats.chisquare(prod_aligned, f_exp=expected)
    return float(chi2_stat), float(p_value)


@dataclass
class KSDriftAlert:
    """Drift alert for a single feature using KS or chi-square test.

    Attributes:
        statistic: Test statistic (KS statistic or chi-square statistic).
        p_value: P-value from the statistical test.
        feature_name: Name of the feature.
        test_type: Type of test used ("ks" or "chi_square").
        warning_threshold: P-value threshold for warning level.
        drift_threshold: P-value threshold for drift level.
        level: Alert level — "no_drift", "warning", or "drift".
        is_drifted: True if p_value <= drift_threshold.
    """

    statistic: float
    p_value: float
    feature_name: str
    test_type: str  # "ks" or "chi_square"
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD
    level: str = field(init=False)
    is_drifted: bool = field(init=False)

    def __post_init__(self) -> None:
        """Classify drift level based on p-value thresholds."""
        if self.p_value <= self.drift_threshold:
            self.level = "drift"
            self.is_drifted = True
        elif self.p_value <= self.warning_threshold:
            self.level = "warning"
            self.is_drifted = False
        else:
            self.level = "no_drift"
            self.is_drifted = False

    def to_dict(self) -> Dict:
        """Return JSON-serializable dict."""
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "feature_name": self.feature_name,
            "test_type": self.test_type,
            "level": self.level,
            "is_drifted": self.is_drifted,
            "warning_threshold": self.warning_threshold,
            "drift_threshold": self.drift_threshold,
        }


@dataclass
class KSDriftReport:
    """Aggregated drift report across numerical and categorical features.

    Attributes:
        feature_alerts: Mapping of feature name → KSDriftAlert.
    """

    feature_alerts: Dict[str, KSDriftAlert]

    def summary(self) -> Dict:
        """Return high-level summary of drift analysis.

        Returns:
            Dict with keys: total_features, drifted_features,
            numerical_drifted, categorical_drifted,
            drifted_feature_names, warning_feature_names.
        """
        all_alerts = list(self.feature_alerts.values())
        drifted = [a for a in all_alerts if a.is_drifted]
        warnings = [a for a in all_alerts if a.level == "warning"]
        num_drifted = [a for a in drifted if a.test_type == "ks"]
        cat_drifted = [a for a in drifted if a.test_type == "chi_square"]

        return {
            "total_features": len(all_alerts),
            "drifted_features": len(drifted),
            "warning_features": len(warnings),
            "numerical_drifted": len(num_drifted),
            "categorical_drifted": len(cat_drifted),
            "drifted_feature_names": [a.feature_name for a in drifted],
            "warning_feature_names": [a.feature_name for a in warnings],
        }

    def to_dict(self) -> Dict:
        """Return JSON-serializable representation of the full report."""
        return {
            "alerts": {k: v.to_dict() for k, v in self.feature_alerts.items()},
            "summary": self.summary(),
        }


class KSDriftDetector:
    """Detect feature drift using KS test (numerical) and chi-square (categorical).

    Typical usage::

        detector = KSDriftDetector(
            numerical_features=["age", "income"],
            categorical_features=["gender", "region"],
        )
        detector.fit(train_df)
        report = detector.detect(production_df)
        print(report.summary())

    Args:
        numerical_features: List of numerical feature column names.
        categorical_features: List of categorical feature column names.
        warning_threshold: P-value threshold for warning level.
        drift_threshold: P-value threshold for drift level.
    """

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    ) -> None:
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.reference_data: Optional[pd.DataFrame] = None

    @classmethod
    def from_config(cls, config: Dict) -> "KSDriftDetector":
        """Create a KSDriftDetector from a configuration dictionary.

        Args:
            config: Dict with keys: numerical_features, categorical_features,
                warning_threshold, drift_threshold.

        Returns:
            Configured KSDriftDetector instance.
        """
        return cls(
            numerical_features=config.get("numerical_features", []),
            categorical_features=config.get("categorical_features", []),
            warning_threshold=config.get("warning_threshold", DEFAULT_WARNING_THRESHOLD),
            drift_threshold=config.get("drift_threshold", DEFAULT_DRIFT_THRESHOLD),
        )

    @classmethod
    def auto_detect(
        cls,
        reference: pd.DataFrame,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    ) -> "KSDriftDetector":
        """Create a KSDriftDetector by auto-detecting feature types.

        Numerical features are those with numeric dtypes.
        Categorical features are those with object or category dtypes.

        Args:
            reference: Reference DataFrame to infer types from.
            warning_threshold: P-value threshold for warning level.
            drift_threshold: P-value threshold for drift level.

        Returns:
            KSDriftDetector with auto-detected feature lists.
        """
        numerical = [
            col for col in reference.columns
            if pd.api.types.is_numeric_dtype(reference[col])
        ]
        categorical = [
            col for col in reference.columns
            if pd.api.types.is_object_dtype(reference[col])
            or isinstance(reference[col].dtype, pd.CategoricalDtype)
        ]
        detector = cls(
            numerical_features=numerical,
            categorical_features=categorical,
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
        )
        detector.fit(reference)
        return detector

    def fit(self, reference: pd.DataFrame) -> "KSDriftDetector":
        """Store reference distribution data.

        Args:
            reference: DataFrame of reference (training) feature values.

        Returns:
            self (for method chaining).
        """
        self.reference_data = reference.copy()
        return self

    def detect(
        self,
        production: pd.DataFrame,
        features: Optional[Sequence[str]] = None,
    ) -> KSDriftReport:
        """Run KS/chi-square tests and return a drift report.

        Args:
            production: DataFrame of production feature values.
            features: Optional subset of feature names to evaluate.

        Returns:
            KSDriftReport with per-feature alerts.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.reference_data is None:
            raise RuntimeError(
                "KSDriftDetector has not been fit. Call fit() with reference data first."
            )

        # Determine which features to evaluate
        if features is not None:
            num_feats = [f for f in features if f in self.numerical_features]
            cat_feats = [f for f in features if f in self.categorical_features]
        else:
            num_feats = self.numerical_features
            cat_feats = self.categorical_features

        feature_alerts: Dict[str, KSDriftAlert] = {}

        # KS test for numerical features
        for col in num_feats:
            stat, p_value = compute_ks_statistic(
                self.reference_data[col].values,
                production[col].values,
            )
            feature_alerts[col] = KSDriftAlert(
                statistic=stat,
                p_value=p_value,
                feature_name=col,
                test_type="ks",
                warning_threshold=self.warning_threshold,
                drift_threshold=self.drift_threshold,
            )

        # Chi-square test for categorical features
        for col in cat_feats:
            stat, p_value = compute_chi_square(
                self.reference_data[col].values,
                production[col].values,
            )
            feature_alerts[col] = KSDriftAlert(
                statistic=stat,
                p_value=p_value,
                feature_name=col,
                test_type="chi_square",
                warning_threshold=self.warning_threshold,
                drift_threshold=self.drift_threshold,
            )

        return KSDriftReport(feature_alerts=feature_alerts)
