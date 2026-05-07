"""
SHAP Explainer for Churn Prediction Models.

Provides model-agnostic SHAP-based explanations for trained churn models:
- Global feature importance via mean absolute SHAP values
- Individual prediction explanations (local interpretability)
- Summary plots (beeswarm), bar plots, force plots, dependence plots
- DataFrame export of SHAP values for downstream analysis

Supports both XGBoost and LightGBM backends via shap.TreeExplainer.
All plot methods save to file (non-interactive) for pipeline integration.
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class ShapExplainer:
    """SHAP-based model explainer for churn prediction models.

    Uses shap.TreeExplainer for gradient boosting models (XGBoost/LightGBM)
    to compute SHAP values efficiently. Provides global and local
    interpretability methods, plus plot generation utilities.

    Args:
        model: A trained MLChurnModel instance (with .model and .model_type).
        background_data: Training data DataFrame used as background for
            SHAP value computation.
        config: Configuration dictionary (for optional settings).
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize SHAP explainer with a trained model and background data.

        Args:
            model: Trained MLChurnModel with .model and .model_type attributes.
            background_data: Training data used as reference distribution.
            config: Optional configuration dictionary.
        """
        self.ml_model = model
        self.config = config or {}
        self.background_data = background_data
        self.feature_names = list(background_data.columns) if background_data is not None else []

        # Build the SHAP TreeExplainer from the underlying model
        self._explainer = self._create_explainer(model)
        self._shap_values_cache: Optional[np.ndarray] = None
        self._cache_key: Optional[int] = None

    def _ensure_feature_names(self, X: pd.DataFrame) -> None:
        """Ensure feature names are available for SHAP outputs."""
        if not self.feature_names:
            self.feature_names = list(X.columns)

    def _create_explainer(self, model: Any) -> shap.TreeExplainer:
        """Create a SHAP TreeExplainer from the ML model.

        Extracts the underlying XGBoost or LightGBM model object
        and wraps it in a shap.TreeExplainer.

        Args:
            model: Trained MLChurnModel instance.

        Returns:
            shap.TreeExplainer instance.
        """
        raw_model = model.model
        logger.info(
            f"Creating SHAP TreeExplainer for {model.model_type} model"
        )
        return shap.TreeExplainer(raw_model)

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the given data.

        Uses caching to avoid recomputation when called with the same data.

        Args:
            X: Feature DataFrame of shape (n_samples, n_features).

        Returns:
            numpy array of SHAP values with shape (n_samples, n_features).
        """
        self._ensure_feature_names(X)
        cache_key = id(X)
        if self._shap_values_cache is not None and self._cache_key == cache_key:
            return self._shap_values_cache

        shap_values = self._explainer.shap_values(X)

        # Handle different SHAP output formats
        # For binary classifiers, shap_values may be a list of 2 arrays
        # (one per class). We want the positive class (churn=1).
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.array(shap_values)

        self._shap_values_cache = shap_values
        self._cache_key = cache_key

        logger.info(
            f"Computed SHAP values: shape={shap_values.shape}"
        )
        return shap_values

    def global_feature_importance(
        self, X: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute global feature importance as mean |SHAP| values.

        Returns features sorted by importance in descending order.

        Args:
            X: Feature DataFrame to compute SHAP values over.

        Returns:
            OrderedDict mapping feature name to mean absolute SHAP value,
            sorted descending by importance.
        """
        shap_values = self.compute_shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Sort descending by importance
        sorted_indices = np.argsort(-mean_abs_shap)
        importance = OrderedDict()
        for idx in sorted_indices:
            importance[self.feature_names[idx]] = float(mean_abs_shap[idx])

        return importance

    def get_top_features(
        self, X: pd.DataFrame, k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get the top-k most important features.

        Args:
            X: Feature DataFrame for SHAP computation.
            k: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples, sorted descending.
        """
        importance = self.global_feature_importance(X)
        return list(importance.items())[:k]

    def top_features(
        self, X: pd.DataFrame, n: int = 10
    ) -> List[Tuple[str, float]]:
        """Backward-compatible alias used by CLI integration."""
        return self.get_top_features(X, k=n)

    def explain_individual(
        self, sample: pd.Series
    ) -> Dict[str, float]:
        """Explain a single prediction using SHAP values.

        Returns per-feature SHAP contributions and the base value
        (expected model output over the background data).

        Args:
            sample: A single-row pd.Series with feature values.

        Returns:
            Dict with feature names as keys and SHAP values as values,
            plus a 'base_value' key with the expected base prediction.
        """
        sample_df = pd.DataFrame([sample])
        shap_values = self.compute_shap_values(sample_df)
        # Invalidate cache since this was a single-sample computation
        self._shap_values_cache = None
        self._cache_key = None

        explanation = {}
        for i, feat in enumerate(self.feature_names):
            explanation[feat] = float(shap_values[0, i])

        # Get base value (expected value)
        base_value = self._explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        explanation["base_value"] = float(base_value)

        return explanation

    def get_explanation_dataframe(
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Return SHAP values as a DataFrame with feature column names.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame of SHAP values with same shape and columns as X.
        """
        shap_values = self.compute_shap_values(X)
        return pd.DataFrame(shap_values, columns=self.feature_names, index=X.index)

    def export_local_explanations(
        self,
        X: pd.DataFrame,
        output_path: str,
        prediction_probabilities: Optional[Union[np.ndarray, List[float]]] = None,
        customer_ids: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        top_n_samples: int = 5,
        top_k_features: int = 10,
    ) -> pd.DataFrame:
        """Persist representative high-risk local SHAP explanations.

        The export is intentionally row-oriented so downstream checklist and
        dashboard code can validate a stable schema without parsing nested
        objects. Each selected customer contributes its top feature-level SHAP
        drivers sorted by absolute contribution.
        """
        if X.empty:
            raise ValueError("X must contain at least one row.")

        self._ensure_feature_names(X)
        n_samples = min(max(int(top_n_samples), 1), len(X))
        k_features = min(max(int(top_k_features), 1), X.shape[1])

        if prediction_probabilities is None:
            raw_model = getattr(self.ml_model, "predict_proba", None)
            if raw_model is None:
                probabilities = np.zeros(len(X), dtype=np.float64)
            else:
                raw_probabilities = np.asarray(raw_model(X), dtype=np.float64)
                probabilities = (
                    raw_probabilities[:, 1]
                    if raw_probabilities.ndim == 2 and raw_probabilities.shape[1] > 1
                    else raw_probabilities.ravel()
                )
        else:
            probabilities = np.asarray(prediction_probabilities, dtype=np.float64).ravel()

        if probabilities.shape[0] != len(X):
            raise ValueError("prediction_probabilities length must match X.")

        if customer_ids is None:
            customer_arr = X.index.to_numpy()
        else:
            customer_arr = np.asarray(customer_ids)
            if customer_arr.shape[0] != len(X):
                raise ValueError("customer_ids length must match X.")

        selected_positions = np.argsort(-probabilities)[:n_samples]
        shap_values = self.compute_shap_values(X)
        base_value = self._explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        rows = []
        for sample_rank, pos in enumerate(selected_positions, start=1):
            contributions = shap_values[pos]
            top_feature_positions = np.argsort(-np.abs(contributions))[:k_features]
            for feature_rank, feat_pos in enumerate(top_feature_positions, start=1):
                feature = self.feature_names[feat_pos]
                shap_value = float(contributions[feat_pos])
                rows.append({
                    "customer_id": customer_arr[pos],
                    "sample_rank": sample_rank,
                    "feature_rank": feature_rank,
                    "predicted_churn_probability": float(probabilities[pos]),
                    "base_value": float(base_value),
                    "feature": feature,
                    "feature_value": float(X.iloc[pos][feature]),
                    "shap_value": shap_value,
                    "abs_shap_value": abs(shap_value),
                })

        result = pd.DataFrame(rows).sort_values(
            ["sample_rank", "abs_shap_value"],
            ascending=[True, False],
        ).reset_index(drop=True)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        logger.info("Saved local SHAP explanations to %s", output)
        return result

    # ------------------------------------------------------------------
    # Plot methods (all save to file, non-interactive)
    # ------------------------------------------------------------------

    def save_summary_plot(
        self,
        X: pd.DataFrame,
        output_path: str,
        max_display: int = 20,
    ) -> None:
        """Generate and save a SHAP summary (beeswarm) plot.

        Shows the distribution of SHAP values across all samples
        for the top features.

        Args:
            X: Feature DataFrame.
            output_path: File path for the saved PNG.
            max_display: Maximum number of features to display.
        """
        shap_values = self.compute_shap_values(X)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP summary plot to {output_path}")

    def summary_plot(
        self,
        X: pd.DataFrame,
        save_path: str,
        max_display: int = 20,
    ) -> None:
        """Backward-compatible wrapper used by older callers."""
        self.save_summary_plot(X, save_path, max_display=max_display)

    def save_force_plot(
        self,
        sample: pd.Series,
        output_path: str,
    ) -> None:
        """Generate and save a SHAP force plot for an individual prediction.

        Shows how each feature contributes to pushing the prediction
        away from the base value.

        Args:
            sample: Single sample (pd.Series) to explain.
            output_path: File path for the saved image.
        """
        sample_df = pd.DataFrame([sample])
        shap_values = self.compute_shap_values(sample_df)
        # Clear cache after single-sample computation
        self._shap_values_cache = None
        self._cache_key = None

        base_value = self._explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        # Use waterfall plot as a matplotlib-native alternative to force plot
        # which requires JS rendering
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=sample.values,
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(14, 6))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP force/waterfall plot to {output_path}")

    def save_dependence_plot(
        self,
        X: pd.DataFrame,
        feature_name: str,
        output_path: str,
        interaction_feature: Optional[str] = None,
    ) -> None:
        """Generate and save a SHAP dependence plot for a feature.

        Shows how the SHAP value of a feature varies with its value,
        optionally colored by an interaction feature.

        Args:
            X: Feature DataFrame.
            feature_name: Name of the feature to plot.
            output_path: File path for the saved PNG.
            interaction_feature: Optional feature for interaction coloring.
        """
        shap_values = self.compute_shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Saved SHAP dependence plot for '{feature_name}' to {output_path}"
        )

    def save_bar_plot(
        self,
        X: pd.DataFrame,
        output_path: str,
        max_display: int = 20,
    ) -> None:
        """Generate and save a SHAP global importance bar chart.

        Shows mean |SHAP| values as horizontal bars.

        Args:
            X: Feature DataFrame.
            output_path: File path for the saved PNG.
            max_display: Maximum number of features to display.
        """
        shap_values = self.compute_shap_values(X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP bar plot to {output_path}")

    def export_top_features(
        self,
        X: pd.DataFrame,
        output_path: str,
        k: int = 10,
    ) -> pd.DataFrame:
        """Save top-k global SHAP features to CSV or JSON.

        Returns the saved DataFrame for immediate downstream use.
        """
        top_features = self.get_top_features(X, k=k)
        result = pd.DataFrame(top_features, columns=["feature", "importance"])

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        if suffix == ".json":
            result.to_json(output, orient="records", indent=2)
        else:
            result.to_csv(output, index=False)

        logger.info("Saved SHAP top-features report to %s", output)
        return result
