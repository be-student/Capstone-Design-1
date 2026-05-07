"""
Uplift Modeling Module for E-Commerce Churn Prediction.

Implements T-Learner and S-Learner meta-learner approaches for estimating
heterogeneous treatment effects (uplift). Supports 4-quadrant customer
segmentation: persuadable, sure_thing, lost_cause, sleeping_dog.

Usage:
    model = UpliftModel(config)
    model.fit(X, treatment, y)
    uplift_scores = model.predict_uplift(X)
    segments = model.segment_customers(uplift_scores)
    auuc = model.compute_auuc(y, uplift_scores, treatment)
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for script/server use
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class UpliftModel:
    """Uplift model using T-Learner or S-Learner meta-learner approach.

    T-Learner: Trains separate models for treatment and control groups,
    estimates uplift as the difference in predicted probabilities.

    S-Learner: Trains a single model with treatment indicator as a feature,
    estimates uplift by comparing predictions with treatment=1 vs treatment=0.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses 'simulation.random_seed' for
        reproducibility and optional 'uplift' section for model params.
    learner : str
        Meta-learner type: 'auto', 't_learner' or 's_learner'. Default:
        'auto', which selects the fitted learner with the best AUUC.
    """

    VALID_LEARNERS = {"auto", "t_learner", "s_learner"}

    def __init__(self, config: dict, learner: str = "auto"):
        if learner not in self.VALID_LEARNERS:
            raise ValueError(
                f"learner must be one of {sorted(self.VALID_LEARNERS)}; got {learner!r}"
            )
        self.config = config
        self.learner = learner
        self.requested_learner = learner
        self.seed = config.get("simulation", {}).get("random_seed", 42)

        # Uplift-specific config
        uplift_cfg = config.get("uplift", {})
        self.n_estimators = uplift_cfg.get("n_estimators", 100)
        self.max_depth = uplift_cfg.get("max_depth", 4)
        self.learning_rate = uplift_cfg.get("learning_rate", 0.1)

        # Segmentation thresholds (quantile-based by default)
        self.segment_thresholds = uplift_cfg.get("segment_thresholds", {
            "high_uplift_quantile": 0.5,
            "low_uplift_quantile": 0.5,
        })

        # Models (populated after fit)
        self._treatment_model = None
        self._control_model = None
        self._single_model = None  # For S-Learner
        self._is_fitted = False
        self._feature_names = None
        self.selection_metrics_: Optional[pd.DataFrame] = None

    def _make_base_model(self):
        """Create a base classifier for uplift estimation."""
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            min_samples_leaf=10,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "UpliftModel":
        """Fit the uplift model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicator (1=treatment, 0=control).
        y : array-like of shape (n_samples,)
            Binary outcome (e.g. churn label).

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=np.float64)
        treatment_arr = np.asarray(treatment).ravel()
        y_arr = np.asarray(y).ravel()

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        treatment_mask = treatment_arr == 1
        control_mask = treatment_arr == 0

        learner_to_fit = self.requested_learner
        if learner_to_fit == "auto":
            self._fit_auto(X, treatment, y)
        elif learner_to_fit == "t_learner":
            self.learner = "t_learner"
            self._fit_t_learner(X_arr, y_arr, treatment_mask, control_mask)
            self.selection_metrics_ = None
        elif learner_to_fit == "s_learner":
            self.learner = "s_learner"
            self._fit_s_learner(X_arr, y_arr, treatment_arr)
            self.selection_metrics_ = None

        self._is_fitted = True
        logger.info("Uplift model fitted using %s approach", self.learner)
        return self

    def _fit_auto(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> None:
        """Fit candidate learners and keep the best available AUUC model."""
        comparison_rows = []
        candidates: Dict[str, "UpliftModel"] = {}
        for learner in ("t_learner", "s_learner"):
            candidate = UpliftModel(self.config, learner=learner)
            candidate.fit(X, treatment, y)
            scores = candidate.predict_uplift(X)
            comparison_rows.append({
                "learner": learner,
                "auuc": candidate.compute_auuc(y, scores, treatment),
                "mean_uplift": float(np.mean(scores)),
                "std_uplift": float(np.std(scores)),
            })
            candidates[learner] = candidate

        comparison = pd.DataFrame(comparison_rows).sort_values(
            ["auuc", "mean_uplift"], ascending=[False, False]
        ).reset_index(drop=True)
        selected = str(comparison.iloc[0]["learner"])
        chosen = candidates[selected]

        self.learner = selected
        self._treatment_model = chosen._treatment_model
        self._control_model = chosen._control_model
        self._single_model = chosen._single_model
        self._feature_names = chosen._feature_names
        self.selection_metrics_ = comparison
        logger.info("Auto-selected uplift learner %s", selected)

    def _fit_t_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment_mask: np.ndarray,
        control_mask: np.ndarray,
    ):
        """Fit T-Learner: separate models for treatment and control."""
        self._treatment_model = self._make_base_model()
        self._control_model = self._make_base_model()

        self._treatment_model.fit(X[treatment_mask], y[treatment_mask])
        self._control_model.fit(X[control_mask], y[control_mask])

    def _fit_s_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ):
        """Fit S-Learner: single model with treatment as a feature."""
        X_with_t = np.column_stack([X, treatment])
        self._single_model = self._make_base_model()
        self._single_model.fit(X_with_t, y)

    def predict_uplift(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict individual treatment effects (uplift scores).

        Uplift = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)

        For churn prediction, a negative uplift means treatment reduces churn
        (good). We return the negated difference so that positive uplift
        means the treatment is beneficial.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        uplift : np.ndarray of shape (n_samples,)
            Uplift scores. Positive = treatment helps (reduces churn).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        X_arr = np.asarray(X, dtype=np.float64)

        if self.learner == "t_learner":
            return self._predict_t_learner(X_arr)
        elif self.learner == "s_learner":
            return self._predict_s_learner(X_arr)
        raise RuntimeError(f"Unsupported fitted learner: {self.learner}")

    def _predict_t_learner(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift using T-Learner."""
        p_treatment = self._treatment_model.predict_proba(X)[:, 1]
        p_control = self._control_model.predict_proba(X)[:, 1]
        # Negative difference = treatment reduces churn = positive uplift
        return (p_control - p_treatment).astype(np.float64)

    def _predict_s_learner(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift using S-Learner."""
        n = X.shape[0]
        X_t1 = np.column_stack([X, np.ones(n)])
        X_t0 = np.column_stack([X, np.zeros(n)])

        p_treatment = self._single_model.predict_proba(X_t1)[:, 1]
        p_control = self._single_model.predict_proba(X_t0)[:, 1]
        return (p_control - p_treatment).astype(np.float64)

    def segment_customers(
        self,
        uplift_scores: Union[np.ndarray, list],
        baseline_churn_probability: Optional[Union[np.ndarray, list]] = None,
        neutral_band: Optional[float] = None,
    ) -> np.ndarray:
        """Segment customers into 4-quadrant uplift categories.

        Quadrants based on uplift score magnitude:
        - persuadable: High positive uplift (treatment helps a lot)
        - sure_thing: Low positive uplift (would be fine without treatment)
        - lost_cause: Low negative uplift (treatment doesn't help much)
        - sleeping_dog: High negative uplift (treatment hurts)

        Parameters
        ----------
        uplift_scores : array-like of shape (n_samples,)
            Predicted uplift scores from predict_uplift.

        Returns
        -------
        segments : np.ndarray of shape (n_samples,)
            Segment labels for each customer.
        """
        scores = np.asarray(uplift_scores, dtype=np.float64)

        if neutral_band is None:
            neutral_band = float(
                self.segment_thresholds.get("neutral_uplift_threshold", 0.05)
            )

        if baseline_churn_probability is not None:
            churn = np.asarray(baseline_churn_probability, dtype=np.float64)
            if churn.shape[0] != scores.shape[0]:
                raise ValueError("baseline_churn_probability length must match scores.")

            high_churn_cut = float(
                self.segment_thresholds.get("high_churn_threshold", 0.5)
            )
            segments = np.full(len(scores), "unknown", dtype=object)

            sleeping = scores < 0
            high_uplift = scores > neutral_band
            neutral_uplift = np.abs(scores) <= neutral_band
            high_churn = churn >= high_churn_cut

            segments[high_uplift & high_churn] = "persuadable"
            segments[(high_uplift & ~high_churn) | (neutral_uplift & ~high_churn)] = "sure_thing"
            segments[neutral_uplift & high_churn] = "lost_cause"
            segments[sleeping] = "sleeping_dog"

            unassigned = segments == "unknown"
            segments[unassigned & high_churn] = "persuadable"
            segments[unassigned & ~high_churn] = "sure_thing"
            return segments

        median_uplift = np.median(scores)

        segments = np.empty(len(scores), dtype=object)

        # Positive uplift = treatment beneficial
        high_positive = scores >= max(median_uplift, 0)
        low_positive = (scores >= 0) & (scores < max(median_uplift, 0))

        # Negative uplift = treatment harmful or no effect
        low_negative = (scores < 0) & (scores >= min(median_uplift, 0))
        high_negative = scores < min(median_uplift, 0)

        segments[high_positive] = "persuadable"
        segments[low_positive] = "sure_thing"
        segments[low_negative] = "lost_cause"
        segments[high_negative] = "sleeping_dog"

        # Fill any remaining with closest category
        unassigned = segments == None  # noqa: E711
        if unassigned.any():
            segments[unassigned & (scores >= 0)] = "sure_thing"
            segments[unassigned & (scores < 0)] = "lost_cause"

        return segments

    def compare_learners(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        learners: Sequence[str] = ("t_learner", "s_learner"),
    ) -> pd.DataFrame:
        """Fit multiple meta-learners and compare them on shared data."""
        rows = []
        for learner in learners:
            model = UpliftModel(self.config, learner=learner)
            model.fit(X, treatment, y)
            uplift = model.predict_uplift(X)
            rows.append(
                {
                    "learner": learner,
                    "auuc": model.compute_auuc(y, uplift, treatment),
                    "mean_uplift": float(np.mean(uplift)),
                    "std_uplift": float(np.std(uplift)),
                }
            )

        return pd.DataFrame(rows).sort_values(
            ["auuc", "mean_uplift"], ascending=[False, False]
        ).reset_index(drop=True)

    def compute_auuc(
        self,
        y: Union[pd.Series, np.ndarray],
        uplift: Union[np.ndarray, list],
        treatment: Union[pd.Series, np.ndarray],
    ) -> float:
        """Compute Area Under the Uplift Curve (AUUC).

        Calculates the area between the model's uplift curve and the
        random targeting baseline.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Binary outcome labels.
        uplift : array-like of shape (n_samples,)
            Predicted uplift scores.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicators.

        Returns
        -------
        auuc : float
            Area Under the Uplift Curve. Higher is better.
        """
        y_arr = np.asarray(y).ravel()
        uplift_arr = np.asarray(uplift).ravel()
        treatment_arr = np.asarray(treatment).ravel()

        # Sort by descending uplift score
        order = np.argsort(-uplift_arr)
        y_sorted = y_arr[order]
        t_sorted = treatment_arr[order]

        n = len(y_arr)
        n_treatment = t_sorted.sum()
        n_control = n - n_treatment

        if n_treatment == 0 or n_control == 0:
            return 0.0

        # Compute cumulative uplift curve
        cum_treatment_outcomes = np.cumsum(y_sorted * t_sorted)
        cum_control_outcomes = np.cumsum(y_sorted * (1 - t_sorted))
        cum_treatment_count = np.cumsum(t_sorted)
        cum_control_count = np.cumsum(1 - t_sorted)

        # Avoid division by zero
        cum_treatment_count = np.maximum(cum_treatment_count, 1)
        cum_control_count = np.maximum(cum_control_count, 1)

        # Uplift at each point: churn reduction from treatment.
        # predict_uplift uses control_churn - treatment_churn, so AUUC must
        # use the same sign. A negative area means the ranking targets
        # customers harmed by treatment first.
        uplift_curve = (
            cum_control_outcomes / cum_control_count
            - cum_treatment_outcomes / cum_treatment_count
        )

        # AUUC = area between model curve and random baseline
        # Using trapezoidal integration normalized by n
        fractions = np.arange(1, n + 1) / n
        # np.trapezoid (NumPy 2.0+) replaces deprecated np.trapz
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auuc = float(_trapz(uplift_curve, fractions))

        return auuc

    @staticmethod
    def analyze_persuadables(
        X: Union[pd.DataFrame, np.ndarray],
        uplift_scores: Union[np.ndarray, list],
        segments: Union[np.ndarray, list, pd.Series],
        baseline_churn_probability: Optional[
            Union[np.ndarray, list, pd.Series]
        ] = None,
        customer_ids: Optional[Sequence[str]] = None,
        top_features: int = 10,
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """Summarize Persuadables characteristics for targeting criteria.

        Returns segment-level data that can be persisted by callers as CSV or
        JSON. Feature lift is computed as Persuadables mean minus the rest of
        the population, making the targeting criteria directly data-backed.
        """
        scores = np.asarray(uplift_scores, dtype=np.float64).ravel()
        seg_arr = np.asarray(segments, dtype=object).ravel()
        if len(scores) != len(seg_arr):
            raise ValueError("uplift_scores and segments must have the same length.")

        if isinstance(X, pd.DataFrame):
            feature_df = X.reset_index(drop=True).copy()
        else:
            arr = np.asarray(X, dtype=np.float64)
            feature_df = pd.DataFrame(
                arr,
                columns=[f"feature_{i}" for i in range(arr.shape[1])],
            )
        if len(feature_df) != len(scores):
            raise ValueError("X length must match uplift_scores.")

        persuadable_mask = seg_arr == "persuadable"
        rest_mask = ~persuadable_mask
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        rows = []
        for col in numeric_cols:
            persuadable_mean = (
                float(feature_df.loc[persuadable_mask, col].mean())
                if persuadable_mask.any()
                else 0.0
            )
            rest_mean = (
                float(feature_df.loc[rest_mask, col].mean())
                if rest_mask.any()
                else 0.0
            )
            rows.append({
                "feature": col,
                "persuadable_mean": persuadable_mean,
                "non_persuadable_mean": rest_mean,
                "lift": persuadable_mean - rest_mean,
                "abs_lift": abs(persuadable_mean - rest_mean),
            })

        feature_lift = pd.DataFrame(rows)
        if not feature_lift.empty:
            feature_lift = feature_lift.sort_values(
                ["abs_lift", "feature"], ascending=[False, True]
            ).head(top_features).reset_index(drop=True)

        top_customers = pd.DataFrame({
            "customer_id": (
                np.asarray(customer_ids)
                if customer_ids is not None
                else np.arange(len(scores))
            ),
            "uplift_score": scores,
            "segment": seg_arr,
        })
        if baseline_churn_probability is not None:
            churn = np.asarray(baseline_churn_probability, dtype=np.float64).ravel()
            if len(churn) != len(scores):
                raise ValueError(
                    "baseline_churn_probability length must match scores."
                )
            top_customers["baseline_churn_probability"] = churn
        top_customers = (
            top_customers[top_customers["segment"] == "persuadable"]
            .sort_values("uplift_score", ascending=False)
            .reset_index(drop=True)
        )

        summary = {
            "total_customers": float(len(scores)),
            "persuadable_count": float(persuadable_mask.sum()),
            "persuadable_rate": (
                float(persuadable_mask.mean()) if len(scores) else 0.0
            ),
            "persuadable_mean_uplift": (
                float(scores[persuadable_mask].mean())
                if persuadable_mask.any()
                else 0.0
            ),
        }
        if baseline_churn_probability is not None and not top_customers.empty:
            summary["persuadable_mean_baseline_churn"] = float(
                top_customers["baseline_churn_probability"].mean()
            )

        return {
            "summary": summary,
            "feature_lift": feature_lift,
            "top_customers": top_customers,
        }

    def save(self, path: str) -> None:
        """Save the fitted model to disk.

        Parameters
        ----------
        path : str
            File path (without extension). Will save as .pkl.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        save_path = Path(path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".pkl")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "learner": self.learner,
            "requested_learner": self.requested_learner,
            "selection_metrics": (
                self.selection_metrics_.to_dict(orient="records")
                if self.selection_metrics_ is not None else None
            ),
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "segment_thresholds": self.segment_thresholds,
            "treatment_model": self._treatment_model,
            "control_model": self._control_model,
            "single_model": self._single_model,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Uplift model saved to %s", save_path)

    @classmethod
    def load(cls, path: str) -> "UpliftModel":
        """Load a saved uplift model from disk.

        Parameters
        ----------
        path : str
            File path (with or without .pkl extension).

        Returns
        -------
        model : UpliftModel
            Loaded model instance.
        """
        load_path = Path(path)
        if not load_path.suffix:
            load_path = load_path.with_suffix(".pkl")

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        # Reconstruct model with minimal config
        config = {
            "simulation": {"random_seed": state["seed"]},
            "uplift": {
                "n_estimators": state["n_estimators"],
                "max_depth": state["max_depth"],
                "learning_rate": state["learning_rate"],
                "segment_thresholds": state["segment_thresholds"],
            },
        }

        model = cls(config, learner=state["learner"])
        model.requested_learner = state.get("requested_learner", state["learner"])
        model._treatment_model = state["treatment_model"]
        model._control_model = state["control_model"]
        model._single_model = state["single_model"]
        model._is_fitted = state["is_fitted"]
        model._feature_names = state["feature_names"]
        if state.get("selection_metrics") is not None:
            model.selection_metrics_ = pd.DataFrame(state["selection_metrics"])

        logger.info("Uplift model loaded from %s", load_path)
        return model


# ---------------------------------------------------------------------------
# Utility: Qini Curve visualisation
# ---------------------------------------------------------------------------

def plot_qini_curve(
    y_true: Union[np.ndarray, "pd.Series"],
    uplift_scores: Union[np.ndarray, list],
    treatment: Union[np.ndarray, "pd.Series"],
    save_path: Optional[str] = None,
) -> None:
    """Plot the Qini Curve for an uplift model.

    The Qini Curve measures the incremental gain from targeting customers
    in order of their predicted uplift score, compared to a random baseline.

    The y-axis shows the cumulative incremental conversions (treated
    positives minus the proportional share of control positives).
    The x-axis shows the fraction of the population targeted.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary outcome labels (1 = churn / converted, 0 = no churn).
    uplift_scores : array-like of shape (n_samples,)
        Predicted uplift scores from :meth:`UpliftModel.predict_uplift`.
        Higher values indicate customers expected to benefit most.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator (1 = treated, 0 = control).
    save_path : str, optional
        File path to save the figure (e.g. ``"results/qini_curve.png"``).
        Supported formats: PNG, PDF, SVG (inferred from extension).
        If ``None``, the figure is displayed interactively instead.

    Returns
    -------
    None

    Example
    -------
    >>> plot_qini_curve(y_test, uplift_scores, treatment,
    ...                 save_path="results/qini_curve.png")
    """
    y_arr = np.asarray(y_true).ravel()
    scores_arr = np.asarray(uplift_scores).ravel()
    treatment_arr = np.asarray(treatment).ravel()

    n = len(y_arr)
    n_treatment = treatment_arr.sum()
    n_control = n - n_treatment

    if n_treatment == 0 or n_control == 0:
        logger.warning("Cannot plot Qini Curve: treatment or control group is empty.")
        return

    # Sort by descending uplift score
    order = np.argsort(-scores_arr)
    y_sorted = y_arr[order]
    t_sorted = treatment_arr[order]

    # Cumulative treated / control positives at each rank
    cum_treat_pos = np.cumsum(y_sorted * t_sorted)
    cum_ctrl_pos = np.cumsum(y_sorted * (1 - t_sorted))
    cum_treat_cnt = np.cumsum(t_sorted)
    cum_ctrl_cnt = np.cumsum(1 - t_sorted)

    # Qini value at each rank k for churn reduction:
    #   Q(k) = expected control churn among treated count - treated churn.
    # Guard against zero control count.
    # Use a safe divisor (minimum 1) to avoid RuntimeWarning from division;
    # the np.where mask ensures treat_ratio is 0 when no control units exist.
    safe_ctrl_cnt = np.where(cum_ctrl_cnt > 0, cum_ctrl_cnt, 1)
    treat_ratio = np.where(
        cum_ctrl_cnt > 0,
        cum_treat_cnt / safe_ctrl_cnt,
        0.0,
    )
    qini_values = cum_ctrl_pos * treat_ratio - cum_treat_pos

    fractions = np.arange(1, n + 1) / n

    # Random baseline: linear from 0 to final Qini value
    random_baseline = np.linspace(0, qini_values[-1], n)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fractions, qini_values, color="steelblue", linewidth=2,
            label="Uplift model")
    ax.plot(fractions, random_baseline, color="grey", linewidth=1.5,
            linestyle="--", label="Random targeting")
    ax.fill_between(fractions, random_baseline, qini_values,
                    alpha=0.15, color="steelblue")

    ax.set_xlabel("Fraction of population targeted", fontsize=12)
    ax.set_ylabel("Cumulative incremental conversions (Qini)", fontsize=12)
    ax.set_title("Qini Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        logger.info("Qini Curve saved to %s", out)
        plt.close(fig)
    else:
        plt.show()
