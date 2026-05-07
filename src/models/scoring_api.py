"""
Real-Time Scoring API for Churn Prediction.

Provides a lightweight inference layer that:
- Loads a trained ML model (or falls back to a heuristic scorer)
- Performs single and batch churn predictions
- Returns churn probability, risk level, and recommended action
- Supports prediction caching via in-memory dict (Redis optional)
- Validates input features

Usage:
    from src.models.scoring_api import ScoringAPI

    api = ScoringAPI(config)
    result = api.predict(features={"customer_id": "C00001", ...})
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features required for scoring (excluding customer_id)
NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "days_since_last_purchase",
    "days_since_last_login",
    "total_purchases",
    "session_count_30d",
    "avg_session_duration",
    "coupon_usage_rate",
    "cart_abandonment_rate",
    "review_count",
    "cs_contact_count",
    "preferred_category_encoded",
    "segment_encoded",
]

REQUIRED_FEATURES = ["customer_id"] + NUMERIC_FEATURES

# Risk level thresholds
RISK_THRESHOLDS = {
    "critical": 0.75,
    "high": 0.50,
    "medium": 0.25,
}

# Recommended actions per risk level
RISK_ACTIONS = {
    "critical": "immediate_personal_outreach",
    "high": "win_back_campaign_with_discount",
    "medium": "engagement_email_campaign",
    "low": "standard_loyalty_program",
}


class ScoringAPI:
    """Real-time churn scoring API.

    Provides predict/predict_batch methods that return churn probability,
    risk level, and recommended retention action for each customer.

    If a trained model is available (via load_model), it is used for
    inference. Otherwise, a heuristic scorer based on feature statistics
    is used as fallback.

    Args:
        config: Application configuration dictionary.
    """

    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize scoring API.

        Args:
            config: Configuration dict (from simulator_config.yaml).
        """
        self.config = config
        self._model: Optional[Any] = None
        self._model_type: str = "heuristic"
        self._model_loaded: bool = False
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl: int = config.get("redis", {}).get(
            "cache_ttl_seconds", 3600
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model: Any = None, path: Optional[str] = None) -> None:
        """Load a trained churn model for inference.

        Args:
            model: A fitted model object with predict_proba(X) -> np.ndarray.
            path: Path to a saved model file (joblib/pickle).
        """
        if model is not None:
            self._model = model
            self._model_type = getattr(model, "model_type", "custom")
            self._model_loaded = True
            logger.info("Loaded model of type %s", self._model_type)
            return

        if path is not None:
            import joblib
            self._model = joblib.load(path)
            self._model_type = getattr(self._model, "model_type", "loaded")
            self._model_loaded = True
            logger.info("Loaded model from %s", path)
            return

        logger.warning("No model provided; using heuristic scorer.")

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model.

        Returns:
            Dict with model_type, version, loaded status.
        """
        return {
            "model_type": self._model_type,
            "model_loaded": self._model_loaded,
            "version": self.VERSION,
        }

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    def get_required_features(self) -> List[str]:
        """Return the list of required input feature names.

        Returns:
            List of feature name strings.
        """
        return list(REQUIRED_FEATURES)

    def validate_features(
        self, features: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate that a feature dict contains all required fields.

        Args:
            features: Input feature dictionary.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        for name in REQUIRED_FEATURES:
            if name not in features:
                errors.append(f"Missing required feature: {name}")
                continue

            val = features[name]
            if name == "customer_id":
                if not isinstance(val, (str, int)):
                    errors.append(
                        f"customer_id must be str or int, got {type(val).__name__}"
                    )
            else:
                if val is not None and not isinstance(val, (int, float, np.integer, np.floating)):
                    errors.append(
                        f"Feature '{name}' must be numeric, got {type(val).__name__}"
                    )

        return (len(errors) == 0, errors)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single customer for churn risk.

        Args:
            features: Dict with customer_id and numeric feature values.

        Returns:
            Dict with customer_id, churn_probability, risk_level,
            recommended_action, and timestamp.

        Raises:
            ValueError: If customer_id is missing.
            KeyError: If features dict is empty.
        """
        if not features:
            raise ValueError("Features dict must not be empty")

        if "customer_id" not in features:
            raise KeyError("customer_id is required")

        customer_id = str(features["customer_id"])
        prob = self._score_single(features)
        risk = self._classify_risk(prob)

        return {
            "customer_id": customer_id,
            "churn_probability": float(prob),
            "risk_level": risk,
            "recommended_action": RISK_ACTIONS[risk],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": self._model_type,
        }

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of customers for churn risk.

        Args:
            data: DataFrame with customer_id column and feature columns.

        Returns:
            DataFrame with customer_id, churn_probability, risk_level,
            recommended_action columns.
        """
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "customer_id", "churn_probability",
                    "risk_level", "recommended_action",
                ]
            )

        probs = self._score_batch(data)
        risks = [self._classify_risk(p) for p in probs]
        actions = [RISK_ACTIONS[r] for r in risks]

        result = pd.DataFrame({
            "customer_id": data["customer_id"].values,
            "churn_probability": probs,
            "risk_level": risks,
            "recommended_action": actions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _score_single(self, features: Dict[str, Any]) -> float:
        """Compute churn probability for a single customer.

        Uses the loaded model if available; falls back to heuristic.
        """
        if self._model is not None and self._model_loaded:
            return self._score_with_model(features)
        return self._heuristic_score(features)

    def _score_batch(self, data: pd.DataFrame) -> np.ndarray:
        """Compute churn probabilities for a batch."""
        if self._model is not None and self._model_loaded:
            return self._batch_score_with_model(data)
        return np.array([
            self._heuristic_score(row.to_dict())
            for _, row in data.iterrows()
        ])

    def _score_with_model(self, features: Dict[str, Any]) -> float:
        """Score using the loaded ML/DL model."""
        feature_cols = [f for f in NUMERIC_FEATURES if f in features]
        vals = []
        for col in feature_cols:
            v = features.get(col, 0.0)
            vals.append(float(v) if v is not None else 0.0)

        df = pd.DataFrame([vals], columns=feature_cols)
        prob = self._model.predict_proba(df)
        if isinstance(prob, np.ndarray):
            return float(prob[0]) if prob.ndim == 1 else float(prob[0, -1])
        return float(prob)

    def _batch_score_with_model(self, data: pd.DataFrame) -> np.ndarray:
        """Batch score using loaded model."""
        feature_cols = [c for c in NUMERIC_FEATURES if c in data.columns]
        X = data[feature_cols].fillna(0.0).copy()
        probs = self._model.predict_proba(X)
        if probs.ndim > 1:
            probs = probs[:, -1]
        return probs

    def _heuristic_score(self, features: Dict[str, Any]) -> float:
        """Simple rule-based churn probability estimate.

        Uses days_since_last_purchase and days_since_last_login as primary
        churn signals, combined with engagement metrics.
        """
        churn_cfg = self.config.get("churn_definition", {})
        purchase_thresh = churn_cfg.get("no_purchase_days", 30)
        login_thresh = churn_cfg.get("no_login_days", 60)

        days_purchase = float(features.get("days_since_last_purchase", 0) or 0)
        days_login = float(features.get("days_since_last_login", 0) or 0)
        frequency = float(features.get("frequency", 5) or 5)
        session_count = float(features.get("session_count_30d", 10) or 10)
        cart_abandon = float(features.get("cart_abandonment_rate", 0.2) or 0.2)

        # Recency signals
        purchase_risk = min(days_purchase / purchase_thresh, 2.0) * 0.3
        login_risk = min(days_login / login_thresh, 2.0) * 0.2

        # Engagement signals
        freq_risk = max(0, 1.0 - frequency / 10.0) * 0.2
        session_risk = max(0, 1.0 - session_count / 20.0) * 0.15
        cart_risk = min(cart_abandon, 1.0) * 0.15

        raw = purchase_risk + login_risk + freq_risk + session_risk + cart_risk
        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def _classify_risk(prob: float) -> str:
        """Map churn probability to risk level category."""
        if prob >= RISK_THRESHOLDS["critical"]:
            return "critical"
        if prob >= RISK_THRESHOLDS["high"]:
            return "high"
        if prob >= RISK_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache_prediction(
        self, customer_id: str, prediction: Dict[str, Any]
    ) -> None:
        """Cache a prediction result.

        Args:
            customer_id: Customer identifier.
            prediction: Prediction result dict.
        """
        self._cache[customer_id] = {
            "prediction": prediction,
            "cached_at": time.time(),
        }

    def get_cached_prediction(
        self, customer_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached prediction if it exists and is fresh.

        Args:
            customer_id: Customer identifier.

        Returns:
            Cached prediction dict or None if not found/expired.
        """
        entry = self._cache.get(customer_id)
        if entry is None:
            return None

        age = time.time() - entry["cached_at"]
        if age > self._cache_ttl:
            del self._cache[customer_id]
            return None

        return entry["prediction"]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the scoring API.

        Returns:
            Dict with status, model_loaded, version, cache_size.
        """
        status = "healthy" if True else "unhealthy"
        return {
            "status": status,
            "model_loaded": self._model_loaded,
            "version": self.VERSION,
            "model_type": self._model_type,
            "cache_size": len(self._cache),
        }
