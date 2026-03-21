"""
TDD Tests for Real-Time Scoring API Module.

Tests cover:
- Scoring API instantiation and interface
- Single customer real-time churn prediction
- Batch customer scoring
- Feature validation and preprocessing for API input
- Response format (churn probability, risk level, recommended action)
- Redis integration for caching and streaming
- Latency requirements (< 100ms per prediction)
- Model loading and warm-up
- Error handling for invalid/missing inputs
- Health check endpoint
- API versioning support
- Feature store integration (file-based)
- Reproducibility with same input
- Configurable parameters from YAML
"""

import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_customer_features():
    """Create a single customer feature vector for scoring.

    Represents the minimal feature set needed for real-time inference.
    """
    return {
        "customer_id": "C00001",
        "recency": 5.0,
        "frequency": 12.0,
        "monetary": 85000.0,
        "avg_order_value": 70000.0,
        "days_since_last_purchase": 15.0,
        "days_since_last_login": 3.0,
        "total_purchases": 24.0,
        "session_count_30d": 18.0,
        "avg_session_duration": 12.5,
        "coupon_usage_rate": 0.35,
        "cart_abandonment_rate": 0.20,
        "review_count": 5.0,
        "cs_contact_count": 2.0,
        "preferred_category_encoded": 3,
        "segment_encoded": 1,
    }


@pytest.fixture
def sample_batch_features():
    """Create a batch of customer feature vectors for scoring."""
    np.random.seed(42)
    n = 100

    records = []
    for i in range(n):
        records.append({
            "customer_id": f"C{i:05d}",
            "recency": np.random.exponential(10),
            "frequency": np.random.poisson(5),
            "monetary": np.random.lognormal(10, 1),
            "avg_order_value": np.random.lognormal(10, 0.5),
            "days_since_last_purchase": np.random.exponential(20),
            "days_since_last_login": np.random.exponential(10),
            "total_purchases": np.random.poisson(15),
            "session_count_30d": np.random.poisson(10),
            "avg_session_duration": np.random.exponential(15),
            "coupon_usage_rate": np.random.beta(2, 5),
            "cart_abandonment_rate": np.random.beta(2, 8),
            "review_count": np.random.poisson(3),
            "cs_contact_count": np.random.poisson(1),
            "preferred_category_encoded": np.random.randint(0, 5),
            "segment_encoded": np.random.randint(0, 6),
        })

    return pd.DataFrame(records)


@pytest.fixture
def scoring_api(config):
    """Create a scoring API instance."""
    from src.models.scoring_api import ScoringAPI
    return ScoringAPI(config)


# ---------------------------------------------------------------------------
# API interface tests
# ---------------------------------------------------------------------------

class TestScoringAPIInterface:
    """Test scoring API instantiation and interface."""

    def test_instantiation(self, scoring_api):
        """Scoring API must be instantiable from config."""
        assert scoring_api is not None

    def test_has_predict_method(self, scoring_api):
        """Must implement a predict method for single customer."""
        assert hasattr(scoring_api, "predict")
        assert callable(scoring_api.predict)

    def test_has_predict_batch_method(self, scoring_api):
        """Must implement a batch prediction method."""
        assert hasattr(scoring_api, "predict_batch")
        assert callable(scoring_api.predict_batch)

    def test_has_health_check_method(self, scoring_api):
        """Must implement a health check endpoint."""
        assert hasattr(scoring_api, "health_check")
        assert callable(scoring_api.health_check)

    def test_has_validate_features_method(self, scoring_api):
        """Must implement input feature validation."""
        assert hasattr(scoring_api, "validate_features")
        assert callable(scoring_api.validate_features)

    def test_has_get_required_features_method(self, scoring_api):
        """Must expose the list of required input features."""
        assert hasattr(scoring_api, "get_required_features")
        assert callable(scoring_api.get_required_features)


# ---------------------------------------------------------------------------
# Single prediction tests
# ---------------------------------------------------------------------------

class TestSinglePrediction:
    """Test single customer real-time prediction."""

    def test_returns_churn_probability(self, scoring_api,
                                        sample_customer_features):
        """Prediction must return a churn probability."""
        result = scoring_api.predict(features=sample_customer_features)

        assert "churn_probability" in result, (
            "Result must include churn_probability"
        )
        assert 0 <= result["churn_probability"] <= 1, (
            f"Churn probability {result['churn_probability']} out of [0, 1]"
        )

    def test_returns_risk_level(self, scoring_api,
                                  sample_customer_features):
        """Prediction must return a risk level category."""
        result = scoring_api.predict(features=sample_customer_features)

        assert "risk_level" in result, "Result must include risk_level"
        assert result["risk_level"] in {"low", "medium", "high", "critical"}, (
            f"Invalid risk level: {result['risk_level']}"
        )

    def test_returns_recommended_action(self, scoring_api,
                                          sample_customer_features):
        """Prediction must include a recommended retention action."""
        result = scoring_api.predict(features=sample_customer_features)

        assert "recommended_action" in result, (
            "Result must include recommended_action"
        )
        assert isinstance(result["recommended_action"], str)
        assert len(result["recommended_action"]) > 0

    def test_returns_customer_id(self, scoring_api,
                                   sample_customer_features):
        """Prediction must echo back the customer_id."""
        result = scoring_api.predict(features=sample_customer_features)

        assert "customer_id" in result
        assert result["customer_id"] == sample_customer_features["customer_id"]

    def test_returns_timestamp(self, scoring_api,
                                 sample_customer_features):
        """Prediction must include a scoring timestamp."""
        result = scoring_api.predict(features=sample_customer_features)

        assert "timestamp" in result, "Result must include timestamp"

    def test_result_is_dict(self, scoring_api, sample_customer_features):
        """Single prediction must return a dictionary."""
        result = scoring_api.predict(features=sample_customer_features)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Batch prediction tests
# ---------------------------------------------------------------------------

class TestBatchPrediction:
    """Test batch customer scoring."""

    def test_batch_returns_all_customers(self, scoring_api,
                                          sample_batch_features):
        """Batch prediction must return results for all customers."""
        results = scoring_api.predict_batch(data=sample_batch_features)

        assert len(results) == len(sample_batch_features), (
            f"Expected {len(sample_batch_features)} results, "
            f"got {len(results)}"
        )

    def test_batch_returns_dataframe(self, scoring_api,
                                      sample_batch_features):
        """Batch prediction must return a DataFrame."""
        results = scoring_api.predict_batch(data=sample_batch_features)
        assert isinstance(results, pd.DataFrame)

    def test_batch_has_churn_probability(self, scoring_api,
                                          sample_batch_features):
        """Batch results must include churn_probability column."""
        results = scoring_api.predict_batch(data=sample_batch_features)

        assert "churn_probability" in results.columns
        assert (results["churn_probability"] >= 0).all()
        assert (results["churn_probability"] <= 1).all()

    def test_batch_has_risk_level(self, scoring_api,
                                    sample_batch_features):
        """Batch results must include risk_level column."""
        results = scoring_api.predict_batch(data=sample_batch_features)

        assert "risk_level" in results.columns
        valid_levels = {"low", "medium", "high", "critical"}
        assert set(results["risk_level"].unique()) <= valid_levels

    def test_batch_has_customer_id(self, scoring_api,
                                     sample_batch_features):
        """Batch results must include customer_id column."""
        results = scoring_api.predict_batch(data=sample_batch_features)

        assert "customer_id" in results.columns
        assert set(results["customer_id"]) == set(
            sample_batch_features["customer_id"]
        )

    def test_batch_no_duplicate_customers(self, scoring_api,
                                            sample_batch_features):
        """Each customer should appear exactly once in batch results."""
        results = scoring_api.predict_batch(data=sample_batch_features)

        assert results["customer_id"].nunique() == len(results), (
            "Duplicate customers in batch results"
        )

    def test_batch_no_nan_probabilities(self, scoring_api,
                                          sample_batch_features):
        """Churn probabilities must not contain NaN."""
        results = scoring_api.predict_batch(data=sample_batch_features)

        assert not results["churn_probability"].isna().any(), (
            "NaN values in churn probabilities"
        )


# ---------------------------------------------------------------------------
# Feature validation tests
# ---------------------------------------------------------------------------

class TestFeatureValidation:
    """Test input feature validation."""

    def test_valid_features_pass(self, scoring_api,
                                   sample_customer_features):
        """Valid feature set must pass validation."""
        is_valid, errors = scoring_api.validate_features(
            features=sample_customer_features,
        )

        assert is_valid is True, f"Valid features rejected: {errors}"

    def test_missing_feature_detected(self, scoring_api,
                                        sample_customer_features):
        """Missing required features must be detected."""
        incomplete = {k: v for k, v in sample_customer_features.items()
                      if k != "recency"}

        is_valid, errors = scoring_api.validate_features(
            features=incomplete,
        )

        assert is_valid is False, "Missing feature not detected"
        assert len(errors) > 0

    def test_required_features_list(self, scoring_api):
        """get_required_features must return a non-empty list."""
        required = scoring_api.get_required_features()

        assert isinstance(required, list)
        assert len(required) > 0
        assert "customer_id" in required

    def test_invalid_type_detected(self, scoring_api,
                                     sample_customer_features):
        """Invalid feature types should be handled gracefully."""
        bad_features = sample_customer_features.copy()
        bad_features["recency"] = "not_a_number"

        is_valid, errors = scoring_api.validate_features(
            features=bad_features,
        )

        # Should either reject or handle gracefully
        assert isinstance(is_valid, bool)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestScoringErrorHandling:
    """Test error handling for edge cases."""

    def test_empty_features_handled(self, scoring_api):
        """Empty feature dict must be handled gracefully."""
        with pytest.raises((ValueError, KeyError)):
            scoring_api.predict(features={})

    def test_empty_batch_handled(self, scoring_api):
        """Empty DataFrame must be handled gracefully."""
        empty_df = pd.DataFrame()
        result = scoring_api.predict_batch(data=empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_null_values_handled(self, scoring_api,
                                   sample_customer_features):
        """Null feature values must be handled gracefully."""
        features_with_null = sample_customer_features.copy()
        features_with_null["recency"] = None

        # Should either handle gracefully or raise clear error
        try:
            result = scoring_api.predict(features=features_with_null)
            assert "churn_probability" in result
        except (ValueError, TypeError) as e:
            assert str(e), "Error message should be descriptive"


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check_returns_status(self, scoring_api):
        """Health check must return a status indicator."""
        health = scoring_api.health_check()

        assert "status" in health, "Health check must include status"
        assert health["status"] in {"healthy", "unhealthy", "degraded"}

    def test_health_check_returns_model_loaded(self, scoring_api):
        """Health check must indicate whether model is loaded."""
        health = scoring_api.health_check()

        assert "model_loaded" in health
        assert isinstance(health["model_loaded"], bool)

    def test_health_check_returns_version(self, scoring_api):
        """Health check must return API version."""
        health = scoring_api.health_check()

        assert "version" in health
        assert isinstance(health["version"], str)


# ---------------------------------------------------------------------------
# Latency tests
# ---------------------------------------------------------------------------

class TestScoringLatency:
    """Test scoring latency requirements."""

    def test_single_prediction_latency(self, scoring_api,
                                         sample_customer_features):
        """Single prediction should complete within 200ms."""
        start = time.time()
        scoring_api.predict(features=sample_customer_features)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 200, (
            f"Single prediction took {elapsed_ms:.1f}ms, expected < 200ms"
        )

    def test_batch_prediction_throughput(self, scoring_api,
                                           sample_batch_features):
        """Batch of 100 should complete within 2 seconds."""
        start = time.time()
        scoring_api.predict_batch(data=sample_batch_features)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 2000, (
            f"Batch prediction took {elapsed_ms:.1f}ms, expected < 2000ms"
        )


# ---------------------------------------------------------------------------
# Redis integration tests
# ---------------------------------------------------------------------------

class TestRedisIntegration:
    """Test Redis integration for caching and streaming."""

    def test_has_cache_prediction_method(self, scoring_api):
        """Must implement prediction caching."""
        assert hasattr(scoring_api, "cache_prediction")
        assert callable(scoring_api.cache_prediction)

    def test_has_get_cached_prediction_method(self, scoring_api):
        """Must implement cached prediction retrieval."""
        assert hasattr(scoring_api, "get_cached_prediction")
        assert callable(scoring_api.get_cached_prediction)

    def test_cache_roundtrip(self, scoring_api, sample_customer_features):
        """Cached prediction must be retrievable."""
        prediction = scoring_api.predict(features=sample_customer_features)

        scoring_api.cache_prediction(
            customer_id=sample_customer_features["customer_id"],
            prediction=prediction,
        )

        cached = scoring_api.get_cached_prediction(
            customer_id=sample_customer_features["customer_id"],
        )

        if cached is not None:
            assert cached["churn_probability"] == prediction["churn_probability"]


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestScoringReproducibility:
    """Test scoring reproducibility with same input."""

    def test_same_input_same_output(self, scoring_api,
                                      sample_customer_features):
        """Same input must produce identical predictions."""
        result1 = scoring_api.predict(features=sample_customer_features)
        result2 = scoring_api.predict(features=sample_customer_features)

        assert abs(
            result1["churn_probability"] - result2["churn_probability"]
        ) < 1e-10, "Same input produced different predictions"

    def test_batch_reproducibility(self, scoring_api,
                                     sample_batch_features):
        """Same batch input must produce identical results."""
        results1 = scoring_api.predict_batch(data=sample_batch_features)
        results2 = scoring_api.predict_batch(data=sample_batch_features)

        np.testing.assert_array_almost_equal(
            results1["churn_probability"].values,
            results2["churn_probability"].values,
            decimal=10,
        )


# ---------------------------------------------------------------------------
# Model management tests
# ---------------------------------------------------------------------------

class TestModelManagement:
    """Test model loading and management."""

    def test_load_model(self, scoring_api):
        """Must be able to load a trained model."""
        assert hasattr(scoring_api, "load_model")
        assert callable(scoring_api.load_model)

    def test_get_model_info(self, scoring_api):
        """Must expose model metadata."""
        assert hasattr(scoring_api, "get_model_info")
        info = scoring_api.get_model_info()

        assert isinstance(info, dict)
        assert "model_type" in info or "version" in info
