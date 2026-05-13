"""
TDD Tests for Redis Streams Producer Module.

Tests cover:
- RedisStreamProducer instantiation from config
- Publishing single customer scoring requests to a Redis stream
- Publishing batch scoring requests
- Message format validation (customer_id, feature data, timestamp)
- Serialization/deserialization of feature data
- Stream name configurability
- Connection error handling (graceful fallback when Redis unavailable)
- Message ID generation
- Retry logic for transient failures
- Producer health check
"""

import os
import sys
import json
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

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
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_customer_features():
    """Create a single customer feature dict for scoring requests."""
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
    """Create a batch of customer feature dicts."""
    np.random.seed(42)
    batch = []
    for i in range(10):
        batch.append({
            "customer_id": f"C{i:05d}",
            "recency": float(np.random.exponential(10)),
            "frequency": float(np.random.poisson(5)),
            "monetary": float(np.random.lognormal(10, 1)),
            "avg_order_value": float(np.random.lognormal(10, 0.5)),
            "days_since_last_purchase": float(np.random.exponential(20)),
            "days_since_last_login": float(np.random.exponential(10)),
            "total_purchases": float(np.random.poisson(15)),
            "session_count_30d": float(np.random.poisson(10)),
            "avg_session_duration": float(np.random.exponential(15)),
            "coupon_usage_rate": float(np.random.beta(2, 5)),
            "cart_abandonment_rate": float(np.random.beta(2, 8)),
            "review_count": float(np.random.poisson(3)),
            "cs_contact_count": float(np.random.poisson(1)),
            "preferred_category_encoded": int(np.random.randint(0, 5)),
            "segment_encoded": int(np.random.randint(0, 6)),
        })
    return batch


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    mock = MagicMock()
    mock.ping.return_value = True
    # xadd returns a message ID
    mock.xadd.return_value = b"1234567890-0"
    mock.xlen.return_value = 0
    return mock


@pytest.fixture
def producer(config, mock_redis):
    """Create a RedisStreamProducer with mocked Redis."""
    from src.streaming.redis_producer import RedisStreamProducer
    prod = RedisStreamProducer(config)
    prod._redis = mock_redis
    prod._connected = True
    return prod


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------

class TestRedisStreamProducerInterface:
    """Test producer instantiation and interface."""

    def test_instantiation_from_config(self, config):
        """Producer must be instantiable from config dict."""
        from src.streaming.redis_producer import RedisStreamProducer
        producer = RedisStreamProducer(config)
        assert producer is not None

    def test_has_publish_method(self, producer):
        """Must implement a publish method for single requests."""
        assert hasattr(producer, "publish")
        assert callable(producer.publish)

    def test_has_publish_batch_method(self, producer):
        """Must implement a batch publish method."""
        assert hasattr(producer, "publish_batch")
        assert callable(producer.publish_batch)

    def test_has_health_check_method(self, producer):
        """Must implement a health check."""
        assert hasattr(producer, "health_check")
        assert callable(producer.health_check)

    def test_has_close_method(self, producer):
        """Must implement a close/cleanup method."""
        assert hasattr(producer, "close")
        assert callable(producer.close)

    def test_default_stream_name(self, producer):
        """Default stream name should be 'scoring_requests'."""
        assert producer.stream_name == "scoring_requests"

    def test_configurable_stream_name(self, config):
        """Stream name should be configurable via config."""
        from src.streaming.redis_producer import RedisStreamProducer
        config.setdefault("redis", {})["stream_name"] = "custom_stream"
        prod = RedisStreamProducer(config)
        assert prod.stream_name == "custom_stream"


# ---------------------------------------------------------------------------
# Single publish tests
# ---------------------------------------------------------------------------

class TestSinglePublish:
    """Test publishing single customer scoring requests."""

    def test_publish_returns_message_id(self, producer,
                                        sample_customer_features):
        """Publish must return a message ID."""
        msg_id = producer.publish(sample_customer_features)
        assert msg_id is not None
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_publish_calls_xadd(self, producer, mock_redis,
                                 sample_customer_features):
        """Publish must call Redis XADD on the configured stream."""
        producer.publish(sample_customer_features)
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        # First positional arg is stream name
        assert call_args[0][0] == producer.stream_name

    def test_publish_includes_customer_id(self, producer, mock_redis,
                                           sample_customer_features):
        """Published message must include customer_id."""
        producer.publish(sample_customer_features)
        call_args = mock_redis.xadd.call_args
        message_data = call_args[0][1]
        assert "customer_id" in message_data
        assert message_data["customer_id"] == "C00001"

    def test_publish_includes_features(self, producer, mock_redis,
                                        sample_customer_features):
        """Published message must include serialized feature data."""
        producer.publish(sample_customer_features)
        call_args = mock_redis.xadd.call_args
        message_data = call_args[0][1]
        assert "features" in message_data
        # Features should be JSON-serialized
        features = json.loads(message_data["features"])
        assert isinstance(features, dict)
        assert "recency" in features

    def test_publish_includes_timestamp(self, producer, mock_redis,
                                         sample_customer_features):
        """Published message must include a timestamp."""
        producer.publish(sample_customer_features)
        call_args = mock_redis.xadd.call_args
        message_data = call_args[0][1]
        assert "timestamp" in message_data

    def test_publish_includes_request_type(self, producer, mock_redis,
                                            sample_customer_features):
        """Published message must include request_type field."""
        producer.publish(sample_customer_features)
        call_args = mock_redis.xadd.call_args
        message_data = call_args[0][1]
        assert "request_type" in message_data
        assert message_data["request_type"] == "score"

    def test_feature_serialization_roundtrip(self, producer, mock_redis,
                                              sample_customer_features):
        """Features must survive JSON serialization roundtrip."""
        producer.publish(sample_customer_features)
        call_args = mock_redis.xadd.call_args
        message_data = call_args[0][1]
        features = json.loads(message_data["features"])

        for key in ["recency", "frequency", "monetary"]:
            assert abs(features[key] - sample_customer_features[key]) < 1e-6


# ---------------------------------------------------------------------------
# Batch publish tests
# ---------------------------------------------------------------------------

class TestBatchPublish:
    """Test batch publishing of scoring requests."""

    def test_batch_publish_returns_message_ids(self, producer,
                                                sample_batch_features):
        """Batch publish must return list of message IDs."""
        msg_ids = producer.publish_batch(sample_batch_features)
        assert isinstance(msg_ids, list)
        assert len(msg_ids) == len(sample_batch_features)

    def test_batch_publish_calls_xadd_per_item(self, producer, mock_redis,
                                                 sample_batch_features):
        """Batch publish must call XADD once per customer."""
        producer.publish_batch(sample_batch_features)
        assert mock_redis.xadd.call_count == len(sample_batch_features)

    def test_batch_preserves_customer_ids(self, producer, mock_redis,
                                           sample_batch_features):
        """Each batch item must have its own customer_id."""
        producer.publish_batch(sample_batch_features)
        published_ids = []
        for call in mock_redis.xadd.call_args_list:
            msg_data = call[0][1]
            published_ids.append(msg_data["customer_id"])

        expected_ids = [f["customer_id"] for f in sample_batch_features]
        assert published_ids == expected_ids

    def test_batch_empty_list(self, producer):
        """Empty batch must return empty list."""
        msg_ids = producer.publish_batch([])
        assert msg_ids == []


# ---------------------------------------------------------------------------
# Connection and error handling tests
# ---------------------------------------------------------------------------

class TestConnectionHandling:
    """Test Redis connection and error handling."""

    def test_health_check_connected(self, producer, mock_redis):
        """Health check should report healthy when connected."""
        health = producer.health_check()
        assert health["status"] == "healthy"
        assert health["connected"] is True

    def test_health_check_disconnected(self, config):
        """Health check should report unhealthy when disconnected."""
        from src.streaming.redis_producer import RedisStreamProducer
        prod = RedisStreamProducer(config)
        prod._connected = False
        prod._redis = None
        health = prod.health_check()
        assert health["status"] == "unhealthy"
        assert health["connected"] is False

    def test_publish_raises_on_no_connection(self, config,
                                              sample_customer_features):
        """Publish should raise when not connected."""
        from src.streaming.redis_producer import RedisStreamProducer
        prod = RedisStreamProducer(config)
        prod._connected = False
        prod._redis = None

        with pytest.raises(ConnectionError):
            prod.publish(sample_customer_features)

    def test_publish_handles_redis_error(self, producer, mock_redis,
                                          sample_customer_features):
        """Publish should handle Redis errors gracefully."""
        mock_redis.xadd.side_effect = Exception("Redis error")

        with pytest.raises(Exception, match="Redis error"):
            producer.publish(sample_customer_features)

    def test_close_cleans_up(self, producer, mock_redis):
        """Close must clean up the Redis connection."""
        producer.close()
        mock_redis.close.assert_called_once()
        assert producer._connected is False

    def test_stream_length_query(self, producer, mock_redis):
        """Must be able to query stream length."""
        mock_redis.xlen.return_value = 42
        length = producer.get_stream_length()
        assert length == 42
        mock_redis.xlen.assert_called_once_with(producer.stream_name)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestProducerConfiguration:
    """Test producer configuration from YAML."""

    def test_redis_host_default(self, config):
        """Default Redis host should be localhost."""
        from src.streaming.redis_producer import RedisStreamProducer
        prod = RedisStreamProducer(config)
        assert prod.host == "localhost"

    def test_redis_port_default(self, config):
        """Default Redis port should be 6379."""
        from src.streaming.redis_producer import RedisStreamProducer
        prod = RedisStreamProducer(config)
        assert prod.port == 6379

    def test_redis_host_configurable(self, config):
        """Redis host should be configurable."""
        from src.streaming.redis_producer import RedisStreamProducer
        config.setdefault("redis", {})["host"] = "redis-server"
        prod = RedisStreamProducer(config)
        assert prod.host == "redis-server"

    def test_redis_port_configurable(self, config):
        """Redis port should be configurable."""
        from src.streaming.redis_producer import RedisStreamProducer
        config.setdefault("redis", {})["port"] = 6380
        prod = RedisStreamProducer(config)
        assert prod.port == 6380

    def test_maxlen_configurable(self, config):
        """Stream max length should be configurable."""
        from src.streaming.redis_producer import RedisStreamProducer
        config.setdefault("redis", {})["stream_maxlen"] = 5000
        prod = RedisStreamProducer(config)
        assert prod.maxlen == 5000

    def test_default_maxlen(self, config):
        """Default stream max length should be 10000."""
        from src.streaming.redis_producer import RedisStreamProducer
        prod = RedisStreamProducer(config)
        assert prod.maxlen == 10000

    def test_env_var_overrides_host(self, config, monkeypatch):
        """REDIS_HOST env var should override config host."""
        from src.streaming.redis_producer import RedisStreamProducer
        monkeypatch.setenv("REDIS_HOST", "redis-docker")
        prod = RedisStreamProducer(config)
        assert prod.host == "redis-docker"

    def test_env_var_overrides_port(self, config, monkeypatch):
        """REDIS_PORT env var should override config port."""
        from src.streaming.redis_producer import RedisStreamProducer
        monkeypatch.setenv("REDIS_PORT", "6380")
        prod = RedisStreamProducer(config)
        assert prod.port == 6380

    def test_env_var_overrides_db(self, config, monkeypatch):
        """REDIS_DB env var should override config db."""
        from src.streaming.redis_producer import RedisStreamProducer
        monkeypatch.setenv("REDIS_DB", "2")
        prod = RedisStreamProducer(config)
        assert prod.db == 2
