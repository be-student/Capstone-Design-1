"""
TDD Tests for Redis Streams Consumer Module.

Tests cover:
- RedisStreamConsumer instantiation from config
- Reading scoring requests from a Redis stream
- Processing messages with ScoringAPI integration
- Writing prediction results to the response stream
- Message acknowledgement
- Consumer group management
- Connection error handling
- Health check
- Graceful shutdown
- End-to-end: produce -> consume -> response
"""

import json
import sys
import time

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

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
def sample_message_data():
    """Create a sample Redis stream message data dict."""
    features = {
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
    return {
        "customer_id": "C00001",
        "features": json.dumps(features),
        "timestamp": "2024-06-15T10:30:00+00:00",
        "request_type": "score",
    }


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for consumer tests."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.xgroup_create.return_value = True
    mock.xreadgroup.return_value = None  # no messages by default
    mock.xadd.return_value = b"9999999999-0"
    mock.xack.return_value = 1
    mock.xlen.return_value = 0
    return mock


@pytest.fixture
def consumer(config, mock_redis):
    """Create a RedisStreamConsumer with mocked Redis."""
    from src.streaming.redis_consumer import RedisStreamConsumer
    cons = RedisStreamConsumer(config)
    cons._redis = mock_redis
    cons._connected = True
    return cons


@pytest.fixture
def consumer_with_api(consumer, config):
    """Create a consumer with a ScoringAPI attached."""
    from src.models.scoring_api import ScoringAPI
    api = ScoringAPI(config)
    consumer.set_scoring_api(api)
    return consumer


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------

class TestRedisStreamConsumerInterface:
    """Test consumer instantiation and interface."""

    def test_instantiation_from_config(self, config):
        """Consumer must be instantiable from config dict."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        consumer = RedisStreamConsumer(config)
        assert consumer is not None

    def test_has_process_message_method(self, consumer):
        """Must implement a process_message method."""
        assert hasattr(consumer, "process_message")
        assert callable(consumer.process_message)

    def test_has_process_one_method(self, consumer):
        """Must implement a process_one method for single message."""
        assert hasattr(consumer, "process_one")
        assert callable(consumer.process_one)

    def test_has_start_method(self, consumer):
        """Must implement a start method for event loop."""
        assert hasattr(consumer, "start")
        assert callable(consumer.start)

    def test_has_stop_method(self, consumer):
        """Must implement a stop method."""
        assert hasattr(consumer, "stop")
        assert callable(consumer.stop)

    def test_has_health_check_method(self, consumer):
        """Must implement a health check."""
        assert hasattr(consumer, "health_check")
        assert callable(consumer.health_check)

    def test_has_close_method(self, consumer):
        """Must implement a close/cleanup method."""
        assert hasattr(consumer, "close")
        assert callable(consumer.close)

    def test_default_stream_names(self, consumer):
        """Default stream names from config."""
        assert consumer.request_stream == "scoring_requests"
        assert consumer.response_stream == "scoring_responses"

    def test_consumer_group_name(self, consumer):
        """Consumer group should have a default name."""
        assert consumer.consumer_group == "scoring_consumers"

    def test_consumer_name(self, consumer):
        """Consumer should have a unique name."""
        assert consumer.consumer_name == "consumer-1"


# ---------------------------------------------------------------------------
# Message processing tests
# ---------------------------------------------------------------------------

class TestMessageProcessing:
    """Test processing individual messages."""

    def test_process_message_returns_prediction(
        self, consumer_with_api, sample_message_data
    ):
        """Processing a message must return a prediction dict."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        assert isinstance(result, dict)
        assert "churn_probability" in result
        assert "risk_level" in result
        assert "customer_id" in result

    def test_process_message_valid_probability(
        self, consumer_with_api, sample_message_data
    ):
        """Churn probability must be in [0, 1]."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        prob = result["churn_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range"

    def test_process_message_valid_risk_level(
        self, consumer_with_api, sample_message_data
    ):
        """Risk level must be one of the valid categories."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        assert result["risk_level"] in {"low", "medium", "high", "critical"}

    def test_process_message_includes_customer_id(
        self, consumer_with_api, sample_message_data
    ):
        """Result must include the original customer_id."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        assert result["customer_id"] == "C00001"

    def test_process_message_includes_metadata(
        self, consumer_with_api, sample_message_data
    ):
        """Result must include request metadata."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        assert "request_message_id" in result
        assert result["request_message_id"] == "1234567890-0"
        assert "scored_at" in result

    def test_process_message_handles_invalid_json(
        self, consumer_with_api
    ):
        """Must handle invalid JSON in features gracefully."""
        bad_data = {
            "customer_id": "C00002",
            "features": "not-valid-json{{{",
            "timestamp": "2024-06-15T10:30:00+00:00",
            "request_type": "score",
        }
        result = consumer_with_api.process_message("1234567890-1", bad_data)
        assert isinstance(result, dict)
        assert "churn_probability" in result

    def test_process_message_handles_missing_features(
        self, consumer_with_api
    ):
        """Must handle missing features field."""
        minimal_data = {
            "customer_id": "C00003",
            "request_type": "score",
        }
        result = consumer_with_api.process_message("1234567890-2", minimal_data)
        assert isinstance(result, dict)
        assert "churn_probability" in result


# ---------------------------------------------------------------------------
# Response writing tests
# ---------------------------------------------------------------------------

class TestResponseWriting:
    """Test writing prediction results to the response stream."""

    def test_write_response_calls_xadd(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """Response must be written to the response stream via XADD."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        consumer_with_api._write_response(result)
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == consumer_with_api.response_stream

    def test_response_includes_probability(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """Written response must include churn_probability."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        consumer_with_api._write_response(result)
        call_args = mock_redis.xadd.call_args
        msg_data = call_args[0][1]
        assert "churn_probability" in msg_data

    def test_response_includes_customer_id(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """Written response must include customer_id."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        consumer_with_api._write_response(result)
        call_args = mock_redis.xadd.call_args
        msg_data = call_args[0][1]
        assert msg_data["customer_id"] == "C00001"

    def test_response_includes_risk_level(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """Written response must include risk_level."""
        result = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        consumer_with_api._write_response(result)
        call_args = mock_redis.xadd.call_args
        msg_data = call_args[0][1]
        assert "risk_level" in msg_data


# ---------------------------------------------------------------------------
# Message acknowledgement tests
# ---------------------------------------------------------------------------

class TestMessageAcknowledgement:
    """Test message ACK behaviour."""

    def test_acknowledge_calls_xack(self, consumer, mock_redis):
        """ACK must call XACK on the request stream."""
        consumer._acknowledge("1234567890-0")
        mock_redis.xack.assert_called_once_with(
            consumer.request_stream,
            consumer.consumer_group,
            "1234567890-0",
        )


# ---------------------------------------------------------------------------
# process_one integration tests
# ---------------------------------------------------------------------------

class TestProcessOne:
    """Test single-message processing via process_one."""

    def test_process_one_returns_result(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """process_one should return a prediction result dict."""
        mock_redis.xreadgroup.return_value = [
            (
                consumer_with_api.request_stream,
                [("1234567890-0", sample_message_data)],
            )
        ]
        result = consumer_with_api.process_one()
        assert result is not None
        assert "churn_probability" in result

    def test_process_one_returns_none_when_empty(
        self, consumer_with_api, mock_redis
    ):
        """process_one should return None if no messages."""
        mock_redis.xreadgroup.return_value = None
        result = consumer_with_api.process_one()
        assert result is None

    def test_process_one_acks_message(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """process_one must ACK the processed message."""
        mock_redis.xreadgroup.return_value = [
            (
                consumer_with_api.request_stream,
                [("1234567890-0", sample_message_data)],
            )
        ]
        consumer_with_api.process_one()
        mock_redis.xack.assert_called_once()

    def test_process_one_writes_response(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """process_one must write a response to the response stream."""
        mock_redis.xreadgroup.return_value = [
            (
                consumer_with_api.request_stream,
                [("1234567890-0", sample_message_data)],
            )
        ]
        consumer_with_api.process_one()
        # xadd should have been called for the response
        mock_redis.xadd.assert_called_once()

    def test_process_one_increments_counter(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """process_one must increment processed_count."""
        mock_redis.xreadgroup.return_value = [
            (
                consumer_with_api.request_stream,
                [("1234567890-0", sample_message_data)],
            )
        ]
        initial = consumer_with_api.processed_count
        consumer_with_api.process_one()
        assert consumer_with_api.processed_count == initial + 1

    def test_process_one_raises_when_disconnected(self, config):
        """process_one must raise when not connected."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        cons = RedisStreamConsumer(config)
        cons._connected = False
        with pytest.raises(ConnectionError):
            cons.process_one()


# ---------------------------------------------------------------------------
# Event loop tests
# ---------------------------------------------------------------------------

class TestEventLoop:
    """Test the consumer event loop (start/stop)."""

    def test_start_with_max_iterations(
        self, consumer_with_api, mock_redis, sample_message_data
    ):
        """start(max_iterations=N) should stop after N iterations."""
        mock_redis.xreadgroup.return_value = [
            (
                consumer_with_api.request_stream,
                [("1234567890-0", sample_message_data)],
            )
        ]
        consumer_with_api.start(max_iterations=3)
        assert mock_redis.xreadgroup.call_count == 3

    def test_start_with_no_messages(
        self, consumer_with_api, mock_redis
    ):
        """start should handle empty reads gracefully."""
        mock_redis.xreadgroup.return_value = None
        consumer_with_api.start(max_iterations=2)
        assert mock_redis.xreadgroup.call_count == 2

    def test_stop_sets_running_false(self, consumer_with_api):
        """stop() should set _running to False."""
        consumer_with_api._running = True
        consumer_with_api.stop()
        assert consumer_with_api._running is False

    def test_start_raises_when_disconnected(self, config):
        """start must raise when not connected."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        cons = RedisStreamConsumer(config)
        cons._connected = False
        with pytest.raises(ConnectionError):
            cons.start()


# ---------------------------------------------------------------------------
# Connection handling tests
# ---------------------------------------------------------------------------

class TestConsumerConnection:
    """Test Redis connection and error handling."""

    def test_health_check_connected(self, consumer, mock_redis):
        """Health check should report healthy when connected."""
        health = consumer.health_check()
        assert health["status"] == "healthy"
        assert health["connected"] is True

    def test_health_check_disconnected(self, config):
        """Health check should report unhealthy when disconnected."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        cons = RedisStreamConsumer(config)
        cons._connected = False
        cons._redis = None
        health = cons.health_check()
        assert health["status"] == "unhealthy"
        assert health["connected"] is False

    def test_close_cleans_up(self, consumer, mock_redis):
        """Close must clean up the Redis connection."""
        consumer.close()
        mock_redis.close.assert_called_once()
        assert consumer._connected is False

    def test_health_check_includes_processed_count(self, consumer):
        """Health check should report processed count."""
        health = consumer.health_check()
        assert "processed_count" in health
        assert health["processed_count"] == 0


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestConsumerConfiguration:
    """Test consumer configuration from YAML."""

    def test_redis_host_default(self, config):
        """Default Redis host should be localhost."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        cons = RedisStreamConsumer(config)
        assert cons.host == "localhost"

    def test_redis_port_default(self, config):
        """Default Redis port should be 6379."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        cons = RedisStreamConsumer(config)
        assert cons.port == 6379

    def test_configurable_consumer_group(self, config):
        """Consumer group should be configurable."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        config.setdefault("redis", {})["consumer_group"] = "my_group"
        cons = RedisStreamConsumer(config)
        assert cons.consumer_group == "my_group"

    def test_configurable_batch_size(self, config):
        """Batch size should be configurable."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        config.setdefault("redis", {})["consumer_batch_size"] = 50
        cons = RedisStreamConsumer(config)
        assert cons.batch_size == 50

    def test_env_var_overrides_host(self, config, monkeypatch):
        """REDIS_HOST env var should override config host."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        monkeypatch.setenv("REDIS_HOST", "redis-docker")
        cons = RedisStreamConsumer(config)
        assert cons.host == "redis-docker"

    def test_env_var_overrides_port(self, config, monkeypatch):
        """REDIS_PORT env var should override config port."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        monkeypatch.setenv("REDIS_PORT", "6380")
        cons = RedisStreamConsumer(config)
        assert cons.port == 6380

    def test_env_var_overrides_db(self, config, monkeypatch):
        """REDIS_DB env var should override config db."""
        from src.streaming.redis_consumer import RedisStreamConsumer
        monkeypatch.setenv("REDIS_DB", "2")
        cons = RedisStreamConsumer(config)
        assert cons.db == 2


# ---------------------------------------------------------------------------
# ScoringAPI integration tests
# ---------------------------------------------------------------------------

class TestScoringAPIIntegration:
    """Test consumer integration with ScoringAPI."""

    def test_set_scoring_api(self, consumer, config):
        """Must be able to attach a ScoringAPI."""
        from src.models.scoring_api import ScoringAPI
        api = ScoringAPI(config)
        consumer.set_scoring_api(api)
        assert consumer._scoring_api is api

    def test_lazy_creates_scoring_api(self, consumer):
        """_get_scoring_api creates API lazily if not set."""
        api = consumer._get_scoring_api()
        assert api is not None

    def test_reproducible_predictions(
        self, consumer_with_api, sample_message_data
    ):
        """Same message should produce identical predictions."""
        r1 = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        r2 = consumer_with_api.process_message(
            "1234567890-0", sample_message_data
        )
        assert abs(
            r1["churn_probability"] - r2["churn_probability"]
        ) < 1e-10
