"""
Redis Streams Producer for Customer Scoring Requests.

Publishes customer feature data to a Redis Stream for real-time
churn scoring. Messages include customer_id, serialized features,
timestamp, and request type.

Usage:
    from src.streaming.redis_producer import RedisStreamProducer

    producer = RedisStreamProducer(config)
    producer.connect()
    msg_id = producer.publish(customer_features)
    producer.close()
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RedisStreamProducer:
    """Publishes customer scoring requests to a Redis Stream.

    Attributes:
        host: Redis server hostname.
        port: Redis server port.
        stream_name: Name of the Redis stream to publish to.
        maxlen: Maximum stream length (approximate trimming).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Redis Stream producer.

        Args:
            config: Application configuration dictionary. Redis settings
                are read from config["redis"] if present. Environment
                variables REDIS_HOST and REDIS_PORT override config values
                (for Docker deployment).
        """
        redis_config = config.get("redis", {})
        self.host: str = os.environ.get(
            "REDIS_HOST", redis_config.get("host", "localhost")
        )
        self.port: int = int(os.environ.get(
            "REDIS_PORT", redis_config.get("port", 6379)
        ))
        self.db: int = int(os.environ.get(
            "REDIS_DB", redis_config.get("db", 0)
        ))
        self.stream_name: str = redis_config.get(
            "stream_name", "scoring_requests"
        )
        self.maxlen: int = redis_config.get("stream_maxlen", 10000)
        self._redis: Optional[Any] = None
        self._connected: bool = False
        self._config = config

    def connect(self) -> bool:
        """Establish connection to Redis server.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            import redis
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            self._redis.ping()
            self._connected = True
            logger.info(
                "Connected to Redis at %s:%d", self.host, self.port
            )
            return True
        except Exception as exc:
            logger.warning("Failed to connect to Redis: %s", exc)
            self._connected = False
            return False

    def publish(self, features: Dict[str, Any]) -> str:
        """Publish a single customer scoring request to the stream.

        Args:
            features: Dictionary containing customer_id and feature
                values for scoring.

        Returns:
            Message ID assigned by Redis.

        Raises:
            ConnectionError: If not connected to Redis.
            ValueError: If customer_id is missing from features.
        """
        if not self._connected or self._redis is None:
            raise ConnectionError(
                "Not connected to Redis. Call connect() first."
            )

        customer_id = features.get("customer_id")
        if not customer_id:
            raise ValueError("features must include 'customer_id'")

        # Separate customer_id from numeric features for serialization
        feature_data = {
            k: v for k, v in features.items() if k != "customer_id"
        }

        message = {
            "customer_id": str(customer_id),
            "features": json.dumps(feature_data, default=_json_serializer),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_type": "score",
        }

        msg_id = self._redis.xadd(
            self.stream_name, message, maxlen=self.maxlen, approximate=True
        )

        # Redis may return bytes or str depending on decode_responses
        if isinstance(msg_id, bytes):
            msg_id = msg_id.decode("utf-8")

        logger.debug(
            "Published scoring request for %s, msg_id=%s",
            customer_id, msg_id,
        )
        return str(msg_id)

    def publish_batch(
        self, features_list: List[Dict[str, Any]]
    ) -> List[str]:
        """Publish a batch of customer scoring requests.

        Args:
            features_list: List of feature dictionaries, each containing
                customer_id and feature values.

        Returns:
            List of message IDs, one per published request.
        """
        if not features_list:
            return []

        msg_ids: List[str] = []
        for features in features_list:
            msg_id = self.publish(features)
            msg_ids.append(msg_id)

        logger.info(
            "Published batch of %d scoring requests", len(msg_ids)
        )
        return msg_ids

    def get_stream_length(self) -> int:
        """Get the current length of the scoring stream.

        Returns:
            Number of messages in the stream.

        Raises:
            ConnectionError: If not connected to Redis.
        """
        if not self._connected or self._redis is None:
            raise ConnectionError("Not connected to Redis.")
        return self._redis.xlen(self.stream_name)

    def health_check(self) -> Dict[str, Any]:
        """Check producer health and Redis connectivity.

        Returns:
            Dictionary with status, connected flag, and stream info.
        """
        health: Dict[str, Any] = {
            "connected": self._connected,
            "host": self.host,
            "port": self.port,
            "stream_name": self.stream_name,
        }

        if self._connected and self._redis is not None:
            try:
                self._redis.ping()
                health["status"] = "healthy"
                health["stream_length"] = self._redis.xlen(
                    self.stream_name
                )
            except Exception:
                health["status"] = "unhealthy"
                self._connected = False
        else:
            health["status"] = "unhealthy"

        return health

    def close(self) -> None:
        """Close the Redis connection and clean up resources."""
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
        self._redis = None
        self._connected = False
        logger.info("Redis producer connection closed.")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy/pandas types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.
    """
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
