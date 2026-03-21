"""
Redis Streams Consumer for Real-Time Churn Scoring.

Reads scoring requests from a Redis Stream, performs churn prediction
using a loaded model (or heuristic fallback), and writes results back
to a response stream in Redis.

Usage:
    from src.streaming.redis_consumer import RedisStreamConsumer

    consumer = RedisStreamConsumer(config)
    consumer.connect()
    consumer.start()  # blocking event loop
    # or
    consumer.process_one()  # process a single message

Architecture:
    scoring_requests (stream) -> Consumer -> scoring_responses (stream)
                                    |
                              ScoringAPI (model)
"""

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RedisStreamConsumer:
    """Consumes scoring requests from Redis Streams and writes predictions.

    Reads messages from the ``scoring_requests`` stream, invokes the
    ScoringAPI for churn prediction, and publishes results to the
    ``scoring_responses`` stream.

    Attributes:
        host: Redis server hostname.
        port: Redis server port.
        request_stream: Name of the input stream.
        response_stream: Name of the output stream.
        consumer_group: Consumer group name for coordinated consumption.
        consumer_name: Unique name within the consumer group.
        batch_size: Messages to read per XREADGROUP call.
        block_ms: Milliseconds to block waiting for new messages.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Redis Stream consumer.

        Args:
            config: Application configuration dictionary. Redis settings
                are read from config["redis"]. Environment variables
                REDIS_HOST and REDIS_PORT override config values
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
        self.request_stream: str = redis_config.get(
            "stream_name", "scoring_requests"
        )
        self.response_stream: str = redis_config.get(
            "response_stream", "scoring_responses"
        )
        self.consumer_group: str = redis_config.get(
            "consumer_group", "scoring_consumers"
        )
        self.consumer_name: str = redis_config.get(
            "consumer_name", "consumer-1"
        )
        self.batch_size: int = redis_config.get("consumer_batch_size", 10)
        self.block_ms: int = redis_config.get("consumer_block_ms", 1000)
        self.maxlen: int = redis_config.get("stream_maxlen", 10000)

        self._redis: Optional[Any] = None
        self._connected: bool = False
        self._running: bool = False
        self._scoring_api: Optional[Any] = None
        self._config = config
        self._processed_count: int = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish connection to Redis and set up consumer group.

        Returns:
            True if connection and group setup successful.
        """
        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            self._redis.ping()
            self._connected = True
            self._ensure_consumer_group()
            logger.info(
                "Consumer connected to Redis at %s:%d", self.host, self.port
            )
            return True
        except Exception as exc:
            logger.warning("Failed to connect to Redis: %s", exc)
            self._connected = False
            return False

    def _ensure_consumer_group(self) -> None:
        """Create the consumer group if it doesn't already exist."""
        if self._redis is None:
            return
        try:
            self._redis.xgroup_create(
                self.request_stream,
                self.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                "Created consumer group '%s' on '%s'",
                self.consumer_group, self.request_stream,
            )
        except Exception:
            # Group already exists — that's fine
            pass

    def close(self) -> None:
        """Close the Redis connection and stop processing."""
        self._running = False
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
        self._redis = None
        self._connected = False
        logger.info("Redis consumer connection closed.")

    # ------------------------------------------------------------------
    # Model / ScoringAPI
    # ------------------------------------------------------------------

    def set_scoring_api(self, scoring_api: Any) -> None:
        """Attach a ScoringAPI instance for predictions.

        Args:
            scoring_api: A ScoringAPI (or compatible) object with
                a predict(features=dict) method.
        """
        self._scoring_api = scoring_api

    def _get_scoring_api(self) -> Any:
        """Lazily create a ScoringAPI if none is attached."""
        if self._scoring_api is None:
            from src.models.scoring_api import ScoringAPI
            self._scoring_api = ScoringAPI(self._config)
            logger.info("Created default ScoringAPI (heuristic mode)")
        return self._scoring_api

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def process_message(self, message_id: str, data: Dict[str, str]) -> Dict[str, Any]:
        """Process a single scoring request message.

        Deserializes features from the message, calls the scoring API,
        and returns the prediction result.

        Args:
            message_id: Redis stream message ID.
            data: Message fields as string dict (from XREADGROUP).

        Returns:
            Prediction result dict with churn_probability, risk_level, etc.
        """
        customer_id = data.get("customer_id", "unknown")
        features_json = data.get("features", "{}")
        request_type = data.get("request_type", "score")

        try:
            features = json.loads(features_json)
        except json.JSONDecodeError:
            logger.error(
                "Invalid JSON in features for message %s", message_id
            )
            features = {}

        # Reconstruct full feature dict with customer_id
        features["customer_id"] = customer_id

        api = self._get_scoring_api()
        result = api.predict(features=features)

        # Enrich with message metadata
        result["request_message_id"] = message_id
        result["request_type"] = request_type
        result["scored_at"] = datetime.now(timezone.utc).isoformat()

        return result

    def _write_response(self, result: Dict[str, Any]) -> Optional[str]:
        """Write a prediction result to the response stream.

        Args:
            result: Prediction result dict.

        Returns:
            Message ID of the response, or None on failure.
        """
        if not self._connected or self._redis is None:
            logger.warning("Cannot write response: not connected")
            return None

        try:
            message = {
                "customer_id": str(result.get("customer_id", "")),
                "churn_probability": str(result.get("churn_probability", 0.0)),
                "risk_level": str(result.get("risk_level", "unknown")),
                "recommended_action": str(result.get("recommended_action", "")),
                "model_type": str(result.get("model_type", "")),
                "scored_at": str(result.get("scored_at", "")),
                "request_message_id": str(
                    result.get("request_message_id", "")
                ),
            }

            msg_id = self._redis.xadd(
                self.response_stream,
                message,
                maxlen=self.maxlen,
                approximate=True,
            )

            if isinstance(msg_id, bytes):
                msg_id = msg_id.decode("utf-8")

            return str(msg_id)

        except Exception as exc:
            logger.error("Failed to write response: %s", exc)
            return None

    def _acknowledge(self, message_id: str) -> None:
        """Acknowledge a processed message in the consumer group.

        Args:
            message_id: Redis stream message ID to acknowledge.
        """
        if self._redis is not None:
            try:
                self._redis.xack(
                    self.request_stream,
                    self.consumer_group,
                    message_id,
                )
            except Exception as exc:
                logger.warning("Failed to ACK message %s: %s", message_id, exc)

    def process_one(self) -> Optional[Dict[str, Any]]:
        """Read and process a single message from the stream.

        Performs one XREADGROUP call with a short block time, processes
        the first available message, writes the response, and ACKs.

        Returns:
            Prediction result dict, or None if no message available.
        """
        if not self._connected or self._redis is None:
            raise ConnectionError("Not connected to Redis.")

        try:
            messages = self._redis.xreadgroup(
                self.consumer_group,
                self.consumer_name,
                {self.request_stream: ">"},
                count=1,
                block=100,  # short block for single processing
            )
        except Exception as exc:
            logger.error("XREADGROUP error: %s", exc)
            return None

        if not messages:
            return None

        # messages format: [(stream_name, [(msg_id, data), ...])]
        stream_name, msg_list = messages[0]
        if not msg_list:
            return None

        msg_id, data = msg_list[0]
        result = self.process_message(msg_id, data)
        self._write_response(result)
        self._acknowledge(msg_id)
        self._processed_count += 1

        logger.debug(
            "Processed message %s for customer %s (prob=%.4f)",
            msg_id,
            result.get("customer_id"),
            result.get("churn_probability", 0.0),
        )

        return result

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def start(self, max_iterations: Optional[int] = None) -> None:
        """Start the consumer event loop.

        Continuously reads from the stream, processes messages, and
        writes responses. Gracefully stops on SIGINT/SIGTERM.

        Args:
            max_iterations: Maximum number of read cycles (None = infinite).
        """
        if not self._connected or self._redis is None:
            raise ConnectionError("Not connected to Redis. Call connect().")

        self._running = True
        iteration = 0

        # Graceful shutdown on signals
        def _handle_signal(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, stopping consumer...", signum)
            self._running = False

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (OSError, ValueError):
            # Signal handling may not be available in all contexts
            pass

        logger.info(
            "Consumer started. Reading from '%s', writing to '%s'.",
            self.request_stream, self.response_stream,
        )

        while self._running:
            if max_iterations is not None and iteration >= max_iterations:
                break

            try:
                messages = self._redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.request_stream: ">"},
                    count=self.batch_size,
                    block=self.block_ms,
                )
            except Exception as exc:
                logger.error("XREADGROUP error: %s", exc)
                time.sleep(1)
                iteration += 1
                continue

            if not messages:
                iteration += 1
                continue

            stream_name, msg_list = messages[0]
            for msg_id, data in msg_list:
                try:
                    result = self.process_message(msg_id, data)
                    self._write_response(result)
                    self._acknowledge(msg_id)
                    self._processed_count += 1
                except Exception as exc:
                    logger.error(
                        "Error processing message %s: %s", msg_id, exc
                    )

            iteration += 1

        logger.info(
            "Consumer stopped. Processed %d messages total.",
            self._processed_count,
        )

    def stop(self) -> None:
        """Signal the consumer to stop its event loop."""
        self._running = False

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Check consumer health and Redis connectivity.

        Returns:
            Dict with status, connected, processed_count, and stream info.
        """
        health: Dict[str, Any] = {
            "connected": self._connected,
            "host": self.host,
            "port": self.port,
            "request_stream": self.request_stream,
            "response_stream": self.response_stream,
            "consumer_group": self.consumer_group,
            "consumer_name": self.consumer_name,
            "processed_count": self._processed_count,
        }

        if self._connected and self._redis is not None:
            try:
                self._redis.ping()
                health["status"] = "healthy"
                health["request_stream_length"] = self._redis.xlen(
                    self.request_stream
                )
                health["response_stream_length"] = self._redis.xlen(
                    self.response_stream
                )
            except Exception:
                health["status"] = "unhealthy"
                self._connected = False
        else:
            health["status"] = "unhealthy"

        return health

    @property
    def processed_count(self) -> int:
        """Number of messages processed since start."""
        return self._processed_count
