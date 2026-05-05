"""
src/stream_producer.py
-----------------------
Production-grade async batch producer for the
Real-Time ML Observability Platform.

Architecture:
    FraudModelSimulator -> StreamProducer -> InMemoryEventBroker -> subscribers
                                                                  |- drift_detector
                                                                  |- api
                                                                  |- dashboard

Design goals:
    - Keep producer data-source agnostic
    - Use bounded in-memory buffering for local development
    - Preserve a stable event payload contract for downstream consumers
    - Expose useful producer and broker health metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque

from config.settings import get_settings
from src.model_simulator import FraudModelSimulator, PredictionRecord

logger = logging.getLogger(__name__)


__all__ = [
    "StreamProducer",
    "InMemoryEventBroker",
    "BatchHistory",
    "ProducerMetrics",
    "build_batch_summary",
    "get_event_broker",
    "get_batch_history",
]


# ────────────────────────────────────────────────────────────────
# DATA CONTAINERS
# ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ProducerMetrics:
    batches_sent: int = 0
    messages_published: int = 0
    bytes_sent: int = 0
    dropped_messages: int = 0
    batch_generation_failures: int = 0
    publish_failures: int = 0
    publish_timeouts: int = 0
    last_batch_latency_ms: float = 0.0
    last_batch_id: int = 0


@dataclass(slots=True)
class BatchHistory:
    """
    Bounded in-memory log of recent successfully published batch summaries.
    """
    max_size: int
    _items: Deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.max_size)

    def append(self, item: dict[str, Any]) -> None:
        self._items.append(item)

    def latest(self) -> dict[str, Any] | None:
        return self._items[-1] if self._items else None

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


# ────────────────────────────────────────────────────────────────
# FAN-OUT BROKER
# ────────────────────────────────────────────────────────────────

class InMemoryEventBroker:
    """
    In-process fan-out event broker.

    Each subscriber gets its own bounded queue.
    One published message is delivered independently to all subscribers.

    Backpressure strategy:
        If a subscriber queue is full, drop the oldest message and keep the newest.
    """

    def __init__(self, queue_maxsize: int) -> None:
        self._queue_maxsize = queue_maxsize
        self._subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._dropped_by_subscriber: dict[str, int] = {}

        self._total_publish_calls = 0
        self._total_messages_enqueued = 0
        self._total_subscribers_registered = 0

    async def subscribe(self, name: str) -> asyncio.Queue[dict[str, Any]]:
        if name in self._subscribers:
            raise ValueError(
                f"Subscriber '{name}' already registered. Call unsubscribe() first."
            )

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers[name] = queue
        self._dropped_by_subscriber[name] = 0
        self._total_subscribers_registered += 1

        logger.info(
            "Broker subscriber registered | name=%s | maxsize=%d",
            name,
            self._queue_maxsize,
        )
        return queue

    async def unsubscribe(self, name: str) -> None:
        if name in self._subscribers:
            del self._subscribers[name]
            del self._dropped_by_subscriber[name]
            logger.info("Broker subscriber removed | name=%s", name)

    async def publish(self, message: dict[str, Any]) -> dict[str, int]:
        """
        Deliver one message to all subscribers.

        Returns:
            {
                "subscribers": <count>,
                "published": <successful deliveries>,
                "dropped": <rotated queue insertions>,
            }
        """
        self._total_publish_calls += 1

        published = 0
        dropped = 0

        for name, queue in list(self._subscribers.items()):
            # defensive copy so one consumer cannot mutate another consumer's view
            payload = dict(message)

            try:
                queue.put_nowait(payload)
                published += 1
                self._total_messages_enqueued += 1
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                    self._dropped_by_subscriber[name] += 1
                    published += 1
                    dropped += 1
                    self._total_messages_enqueued += 1

                    logger.warning(
                        "Queue full — dropped oldest | subscriber=%s | batch_id=%s",
                        name,
                        message.get("batch_id"),
                    )
                except Exception:
                    dropped += 1
                    logger.exception(
                        "Failed to rotate queue | subscriber=%s | batch_id=%s",
                        name,
                        message.get("batch_id"),
                    )

        return {
            "subscribers": len(self._subscribers),
            "published": published,
            "dropped": dropped,
        }

    async def get_stats(self) -> dict[str, Any]:
        return {
            "subscriber_count": len(self._subscribers),
            "subscriber_names": sorted(self._subscribers.keys()),
            "total_publish_calls": self._total_publish_calls,
            "total_messages_enqueued": self._total_messages_enqueued,
            "total_subscribers_registered": self._total_subscribers_registered,
            "subscribers": {
                name: {
                    "queue_size": queue.qsize(),
                    "queue_maxsize": queue.maxsize,
                    "queue_full_pct": round(queue.qsize() / queue.maxsize * 100, 1),
                    "dropped_messages": self._dropped_by_subscriber[name],
                }
                for name, queue in self._subscribers.items()
            },
        }


# ────────────────────────────────────────────────────────────────
# LAZY SINGLETONS
# ────────────────────────────────────────────────────────────────

_event_broker: InMemoryEventBroker | None = None
_batch_history: BatchHistory | None = None


def get_event_broker() -> InMemoryEventBroker:
    global _event_broker
    if _event_broker is None:
        cfg = get_settings()
        _event_broker = InMemoryEventBroker(queue_maxsize=cfg.monitoring.queue_maxsize)
        logger.info(
            "EventBroker singleton created | queue_maxsize=%d",
            cfg.monitoring.queue_maxsize,
        )
    return _event_broker


def get_batch_history() -> BatchHistory:
    global _batch_history
    if _batch_history is None:
        cfg = get_settings()
        _batch_history = BatchHistory(max_size=cfg.monitoring.max_batch_log_size)
        logger.info(
            "BatchHistory singleton created | max_size=%d",
            cfg.monitoring.max_batch_log_size,
        )
    return _batch_history


# ────────────────────────────────────────────────────────────────
# SUMMARY BUILDER
# ────────────────────────────────────────────────────────────────

def build_batch_summary(
    batch: list[PredictionRecord],
    *,
    batch_id: int,
    processing_ms: float,
    fraud_rate: float,
    avg_confidence: float,
    drift_injected: bool,
) -> dict[str, Any]:
    """
    Convert a raw batch into a compact event payload.

    This is the contract downstream consumers depend on.
    """
    feature_means: dict[str, float] = {}

    if batch:
        for feature in batch[0].features.keys():
            values = [float(record.features.get(feature, 0.0)) for record in batch]
            feature_means[feature] = round(sum(values) / len(values), 3)

    return {
        "event_version": "1.0",
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_size": len(batch),
        "fraud_rate": fraud_rate,
        "avg_confidence": avg_confidence,
        "drift_injected": drift_injected,
        "processing_ms": round(processing_ms, 2),
        "feature_means": feature_means,
        "predictions_sample": [
            {
                "prediction": record.prediction,
                "label": record.label,
                "confidence": record.confidence,
            }
            for record in batch[:10]
        ],
    }


# ────────────────────────────────────────────────────────────────
# STREAM PRODUCER
# ────────────────────────────────────────────────────────────────

class StreamProducer:
    """
    Continuously generates batches from FraudModelSimulator and publishes
    compact summaries to the event broker.
    """

    _PUBLISH_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        *,
        simulator: FraudModelSimulator | None = None,
        broker: InMemoryEventBroker | None = None,
        history: BatchHistory | None = None,
    ) -> None:
        cfg = get_settings()

        self._interval = cfg.model.interval_seconds
        self._batch_size = cfg.model.batch_size
        self._broker = broker or get_event_broker()
        self._history = history or get_batch_history()
        self._simulator = simulator or FraudModelSimulator()
        self._running = False
        self._metrics = ProducerMetrics()

        logger.info(
            "StreamProducer initialized | interval=%.2fs | batch_size=%d | mode=%s",
            self._interval,
            self._batch_size,
            cfg.model.data_mode,
        )

    # ── PUBLIC ────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            logger.warning("StreamProducer.start() called while already running")
            return

        self._running = True
        logger.info("StreamProducer started")

        try:
            while self._running:
                await self._produce_one_batch()
                await asyncio.sleep(self._interval)

        except asyncio.CancelledError:
            logger.warning("StreamProducer task cancelled")
            self._running = False
            raise

        finally:
            logger.info(
                "StreamProducer stopped | batches=%d | published=%d | bytes=%d | "
                "dropped=%d | generation_failures=%d | publish_failures=%d | timeouts=%d",
                self._metrics.batches_sent,
                self._metrics.messages_published,
                self._metrics.bytes_sent,
                self._metrics.dropped_messages,
                self._metrics.batch_generation_failures,
                self._metrics.publish_failures,
                self._metrics.publish_timeouts,
            )

    def stop(self) -> None:
        self._running = False
        logger.info("StreamProducer stop signal received")

    def is_running(self) -> bool:
        return self._running

    async def get_stats(self) -> dict[str, Any]:
        cfg = get_settings()
        broker_stats = await self._broker.get_stats()

        return {
            "running": self._running,
            "mode": cfg.model.data_mode,
            **asdict(self._metrics),
            "last_batch_latency_ms": round(self._metrics.last_batch_latency_ms, 2),
            "history_size": len(self._history),
            "broker": broker_stats,
        }

    # ── PRIVATE ───────────────────────────────────────────────

    async def _produce_one_batch(self) -> None:
        started = time.perf_counter()

        try:
            batch = self._simulator.generate_batch(self._batch_size)
        except Exception:
            self._metrics.batch_generation_failures += 1
            logger.exception(
                "Failed to generate batch | next_batch_id=%d",
                self._metrics.last_batch_id + 1,
            )
            return

        try:
            stats = self._simulator.stats
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            summary = build_batch_summary(
                batch,
                batch_id=stats.batches_processed,
                processing_ms=elapsed_ms,
                fraud_rate=stats.fraud_rate,
                avg_confidence=stats.avg_confidence,
                drift_injected=stats.drift_injected,
            )

            payload_bytes = len(
                json.dumps(summary, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            )

            try:
                publish_stats = await asyncio.wait_for(
                    self._broker.publish(summary),
                    timeout=self._PUBLISH_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                self._metrics.publish_timeouts += 1
                logger.error(
                    "Broker publish timed out | batch_id=%d | timeout=%.1fs",
                    stats.batches_processed,
                    self._PUBLISH_TIMEOUT_SECONDS,
                )
                return

            self._history.append(summary)

            self._metrics.batches_sent += 1
            self._metrics.messages_published += publish_stats["published"]
            self._metrics.dropped_messages += publish_stats["dropped"]
            self._metrics.bytes_sent += payload_bytes
            self._metrics.last_batch_latency_ms = elapsed_ms
            self._metrics.last_batch_id = stats.batches_processed

            logger.info(
                "Batch produced | id=%03d | fraud=%.1f%% | confidence=%.3f | "
                "drift=%s | latency=%.2fms | subscribers=%d | dropped=%d",
                stats.batches_processed,
                stats.fraud_rate * 100,
                stats.avg_confidence,
                stats.drift_injected,
                elapsed_ms,
                publish_stats["subscribers"],
                publish_stats["dropped"],
            )

        except Exception:
            self._metrics.publish_failures += 1
            logger.exception(
                "StreamProducer failed after generation | batch_id=%d",
                self._metrics.last_batch_id + 1,
            )


# ────────────────────────────────────────────────────────────────
# ENTRYPOINT TEST
# ────────────────────────────────────────────────────────────────

async def _consumer(name: str, duration_seconds: float = 8.0) -> None:
    queue = await get_event_broker().subscribe(name)
    deadline = time.perf_counter() + duration_seconds

    try:
        while time.perf_counter() < deadline:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                logger.info(
                    "Consumer received | name=%s | batch_id=%03d | fraud=%.1f%% | drift=%s",
                    name,
                    item["batch_id"],
                    item["fraud_rate"] * 100,
                    item["drift_injected"],
                )
            except asyncio.TimeoutError:
                continue
    finally:
        await get_event_broker().unsubscribe(name)


async def _test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = get_settings()
    duration = 8 * cfg.model.interval_seconds
    producer = StreamProducer()

    producer_task = asyncio.create_task(producer.start())

    try:
        await asyncio.gather(
            _consumer("drift_detector", duration_seconds=duration),
            _consumer("dashboard", duration_seconds=duration),
        )
    finally:
        producer.stop()
        await producer_task

    print("\n── Producer stats ──")
    stats = await producer.get_stats()
    print(json.dumps(stats, indent=2))

    print("\n── Latest batch ──")
    latest = get_batch_history().latest()
    if latest:
        print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    asyncio.run(_test())