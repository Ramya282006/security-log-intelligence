"""Kafka Event Consumer and Real-Time Transformer Anomaly Scorer.

Consumes process events from Apache Kafka, scores each event through the existing
Transformer inference engine and anomaly analyzer using only preceding event history,
and maintains live state for the analyst dashboard.
"""

from collections import deque
import json
import logging
import threading
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.inference.engine import ProcessInferenceEngine
from intelligence.inference.anomaly import ProcessAnomalyAnalyzer
from intelligence.kafka.scope import ScopeManager

logger = logging.getLogger(__name__)


class ProcessEventKafkaConsumer:
    """Consumes Kafka process-event stream and performs causal Transformer anomaly scoring."""

    def __init__(
        self,
        engine: ProcessInferenceEngine,
        analyzer: ProcessAnomalyAnalyzer,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "process-events",
        group_id: str = "security-intelligence-group",
        max_history: int = 80,
        threshold: float = 12.0,
        scope_manager: Optional[ScopeManager] = None,
        scope_type: str = "organization",
        scope_target_id: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.analyzer = analyzer
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.max_history = max_history
        self.threshold = threshold
        self.scope_manager = scope_manager or ScopeManager()
        self.active_scope = {
            "type": scope_type or "organization",
            "target_id": scope_target_id,
        }

        self._consumer = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        # In-memory processing state
        self.streamed_history: List[Dict[str, Any]] = []
        self.processed_seq_ids: Set[Any] = set()
        self.recent_events: deque = deque(maxlen=50)
        self.total_events: int = 0
        self.normal_events: int = 0
        self.unusual_events: int = 0
        self.latest_detection: Optional[Dict[str, Any]] = None
        self.last_score: Optional[Dict[str, Any]] = None

    def connect(self) -> bool:
        """Connect to Kafka topic as a consumer."""
        try:
            from kafka import KafkaConsumer

            self._consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=1000,
            )
            logger.info("Kafka consumer connected to %s on topic '%s'", self.bootstrap_servers, self.topic)
            return True
        except Exception as e:
            logger.error("Failed to connect Kafka consumer: %s", e)
            self._consumer = None
            return False

    @property
    def is_connected(self) -> bool:
        return self._consumer is not None

    def reset_state(self) -> None:
        """Clear event history, seen IDs, and counters."""
        with self._lock:
            self.streamed_history.clear()
            self.processed_seq_ids.clear()
            self.recent_events.clear()
            self.total_events = 0
            self.normal_events = 0
            self.unusual_events = 0
            self.latest_detection = None
            self.last_score = None

    def set_scope(self, scope_type: str, target_id: Optional[str] = None) -> Dict[str, Any]:
        """Update active monitoring scope and reset stream state cleanly."""
        with self._lock:
            self.active_scope = {
                "type": scope_type,
                "target_id": target_id,
            }
            self.streamed_history.clear()
            self.processed_seq_ids.clear()
            self.recent_events.clear()
            self.total_events = 0
            self.normal_events = 0
            self.unusual_events = 0
            self.latest_detection = None
            self.last_score = None
            label = self.scope_manager.get_scope_label(scope_type, target_id)
            logger.info("Monitoring scope updated to: %s", label)
            return {
                "status": "ok",
                "scope_type": scope_type,
                "target_id": target_id,
                "label": label,
            }

    def process_event(self, incoming_event: Dict[str, Any]) -> Dict[str, Any]:
        """Process one incoming event through the causal Transformer anomaly pipeline.

        Strict causal guarantee: The incoming event is evaluated against preceding
        history BEFORE being appended to that history.
        """
        with self._lock:
            # 0. Check scope filtering
            if not self.scope_manager.matches_scope(incoming_event, self.active_scope["type"], self.active_scope.get("target_id")):
                return {
                    "status": "filtered_out",
                    "reason": "outside_scope",
                    "computer": incoming_event.get("computer"),
                    "active_scope": self.active_scope,
                }

            seq_id = incoming_event.get("seq_id")
            if seq_id is not None:
                if seq_id in self.processed_seq_ids:
                    logger.debug("Duplicate event skipped: seq_id=%s", seq_id)
                    return {"status": "duplicate_skipped", "seq_id": seq_id}
                self.processed_seq_ids.add(seq_id)

            # 1. Causal Anomaly Scoring against preceding history
            score_dict: Optional[Dict[str, Any]] = None
            is_unusual = False
            short_reason = "Routine activity matching baseline"

            if self.streamed_history:
                preds, score = self.analyzer.evaluate_observation(
                    self.streamed_history, incoming_event, custom_threshold=self.threshold
                )
                is_unusual = score.is_anomaly
                score_dict = {
                    "total_surprise": round(score.total_surprise, 2),
                    "is_anomaly": score.is_anomaly,
                    "severity": score.severity,
                    "threshold": round(score.threshold, 2),
                    "head_surprises": {k: round(v, 2) for k, v in score.head_surprises.items()},
                    "dominant_factor": score.dominant_factor,
                    "explanation": score.explanation,
                }
                if is_unusual:
                    short_reason = (
                        f"Unusual {score.dominant_factor}: observed '{incoming_event.get(score.dominant_factor)}' "
                        f"(surprise: {score.head_surprises.get(score.dominant_factor, 0.0):.1f})"
                    )

            # 2. Append event to chronological history
            self.streamed_history.append(dict(incoming_event))
            if len(self.streamed_history) > self.max_history:
                self.streamed_history = self.streamed_history[-self.max_history:]

            # 3. Update summary metrics
            self.total_events += 1
            if is_unusual:
                self.unusual_events += 1
                self.latest_detection = {
                    "observed": {
                        "user": incoming_event.get("user"),
                        "computer": incoming_event.get("computer"),
                        "process": incoming_event.get("process"),
                        "event_type": incoming_event.get("event_type"),
                        "source": incoming_event.get("source"),
                    },
                    "anomaly_score": score_dict["total_surprise"] if score_dict else 0.0,
                    "threshold": score_dict["threshold"] if score_dict else 0.0,
                    "severity": score_dict["severity"] if score_dict else "NORMAL",
                    "reason": short_reason,
                    "timestamp": time.time(),
                }
            else:
                self.normal_events += 1

            self.last_score = score_dict

            # 4. Record event entry for UI
            event_entry = {
                "seq_id": seq_id or self.total_events,
                "user": incoming_event.get("user", "Unknown"),
                "computer": incoming_event.get("computer", "Unknown"),
                "process": incoming_event.get("process", "Unknown"),
                "event_type": incoming_event.get("event_type", "Unknown"),
                "status": "UNUSUAL" if is_unusual else "NORMAL",
                "score": score_dict["total_surprise"] if score_dict else None,
                "reason": short_reason if is_unusual else "",
                "source": incoming_event.get("source", "LANL Dataset"),
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self.recent_events.appendleft(event_entry)

            return {
                "status": "processed",
                "event": event_entry,
                "is_unusual": is_unusual,
                "total_events": self.total_events,
                "normal_events": self.normal_events,
                "unusual_events": self.unusual_events,
            }

    def start(self) -> bool:
        """Start consumer thread in background."""
        if self._consumer_thread and self._consumer_thread.is_alive():
            return True

        if not self._consumer:
            if not self.connect():
                return False

        self._stop_event.clear()
        self._pause_event.clear()

        def _consume_loop():
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.2)
                    continue

                try:
                    records = self._consumer.poll(timeout_ms=500, max_records=5)
                    for topic_partition, messages in records.items():
                        for msg in messages:
                            if self._stop_event.is_set():
                                break
                            if self._pause_event.is_set():
                                break
                            self.process_event(msg.value)
                except Exception as e:
                    logger.error("Consumer loop error: %s", e)
                    time.sleep(1.0)

        self._consumer_thread = threading.Thread(target=_consume_loop, daemon=True)
        self._consumer_thread.start()
        return True

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=2.0)
        self._consumer_thread = None

    def close(self) -> None:
        self.stop()
        if self._consumer:
            try:
                self._consumer.close()
            except Exception:
                pass
            self._consumer = None

    def get_status(self) -> Dict[str, Any]:
        """Return snapshot for dashboard rendering."""
        with self._lock:
            return {
                "kafka_connected": self.is_connected,
                "bootstrap_servers": self.bootstrap_servers,
                "topic": self.topic,
                "is_running": (self._consumer_thread is not None and self._consumer_thread.is_alive()),
                "is_paused": self._pause_event.is_set(),
                "total_events": self.total_events,
                "normal_events": self.normal_events,
                "unusual_events": self.unusual_events,
                "latest_detection": self.latest_detection,
                "recent_events": list(self.recent_events),
                "threshold": self.threshold,
                "active_scope": {
                    "type": self.active_scope["type"],
                    "target_id": self.active_scope.get("target_id"),
                    "label": self.scope_manager.get_scope_label(self.active_scope["type"], self.active_scope.get("target_id")),
                },
                "scope_metadata": self.scope_manager.get_scope_metadata(),
            }
