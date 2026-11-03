"""Kafka Event Producer for LANL Process Intelligence.

Streams genuine LANL process events into Apache Kafka with full support for
Single Host, Entire Domain (DOM1, LAB_DOM), and Entire Organization monitoring scopes,
interleaving clearly-tagged synthetic demonstration events for repeatable testing.
"""

import json
import logging
import threading
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.inference.replay import ProcessEventReplayer
from intelligence.kafka.scope import ScopeManager

logger = logging.getLogger(__name__)


class ProcessEventKafkaProducer:
    """Publishes genuine LANL process events and synthetic anomalies to Kafka across monitoring scopes."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "process-events",
        replayer: Optional[ProcessEventReplayer] = None,
        scope_manager: Optional[ScopeManager] = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.replayer = replayer or ProcessEventReplayer()
        self.scope_manager = scope_manager or ScopeManager()

        self._producer = None
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._published_count = 0
        self._lock = threading.Lock()

        # Build non-test host-to-events index across all available sequences
        self._host_index: Dict[str, List[Dict[str, Any]]] = self._build_host_event_index()

    def _build_host_event_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Index all genuine historical events from non-test sequences by computer identifier."""
        all_seqs = self.replayer.list_available_sequences(None)  # Strict non-test only
        index: Dict[str, List[Dict[str, Any]]] = {}

        for s in all_seqs:
            evs = self.replayer.get_sequence_events(s["sequence_index"])
            for e in evs:
                h = e.get("computer", "")
                if h and h != "<UNK>":
                    if h not in index:
                        index[h] = []
                    index[h].append(dict(e))

        return index

    def connect(self) -> bool:
        """Initialize connection to Kafka broker."""
        try:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
                acks=1,
                request_timeout_ms=5000,
                retries=3,
            )
            logger.info("Kafka producer successfully connected to %s", self.bootstrap_servers)
            return True
        except Exception as e:
            logger.error("Failed to connect Kafka producer to %s: %s", self.bootstrap_servers, e)
            self._producer = None
            return False

    @property
    def is_connected(self) -> bool:
        return self._producer is not None

    def publish_event(self, event_data: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Publish a single process event to Kafka."""
        if not self._producer:
            if not self.connect():
                return False

        try:
            future = self._producer.send(self.topic, key=key, value=event_data)
            future.get(timeout=5.0)
            with self._lock:
                self._published_count += 1
            return True
        except Exception as e:
            logger.error("Error publishing event to Kafka topic '%s': %s", self.topic, e)
            return False

    def build_stream(
        self,
        scope_type: str = "organization",
        target_id: Optional[str] = None,
        max_events: int = 40,
    ) -> List[Dict[str, Any]]:
        """Construct a scoped sequence of genuine LANL events with synthetic anomalies mixed in.

        Supports:
        - 'single_host': Replays events belonging exclusively to target_id.
        - 'domain': Interleaves events belonging to all hosts in target_id (e.g. DOM1, LAB_DOM).
        - 'organization': Interleaves events across all available hosts in the organization.
        """
        # Case A: Single Host Scope
        if scope_type == "single_host":
            if not target_id:
                return []
            host = target_id.upper()
            pool = self._host_index.get(host, [])
            if not pool:
                return []

            stream: List[Dict[str, Any]] = []
            for step in range(1, max_events + 1):
                if step == min(6, max_events) and max_events >= 2:
                    # Injected rare process anomaly on this host
                    base = pool[(step - 1) % len(pool)]
                    anom = self.replayer.generate_simulated_anomaly_event(base, anomaly_type="rare_process")
                    anom["seq_id"] = step
                    anom["computer"] = host
                    stream.append(anom)
                else:
                    ev = dict(pool[(step - 1) % len(pool)])
                    ev["seq_id"] = step
                    ev["computer"] = host
                    stream.append(ev)
            return stream

        # Case B: Domain Scope
        if scope_type == "domain":
            if not target_id:
                return []
            dom_id = target_id.upper()
            dom_hosts = self.scope_manager.get_hosts_for_domain(dom_id)
            active_hosts = [h for h in dom_hosts if h in self._host_index and self._host_index[h]]
            if not active_hosts:
                return []

            stream: List[Dict[str, Any]] = []
            host_ptrs = {h: 0 for h in active_hosts}

            for step in range(1, max_events + 1):
                cur_host = active_hosts[(step - 1) % len(active_hosts)]
                pool = self._host_index[cur_host]
                ptr = host_ptrs[cur_host]
                host_ptrs[cur_host] = ptr + 1

                if step == min(6, max_events) and max_events >= 2:
                    base = pool[ptr % len(pool)]
                    anom = self.replayer.generate_simulated_anomaly_event(base, anomaly_type="rare_process")
                    anom["seq_id"] = step
                    anom["computer"] = cur_host
                    stream.append(anom)
                else:
                    ev = dict(pool[ptr % len(pool)])
                    ev["seq_id"] = step
                    ev["computer"] = cur_host
                    stream.append(ev)
            return stream

        # Case C: Organization Scope (Default multi-host demonstration stream)
        # Interleave hosts fairly: core validation hosts + other enterprise and lab hosts
        ordered_hosts = [
            "C2519", "C1014", "C2519", "C19932", "C2519", "C2519",
            "C1014", "C9945", "C2519", "C19932", "C2519", "C1014",
            "C9945", "C2519", "C2519", "C1014", "C19932", "C2519",
            "C9945", "C1014", "C2519", "C10577", "C2669", "C3422",
            "C108", "C1268", "C231", "C19038", "C1810", "C22409",
            "C395", "C636", "C10", "C9692", "C2091", "C457", "C126"
        ]
        # Filter to hosts present in index
        stream_host_list = [h for h in ordered_hosts if h in self._host_index]
        if not stream_host_list:
            stream_host_list = sorted(list(self._host_index.keys()))

        host_ptrs = {h: 0 for h in stream_host_list}
        stream: List[Dict[str, Any]] = []

        for step in range(1, max_events + 1):
            if step == 6:
                # Step 6: Synthetic rare process on C2519
                pool = self._host_index.get("C2519", self._host_index[stream_host_list[0]])
                base = pool[0]
                anom = self.replayer.generate_simulated_anomaly_event(base, anomaly_type="rare_process")
                anom["seq_id"] = step
                anom["computer"] = base.get("computer", "C2519")
                stream.append(anom)
            elif step == 14:
                # Step 14: Synthetic unusual user on C2519
                pool = self._host_index.get("C2519", self._host_index[stream_host_list[0]])
                base = pool[min(2, len(pool) - 1)]
                anom = self.replayer.generate_simulated_anomaly_event(base, anomaly_type="unusual_user")
                anom["seq_id"] = step
                anom["computer"] = base.get("computer", "C2519")
                stream.append(anom)
            elif step == 22:
                # Step 22: Synthetic unusual host (C10577 in LAB_DOM)
                pool = self._host_index.get("C10577", self._host_index[stream_host_list[0]])
                base = pool[0]
                anom = self.replayer.generate_simulated_anomaly_event(base, anomaly_type="unusual_host")
                anom["seq_id"] = step
                anom["computer"] = "C10577"
                stream.append(anom)
            else:
                cur_host = stream_host_list[(step - 1) % len(stream_host_list)]
                pool = self._host_index[cur_host]
                ptr = host_ptrs[cur_host]
                host_ptrs[cur_host] = ptr + 1

                ev = dict(pool[ptr % len(pool)])
                ev["seq_id"] = step
                ev["computer"] = cur_host
                stream.append(ev)

        return stream

    def build_demonstration_stream(self, max_events: int = 40) -> List[Dict[str, Any]]:
        """Construct demonstration stream preserving baseline assertions."""
        return self.build_stream(scope_type="organization", target_id=None, max_events=max_events)

    def publish_demo_batch(
        self,
        count: int = 15,
        scope_type: str = "organization",
        target_id: Optional[str] = None,
    ) -> int:
        """Publish a fixed batch of scoped demonstration events synchronously."""
        events = self.build_stream(scope_type=scope_type, target_id=target_id, max_events=count)
        sent = 0
        for ev in events:
            ok = self.publish_event(ev, key=str(ev.get("seq_id", sent + 1)))
            if ok:
                sent += 1
        return sent

    def start_streaming(
        self,
        interval_sec: float = 0.8,
        total_events: int = 40,
        scope_type: str = "organization",
        target_id: Optional[str] = None,
    ) -> bool:
        """Start streaming events for the designated scope in a background thread."""
        if self._streaming_thread and self._streaming_thread.is_alive():
            return False

        if not self._producer:
            if not self.connect():
                return False

        self._stop_event.clear()
        self._pause_event.clear()

        def _worker():
            events = self.build_stream(scope_type=scope_type, target_id=target_id, max_events=total_events)
            for ev in events:
                if self._stop_event.is_set():
                    break
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.1)

                self.publish_event(ev, key=str(ev.get("seq_id")))
                time.sleep(interval_sec)

        self._streaming_thread = threading.Thread(target=_worker, daemon=True)
        self._streaming_thread.start()
        return True

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)
        self._streaming_thread = None

    def close(self) -> None:
        self.stop()
        if self._producer:
            try:
                self._producer.flush(timeout=3.0)
                self._producer.close(timeout=3.0)
            except Exception:
                pass
            self._producer = None

    @property
    def published_count(self) -> int:
        with self._lock:
            return self._published_count
