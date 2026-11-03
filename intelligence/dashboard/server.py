"""Process Activity Intelligence Analyst Dashboard HTTP Server.

Serves an interactive security analyst dashboard providing:
1. Progressive sequence replay from real LANL validation data.
2. Real-time Transformer next-event predictions and confidence distributions.
3. Negative log likelihood surprise scoring with validation-calibrated thresholds.
4. Flagged event investigation timeline with evidence attribution.
5. Interactive controls to inject simulated anomalies or submit custom events.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.inference.engine import ProcessInferenceEngine
from intelligence.inference.anomaly import ProcessAnomalyAnalyzer
from intelligence.inference.replay import ProcessEventReplayer
from intelligence.kafka.producer import ProcessEventKafkaProducer
from intelligence.kafka.consumer import ProcessEventKafkaConsumer
from intelligence.kafka.scope import ScopeManager

STATIC_DIR = Path(__file__).parent / "static"


class DashboardSessionState:
    """Maintains in-memory state for live sequence replay and analyst triage."""

    def __init__(self, engine: ProcessInferenceEngine, analyzer: ProcessAnomalyAnalyzer, replayer: ProcessEventReplayer) -> None:
        self.engine = engine
        self.analyzer = analyzer
        self.replayer = replayer

        self.available_sequences = self.replayer.list_available_sequences(split="validation")
        self.active_sequence_index: int = self.available_sequences[0]["sequence_index"] if self.available_sequences else 0
        self.full_sequence_events: List[Dict[str, Any]] = []
        self.current_step_idx: int = 0
        self.streamed_history: List[Dict[str, Any]] = []
        self.flagged_timeline: List[Dict[str, Any]] = []
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_score: Optional[Dict[str, Any]] = None
        self.custom_threshold: Optional[float] = None

        # Monitoring Scope & Kafka integration
        self.scope_manager = ScopeManager()
        self.kafka_producer = ProcessEventKafkaProducer(replayer=self.replayer)
        self.kafka_consumer = ProcessEventKafkaConsumer(
            engine=self.engine,
            analyzer=self.analyzer,
            threshold=12.0,
            scope_manager=self.scope_manager,
            scope_type="organization",
        )

        self._load_active_sequence(self.active_sequence_index)

    def _load_active_sequence(self, seq_idx: int) -> None:
        self.active_sequence_index = seq_idx
        self.full_sequence_events = self.replayer.get_sequence_events(seq_idx)
        self.current_step_idx = 0
        self.streamed_history = []
        self.flagged_timeline = []
        self.last_prediction = None
        self.last_score = None

    def reset(self) -> None:
        self._load_active_sequence(self.active_sequence_index)

    def select_sequence(self, seq_idx: int) -> Dict[str, Any]:
        self._load_active_sequence(seq_idx)
        return {
            "status": "ok",
            "active_sequence_index": self.active_sequence_index,
            "total_events": len(self.full_sequence_events),
        }

    def step(self) -> Dict[str, Any]:
        """Advance to the next event, evaluate surprise, and predict the subsequent event."""
        if self.current_step_idx >= len(self.full_sequence_events):
            return {
                "status": "completed",
                "message": "Sequence replay completed. Reset or select another sequence.",
            }

        next_event = dict(self.full_sequence_events[self.current_step_idx])
        self.current_step_idx += 1

        return self._process_incoming_event(next_event)

    def inject_anomaly(self, anomaly_type: str = "rare_process") -> Dict[str, Any]:
        """Inject a simulated demonstration event to demonstrate divergence."""
        if not self.streamed_history:
            # Need at least one event in history to form context
            base_event = self.full_sequence_events[0] if self.full_sequence_events else {
                "user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "Start"
            }
        else:
            base_event = self.streamed_history[-1]

        sim_event = self.replayer.generate_simulated_anomaly_event(base_event, anomaly_type)
        return self._process_incoming_event(sim_event)

    def submit_custom_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a user-submitted event and evaluate surprise."""
        custom_event = {
            "user": str(event_data.get("user", "U8946@DOM1")),
            "computer": str(event_data.get("computer", "C19038")),
            "process": str(event_data.get("process", "P4")),
            "event_type": str(event_data.get("event_type", "Start")),
            "time_delta": float(event_data.get("time_delta", 1.0)),
            "event_duration": float(event_data.get("event_duration", 0.0)),
            "event_count": len(self.streamed_history) + 1,
            "sequence_position": len(self.streamed_history) + 1,
            "source": "Analyst Manual Submission (Custom Test)",
        }
        return self._process_incoming_event(custom_event)

    def _process_incoming_event(self, incoming_event: Dict[str, Any]) -> Dict[str, Any]:
        """Run Transformer inference, calculate surprise, and update timeline."""
        # If there is history, evaluate surprise of incoming event
        score_dict: Optional[Dict[str, Any]] = None
        if self.streamed_history:
            preds, score = self.analyzer.evaluate_observation(
                self.streamed_history, incoming_event, custom_threshold=self.custom_threshold
            )
            score_dict = {
                "total_surprise": round(score.total_surprise, 2),
                "is_anomaly": score.is_anomaly,
                "severity": score.severity,
                "threshold": round(score.threshold, 2),
                "head_surprises": {k: round(v, 2) for k, v in score.head_surprises.items()},
                "dominant_factor": score.dominant_factor,
                "explanation": score.explanation,
                "investigation_recommendation": score.investigation_recommendation,
                "confidence_warning": score.confidence_warning,
            }
            if score.is_anomaly:
                self.flagged_timeline.append({
                    "step": len(self.streamed_history) + 1,
                    "event": incoming_event,
                    "score": score_dict,
                })

        # Append to history
        self.streamed_history.append(incoming_event)

        # Predict next event given updated history
        next_preds = self.engine.predict_next(self.streamed_history, top_k=3)
        formatted_preds = {}
        for feat, data in next_preds.items():
            formatted_preds[feat] = {
                "predicted_identifier": data["predicted_identifier"],
                "confidence_pct": round(data["confidence"] * 100.0, 1),
                "top_candidates": [
                    {"identifier": c["identifier"], "probability_pct": round(c["probability"] * 100.0, 1)}
                    for c in data["top_candidates"]
                ],
            }

        self.last_prediction = formatted_preds
        self.last_score = score_dict

        return {
            "status": "ok",
            "current_step": len(self.streamed_history),
            "total_sequence_steps": len(self.full_sequence_events),
            "incoming_event": incoming_event,
            "anomaly_score": score_dict,
            "next_prediction": formatted_preds,
            "flagged_count": len(self.flagged_timeline),
        }

    def set_monitoring_scope(self, scope_type: str, target_id: Optional[str] = None) -> Dict[str, Any]:
        """Update active monitoring scope for Kafka event processing."""
        if self.kafka_consumer:
            return self.kafka_consumer.set_scope(scope_type, target_id)
        return {"status": "error", "message": "Kafka consumer not initialized"}

    def start_kafka_stream(
        self,
        interval_sec: float = 0.8,
        total_events: int = 40,
        scope_type: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start both Kafka consumer and producer stream with selected monitoring scope."""
        self.stop_kafka_stream()

        if scope_type and self.kafka_consumer:
            self.kafka_consumer.set_scope(scope_type, target_id)
        elif self.kafka_consumer:
            scope_type = self.kafka_consumer.active_scope.get("type", "organization")
            target_id = self.kafka_consumer.active_scope.get("target_id")

        c_ok = self.kafka_consumer.start()
        if not c_ok:
            return {"status": "error", "message": "Failed to connect Kafka consumer to broker"}

        p_ok = self.kafka_producer.start_streaming(
            interval_sec=interval_sec,
            total_events=total_events,
            scope_type=scope_type or "organization",
            target_id=target_id,
        )
        if not p_ok:
            self.kafka_consumer.stop()
            return {"status": "error", "message": "Failed to connect Kafka producer to broker"}

        return {
            "status": "ok",
            "message": "Kafka event stream and real-time inference started",
            "active_scope": self.kafka_consumer.active_scope if self.kafka_consumer else None,
        }

    def pause_kafka_stream(self) -> Dict[str, Any]:
        if self.kafka_producer:
            self.kafka_producer.pause()
        if self.kafka_consumer:
            self.kafka_consumer.pause()
        return {"status": "ok", "message": "Kafka stream paused"}

    def resume_kafka_stream(self) -> Dict[str, Any]:
        if self.kafka_producer:
            self.kafka_producer.resume()
        if self.kafka_consumer:
            self.kafka_consumer.resume()
        return {"status": "ok", "message": "Kafka stream resumed"}

    def stop_kafka_stream(self) -> Dict[str, Any]:
        if self.kafka_producer:
            self.kafka_producer.stop()
        if self.kafka_consumer:
            self.kafka_consumer.stop()
            self.kafka_consumer.reset_state()
        return {"status": "ok", "message": "Kafka stream stopped and reset"}

    def get_kafka_status(self) -> Dict[str, Any]:
        if self.kafka_consumer:
            return self.kafka_consumer.get_status()
        return {
            "kafka_connected": False,
            "is_running": False,
            "total_events": 0,
            "normal_events": 0,
            "unusual_events": 0,
            "latest_detection": None,
            "recent_events": [],
        }


def make_dashboard_handler(session: DashboardSessionState):
    """Factory creating the HTTP request handler bound to the session state."""

    class DashboardHandler(BaseHTTPRequestHandler):

        def _send_json(self, data: Any, status: int = 200) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        def _send_html(self, html_content: str) -> None:
            payload = html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                html_file = STATIC_DIR / "index.html"
                if html_file.exists():
                    self._send_html(html_file.read_text(encoding="utf-8"))
                else:
                    self._send_json({"error": "Static HTML not found"}, status=404)
            elif path == "/api/status":
                data = {
                    "checkpoint": session.engine.checkpoint_meta["checkpoint_name"],
                    "checkpoint_epoch": session.engine.checkpoint_meta["epoch"],
                    "best_val_loss": session.engine.checkpoint_meta["best_val_loss"],
                    "thresholds": session.analyzer.thresholds,
                    "active_sequence_index": session.active_sequence_index,
                    "streamed_steps": len(session.streamed_history),
                    "total_sequence_steps": len(session.full_sequence_events),
                    "flagged_count": len(session.flagged_timeline),
                    "last_score": session.last_score,
                    "last_prediction": session.last_prediction,
                }
                self._send_json(data)
            elif path == "/api/sequences":
                self._send_json(session.available_sequences)
            elif path == "/api/flagged":
                self._send_json(session.flagged_timeline)
            elif path == "/api/history":
                self._send_json(session.streamed_history)
            elif path == "/api/kafka/status":
                self._send_json(session.get_kafka_status())
            elif path == "/api/kafka/scope":
                self._send_json({
                    "active_scope": session.kafka_consumer.active_scope if session.kafka_consumer else None,
                    "metadata": session.scope_manager.get_scope_metadata(),
                })
            else:
                self._send_json({"error": "Not Found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            content_len = int(self.headers.get("Content-Length", 0))
            body_data = {}
            if content_len > 0:
                raw_body = self.rfile.read(content_len).decode("utf-8")
                try:
                    body_data = json.loads(raw_body)
                except Exception:
                    pass

            if path == "/api/step":
                res = session.step()
                self._send_json(res)
            elif path == "/api/reset":
                session.reset()
                self._send_json({"status": "ok", "message": "Sequence reset to beginning."})
            elif path == "/api/select_sequence":
                seq_idx = int(body_data.get("sequence_index", 0))
                res = session.select_sequence(seq_idx)
                self._send_json(res)
            elif path == "/api/inject_anomaly":
                anomaly_type = str(body_data.get("anomaly_type", "rare_process"))
                res = session.inject_anomaly(anomaly_type)
                self._send_json(res)
            elif path == "/api/submit_event":
                res = session.submit_custom_event(body_data)
                self._send_json(res)
            elif path == "/api/set_threshold":
                thresh = float(body_data.get("threshold", 11.5))
                session.custom_threshold = thresh
                self._send_json({"status": "ok", "custom_threshold": thresh})
            elif path == "/api/kafka/start":
                interval = float(body_data.get("interval", 0.8))
                count = int(body_data.get("count", 40))
                scope_type = body_data.get("scope_type")
                target_id = body_data.get("target_id")
                res = session.start_kafka_stream(
                    interval_sec=interval, total_events=count, scope_type=scope_type, target_id=target_id
                )
                self._send_json(res)
            elif path == "/api/kafka/scope":
                scope_type = str(body_data.get("scope_type", "organization"))
                target_id = body_data.get("target_id")
                res = session.set_monitoring_scope(scope_type, target_id)
                self._send_json(res)
            elif path == "/api/kafka/pause":
                res = session.pause_kafka_stream()
                self._send_json(res)
            elif path == "/api/kafka/resume":
                res = session.resume_kafka_stream()
                self._send_json(res)
            elif path == "/api/kafka/stop":
                res = session.stop_kafka_stream()
                self._send_json(res)
            else:
                self._send_json({"error": "Unknown API endpoint"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:
            # Suppress default noisy access logs to keep terminal readable
            pass

    return DashboardHandler


def create_dashboard_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    checkpoint_path: Optional[Path] = None,
) -> HTTPServer:
    """Create and configure the HTTP dashboard server."""
    engine = ProcessInferenceEngine(checkpoint_path=checkpoint_path)
    analyzer = ProcessAnomalyAnalyzer(engine)
    replayer = ProcessEventReplayer()
    session = DashboardSessionState(engine, analyzer, replayer)

    handler = make_dashboard_handler(session)
    server = HTTPServer((host, port), handler)
    return server


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8080,
    checkpoint_path: Optional[Path] = None,
) -> None:
    """Start the dashboard server and serve indefinitely."""
    server = create_dashboard_server(host=host, port=port, checkpoint_path=checkpoint_path)
    print("=" * 72)
    print(f"  Process Activity Intelligence Analyst Dashboard")
    print(f"  Running locally at: http://{host}:{port}/")
    print(f"  Loaded model: {Path(checkpoint_path).name if checkpoint_path else 'models/lanl_run2_best.pt'}")
    print("  Press Ctrl+C to terminate the dashboard server.")
    print("=" * 72)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_dashboard(host="127.0.0.1", port=8080)

