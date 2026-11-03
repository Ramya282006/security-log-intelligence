"""Integration and unit tests for Apache Kafka pipeline."""

import unittest
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.inference.engine import ProcessInferenceEngine
from intelligence.inference.anomaly import ProcessAnomalyAnalyzer
from intelligence.inference.replay import ProcessEventReplayer
from intelligence.kafka.producer import ProcessEventKafkaProducer
from intelligence.kafka.consumer import ProcessEventKafkaConsumer


class TestKafkaPipeline(unittest.TestCase):
    """Test suite for Kafka producer, consumer, and streaming logic."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ProcessInferenceEngine()
        cls.analyzer = ProcessAnomalyAnalyzer(cls.engine)
        cls.replayer = ProcessEventReplayer()

    def test_producer_demonstration_stream_structure(self):
        """Verify demonstration stream mixes genuine LANL events with synthetic anomalies."""
        producer = ProcessEventKafkaProducer(replayer=self.replayer)
        stream = producer.build_demonstration_stream(max_events=25)

        self.assertEqual(len(stream), 25)
        # Verify step 6 is synthetic rare process
        self.assertEqual(stream[5]["seq_id"], 6)
        self.assertEqual(stream[5]["process"], "P8881")
        self.assertIn("Synthetic Anomaly", stream[5]["source"])

        # Verify step 14 is unusual user
        self.assertEqual(stream[13]["seq_id"], 14)
        self.assertIn("Synthetic Anomaly", stream[13]["source"])

        # Verify step 1 is genuine LANL
        self.assertIn("LANL Dataset", stream[0]["source"])

    def test_consumer_causal_scoring_and_counters(self):
        """Verify consumer causal scoring updates counters and scores without target leakage."""
        consumer = ProcessEventKafkaConsumer(
            engine=self.engine,
            analyzer=self.analyzer,
            threshold=12.0,
        )
        consumer.reset_state()

        # Use genuine validation sequence events
        valid_events = self.replayer.get_sequence_events(0)
        ev1 = dict(valid_events[0])
        ev1["seq_id"] = 1
        res1 = consumer.process_event(ev1)
        self.assertEqual(res1["status"], "processed")
        self.assertFalse(res1["is_unusual"])
        self.assertEqual(consumer.total_events, 1)
        self.assertEqual(consumer.normal_events, 1)
        self.assertEqual(consumer.unusual_events, 0)
        self.assertEqual(len(consumer.streamed_history), 1)

        # Step 2: Normal continuation (actual score ~1.03)
        ev2 = dict(valid_events[1])
        ev2["seq_id"] = 2
        res2 = consumer.process_event(ev2)
        self.assertEqual(res2["status"], "processed")
        self.assertFalse(res2["is_unusual"])
        self.assertEqual(consumer.total_events, 2)
        self.assertEqual(consumer.normal_events, 2)

        # Step 3: Injected extreme surprise event (rare process P8881, actual score ~16.99)
        ev3 = self.replayer.generate_simulated_anomaly_event(valid_events[1], anomaly_type="rare_process")
        ev3["seq_id"] = 3
        res3 = consumer.process_event(ev3)
        self.assertEqual(res3["status"], "processed")
        self.assertTrue(res3["is_unusual"])
        self.assertEqual(consumer.total_events, 3)
        self.assertEqual(consumer.unusual_events, 1)
        self.assertIsNotNone(consumer.latest_detection)
        self.assertGreater(consumer.latest_detection["anomaly_score"], 12.0)

    def test_consumer_duplicate_event_rejection(self):
        """Verify events with identical seq_id are skipped to prevent duplicate scoring."""
        consumer = ProcessEventKafkaConsumer(
            engine=self.engine,
            analyzer=self.analyzer,
        )
        consumer.reset_state()

        ev = {"seq_id": 101, "user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "Start"}
        res1 = consumer.process_event(ev)
        self.assertEqual(res1["status"], "processed")

        res2 = consumer.process_event(ev)
        self.assertEqual(res2["status"], "duplicate_skipped")
        self.assertEqual(consumer.total_events, 1)

    def test_consumer_reset_state(self):
        """Verify reset clears all history and counters cleanly."""
        consumer = ProcessEventKafkaConsumer(
            engine=self.engine,
            analyzer=self.analyzer,
        )
        ev = {"seq_id": 201, "user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "Start"}
        consumer.process_event(ev)
        self.assertEqual(consumer.total_events, 1)

        consumer.reset_state()
        self.assertEqual(consumer.total_events, 0)
        self.assertEqual(consumer.normal_events, 0)
        self.assertEqual(consumer.unusual_events, 0)
        self.assertEqual(len(consumer.streamed_history), 0)
        self.assertIsNone(consumer.latest_detection)


if __name__ == "__main__":
    unittest.main()
