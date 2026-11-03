"""Unit tests for Monitoring Scope Selector and Scope-Based Filtering."""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.inference.engine import ProcessInferenceEngine
from intelligence.inference.anomaly import ProcessAnomalyAnalyzer
from intelligence.inference.replay import ProcessEventReplayer
from intelligence.kafka.consumer import ProcessEventKafkaConsumer
from intelligence.kafka.producer import ProcessEventKafkaProducer
from intelligence.kafka.scope import ScopeManager


class TestMonitoringScopeManager(unittest.TestCase):
    """Test suite for ScopeManager mapping and filtering rules."""

    def setUp(self):
        self.scope_mgr = ScopeManager()

    def test_organization_scope_matches_all_hosts(self):
        """Organization scope matches every event with a valid host."""
        ev1 = {"computer": "C2519", "process": "P4"}
        ev2 = {"computer": "C1014", "process": "P4"}
        ev3 = {"computer": "C10577", "process": "P4"}

        self.assertTrue(self.scope_mgr.matches_scope(ev1, "organization"))
        self.assertTrue(self.scope_mgr.matches_scope(ev2, "organization"))
        self.assertTrue(self.scope_mgr.matches_scope(ev3, "organization"))

    def test_single_host_filtering(self):
        """Single host scope matches only the specified host."""
        ev_target = {"computer": "C2519", "process": "P4"}
        ev_other = {"computer": "C1014", "process": "P4"}

        self.assertTrue(self.scope_mgr.matches_scope(ev_target, "single_host", target_id="C2519"))
        self.assertFalse(self.scope_mgr.matches_scope(ev_other, "single_host", target_id="C2519"))

        # Case-insensitive host match
        self.assertTrue(self.scope_mgr.matches_scope(ev_target, "single_host", target_id="c2519"))

    def test_single_host_unspecified_target(self):
        """Single host scope with None target_id safely returns False."""
        ev = {"computer": "C2519", "process": "P4"}
        self.assertFalse(self.scope_mgr.matches_scope(ev, "single_host", target_id=None))

    def test_domain_filtering_valid_mappings(self):
        """Domain scope matches hosts mapped to that domain."""
        ev_dom1_a = {"computer": "C2519", "process": "P4"}
        ev_dom1_b = {"computer": "C1014", "process": "P4"}
        ev_lab = {"computer": "C10577", "process": "P4"}

        # DOM1 covers enterprise hosts
        self.assertTrue(self.scope_mgr.matches_scope(ev_dom1_a, "domain", target_id="DOM1"))
        self.assertTrue(self.scope_mgr.matches_scope(ev_dom1_b, "domain", target_id="DOM1"))
        self.assertFalse(self.scope_mgr.matches_scope(ev_lab, "domain", target_id="DOM1"))

        # LAB_DOM covers lab hosts
        self.assertTrue(self.scope_mgr.matches_scope(ev_lab, "domain", target_id="LAB_DOM"))
        self.assertFalse(self.scope_mgr.matches_scope(ev_dom1_a, "domain", target_id="LAB_DOM"))

    def test_domain_filtering_invalid_domain(self):
        """Invalid or unmapped domain safely rejects events."""
        ev = {"computer": "C2519", "process": "P4"}
        self.assertFalse(self.scope_mgr.matches_scope(ev, "domain", target_id="NON_EXISTENT_DOM"))

    def test_missing_or_empty_host_handling(self):
        """Events missing computer field are safely rejected across all scopes."""
        ev_missing = {"process": "P4"}
        ev_empty = {"computer": "", "process": "P4"}

        for scope in ["single_host", "domain", "organization"]:
            self.assertFalse(self.scope_mgr.matches_scope(ev_missing, scope, target_id="C2519"))
            self.assertFalse(self.scope_mgr.matches_scope(ev_empty, scope, target_id="C2519"))

    def test_scope_labels(self):
        """Scope manager returns human-readable labels."""
        self.assertIn("C2519", self.scope_mgr.get_scope_label("single_host", "C2519"))
        self.assertIn("DOM1", self.scope_mgr.get_scope_label("domain", "DOM1"))
        self.assertIn("Entire Organization", self.scope_mgr.get_scope_label("organization"))

    def test_all_vocabulary_hosts_categorized(self):
        """Every host in vocabulary is mapped to either DOM1 or LAB_DOM and has metadata."""
        meta = self.scope_mgr.get_scope_metadata()
        hosts = meta["hosts"]
        self.assertEqual(len(hosts), 38)
        self.assertEqual(meta["organization"]["total_hosts"], 38)
        self.assertEqual(meta["organization"]["replayable_hosts_count"], 38)

        domains_by_id = {d["id"]: d for d in meta["domains"]}
        self.assertEqual(domains_by_id["DOM1"]["total_hosts_count"], 35)
        self.assertEqual(domains_by_id["LAB_DOM"]["total_hosts_count"], 3)
        self.assertEqual(set(domains_by_id["LAB_DOM"]["hosts"]), {"C10577", "C2669", "C3422"})

        # Verify all 38 hosts have replayable events > 0
        for h in hosts:
            self.assertGreater(h["event_count"], 0, f"Host {h['id']} has 0 events")
            self.assertTrue(h["has_events"])


class TestProducerStreamGeneration(unittest.TestCase):
    """Test suite for ProcessEventKafkaProducer multi-scope event stream generation."""

    @classmethod
    def setUpClass(cls):
        cls.producer = ProcessEventKafkaProducer()

    def test_all_38_hosts_produce_valid_streams(self):
        """Every single host in the 38-host vocabulary can generate a dedicated stream."""
        vocab_hosts = self.producer.scope_manager.get_all_hosts()
        self.assertEqual(len(vocab_hosts), 38)

        for host in vocab_hosts:
            stream = self.producer.build_stream("single_host", host, max_events=10)
            self.assertGreater(len(stream), 0, f"Host {host} stream is empty")
            # All events in single_host stream MUST belong to that host
            for ev in stream:
                self.assertEqual(ev["computer"], host, f"Event {ev} computer is not {host}")
                self.assertIn("seq_id", ev)
                self.assertIn("user", ev)
                self.assertIn("process", ev)
                self.assertIn("event_type", ev)

    def test_domain_lab_dom_stream(self):
        """LAB_DOM stream produces events strictly from LAB_DOM hosts (C10577, C2669, C3422)."""
        stream = self.producer.build_stream("domain", "LAB_DOM", max_events=30)
        self.assertGreater(len(stream), 0)
        hosts = set(ev["computer"] for ev in stream)
        lab_hosts = {"C10577", "C2669", "C3422"}
        self.assertTrue(hosts.issubset(lab_hosts), f"Stream contained non-LAB hosts: {hosts - lab_hosts}")
        # LAB_DOM has multiple hosts represented
        self.assertGreaterEqual(len(hosts), 2)

    def test_domain_dom1_stream(self):
        """DOM1 stream produces events strictly from DOM1 hosts, excluding LAB_DOM."""
        stream = self.producer.build_stream("domain", "DOM1", max_events=30)
        self.assertGreater(len(stream), 0)
        hosts = set(ev["computer"] for ev in stream)
        lab_hosts = {"C10577", "C2669", "C3422"}
        self.assertTrue(hosts.isdisjoint(lab_hosts), f"Stream contained LAB hosts: {hosts & lab_hosts}")

    def test_organization_stream_multi_host(self):
        """Organization stream interleaves events across multiple hosts and domains."""
        stream = self.producer.build_stream("organization", max_events=50)
        self.assertGreater(len(stream), 0)
        hosts = set(ev["computer"] for ev in stream)
        self.assertGreaterEqual(len(hosts), 10, "Organization stream should interleave many hosts")


class TestKafkaConsumerScopeFiltering(unittest.TestCase):
    """Test suite for ProcessEventKafkaConsumer scope filtering and state isolation."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ProcessInferenceEngine()
        cls.analyzer = ProcessAnomalyAnalyzer(cls.engine)
        cls.replayer = ProcessEventReplayer()

    def setUp(self):
        self.consumer = ProcessEventKafkaConsumer(
            engine=self.engine,
            analyzer=self.analyzer,
            threshold=12.0,
            scope_type="organization",
        )

    def test_consumer_single_host_filtering_and_counts(self):
        """Consumer filters events by single host, updating counters only for in-scope events."""
        self.consumer.set_scope("single_host", target_id="C2519")

        # Event 1: Host C2519 (in-scope)
        ev1 = {"seq_id": 1, "user": "U1723@DOM1", "computer": "C2519", "process": "P810", "event_type": "Start"}
        res1 = self.consumer.process_event(ev1)
        self.assertEqual(res1["status"], "processed")
        self.assertEqual(self.consumer.total_events, 1)

        # Event 2: Host C1014 (outside-scope)
        ev2 = {"seq_id": 2, "user": "U342@DOM1", "computer": "C1014", "process": "P4", "event_type": "Start"}
        res2 = self.consumer.process_event(ev2)
        self.assertEqual(res2["status"], "filtered_out")
        self.assertEqual(res2["reason"], "outside_scope")
        # Total events and history MUST NOT increase for filtered event
        self.assertEqual(self.consumer.total_events, 1)
        self.assertEqual(len(self.consumer.streamed_history), 1)

        # Event 3: Host C2519 continuation (in-scope)
        ev3 = {"seq_id": 3, "user": "U1723@DOM1", "computer": "C2519", "process": "P810", "event_type": "End"}
        res3 = self.consumer.process_event(ev3)
        self.assertEqual(res3["status"], "processed")
        self.assertEqual(self.consumer.total_events, 2)
        self.assertEqual(len(self.consumer.streamed_history), 2)

    def test_consumer_domain_filtering(self):
        """Consumer includes all domain hosts and excludes out-of-domain hosts."""
        self.consumer.set_scope("domain", target_id="DOM1")

        # C2519 and C1014 are in DOM1
        ev_dom1_1 = {"seq_id": 10, "user": "U1723@DOM1", "computer": "C2519", "process": "P810", "event_type": "Start"}
        ev_dom1_2 = {"seq_id": 11, "user": "U342@DOM1", "computer": "C1014", "process": "P4", "event_type": "Start"}
        # C10577 is in LAB_DOM
        ev_lab = {"seq_id": 12, "user": "U3277@C2519", "computer": "C10577", "process": "P10239", "event_type": "Start"}

        self.assertEqual(self.consumer.process_event(ev_dom1_1)["status"], "processed")
        self.assertEqual(self.consumer.process_event(ev_dom1_2)["status"], "processed")
        self.assertEqual(self.consumer.process_event(ev_lab)["status"], "filtered_out")

        self.assertEqual(self.consumer.total_events, 2)

    def test_consumer_lab_dom_filtering(self):
        """Consumer with LAB_DOM scope processes C10577 and C2669, rejecting DOM1 hosts."""
        self.consumer.set_scope("domain", target_id="LAB_DOM")

        ev_lab1 = {"seq_id": 20, "user": "U3277@C2519", "computer": "C10577", "process": "P10239", "event_type": "Start"}
        ev_lab2 = {"seq_id": 21, "user": "U3277@C2519", "computer": "C2669", "process": "P4", "event_type": "Start"}
        ev_dom1 = {"seq_id": 22, "user": "U1723@DOM1", "computer": "C2519", "process": "P810", "event_type": "Start"}

        self.assertEqual(self.consumer.process_event(ev_lab1)["status"], "processed")
        self.assertEqual(self.consumer.process_event(ev_lab2)["status"], "processed")
        self.assertEqual(self.consumer.process_event(ev_dom1)["status"], "filtered_out")
        self.assertEqual(self.consumer.total_events, 2)

    def test_end_to_end_scoring_single_host_and_lab_dom(self):
        """End-to-end event stream processing with causal Transformer anomaly scoring."""
        producer = ProcessEventKafkaProducer()

        # 1. Test single host stream (e.g. LAB_DOM host C10577)
        self.consumer.set_scope("single_host", target_id="C10577")
        stream_c10577 = producer.build_stream("single_host", "C10577", max_events=8)
        for ev in stream_c10577:
            res = self.consumer.process_event(ev)
            self.assertEqual(res["status"], "processed")
            self.assertIn("score", res["event"])
            self.assertIn(res["event"]["status"], ["NORMAL", "UNUSUAL"])

        self.assertEqual(self.consumer.total_events, len(stream_c10577))
        # Check an anomaly was flagged
        self.assertGreater(self.consumer.unusual_events, 0)

        # 2. Test LAB_DOM stream
        self.consumer.set_scope("domain", target_id="LAB_DOM")
        stream_lab = producer.build_stream("domain", "LAB_DOM", max_events=10)
        for ev in stream_lab:
            res = self.consumer.process_event(ev)
            self.assertEqual(res["status"], "processed")

        self.assertEqual(self.consumer.total_events, len(stream_lab))

    def test_scope_change_resets_state_safely(self):
        """Switching scope clears event history and counters to prevent state leakage."""
        self.consumer.set_scope("single_host", target_id="C2519")
        ev = {"seq_id": 1, "user": "U1723@DOM1", "computer": "C2519", "process": "P810", "event_type": "Start"}
        self.consumer.process_event(ev)
        self.assertEqual(self.consumer.total_events, 1)

        # Switch to different host
        res = self.consumer.set_scope("single_host", target_id="C1014")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self.consumer.total_events, 0)
        self.assertEqual(len(self.consumer.streamed_history), 0)
        self.assertEqual(len(self.consumer.recent_events), 0)
        self.assertIsNone(self.consumer.latest_detection)


if __name__ == "__main__":
    unittest.main()

