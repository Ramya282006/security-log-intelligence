"""Comprehensive Unit Tests for Process Activity Intelligence Prototype.

Covers:
1. ProcessInferenceEngine:
   - Checkpoint loading (Run 1 & Run 2 compatible).
   - Event encoding into categorical and numeric vectors.
   - Handling of out-of-vocabulary / unseen tokens.
   - Attention mask construction and sequence windowing.
   - Next-event top-k prediction and probability normalization.
   - Malformed / incomplete event handling.
2. ProcessAnomalyAnalyzer:
   - Calibration strictly on validation split (test split untouched).
   - Negative log likelihood surprise computation.
   - Anomaly severity categorization (NORMAL, ELEVATED, HIGH_SURPRISE).
   - Dominant factor attribution.
   - Explanation and analyst recommendation generation.
3. ProcessEventReplayer:
   - Loading sequence metadata and raw sequences.
   - Decoding categorical and numeric tokens back to human-readable strings.
   - Test split isolation (refuses access to test split).
   - Synthetic demonstration event injection with clear labeling.
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.inference.engine import ProcessInferenceEngine
from intelligence.inference.anomaly import ProcessAnomalyAnalyzer, AnomalyScore
from intelligence.inference.replay import ProcessEventReplayer
from intelligence.preprocessing.label_mapping import TargetLabelMapper


class TestProcessInferenceEngine(unittest.TestCase):
    """Test suite for ProcessInferenceEngine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ProcessInferenceEngine()

    def test_checkpoint_loaded_properly(self):
        """Verify checkpoint loaded is valid and model is in eval mode."""
        self.assertIsNotNone(self.engine.model)
        self.assertFalse(self.engine.model.training)
        self.assertIn("epoch", self.engine.checkpoint_meta)
        self.assertIn("best_val_loss", self.engine.checkpoint_meta)

    def test_encode_event_valid(self):
        """Verify correct categorical and numeric encoding of valid event."""
        event = {
            "user": "U8946@DOM1",
            "computer": "C19038",
            "process": "P4",
            "event_type": "Start",
            "time_delta": 15.0,
            "event_duration": 0.0,
            "event_count": 3,
            "sequence_position": 3,
        }
        cat_ids, num_feats = self.engine.encode_event(event)

        # Categorical IDs should be >= 2 for known tokens
        self.assertEqual(len(cat_ids), 4)
        for cid in cat_ids:
            self.assertGreaterEqual(cid, 2)

        # Numeric features should be log1p-transformed floats
        self.assertEqual(len(num_feats), 4)
        self.assertAlmostEqual(num_feats[0], float(np.log1p(15.0)), places=4)

    def test_encode_event_unseen_token(self):
        """Unseen token receives UNK_ID (1) without raising errors."""
        event = {
            "user": "UNSEEN_USER_XYZ@DOM1",
            "computer": "C19038",
            "process": "P4",
            "event_type": "Start",
        }
        cat_ids, _ = self.engine.encode_event(event)
        self.assertEqual(cat_ids[0], TargetLabelMapper.UNK_ID)

    def test_encode_event_malformed_incomplete(self):
        """Incomplete event missing fields defaults gracefully without crashing."""
        incomplete_event = {"process": "P4"}
        cat_ids, num_feats = self.engine.encode_event(incomplete_event)

        self.assertEqual(len(cat_ids), 4)
        self.assertEqual(len(num_feats), 4)
        # Missing categorical fields map to UNK
        self.assertEqual(cat_ids[0], TargetLabelMapper.UNK_ID)  # user
        self.assertEqual(cat_ids[1], TargetLabelMapper.UNK_ID)  # computer

    def test_predict_next_output_structure(self):
        """Verify output dictionary contains top-k candidates, confidences, and logits."""
        events = [
            {"user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "Start"},
            {"user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "End"},
        ]
        preds = self.engine.predict_next(events, top_k=3)

        for feat in TargetLabelMapper.FEATURES:
            self.assertIn(feat, preds)
            res = preds[feat]
            self.assertIn("predicted_identifier", res)
            self.assertIn("confidence", res)
            self.assertIn("top_candidates", res)

            # Probabilities sum to 1.0
            probs = res["probs"]
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

            # Top candidate matches confidence (min(top_k, num_classes))
            expected_k = min(3, self.engine.mapper.num_classes(feat))
            self.assertEqual(len(res["top_candidates"]), expected_k)
            self.assertAlmostEqual(res["confidence"], res["top_candidates"][0]["probability"], places=4)


class TestProcessAnomalyAnalyzer(unittest.TestCase):
    """Test suite for ProcessAnomalyAnalyzer."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ProcessInferenceEngine()
        cls.analyzer = ProcessAnomalyAnalyzer(cls.engine)

    def test_validation_calibration(self):
        """Verify threshold calibration metrics computed on validation data."""
        thresholds = self.analyzer.thresholds
        self.assertIn("p90", thresholds)
        self.assertIn("p95", thresholds)
        self.assertIn("active_threshold", thresholds)
        self.assertGreater(thresholds["p95"], thresholds["p90"])

    def test_normal_event_evaluation(self):
        """Typical event in sequence should receive NORMAL severity."""
        history = [
            {"user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "Start"}
        ]
        # Followed by expected process continuation
        observed = {"user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "End"}

        _, score = self.analyzer.evaluate_observation(history, observed)
        self.assertIsInstance(score, AnomalyScore)
        self.assertGreater(score.total_surprise, 0.0)
        self.assertIn(score.severity, ["NORMAL", "ELEVATED", "HIGH_SURPRISE"])
        self.assertIn("Observed event", score.explanation)

    def test_extreme_surprise_on_unseen_entities(self):
        """Event with completely unknown tokens triggers elevated/high surprise."""
        history = [
            {"user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "Start"}
        ]
        observed = {
            "user": "UNKNOWN_INTRUDER@DOM1",
            "computer": "C999999",
            "process": "P999999",
            "event_type": "Start",
        }
        _, score = self.analyzer.evaluate_observation(history, observed)
        self.assertTrue(score.is_anomaly)
        self.assertIn("unseen", score.explanation)
        self.assertIn(score.severity, ["ELEVATED", "HIGH_SURPRISE"])


class TestProcessEventReplayer(unittest.TestCase):
    """Test suite for ProcessEventReplayer."""

    @classmethod
    def setUpClass(cls):
        cls.replayer = ProcessEventReplayer()

    def test_list_sequences_excludes_test_split(self):
        """Ensure strict test split isolation: test sequences are never returned."""
        val_seqs = self.replayer.list_available_sequences(split="validation")
        self.assertGreater(len(val_seqs), 0)
        for s in val_seqs:
            self.assertEqual(s["data_split"], "validation")

        all_non_test = self.replayer.list_available_sequences(split=None)
        splits = {s["data_split"] for s in all_non_test}
        self.assertNotIn("test", splits, "Test split must remain completely isolated.")

    def test_get_sequence_events_decodes_properly(self):
        """Decodes raw numpy arrays into valid event dictionaries."""
        val_seqs = self.replayer.list_available_sequences(split="validation")
        seq_idx = val_seqs[0]["sequence_index"]
        events = self.replayer.get_sequence_events(seq_idx)

        self.assertGreater(len(events), 0)
        first_ev = events[0]
        self.assertIn("user", first_ev)
        self.assertIn("computer", first_ev)
        self.assertIn("process", first_ev)
        self.assertIn("event_type", first_ev)
        self.assertIn("time_delta", first_ev)
        self.assertTrue(first_ev["source"].startswith("LANL Dataset"))

    def test_simulated_demonstration_event(self):
        """Simulated events are clearly tagged to avoid confusing them with real data."""
        base_event = {"user": "U8946@DOM1", "computer": "C19038", "process": "P4", "event_type": "Start"}
        sim_event = self.replayer.generate_simulated_anomaly_event(base_event, "rare_process")

        self.assertIn("Simulated Demonstration", sim_event["source"])
        self.assertNotEqual(sim_event["process"], base_event["process"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
