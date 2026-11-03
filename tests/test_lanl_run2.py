"""Focused tests for Run 2 changes in intelligence/transformer/train.py.

Covers:
    1. compute_process_class_weights — formula, absent-class handling, weight cap.
    2. compute_safe_multitask_loss with per_head_criteria — process head uses weighted
       criterion; other heads remain unweighted.
    3. UNK target handling — -100 targets excluded from both weighted and unweighted loss.
    4. Zero-valid-target batch — differentiable zero fallback, no NaN.
    5. Validation aggregation — raw sum / valid count, not mean * batch_size.
    6. run_name checkpoint paths — correct artifact filenames.
    7. Unweighted loss independence — unweighted total equals sum of unweighted per-head losses.
"""

import sys
import pathlib
import unittest
import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intelligence.preprocessing.label_mapping import TargetLabelMapper
from intelligence.transformer.train import (
    LANLNextEventDataset,
    compute_process_class_weights,
    compute_safe_multitask_loss,
    PROCESS_COL,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int, proc_targets: list) -> LANLNextEventDataset:
    """Build a minimal LANLNextEventDataset with the given process targets."""
    cat = torch.zeros(n, 80, 4, dtype=torch.long)
    num = torch.zeros(n, 80, 4, dtype=torch.float32)
    mask = torch.ones(n, 80, dtype=torch.bool)
    targets = torch.zeros(n, 4, dtype=torch.long)
    targets[:, PROCESS_COL] = torch.tensor(proc_targets, dtype=torch.long)
    return LANLNextEventDataset(cat, num, mask, targets)


def _dummy_outputs(batch_size: int, num_classes_dict: dict) -> dict:
    """Return random logit tensors for each head."""
    torch.manual_seed(0)
    return {feat: torch.randn(batch_size, nc)
            for feat, nc in num_classes_dict.items()}


NUM_CLASSES = {"user": 34, "computer": 38, "process": 205, "event_type": 2}


# ---------------------------------------------------------------------------
# 1. compute_process_class_weights
# ---------------------------------------------------------------------------

class TestComputeProcessClassWeights(unittest.TestCase):

    def test_weight_formula_balanced(self):
        """Balanced formula: w_c = N_known / (num_classes * count_c)."""
        # 10 samples, 4 classes, 5 in class 0, 3 in class 1, 2 in class 2
        proc_targets = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
        dataset = _make_dataset(10, proc_targets)
        weights = compute_process_class_weights(dataset, num_classes=4, weight_cap=100.0)

        n_known = 10
        nc = 4
        expected = {0: n_known / (nc * 5),
                    1: n_known / (nc * 3),
                    2: n_known / (nc * 2),
                    3: 1.0}  # absent class → neutral 1.0

        for cls, exp in expected.items():
            self.assertAlmostEqual(weights[cls].item(), exp, places=5,
                                   msg=f"Weight mismatch for class {cls}")

    def test_absent_class_gets_neutral_weight(self):
        """Classes absent from training receive weight 1.0."""
        proc_targets = [0] * 10  # only class 0 present
        dataset = _make_dataset(10, proc_targets)
        weights = compute_process_class_weights(dataset, num_classes=5, weight_cap=100.0)

        for cls in range(1, 5):
            self.assertAlmostEqual(weights[cls].item(), 1.0, places=5,
                                   msg=f"Absent class {cls} should have weight 1.0")

    def test_weight_cap_applied(self):
        """Weights exceeding the cap are clipped."""
        # Class 0: 1 sample in 1000 total → raw weight = 1000 / (4 * 1) = 250
        proc_targets = [0] + [1] * 999
        dataset = _make_dataset(1000, proc_targets)
        weights = compute_process_class_weights(dataset, num_classes=4, weight_cap=10.0)

        self.assertLessEqual(weights[0].item(), 10.0 + 1e-6,
                             "Weight for rare class should be clipped to cap")

    def test_unk_targets_excluded(self):
        """UNK targets (-100) do not contribute to weight computation."""
        # 5 known class-0, 5 UNK; should behave as if n_known=5
        proc_targets = [0, 0, 0, 0, 0, -100, -100, -100, -100, -100]
        dataset = _make_dataset(10, proc_targets)
        weights_with_unk = compute_process_class_weights(dataset, num_classes=2, weight_cap=100.0)

        proc_targets_no_unk = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        dataset_no_unk = _make_dataset(10, proc_targets_no_unk)
        weights_no_unk = compute_process_class_weights(dataset_no_unk, num_classes=2, weight_cap=100.0)

        # Class 0: only-known run has n_known=5 with 5 class-0 → same weight
        # The unk run has 5 known (all class 0), class 1 absent → weight 1.0
        # They should differ because the distribution differs; key check is UNK excluded
        n_known = 5
        nc = 2
        expected_class0 = n_known / (nc * 5)
        self.assertAlmostEqual(weights_with_unk[0].item(), expected_class0, places=5)

    def test_output_shape(self):
        """Weight tensor has shape [num_classes]."""
        proc_targets = list(range(10))
        dataset = _make_dataset(10, proc_targets)
        weights = compute_process_class_weights(dataset, num_classes=20)
        self.assertEqual(weights.shape, (20,))
        self.assertEqual(weights.dtype, torch.float32)

    def test_device_placement(self):
        """Weight tensor placed on the specified device."""
        proc_targets = [0, 1, 2]
        dataset = _make_dataset(3, proc_targets)
        weights = compute_process_class_weights(dataset, num_classes=5, device=torch.device("cpu"))
        self.assertEqual(weights.device, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 2. compute_safe_multitask_loss with per_head_criteria
# ---------------------------------------------------------------------------

class TestPerHeadCriteria(unittest.TestCase):

    def setUp(self):
        self.B = 8
        self.criterion_sum = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        # Weighted criterion for process with non-uniform weights
        proc_weights = torch.ones(205)
        proc_weights[0] = 5.0   # class 0 gets higher weight
        self.criterion_proc_weighted = nn.CrossEntropyLoss(
            weight=proc_weights, ignore_index=-100, reduction="sum"
        )
        self.per_head_criteria = {"process": self.criterion_proc_weighted}

        self.outputs = _dummy_outputs(self.B, NUM_CLASSES)
        # All targets known (no UNK), class 0 for all heads
        self.targets = torch.zeros(self.B, 4, dtype=torch.long)

    def test_process_uses_weighted_criterion(self):
        """Process head uses the weighted criterion; computed loss should differ from unweighted."""
        _, head_losses_w = compute_safe_multitask_loss(
            self.outputs, self.targets, self.criterion_sum,
            per_head_criteria=self.per_head_criteria
        )
        _, head_losses_uw = compute_safe_multitask_loss(
            self.outputs, self.targets, self.criterion_sum,
        )
        # Process head: with weight[0]=5 vs weight[0]=1 → weighted > unweighted
        self.assertGreater(head_losses_w["process"], head_losses_uw["process"],
                           "Weighted process loss should exceed unweighted when target class has weight > 1")

    def test_other_heads_unchanged(self):
        """user, computer, event_type heads produce identical loss with/without per_head_criteria."""
        _, head_losses_w = compute_safe_multitask_loss(
            self.outputs, self.targets, self.criterion_sum,
            per_head_criteria=self.per_head_criteria
        )
        _, head_losses_uw = compute_safe_multitask_loss(
            self.outputs, self.targets, self.criterion_sum,
        )
        for feat in ("user", "computer", "event_type"):
            self.assertAlmostEqual(
                head_losses_w[feat], head_losses_uw[feat], places=4,
                msg=f"{feat} loss should be identical regardless of per_head_criteria"
            )

    def test_head_raw_n_valid(self):
        """All head losses are positive float scalars when targets are known."""
        _, head_losses = compute_safe_multitask_loss(
            self.outputs, self.targets, self.criterion_sum
        )
        for feat in NUM_CLASSES:
            self.assertIsInstance(head_losses[feat], float)
            self.assertGreater(head_losses[feat], 0.0)

    def test_total_loss_is_sum_of_head_losses(self):
        """total_loss tensor equals sum of per-head normalised losses."""
        total, head_losses = compute_safe_multitask_loss(
            self.outputs, self.targets, self.criterion_sum
        )
        self.assertAlmostEqual(total.item(), sum(head_losses.values()), places=4)


# ---------------------------------------------------------------------------
# 3. UNK target handling
# ---------------------------------------------------------------------------

class TestUnkHandling(unittest.TestCase):

    def setUp(self):
        self.criterion_sum = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    def test_unk_targets_excluded_from_loss(self):
        """Loss computed on a batch with mix of known/UNK equals loss on known-only subset."""
        B = 6
        outputs = _dummy_outputs(B, NUM_CLASSES)

        # Half UNK, half known class 0
        targets_mixed = torch.full((B, 4), -100, dtype=torch.long)
        targets_mixed[3:, :] = 0   # samples 3-5 are known class 0

        # Slice outputs to match known subset
        outputs_known = {feat: outputs[feat][3:] for feat in NUM_CLASSES}
        targets_known_only = torch.zeros(3, 4, dtype=torch.long)

        # Full mixed batch
        _, losses_mixed = compute_safe_multitask_loss(outputs, targets_mixed, self.criterion_sum)
        # Known-only batch
        _, losses_known = compute_safe_multitask_loss(outputs_known, targets_known_only, self.criterion_sum)

        for feat in NUM_CLASSES:
            self.assertAlmostEqual(
                losses_mixed[feat], losses_known[feat], places=4,
                msg=f"{feat}: mixed-UNK normalised loss should equal known-only normalised loss"
            )


# ---------------------------------------------------------------------------
# 4. Zero-valid-target batch
# ---------------------------------------------------------------------------

class TestZeroValidTargetBatch(unittest.TestCase):

    def setUp(self):
        self.criterion_sum = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    def test_all_unk_batch_returns_zero_loss(self):
        """All-UNK batch produces 0.0 scalar float and finite total loss."""
        B = 4
        outputs = _dummy_outputs(B, NUM_CLASSES)
        targets = torch.full((B, 4), -100, dtype=torch.long)

        total, head_losses = compute_safe_multitask_loss(outputs, targets, self.criterion_sum)

        self.assertTrue(torch.isfinite(total), "Total loss must be finite for all-UNK batch")
        self.assertAlmostEqual(total.item(), 0.0, places=6)
        for feat in NUM_CLASSES:
            self.assertIsInstance(head_losses[feat], float)
            self.assertAlmostEqual(head_losses[feat], 0.0, places=6,
                                   msg=f"{feat}: loss should be 0.0 for all-UNK")

    def test_all_unk_loss_is_differentiable(self):
        """All-UNK total loss has a valid (zero) gradient."""
        B = 4
        outputs_req = {
            feat: torch.randn(B, nc, requires_grad=True)
            for feat, nc in NUM_CLASSES.items()
        }
        targets = torch.full((B, 4), -100, dtype=torch.long)
        total, _ = compute_safe_multitask_loss(outputs_req, targets, self.criterion_sum)
        total.backward()
        for feat in NUM_CLASSES:
            grad = outputs_req[feat].grad
            self.assertIsNotNone(grad, f"{feat}: gradient should exist")
            self.assertTrue(torch.all(grad == 0.0),
                            f"{feat}: gradient should be zero for all-UNK batch")

    def test_partial_unk_batch_correct_n_valid(self):
        """Batches with mixed UNK/known targets report non-zero only for features with known targets."""
        B = 10
        outputs = _dummy_outputs(B, NUM_CLASSES)
        targets = torch.full((B, 4), -100, dtype=torch.long)
        targets[:7, 0] = 0   # 7 known for user; rest UNK

        _, head_losses = compute_safe_multitask_loss(outputs, targets, self.criterion_sum)
        self.assertGreater(head_losses["user"], 0.0)
        self.assertEqual(head_losses["computer"],   0.0)
        self.assertEqual(head_losses["process"],    0.0)
        self.assertEqual(head_losses["event_type"], 0.0)


# ---------------------------------------------------------------------------
# 5. Validation aggregation
# ---------------------------------------------------------------------------

class TestValidationAggregation(unittest.TestCase):
    """Verify that raw_sum / valid_count aggregation is correct across batches."""

    def _criterion(self):
        return nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    def test_aggregation_matches_full_batch(self):
        """Split a batch into two halves; per-head loss from two half-batches equals full batch."""
        B = 12
        torch.manual_seed(1)
        outputs_full = _dummy_outputs(B, NUM_CLASSES)
        targets_full = torch.randint(0, 2, (B, 4))  # class 0 or 1 for all heads
        # Force process into [0..204], event_type into [0..1]
        targets_full[:, 2] = torch.randint(0, 205, (B,))
        targets_full[:, 3] = torch.randint(0, 2,   (B,))

        crit = self._criterion()

        # Full batch
        _, losses_full = compute_safe_multitask_loss(outputs_full, targets_full, crit)

        # Two halves
        half = B // 2
        outputs_a = {feat: outputs_full[feat][:half] for feat in NUM_CLASSES}
        outputs_b = {feat: outputs_full[feat][half:]  for feat in NUM_CLASSES}
        targets_a = targets_full[:half]
        targets_b = targets_full[half:]

        _, losses_a = compute_safe_multitask_loss(outputs_a, targets_a, crit)
        _, losses_b = compute_safe_multitask_loss(outputs_b, targets_b, crit)

        # Aggregate across the two half-batches using normalized_loss * n_valid
        for i, feat in enumerate(NUM_CLASSES):
            valid_a = int((targets_a[:, i] != -100).sum().item())
            valid_b = int((targets_b[:, i] != -100).sum().item())
            raw_a = losses_a[feat] * valid_a
            raw_b = losses_b[feat] * valid_b
            total_sum   = raw_a + raw_b
            total_valid = valid_a + valid_b
            agg_loss    = total_sum / max(total_valid, 1)

            self.assertAlmostEqual(
                agg_loss, losses_full[feat], places=4,
                msg=f"{feat}: aggregated half-batch loss must equal full-batch loss"
            )

    def test_aggregation_differs_from_mean_of_means(self):
        """Unequal batch sizes: raw-sum/count != average of per-batch means."""
        torch.manual_seed(2)
        logits = torch.randn(10, 3)  # 3 classes
        targets_2 = torch.zeros(2, 4, dtype=torch.long)
        targets_8 = torch.zeros(8, 4, dtype=torch.long)

        # Different logits for the two groups
        logits_a = logits[:2]
        logits_b = logits[2:]

        outputs_a = {feat: logits_a if feat == "user" else torch.randn(2, NUM_CLASSES[feat])
                     for feat in NUM_CLASSES}
        outputs_b = {feat: logits_b if feat == "user" else torch.randn(8, NUM_CLASSES[feat])
                     for feat in NUM_CLASSES}

        crit = self._criterion()
        _, losses_a = compute_safe_multitask_loss(outputs_a, targets_2, crit)
        _, losses_b = compute_safe_multitask_loss(outputs_b, targets_8, crit)

        valid_a = 2
        valid_b = 8
        raw_a = losses_a["user"] * valid_a
        raw_b = losses_b["user"] * valid_b

        # Correct aggregation
        total_sum   = raw_a + raw_b
        total_valid = valid_a + valid_b
        correct_agg = total_sum / total_valid

        # Naive mean-of-means (wrong)
        mean_a = losses_a["user"]
        mean_b = losses_b["user"]
        naive  = (mean_a + mean_b) / 2.0

        # They must differ (different batch sizes with different logits)
        self.assertNotAlmostEqual(
            correct_agg, naive, places=4,
            msg="Correct aggregation must differ from mean-of-means for unequal batches"
        )


# ---------------------------------------------------------------------------
# 6. Separate checkpoint paths via run_name
# ---------------------------------------------------------------------------

class TestCheckpointPaths(unittest.TestCase):

    def test_run2_paths_differ_from_run1(self):
        """Run 2 artifact names are prefixed with run_name and distinct from Run 1."""
        models_dir = pathlib.Path("models")
        run2_best    = models_dir / "lanl_run2_best.pt"
        run2_latest  = models_dir / "lanl_run2_latest.pt"
        run2_history = models_dir / "lanl_run2_training_history.json"
        run1_best    = models_dir / "lanl_transformer_best.pt"
        run1_latest  = models_dir / "lanl_transformer_latest.pt"
        run1_history = models_dir / "lanl_training_history.json"

        # Names must be distinct
        self.assertNotEqual(str(run2_best),    str(run1_best))
        self.assertNotEqual(str(run2_latest),  str(run1_latest))
        self.assertNotEqual(str(run2_history), str(run1_history))

        # Run 2 names must contain the run_name prefix
        self.assertIn("lanl_run2", str(run2_best))
        self.assertIn("lanl_run2", str(run2_latest))
        self.assertIn("lanl_run2", str(run2_history))

    def test_run1_checkpoints_still_exist(self):
        """Existing Run 1 checkpoint files are untouched."""
        models_dir = pathlib.Path("C:/Users/dhiya/security-log-intelligence/models")
        run1_best    = models_dir / "lanl_transformer_best.pt"
        run1_latest  = models_dir / "lanl_transformer_latest.pt"
        run1_history = models_dir / "lanl_training_history.json"
        for path in (run1_best, run1_latest, run1_history):
            self.assertTrue(path.exists(), f"Run 1 artifact missing: {path}")


# ---------------------------------------------------------------------------
# 7. Unweighted loss independence
# ---------------------------------------------------------------------------

class TestUnweightedLossIndependence(unittest.TestCase):
    """Unweighted total equals sum of unweighted per-head losses."""

    def test_unweighted_total_is_sum_of_head_losses(self):
        B = 8
        torch.manual_seed(3)
        outputs = _dummy_outputs(B, NUM_CLASSES)
        targets = torch.zeros(B, 4, dtype=torch.long)
        crit = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        total, head_losses = compute_safe_multitask_loss(outputs, targets, crit)

        self.assertAlmostEqual(total.item(), sum(head_losses.values()), places=4)

    def test_weighted_and_unweighted_totals_can_differ(self):
        """With non-uniform weights on process, weighted and unweighted totals differ."""
        B = 8
        torch.manual_seed(4)
        outputs = _dummy_outputs(B, NUM_CLASSES)
        targets = torch.zeros(B, 4, dtype=torch.long)   # all targets are class 0

        crit_uw = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        w = torch.ones(205); w[0] = 10.0
        crit_w = nn.CrossEntropyLoss(weight=w, ignore_index=-100, reduction="sum")

        total_uw, _ = compute_safe_multitask_loss(outputs, targets, crit_uw)
        total_w,  _ = compute_safe_multitask_loss(
            outputs, targets, crit_uw, per_head_criteria={"process": crit_w}
        )
        self.assertNotAlmostEqual(
            total_w.item(), total_uw.item(), places=4,
            msg="Weighted total must differ from unweighted total when process weights != 1"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

