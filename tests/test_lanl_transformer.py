"""Validation tests for SecurityEventTransformer next-event prediction."""

import sys
import unittest
import warnings
from pathlib import Path
import numpy as np
import torch

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.transformer.model import SecurityEventTransformer


class TestLANLTransformer(unittest.TestCase):
    """Validation test suite for SecurityEventTransformer forward pass and data integrity."""

    @classmethod
    def setUpClass(cls):
        dataset_path = PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "next_event" / "train.npz"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        raw_data = np.load(dataset_path)
        cls.raw_cat = raw_data["categorical_inputs"][:4]
        cls.raw_num = raw_data["numeric_inputs"][:4]
        cls.raw_mask = raw_data["attention_masks"][:4]

        # Convert to PyTorch tensors with correct data types
        cls.categorical_inputs = torch.tensor(cls.raw_cat, dtype=torch.long)
        cls.numeric_inputs = torch.tensor(cls.raw_num, dtype=torch.float32)
        cls.attention_masks = torch.tensor(cls.raw_mask, dtype=torch.bool)

        # Instantiate model in evaluation mode
        cls.model = SecurityEventTransformer()
        cls.model.eval()

    def test_embedding_index_ranges(self):
        """Verify that no embedding index in the batch exceeds its embedding vocabulary range."""
        vocab_limits = {
            "user": 36,
            "computer": 40,
            "process": 207,
            "event_type": 4,
        }

        user_indices = self.categorical_inputs[:, :, 0]
        computer_indices = self.categorical_inputs[:, :, 1]
        process_indices = self.categorical_inputs[:, :, 2]
        event_type_indices = self.categorical_inputs[:, :, 3]

        self.assertTrue((user_indices >= 0).all() and (user_indices < vocab_limits["user"]).all(),
                        f"User indices exceeded range [0, {vocab_limits['user']})")
        self.assertTrue((computer_indices >= 0).all() and (computer_indices < vocab_limits["computer"]).all(),
                        f"Computer indices exceeded range [0, {vocab_limits['computer']})")
        self.assertTrue((process_indices >= 0).all() and (process_indices < vocab_limits["process"]).all(),
                        f"Process indices exceeded range [0, {vocab_limits['process']})")
        self.assertTrue((event_type_indices >= 0).all() and (event_type_indices < vocab_limits["event_type"]).all(),
                        f"Event type indices exceeded range [0, {vocab_limits['event_type']})")

    def test_forward_pass_output_shapes(self):
        """Verify the four output head shapes: user [4, 34], computer [4, 38], process [4, 205], event_type [4, 2]."""
        with torch.no_grad():
            outputs = self.model(
                self.categorical_inputs,
                self.numeric_inputs,
                self.attention_masks,
            )

        expected_shapes = {
            "user": (4, 34),
            "computer": (4, 38),
            "process": (4, 205),
            "event_type": (4, 2),
        }

        for head_name, expected_shape in expected_shapes.items():
            self.assertIn(head_name, outputs, f"Missing output head: {head_name}")
            actual_shape = tuple(outputs[head_name].shape)
            self.assertEqual(
                actual_shape,
                expected_shape,
                f"Head '{head_name}' expected shape {expected_shape}, got {actual_shape}",
            )

    def test_outputs_are_finite(self):
        """Verify that every output contains finite values (no NaN or Inf)."""
        with torch.no_grad():
            outputs = self.model(
                self.categorical_inputs,
                self.numeric_inputs,
                self.attention_masks,
            )

        for head_name, tensor in outputs.items():
            self.assertTrue(
                torch.isfinite(tensor).all().item(),
                f"Head '{head_name}' contains non-finite values (NaN or Inf)",
            )

    def test_last_valid_event_selection(self):
        """Verify last-valid-event selection works properly for different sequence lengths."""
        lengths = self.attention_masks.sum(dim=1).long()
        last_indices = lengths - 1

        # Check that test samples have differing sequence lengths: [1, 2, 3, 4]
        unique_lengths = torch.unique(lengths)
        self.assertGreater(
            len(unique_lengths),
            1,
            f"Expected multiple distinct sequence lengths in batch, got lengths {lengths.tolist()}",
        )
        self.assertEqual(lengths.tolist(), [1, 2, 3, 4])
        self.assertEqual(last_indices.tolist(), [0, 1, 2, 3])

        # Verify that all last_indices point to valid (True) positions in attention_masks
        for b in range(len(last_indices)):
            idx = last_indices[b].item()
            self.assertTrue(
                bool(self.attention_masks[b, idx]),
                f"Batch {b}: last_index {idx} does not correspond to a valid event",
            )
            # If not at the end of the sequence, the token immediately after must be padding (False)
            if idx + 1 < self.attention_masks.size(1):
                self.assertFalse(
                    bool(self.attention_masks[b, idx + 1]),
                    f"Batch {b}: token at {idx + 1} immediately after last_index {idx} should be padding",
                )

    def test_runtime_warnings_and_errors(self):
        """Check for runtime warnings or errors during evaluation."""
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            with torch.no_grad():
                outputs = self.model(
                    self.categorical_inputs,
                    self.numeric_inputs,
                    self.attention_masks,
                )
                self.assertEqual(len(outputs), 4)

            # Ensure no abnormal runtime warnings (like numerical instability or deprecation)
            for w in recorded_warnings:
                self.assertNotIn(
                    w.category,
                    [RuntimeWarning, DeprecationWarning, FutureWarning],
                    f"Unexpected warning category: {w.category.__name__}: {w.message}",
                )


if __name__ == "__main__":
    unittest.main()
