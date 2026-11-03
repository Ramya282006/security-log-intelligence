"""Automated tests for TargetLabelMapper utility."""

import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.preprocessing.label_mapping import TargetLabelMapper
from intelligence.transformer.model import SecurityEventTransformer


class TestTargetLabelMapper(unittest.TestCase):
    """Test suite verifying TargetLabelMapper correctness, bijectivity, and edge case safety."""

    @classmethod
    def setUpClass(cls):
        cls.mapper = TargetLabelMapper()
        cls.vocab_path = PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "lanl_vocabularies.json"
        with open(cls.vocab_path, "r", encoding="utf-8") as f:
            cls.raw_vocabs = json.load(f)

    def test_vocabulary_mappings_and_dimensions(self):
        """Verify that all four vocabulary mappings match expected head dimensions."""
        expected_dims = {
            "user": 34,
            "computer": 38,
            "process": 205,
            "event_type": 2,
        }
        self.assertEqual(self.mapper.head_dimensions, expected_dims)

        for feat, expected_dim in expected_dims.items():
            self.assertEqual(self.mapper.num_classes(feat), expected_dim)
            raw_dict = self.raw_vocabs[feat]
            self.assertEqual(len(raw_dict), expected_dim)

    def test_dimensions_match_model_architecture(self):
        """Verify mapper dimensions exactly equal model output head dimensions."""
        model = SecurityEventTransformer()
        self.assertEqual(self.mapper.num_classes("user"), model.user_head.out_features)
        self.assertEqual(self.mapper.num_classes("computer"), model.computer_head.out_features)
        self.assertEqual(self.mapper.num_classes("process"), model.process_head.out_features)
        self.assertEqual(self.mapper.num_classes("event_type"), model.event_type_head.out_features)

    def test_forward_and_reverse_mapping_single_elements(self):
        """Verify bijectivity of single-element conversions for all vocabulary entities."""
        for feat in self.mapper.features:
            raw_dict = self.raw_vocabs[feat]
            num_classes = self.mapper.num_classes(feat)
            observed_classes = set()

            for name, orig_id in raw_dict.items():
                # Forward: ID -> class index
                cls_idx = self.mapper.to_class_index(orig_id, feat)
                self.assertGreaterEqual(cls_idx, 0)
                self.assertLess(cls_idx, num_classes)
                observed_classes.add(cls_idx)

                # Reverse: class index -> ID
                reconstructed_id = self.mapper.to_original_id(cls_idx, feat)
                self.assertEqual(reconstructed_id, orig_id)

                # String recovery from class index and original ID
                self.assertEqual(self.mapper.to_string_identifier(cls_idx, feat, from_class_index=True), name)
                self.assertEqual(self.mapper.to_string_identifier(orig_id, feat, from_class_index=False), name)

            # Ensure all class indices 0..N-1 are covered with no gaps
            self.assertEqual(observed_classes, set(range(num_classes)))

    def test_forward_and_reverse_mapping_vectorized(self):
        """Verify vectorized conversions preserve types, devices, and exact values."""
        for feat in self.mapper.features:
            raw_ids = list(self.raw_vocabs[feat].values())

            # Python list
            cls_list = self.mapper.to_class_indices(raw_ids, feat)
            self.assertIsInstance(cls_list, list)
            rec_list = self.mapper.to_original_ids(cls_list, feat)
            self.assertEqual(rec_list, raw_ids)

            # NumPy array
            np_ids = np.array(raw_ids, dtype=np.int32)
            cls_np = self.mapper.to_class_indices(np_ids, feat)
            self.assertIsInstance(cls_np, np.ndarray)
            rec_np = self.mapper.to_original_ids(cls_np, feat)
            self.assertTrue(np.array_equal(rec_np, np_ids))

            # PyTorch Tensor
            torch_ids = torch.tensor(raw_ids, dtype=torch.long)
            cls_torch = self.mapper.to_class_indices(torch_ids, feat)
            self.assertIsInstance(cls_torch, torch.Tensor)
            rec_torch = self.mapper.to_original_ids(cls_torch, feat)
            self.assertTrue(torch.equal(rec_torch, torch_ids))

    def test_unk_conversion_to_minus_100(self):
        """Verify that UNK (ID 1) maps to -100 and can be recovered with allow_unknown=True."""
        for feat in self.mapper.features:
            # Scalar mapping
            cls_idx = self.mapper.to_class_index(1, feat)
            self.assertEqual(cls_idx, -100)

            # String identifiers
            self.assertEqual(self.mapper.to_string_identifier(1, feat), "<UNK>")
            self.assertEqual(self.mapper.to_string_identifier(-100, feat, from_class_index=True), "<UNK>")

            # Reverse with allow_unknown=True
            self.assertEqual(self.mapper.to_original_id(-100, feat, allow_unknown=True), 1)

            # Reverse with allow_unknown=False should raise ValueError
            with self.assertRaises(ValueError):
                self.mapper.to_original_id(-100, feat, allow_unknown=False)

            # Vectorized mapping
            arr_with_unk = np.array([1, 2, 1], dtype=np.int64)
            mapped = self.mapper.to_class_indices(arr_with_unk, feat)
            self.assertEqual(mapped[0], -100)
            self.assertEqual(mapped[2], -100)

            reconstructed = self.mapper.to_original_ids(mapped, feat, allow_unknown=True)
            self.assertTrue(np.array_equal(reconstructed, arr_with_unk))

    def test_padding_id_rejection(self):
        """Verify that padding ID (0) is strictly rejected as a target label."""
        for feat in self.mapper.features:
            with self.assertRaises(ValueError) as ctx:
                self.mapper.to_class_index(0, feat)
            self.assertIn("Padding ID 0", str(ctx.exception))

            with self.assertRaises(ValueError):
                self.mapper.to_class_indices([0, 2], feat)

            # Target matrix with 0 must be rejected
            with self.assertRaises(ValueError):
                bad_matrix = np.array([[2, 2, 2, 2], [0, 2, 2, 2]])
                self.mapper.transform_targets(bad_matrix)

            # Original ID 0 string is allowed for decoding input sequences
            self.assertEqual(self.mapper.to_string_identifier(0, feat, from_class_index=False), "<PAD>")

            # Model prediction index 0 is a valid class (the first class), NOT padding
            first_class_str = self.mapper.to_string_identifier(0, feat, from_class_index=True)
            self.assertNotEqual(first_class_str, "<PAD>")

    def test_invalid_id_handling(self):
        """Verify that out-of-range or unknown IDs raise clear ValueErrors."""
        for feat in self.mapper.features:
            # Negative ID
            with self.assertRaises(ValueError):
                self.mapper.to_class_index(-5, feat)

            # Non-existent high ID
            with self.assertRaises(ValueError):
                self.mapper.to_class_index(99999, feat)

            with self.assertRaises(ValueError):
                self.mapper.to_class_indices([2, 99999], feat)

    def test_invalid_prediction_index_handling(self):
        """Verify that out-of-range class prediction indices raise clear ValueErrors."""
        for feat in self.mapper.features:
            num_cls = self.mapper.num_classes(feat)

            # Negative index
            with self.assertRaises(ValueError):
                self.mapper.to_original_id(-1, feat)

            # Out of bounds index
            with self.assertRaises(ValueError):
                self.mapper.to_original_id(num_cls, feat)

            with self.assertRaises(ValueError):
                self.mapper.to_original_ids([0, num_cls], feat)

    def test_duplicate_mapping_detection(self):
        """Verify that vocabularies with duplicate IDs or duplicate strings are rejected."""
        # Duplicate integer ID in custom vocabulary
        bad_vocab_dup_id = {
            "user": {"U1": 2, "U2": 2},  # duplicate ID 2
            "computer": {"C1": 2},
            "process": {"P1": 2},
            "event_type": {"E1": 2},
        }
        with self.assertRaises(ValueError) as ctx:
            TargetLabelMapper(vocabularies=bad_vocab_dup_id)
        self.assertIn("Duplicate integer IDs", str(ctx.exception))

        # Reserved ID 0 in vocabulary
        bad_vocab_pad = {
            "user": {"U1": 0},
            "computer": {"C1": 2},
            "process": {"P1": 2},
            "event_type": {"E1": 2},
        }
        with self.assertRaises(ValueError) as ctx:
            TargetLabelMapper(vocabularies=bad_vocab_pad)
        self.assertIn("Reserved padding ID (0)", str(ctx.exception))

        # Reserved ID 1 in vocabulary
        bad_vocab_unk = {
            "user": {"U1": 1},
            "computer": {"C1": 2},
            "process": {"P1": 2},
            "event_type": {"E1": 2},
        }
        with self.assertRaises(ValueError) as ctx:
            TargetLabelMapper(vocabularies=bad_vocab_unk)
        self.assertIn("Reserved UNK ID (1)", str(ctx.exception))

    def test_edge_case_batch_only_unk_values(self):
        """Verify that a batch containing only UNK values (1) is never treated as valid class labels."""
        # Create an all-UNK batch of shape (16, 4)
        all_unk_batch = np.full((16, 4), 1, dtype=np.int64)

        mapped = self.mapper.transform_targets(all_unk_batch)

        # 1. All elements must strictly equal -100
        self.assertTrue((mapped == -100).all())

        # 2. No element should fall in [0, num_classes-1] for any feature
        for i, feat in enumerate(self.mapper.features):
            col = mapped[:, i]
            num_cls = self.mapper.num_classes(feat)
            valid_mask = (col >= 0) & (col < num_cls)
            self.assertEqual(valid_mask.sum(), 0, f"Found accidental valid class labels in all-UNK {feat} batch!")

            # 3. Model prediction recovery with allow_unknown=False MUST fail
            with self.assertRaises(ValueError) as ctx:
                self.mapper.to_original_ids(col, feat, allow_unknown=False)
            self.assertIn("not a valid prediction", str(ctx.exception))

            # 4. Target recovery with allow_unknown=True succeeds and reproduces UNK (1)
            recovered_col = self.mapper.to_original_ids(col, feat, allow_unknown=True)
            self.assertTrue((recovered_col == 1).all())

        # Test with PyTorch Tensor as well
        torch_all_unk = torch.full((8, 4), 1, dtype=torch.long)
        torch_mapped = self.mapper.transform_targets(torch_all_unk)
        self.assertTrue((torch_mapped == -100).all())
        torch_rec = self.mapper.inverse_transform_targets(torch_mapped, allow_unknown=True)
        self.assertTrue(torch.equal(torch_rec, torch_all_unk))

    def test_exact_reconstruction_across_all_dataset_splits(self):
        """Verify exact round-trip reconstruction for all targets across train, validation, and test."""
        splits = ["train", "validation", "test"]

        for split in splits:
            npz_path = PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "next_event" / f"{split}.npz"
            raw_targets = np.load(npz_path)["targets"]

            # NumPy round trip
            mapped_classes = self.mapper.transform_targets(raw_targets)
            reconstructed_np = self.mapper.inverse_transform_targets(mapped_classes, allow_unknown=True)
            self.assertTrue(
                np.array_equal(raw_targets, reconstructed_np),
                f"NumPy reconstruction failed for split: {split}",
            )

            # PyTorch round trip
            torch_targets = torch.tensor(raw_targets, dtype=torch.long)
            torch_mapped = self.mapper.transform_targets(torch_targets)
            torch_reconstructed = self.mapper.inverse_transform_targets(torch_mapped, allow_unknown=True)
            self.assertTrue(
                torch.equal(torch_targets, torch_reconstructed),
                f"PyTorch reconstruction failed for split: {split}",
            )


if __name__ == "__main__":
    unittest.main()
