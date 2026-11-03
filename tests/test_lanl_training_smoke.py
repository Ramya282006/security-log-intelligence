"""Smoke test for LANL Transformer training pipeline integration."""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.preprocessing.label_mapping import TargetLabelMapper
from intelligence.transformer.model import SecurityEventTransformer
from intelligence.transformer.train import (
    LANLNextEventDataset,
    compute_safe_multitask_loss,
    load_lanl_split,
    save_checkpoint,
    load_checkpoint,
)


class TestLANLTrainingPipelineSmoke(unittest.TestCase):
    """Smoke tests validating the complete forward-backward step, safe loss, and checkpointing."""

    @classmethod
    def setUpClass(cls):
        cls.mapper = TargetLabelMapper()
        cls.train_dataset = load_lanl_split("train", mapper=cls.mapper)

    def test_pipeline_smoke_single_step(self):
        """Execute one complete forward pass, safe loss, backward pass, and optimizer step."""
        # 1. Take a small batch of 4 samples from the real training dataset
        cat_batch = self.train_dataset.categorical_inputs[:4]
        num_batch = self.train_dataset.numeric_inputs[:4]
        mask_batch = self.train_dataset.attention_masks[:4]
        target_batch = self.train_dataset.targets[:4]

        # 2. Instantiate model and optimizer
        model = SecurityEventTransformer()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion_sum = nn.CrossEntropyLoss(
            ignore_index=TargetLabelMapper.IGNORE_INDEX,
            reduction="sum",
        )

        model.train()
        optimizer.zero_grad()

        # 3. Forward pass
        outputs = model(cat_batch, num_batch, mask_batch)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(set(outputs.keys()), set(TargetLabelMapper.FEATURES))

        for feat in TargetLabelMapper.FEATURES:
            self.assertEqual(outputs[feat].shape[0], 4)
            self.assertEqual(outputs[feat].shape[1], self.mapper.num_classes(feat))
            self.assertTrue(torch.isfinite(outputs[feat]).all().item())

        # 4. Safe loss calculation
        loss, head_losses = compute_safe_multitask_loss(outputs, target_batch, criterion_sum)

        self.assertTrue(torch.isfinite(loss).item(), "Total loss must be finite")
        self.assertGreater(loss.item(), 0.0, "Total loss should be positive for un-trained model")

        for feat in TargetLabelMapper.FEATURES:
            self.assertIn(feat, head_losses)
            self.assertTrue(np.isfinite(head_losses[feat]))
            self.assertGreater(head_losses[feat], 0.0)

        # 5. Backward pass
        loss.backward()

        # 6. Verify finite gradients on all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for parameter: {name}")
                self.assertTrue(
                    torch.isfinite(param.grad).all().item(),
                    f"Non-finite gradient in parameter: {name}",
                )

        # 7. Optimizer step
        optimizer.step()

    def test_safe_loss_handles_all_unk_batch(self):
        """Confirm that an all-UNK batch produces a differentiable zero loss without NaNs."""
        cat_batch = self.train_dataset.categorical_inputs[:4]
        num_batch = self.train_dataset.numeric_inputs[:4]
        mask_batch = self.train_dataset.attention_masks[:4]

        # Synthetic target batch where every single target is UNK (-100)
        all_unk_targets = torch.full((4, 4), TargetLabelMapper.IGNORE_INDEX, dtype=torch.long)

        model = SecurityEventTransformer()
        criterion_sum = nn.CrossEntropyLoss(
            ignore_index=TargetLabelMapper.IGNORE_INDEX,
            reduction="sum",
        )

        model.train()
        model.zero_grad()
        outputs = model(cat_batch, num_batch, mask_batch)

        loss, head_losses = compute_safe_multitask_loss(outputs, all_unk_targets, criterion_sum)

        # Loss must be strictly zero (not NaN)
        self.assertFalse(torch.isnan(loss).item(), "Loss must not be NaN on all-UNK batch")
        self.assertEqual(loss.item(), 0.0, "Loss on all-UNK batch must be 0.0")

        for feat in TargetLabelMapper.FEATURES:
            self.assertEqual(head_losses[feat], 0.0)

        # Backward must complete without errors and produce zero gradients
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all().item())
                self.assertEqual(param.grad.abs().sum().item(), 0.0)

    def test_checkpoint_save_and_load(self):
        """Verify checkpoint serialization and deserialization."""
        model = SecurityEventTransformer()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_ckpt.pt"
            history = [{"epoch": 1, "train_loss": 5.2, "val_loss": 4.8}]

            save_checkpoint(
                checkpoint_path=ckpt_path,
                epoch=1,
                model=model,
                optimizer=optimizer,
                best_val_loss=4.8,
                train_history=history,
                config={"batch_size": 16},
            )

            self.assertTrue(ckpt_path.exists())

            # Load into new model
            new_model = SecurityEventTransformer()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
            loaded_state = load_checkpoint(ckpt_path, new_model, new_optimizer)

            self.assertEqual(loaded_state["epoch"], 1)
            self.assertEqual(loaded_state["best_val_loss"], 4.8)
            self.assertEqual(loaded_state["train_history"], history)
            self.assertEqual(loaded_state["config"]["batch_size"], 16)

            # Check that weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()

