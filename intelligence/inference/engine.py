"""Inference engine for LANL Process Activity Transformer."""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.preprocessing.label_mapping import TargetLabelMapper
from intelligence.transformer.model import SecurityEventTransformer

DEFAULT_VOCAB_PATH = PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "lanl_vocabularies.json"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "lanl_run2_best.pt"
FALLBACK_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "lanl_transformer_best.pt"


class ProcessInferenceEngine:
    """Inference engine for predicting next process activity events using the trained Transformer.

    Encapsulates checkpoint loading, sequence encoding, token mapping, model forward
    pass, and extraction of top-k class predictions with softmax confidences.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        max_seq_length: int = 80,
    ) -> None:
        """Initialize the inference engine.

        Args:
            checkpoint_path: Path to the trained Transformer checkpoint (.pt).
            vocab_path: Path to lanl_vocabularies.json.
            device: Torch device (defaults to CPU).
            max_seq_length: Maximum sequence length (default 80).
        """
        self.max_seq_length = max_seq_length
        self.device = device or torch.device("cpu")
        self.vocab_path = Path(vocab_path or DEFAULT_VOCAB_PATH)
        self.mapper = TargetLabelMapper(vocab_path=self.vocab_path)

        # Determine checkpoint path
        ckpt_to_load = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        if not ckpt_to_load.exists():
            if FALLBACK_CHECKPOINT_PATH.exists():
                ckpt_to_load = FALLBACK_CHECKPOINT_PATH
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_to_load}")

        self.checkpoint_path = ckpt_to_load
        self.model, self.checkpoint_meta = self._load_model(self.checkpoint_path)

    def _load_model(self, path: Path) -> Tuple[SecurityEventTransformer, Dict[str, Any]]:
        """Load Transformer model from checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        model = SecurityEventTransformer(
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            feedforward_dim=128,
            dropout=0.0,
            max_seq_length=self.max_seq_length,
        ).to(self.device)

        model.load_state_dict(state["model_state_dict"])
        model.eval()

        meta = {
            "epoch": state.get("epoch"),
            "best_val_loss": state.get("best_val_loss"),
            "config": state.get("config", {}),
            "checkpoint_name": path.name,
        }
        return model, meta

    def encode_event(self, event: Dict[str, Any]) -> Tuple[List[int], List[float]]:
        """Convert a raw event dictionary into categorical and numeric feature vectors.

        Args:
            event: Event dictionary containing keys:
                - user: string identifier (e.g. 'U8946@DOM1')
                - computer: string identifier (e.g. 'C19038')
                - process: string identifier (e.g. 'P4')
                - event_type: string ('Start' or 'End')
                Optional numeric keys:
                - time_delta: float seconds (log-transformed internally if >= 0)
                - event_duration: float seconds
                - event_count: int/float
                - sequence_position: int/float

        Returns:
            Tuple of (categorical_ids, numeric_features)
        """
        # Categorical IDs (0: PAD, 1: UNK, 2..N+1: Known)
        cat_ids: List[int] = []
        for feat in TargetLabelMapper.FEATURES:
            val = event.get(feat, "")
            if isinstance(val, (int, np.integer)):
                orig_id = int(val)
            elif str(val) in self.mapper._str_to_id[feat]:
                orig_id = self.mapper._str_to_id[feat][str(val)]
            else:
                orig_id = TargetLabelMapper.UNK_ID
            cat_ids.append(orig_id)

        # Numeric features: [time_delta, event_duration, event_count, sequence_position]
        raw_delta = float(event.get("time_delta", 0.0))
        raw_dur = float(event.get("event_duration", 0.0))
        raw_cnt = float(event.get("event_count", 1.0))
        raw_pos = float(event.get("sequence_position", 1.0))

        # Log1p transforms for numerical stability (matching dataset preprocessing)
        num_feats = [
            float(np.log1p(max(raw_delta, 0.0))),
            float(np.log1p(max(raw_dur, 0.0))),
            float(np.log1p(max(raw_cnt, 0.0))),
            float(np.log1p(max(raw_pos, 0.0))),
        ]

        return cat_ids, num_feats

    def build_model_inputs(
        self,
        events: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct model tensors from a sequence of raw events.

        Truncates to the last max_seq_length events or pads shorter sequences with 0.

        Args:
            events: List of raw event dicts in chronological order.

        Returns:
            Tuple of:
                categorical_inputs: LongTensor of shape [1, 80, 4]
                numeric_inputs: FloatTensor of shape [1, 80, 4]
                attention_masks: BoolTensor of shape [1, 80] (True for valid events)
        """
        if not events:
            raise ValueError("Event sequence must contain at least one event.")

        # Keep last max_seq_length events
        window = events[-self.max_seq_length :]
        seq_len = len(window)

        cat_matrix = np.zeros((self.max_seq_length, 4), dtype=np.int64)
        num_matrix = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        attn_mask = np.zeros(self.max_seq_length, dtype=bool)

        for step_idx, ev in enumerate(window):
            # Update sequence position feature dynamically if not set
            ev_copy = dict(ev)
            if "sequence_position" not in ev_copy:
                ev_copy["sequence_position"] = step_idx + 1
            if "event_count" not in ev_copy:
                ev_copy["event_count"] = step_idx + 1

            cat_ids, num_feats = self.encode_event(ev_copy)
            cat_matrix[step_idx] = cat_ids
            num_matrix[step_idx] = num_feats
            attn_mask[step_idx] = True

        cat_tensor = torch.tensor(cat_matrix, dtype=torch.long, device=self.device).unsqueeze(0)
        num_tensor = torch.tensor(num_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(attn_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        return cat_tensor, num_tensor, mask_tensor

    def predict_next(
        self,
        events: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run Transformer inference on an event sequence and predict next-event features.

        Args:
            events: Chronological sequence of event dicts.
            top_k: Number of top candidate predictions to return per head.

        Returns:
            Dictionary containing prediction analysis for user, computer, process,
            and event_type heads, including:
                - predicted_id: String identifier of top-1 class
                - confidence: Softmax probability of top-1 class
                - top_candidates: List of dicts with (identifier, probability, class_index)
                - logits: Raw output logits tensor
                - probs: Full probability distribution tensor
        """
        cat_t, num_t, mask_t = self.build_model_inputs(events)

        with torch.no_grad():
            outputs = self.model(cat_t, num_t, mask_t)

        results: Dict[str, Any] = {}
        for feat in TargetLabelMapper.FEATURES:
            logits = outputs[feat].squeeze(0)  # [num_classes]
            probs = F.softmax(logits, dim=-1)

            top_k_count = min(top_k, probs.size(0))
            top_probs, top_indices = torch.topk(probs, k=top_k_count)

            top_candidates = []
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                str_id = self.mapper.to_string_identifier(idx, feat, from_class_index=True)
                top_candidates.append({
                    "identifier": str_id,
                    "probability": float(prob),
                    "class_index": int(idx),
                })

            results[feat] = {
                "predicted_identifier": top_candidates[0]["identifier"],
                "confidence": top_candidates[0]["probability"],
                "class_index": top_candidates[0]["class_index"],
                "top_candidates": top_candidates,
                "logits": logits.cpu(),
                "probs": probs.cpu(),
            }

        return results
