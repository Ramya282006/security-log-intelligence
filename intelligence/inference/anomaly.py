"""Anomaly and surprise intelligence module for process activity events."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.preprocessing.label_mapping import TargetLabelMapper
from intelligence.inference.engine import ProcessInferenceEngine


@dataclass
class AnomalyScore:
    """Encapsulates surprise and anomaly scoring for an observed process event."""

    total_surprise: float
    is_anomaly: bool
    severity: str  # 'NORMAL', 'ELEVATED', 'HIGH_SURPRISE'
    threshold: float
    head_surprises: Dict[str, float]
    dominant_factor: str
    explanation: str
    investigation_recommendation: str
    confidence_warning: Optional[str] = None


class ProcessAnomalyAnalyzer:
    """Analyzes prediction surprise of observed events against the Transformer model."""

    def __init__(
        self,
        engine: ProcessInferenceEngine,
        threshold_percentile: float = 95.0,
        validation_npz_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the anomaly analyzer and calibrate thresholds against validation data.

        Args:
            engine: Initialized ProcessInferenceEngine.
            threshold_percentile: Percentile of validation surprise to flag as anomaly (default 95.0).
            validation_npz_path: Optional path to validation.npz for calibration.
        """
        self.engine = engine
        self.threshold_percentile = threshold_percentile
        self.mapper = engine.mapper

        val_path = Path(
            validation_npz_path
            or (PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "next_event" / "validation.npz")
        )
        self.thresholds = self._calibrate_from_validation(val_path)

    def _calibrate_from_validation(self, val_path: Path) -> Dict[str, float]:
        """Compute empirical surprise distribution strictly from validation data.

        Preserves test isolation: test.npz is never touched.
        """
        if not val_path.exists():
            # Conservative default thresholds if validation split is absent
            return {
                "p90": 8.0,
                "p95": 11.5,
                "p99": 16.0,
                "mean": 6.5,
                "active_threshold": 11.5,
            }

        val_data = np.load(val_path)
        cat_inputs = torch.tensor(val_data["categorical_inputs"], dtype=torch.long, device=self.engine.device)
        num_inputs = torch.tensor(val_data["numeric_inputs"], dtype=torch.float32, device=self.engine.device)
        attn_masks = torch.tensor(val_data["attention_masks"], dtype=torch.bool, device=self.engine.device)
        raw_targets = val_data["targets"]

        # Map targets to contiguous class indices (-100 for UNK)
        mapped_targets = self.mapper.transform_targets(raw_targets)

        surprises: List[float] = []
        batch_size = 32
        n_samples = len(mapped_targets)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                out = self.engine.model(
                    cat_inputs[i:end_i],
                    num_inputs[i:end_i],
                    attn_masks[i:end_i],
                )

                batch_surprises = torch.zeros(end_i - i, device=self.engine.device)
                for head_idx, feat in enumerate(TargetLabelMapper.FEATURES):
                    logits = out[feat]
                    log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
                    targs = torch.tensor(
                        mapped_targets[i:end_i, head_idx],
                        dtype=torch.long,
                        device=self.engine.device,
                    )

                    valid_mask = targs >= 0
                    if valid_mask.any():
                        # Negative log-likelihood: -log P(y_true)
                        nll = -log_probs[valid_mask, targs[valid_mask]]
                        batch_surprises[valid_mask] += nll

                    # UNK targets receive max surprise penalty
                    unk_mask = targs == TargetLabelMapper.IGNORE_INDEX
                    if unk_mask.any():
                        batch_surprises[unk_mask] += 10.0  # ~ -log(4.5e-5)

                surprises.extend(batch_surprises.cpu().tolist())

        surprises_arr = np.array(surprises, dtype=np.float64)
        p90 = float(np.percentile(surprises_arr, 90.0))
        p95 = float(np.percentile(surprises_arr, 95.0))
        p99 = float(np.percentile(surprises_arr, 99.0))
        mean_val = float(np.mean(surprises_arr))

        active = float(np.percentile(surprises_arr, self.threshold_percentile))

        return {
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "mean": mean_val,
            "active_threshold": active,
            "n_calibration_samples": len(surprises_arr),
        }

    def evaluate_observation(
        self,
        history_events: List[Dict[str, Any]],
        observed_event: Dict[str, Any],
        custom_threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], AnomalyScore]:
        """Compute the model's next-event predictions and evaluate surprise of the observed event.

        Args:
            history_events: Prior sequence of event dicts.
            observed_event: The actual event that occurred next.
            custom_threshold: Optional threshold override.

        Returns:
            Tuple of (prediction_results, AnomalyScore).
        """
        predictions = self.engine.predict_next(history_events)
        threshold = custom_threshold if custom_threshold is not None else self.thresholds["active_threshold"]

        head_surprises: Dict[str, float] = {}
        total_surprise = 0.0
        unk_penalties: List[str] = []

        for feat in TargetLabelMapper.FEATURES:
            observed_val = observed_event.get(feat, "")
            if isinstance(observed_val, (int, np.integer)):
                orig_id = int(observed_val)
            elif str(observed_val) in self.mapper._str_to_id[feat]:
                orig_id = self.mapper._str_to_id[feat][str(observed_val)]
            else:
                orig_id = TargetLabelMapper.UNK_ID

            probs = predictions[feat]["probs"]  # [num_classes]

            if orig_id == TargetLabelMapper.UNK_ID:
                # Unseen vocabulary token
                surprise_h = 10.0  # high penalty
                unk_penalties.append(f"unseen {feat} '{observed_val}'")
            else:
                class_idx = self.mapper.to_class_index(orig_id, feat)
                prob_observed = float(probs[class_idx].item())
                # Epsilon guard against log(0)
                prob_safe = max(prob_observed, 1e-7)
                surprise_h = float(-np.log(prob_safe))

            head_surprises[feat] = surprise_h
            total_surprise += surprise_h

        # Dominant factor: head with highest surprise
        dominant_factor = max(head_surprises, key=head_surprises.get)
        max_surprise_val = head_surprises[dominant_factor]

        # Severity categorization
        if total_surprise >= self.thresholds.get("p99", threshold * 1.3):
            severity = "HIGH_SURPRISE"
            is_anomaly = True
        elif total_surprise >= threshold:
            severity = "ELEVATED"
            is_anomaly = True
        else:
            severity = "NORMAL"
            is_anomaly = False

        # Build evidence-based explanation
        observed_dominant_str = observed_event.get(dominant_factor, "Unknown")
        predicted_dominant_str = predictions[dominant_factor]["predicted_identifier"]
        dominant_conf = predictions[dominant_factor]["confidence"] * 100.0

        explanation_parts = [
            f"Observed event scored total surprise of {total_surprise:.2f} (threshold: {threshold:.2f}).",
            f"Primary deviation occurred in '{dominant_factor}' (surprise: {max_surprise_val:.2f}): "
            f"observed '{observed_dominant_str}' while model predicted '{predicted_dominant_str}' ({dominant_conf:.1f}% confidence).",
        ]
        if unk_penalties:
            explanation_parts.append(f"Includes out-of-vocabulary penalty for: {', '.join(unk_penalties)}.")

        explanation = " ".join(explanation_parts)

        # Investigation recommendation
        if severity == "HIGH_SURPRISE":
            recommendation = (
                f"Flagged for triage: Investigate execution of process '{observed_event.get('process')}' "
                f"by user '{observed_event.get('user')}' on host '{observed_event.get('computer')}'. "
                f"Verify parent process tree, command-line arguments, and whether this user normally executes this process."
            )
        elif severity == "ELEVATED":
            recommendation = (
                f"Low-priority review: Event represents an uncommon transition for '{dominant_factor}'. "
                f"Monitor subsequent process events in this sequence for clustering of unusual actions."
            )
        else:
            recommendation = "Routine process activity matching baseline temporal distribution. No action required."

        confidence_warning = (
            "Note: Model surprise measures divergence from baseline training sequences, "
            "not verified maliciousness. Novel benign administrative activity can also trigger surprise."
        )

        score = AnomalyScore(
            total_surprise=total_surprise,
            is_anomaly=is_anomaly,
            severity=severity,
            threshold=threshold,
            head_surprises=head_surprises,
            dominant_factor=dominant_factor,
            explanation=explanation,
            investigation_recommendation=recommendation,
            confidence_warning=confidence_warning,
        )

        return predictions, score
