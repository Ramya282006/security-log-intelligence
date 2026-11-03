"""LANL Next-Event Training Pipeline for SecurityEventTransformer.

This module loads the prepared LANL group-based dataset splits (train, validation, test),
maps categorical targets using TargetLabelMapper, trains the lightweight custom Transformer
using safe multi-task loss computation with NaN prevention, tracks per-head and joint metrics,
and saves checkpoints for resuming interrupted training.

Conventions:
    - Input sequence length: max 80 tokens
    - Inputs: categorical_inputs [B, 80, 4], numeric_inputs [B, 80, 4], attention_masks [B, 80]
    - Targets: user (34), computer (38), process (205), event_type (2)
    - UNK target IDs (1) are mapped to IGNORE_INDEX = -100

Run 2 additions:
    - Clipped inverse-frequency class weights applied to the process head only.
    - Validation loss aggregated as raw-sum / valid-count per head (not mean * batch_size).
    - Both weighted and unweighted validation losses recorded each epoch.
    - Unweighted validation loss used for best-checkpoint selection (comparable to Run 1).
    - Checkpoint paths parameterised by run_name to preserve Run 1 artifacts.
"""

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.preprocessing.label_mapping import TargetLabelMapper
from intelligence.transformer.model import SecurityEventTransformer

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "next_event"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# Column index of the process feature in the [user, computer, process, event_type] target matrix
PROCESS_COL: int = 2


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LANLNextEventDataset(Dataset):
    """PyTorch Dataset holding LANL next-event sequences and mapped target labels.
    
    Each item contains:
        - categorical_inputs: torch.LongTensor of shape [seq_len, 4]
        - numeric_inputs: torch.FloatTensor of shape [seq_len, 4]
        - attention_masks: torch.BoolTensor of shape [seq_len]
        - targets: torch.LongTensor of shape [4] (values in [-100, num_classes-1])
    """

    def __init__(
        self,
        categorical_inputs: Union[np.ndarray, torch.Tensor],
        numeric_inputs: Union[np.ndarray, torch.Tensor],
        attention_masks: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Initialize the dataset with converted tensors."""
        if not isinstance(categorical_inputs, torch.Tensor):
            self.categorical_inputs = torch.as_tensor(categorical_inputs, dtype=torch.long)
        else:
            self.categorical_inputs = categorical_inputs.to(dtype=torch.long)

        if not isinstance(numeric_inputs, torch.Tensor):
            self.numeric_inputs = torch.as_tensor(numeric_inputs, dtype=torch.float32)
        else:
            self.numeric_inputs = numeric_inputs.to(dtype=torch.float32)

        if not isinstance(attention_masks, torch.Tensor):
            self.attention_masks = torch.as_tensor(attention_masks, dtype=torch.bool)
        else:
            self.attention_masks = attention_masks.to(dtype=torch.bool)

        if not isinstance(targets, torch.Tensor):
            self.targets = torch.as_tensor(targets, dtype=torch.long)
        else:
            self.targets = targets.to(dtype=torch.long)

        self._num_samples = len(self.targets)
        assert len(self.categorical_inputs) == self._num_samples, "Mismatched categorical inputs length"
        assert len(self.numeric_inputs) == self._num_samples, "Mismatched numeric inputs length"
        assert len(self.attention_masks) == self._num_samples, "Mismatched attention masks length"

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.categorical_inputs[idx],
            self.numeric_inputs[idx],
            self.attention_masks[idx],
            self.targets[idx],
        )


def load_lanl_split(
    split_name: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    mapper: Optional[TargetLabelMapper] = None,
) -> LANLNextEventDataset:
    """Load a prepared LANL NPZ split and map its targets to class indices.
    
    Args:
        split_name: 'train', 'validation', or 'test'.
        data_dir: Directory containing train.npz, validation.npz, test.npz.
        mapper: TargetLabelMapper instance. If None, instantiates default mapper.
        
    Returns:
        LANLNextEventDataset instance with mapped targets.
    """
    if mapper is None:
        mapper = TargetLabelMapper()

    file_path = data_dir / f"{split_name}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset split not found at: {file_path}")

    raw_data = np.load(file_path)
    cat_inputs = raw_data["categorical_inputs"]
    num_inputs = raw_data["numeric_inputs"]
    attn_masks = raw_data["attention_masks"]
    raw_targets = raw_data["targets"]

    # Map target IDs (2..N+1 -> 0..N-1, 1 -> -100, 0 -> error)
    mapped_targets = mapper.transform_targets(raw_targets)

    return LANLNextEventDataset(
        categorical_inputs=cat_inputs,
        numeric_inputs=num_inputs,
        attention_masks=attn_masks,
        targets=mapped_targets,
    )


def compute_process_class_weights(
    train_dataset: LANLNextEventDataset,
    num_classes: int = 205,
    weight_cap: float = 10.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute clipped inverse-frequency class weights for the process prediction head.

    Uses the sklearn 'balanced' formula: w_c = N_known / (num_classes * count_c).

    Only training targets are used. Classes absent from training receive weight 1.0
    (neutral placeholder; they never appear as training targets so the weight is
    never applied to the loss numerator).

    UNK targets (-100) are excluded before computing counts.

    Args:
        train_dataset: Training LANLNextEventDataset with mapped targets already applied.
        num_classes: Number of process output classes (205).
        weight_cap: Maximum allowed weight before clipping. Applied after computing
            inverse-frequency weights. Default 10.0.
        device: Torch device for the returned tensor. If None, returns a CPU tensor.

    Returns:
        Float32 tensor of shape [num_classes] suitable for passing as the `weight`
        argument to nn.CrossEntropyLoss.
    """
    process_targets = train_dataset.targets[:, PROCESS_COL].numpy()
    known_targets = process_targets[process_targets >= 0]  # exclude UNK (-100)
    n_known = len(known_targets)

    if n_known == 0:
        raise ValueError("Training split contains zero known process targets — cannot compute class weights.")

    counts = np.zeros(num_classes, dtype=np.float64)
    for cls_idx in known_targets:
        counts[int(cls_idx)] += 1.0

    # Inverse-frequency: balanced formula. Absent classes get neutral weight 1.0.
    weights = np.ones(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = n_known / (num_classes * counts[c])

    # Clip to prevent extreme gradients from classes with very few samples
    weights = np.clip(weights, a_min=None, a_max=weight_cap)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return weight_tensor


def compute_safe_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    criterion_sum: nn.Module,
    features: Tuple[str, ...] = ("user", "computer", "process", "event_type"),
    per_head_criteria: Optional[Dict[str, nn.Module]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute independent multi-task loss with safe guarding against NaN on all-UNK batches.

    When a batch contains zero valid targets for a head (all targets == -100), computing
    reduction='mean' would yield 0/0 = NaN. This function guarantees:
        1. If n_valid > 0: loss is criterion_raw_sum / n_valid (per-sample normalisation).
        2. If n_valid == 0: returns logits.sum() * 0.0, a differentiable zero scalar
           that avoids graph breakage and produces zero gradient without NaNs, and returns
           head_losses[feat] = 0.0.

    The per-head criterion is selected from per_head_criteria if provided; otherwise
    criterion_sum is used. This enables process-head class weighting while leaving the
    other three heads unchanged.

    Args:
        outputs: Dict mapping feature name to logits tensor of shape [batch_size, num_classes].
        targets: Tensor of shape [batch_size, 4] with mapped class indices (or -100).
        criterion_sum: nn.CrossEntropyLoss with ignore_index=-100 and reduction='sum'.
            Used for any feature not present in per_head_criteria.
        features: Order of features corresponding to target columns.
        per_head_criteria: Optional dict mapping feature name to a head-specific
            nn.CrossEntropyLoss. If a feature is in this dict, that criterion is used
            instead of criterion_sum.

    Returns:
        Tuple of (total_loss_tensor, dict_of_float_head_losses).
            total_loss: Scalar tensor — sum of per-head normalised losses, suitable
                for backward(). Device and dtype match the model outputs.
            head_losses: Dict mapping each feature to its normalised loss scalar (float).
                If n_valid == 0 for a feature, its value is 0.0.
    """
    loss_list: List[torch.Tensor] = []
    head_losses: Dict[str, float] = {}

    for i, feat in enumerate(features):
        logits = outputs[feat]
        col_targets = targets[:, i]

        valid_mask = col_targets != TargetLabelMapper.IGNORE_INDEX
        n_valid = valid_mask.sum()

        # Select criterion: per-head override if provided, else shared criterion
        head_criterion = (
            per_head_criteria[feat]
            if (per_head_criteria and feat in per_head_criteria)
            else criterion_sum
        )

        if n_valid == 0:
            # Differentiable zero — preserves computation graph, produces zero gradient
            head_loss = logits.sum() * 0.0
            head_losses[feat] = 0.0
        else:
            raw_sum = head_criterion(logits, col_targets)  # scalar tensor (reduction='sum')
            head_loss = raw_sum / n_valid.to(dtype=logits.dtype)
            head_losses[feat] = head_loss.item()

        loss_list.append(head_loss)

    total_loss = torch.stack(loss_list).sum()
    return total_loss, head_losses


def compute_head_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature: str,
) -> Dict[str, float]:
    """Calculate known-target accuracy, effective accuracy, UNK count, and coverage for a head.
    
    Args:
        predictions: 1D array of predicted class indices [0..num_classes-1].
        targets: 1D array of true class indices [-100..num_classes-1].
        feature: Feature name string.
        
    Returns:
        Dictionary of performance metrics.
    """
    total = len(targets)
    known_mask = (targets != TargetLabelMapper.IGNORE_INDEX)
    n_known = int(known_mask.sum())
    n_unk = total - n_known

    coverage = (n_known / total) if total > 0 else 0.0
    unk_rate = (n_unk / total) if total > 0 else 0.0

    if n_known > 0:
        correct_known = int((predictions[known_mask] == targets[known_mask]).sum())
        known_accuracy = correct_known / n_known
    else:
        correct_known = 0
        known_accuracy = 0.0

    effective_accuracy = (correct_known / total) if total > 0 else 0.0

    return {
        f"{feature}_total": total,
        f"{feature}_known_count": n_known,
        f"{feature}_unk_count": n_unk,
        f"{feature}_coverage": coverage,
        f"{feature}_unk_rate": unk_rate,
        f"{feature}_known_accuracy": known_accuracy,
        f"{feature}_effective_accuracy": effective_accuracy,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_sum: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    per_head_criteria: Optional[Dict[str, nn.Module]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Train the model for one full epoch across all batches.

    Loss aggregation: for each head, accumulates raw criterion sums and valid-target
    counts across batches, then divides at the end. This avoids the bias introduced
    by averaging over batch-level normalised losses when batch sizes differ.

    Args:
        model: SecurityEventTransformer instance.
        dataloader: Training DataLoader.
        criterion_sum: CrossEntropyLoss with reduction='sum' and ignore_index=-100.
        optimizer: Optimizer instance.
        device: Computation device.
        per_head_criteria: Optional dict of per-head criterion overrides.

    Returns:
        Tuple of (avg_total_loss, avg_head_losses_dict).
        avg_total_loss = sum of per-head (raw_sum / valid_count).
        avg_head_losses_dict maps each feature to its per-valid-target loss.
    """
    model.train()

    # Accumulate raw loss sums and valid-target counts per head
    head_loss_sum: Dict[str, float] = {f: 0.0 for f in TargetLabelMapper.FEATURES}
    head_valid_cnt: Dict[str, int]  = {f: 0   for f in TargetLabelMapper.FEATURES}

    for cat_inputs, num_inputs, attn_masks, targets in dataloader:
        cat_inputs = cat_inputs.to(device)
        num_inputs = num_inputs.to(device)
        attn_masks = attn_masks.to(device)
        targets    = targets.to(device)

        optimizer.zero_grad()
        outputs = model(cat_inputs, num_inputs, attn_masks)
        loss, batch_head_losses = compute_safe_multitask_loss(
            outputs, targets, criterion_sum, per_head_criteria=per_head_criteria
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Encountered non-finite loss ({loss.item()}) at training step.")

        loss.backward()
        optimizer.step()

        for i, feat in enumerate(TargetLabelMapper.FEATURES):
            col_targets = targets[:, i]
            n_valid = int((col_targets != TargetLabelMapper.IGNORE_INDEX).sum().item())
            raw_sum = batch_head_losses[feat] * n_valid
            head_loss_sum[feat]  += raw_sum
            head_valid_cnt[feat] += n_valid

    # Final per-head average: raw_sum / valid_count
    avg_head_losses = {
        f: head_loss_sum[f] / max(head_valid_cnt[f], 1)
        for f in TargetLabelMapper.FEATURES
    }
    avg_total_loss = sum(avg_head_losses.values())
    return avg_total_loss, avg_head_losses


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_sum: nn.Module,
    device: torch.device,
    per_head_criteria: Optional[Dict[str, nn.Module]] = None,
    unweighted_criterion_sum: Optional[nn.Module] = None,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """Evaluate the model on a validation or test DataLoader.

    When per_head_criteria is provided (e.g. weighted process head) and
    unweighted_criterion_sum is also provided, this function computes both:
        - Weighted losses  (for logging the training signal)
        - Unweighted losses (for best-checkpoint selection, comparable to Run 1)

    Loss aggregation: raw criterion sums and valid-target counts are accumulated
    across batches per head. Division happens once at the end to avoid the
    batch-size-weighting bias from averaging normalised per-batch losses.

    Neither backward() nor optimizer.step() is called.

    Args:
        model: SecurityEventTransformer instance.
        dataloader: Evaluation DataLoader.
        criterion_sum: CrossEntropyLoss with reduction='sum' and ignore_index=-100.
        device: Computation device.
        per_head_criteria: Optional per-head criterion overrides (e.g. weighted process).
        unweighted_criterion_sum: If provided, an additional unweighted CE criterion is
            used to compute a second set of losses without per_head_criteria overrides.
            The unweighted total is stored in metrics['total_loss_unweighted'] and is
            suitable for checkpoint selection and comparison with Run 1.

    Returns:
        Tuple of (avg_total_loss_weighted, avg_head_losses_weighted, metrics_dict).
        metrics_dict additionally contains 'total_loss_unweighted' and
        'head_losses_unweighted'.
    """
    model.eval()

    # Weighted accumulators (primary)
    head_loss_sum_w:  Dict[str, float] = {f: 0.0 for f in TargetLabelMapper.FEATURES}
    head_valid_cnt_w: Dict[str, int]   = {f: 0   for f in TargetLabelMapper.FEATURES}

    # Unweighted accumulators (for checkpoint selection)
    compute_uw = unweighted_criterion_sum is not None
    head_loss_sum_uw:  Dict[str, float] = {f: 0.0 for f in TargetLabelMapper.FEATURES}
    head_valid_cnt_uw: Dict[str, int]   = {f: 0   for f in TargetLabelMapper.FEATURES}

    total_samples = 0
    all_preds:   Dict[str, List] = {f: [] for f in TargetLabelMapper.FEATURES}
    all_targets: Dict[str, List] = {f: [] for f in TargetLabelMapper.FEATURES}

    with torch.no_grad():
        for cat_inputs, num_inputs, attn_masks, targets in dataloader:
            cat_inputs = cat_inputs.to(device)
            num_inputs = num_inputs.to(device)
            attn_masks = attn_masks.to(device)
            targets    = targets.to(device)
            total_samples += len(targets)

            outputs = model(cat_inputs, num_inputs, attn_masks)

            # Weighted pass
            _, batch_head_losses_w = compute_safe_multitask_loss(
                outputs, targets, criterion_sum, per_head_criteria=per_head_criteria
            )
            for i, feat in enumerate(TargetLabelMapper.FEATURES):
                col_targets = targets[:, i]
                n_valid = int((col_targets != TargetLabelMapper.IGNORE_INDEX).sum().item())
                head_loss_sum_w[feat]  += batch_head_losses_w[feat] * n_valid
                head_valid_cnt_w[feat] += n_valid

            # Unweighted pass (no per_head_criteria → all heads use criterion_sum)
            if compute_uw:
                _, batch_head_losses_uw = compute_safe_multitask_loss(
                    outputs, targets, unweighted_criterion_sum
                )
                for i, feat in enumerate(TargetLabelMapper.FEATURES):
                    col_targets = targets[:, i]
                    n_valid = int((col_targets != TargetLabelMapper.IGNORE_INDEX).sum().item())
                    head_loss_sum_uw[feat]  += batch_head_losses_uw[feat] * n_valid
                    head_valid_cnt_uw[feat] += n_valid

            for i, feat in enumerate(TargetLabelMapper.FEATURES):
                preds = torch.argmax(outputs[feat], dim=1).cpu().numpy()
                all_preds[feat].extend(preds)
                all_targets[feat].extend(targets[:, i].cpu().numpy())

    # Final weighted averages
    avg_head_losses_w = {
        f: head_loss_sum_w[f] / max(head_valid_cnt_w[f], 1)
        for f in TargetLabelMapper.FEATURES
    }
    avg_total_loss_w = sum(avg_head_losses_w.values())

    # Final unweighted averages
    if compute_uw:
        avg_head_losses_uw = {
            f: head_loss_sum_uw[f] / max(head_valid_cnt_uw[f], 1)
            for f in TargetLabelMapper.FEATURES
        }
        avg_total_loss_uw = sum(avg_head_losses_uw.values())
    else:
        # No separate unweighted pass requested — both are identical
        avg_head_losses_uw = avg_head_losses_w
        avg_total_loss_uw  = avg_total_loss_w

    # Compute comprehensive per-head metrics
    metrics: Dict[str, Any] = {
        "total_loss":              avg_total_loss_w,
        "total_loss_unweighted":   avg_total_loss_uw,
        "head_losses":             avg_head_losses_w,
        "head_losses_unweighted":  avg_head_losses_uw,
        "total_samples":           total_samples,
    }

    pred_matrix_cols = []
    targ_matrix_cols = []

    for feat in TargetLabelMapper.FEATURES:
        feat_preds = np.array(all_preds[feat],   dtype=np.int64)
        feat_targs = np.array(all_targets[feat], dtype=np.int64)
        pred_matrix_cols.append(feat_preds)
        targ_matrix_cols.append(feat_targs)

        feat_metrics = compute_head_metrics(feat_preds, feat_targs, feat)
        metrics.update(feat_metrics)

    # Joint all-four accuracy
    pred_mat = np.column_stack(pred_matrix_cols)
    targ_mat = np.column_stack(targ_matrix_cols)

    all_known_mask = (targ_mat != TargetLabelMapper.IGNORE_INDEX).all(axis=1)
    n_all_known = int(all_known_mask.sum())
    metrics["joint_all_known_samples"] = n_all_known

    if n_all_known > 0:
        all_correct_mask = (pred_mat == targ_mat).all(axis=1) & all_known_mask
        metrics["joint_all_four_known_accuracy"] = int(all_correct_mask.sum()) / n_all_known
    else:
        metrics["joint_all_four_known_accuracy"] = 0.0

    return avg_total_loss_w, avg_head_losses_w, metrics


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    train_history: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training state checkpoint to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "train_history": train_history,
        "config": config or {},
    }
    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load training state checkpoint from disk."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    map_location = device or torch.device("cpu")
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def run_training_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 42,
    resume: bool = True,
    evaluate_test_at_end: bool = False,
    run_name: str = "lanl_run2",
) -> Dict[str, Any]:
    """Execute end-to-end LANL Transformer training and validation.

    Checkpoint paths are derived from run_name so that multiple runs never
    overwrite each other's artifacts:
        models/{run_name}_best.pt
        models/{run_name}_latest.pt
        models/{run_name}_training_history.json

    The process prediction head uses inverse-frequency class weights computed
    exclusively from training targets. The other three heads (user, computer,
    event_type) use the standard unweighted CrossEntropyLoss.

    Best-checkpoint selection uses the unweighted total validation loss so that
    results are directly comparable across runs regardless of process head weighting.

    Args:
        data_dir: Directory containing train.npz, validation.npz, test.npz.
        models_dir: Output directory for checkpoints.
        epochs: Target number of epochs.
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        seed: Random seed.
        resume: If True, resumes from latest checkpoint if available.
        evaluate_test_at_end: If True, evaluates test set after training completes.
        run_name: Prefix for all saved artifact filenames. Use distinct names for
            each run to prevent overwriting earlier checkpoints.

    Returns:
        Summary dictionary containing training history and best metrics.
    """
    set_seed(seed)
    device = torch.device("cpu")  # CPU-compatible PyTorch

    models_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path   = models_dir / f"{run_name}_best.pt"
    latest_checkpoint_path = models_dir / f"{run_name}_latest.pt"
    history_path           = models_dir / f"{run_name}_training_history.json"

    mapper = TargetLabelMapper()

    # Load pre-existing dataset splits (never create random splits)
    train_dataset = load_lanl_split("train",      data_dir=data_dir, mapper=mapper)
    val_dataset   = load_lanl_split("validation", data_dir=data_dir, mapper=mapper)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model     = SecurityEventTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Shared unweighted criterion — used for user, computer, event_type heads
    # and for the unweighted validation pass (checkpoint selection).
    criterion_sum = nn.CrossEntropyLoss(
        ignore_index=TargetLabelMapper.IGNORE_INDEX,
        reduction="sum",
    )

    # Process-head class weights — inverse-frequency from training targets only
    process_class_weights = compute_process_class_weights(
        train_dataset, num_classes=205, weight_cap=10.0, device=device
    )
    criterion_process_weighted = nn.CrossEntropyLoss(
        weight=process_class_weights,
        ignore_index=TargetLabelMapper.IGNORE_INDEX,
        reduction="sum",
    )

    # Only process head uses the weighted criterion
    per_head_criteria: Dict[str, nn.Module] = {"process": criterion_process_weighted}

    start_epoch  = 1
    best_val_loss = float("inf")   # tracked using UNWEIGHTED loss for comparability
    train_history: List[Dict[str, Any]] = []

    if resume and latest_checkpoint_path.exists():
        state = load_checkpoint(latest_checkpoint_path, model, optimizer, device)
        start_epoch   = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        train_history = state.get("train_history", [])
        print(f"Resumed from epoch {state['epoch']} (Best unweighted val loss: {best_val_loss:.4f})")

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "run_name": run_name,
        "process_weight_cap": 10.0,
    }

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_head_losses = train_epoch(
            model, train_loader, criterion_sum, optimizer, device,
            per_head_criteria=per_head_criteria,
        )
        val_loss_w, val_head_losses_w, val_metrics = evaluate(
            model, val_loader, criterion_sum, device,
            per_head_criteria=per_head_criteria,
            unweighted_criterion_sum=criterion_sum,
        )

        val_loss_uw = val_metrics["total_loss_unweighted"]

        if not np.isfinite(train_loss) or not np.isfinite(val_loss_w):
            print(
                f"ERROR: Non-finite loss at epoch {epoch} "
                f"(train: {train_loss}, val_w: {val_loss_w}). Stopping safely."
            )
            break

        # Checkpoint selection uses UNWEIGHTED validation loss for Run 1 comparability
        is_best = val_loss_uw < best_val_loss
        if is_best:
            best_val_loss = val_loss_uw
            save_checkpoint(
                best_checkpoint_path, epoch, model, optimizer,
                best_val_loss, train_history, config
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_head_losses": train_head_losses,
            "val_loss_weighted":   val_loss_w,
            "val_loss_unweighted": val_loss_uw,
            "val_head_losses_weighted":   val_head_losses_w,
            "val_head_losses_unweighted": val_metrics["head_losses_unweighted"],
            "val_metrics": val_metrics,
            "is_best": is_best,
        }
        train_history.append(epoch_record)
        save_checkpoint(
            latest_checkpoint_path, epoch, model, optimizer,
            best_val_loss, train_history, config
        )

        best_tag = " [BEST]" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss (W): {train_loss:.4f} | "
            f"Val Loss (UW): {val_loss_uw:.4f}{best_tag} | "
            f"Val Loss (W): {val_loss_w:.4f} | "
            f"Val Known Acc (U/C/P/E): "
            f"{val_metrics['user_known_accuracy']:.2f} / "
            f"{val_metrics['computer_known_accuracy']:.2f} / "
            f"{val_metrics['process_known_accuracy']:.2f} / "
            f"{val_metrics['event_type_known_accuracy']:.2f}"
        )

    # Save complete history JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2)

    results: Dict[str, Any] = {
        "best_val_loss_unweighted": best_val_loss,
        "epochs_completed": len(train_history),
        "best_checkpoint": str(best_checkpoint_path),
        "history_file": str(history_path),
    }

    if evaluate_test_at_end:
        test_dataset = load_lanl_split("test", data_dir=data_dir, mapper=mapper)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        load_checkpoint(best_checkpoint_path, model, device=device)
        test_loss_w, test_head_losses_w, test_metrics = evaluate(
            model, test_loader, criterion_sum, device,
            per_head_criteria=per_head_criteria,
            unweighted_criterion_sum=criterion_sum,
        )
        results["test_metrics"] = test_metrics

    return results


if __name__ == "__main__":
    results = run_training_pipeline(
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        seed=42,
        resume=False,
        evaluate_test_at_end=False,
        run_name="lanl_run2",
    )
