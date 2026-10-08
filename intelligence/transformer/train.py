"""Training Pipeline for SecurityEventTransformer.

This module loads labeled security event sequences, tokenizes them,
trains the lightweight custom Transformer encoder, saves the best model
checkpoint based on validation loss, and evaluates on an unseen test set.
"""

from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.preprocessing.tokenizer import SecurityEventTokenizer
from intelligence.transformer.model import SecurityEventTransformer

# Import sklearn metrics with fallback implementations for beginner resilience
try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
except ImportError:
    # Graceful fallback implementations if scikit-learn is not yet installed
    def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
        return sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(len(y_true), 1)

    def precision_score(y_true: List[int], y_pred: List[int], zero_division: int = 0) -> float:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        return tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)

    def recall_score(y_true: List[int], y_pred: List[int], zero_division: int = 0) -> float:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        return tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)

    def f1_score(y_true: List[int], y_pred: List[int], zero_division: int = 0) -> float:
        p = precision_score(y_true, y_pred, zero_division=zero_division)
        r = recall_score(y_true, y_pred, zero_division=zero_division)
        return (2 * p * r) / (p + r) if (p + r) > 0 else float(zero_division)

    def train_test_split(
        x: List[str],
        y: List[int],
        test_size: float = 0.30,
        random_state: int = 42,
        stratify: List[int] = None,
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        rng = random.Random(random_state)
        if stratify is not None:
            groups: Dict[int, List[Tuple[str, int]]] = {}
            for item_x, item_y in zip(x, y):
                groups.setdefault(item_y, []).append((item_x, item_y))
            x_tr, x_te, y_tr, y_te = [], [], [], []
            for _, items in groups.items():
                shuffled = list(items)
                rng.shuffle(shuffled)
                n_te = max(1, int(round(len(shuffled) * test_size)))
                for it_x, it_y in shuffled[n_te:]:
                    x_tr.append(it_x)
                    y_tr.append(it_y)
                for it_x, it_y in shuffled[:n_te]:
                    x_te.append(it_x)
                    y_te.append(it_y)
            return x_tr, x_te, y_tr, y_te
        else:
            combined = list(zip(x, y))
            rng.shuffle(combined)
            n_te = int(round(len(combined) * test_size))
            return (
                [it[0] for it in combined[n_te:]],
                [it[0] for it in combined[:n_te]],
                [it[1] for it in combined[n_te:]],
                [it[1] for it in combined[:n_te]],
            )


def set_seed(seed: int = 42) -> None:
    """Set random seeds across libraries for deterministic reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SecuritySequenceDataset(Dataset):
    """PyTorch Dataset holding tokenized event sequences and binary labels."""

    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor) -> None:
        """Initialize dataset.

        Args:
            sequences: 2D LongTensor of shape [num_samples, max_seq_length].
            labels: 1D LongTensor of shape [num_samples].
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Args:
        model: Transformer model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Computation device (CPU or CUDA).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate the model on validation or test data.

    Args:
        model: Transformer model.
        dataloader: Evaluation DataLoader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Tuple of (average_loss, accuracy, all_predictions, all_targets).
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc, all_preds, all_targets


def run_training_pipeline(
    csv_path: Path = PROJECT_ROOT / "data" / "sample" / "security_sequences.csv",
    models_dir: Path = PROJECT_ROOT / "models",
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> Dict[str, float]:
    """Execute the end-to-end training, validation, checkpointing, and testing pipeline.

    Args:
        csv_path: Path to dataset CSV.
        models_dir: Directory where the best model will be saved.
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        learning_rate: Adam optimizer learning rate.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dictionary containing test evaluation metrics.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("LIGHTWEIGHT SECURITY LOG INTELLIGENCE - TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset path: {csv_path}")

    # 1. Load CSV dataset
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset CSV must contain 'sequence' and 'label' columns.")

    raw_sequences = df["sequence"].astype(str).tolist()
    raw_labels = df["label"].astype(int).tolist()

    # 2. Fit Tokenizer
    tokenizer = SecurityEventTokenizer()
    tokenizer.fit(raw_sequences)
    print(f"Fitted Tokenizer vocabulary size: {tokenizer.vocab_size} tokens")

    # 3. Dataset split: 70% Train, 15% Validation, 15% Test
    # First split off 30% for combined val/test
    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        raw_sequences,
        raw_labels,
        test_size=0.30,
        random_state=seed,
        stratify=raw_labels,
    )

    # Split remaining 30% equally into 15% validation and 15% test
    val_seqs, test_seqs, val_labels, test_labels = train_test_split(
        temp_seqs,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )

    print(
        f"Split sizes -> Train: {len(train_seqs)} (70%), "
        f"Val: {len(val_seqs)} (15%), Test: {len(test_seqs)} (15%)"
    )

    # 4. Tokenize and pad batches
    X_train = tokenizer.encode_batch(train_seqs)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    X_val = tokenizer.encode_batch(val_seqs, max_length=X_train.shape[1])
    y_val = torch.tensor(val_labels, dtype=torch.long)

    X_test = tokenizer.encode_batch(test_seqs, max_length=X_train.shape[1])
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_loader = DataLoader(
        SecuritySequenceDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SecuritySequenceDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SecuritySequenceDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    # 5. Instantiate Model
    model = SecurityEventTransformer(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instantiated SecurityEventTransformer ({total_params:,} trainable parameters)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "security_event_transformer.pt"
    best_val_loss = float("inf")

    # 6. Training Loop
    print("\nStarting Training Loop:")
    print("-" * 60)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            saved_indicator = "-> Saved Best Model"
        else:
            saved_indicator = ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} {saved_indicator}"
        )

    # 7. Test Evaluation with Best Model
    print("-" * 60)
    print("Loading best checkpoint for test set evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)

    # 8. Concise Training Summary
    print("\n" + "=" * 60)
    print("CONCISE TRAINING & EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Dataset Size:    {len(df)} sequences")
    print(f"Split Proportions:     Train: {len(train_seqs)} | Val: {len(val_seqs)} | Test: {len(test_seqs)}")
    print(f"Vocabulary Size:       {tokenizer.vocab_size} tokens")
    print(f"Best Validation Loss:  {best_val_loss:.4f}")
    print(f"Saved Checkpoint:      {best_model_path}")
    print("-" * 60)
    print(f"Test Loss:             {test_loss:.4f}")
    print(f"Test Accuracy:         {test_acc:.4f} ({test_acc * 100:.1f}%)")
    print(f"Test Precision:        {precision:.4f}")
    print(f"Test Recall:           {recall:.4f}")
    print(f"Test F1 Score:         {f1:.4f}")
    print("=" * 60)

    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    run_training_pipeline()
