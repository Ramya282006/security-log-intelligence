"""Lightweight Transformer Model for Security Event Sequences.

This module implements a small, custom Transformer encoder in PyTorch designed
for binary classification (normal vs. suspicious) of security event sequences.
It is lightweight, efficient on CPU, and self-contained for academic/college
security intelligence projects.
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize sinusoidal positional encodings.

        Args:
            embedding_dim: Dimension of token embeddings (d_model).
            max_seq_length: Maximum allowed sequence length.
            dropout: Dropout probability applied to position-augmented embeddings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        # Register buffer to include in state_dict without being a parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape [batch_size, sequence_length, embedding_dim].

        Returns:
            Embeddings with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class SecurityEventTransformer(nn.Module):
    """Lightweight Transformer encoder for binary classification of event sequences.

    Architecture pipeline:
        1. Token Embedding: converts token indices to dense vectors.
        2. Positional Encoding: injects sequential order information.
        3. TransformerEncoder: applies multi-head self-attention layers.
        4. Sequence-level Pooling: aggregates token embeddings via mean pooling.
        5. Classification Head: outputs 2-class logits (0: normal, 1: suspicious).
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        max_seq_length: int = 128,
        padding_idx: Optional[int] = 0,
    ) -> None:
        """Initialize the SecurityEventTransformer model.

        Args:
            vocab_size: Size of vocabulary (unique security event types).
            embedding_dim: Dimensionality of token embeddings (must be divisible by num_heads).
            num_heads: Number of attention heads in each Transformer layer.
            num_layers: Number of Transformer encoder layers.
            feedforward_dim: Hidden dimension in the feedforward network sub-layer.
            dropout: Dropout probability.
            max_seq_length: Maximum sequence length supported.
            padding_idx: Index reserved for padding tokens (None if no padding token).
        """
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx

        # 1. Token Embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # 4 & 5. Sequence-level pooling + Binary classification head
        # Logits: class 0 = normal, class 1 = suspicious
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform forward pass on a batch of tokenized event sequences.

        Args:
            x: Tensor of tokenized event sequences of shape [batch_size, sequence_length].
            mask: Optional tensor indicating valid tokens. Can be boolean mask where
                True indicates non-padding tokens, or binary float mask [batch_size, sequence_length].

        Returns:
            Logits tensor of shape [batch_size, 2] suitable for nn.CrossEntropyLoss:
                - index 0: normal
                - index 1: suspicious
        """
        # 1. Token Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embeddings = self.token_embedding(x)

        # 2. Positional Encoding: [batch_size, seq_len, embedding_dim]
        encoded = self.positional_encoding(embeddings)

        # Build key padding mask for Transformer (True indicates tokens to ignore)
        src_key_padding_mask: Optional[torch.Tensor] = None
        if mask is not None:
            if mask.dtype == torch.bool:
                src_key_padding_mask = ~mask
            else:
                src_key_padding_mask = mask == 0
        elif self.padding_idx is not None:
            src_key_padding_mask = x == self.padding_idx

        # 3. Transformer Encoder: [batch_size, seq_len, embedding_dim]
        transformer_out = self.transformer_encoder(
            encoded,
            src_key_padding_mask=src_key_padding_mask,
        )

        # 4. Sequence-level Pooling: [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim]
        if src_key_padding_mask is not None:
            # Mean pooling over valid non-padding tokens
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
            sum_embeddings = torch.sum(transformer_out * valid_mask, dim=1)
            token_counts = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / token_counts
        else:
            pooled = torch.mean(transformer_out, dim=1)

        # 5. Binary Classification Head: [batch_size, embedding_dim] -> [batch_size, 2]
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    print("Module 'model' loaded successfully.")
    print("Example usage:")
    print("    from intelligence.transformer.model import SecurityEventTransformer")
    print("    model = SecurityEventTransformer(vocab_size=100, embedding_dim=64)")
    print("    # dummy_input = torch.randint(0, 100, (4, 16))")
    print("    # logits = model(dummy_input)  # shape: [4, 2]")
