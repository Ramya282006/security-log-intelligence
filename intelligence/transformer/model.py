
"""Lightweight Transformer for LANL next-event prediction."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        max_seq_length=128,
        dropout=0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_length).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(max_seq_length, embedding_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].size(1)]
        )

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SecurityEventTransformer(nn.Module):
    """Predict four categorical attributes of the next LANL event."""

    def __init__(
        self,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
        dropout=0.1,
        max_seq_length=80,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        # Vocabulary sizes include padding index 0.
        self.user_embedding = nn.Embedding(36, embedding_dim, padding_idx=0)
        self.computer_embedding = nn.Embedding(40, embedding_dim, padding_idx=0)
        self.process_embedding = nn.Embedding(207, embedding_dim, padding_idx=0)
        self.event_type_embedding = nn.Embedding(4, embedding_dim, padding_idx=0)

        # Convert four numeric features into embedding_dim dimensions.
        self.numeric_projection = nn.Linear(4, embedding_dim)

        self.positional_encoding = PositionalEncoding(
            embedding_dim,
            max_seq_length,
            dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

        # Four next-event prediction heads.
        self.user_head = nn.Linear(embedding_dim, 34)
        self.computer_head = nn.Linear(embedding_dim, 38)
        self.process_head = nn.Linear(embedding_dim, 205)
        self.event_type_head = nn.Linear(embedding_dim, 2)

    def forward(self, categorical_inputs, numeric_inputs, attention_masks):

        # Split the four categorical features.
        user = self.user_embedding(categorical_inputs[:, :, 0])
        computer = self.computer_embedding(categorical_inputs[:, :, 1])
        process = self.process_embedding(categorical_inputs[:, :, 2])
        event_type = self.event_type_embedding(categorical_inputs[:, :, 3])

        # Combine categorical embeddings.
        categorical = user + computer + process + event_type

        # Project numeric features.
        numeric = self.numeric_projection(numeric_inputs)

        # Combine categorical and numerical information.
        x = categorical + numeric

        # Add positional information.
        x = self.positional_encoding(x)

        # True in attention_masks means valid event.
        padding_mask = ~attention_masks.bool()

        # Transformer encoder.
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Find the last valid event in each sequence.
        lengths = attention_masks.sum(dim=1).long()
        last_indices = lengths - 1

        batch_indices = torch.arange(
            encoded.size(0),
            device=encoded.device,
        )

        last_event = encoded[batch_indices, last_indices]

        last_event = self.dropout(last_event)

        # Predict the four attributes of the next event.
        return {
            "user": self.user_head(last_event),
            "computer": self.computer_head(last_event),
            "process": self.process_head(last_event),
            "event_type": self.event_type_head(last_event),
        }

