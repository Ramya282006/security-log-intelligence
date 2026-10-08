"""Security Event Tokenizer Module for Lightweight Security Log Intelligence.

This module converts space-delimited security event sequences into integer token IDs
and padded PyTorch LongTensors suitable for the SecurityEventTransformer model.
"""

from typing import Dict, List, Optional, Union
import torch


class SecurityEventTokenizer:
    """Custom tokenizer converting security event type sequences into integer token IDs.

    Special Tokens:
        <PAD> (ID: 0) - Padding token for variable length sequences in batches.
        <UNK> (ID: 1) - Unknown token for events not observed during vocabulary fitting.
    """

    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_ID: int = 0
    UNK_ID: int = 1

    def __init__(self, sequences: Optional[List[str]] = None) -> None:
        """Initialize the tokenizer with default special tokens and optional initial sequences.

        Args:
            sequences: Optional list of event sequence strings to fit immediately.
        """
        self.token_to_id: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        self.id_to_token: Dict[int, str] = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN,
        }

        if sequences:
            self.fit(sequences)

    @property
    def vocab_size(self) -> int:
        """Total size of vocabulary, including special tokens."""
        return len(self.token_to_id)

    def fit(self, sequences: List[str]) -> "SecurityEventTokenizer":
        """Build the vocabulary from a list of event sequences.

        Args:
            sequences: List of space-delimited event sequence strings
                (e.g., ["LOGIN_FAILURE LOGIN_SUCCESS", "FILE_ACCESS LOGOUT"]).

        Returns:
            Self instance for convenient method chaining.
        """
        for seq in sequences:
            if isinstance(seq, str):
                tokens = seq.strip().split()
            elif isinstance(seq, (list, tuple)):
                tokens = [str(t) for t in seq]
            else:
                continue

            for token in tokens:
                token = token.strip()
                if token and token not in self.token_to_id:
                    new_id = len(self.token_to_id)
                    self.token_to_id[token] = new_id
                    self.id_to_token[new_id] = token

        return self

    def encode(self, sequence: Union[str, List[str]]) -> List[int]:
        """Convert a single event sequence into a list of integer token IDs.

        Args:
            sequence: Space-delimited string of event types or list of event tokens.

        Returns:
            List of integer token IDs. Unknown event types are mapped to <UNK> (1).
        """
        if isinstance(sequence, str):
            tokens = sequence.strip().split()
        elif isinstance(sequence, (list, tuple)):
            tokens = [str(t) for t in sequence]
        else:
            return []

        return [self.token_to_id.get(token, self.UNK_ID) for token in tokens if token]

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode a list of integer token IDs back to a space-delimited string.

        Args:
            token_ids: List or 1D Tensor of integer token IDs.
            skip_special_tokens: If True, excludes <PAD> and <UNK> tokens from output.

        Returns:
            Space-delimited string of reconstructed event types.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        tokens: List[str] = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and tid in (self.PAD_ID, self.UNK_ID):
                continue
            tokens.append(token)

        return " ".join(tokens)

    def encode_batch(
        self,
        sequences: List[Union[str, List[str]]],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode a batch of event sequences into a padded 2D PyTorch LongTensor.

        Args:
            sequences: List of event sequence strings or token lists.
            max_length: Optional maximum sequence length. If specified, sequences
                exceeding this length will be truncated. If None, sequences will
                be padded to the longest sequence in the batch.

        Returns:
            torch.LongTensor of shape [batch_size, sequence_length].
        """
        encoded_list: List[List[int]] = [self.encode(seq) for seq in sequences]

        if not encoded_list:
            target_len = max_length if max_length is not None else 0
            return torch.empty((0, target_len), dtype=torch.long)

        if max_length is not None:
            target_len = max_length
        else:
            target_len = max((len(tokens) for tokens in encoded_list), default=0)

        batch_padded: List[List[int]] = []
        for tokens in encoded_list:
            if len(tokens) >= target_len:
                padded = tokens[:target_len]
            else:
                padded = tokens + [self.PAD_ID] * (target_len - len(tokens))
            batch_padded.append(padded)

        return torch.tensor(batch_padded, dtype=torch.long)


if __name__ == "__main__":
    print("Module 'tokenizer' loaded successfully.")
    print("Example usage:")
    print("    from intelligence.preprocessing.tokenizer import SecurityEventTokenizer")
    print("    tokenizer = SecurityEventTokenizer()")
    print("    tokenizer.fit(['LOGIN_SUCCESS FILE_ACCESS LOGOUT'])")
    print("    print('Vocab size:', tokenizer.vocab_size)")
    print("    # tokens = tokenizer.encode_batch(['LOGIN_SUCCESS LOGOUT'])")
