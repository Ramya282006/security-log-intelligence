"""Target Label Mapping Utility for LANL Transformer Next-Event Prediction.

This module provides bidirectional mapping between raw LANL categorical target IDs,
0-indexed model prediction class labels, and human-readable string identifiers.

Token Conventions:
    - ID 0: <PAD> (Padding token; prohibited as a target label).
    - ID 1: <UNK> (Out-of-vocabulary / unseen token; mapped to IGNORE_INDEX = -100).
    - IDs >= 2: Known vocabulary entities (mapped to contiguous class indices 0..N-1).
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "lanl_vocabularies.json"


class TargetLabelMapper:
    """Bidirectional label mapper for the four LANL Transformer next-event prediction heads.
    
    Attributes:
        FEATURES: Tuple of the four target feature names in column order.
        PAD_ID: Reserved padding integer ID (0).
        UNK_ID: Reserved unknown / out-of-vocabulary integer ID (1).
        IGNORE_INDEX: Sentinel index (-100) for loss computation ignoring unknown targets.
        PAD_TOKEN: String representation of padding token ('<PAD>').
        UNK_TOKEN: String representation of unknown token ('<UNK>').
    """

    FEATURES: Tuple[str, ...] = ("user", "computer", "process", "event_type")
    PAD_ID: int = 0
    UNK_ID: int = 1
    IGNORE_INDEX: int = -100
    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"

    def __init__(
        self,
        vocabularies: Optional[Dict[str, Dict[str, int]]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the target label mapper with vocabulary dictionaries.
        
        Args:
            vocabularies: Optional dictionary containing vocabulary mappings for all 4 features.
            vocab_path: Optional file path to lanl_vocabularies.json. Defaults to standard dataset location.
            
        Raises:
            FileNotFoundError: If vocab_path does not exist.
            ValueError: If vocabulary structure is invalid, contains duplicates, or includes reserved IDs.
        """
        if vocabularies is None:
            path = Path(vocab_path) if vocab_path is not None else DEFAULT_VOCAB_PATH
            if not path.exists():
                raise FileNotFoundError(f"Vocabulary file not found at: {path}")
            with open(path, "r", encoding="utf-8") as f:
                vocabularies = json.load(f)

        self._validate_and_build_mappings(vocabularies)

    def _validate_and_build_mappings(self, vocabularies: Dict[str, Dict[str, int]]) -> None:
        """Validate vocabulary structure and build internal lookup tables."""
        if not isinstance(vocabularies, dict):
            raise ValueError(f"Vocabularies must be a dictionary, got {type(vocabularies).__name__}")

        self._id_to_class: Dict[str, Dict[int, int]] = {}
        self._class_to_id: Dict[str, Dict[int, int]] = {}
        self._id_to_str: Dict[str, Dict[int, str]] = {}
        self._str_to_id: Dict[str, Dict[str, int]] = {}
        self._class_to_str: Dict[str, Dict[int, str]] = {}
        self._num_classes: Dict[str, int] = {}

        # Vectorized lookup tables (NumPy arrays) for O(1) array operations
        self._np_forward_lookup: Dict[str, np.ndarray] = {}
        self._np_reverse_lookup: Dict[str, np.ndarray] = {}

        for feat in self.FEATURES:
            if feat not in vocabularies:
                raise ValueError(f"Missing required feature '{feat}' in vocabulary dictionary.")

            raw_vocab = vocabularies[feat]
            if not isinstance(raw_vocab, dict) or len(raw_vocab) == 0:
                raise ValueError(f"Vocabulary for feature '{feat}' must be a non-empty dictionary.")

            # Validate for duplicate strings and duplicate integer IDs
            string_keys = list(raw_vocab.keys())
            id_values = list(raw_vocab.values())

            if len(string_keys) != len(set(string_keys)):
                dup_strings = [k for k, count in Counter(string_keys).items() if count > 1]
                raise ValueError(f"Duplicate string identifiers in feature '{feat}': {dup_strings}")

            if len(id_values) != len(set(id_values)):
                dup_ids = [k for k, count in Counter(id_values).items() if count > 1]
                raise ValueError(f"Duplicate integer IDs in feature '{feat}': {dup_ids}")

            # Check that reserved IDs 0 and 1 are not in vocabulary entries
            if self.PAD_ID in id_values:
                raise ValueError(f"Reserved padding ID ({self.PAD_ID}) cannot be a named class in '{feat}'.")
            if self.UNK_ID in id_values:
                raise ValueError(f"Reserved UNK ID ({self.UNK_ID}) cannot be a named class in '{feat}'.")

            # Validate IDs are positive integers >= 2
            for val in id_values:
                if not isinstance(val, (int, np.integer)):
                    raise ValueError(f"All vocabulary IDs for '{feat}' must be integers, got {type(val)}: {val}")
                if val < 2:
                    raise ValueError(f"Vocabulary ID {val} for '{feat}' is invalid (must be >= 2).")

            # Sort deterministically by original integer ID to assign contiguous class indices
            sorted_items = sorted(raw_vocab.items(), key=lambda item: item[1])
            num_classes = len(sorted_items)
            self._num_classes[feat] = num_classes

            id_to_class: Dict[int, int] = {}
            class_to_id: Dict[int, int] = {}
            id_to_str: Dict[int, str] = {}
            str_to_id: Dict[str, int] = {}
            class_to_str: Dict[int, str] = {}

            max_id = max(id_values)
            # Lookup table initialized to -999 (invalid sentinel)
            forward_arr = np.full(max_id + 1, -999, dtype=np.int64)
            forward_arr[self.UNK_ID] = self.IGNORE_INDEX  # 1 -> -100

            reverse_arr = np.full(num_classes, -999, dtype=np.int64)

            for class_idx, (name, orig_id) in enumerate(sorted_items):
                id_to_class[orig_id] = class_idx
                class_to_id[class_idx] = orig_id
                id_to_str[orig_id] = name
                str_to_id[name] = orig_id
                class_to_str[class_idx] = name

                forward_arr[orig_id] = class_idx
                reverse_arr[class_idx] = orig_id

            self._id_to_class[feat] = id_to_class
            self._class_to_id[feat] = class_to_id
            self._id_to_str[feat] = id_to_str
            self._str_to_id[feat] = str_to_id
            self._class_to_str[feat] = class_to_str

            self._np_forward_lookup[feat] = forward_arr
            self._np_reverse_lookup[feat] = reverse_arr

    @property
    def features(self) -> Tuple[str, ...]:
        """Names of all supported features in column order."""
        return self.FEATURES

    @property
    def head_dimensions(self) -> Dict[str, int]:
        """Expected output head dimensions for each of the four features."""
        return dict(self._num_classes)

    def num_classes(self, feature: str) -> int:
        """Return the number of valid target classes for a given feature."""
        self._check_feature(feature)
        return self._num_classes[feature]

    def _check_feature(self, feature: str) -> None:
        if feature not in self.FEATURES:
            raise ValueError(f"Unknown feature '{feature}'. Supported features are: {self.FEATURES}")

    # -------------------------------------------------------------------------
    # Single-element mapping functions
    # -------------------------------------------------------------------------

    def to_class_index(self, original_id: int, feature: str) -> int:
        """Convert a single original LANL integer ID into a 0-indexed class index.
        
        Args:
            original_id: Integer LANL ID.
            feature: Feature name ('user', 'computer', 'process', 'event_type').
            
        Returns:
            Contiguous class index in [0, num_classes - 1], or -100 if original_id is UNK (1).
            
        Raises:
            ValueError: If original_id is 0 (PAD) or unknown/invalid.
        """
        self._check_feature(feature)
        if original_id == self.PAD_ID:
            raise ValueError(
                f"Padding ID {self.PAD_ID} cannot be converted to a target class for '{feature}'."
            )
        if original_id == self.UNK_ID:
            return self.IGNORE_INDEX

        if original_id in self._id_to_class[feature]:
            return self._id_to_class[feature][original_id]

        raise ValueError(
            f"Original ID {original_id} is not present in vocabulary for feature '{feature}'."
        )

    def to_original_id(self, class_index: int, feature: str, allow_unknown: bool = False) -> int:
        """Convert a single model class index prediction back into an original LANL ID.
        
        Args:
            class_index: Model prediction class index.
            feature: Feature name.
            allow_unknown: If True, class_index == -100 is converted to UNK ID (1).
            
        Returns:
            Original LANL integer ID (>= 2, or 1 if allow_unknown and class_index == -100).
            
        Raises:
            ValueError: If class_index is invalid or out of range.
        """
        self._check_feature(feature)
        if class_index == self.IGNORE_INDEX:
            if allow_unknown:
                return self.UNK_ID
            raise ValueError(
                f"Class index {self.IGNORE_INDEX} represents an unknown/ignored label, "
                f"not a valid model prediction index for '{feature}'."
            )

        if class_index in self._class_to_id[feature]:
            return self._class_to_id[feature][class_index]

        num_cls = self._num_classes[feature]
        raise ValueError(
            f"Invalid class index {class_index} for feature '{feature}'. "
            f"Must be in range [0, {num_cls - 1}]."
        )

    def to_string_identifier(
        self,
        identifier: int,
        feature: str,
        from_class_index: bool = False,
    ) -> str:
        """Recover the original human-readable string identifier.
        
        Args:
            identifier: Either an original ID or a model class index.
            feature: Feature name.
            from_class_index: Set to True if identifier is a model class index.
            
        Returns:
            String name (e.g., 'U1289@DOM1', 'C108', '<UNK>', '<PAD>').
            
        Raises:
            ValueError: If identifier is invalid or out of range.
        """
        self._check_feature(feature)
        if from_class_index:
            if identifier == self.IGNORE_INDEX:
                return self.UNK_TOKEN
            if identifier in self._class_to_str[feature]:
                return self._class_to_str[feature][identifier]
            num_cls = self._num_classes[feature]
            raise ValueError(
                f"Invalid class index {identifier} for feature '{feature}'. "
                f"Must be in range [0, {num_cls - 1}]."
            )
        else:
            if identifier == self.PAD_ID:
                return self.PAD_TOKEN
            if identifier == self.UNK_ID:
                return self.UNK_TOKEN
            if identifier in self._id_to_str[feature]:
                return self._id_to_str[feature][identifier]
            raise ValueError(
                f"Original ID {identifier} not found in vocabulary for feature '{feature}'."
            )

    # -------------------------------------------------------------------------
    # Batch / Vectorized conversion functions
    # -------------------------------------------------------------------------

    def to_class_indices(
        self,
        original_ids: Union[List[int], np.ndarray, torch.Tensor],
        feature: str,
    ) -> Union[List[int], np.ndarray, torch.Tensor]:
        """Convert a sequence or tensor of original LANL IDs into model class indices.
        
        Args:
            original_ids: 1D collection of integer IDs (list, NumPy array, or torch.Tensor).
            feature: Feature name.
            
        Returns:
            Collection of same type containing 0-indexed class labels and -100 for UNKs.
            
        Raises:
            ValueError: If any ID is 0 (PAD) or unknown.
        """
        self._check_feature(feature)
        is_torch = isinstance(original_ids, torch.Tensor)
        is_list = isinstance(original_ids, (list, tuple))

        if is_torch:
            arr = original_ids.detach().cpu().numpy()
        elif is_list:
            arr = np.asarray(original_ids, dtype=np.int64)
        else:
            arr = np.asarray(original_ids)

        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"original_ids must contain integers, got dtype {arr.dtype}")

        # Validate padding ID 0
        if np.any(arr == self.PAD_ID):
            raise ValueError(
                f"Found padding ID ({self.PAD_ID}) in '{feature}' target IDs. "
                "Padding is invalid as a target label."
            )

        lookup = self._np_forward_lookup[feature]
        max_valid_id = len(lookup) - 1

        # Check bounds
        out_of_bounds = (arr < 0) | (arr > max_valid_id)
        if np.any(out_of_bounds):
            bad_val = int(arr[out_of_bounds].flat[0])
            raise ValueError(
                f"Invalid original ID {bad_val} for feature '{feature}': out of vocabulary range."
            )

        # Apply lookup
        mapped = lookup[arr]

        # Check for non-existent IDs that had sentinel -999
        unmapped = (mapped == -999)
        if np.any(unmapped):
            bad_val = int(arr[unmapped].flat[0])
            raise ValueError(
                f"Original ID {bad_val} for feature '{feature}' is not present in vocabulary."
            )

        if is_torch:
            return torch.as_tensor(mapped, dtype=torch.long, device=original_ids.device)
        elif is_list:
            return mapped.tolist()
        return mapped

    def to_original_ids(
        self,
        class_indices: Union[List[int], np.ndarray, torch.Tensor],
        feature: str,
        allow_unknown: bool = False,
    ) -> Union[List[int], np.ndarray, torch.Tensor]:
        """Convert a sequence or tensor of model class indices back into original LANL IDs.
        
        Args:
            class_indices: 1D collection of model class indices.
            feature: Feature name.
            allow_unknown: If True, maps -100 back to UNK ID 1.
            
        Returns:
            Collection of same type containing original LANL IDs.
            
        Raises:
            ValueError: If any index is invalid or if -100 is encountered when allow_unknown=False.
        """
        self._check_feature(feature)
        is_torch = isinstance(class_indices, torch.Tensor)
        is_list = isinstance(class_indices, (list, tuple))

        if is_torch:
            arr = class_indices.detach().cpu().numpy()
        elif is_list:
            arr = np.asarray(class_indices, dtype=np.int64)
        else:
            arr = np.asarray(class_indices)

        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"class_indices must contain integers, got dtype {arr.dtype}")

        reverse_arr = self._np_reverse_lookup[feature]
        num_cls = len(reverse_arr)

        if allow_unknown:
            is_unk = (arr == self.IGNORE_INDEX)
            valid_class = (arr >= 0) & (arr < num_cls)
            invalid = ~(is_unk | valid_class)
            if np.any(invalid):
                bad_val = int(arr[invalid].flat[0])
                raise ValueError(
                    f"Invalid class index {bad_val} for feature '{feature}'. "
                    f"Must be in [0, {num_cls - 1}] or {self.IGNORE_INDEX}."
                )
            result = np.where(is_unk, self.UNK_ID, reverse_arr[np.clip(arr, 0, num_cls - 1)])
        else:
            invalid = (arr < 0) | (arr >= num_cls)
            if np.any(invalid):
                bad_val = int(arr[invalid].flat[0])
                msg = f"Invalid class index {bad_val} for feature '{feature}'. Must be in [0, {num_cls - 1}]."
                if bad_val == self.IGNORE_INDEX:
                    msg += " (-100 is an ignored UNK label, not a valid prediction. Set allow_unknown=True if recovering targets)."
                raise ValueError(msg)
            result = reverse_arr[arr]

        if is_torch:
            return torch.as_tensor(result, dtype=torch.long, device=class_indices.device)
        elif is_list:
            return result.tolist()
        return result

    # -------------------------------------------------------------------------
    # Full (N, 4) targets transformation
    # -------------------------------------------------------------------------

    def transform_targets(
        self,
        targets: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform an (N, 4) matrix of original LANL target IDs to (N, 4) model class indices.
        
        Column order must be: ('user', 'computer', 'process', 'event_type').
        Known IDs (>= 2) are mapped to 0-indexed class indices.
        UNK IDs (1) are mapped to -100 (IGNORE_INDEX).
        Padding IDs (0) or unrecognized IDs trigger ValueError.
        
        Args:
            targets: (N, 4) matrix as NumPy array or PyTorch Tensor.
            
        Returns:
            (N, 4) matrix of mapped class indices with matching type and device.
        """
        is_torch = isinstance(targets, torch.Tensor)
        if is_torch:
            shape = targets.shape
            device = targets.device
            raw = targets.detach().cpu().numpy()
        else:
            raw = np.asarray(targets)
            shape = raw.shape
            device = None

        if len(shape) != 2 or shape[1] != len(self.FEATURES):
            raise ValueError(
                f"targets must have shape (N, {len(self.FEATURES)}), got {shape}"
            )

        cols = []
        for i, feat in enumerate(self.FEATURES):
            col_mapped = self.to_class_indices(raw[:, i], feature=feat)
            cols.append(col_mapped)

        result = np.column_stack(cols)

        if is_torch:
            return torch.as_tensor(result, dtype=torch.long, device=device)
        return result

    def inverse_transform_targets(
        self,
        class_indices: Union[np.ndarray, torch.Tensor],
        allow_unknown: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert an (N, 4) matrix of model class indices back to (N, 4) original LANL target IDs.
        
        Args:
            class_indices: (N, 4) matrix of class indices.
            allow_unknown: If True, maps -100 back to UNK ID 1.
            
        Returns:
            (N, 4) matrix of original LANL target IDs with matching type and device.
        """
        is_torch = isinstance(class_indices, torch.Tensor)
        if is_torch:
            shape = class_indices.shape
            device = class_indices.device
            raw = class_indices.detach().cpu().numpy()
        else:
            raw = np.asarray(class_indices)
            shape = raw.shape
            device = None

        if len(shape) != 2 or shape[1] != len(self.FEATURES):
            raise ValueError(
                f"class_indices must have shape (N, {len(self.FEATURES)}), got {shape}"
            )

        cols = []
        for i, feat in enumerate(self.FEATURES):
            col_orig = self.to_original_ids(raw[:, i], feature=feat, allow_unknown=allow_unknown)
            cols.append(col_orig)

        result = np.column_stack(cols)

        if is_torch:
            return torch.as_tensor(result, dtype=torch.long, device=device)
        return result

