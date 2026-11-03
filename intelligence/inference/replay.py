"""Process Event Sequence Replay and Stream Simulator."""

import csv
from pathlib import Path
import sys
from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.preprocessing.label_mapping import TargetLabelMapper

DATA_DIR = PROJECT_ROOT / "data" / "lanl_transformer_dataset"


class ProcessEventReplayer:
    """Replays real LANL process sequences and provides simulated demonstration streams."""

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the replayer with dataset and vocabularies.

        Args:
            dataset_dir: Directory containing X_sequences.npy, attention_masks.npy, sequence_metadata.csv.
            vocab_path: Path to lanl_vocabularies.json.
        """
        self.data_dir = Path(dataset_dir or DATA_DIR)
        self.vocab_path = Path(vocab_path or (self.data_dir / "lanl_vocabularies.json"))
        self.mapper = TargetLabelMapper(vocab_path=self.vocab_path)

        self.metadata = self._load_metadata()
        self.X_sequences = np.load(self.data_dir / "X_sequences.npy")
        self.attention_masks = np.load(self.data_dir / "attention_masks.npy")

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load sequence_metadata.csv rows."""
        meta_file = self.data_dir / "sequence_metadata.csv"
        with open(meta_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def list_available_sequences(self, split: Optional[str] = "validation") -> List[Dict[str, Any]]:
        """List sequence summaries filtered by split.

        Args:
            split: 'validation', 'train', or None for all non-test sequences.
                Note: test split is never exposed.

        Returns:
            List of sequence metadata dicts with index and length.
        """
        results = []
        for idx, row in enumerate(self.metadata):
            # Strict isolation: never expose test split
            if row.get("data_split") == "test":
                continue
            if split and row.get("data_split") != split:
                continue

            results.append({
                "sequence_index": idx,
                "incident_id": row.get("incident_id"),
                "overlap_group_id": row.get("overlap_group_id"),
                "data_split": row.get("data_split"),
                "sequence_length": int(row.get("sequence_length", 0)),
            })
        return results

    def get_sequence_events(
        self,
        sequence_index: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve and decode the events of a specific sequence.

        Args:
            sequence_index: Index in [0..114] of the sequence.

        Returns:
            List of decoded event dictionaries in chronological order.
        """
        meta = self.metadata[sequence_index]
        if meta.get("data_split") == "test":
            raise PermissionError("Access to the test split is strictly restricted.")

        raw_seq = self.X_sequences[sequence_index]  # [80, 8]
        mask = self.attention_masks[sequence_index]  # [80]
        actual_len = int(mask.sum())

        events: List[Dict[str, Any]] = []
        for step_idx in range(actual_len):
            step_data = raw_seq[step_idx]

            # Categorical IDs
            user_id = int(round(step_data[0]))
            comp_id = int(round(step_data[1]))
            proc_id = int(round(step_data[2]))
            event_type_id = int(round(step_data[3]))

            # Decoded string identifiers
            user_str = self.mapper.to_string_identifier(user_id, "user")
            comp_str = self.mapper.to_string_identifier(comp_id, "computer")
            proc_str = self.mapper.to_string_identifier(proc_id, "process")
            event_type_str = self.mapper.to_string_identifier(event_type_id, "event_type")

            # Numeric features (inverse log1p to show human-readable approx seconds)
            time_delta_sec = float(max(0.0, np.expm1(step_data[4])))
            event_duration_sec = float(max(0.0, np.expm1(step_data[5])))
            event_count = int(round(np.expm1(step_data[6])))
            seq_position = int(round(np.expm1(step_data[7])))

            events.append({
                "step": step_idx + 1,
                "user": user_str,
                "computer": comp_str,
                "process": proc_str,
                "event_type": event_type_str,
                "time_delta": round(time_delta_sec, 2),
                "event_duration": round(event_duration_sec, 2),
                "event_count": max(1, event_count),
                "sequence_position": max(1, seq_position),
                "incident_id": meta.get("incident_id"),
                "source": f"LANL Dataset ({meta.get('data_split', 'unknown').capitalize()})",
            })

        return events

    def stream_sequence(
        self,
        sequence_index: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """Progressive streaming generator for a sequence."""
        events = self.get_sequence_events(sequence_index)
        for ev in events:
            yield ev

    def generate_simulated_anomaly_event(
        self,
        baseline_event: Dict[str, Any],
        anomaly_type: str = "rare_process",
    ) -> Dict[str, Any]:
        """Generate a synthetic test event designed to trigger elevated surprise.

        Clearly tagged as 'Simulated Demonstration' to distinguish from real LANL data.

        Args:
            baseline_event: Context event to mutate.
            anomaly_type: 'rare_process', 'unusual_user', 'unusual_host'.

        Returns:
            A clearly tagged synthetic event dict.
        """
        ev = dict(baseline_event)
        ev["source"] = "Simulated Demonstration (Synthetic Anomaly)"

        if anomaly_type == "rare_process":
            # Select an anomalous process that diverges from baseline on this host
            comp = baseline_event.get("computer", "")
            if comp == "C10577":
                ev["process"] = "P4"
                ev["user"] = "U3277@C2519"
            else:
                cur_proc = baseline_event.get("process", "")
                ev["process"] = "P8881" if cur_proc != "P8881" else "P4"
            ev["event_type"] = "Start"
        elif anomaly_type == "unusual_user":
            # Switch to a user that rarely touches this machine
            ev["user"] = "U3277@C2519"
        elif anomaly_type == "unusual_host":
            ev["computer"] = "C10577"
        else:
            ev["process"] = "P10239"

        ev["time_delta"] = 0.5
        return ev

