"""Sequence Builder Module for Lightweight Security Log Intelligence.

This module ingests raw security event logs from CSV files, validates schema
integrity, and transforms them into time-ordered event sequences grouped by
incident ID for downstream graph or transformer-based security analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "user",
    "src_ip",
    "dst_host",
    "event_type",
    "status",
    "severity",
    "incident_id",
)


def build_incident_sequences(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read security events from a CSV and construct ordered sequences by incident.

    Args:
        csv_path: Path to the security events CSV file.

    Returns:
        List of dictionaries where each item represents an incident sequence with:
            - incident_id: The identifier of the incident.
            - events: Chronologically ordered list of event_type strings.
            - start_time: Datetime of the first event in the incident.
            - end_time: Datetime of the last event in the incident.
            - event_count: Total number of events associated with the incident.

    Raises:
        FileNotFoundError: If the provided CSV file does not exist.
        ValueError: If required columns are missing from the CSV file.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Event log file not found at: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s) in CSV: {', '.join(missing_columns)}. "
            f"Expected schema must include: {', '.join(REQUIRED_COLUMNS)}"
        )

    if df.empty:
        return []

    # Convert timestamp to pandas datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort events by incident_id and timestamp
    sorted_df = df.sort_values(by=["incident_id", "timestamp"])

    # Group by incident_id and create ordered sequences
    sequences: List[Dict[str, Any]] = []
    for incident_id, group in sorted_df.groupby("incident_id", sort=False):
        events = group["event_type"].tolist()
        sequences.append(
            {
                "incident_id": incident_id,
                "events": events,
                "start_time": group["timestamp"].iloc[0],
                "end_time": group["timestamp"].iloc[-1],
                "event_count": len(events),
            }
        )

    return sequences


if __name__ == "__main__":
    # Demonstration of expected function usage:
    # example_path = "data/sample/sample_events.csv"
    # incident_sequences = build_incident_sequences(example_path)
    # for seq in incident_sequences:
    #     print(seq)
    print("Module 'sequence_builder' loaded successfully.")
    print("Example usage:")
    print("    from intelligence.preprocessing.sequence_builder import build_incident_sequences")
    print("    sequences = build_incident_sequences('data/sample/sample_events.csv')")
