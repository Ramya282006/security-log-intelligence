"""Temporal Attack Graph Module for Lightweight Security Log Intelligence.

This module models security incidents as directed temporal graphs using NetworkX.
It supports both:
1. Batch construction from incident sequences (e.g., from sequence_builder.py).
2. Incremental updates as new security events arrive in real-time streaming pipelines.
"""

from typing import Any, Dict, List, Optional
import networkx as nx
import pandas as pd


class TemporalAttackGraph:
    """Directed temporal graph representation for security incident event sequences.
    
    Supports both batch creation from historic sequences and real-time streaming ingestion
    via incremental event updates.
    """

    def __init__(self, sequences: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize an empty directed temporal attack graph.

        Args:
            sequences: Optional list of incident sequence dictionaries to build in batch.
        """
        self.graph: nx.DiGraph = nx.DiGraph()
        # Internal state tracking the most recent event_id for each incident
        self._latest_event_by_incident: Dict[str, str] = {}

        if sequences:
            self.build_from_sequences(sequences)

    def add_event(self, event: Dict[str, Any]) -> None:
        """Add a single security event incrementally to the graph.

        Connects consecutive events within the same incident:
            previous_event -> current_event
        Stores `time_delta_seconds` on the directed edge.

        Args:
            event: Dictionary representing a security event with required keys:
                - event_id
                - incident_id
                - timestamp
                and optional keys:
                - event_type, user, src_ip, dst_host, severity

        Raises:
            ValueError: If incident_id, event_id, or timestamp is missing.
        """
        if not isinstance(event, dict):
            raise ValueError("Event must be a dictionary.")

        # Validate required fields
        for field in ("incident_id", "event_id", "timestamp"):
            if field not in event or event[field] is None or event[field] == "":
                raise ValueError(f"Missing required event field: '{field}'")

        event_id = str(event["event_id"])

        # Prevent duplicate nodes
        if self.graph.has_node(event_id):
            return

        incident_id = str(event["incident_id"])
        timestamp = pd.to_datetime(event["timestamp"])

        # Build node attributes dictionary
        node_attrs: Dict[str, Any] = {
            "event_id": event_id,
            "incident_id": incident_id,
            "timestamp": timestamp,
        }

        optional_fields = ("event_type", "user", "src_ip", "dst_host", "severity")
        for field in optional_fields:
            if field in event and event[field] is not None:
                node_attrs[field] = event[field]

        # Add event as a node in the graph
        self.graph.add_node(event_id, **node_attrs)

        # Connect with the most recent event of the same incident if chronological
        if incident_id in self._latest_event_by_incident:
            prev_event_id = self._latest_event_by_incident[incident_id]
            prev_ts = self.graph.nodes[prev_event_id].get("timestamp")

            # Only create an edge if incoming event is not earlier than the latest event
            if prev_ts is None or timestamp >= prev_ts:
                time_delta_seconds = self._calculate_time_delta(prev_ts, timestamp)
                self.graph.add_edge(
                    prev_event_id,
                    event_id,
                    time_delta_seconds=time_delta_seconds,
                )
                self._latest_event_by_incident[incident_id] = event_id
        else:
            # First event for this incident
            self._latest_event_by_incident[incident_id] = event_id

    def build_from_sequences(
        self, sequences: Optional[List[Dict[str, Any]]]
    ) -> nx.DiGraph:
        """Build the graph from complete incident sequences in batch mode.

        Args:
            sequences: List of incident sequence dictionaries produced by
                the sequence builder.

        Returns:
            The populated networkx.DiGraph instance.
        """
        self.graph.clear()
        self._latest_event_by_incident.clear()

        if not sequences:
            return self.graph

        for seq in sequences:
            incident_id = seq.get("incident_id")
            if not incident_id:
                continue

            events = seq.get("events", [])
            if not events:
                continue

            for idx, event in enumerate(events):
                if isinstance(event, dict):
                    event_dict = dict(event)
                    if "incident_id" not in event_dict:
                        event_dict["incident_id"] = incident_id
                    self.add_event(event_dict)
                else:
                    # Support event_type string sequences from sequence_builder.py
                    ts = seq.get("start_time") if idx == 0 else seq.get("end_time", seq.get("start_time"))
                    event_dict = {
                        "event_id": f"{incident_id}_evt_{idx + 1}",
                        "event_type": str(event),
                        "incident_id": incident_id,
                        "timestamp": ts,
                    }
                    self.add_event(event_dict)

        return self.graph

    def get_incident_subgraph(self, incident_id: str) -> nx.DiGraph:
        """Extract a directed subgraph containing only events from a specific incident.

        Args:
            incident_id: The identifier of the incident.

        Returns:
            A copy of the subgraph for the incident, or an empty DiGraph if not found.
        """
        nodes = self.get_attack_path(incident_id)
        if not nodes:
            return nx.DiGraph()
        return self.graph.subgraph(nodes).copy()

    def get_attack_path(self, incident_id: str) -> List[str]:
        """Return the chronologically ordered event nodes for the given incident.

        Args:
            incident_id: The identifier of the incident.

        Returns:
            List of event_id strings in chronological order.
        """
        incident_str = str(incident_id)
        incident_nodes = [
            node
            for node, data in self.graph.nodes(data=True)
            if str(data.get("incident_id")) == incident_str
        ]

        def get_timestamp(node_id: str) -> Any:
            ts = self.graph.nodes[node_id].get("timestamp")
            return pd.to_datetime(ts) if ts is not None else pd.Timestamp.min

        incident_nodes.sort(key=get_timestamp)
        return incident_nodes

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Compute structural statistics of the temporal attack graph.

        Returns:
            Dictionary containing node_count, edge_count, incident_count,
            and average_degree.
        """
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()

        incidents = {
            data["incident_id"]
            for _, data in self.graph.nodes(data=True)
            if "incident_id" in data
        }
        incident_count = len(incidents)

        average_degree = (
            float(sum(dict(self.graph.degree()).values()) / node_count)
            if node_count > 0
            else 0.0
        )

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "incident_count": incident_count,
            "average_degree": average_degree,
        }

    @staticmethod
    def _calculate_time_delta(t1: Any, t2: Any) -> float:
        """Calculate time difference in seconds between two timestamps.

        Args:
            t1: Previous event timestamp.
            t2: Current event timestamp.

        Returns:
            Time difference in seconds as a float (>= 0.0), or 0.0 if cannot be computed.
        """
        if t1 is None or t2 is None:
            return 0.0
        try:
            ts1 = pd.to_datetime(t1)
            ts2 = pd.to_datetime(t2)
            delta = (ts2 - ts1).total_seconds()
            return max(0.0, float(delta))
        except Exception:
            return 0.0


if __name__ == "__main__":
    print("Module 'temporal_attack_graph' loaded successfully.")
    print("Example usage:")
    print("    from intelligence.graph.temporal_attack_graph import TemporalAttackGraph")
    print("    tag = TemporalAttackGraph()")
    print("    tag.add_event({'event_id': 'EVT001', 'incident_id': 'INC01', 'timestamp': '2026-10-08 10:00:00'})")
    print("    print(tag.get_graph_statistics())")
