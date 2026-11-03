"""Monitoring scope definitions and host-to-domain/organization mapping."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAPPING_FILE = PROJECT_ROOT / "data" / "lanl_transformer_dataset" / "scope_mappings.json"

DEFAULT_ORGANIZATION_NAME = "LANL Enterprise Network"

DEFAULT_DOMAINS: Dict[str, Dict[str, Any]] = {
    "DOM1": {
        "name": "DOM1 (Primary Enterprise Domain)",
        "description": "Central Active Directory enterprise domain authenticating LANL staff workstations.",
        "hosts": [
            "C10", "C10005", "C1014", "C108", "C1119", "C12448", "C126",
            "C1268", "C1448", "C1461", "C16401", "C1810", "C19038", "C1980",
            "C19932", "C2091", "C20966", "C21963", "C22409", "C231", "C2519",
            "C366", "C395", "C457", "C633", "C636", "C728", "C8209", "C828",
            "C8585", "C886", "C90", "C9692", "C9723", "C9945"
        ],
    },
    "LAB_DOM": {
        "name": "LAB_DOM (Research & Perimeter Testbed)",
        "description": "Isolated computing environment for specialized laboratory hosts.",
        "hosts": ["C10577", "C2669", "C3422"],
    },
}


class ScopeManager:
    """Manages monitoring scopes, host-to-domain mappings, and scope filtering."""

    def __init__(self, mapping_file: Optional[Path] = None, compute_counts: bool = True) -> None:
        self.mapping_file = Path(mapping_file or MAPPING_FILE)
        self.organization_name = DEFAULT_ORGANIZATION_NAME
        self.domains: Dict[str, Dict[str, Any]] = dict(DEFAULT_DOMAINS)
        self._host_to_domain: Dict[str, str] = {}
        self._known_hosts: Set[str] = set()
        self._host_event_counts: Dict[str, int] = {}
        self._replayable_hosts: Set[str] = set()

        self._load_or_initialize_mappings()
        if compute_counts:
            self._compute_replayable_hosts()

    def _load_or_initialize_mappings(self) -> None:
        """Load mappings from file or initialize defaults."""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.organization_name = data.get("organization_name", DEFAULT_ORGANIZATION_NAME)
                    self.domains = data.get("domains", DEFAULT_DOMAINS)
            except Exception as e:
                logger.warning("Could not read scope mapping file, using defaults: %s", e)

        # Build reverse index host -> domain
        self._host_to_domain.clear()
        self._known_hosts.clear()
        for dom_id, dom_info in self.domains.items():
            for h in dom_info.get("hosts", []):
                self._host_to_domain[h] = dom_id
                self._known_hosts.add(h)

    def _compute_replayable_hosts(self) -> None:
        """Scan available non-test dataset sequences to index genuine replayable event counts."""
        try:
            from intelligence.inference.replay import ProcessEventReplayer

            replayer = ProcessEventReplayer()
            all_seqs = replayer.list_available_sequences(None)  # Strict non-test only
            counts: Dict[str, int] = {}
            for s in all_seqs:
                evs = replayer.get_sequence_events(s["sequence_index"])
                for e in evs:
                    h = e.get("computer", "")
                    if h and h != "<UNK>":
                        counts[h] = counts.get(h, 0) + 1

            self._host_event_counts = counts
            self._replayable_hosts = set(counts.keys())
        except Exception as e:
            logger.warning("Could not scan dataset for replayable hosts: %s", e)
            self._host_event_counts = {h: 1 for h in self._known_hosts}
            self._replayable_hosts = set(self._known_hosts)

    def get_domain_for_host(self, host: str) -> Optional[str]:
        """Return domain name for a given host identifier."""
        return self._host_to_domain.get(host)

    def get_hosts_for_domain(self, domain_id: str) -> List[str]:
        """Return list of hosts mapped to a domain."""
        dom = self.domains.get(domain_id)
        return list(dom.get("hosts", [])) if dom else []

    def get_all_hosts(self) -> List[str]:
        """Return all known hosts sorted."""
        return sorted(list(self._known_hosts))

    def get_replayable_hosts(self) -> List[str]:
        """Return all hosts that actually have replayable events."""
        return sorted(list(self._replayable_hosts))

    def has_replayable_events(self, host: str) -> bool:
        """Check whether a host has replayable events."""
        return self._host_event_counts.get(host, 0) > 0

    def get_event_count_for_host(self, host: str) -> int:
        """Return number of replayable events for a specific host."""
        return self._host_event_counts.get(host, 0)

    def get_scope_metadata(self) -> Dict[str, Any]:
        """Return scope choices, counts, and descriptions for frontend dropdowns."""
        total_org_events = sum(self._host_event_counts.values())
        return {
            "organization": {
                "name": self.organization_name,
                "total_hosts": len(self._known_hosts),
                "replayable_hosts_count": len(self._replayable_hosts),
                "total_events": total_org_events,
                "description": f"Covers all {len(self._replayable_hosts)} hosts across the {self.organization_name} dataset ({total_org_events:,} total events).",
            },
            "domains": [
                {
                    "id": dom_id,
                    "name": info.get("name", dom_id),
                    "description": info.get("description", ""),
                    "total_hosts_count": len(info.get("hosts", [])),
                    "replayable_hosts_count": len([h for h in info.get("hosts", []) if self.has_replayable_events(h)]),
                    "total_events": sum(self._host_event_counts.get(h, 0) for h in info.get("hosts", [])),
                    "hosts": info.get("hosts", []),
                }
                for dom_id, info in self.domains.items()
            ],
            "hosts": [
                {
                    "id": h,
                    "domain": self.get_domain_for_host(h) or "Unknown",
                    "event_count": self._host_event_counts.get(h, 0),
                    "has_events": self.has_replayable_events(h),
                }
                for h in sorted(list(self._known_hosts))
            ],
        }

    def matches_scope(
        self,
        event: Dict[str, Any],
        scope_type: str,
        target_id: Optional[str] = None,
    ) -> bool:
        """Evaluate whether an event falls within the specified monitoring scope."""
        host = event.get("computer", "")
        if not host:
            return False

        if scope_type == "organization":
            return True

        if scope_type == "single_host":
            if not target_id:
                return False
            return host.upper() == target_id.upper()

        if scope_type == "domain":
            if not target_id or target_id not in self.domains:
                return False
            allowed_hosts = set(self.get_hosts_for_domain(target_id))
            return host in allowed_hosts

        return False

    def get_scope_label(self, scope_type: str, target_id: Optional[str] = None) -> str:
        """Human-readable scope label for dashboard display."""
        if scope_type == "single_host":
            if not target_id:
                return "Host: Unspecified"
            count = self.get_event_count_for_host(target_id)
            dom = self.get_domain_for_host(target_id) or "Unknown"
            return f"Host: {target_id} ({count} events, Domain: {dom})"
        elif scope_type == "domain":
            dom_info = self.domains.get(target_id, {})
            dom_name = dom_info.get("name", target_id)
            replayable = len([h for h in dom_info.get("hosts", []) if self.has_replayable_events(h)])
            return f"Domain: {dom_name} ({replayable} replayable hosts)"
        elif scope_type == "organization":
            return f"Entire Organization ({self.organization_name} - {len(self._replayable_hosts)} hosts)"
        return f"Unknown Scope ({scope_type})"
