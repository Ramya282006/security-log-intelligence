import urllib.request
import json
import time
import sys
from pathlib import Path
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.dashboard.server import create_dashboard_server

server = create_dashboard_server(port=8092)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
time.sleep(1.0)

try:
    # 1. Check GET /api/kafka/scope
    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/scope")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Scope Metadata - Active:", data["active_scope"]["type"], "| Total hosts:", len(data["metadata"]["hosts"]))

    # 2. Update to Single Host: C2519
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/scope",
        data=json.dumps({"scope_type": "single_host", "target_id": "C2519"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Set Scope Single Host:", data["label"])

    # 3. Start stream with single host scope
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/start",
        data=json.dumps({"interval": 0.1, "count": 20, "scope_type": "single_host", "target_id": "C2519"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Start Response:", data["status"])

    time.sleep(3.5)

    # 4. Check status: all recent events MUST be C2519
    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/status")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print(f"Active scope: {data['active_scope']['label']} | Total events: {data['total_events']}")
        hosts = set(e["computer"] for e in data["recent_events"])
        print(f"Hosts in recent events: {hosts}")
        assert hosts.issubset({"C2519"}), f"Expected only C2519, got {hosts}"
        assert data["total_events"] > 0, "Expected at least 1 C2519 event"

    # 5. Stop stream
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/stop",
        data=b"{}",
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        pass

    # 6. Test Domain Scope DOM1
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/scope",
        data=json.dumps({"scope_type": "domain", "target_id": "DOM1"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Set Scope Domain DOM1:", data["label"])

    # Start domain stream
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/start",
        data=json.dumps({"interval": 0.1, "count": 20, "scope_type": "domain", "target_id": "DOM1"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        pass

    time.sleep(3.5)

    # Verify domain hosts (C2519, C1014, C19932, C9945 in DOM1) and no LAB_DOM (C10577)
    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/status")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        hosts_domain = set(e["computer"] for e in data["recent_events"])
        print(f"Domain scope active: {data['active_scope']['label']} | Total: {data['total_events']} | Hosts: {hosts_domain}")
        assert "C10577" not in hosts_domain, "Lab host C10577 should not be in DOM1"

    # Stop stream
    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/stop", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        pass

    # 7. Test Domain Scope LAB_DOM
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/scope",
        data=json.dumps({"scope_type": "domain", "target_id": "LAB_DOM"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Set Scope Domain LAB_DOM:", data["label"])

    # Start LAB_DOM stream
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/start",
        data=json.dumps({"interval": 0.1, "count": 15, "scope_type": "domain", "target_id": "LAB_DOM"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        pass

    time.sleep(3.0)

    # Verify LAB_DOM hosts (strictly subset of C10577, C2669, C3422, and no DOM1 hosts)
    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/status")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        hosts_lab = set(e["computer"] for e in data["recent_events"])
        print(f"LAB_DOM scope active: {data['active_scope']['label']} | Total: {data['total_events']} | Hosts: {hosts_lab}")
        lab_valid = {"C10577", "C2669", "C3422"}
        assert hosts_lab.issubset(lab_valid), f"LAB_DOM should only contain {lab_valid}, got {hosts_lab}"
        assert data["total_events"] > 0, "LAB_DOM should have streamed events"

    # Stop stream
    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/stop", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        pass

    # 8. Test Single Host in LAB_DOM (C10577)
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/scope",
        data=json.dumps({"scope_type": "single_host", "target_id": "C10577"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Set Scope Single Host C10577:", data["label"])

    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/kafka/start",
        data=json.dumps({"interval": 0.1, "count": 10, "scope_type": "single_host", "target_id": "C10577"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        pass

    time.sleep(2.5)

    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/status")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        hosts_c10577 = set(e["computer"] for e in data["recent_events"])
        print(f"Host C10577 scope: Total events: {data['total_events']} | Hosts: {hosts_c10577}")
        assert hosts_c10577.issubset({"C10577"}), f"Expected strictly C10577, got {hosts_c10577}"
        assert data["total_events"] > 0, "C10577 should stream events"

    # Stop stream
    req = urllib.request.Request("http://127.0.0.1:8092/api/kafka/stop", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        pass

    print("END-TO-END MONITORING SCOPE VERIFICATION SUCCESSFUL!")
finally:
    server.shutdown()
    server.server_close()

