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

server = create_dashboard_server(port=8089)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
time.sleep(1.0)

# 1. Test status
req = urllib.request.Request("http://127.0.0.1:8089/api/kafka/status")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("Initial status:", data["kafka_connected"], "is_running:", data["is_running"])

# 2. Test start stream
req = urllib.request.Request(
    "http://127.0.0.1:8089/api/kafka/start",
    data=json.dumps({"interval": 0.1, "count": 8}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    print("Start response:", resp.read().decode())

time.sleep(3.0)

# 3. Test status during/after streaming
req = urllib.request.Request("http://127.0.0.1:8089/api/kafka/status")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print(f"Streamed total: {data['total_events']} | Normal: {data['normal_events']} | Unusual: {data['unusual_events']}")
    if data["latest_detection"]:
        print(f"Latest detection: {data['latest_detection']['reason']}")

# 4. Test stop
req = urllib.request.Request(
    "http://127.0.0.1:8089/api/kafka/stop",
    data=b"{}",
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    print("Stop response:", resp.read().decode())

server.shutdown()
server.server_close()
print("Dashboard Kafka API verification SUCCESSFUL!")
