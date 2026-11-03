"""Unit tests for the Analyst Dashboard server and API endpoints."""

import json
from pathlib import Path
import sys
import unittest
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import time
from intelligence.dashboard.server import create_dashboard_server


class TestDashboardServer(unittest.TestCase):
    """Integration test suite for the dashboard HTTP server."""

    @classmethod
    def setUpClass(cls):
        cls.port = 8899
        cls.host = "127.0.0.1"
        cls.server = create_dashboard_server(host=cls.host, port=cls.port)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path: str):
        url = f"http://{self.host}:{self.port}{path}"
        req = Request(url)
        with urlopen(req) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            return resp.status, content_type, data

    def _post(self, path: str, payload: dict):
        url = f"http://{self.host}:{self.port}{path}"
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            data = resp.read()
            return resp.status, json.loads(data.decode("utf-8"))

    def test_root_serves_html(self):
        """Root endpoint serves HTML dashboard page."""
        status, ctype, data = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        html_str = data.decode("utf-8")
        self.assertIn("Process Activity Intelligence", html_str)

    def test_api_status(self):
        """API status returns model metadata and calibration thresholds."""
        status, ctype, data = self._get("/api/status")
        self.assertEqual(status, 200)
        res = json.loads(data.decode("utf-8"))
        self.assertIn("checkpoint", res)
        self.assertIn("thresholds", res)
        self.assertIn("p95", res["thresholds"])

    def test_api_sequences_no_test_split(self):
        """Sequences list returns available non-test validation sequences."""
        status, ctype, data = self._get("/api/sequences")
        self.assertEqual(status, 200)
        seqs = json.loads(data.decode("utf-8"))
        self.assertGreater(len(seqs), 0)
        for s in seqs:
            self.assertNotEqual(s.get("data_split"), "test", "Test split must remain isolated.")

    def test_step_execution_and_prediction(self):
        """Step endpoint advances sequence and computes predictions."""
        # Reset first
        self._post("/api/reset", {})

        # Step 1: first event
        status, res = self._post("/api/step", {})
        self.assertEqual(status, 200)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["current_step"], 1)
        self.assertIn("incoming_event", res)
        self.assertIn("next_prediction", res)

        # Step 2: second event (now has history -> generates surprise score)
        status, res2 = self._post("/api/step", {})
        self.assertEqual(status, 200)
        self.assertEqual(res2["current_step"], 2)
        self.assertIn("anomaly_score", res2)
        self.assertIsNotNone(res2["anomaly_score"])
        self.assertIn("total_surprise", res2["anomaly_score"])

    def test_inject_anomaly_simulated_event(self):
        """Simulated anomaly is processed and tagged properly."""
        status, res = self._post("/api/inject_anomaly", {"anomaly_type": "rare_process"})
        self.assertEqual(status, 200)
        self.assertEqual(res["status"], "ok")
        self.assertIn("Simulated Demonstration", res["incoming_event"]["source"])
        self.assertIn("anomaly_score", res)

    def test_submit_custom_event(self):
        """Custom analyst-submitted event is evaluated properly."""
        payload = {
            "user": "U8946@DOM1",
            "computer": "C19038",
            "process": "P4",
            "event_type": "Start",
        }
        status, res = self._post("/api/submit_event", payload)
        self.assertEqual(status, 200)
        self.assertEqual(res["status"], "ok")
        self.assertIn("Manual Submission", res["incoming_event"]["source"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

