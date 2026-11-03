"""Process Activity Intelligence Analyst Prototype Launcher.

Usage:
    .venv\\Scripts\\python.exe dashboard.py [--port 8080] [--host 127.0.0.1] [--checkpoint models/lanl_run2_best.pt]
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.dashboard.server import run_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Activity Intelligence Analyst Dashboard Prototype"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Local port to bind HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Local host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Transformer checkpoint (defaults to models/lanl_run2_best.pt)",
    )

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    run_dashboard(host=args.host, port=args.port, checkpoint_path=ckpt_path)


if __name__ == "__main__":
    main()

