"""Interactive Research Inspection Dashboard Launcher (Phase 12).

Launches the local FastAPI and glassmorphic web dashboard for visualizing
OCR bounding box heatmaps, selective correction audit trails, 3-variant RAG
question comparisons, and LaTeX table publication exports.
"""

import argparse
import sys
from src.dashboard.app import run_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR-Aware Multilingual RAG Interactive Inspection Dashboard"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reloading for development",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("LAUNCHING OCR-AWARE MULTILINGUAL RAG RESEARCH DASHBOARD")
    print(f"URL: http://{args.host}:{args.port}")
    print("=" * 80)
    run_dashboard(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
