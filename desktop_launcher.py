from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from streamlit.web import cli as stcli


def find_available_port(start_port: int, span: int = 30) -> int:
    for port in range(start_port, start_port + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def main() -> None:
    parser = argparse.ArgumentParser(description="Start StockLab desktop app")
    parser.add_argument("--port", type=int, default=8501, help="Preferred local port")
    args = parser.parse_args()

    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app_path = base_dir / "app.py"
    port = find_available_port(args.port)
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode",
        "false",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
