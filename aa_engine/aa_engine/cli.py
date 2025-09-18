from __future__ import annotations
import argparse, json, sys, pathlib
from .utils.logging import get_logger

def main():
    parser = argparse.ArgumentParser(prog="aa-cli", description="AA Engine CLI (skeleton)")
    sub = parser.add_subparsers(dest="cmd")

    p_fore = sub.add_parser("forecast", help="Run forecaster (skeleton)")
    p_fore.add_argument("--config", type=str, default="configs/forecaster.yaml")

    p_dbg = sub.add_parser("show-config", help="Print a config file")
    p_dbg.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    log = get_logger()

    if args.cmd == "show-config":
        p = pathlib.Path(args.config)
        if not p.exists():
            log.error(f"Config not found: {p}")
            sys.exit(1)
        print(p.read_text())
        sys.exit(0)

    if args.cmd == "forecast":
        log.info("Forecast placeholder — wire up once Part 1 is implemented.")
        log.info(f"Using config: {args.config}")
        print(json.dumps({"status": "ok", "msg": "forecaster runner stub"}, indent=2))
        sys.exit(0)

    parser.print_help()
