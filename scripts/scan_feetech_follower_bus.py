#!/usr/bin/env python3
"""Scan a serial port for Feetech servos (same probe as LeRobot's FeetechMotorsBus.scan_port).

Use this outside ROS to verify the follower (or leader) bus before launch:

    ./scripts/scan_feetech_follower_bus.py
    ./scripts/scan_feetech_follower_bus.py /dev/ttyACM0

Requires the same Python environment where ``lerobot`` is installed (e.g. conda ``grover``).
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="FeetechMotorsBus.scan_port() helper for SO-101 USB serial.")
    parser.add_argument(
        "port",
        nargs="?",
        default="/dev/so101_follower",
        help="Serial device or symlink (default: %(default)s)",
    )
    args = parser.parse_args()

    resolved = os.path.realpath(args.port)
    print(f"requested: {args.port}")
    print(f"resolved:  {resolved}")

    try:
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as e:
        print("error: could not import lerobot — activate the env where lerobot is installed.", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    found = FeetechMotorsBus.scan_port(resolved)
    print("scan:", found)
    return 0 if found else 2


if __name__ == "__main__":
    raise SystemExit(main())
