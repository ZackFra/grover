#!/usr/bin/env python3
"""Emit LeRobot-style gi_jane.json from so101_follower_calibration.yaml (joint key `gripper`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML required: pip install pyyaml") from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to so101_follower_calibration.yaml",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output gi_jane.json path",
    )
    args = parser.parse_args()

    data = yaml.safe_load(args.input.read_text())
    joints = data["joints"]
    out: dict = {}
    key_map = {"gripper_joint": "gripper"}
    for joint_name, fields in joints.items():
        key = key_map.get(joint_name, joint_name)
        entry = {
            "id": fields["id"],
            "drive_mode": 0,
            "homing_offset": fields["homing_offset"],
            "range_min": fields["range_min"],
            "range_max": fields["range_max"],
        }
        out[key] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=4) + "\n")


if __name__ == "__main__":
    main()
