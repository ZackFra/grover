#!/usr/bin/env python3
"""Snapshot the current /joint_states into a named YAML under config/poses/.

Companion to scripts/moveit_goto_joints.py: capture a pose with this script,
then drive the arm back to it later with the goto script (planned through
MoveIt2, collision-checked).

Usage:
    # With so101_bringup running:
    python3 scripts/snapshot_joints.py home          # -> config/poses/home.yaml
    python3 scripts/snapshot_joints.py inspect_board
    python3 scripts/snapshot_joints.py --out my.yaml home  # override path
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSES_DIR = REPO_ROOT / "config" / "poses"


def _wait_for_joint_state(topic: str, timeout_s: float) -> JointState:
    """Spin briefly until one JointState arrives on `topic`; return it."""
    node = Node("snapshot_joints")
    received: list[JointState] = []

    def _cb(msg: JointState) -> None:
        received.append(msg)

    node.create_subscription(JointState, topic, _cb, 10)

    deadline = time.monotonic() + timeout_s
    try:
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()

    if not received:
        raise TimeoutError(
            f"No /joint_states received on '{topic}' within {timeout_s:.1f}s. "
            "Is so101_bringup running?"
        )
    return received[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        help="Pose name (writes config/poses/NAME.yaml). Ignored if --out given.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Explicit output path. Overrides the default config/poses/NAME.yaml.",
    )
    parser.add_argument(
        "--topic",
        default="/joint_states",
        help="JointState topic to read (default: /joint_states).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a JointState message (default: 5.0).",
    )
    args = parser.parse_args()

    out_path: Path = args.out or (DEFAULT_POSES_DIR / f"{args.name}.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    try:
        msg = _wait_for_joint_state(args.topic, args.timeout)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        rclpy.shutdown()
        return 1
    rclpy.shutdown()

    joints = {str(n): float(p) for n, p in zip(msg.name, msg.position, strict=True)}
    payload = {
        "joints": joints,
        "stamp": {
            "sec": int(msg.header.stamp.sec),
            "nanosec": int(msg.header.stamp.nanosec),
        },
        "frame_id": msg.header.frame_id,
        "source_topic": args.topic,
    }

    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))

    print(f"Saved {len(joints)} joint positions to {out_path.relative_to(REPO_ROOT)}:")
    for name in sorted(joints):
        print(f"  {name:14s} {joints[name]:+.6f}")
    print()
    print("Replay with:")
    print(f"  python3 scripts/moveit_goto_joints.py {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
