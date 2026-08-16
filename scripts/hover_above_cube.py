#!/usr/bin/env python3
"""Wait for a stable /red_cube/hover_pose, then MoveIt-execute to it.

Leaves the gripper above the cube (same XY, +Z offset from detect_red_cube.py)
with the orientation copied from the live gripper TF. Stops there so a
LeRobot policy can run the grasp. Do not run teleop or RViz Plan & Execute
at the same time.

Usage:
    # bringup already running, not teleop
    python3 scripts/moveit_goto_joints.py top_view
    python3 scripts/detect_red_cube.py          # leave running; check /red_cube/overlay
    python3 scripts/hover_above_cube.py         # this script
    python3 scripts/hover_above_cube.py --plan-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# Same directory as this file (python3 scripts/hover_above_cube.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moveit_goto_pose import format_moveit_result, send_pose  # noqa: E402


def _span(values: list[float]) -> float:
    return max(values) - min(values) if values else float("inf")


def wait_for_stable_hover(
    topic: str,
    *,
    samples: int,
    max_age_s: float,
    max_span_m: float,
    timeout_s: float,
) -> PoseStamped:
    """Collect `samples` fresh poses whose XYZ box is smaller than `max_span_m`."""
    node = Node("hover_above_cube_wait")
    buffer: list[tuple[float, PoseStamped]] = []

    def _cb(msg: PoseStamped) -> None:
        buffer.append((time.monotonic(), msg))

    node.create_subscription(PoseStamped, topic, _cb, qos_profile_sensor_data)
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    n_seen = 0
    last_span = (float("inf"), float("inf"), float("inf"))
    n_fresh = 0
    stable: PoseStamped | None = None
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            n_seen = len(buffer)
            fresh = [(t, m) for t, m in buffer if (now - t) <= max_age_s]
            buffer[:] = fresh[-max(samples * 4, samples) :]
            n_fresh = len(fresh)
            if n_fresh >= 2:
                window = [m for _, m in fresh[-min(samples, n_fresh) :]]
                xs = [p.pose.position.x for p in window]
                ys = [p.pose.position.y for p in window]
                zs = [p.pose.position.z for p in window]
                last_span = (_span(xs), _span(ys), _span(zs))
            if now - last_print >= 2.0:
                print(
                    f"  waiting: {n_fresh} fresh / {n_seen} total, "
                    f"span dx={last_span[0]:.3f} dy={last_span[1]:.3f} "
                    f"dz={last_span[2]:.3f} m (need {samples} samples, "
                    f"span <= {max_span_m:.3f})",
                    flush=True,
                )
                last_print = now
            if n_fresh < samples:
                continue
            window = [m for _, m in fresh[-samples:]]
            xs = [p.pose.position.x for p in window]
            ys = [p.pose.position.y for p in window]
            zs = [p.pose.position.z for p in window]
            last_span = (_span(xs), _span(ys), _span(zs))
            if max(last_span) > max_span_m:
                continue
            avg = PoseStamped()
            avg.header = window[-1].header
            avg.pose.position.x = sorted(xs)[len(xs) // 2]
            avg.pose.position.y = sorted(ys)[len(ys) // 2]
            avg.pose.position.z = sorted(zs)[len(zs) // 2]
            avg.pose.orientation = window[-1].pose.orientation
            stable = avg
            break
    finally:
        node.destroy_node()

    if stable is None:
        raise TimeoutError(
            f"No stable pose on '{topic}' within {timeout_s:.1f}s "
            f"(got {n_fresh} fresh / {n_seen} total, "
            f"span dx={last_span[0]:.3f} dy={last_span[1]:.3f} "
            f"dz={last_span[2]:.3f} m; need {samples} samples, "
            f"age <= {max_age_s:.2f}s, xyz span <= {max_span_m:.3f} m). "
            "Restart detect_red_cube.py so /red_cube/hover_pose uses sensor QoS."
        )
    return stable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/red_cube/hover_pose")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Poses to median-filter once they agree (default 5).",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=1.0,
        help="Drop samples older than this many seconds (default 1.0).",
    )
    parser.add_argument(
        "--max-span",
        type=float,
        default=0.05,
        help="Max XYZ range (m) across the sample window (default 0.05; D405 Z is noisy).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a stable hover pose (default 10).",
    )
    parser.add_argument("--group", default="arm")
    parser.add_argument("--link", default="gripper")
    parser.add_argument("--vel", type=float, default=0.3)
    parser.add_argument("--accel", type=float, default=0.3)
    parser.add_argument("--position-radius", type=float, default=0.015)
    parser.add_argument("--orient-tol", type=float, default=0.5)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan but don't execute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.samples < 1:
        print("ERROR: --samples must be >= 1", file=sys.stderr)
        return 1

    rclpy.init()
    try:
        try:
            pose = wait_for_stable_hover(
                args.topic,
                samples=args.samples,
                max_age_s=args.max_age,
                max_span_m=args.max_span,
                timeout_s=args.timeout,
            )
        except TimeoutError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        p = pose.pose.position
        print(
            f"Stable hover in '{pose.header.frame_id}': "
            f"xyz=({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f})"
        )
        result = send_pose(
            pose,
            group=args.group,
            link_name=args.link,
            plan_only=args.plan_only,
            vel_scale=args.vel,
            accel_scale=args.accel,
            position_radius=args.position_radius,
            orient_tol=args.orient_tol,
            planning_time_s=args.planning_time,
            attempts=args.attempts,
        )
    finally:
        rclpy.shutdown()

    exit_code, message = format_moveit_result(result)
    print(f"\n{message}")
    if exit_code == 0 and not args.plan_only:
        print("Parked above the cube. Start the RL grasp policy from here.")
        print("Do not run teleop or Plan & Execute while the policy owns the arm.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
