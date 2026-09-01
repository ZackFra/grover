#!/usr/bin/env python3
"""Drive the SO-101 arm to a saved joint pose via MoveIt2 planning + execution.

Sends a MoveGroup action goal on /move_action with joint_constraints built
from a pose YAML written by scripts/snapshot_joints.py (or any YAML with a
top-level `joints:` mapping of joint_name -> radians).

The gripper is not in the ``arm`` planning group, so MoveIt never sees it.
If the YAML has ``gripper_joint``, this script also publishes that angle on
``/gripper_controller/commands`` (ForwardCommandController) after the arm
move. ``top_view`` is closed (~-0.18 rad); ``top_view_open`` is open (~1.68).

Usage:
    # With so101_bringup running:
    python3 scripts/moveit_goto_joints.py home
    python3 scripts/moveit_goto_joints.py top_view          # arm + close gripper
    python3 scripts/moveit_goto_joints.py top_view_open     # arm + open gripper
    python3 scripts/moveit_goto_joints.py home --plan-only   # preview, no execute
    python3 scripts/moveit_goto_joints.py home --vel 0.5 --accel 0.5
    python3 scripts/moveit_goto_joints.py path/to/custom.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import rclpy
import yaml
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    PlanningOptions,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSES_DIR = REPO_ROOT / "config" / "poses"

ARM_GROUP_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper_joint"
GRIPPER_COMMAND_TOPIC = "/gripper_controller/commands"

# Reverse-map MoveItErrorCodes constants (val -> name) for human-readable errors.
_MOVEIT_ERROR_NAMES: dict[int, str] = {
    int(getattr(MoveItErrorCodes, attr)): attr
    for attr in dir(MoveItErrorCodes)
    if not attr.startswith("_") and isinstance(getattr(MoveItErrorCodes, attr), int)
}


def load_pose(name_or_path: str) -> tuple[dict[str, float], Path]:
    """Resolve 'NAME' to config/poses/NAME.yaml, or accept an explicit path."""
    raw = Path(name_or_path)
    if raw.exists():
        path = raw
    else:
        path = DEFAULT_POSES_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Pose YAML not found: tried '{raw}' and '{path}'."
        )

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "joints" not in data:
        raise ValueError(f"Pose YAML missing top-level 'joints:' map: {path}")
    joints = data["joints"]
    if not isinstance(joints, dict) or not joints:
        raise ValueError(f"'joints' must be a non-empty mapping in {path}")
    return {str(n): float(p) for n, p in joints.items()}, path


def filter_joints(
    all_joints: dict[str, float], allow: Iterable[str]
) -> tuple[dict[str, float], list[str]]:
    allow_set = set(allow)
    kept = {n: p for n, p in all_joints.items() if n in allow_set}
    skipped = [n for n in all_joints if n not in allow_set]
    return kept, skipped


class MoveGroupGoaler(Node):
    """Tiny single-shot client for /move_action."""

    def __init__(self, group_name: str, action_name: str = "/move_action") -> None:
        super().__init__("moveit_goto_joints")
        self._group = group_name
        self._client = ActionClient(self, MoveGroup, action_name)
        self._action_name = action_name

    def send(
        self,
        joint_positions: dict[str, float],
        *,
        plan_only: bool,
        vel_scale: float,
        accel_scale: float,
        tolerance_rad: float,
        planning_time_s: float,
        attempts: int,
        server_wait_s: float = 10.0,
    ) -> MoveGroup.Result | None:
        constraints = Constraints()
        for name, pos in joint_positions.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = tolerance_rad
            jc.tolerance_below = tolerance_rad
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        request = MotionPlanRequest()
        request.group_name = self._group
        request.num_planning_attempts = attempts
        request.allowed_planning_time = planning_time_s
        request.max_velocity_scaling_factor = vel_scale
        request.max_acceleration_scaling_factor = accel_scale
        request.goal_constraints.append(constraints)

        options = PlanningOptions()
        options.plan_only = plan_only

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = options

        self.get_logger().info(
            f"Waiting for action server at {self._action_name} ..."
        )
        if not self._client.wait_for_server(timeout_sec=server_wait_s):
            self.get_logger().error(
                f"{self._action_name} not available within {server_wait_s:.1f}s. "
                "Is move_group running (so101_bringup)?"
            )
            return None

        self.get_logger().info(
            f"Sending {len(joint_positions)} joint constraints; "
            f"plan_only={plan_only}, vel={vel_scale}, accel={accel_scale}, "
            f"tol={tolerance_rad} rad, attempts={attempts}, planning_time={planning_time_s}s."
        )

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Goal rejected by move_group.")
            return None

        self.get_logger().info("Goal accepted; waiting for result ...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapper = result_future.result()
        return wrapper.result if wrapper is not None else None

    def send_gripper(self, position: float, *, settle_s: float = 0.4) -> None:
        """Command the FCC gripper (not in the MoveIt arm group)."""
        pub = self.create_publisher(Float64MultiArray, GRIPPER_COMMAND_TOPIC, 10)
        msg = Float64MultiArray()
        msg.data = [float(position)]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f"No subscriber on {GRIPPER_COMMAND_TOPIC}; gripper command may be dropped."
            )
        for _ in range(8):
            pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        t0 = time.monotonic()
        while time.monotonic() - t0 < settle_s:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().info(
            f"Gripper {GRIPPER_JOINT}={position:+.4f} rad -> {GRIPPER_COMMAND_TOPIC}"
        )


def execute_named_pose(
    pose: str,
    *,
    group: str = "arm",
    vel: float = 0.3,
    accel: float = 0.3,
    tolerance: float = 0.01,
    planning_time: float = 5.0,
    attempts: int = 10,
    plan_only: bool = False,
    joints: Iterable[str] | None = None,
    move_gripper: bool = True,
) -> int:
    """Plan/execute a named pose. Inits rclpy only if this process has not already."""
    try:
        all_joints, path = load_pose(pose)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    allow = tuple(joints) if joints else ARM_GROUP_JOINTS
    arm_positions, skipped = filter_joints(all_joints, allow)
    gripper_pos = all_joints.get(GRIPPER_JOINT) if move_gripper else None
    skipped = [n for n in skipped if not (n == GRIPPER_JOINT and gripper_pos is not None)]
    if not arm_positions:
        print(
            f"ERROR: no joints in {path} match the allowed set {sorted(allow)}. "
            f"YAML has: {sorted(all_joints)}",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded pose: {path.relative_to(REPO_ROOT) if path.is_absolute() else path}")
    print(f"  group: {group}")
    print(f"  constraining {len(arm_positions)} joint(s):")
    for name in sorted(arm_positions):
        print(f"    {name:14s} {arm_positions[name]:+.6f}")
    if gripper_pos is not None:
        print(
            f"  gripper {GRIPPER_JOINT}={gripper_pos:+.6f} "
            f"via {GRIPPER_COMMAND_TOPIC}"
            + (" (skipped, --plan-only)" if plan_only else "")
        )
    if skipped:
        print(f"  skipping {len(skipped)} joint(s) not in group: {sorted(skipped)}")
    print()

    own_ctx = not rclpy.ok()
    if own_ctx:
        rclpy.init()
    try:
        node = MoveGroupGoaler(group_name=group)
        result = node.send(
            arm_positions,
            plan_only=plan_only,
            vel_scale=vel,
            accel_scale=accel,
            tolerance_rad=tolerance,
            planning_time_s=planning_time,
            attempts=attempts,
        )
        arm_ok = (
            result is not None
            and int(result.error_code.val) == int(MoveItErrorCodes.SUCCESS)
        )
        if arm_ok and gripper_pos is not None and not plan_only:
            node.send_gripper(gripper_pos)
        node.destroy_node()
    finally:
        if own_ctx:
            rclpy.shutdown()

    if result is None:
        return 2

    code = int(result.error_code.val)
    name = _MOVEIT_ERROR_NAMES.get(code, f"UNKNOWN({code})")
    if code == int(MoveItErrorCodes.SUCCESS):
        print(f"\nMoveIt result: SUCCESS ({code})")
        return 0
    print(f"\nMoveIt result: {name} ({code})  -- see moveit_msgs/MoveItErrorCodes")
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pose",
        help="Pose name (config/poses/NAME.yaml) or path to a YAML file.",
    )
    parser.add_argument(
        "--group",
        default="arm",
        help="MoveIt planning group (default: arm).",
    )
    parser.add_argument(
        "--vel",
        type=float,
        default=0.3,
        help="max_velocity_scaling_factor in [0, 1] (default 0.3).",
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=0.3,
        help="max_acceleration_scaling_factor in [0, 1] (default 0.3).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Per-joint tolerance (radians) on the goal constraint (default 0.01).",
    )
    parser.add_argument(
        "--planning-time",
        type=float,
        default=5.0,
        help="MoveIt allowed_planning_time, seconds (default 5.0).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="num_planning_attempts (default 10).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan but don't execute (useful for previewing in RViz).",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=None,
        help=(
            "Override the joints constrained from the YAML (default: the 5 "
            "joints of the 'arm' planning group). gripper_joint is sent on "
            "/gripper_controller/commands unless --no-gripper."
        ),
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Do not command gripper_joint even if it is in the YAML.",
    )
    args = parser.parse_args()
    return execute_named_pose(
        args.pose,
        group=args.group,
        vel=args.vel,
        accel=args.accel,
        tolerance=args.tolerance,
        planning_time=args.planning_time,
        attempts=args.attempts,
        plan_only=args.plan_only,
        joints=args.joints,
        move_gripper=not args.no_gripper,
    )


if __name__ == "__main__":
    raise SystemExit(main())
