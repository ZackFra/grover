#!/usr/bin/env python3
"""Drive the SO-101 gripper to a Cartesian pose via MoveIt2.

Sends a MoveGroup action goal on /move_action with a position sphere on the
`gripper` link plus a loose orientation constraint. Keep orientation from the
live top-down pose (e.g. copied onto /red_cube/hover_pose by detect_red_cube.py);
the arm is 5-DOF so a full 6-DOF IK goal often fails.

Usage:
    # With so101_bringup running, detector publishing hover:
    python3 scripts/moveit_goto_pose.py --topic /red_cube/hover_pose
    python3 scripts/moveit_goto_pose.py --topic /red_cube/hover_pose --plan-only
    python3 scripts/moveit_goto_pose.py --frame world --xyz 0.20 0.00 0.15
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Sequence

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

# Reverse-map MoveItErrorCodes constants (val -> name) for human-readable errors.
_MOVEIT_ERROR_NAMES: dict[int, str] = {
    int(getattr(MoveItErrorCodes, attr)): attr
    for attr in dir(MoveItErrorCodes)
    if not attr.startswith("_") and isinstance(getattr(MoveItErrorCodes, attr), int)
}


def format_moveit_result(result: MoveGroup.Result | None) -> tuple[int, str]:
    """Return (process_exit_code, message) for a MoveGroup result."""
    if result is None:
        return 2, "No result from move_group (server missing or goal rejected)."
    code = int(result.error_code.val)
    name = _MOVEIT_ERROR_NAMES.get(code, f"UNKNOWN({code})")
    if code == int(MoveItErrorCodes.SUCCESS):
        return 0, f"MoveIt result: SUCCESS ({code})"
    return 3, f"MoveIt result: {name} ({code})  -- see moveit_msgs/MoveItErrorCodes"


def wait_for_pose(topic: str, timeout_s: float) -> PoseStamped:
    """Spin until one PoseStamped arrives on `topic`."""
    node = Node("moveit_goto_pose_wait")
    received: list[PoseStamped] = []

    def _cb(msg: PoseStamped) -> None:
        received.append(msg)

    node.create_subscription(PoseStamped, topic, _cb, 10)
    deadline = time.monotonic() + timeout_s
    try:
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()

    if not received:
        raise TimeoutError(
            f"No PoseStamped on '{topic}' within {timeout_s:.1f}s. "
            "Is detect_red_cube.py running and seeing the cube?"
        )
    return received[0]


def pose_from_xyz_quat(
    frame: str,
    xyz: Sequence[float],
    quat_xyzw: Sequence[float] | None,
) -> PoseStamped:
    if len(xyz) != 3:
        raise ValueError("--xyz needs three numbers: x y z")
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = float(xyz[0])
    msg.pose.position.y = float(xyz[1])
    msg.pose.position.z = float(xyz[2])
    if quat_xyzw is None:
        msg.pose.orientation.w = 1.0
        return msg
    if len(quat_xyzw) != 4:
        raise ValueError("--quat needs four numbers: x y z w")
    msg.pose.orientation.x = float(quat_xyzw[0])
    msg.pose.orientation.y = float(quat_xyzw[1])
    msg.pose.orientation.z = float(quat_xyzw[2])
    msg.pose.orientation.w = float(quat_xyzw[3])
    return msg


def build_pose_constraints(
    pose: PoseStamped,
    *,
    link_name: str,
    position_radius: float,
    orient_tol: float,
) -> Constraints:
    constraints = Constraints()
    constraints.name = "gripper_hover"

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [float(position_radius)]

    region_pose = Pose()
    region_pose.position = pose.pose.position
    region_pose.orientation.w = 1.0

    volume = BoundingVolume()
    volume.primitives.append(sphere)
    volume.primitive_poses.append(region_pose)

    position = PositionConstraint()
    position.header = pose.header
    position.link_name = link_name
    position.constraint_region = volume
    position.weight = 1.0
    constraints.position_constraints.append(position)

    # 5-DOF: skip orientation unless the caller asked for a finite tolerance.
    # KDL still won't match a 6-DOF quat; a tight OrientationConstraint rejects
    # every position-only IK sample.
    if float(orient_tol) > 0.0:
        orientation = OrientationConstraint()
        orientation.header = pose.header
        orientation.link_name = link_name
        orientation.orientation = pose.pose.orientation
        # Keep the gripper pointing the same way (down from top_view). Yaw about
        # world Z is free — the arm is 5-DOF and cannot hit a full 6-DOF quat.
        orientation.absolute_x_axis_tolerance = float(orient_tol)
        orientation.absolute_y_axis_tolerance = float(orient_tol)
        orientation.absolute_z_axis_tolerance = math.pi
        if hasattr(OrientationConstraint, "XYZ_EULER_ANGLES"):
            orientation.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
        orientation.weight = 1.0
        constraints.orientation_constraints.append(orientation)
    return constraints


class MoveGroupPoseGoaler(Node):
    """Single-shot /move_action client for a gripper pose."""

    def __init__(self, group_name: str, action_name: str = "/move_action") -> None:
        super().__init__("moveit_goto_pose")
        self._group = group_name
        self._action_name = action_name
        self._client = ActionClient(self, MoveGroup, action_name)

    def send(
        self,
        pose: PoseStamped,
        *,
        link_name: str,
        plan_only: bool,
        vel_scale: float,
        accel_scale: float,
        position_radius: float,
        orient_tol: float,
        planning_time_s: float,
        attempts: int,
        extra_joint_constraints: list[JointConstraint] | None = None,
        server_wait_s: float = 10.0,
    ) -> MoveGroup.Result | None:
        request = MotionPlanRequest()
        request.group_name = self._group
        request.num_planning_attempts = attempts
        request.allowed_planning_time = planning_time_s
        request.max_velocity_scaling_factor = vel_scale
        request.max_acceleration_scaling_factor = accel_scale
        request.start_state.is_diff = True
        goal_constraints = build_pose_constraints(
            pose,
            link_name=link_name,
            position_radius=position_radius,
            orient_tol=orient_tol,
        )
        if extra_joint_constraints:
            for jc in extra_joint_constraints:
                goal_constraints.joint_constraints.append(jc)
                path_jc = JointConstraint()
                path_jc.joint_name = jc.joint_name
                path_jc.position = jc.position
                path_jc.tolerance_above = jc.tolerance_above
                path_jc.tolerance_below = jc.tolerance_below
                path_jc.weight = jc.weight
                request.path_constraints.joint_constraints.append(path_jc)
            request.path_constraints.name = "hold_wrist"
        request.goal_constraints.append(goal_constraints)

        options = PlanningOptions()
        options.plan_only = plan_only

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = options

        self.get_logger().info(f"Waiting for action server at {self._action_name} ...")
        if not self._client.wait_for_server(timeout_sec=server_wait_s):
            self.get_logger().error(
                f"{self._action_name} not available within {server_wait_s:.1f}s. "
                "Is move_group running (so101_bringup)?"
            )
            return None

        p = pose.pose.position
        q = pose.pose.orientation
        self.get_logger().info(
            f"Goal {link_name} in '{pose.header.frame_id}': "
            f"xyz=({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f}) "
            f"quat=({q.x:+.3f}, {q.y:+.3f}, {q.z:+.3f}, {q.w:+.3f}); "
            f"pos_r={position_radius:.3f} m, orient_tol={orient_tol:.2f} rad, "
            f"wrist_holds={len(extra_joint_constraints or [])}, "
            f"plan_only={plan_only}"
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


def send_pose(
    pose: PoseStamped,
    *,
    group: str = "arm",
    link_name: str = "gripper",
    plan_only: bool = False,
    vel_scale: float = 0.3,
    accel_scale: float = 0.3,
    position_radius: float = 0.03,
    orient_tol: float = 0.0,
    planning_time_s: float = 5.0,
    attempts: int = 10,
    extra_joint_constraints: list[JointConstraint] | None = None,
) -> MoveGroup.Result | None:
    """Create a goaler node, send `pose`, destroy the node. Caller owns rclpy.init."""
    node = MoveGroupPoseGoaler(group_name=group)
    try:
        return node.send(
            pose,
            link_name=link_name,
            plan_only=plan_only,
            vel_scale=vel_scale,
            accel_scale=accel_scale,
            position_radius=position_radius,
            orient_tol=orient_tol,
            planning_time_s=planning_time_s,
            attempts=attempts,
            extra_joint_constraints=extra_joint_constraints,
        )
    finally:
        node.destroy_node()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default=None,
        help="Wait for one PoseStamped on this topic (e.g. /red_cube/hover_pose).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for --topic (default 5.0).",
    )
    parser.add_argument("--frame", default="world", help="Used with --xyz.")
    parser.add_argument(
        "--xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="One-shot target position in --frame (meters).",
    )
    parser.add_argument(
        "--quat",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "W"),
        default=None,
        help="Optional orientation (default identity). Prefer --topic from the detector.",
    )
    parser.add_argument("--group", default="arm")
    parser.add_argument("--link", default="gripper", help="MoveIt tip link (default gripper).")
    parser.add_argument("--vel", type=float, default=0.3)
    parser.add_argument("--accel", type=float, default=0.3)
    parser.add_argument(
        "--position-radius",
        type=float,
        default=0.03,
        help="PositionConstraint sphere radius in meters (default 0.03).",
    )
    parser.add_argument(
        "--orient-tol",
        type=float,
        default=0.0,
        help="Per-axis orientation tolerance in radians; 0 skips the constraint (default, required for 5-DOF IK).",
    )
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan but don't execute (useful for previewing in RViz).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if (args.topic is None) == (args.xyz is None):
        print("ERROR: pass exactly one of --topic or --xyz.", file=sys.stderr)
        return 1

    rclpy.init()
    try:
        if args.topic:
            try:
                pose = wait_for_pose(args.topic, args.timeout)
            except TimeoutError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        else:
            pose = pose_from_xyz_quat(args.frame, args.xyz, args.quat)

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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
