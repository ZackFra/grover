#!/usr/bin/env python3
"""Expose FollowJointTrajectory actions for MoveIt on ForwardCommandController arms.

MoveIt Plan & Execute expects `arm_controller/follow_joint_trajectory` (and the
gripper equivalent). ros2_control FCC only exposes `/arm_controller/commands`
(Float64MultiArray). This node bridges the two so trajectories can execute
without JointTrajectoryController (which segfaults on Jazzy for this robot).
"""

from __future__ import annotations

import threading
from typing import Iterable, List, Optional, Sequence

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint


def _duration_sec(point: JointTrajectoryPoint) -> float:
    return float(point.time_from_start.sec) + point.time_from_start.nanosec * 1e-9


def _sample_positions(
    points: Sequence[JointTrajectoryPoint], elapsed_sec: float
) -> Optional[List[float]]:
    if not points:
        return None

    if elapsed_sec <= _duration_sec(points[0]):
        return list(points[0].positions)

    for idx in range(len(points) - 1):
        t0 = _duration_sec(points[idx])
        t1 = _duration_sec(points[idx + 1])
        if elapsed_sec <= t1:
            if t1 <= t0:
                return list(points[idx + 1].positions)
            alpha = (elapsed_sec - t0) / (t1 - t0)
            p0 = points[idx].positions
            p1 = points[idx + 1].positions
            return [a + alpha * (b - a) for a, b in zip(p0, p1)]

    return list(points[-1].positions)


def _reorder_positions(
    goal_joint_names: Iterable[str],
    goal_positions: Sequence[float],
    expected_joint_names: Sequence[str],
) -> Optional[List[float]]:
    mapping = {name: pos for name, pos in zip(goal_joint_names, goal_positions)}
    try:
        return [float(mapping[name]) for name in expected_joint_names]
    except KeyError:
        return None


class _JointGroupAdapter:
    def __init__(
        self,
        node: Node,
        *,
        action_name: str,
        command_topic: str,
        joint_names: Sequence[str],
        update_rate_hz: float,
        callback_group: ReentrantCallbackGroup,
    ) -> None:
        self._node = node
        self._joint_names = list(joint_names)
        self._command_topic = command_topic
        self._update_rate_hz = update_rate_hz
        self._publisher = node.create_publisher(Float64MultiArray, command_topic, 10)
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._active_goal: Optional[object] = None

        self._action_server = ActionServer(
            node,
            FollowJointTrajectory,
            action_name,
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=callback_group,
        )

        node.get_logger().info(
            f"FCC adapter: {action_name} -> {command_topic} "
            f"({', '.join(self._joint_names)})"
        )

    def _goal_callback(self, goal_request) -> GoalResponse:
        traj = goal_request.trajectory
        if not traj.joint_names or not traj.points:
            self._node.get_logger().warn(
                f"Rejecting empty trajectory on {self._command_topic}"
            )
            return GoalResponse.REJECT

        positions = _reorder_positions(
            traj.joint_names, traj.points[0].positions, self._joint_names
        )
        if positions is None:
            self._node.get_logger().warn(
                f"Rejecting trajectory with unexpected joints "
                f"{traj.joint_names}; expected {self._joint_names}"
            )
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _publish(self, positions: Sequence[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(v) for v in positions]
        self._publisher.publish(msg)

    def _execute(self, goal_handle):
        with self._lock:
            self._cancel_requested = False
            self._active_goal = goal_handle

        traj = goal_handle.request.trajectory
        points = traj.points
        duration = _duration_sec(points[-1])
        clock = self._node.get_clock()
        start = clock.now()
        rate = self._node.create_rate(self._update_rate_hz)

        self._node.get_logger().info(
            f"Executing {len(points)} waypoint(s) over {duration:.2f}s on "
            f"{self._command_topic}"
        )

        result = FollowJointTrajectory.Result()
        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                    result.error_string = "Canceled"
                    return result

                elapsed = (clock.now() - start).nanoseconds * 1e-9
                sampled = _sample_positions(points, elapsed)
                if sampled is None:
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                    result.error_string = "Empty trajectory"
                    return result

                ordered = _reorder_positions(
                    traj.joint_names, sampled, self._joint_names
                )
                if ordered is None:
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
                    result.error_string = "Joint name mismatch"
                    return result

                self._publish(ordered)

                if elapsed >= duration:
                    self._publish(ordered)
                    goal_handle.succeed()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    result.error_string = ""
                    return result

                rate.sleep()
        finally:
            with self._lock:
                self._active_goal = None

        goal_handle.abort()
        result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
        result.error_string = "Node shutdown"
        return result


class FccTrajectoryAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("fcc_trajectory_adapter")

        self.declare_parameter("update_rate_hz", 50.0)
        self.declare_parameter(
            "arm_joint_names",
            [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
        )
        self.declare_parameter("gripper_joint_names", ["gripper_joint"])
        self.declare_parameter("arm_action_name", "arm_controller/follow_joint_trajectory")
        self.declare_parameter(
            "gripper_action_name", "gripper_controller/follow_joint_trajectory"
        )
        self.declare_parameter("arm_command_topic", "/arm_controller/commands")
        self.declare_parameter("gripper_command_topic", "/gripper_controller/commands")

        update_rate_hz = float(self.get_parameter("update_rate_hz").value)
        callback_group = ReentrantCallbackGroup()

        self._arm = _JointGroupAdapter(
            self,
            action_name=str(self.get_parameter("arm_action_name").value),
            command_topic=str(self.get_parameter("arm_command_topic").value),
            joint_names=list(self.get_parameter("arm_joint_names").value),
            update_rate_hz=update_rate_hz,
            callback_group=callback_group,
        )
        self._gripper = _JointGroupAdapter(
            self,
            action_name=str(self.get_parameter("gripper_action_name").value),
            command_topic=str(self.get_parameter("gripper_command_topic").value),
            joint_names=list(self.get_parameter("gripper_joint_names").value),
            update_rate_hz=update_rate_hz,
            callback_group=callback_group,
        )


def main() -> None:
    rclpy.init()
    node = FccTrajectoryAdapterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
