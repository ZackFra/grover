#!/usr/bin/env bash
# Quick checks while bringup + teleop are running (source ROS + workspace first).
set -euo pipefail
source /opt/ros/jazzy/setup.bash
source "$(cd "$(dirname "$0")/.." && pwd)/install/setup.bash"

echo "=== Servo status (code 0 = OK, 5 = HALT_FOR_COLLISION) ==="
timeout 2 ros2 topic echo /servo_node/status --once 2>/dev/null || echo "(no message yet)"

echo ""
echo "=== Joint jog input rate ==="
timeout 3 ros2 topic hz /servo_node/delta_joint_cmds 2>/dev/null || echo "(no joint jog published)"

echo ""
echo "=== Arm command output rate ==="
timeout 3 ros2 topic hz /arm_controller/commands 2>/dev/null || echo "(no arm commands from servo)"

echo ""
echo "=== Switch command type (0=JOINT_JOG) ==="
ros2 service type /servo_node/switch_command_type 2>/dev/null || true
