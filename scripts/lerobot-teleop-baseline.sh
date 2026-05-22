#!/usr/bin/env bash
# HW teleop: joint_jog → MoveIt Servo (collision checking) → /arm_controller/commands
# For smoother motion, run without camera preview: DISPLAY_DATA=false ./scripts/lerobot-teleop-baseline.sh
DISPLAY_DATA="${DISPLAY_DATA:-true}"

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.id=gi_jane \
  --robot.port=/dev/so101_follower \
  --teleop.type=so101_leader \
  --teleop.id=gi_joe \
  --teleop.port=/dev/so101_leader \
  --display_data="${DISPLAY_DATA}"
