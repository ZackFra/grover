lerobot-teleoperate \
  --robot.type=so101_ros \
  --robot.id=gi_jane \
  --robot.action_type=joint_position \
  --robot.convert_so101_leader_units=true \
  --teleop.type=so101_leader \
  --teleop.id=gi_joe \
  --teleop.port=/dev/so101_leader \
  --display_data=true
