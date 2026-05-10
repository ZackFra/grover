lerobot-teleoperate \
    --robot.type=so101_ros \
    --robot.id=my_follower \
    --robot.convert_so101_leader_units=true \
    --teleop.type=so101_leader \
    --teleop.id=gi_joe \
    --teleop.port=/dev/so101_leader \
    --display_data=true