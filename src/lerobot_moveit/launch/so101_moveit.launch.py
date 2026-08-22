import os

import yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    is_sim_arg = DeclareLaunchArgument(name="is_sim", default_value="True")
    follower_serial_port_arg = DeclareLaunchArgument(
        name="follower_serial_port",
        default_value="/dev/so101_follower",
        description="Feetech USB device path (passed to xacro usb_port when not sim).",
    )
    disable_servo_torque_arg = DeclareLaunchArgument(
        name="disable_servo_torque",
        default_value="false",
        description="Must match hardware launch: passive ros2_control joints / torque off.",
    )
    enable_octomap_arg = DeclareLaunchArgument(
        name="enable_octomap",
        default_value="true",
        description=(
            "When true, move_group loads PointCloudOctomapUpdater from sensors_3d.yaml "
            "and publishes the occupancy map on the monitored planning scene."
        ),
    )

    def launch_moveit_rviz_servo(context):
        """move_group + RViz + servo_node (joint jog teleop via lerobot_ros action_type=joint_jog)."""
        use_sim_time = (
            LaunchConfiguration("is_sim").perform(context).strip().lower() == "true"
        )
        is_sim_mode = use_sim_time

        follower_serial_port = LaunchConfiguration("follower_serial_port")
        use_octomap = (
            LaunchConfiguration("enable_octomap").perform(context).strip().lower()
            == "true"
        )

        pkg_share = get_package_share_directory("lerobot_moveit")
        lerobot_description_dir = get_package_share_directory("lerobot_description")
        so101_urdf_path = os.path.join(lerobot_description_dir, "urdf", "so101.urdf.xacro")

        # MoveItConfigsBuilder ignores sensors: [] and to_moveit_configs() auto-loads
        # sensors_3d.yaml. An empty sensors list also breaks launch_ros ([] -> ()).
        # When octomap is off: block auto-load, then omit sensor keys from node params.
        builder = (
            MoveItConfigsBuilder("so101", package_name="lerobot_moveit")
            .robot_description(
                file_path=so101_urdf_path,
                mappings={
                    "use_sim": LaunchConfiguration("is_sim"),
                    "usb_port": follower_serial_port,
                    "hardware_passive": LaunchConfiguration("disable_servo_torque"),
                    # HW: optical frames come from camera_pose.launch.py, not the URDF.
                    "d405_use_nominal_extrinsics": "true" if is_sim_mode else "false",
                },
            )
            .robot_description_semantic(file_path="config/so101.srdf")
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .planning_pipelines(
                default_planning_pipeline="ompl",
                pipelines=["ompl"],
                load_all=False,
            )
        )
        if use_octomap:
            builder = builder.sensors_3d(file_path="config/sensors_3d.yaml")
        else:
            builder._MoveItConfigsBuilder__moveit_configs.sensors_3d = {
                "_octomap_disabled": True
            }
        moveit_config = builder.to_moveit_configs()
        move_group_params = moveit_config.to_dict()
        if not use_octomap:
            move_group_params.pop("sensors", None)
            move_group_params.pop("d405_wrist_pointcloud", None)
            move_group_params.pop("_octomap_disabled", None)

        servo_yaml_name = "so101_servo_sim.yaml" if is_sim_mode else "so101_servo.yaml"
        with open(os.path.join(pkg_share, "config", servo_yaml_name)) as f:
            servo_yaml = yaml.safe_load(f)
        servo_params = {"moveit_servo": servo_yaml}
        # Required by online_signal_smoothing plugins (Butterworth / AccelerationLimited).
        acceleration_filter_update_period = {
            "update_period": float(servo_yaml["publish_period"]),
        }
        planning_group_name = {"planning_group_name": servo_yaml["move_group_name"]}

        move_group_node_params = [
            move_group_params,
            {"use_sim_time": use_sim_time},
            {"publish_robot_description_semantic": True},
        ]
        if use_octomap:
            move_group_node_params.insert(
                1,
                {
                    "octomap_frame": "base",
                    "octomap_resolution": 0.025,
                    "max_range": 3.0,
                },
            )
        move_group_node = Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=move_group_node_params,
            arguments=["--ros-args", "--log-level", "info"],
        )

        servo_node = Node(
            package="moveit_servo",
            executable="servo_node",
            output="screen",
            parameters=[
                servo_params,
                acceleration_filter_update_period,
                planning_group_name,
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
                {"use_sim_time": use_sim_time},
            ],
        )

        rviz_config_path = os.path.join(pkg_share, "config", "moveit.rviz")
        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config_path],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
                {"use_sim_time": use_sim_time},
            ],
        )

        return [move_group_node, servo_node, rviz_node]

    return LaunchDescription(
        [
            is_sim_arg,
            follower_serial_port_arg,
            disable_servo_torque_arg,
            enable_octomap_arg,
            OpaqueFunction(function=launch_moveit_rviz_servo),
        ]
    )
