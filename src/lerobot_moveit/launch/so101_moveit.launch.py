import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

    is_sim = LaunchConfiguration("is_sim")
    follower_serial_port = LaunchConfiguration("follower_serial_port")

    lerobot_description_dir = get_package_share_directory("lerobot_description")
    so101_urdf_path = os.path.join(lerobot_description_dir, "urdf", "so101.urdf.xacro")

    moveit_config = (
        MoveItConfigsBuilder("so101", package_name="lerobot_moveit")
        .robot_description(
            file_path=so101_urdf_path,
            mappings={
                "use_sim": is_sim,
                "usb_port": follower_serial_port,
            },
        )
        .robot_description_semantic(file_path="config/so101.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": is_sim},
            {"publish_robot_description_semantic": True},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_config_path = os.path.join(
        get_package_share_directory("lerobot_moveit"),
        "config",
        "moveit.rviz",
    )

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
        ],
    )

    return LaunchDescription(
        [
            is_sim_arg,
            follower_serial_port_arg,
            move_group_node,
            rviz_node,
        ]
    )
