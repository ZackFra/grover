from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    update_rate_hz_arg = DeclareLaunchArgument(
        "update_rate_hz",
        default_value="50.0",
        description="Trajectory sampling rate published to FCC command topics.",
    )

    adapter_node = Node(
        package="lerobot_controller",
        executable="fcc_trajectory_adapter.py",
        name="fcc_trajectory_adapter",
        output="screen",
        parameters=[
            {"update_rate_hz": LaunchConfiguration("update_rate_hz")},
        ],
    )

    return LaunchDescription([update_rate_hz_arg, adapter_node])
