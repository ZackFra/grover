import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    controller_manager_name_arg = DeclareLaunchArgument(
        "controller_manager_name",
        default_value="/controller_manager",
        description="Controller manager service name (Gazebo exposes /controller_manager in sim).",
    )
    controller_manager_name = LaunchConfiguration("controller_manager_name")

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            controller_manager_name,
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", controller_manager_name],
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", controller_manager_name],
    )

    # MoveIt Plan & Execute needs FollowJointTrajectory actions; FCC only has
    # command topics. Start adapter after controllers have had time to load.
    fcc_adapter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("lerobot_controller"),
                "launch",
                "fcc_trajectory_adapter.launch.py",
            )
        ),
        launch_arguments={"update_rate_hz": "50.0"}.items(),
    )
    fcc_adapter_delayed = TimerAction(period=6.0, actions=[fcc_adapter])

    return LaunchDescription(
        [
            controller_manager_name_arg,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
            fcc_adapter_delayed,
        ]
    )