import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="True",
        description="Run in simulation mode. Set false for real hardware.",
    )
    follower_serial_port_arg = DeclareLaunchArgument(
        "follower_serial_port",
        default_value="/dev/so101_follower",
        description="Follower serial device used by mandatory pre-configure on real hardware.",
    )

    is_sim = LaunchConfiguration("is_sim")
    follower_serial_port = LaunchConfiguration("follower_serial_port")

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("lerobot_moveit"),
                "launch",
                "so101_moveit.launch.py",
            )
        ),
        launch_arguments={
            "is_sim": is_sim,
            "follower_serial_port": follower_serial_port,
        }.items(),
    )

    def _select_controller_launch(context):
        sim_value = str(is_sim.perform(context)).strip().lower()
        is_sim_mode = sim_value in ("1", "true", "yes", "on")
        if is_sim_mode:
            return [
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(
                            get_package_share_directory("lerobot_description"),
                            "launch",
                            "so101_gazebo.launch.py",
                        )
                    )
                ),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(
                            get_package_share_directory("lerobot_controller"),
                            "launch",
                            "so101_controller.launch.py",
                        )
                    )
                ),
            ]
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("lerobot_controller"),
                        "launch",
                        "so101_controller_hw.launch.py",
                    )
                ),
                launch_arguments={"follower_serial_port": follower_serial_port}.items(),
            )
        ]

    return LaunchDescription(
        [
            is_sim_arg,
            follower_serial_port_arg,
            OpaqueFunction(function=_select_controller_launch),
            moveit_launch,
        ]
    )
