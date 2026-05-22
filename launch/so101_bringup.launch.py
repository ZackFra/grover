import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

_CONFIGURE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "configure_so101_follower_bus.py",
)


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
    start_trajectory_controllers_arg = DeclareLaunchArgument(
        "start_trajectory_controllers",
        default_value="true",
        description=(
            "Hardware only: set false to skip arm/gripper trajectory controllers "
            "(joint_state_broadcaster only — less stiff holding while debugging)."
        ),
    )
    disable_servo_torque_arg = DeclareLaunchArgument(
        "disable_servo_torque",
        default_value="false",
        description=(
            "Hardware only: true = Feetech torque off (passive joints), for posing/calibration by hand."
        ),
    )

    is_sim = LaunchConfiguration("is_sim")
    follower_serial_port = LaunchConfiguration("follower_serial_port")
    start_trajectory_controllers = LaunchConfiguration("start_trajectory_controllers")
    disable_servo_torque = LaunchConfiguration("disable_servo_torque")

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
            "disable_servo_torque": disable_servo_torque,
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

        port = follower_serial_port.perform(context).strip()
        configure_feetech = ExecuteProcess(
            cmd=["python3", _CONFIGURE_SCRIPT, port],
            output="screen",
        )
        hw_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("lerobot_controller"),
                    "launch",
                    "so101_controller_hw.launch.py",
                )
            ),
            launch_arguments={
                "follower_serial_port": follower_serial_port,
                "start_trajectory_controllers": start_trajectory_controllers,
                "disable_servo_torque": disable_servo_torque,
            }.items(),
        )
        return [
            configure_feetech,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=configure_feetech,
                    on_exit=[hw_launch],
                )
            ),
        ]

    return LaunchDescription(
        [
            is_sim_arg,
            follower_serial_port_arg,
            start_trajectory_controllers_arg,
            disable_servo_torque_arg,
            OpaqueFunction(function=_select_controller_launch),
            moveit_launch,
        ]
    )
