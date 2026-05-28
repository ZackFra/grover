import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    follower_serial_port_arg = DeclareLaunchArgument(
        "follower_serial_port",
        default_value="/dev/so101_follower",
        description="Feetech follower serial port (xacro usb_port).",
    )
    start_trajectory_controllers_arg = DeclareLaunchArgument(
        "start_trajectory_controllers",
        default_value="true",
        description=(
            "If false, only joint_state_broadcaster is loaded — no arm/gripper "
            "controllers holding position (less stiff while debugging offsets / RViz)."
        ),
    )
    disable_servo_torque_arg = DeclareLaunchArgument(
        "disable_servo_torque",
        default_value="false",
        description=(
            "If true, URDF uses passive Feetech joints (no position commands); "
            "feetech_ros2_driver disables servo torque on init. Arm/gripper controllers are not started."
        ),
    )
    follower_serial_port = LaunchConfiguration("follower_serial_port")

    urdf_xacro = os.path.join(
        get_package_share_directory("lerobot_description"),
        "urdf",
        "so101.urdf.xacro",
    )
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                urdf_xacro,
                " use_sim:=false usb_port:=",
                follower_serial_port,
                " hardware_passive:=",
                LaunchConfiguration("disable_servo_torque"),
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description, "use_sim_time": False},
            os.path.join(
                get_package_share_directory("lerobot_controller"),
                "config",
                "so101_controllers_hw.yaml",
            ),
        ],
    )

    def register_conditional_spawners(context):
        tc = (
            LaunchConfiguration("start_trajectory_controllers")
            .perform(context)
            .strip()
            .lower()
        )
        dt = (
            LaunchConfiguration("disable_servo_torque")
            .perform(context)
            .strip()
            .lower()
        )
        torque_off = dt in ("true", "1", "yes", "on")
        start_tc = (tc in ("true", "1", "yes", "on")) and not torque_off

        joint_state_broadcaster_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "--controller-manager",
                "/controller_manager",
            ],
            output="screen",
        )
        arm_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["arm_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        )
        gripper_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        )

        # Spawn sequentially — parallel spawners race and try to re-configure
        # controllers that are already active ("can not be configured from active").
        actions = [
            RegisterEventHandler(
                OnProcessStart(
                    target_action=controller_manager,
                    on_start=[joint_state_broadcaster_spawner],
                )
            ),
        ]
        if start_tc:
            fcc_adapter = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("lerobot_controller"),
                        "launch",
                        "fcc_trajectory_adapter.launch.py",
                    )
                ),
                launch_arguments={"update_rate_hz": "100.0"}.items(),
            )
            actions.extend(
                [
                    RegisterEventHandler(
                        OnProcessExit(
                            target_action=joint_state_broadcaster_spawner,
                            on_exit=[arm_controller_spawner],
                        )
                    ),
                    RegisterEventHandler(
                        OnProcessExit(
                            target_action=arm_controller_spawner,
                            on_exit=[gripper_controller_spawner],
                        )
                    ),
                    RegisterEventHandler(
                        OnProcessExit(
                            target_action=gripper_controller_spawner,
                            on_exit=[fcc_adapter],
                        )
                    ),
                ]
            )

        return actions

    return LaunchDescription(
        [
            follower_serial_port_arg,
            start_trajectory_controllers_arg,
            disable_servo_torque_arg,
            robot_state_publisher_node,
            controller_manager,
            OpaqueFunction(function=register_conditional_spawners),
        ]
    )
