import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    follower_serial_port_arg = DeclareLaunchArgument(
        "follower_serial_port",
        default_value="/dev/so101_follower",
        description="Feetech follower serial port used for mandatory pre-configure.",
    )
    start_trajectory_controllers_arg = DeclareLaunchArgument(
        "start_trajectory_controllers",
        default_value="true",
        description=(
            "If false, only joint_state_broadcaster is loaded — no arm/gripper trajectory "
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
            {"use_sim_time": False},
            os.path.join(
                get_package_share_directory("lerobot_controller"),
                "config",
                "so101_controllers.yaml",
            ),
        ],
    )

    configure_feetech_bus = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "lerobot_robot_ros.so101_follower_bus_configure",
            follower_serial_port,
        ],
        output="screen",
    )

    start_controller_manager_after_config = RegisterEventHandler(
        OnProcessExit(
            target_action=configure_feetech_bus,
            on_exit=[controller_manager],
        )
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
        )
        arm_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        )
        gripper_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        )

        on_start = [joint_state_broadcaster_spawner]
        if start_tc:
            on_start.extend([arm_controller_spawner, gripper_controller_spawner])

        return [
            RegisterEventHandler(
                OnProcessStart(
                    target_action=controller_manager,
                    on_start=on_start,
                )
            )
        ]

    return LaunchDescription(
        [
            follower_serial_port_arg,
            start_trajectory_controllers_arg,
            disable_servo_torque_arg,
            robot_state_publisher_node,
            configure_feetech_bus,
            start_controller_manager_after_config,
            OpaqueFunction(function=register_conditional_spawners),
        ]
    )
