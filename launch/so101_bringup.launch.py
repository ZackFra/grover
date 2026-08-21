import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGURE_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "configure_so101_follower_bus.py")
_CAMERA_POSE_LAUNCH = os.path.join(
    _REPO_ROOT, "config", "realsense-d405", "camera_pose.launch.py"
)

# Stagger HW bringup: Feetech + ros2_control first, then D405, then MoveIt.
# MoveIt must start after joint_states + camera are up so the octomap MessageFilter
# (queue=5) can resolve world <- d405_wrist_*_optical_frame at cloud stamps.
_D405_DELAY_S = 2.0
_MOVEIT_DELAY_S = 8.0


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
            "Hardware only: set false to skip arm/gripper controllers "
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
    enable_octomap_arg = DeclareLaunchArgument(
        "enable_octomap",
        default_value="true",
        description=(
            "When true, move_group loads PointCloudOctomapUpdater (occupancy on "
            "/monitored_planning_scene). The D405 wrist camera always starts on HW."
        ),
    )
    enable_d405_arg = DeclareLaunchArgument(
        "enable_d405",
        default_value="true",
        description="Hardware only: start the RealSense D405 wrist camera node.",
    )

    is_sim = LaunchConfiguration("is_sim")
    follower_serial_port = LaunchConfiguration("follower_serial_port")
    start_trajectory_controllers = LaunchConfiguration("start_trajectory_controllers")
    disable_servo_torque = LaunchConfiguration("disable_servo_torque")
    enable_octomap = LaunchConfiguration("enable_octomap")
    enable_d405 = LaunchConfiguration("enable_d405")

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
            "enable_octomap": enable_octomap,
        }.items(),
    )

    def _select_controller_launch(context):
        is_sim_mode = is_sim.perform(context).strip().lower() == "true"
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
                moveit_launch,
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
                # Hand-eye overlay owns optical TF; omit URDF optical frames.
                "d405_use_nominal_extrinsics": "false",
            }.items(),
        )
        camera_pose = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(_CAMERA_POSE_LAUNCH),
            condition=IfCondition(enable_d405),
        )
        # D405 wrist camera (HW only). Launch the node directly so parent
        # bringup args (is_sim, follower_serial_port, …) are not forwarded
        # into rs_launch.py and logged as unsupported-parameter warnings.
        # D405 exposes RGB through depth_module, not rgb_camera — set both
        # depth and color profiles there to keep USB bandwidth low on a shared hub.
        # Disable IR1/IR2 ROS streams (default opens 848x480@30 each) or USB drops
        # frames ("Incomplete video frame", ~24% of expected bytes). Depth still works.
        d405_camera = Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="d405_wrist",
            namespace="",
            output="screen",
            respawn=True,
            respawn_delay=3.0,
            parameters=[
                {
                    "camera_name": "d405_wrist",
                    "device_type": "D405",
                    "publish_tf": False,
                    "enable_infra": False,
                    "enable_infra1": False,
                    "enable_infra2": False,
                    # Sync depth+color in one frameset (needed for textured pointcloud).
                    "enable_sync": True,
                    "pointcloud.enable": True,
                    # LibRealSense RS2_OPTION_STREAM_FILTER: 2 = color (RGB texture).
                    "pointcloud.stream_filter": 2,
                    # Match MoveIt PointCloudOctomapUpdater (SensorDataQoS / BEST_EFFORT).
                    "pointcloud.pointcloud_qos": "SENSOR_DATA",
                    # Still publish XYZ if texture pairing fails (octomap only needs xyz).
                    "pointcloud.allow_no_texture_points": True,
                    "align_depth.enable": True,
                    "enable_color": True,
                    "enable_depth": True,
                    # 640x480@30 gives finer ChArUco corner localization for hand-eye / target
                    # tracking than the previous 424x240@15 (~1.5x angular precision per pixel)
                    # and doubles the frame rate for tighter octomap MessageFilter alignment,
                    # while still under the driver's 848x480@30 fallback bandwidth. @10 is
                    # not supported on D405 (falls back to 848x480@30).
                    "depth_module.depth_profile": "640,480,30",
                    "depth_module.color_profile": "640,480,30",
                }
            ],
        )
        return [
            configure_feetech,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=configure_feetech,
                    on_exit=[
                        hw_launch,
                        camera_pose,
                        TimerAction(
                            period=_D405_DELAY_S,
                            actions=[d405_camera],
                            condition=IfCondition(enable_d405),
                        ),
                        TimerAction(period=_MOVEIT_DELAY_S, actions=[moveit_launch]),
                    ],
                )
            ),
        ]

    return LaunchDescription(
        [
            is_sim_arg,
            follower_serial_port_arg,
            start_trajectory_controllers_arg,
            disable_servo_torque_arg,
            enable_octomap_arg,
            enable_d405_arg,
            OpaqueFunction(function=_select_controller_launch),
        ]
    )
