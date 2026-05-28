import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lerobot_description = get_package_share_directory("lerobot_description")

    model_arg = DeclareLaunchArgument(name="model", default_value=os.path.join(
                                        lerobot_description, "urdf", "so101.urdf.xacro"
                                        ),
                                      description="Absolute path to robot urdf file"
    )

    # RViz is sim-time aware via use_sim_time:=True parameter below. Without
    # that, RViz's TF buffer is stamped with wall-clock while Gazebo publishes
    # TF stamped with sim-clock, lookups fail, and the cloud renders at stale
    # transforms ("ghost" displaced rendering).
    rviz_arg = DeclareLaunchArgument(
        name="rviz",
        default_value="true",
        description="Launch RViz2 with the sim-aware d405_sim.rviz config.",
    )

    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            str(Path(lerobot_description).parent.resolve())
            ]
        )
    
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                LaunchConfiguration("model"),
                " use_sim:=true",
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}]
    )

    # Use our custom world that loads gz-sim-sensors-system so the D405
    # rgbd_camera in so101_gazebo.xacro actually produces image/depth/points.
    world_path = os.path.join(lerobot_description, "worlds", "so101_empty.sdf")

    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory("ros_gz_sim"), "launch"), "/gz_sim.launch.py"]),
                launch_arguments=[
                    ("gz_args", [" -v 4 -r ", world_path, " "]
                    )
                ]
             )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description",
                   "-name", "so101"],
    )

    # /clock bridge is small enough to stay as CLI args; D405 image/depth/points
    # bridging lives in config/d405_wrist_bridge.yaml so the topic remaps to
    # realsense2_camera-compatible names stay readable.
    gz_clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ]
    )

    d405_bridge_config = os.path.join(
        lerobot_description, "config", "d405_wrist_bridge.yaml"
    )

    d405_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="d405_wrist_bridge",
        output="screen",
        parameters=[{
            "config_file": d405_bridge_config,
            "use_sim_time": True,
        }],
    )

    # Spawn ros2_control controllers once the model is in Gazebo. Without these,
    # joint_state_broadcaster never starts, /joint_states stays silent, and
    # robot_state_publisher renders every revolute joint at zero — the result
    # is a robot model collapsed into a single chunk near base. The hardware
    # bringup also auto-spawns these, so we mirror that here.
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    load_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    load_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # Wait until the entity is spawned (gz_ros2_control plugin inside Gazebo
    # only exposes /controller_manager after the model loads).
    spawn_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_spawn_entity,
            on_exit=[
                load_joint_state_broadcaster,
                load_arm_controller,
                load_gripper_controller,
            ],
        )
    )

    rviz_config_path = os.path.join(lerobot_description, "rviz", "d405_sim.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],
        condition=IfCondition(LaunchConfiguration("rviz")),
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        model_arg,
        rviz_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        gz_clock_bridge,
        d405_bridge,
        spawn_controllers,
        rviz_node,
    ])