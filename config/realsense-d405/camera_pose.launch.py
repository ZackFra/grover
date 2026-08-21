""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: gripper -> d405_wrist_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "gripper",
                "--child-frame-id",
                "d405_wrist_color_optical_frame",
                "--x",
                "-0.0135324",
                "--y",
                "-0.0600973",
                "--z",
                "-0.0277452",
                "--qx",
                "0.968012",
                "--qy",
                "-0.0310663",
                "--qz",
                "-0.0144393",
                "--qw",
                "-0.248554",
                # "--roll",
                # "0.503096",
                # "--pitch",
                # "-3.12908",
                # "--yaw",
                # "-3.07421",
            ],
        ),
        # D405 color and depth share one imager (URDF color offset is 0). Pointcloud
        # headers use depth_optical_frame; without this, octomap TF lookups fail
        # when URDF optical frames are omitted for the hand-eye overlay.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "d405_wrist_color_optical_frame",
                "--child-frame-id",
                "d405_wrist_depth_optical_frame",
                "--x",
                "0",
                "--y",
                "0",
                "--z",
                "0",
                "--qx",
                "0",
                "--qy",
                "0",
                "--qz",
                "0",
                "--qw",
                "1",
            ],
        ),
    ]
    return LaunchDescription(nodes)
