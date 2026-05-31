#!/usr/bin/env python3
"""Convert a MoveIt2 hand-eye calibration into a xacro:sensor_d405 <origin>.

The hand-eye plugin's "Save camera pose" emits a static_transform_publisher
launch file for `gripper -> d405_wrist_color_optical_frame`. The xacro macro
`xacro:sensor_d405`'s `<origin>` controls `gripper -> d405_wrist_bottom_screw_frame`
instead, with a fixed `bottom_screw_frame -> color_optical_frame` constant chained
on by realsense2_description (when `use_nominal_extrinsics="true"`). To bake the
calibration into the URDF we have to back out that constant:

    T_origin_new = T_handeye . T_internal^-1

This script:
  1. Parses the saved launch file for the calibrated xyz + quat.
  2. Reads `T_internal` once from the live TF tree (so we don't hardcode anything
     that could drift if realsense2_description updates).
  3. Prints the new <origin> line for so101_base.xacro.

Usage (with bringup already running so TF is alive):

    ros2 launch launch/so101_bringup.launch.py is_sim:=False
    # in another terminal:
    python3 scripts/apply_handeye_to_xacro.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAUNCH = REPO_ROOT / "config" / "realsense-d405" / "camera_pose.launch.py"
DEFAULT_INTERNAL_PARENT = "d405_wrist_bottom_screw_frame"
DEFAULT_INTERNAL_CHILD = "d405_wrist_color_optical_frame"


def parse_static_tf_launch(path: Path) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Return (frame_id, child_frame_id, xyz, quat_xyzw) from a launch file."""
    text = path.read_text()
    pairs = re.findall(r'"--([^"]+)"\s*,\s*"([^"]+)"', text)
    if not pairs:
        raise ValueError(
            f"Could not parse static_transform_publisher arguments from {path}."
        )
    args = dict(pairs)
    try:
        parent = args["frame-id"]
        child = args["child-frame-id"]
        xyz = np.array(
            [float(args["x"]), float(args["y"]), float(args["z"])], dtype=float
        )
        quat = np.array(
            [
                float(args["qx"]),
                float(args["qy"]),
                float(args["qz"]),
                float(args["qw"]),
            ],
            dtype=float,
        )
    except KeyError as exc:
        raise ValueError(f"Missing required arg {exc} in {path}") from exc

    norm = np.linalg.norm(quat)
    if abs(norm - 1.0) > 1e-3:
        print(
            f"WARN: quaternion in {path.name} has norm {norm:.6f}; renormalising.",
            file=sys.stderr,
        )
        quat = quat / norm
    return parent, child, xyz, quat


def lookup_tf_once(
    target: str, source: str, timeout_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Block until `target -> source` is available; return (xyz, quat_xyzw).

    Convention matches `ros2 run tf2_ros tf2_echo target source` -- the result
    is the pose of `source` expressed in `target`.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import (
        Buffer,
        ConnectivityException,
        ExtrapolationException,
        LookupException,
        TransformListener,
    )

    rclpy.init()
    try:
        node = Node("apply_handeye_to_xacro_lookup")
        buf = Buffer()
        TransformListener(buf, node)
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                tf = buf.lookup_transform(target, source, Time())
            except (LookupException, ExtrapolationException, ConnectivityException) as exc:
                last_err = exc
                continue
            t = tf.transform.translation
            r = tf.transform.rotation
            xyz = np.array([t.x, t.y, t.z], dtype=float)
            quat = np.array([r.x, r.y, r.z, r.w], dtype=float)
            return xyz, quat
        raise RuntimeError(
            f"TF {target} -> {source} not available within {timeout_s:.1f}s "
            f"(last error: {last_err}). Is so101_bringup running?"
        )
    finally:
        rclpy.shutdown()


def quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    """Scalar-last quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q_xyzw)
    n = x * x + y * y + z * z + w * w
    if n <= 0.0:
        raise ValueError("Zero-norm quaternion.")
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> scalar-last quaternion (x, y, z, w). Shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def R_to_rpy(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> fixed-XYZ Euler angles (roll, pitch, yaw).

    Matches URDF/xacro `<origin rpy="r p y"/>` convention: R = Rz(yaw) Ry(pitch) Rx(roll).
    """
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))  # clamp against numerical drift
    pitch = math.asin(sp)
    if abs(abs(sp) - 1.0) < 1e-6:
        # Gimbal lock at pitch = +/- pi/2: yaw is degenerate; pick yaw = 0 and read roll.
        yaw = 0.0
        roll = math.atan2(-R[1, 2], R[1, 1])
    else:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw])


def make_T(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat_xyzw)
    T[:3, 3] = xyz
    return T


def split_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = T[:3, 3]
    R = T[:3, :3]
    return xyz, R_to_rpy(R), R_to_quat(R)


def near_gimbal_lock(rpy: np.ndarray, eps_rad: float = math.radians(2.0)) -> bool:
    return abs(abs(rpy[1]) - math.pi / 2.0) < eps_rad


def _fmt_vec(v: np.ndarray) -> str:
    return ", ".join(f"{x:+.6f}" for x in v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        type=Path,
        default=DEFAULT_LAUNCH,
        help="MoveIt-saved static_transform_publisher launch file.",
    )
    parser.add_argument(
        "--internal-parent",
        default=DEFAULT_INTERNAL_PARENT,
        help="Parent frame of the URDF-internal constant.",
    )
    parser.add_argument(
        "--internal-child",
        default=DEFAULT_INTERNAL_CHILD,
        help="Child frame of the URDF-internal constant "
        "(should match the launch's child-frame-id).",
    )
    parser.add_argument(
        "--tf-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the internal TF before giving up.",
    )
    args = parser.parse_args()

    launch_path: Path = args.launch
    if not launch_path.exists():
        print(f"ERROR: launch file not found: {launch_path}", file=sys.stderr)
        return 1

    parent, child, he_xyz, he_quat = parse_static_tf_launch(launch_path)
    print(f"Loaded hand-eye calibration from {launch_path.relative_to(REPO_ROOT)}:")
    print(f"  {parent} -> {child}")
    print(f"    xyz  = ({_fmt_vec(he_xyz)})")
    print(f"    quat = ({_fmt_vec(he_quat)})  (x, y, z, w)")
    print()

    if child != args.internal_child:
        print(
            f"WARN: launch child '{child}' != --internal-child '{args.internal_child}'. "
            "The composition assumes these are the same frame.",
            file=sys.stderr,
        )

    print(
        f"Looking up internal constant via TF: "
        f"{args.internal_parent} -> {args.internal_child} ..."
    )
    in_xyz, in_quat = lookup_tf_once(
        args.internal_parent, args.internal_child, args.tf_timeout
    )
    print(f"    xyz  = ({_fmt_vec(in_xyz)})")
    print(f"    quat = ({_fmt_vec(in_quat)})  (x, y, z, w)")
    print()

    T_handeye = make_T(he_xyz, he_quat)
    T_internal = make_T(in_xyz, in_quat)
    T_origin_new = T_handeye @ np.linalg.inv(T_internal)
    new_xyz, new_rpy, new_quat = split_T(T_origin_new)

    print(
        f"Computed new {parent} -> {args.internal_parent} "
        f"(this is the xacro:sensor_d405 <origin>):"
    )
    print(f"    xyz  = ({_fmt_vec(new_xyz)})")
    print(f"    rpy  = ({_fmt_vec(new_rpy)})  fixed-XYZ, radians")
    print(f"    quat = ({_fmt_vec(new_quat)})  (x, y, z, w)")
    if near_gimbal_lock(new_rpy):
        print()
        print(
            "  WARN: pitch is within 2 deg of +/- pi/2 (gimbal lock for fixed-XYZ Euler).\n"
            "        RPY values are numerically unstable here -- small noise in the\n"
            "        rotation matrix swings yaw across +/- pi. After pasting into the\n"
            "        xacro and rebuilding, verify with:\n"
            "            ros2 run tf2_ros tf2_echo gripper d405_wrist_color_optical_frame\n"
            "        It should match the saved hand-eye xyz/quat to <1e-3.\n"
            "        If it doesn't, keep the calibration as a static_transform_publisher\n"
            "        overlay (use_nominal_extrinsics=\"false\" + camera_pose.launch.py)\n"
            "        instead of baking it into the URDF.",
        )
    print()
    print("Paste this into src/lerobot_description/urdf/so101_base.xacro,")
    print("replacing the existing <origin .../> inside <xacro:sensor_d405 ...>:")
    print()
    print(
        f'\t\t<origin xyz="{new_xyz[0]:.6f} {new_xyz[1]:.6f} {new_xyz[2]:.6f}" '
        f'rpy="{new_rpy[0]:.6f} {new_rpy[1]:.6f} {new_rpy[2]:.6f}" />'
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
