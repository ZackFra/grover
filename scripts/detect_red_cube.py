#!/usr/bin/env python3
"""Detect a red cube from the D405 wrist camera and publish a hover pose.

HSV-segments red in /d405_wrist/color/image_raw, deprojects the blob centroid
with aligned depth + CameraInfo, then TF into `world`. Hover pose is the cube
XY with +Z offset; orientation is copied from the live `gripper` TF (so a
later MoveIt goal keeps the top-down wrist from `top_view`).

Requires so101_bringup (camera + TF). Do not add this to bringup — extra
image subscribers compete with the D405 USB path.

Usage:
    # After: python3 scripts/moveit_goto_joints.py top_view
    python3 scripts/detect_red_cube.py
    # In rqt_image_view pick /red_cube/overlay  (raw | HSV mask | detect)
    # then in another terminal:
    python3 scripts/hover_above_cube.py
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ERROR: OpenCV (cv2) is required in this Python env (pip/conda opencv)."
    ) from exc

# Avoid cv_bridge / image_geometry: those ROS apt wheels are built against
# NumPy 1.x and break in conda envs with NumPy 2 (and image_geometry needs
# the `deprecated` package, which is not on PYTHONPATH in grover).


def _quat_rotate(qx: float, qy: float, qz: float, qw: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Rotate (x, y, z) by quaternion (x, y, z, w)."""
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def _transform_xyz(tf_msg, x: float, y: float, z: float) -> tuple[float, float, float]:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    rx, ry, rz = _quat_rotate(q.x, q.y, q.z, q.w, x, y, z)
    return rx + t.x, ry + t.y, rz + t.z


def _imgmsg_to_array(msg: Image, channels: int, dtype: np.dtype) -> np.ndarray:
    """Unpack sensor_msgs/Image into a writeable (H, W[, C]) array without cv_bridge."""
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    itemsize = dt.itemsize
    if msg.step < msg.width * channels * itemsize:
        raise ValueError(
            f"Image step {msg.step} too small for {msg.width}x{channels} {dt}"
        )
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=msg.height * msg.step)
    rows = raw.reshape(msg.height, msg.step)
    useful = rows[:, : msg.width * channels * itemsize]
    typed = np.frombuffer(useful.tobytes(), dtype=dt)
    shape = (msg.height, msg.width) if channels == 1 else (msg.height, msg.width, channels)
    return typed.reshape(shape).copy()


def _color_to_bgr(msg: Image) -> np.ndarray:
    enc = msg.encoding
    if enc == "bgr8":
        return np.ascontiguousarray(_imgmsg_to_array(msg, 3, np.uint8))
    if enc in ("rgb8", "8UC3"):
        rgb = _imgmsg_to_array(msg, 3, np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if enc in ("rgba8", "rgbA8"):
        rgba = _imgmsg_to_array(msg, 4, np.uint8)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        bgra = _imgmsg_to_array(msg, 4, np.uint8)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if enc == "mono8":
        gray = _imgmsg_to_array(msg, 1, np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported color encoding '{enc}'")


def _depth_to_meters(msg: Image) -> np.ndarray:
    enc = msg.encoding
    if enc in ("16UC1", "mono16"):
        return _imgmsg_to_array(msg, 1, np.uint16).astype(np.float32) / 1000.0
    if enc == "32FC1":
        return _imgmsg_to_array(msg, 1, np.float32)
    raise ValueError(f"Unsupported depth encoding '{enc}' (expected 16UC1 or 32FC1).")


def _bgr_to_imgmsg(bgr: np.ndarray, header) -> Image:
    out = Image()
    out.header = header
    out.height = int(bgr.shape[0])
    out.width = int(bgr.shape[1])
    out.encoding = "bgr8"
    out.is_bigendian = 0
    out.step = out.width * 3
    out.data = np.ascontiguousarray(bgr).tobytes()
    return out


def _deproject(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    """Pixel + optical-axis depth (m) -> camera XYZ using CameraInfo K."""
    return ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m)


def _identity_pose() -> Pose:
    pose = Pose()
    pose.orientation.w = 1.0
    return pose


class RedCubeDetector(Node):
    """HSV red blob -> world cube pose + hover pose above it."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("detect_red_cube")
        self._args = args
        self._fx = self._fy = self._cx = self._cy = 0.0
        self._cam_frame = ""
        self._have_info = False
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = qos_profile_sensor_data
        self._latest_depth: Image | None = None
        self._n_color = 0
        self._n_depth = 0
        self._last_status = "waiting for color"
        self.create_subscription(
            CameraInfo, args.camera_info_topic, self._on_camera_info, sensor_qos
        )
        self.create_subscription(Image, args.color_topic, self._on_color, sensor_qos)
        self.create_subscription(Image, args.depth_topic, self._on_depth, sensor_qos)
        self.create_timer(2.0, self._log_status)

        self._pose_pub = self.create_publisher(
            PoseStamped, "/red_cube/pose", sensor_qos
        )
        self._hover_pub = self.create_publisher(
            PoseStamped, "/red_cube/hover_pose", sensor_qos
        )
        self._overlay_pub = self.create_publisher(
            Image, "/red_cube/overlay", sensor_qos
        )
        self._mask_pub = self.create_publisher(Image, "/red_cube/mask", sensor_qos)
        self._marker_pub = self.create_publisher(MarkerArray, "/red_cube/markers", 10)

        self.get_logger().info(
            "Detecting red cube on "
            f"{args.color_topic} + {args.depth_topic}; "
            f"hover_offset={args.hover_offset:.3f} m, "
            f"world={args.world_frame}, ee={args.ee_frame}. "
            "rqt_image_view topic: /red_cube/overlay"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        # K is row-major 3x3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])
        self._cam_frame = msg.header.frame_id
        self._have_info = self._fx > 1e-9 and self._fy > 1e-9

    def _lookup_tf(self, target: str, source: str, stamp: Time):
        timeout = rclpy.duration.Duration(seconds=0.05)
        try:
            return self._tf_buffer.lookup_transform(target, source, stamp, timeout)
        except TransformException:
            return self._tf_buffer.lookup_transform(
                target, source, Time(), timeout
            )

    def _log_status(self) -> None:
        self.get_logger().info(
            f"{self._last_status}  (color={self._n_color} depth={self._n_depth} "
            f"caminfo={self._have_info})"
        )

    def _on_depth(self, msg: Image) -> None:
        self._n_depth += 1
        self._latest_depth = msg

    def _on_color(self, color_msg: Image) -> None:
        self._n_color += 1
        try:
            raw = _color_to_bgr(color_msg)
        except ValueError as exc:
            self._last_status = f"color convert failed: {exc} ({color_msg.encoding})"
            self.get_logger().warn(self._last_status, throttle_duration_sec=2.0)
            return

        depth_m = None
        if self._latest_depth is not None:
            try:
                depth_m = _depth_to_meters(self._latest_depth)
                if depth_m.shape[:2] != raw.shape[:2]:
                    self._last_status = (
                        f"size mismatch color {raw.shape[:2]} depth {depth_m.shape[:2]}"
                    )
                    annotated = raw.copy()
                    self._draw_status(annotated, self._last_status)
                    self._publish_debug(raw, None, annotated, color_msg)
                    return
            except (ValueError, TypeError) as exc:
                self._last_status = f"bad depth: {exc}"
                annotated = raw.copy()
                self._draw_status(annotated, self._last_status)
                self._publish_debug(raw, None, annotated, color_msg)
                return

        cube_uvz, reason, mask, annotated = self._find_red_centroid(raw, depth_m)
        if not self._have_info:
            reason = "waiting for CameraInfo"
            cube_uvz = None
        if cube_uvz is None:
            self._last_status = reason
            self._draw_status(annotated, reason)
            self._publish_debug(raw, mask, annotated, color_msg)
            self._clear_markers(color_msg)
            return

        u, v, z_m = cube_uvz
        cam_x, cam_y, cam_z = _deproject(
            float(u), float(v), z_m, self._fx, self._fy, self._cx, self._cy
        )

        cam_frame = (
            self._cam_frame
            or color_msg.header.frame_id
            or "d405_wrist_color_optical_frame"
        )
        stamp = Time.from_msg(color_msg.header.stamp)
        try:
            cam_tf = self._lookup_tf(self._args.world_frame, cam_frame, stamp)
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {self._args.world_frame} <- {cam_frame} missing: {exc}",
                throttle_duration_sec=2.0,
            )
            self._last_status = f"no camera TF: {exc}"
            self._draw_status(annotated, "no camera TF")
            self._publish_debug(raw, mask, annotated, color_msg)
            return

        wx, wy, wz = _transform_xyz(cam_tf, cam_x, cam_y, cam_z)

        cube_pose = PoseStamped()
        cube_pose.header.stamp = color_msg.header.stamp
        cube_pose.header.frame_id = self._args.world_frame
        cube_pose.pose = _identity_pose()
        cube_pose.pose.position.x = wx
        cube_pose.pose.position.y = wy
        cube_pose.pose.position.z = wz

        hover_pose = PoseStamped()
        hover_pose.header = cube_pose.header
        hover_pose.pose = _identity_pose()
        hover_pose.pose.position.x = wx
        hover_pose.pose.position.y = wy
        hover_pose.pose.position.z = wz + self._args.hover_offset

        try:
            ee_tf = self._lookup_tf(self._args.world_frame, self._args.ee_frame, stamp)
            hover_pose.pose.orientation = ee_tf.transform.rotation
        except TransformException:
            pass

        self._pose_pub.publish(cube_pose)
        self._hover_pub.publish(hover_pose)
        self._publish_markers(cube_pose, hover_pose)

        cv2.circle(annotated, (int(u), int(v)), 6, (0, 255, 0), 2)
        self._last_status = "ok"
        self._draw_status(
            annotated,
            f"cube ({wx:+.3f}, {wy:+.3f}, {wz:+.3f})  "
            f"hover z={hover_pose.pose.position.z:+.3f}",
        )
        self._publish_debug(raw, mask, annotated, color_msg)

    def _find_red_centroid(
        self, raw: np.ndarray, depth_m: np.ndarray | None
    ) -> tuple[tuple[int, int, float] | None, str, np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        a = self._args
        mask1 = cv2.inRange(
            hsv, np.array([a.h_low, a.s_min, a.v_min]), np.array([a.h_high, 255, 255])
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([a.h_low2, a.s_min, a.v_min]),
            np.array([a.h_high2, 255, 255]),
        )
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        annotated = raw.copy()
        tint = annotated.copy()
        tint[mask > 0] = (0, 0, 255)
        cv2.addWeighted(tint, 0.35, annotated, 0.65, 0, dst=annotated)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(raw.shape[0] * raw.shape[1])
        max_area = frame_area * a.max_area_frac
        best = None
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < a.min_area or area > max_area:
                continue
            if area > best_area:
                best_area = area
                best = contour
        if best is None:
            return (
                None,
                f"no red cube (need {a.min_area:.0f}–{max_area:.0f} px; table blobs ignored)",
                mask,
                annotated,
            )

        cv2.drawContours(annotated, [best], -1, (0, 255, 255), 2)
        moments = cv2.moments(best)
        if moments["m00"] < 1.0:
            return None, "no red cube (empty moments)", mask, annotated
        u = int(moments["m10"] / moments["m00"])
        v = int(moments["m01"] / moments["m00"])
        cv2.circle(annotated, (u, v), 5, (0, 255, 0), 2)

        if depth_m is None:
            return None, "red blob, waiting for depth", mask, annotated
        r = max(1, a.depth_window)
        h, w = depth_m.shape[:2]
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > a.min_depth) & (patch < a.max_depth)]
        if valid.size == 0:
            return None, f"red blob @({u},{v}) but no depth in window", mask, annotated
        return (u, v, float(np.median(valid))), "ok", mask, annotated

    def _draw_status(self, bgr: np.ndarray, text: str) -> None:
        cv2.putText(
            bgr,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _labeled_panel(self, img: np.ndarray, title: str) -> np.ndarray:
        panel = img.copy()
        bar = np.full((28, panel.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(
            bar,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return np.vstack((bar, panel))

    def _publish_debug(
        self,
        raw: np.ndarray,
        mask: np.ndarray | None,
        annotated: np.ndarray,
        src: Image,
    ) -> None:
        if mask is None:
            mask = np.zeros(raw.shape[:2], dtype=np.uint8)
        mask_bgr = np.zeros_like(raw)
        mask_bgr[mask > 0] = (0, 0, 255)
        mosaic = np.hstack(
            (
                self._labeled_panel(raw, "raw"),
                self._labeled_panel(mask_bgr, "HSV mask"),
                self._labeled_panel(annotated, f"detect  #{self._n_color}"),
            )
        )
        self._overlay_pub.publish(_bgr_to_imgmsg(mosaic, src.header))
        self._mask_pub.publish(_bgr_to_imgmsg(mask_bgr, src.header))
        if self._args.show:
            try:
                cv2.imshow("red_cube overlay", mosaic)
                cv2.waitKey(1)
            except cv2.error:
                pass

    def _marker(
        self,
        marker_id: int,
        pose: PoseStamped,
        mtype: int,
        scale: float,
        color: tuple[float, float, float],
        ns: str,
    ) -> Marker:
        marker = Marker()
        marker.header = pose.header
        marker.ns = ns
        marker.id = marker_id
        marker.type = mtype
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.8)
        marker.lifetime.nanosec = 400_000_000
        return marker

    def _publish_markers(self, cube: PoseStamped, hover: PoseStamped) -> None:
        array = MarkerArray()
        array.markers.append(
            self._marker(0, cube, Marker.CUBE, 0.04, (1.0, 0.0, 0.0), "red_cube")
        )
        array.markers.append(
            self._marker(1, hover, Marker.SPHERE, 0.02, (0.0, 0.8, 1.0), "red_cube")
        )
        self._marker_pub.publish(array)

    def _clear_markers(self, src: Image) -> None:
        array = MarkerArray()
        for marker_id in (0, 1):
            marker = Marker()
            marker.header.stamp = src.header.stamp
            marker.header.frame_id = self._args.world_frame
            marker.ns = "red_cube"
            marker.id = marker_id
            marker.action = Marker.DELETE
            array.markers.append(marker)
        self._marker_pub.publish(array)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-topic", default="/d405_wrist/color/image_raw")
    parser.add_argument(
        "--depth-topic", default="/d405_wrist/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--camera-info-topic", default="/d405_wrist/color/camera_info")
    parser.add_argument("--world-frame", default="world")
    parser.add_argument("--ee-frame", default="gripper")
    parser.add_argument(
        "--hover-offset",
        type=float,
        default=0.08,
        help="Meters added to cube Z in world for /red_cube/hover_pose (default 0.08).",
    )
    parser.add_argument("--min-area", type=float, default=80.0)
    parser.add_argument(
        "--max-area-frac",
        type=float,
        default=0.08,
        help="Ignore blobs larger than this fraction of the image (rejects the table).",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.08,
        help="Ignore depth closer than this (m); rejects near-field gripper.",
    )
    parser.add_argument("--max-depth", type=float, default=1.5)
    parser.add_argument(
        "--depth-window",
        type=int,
        default=5,
        help="Half-size of the median-depth window around the centroid (pixels).",
    )
    parser.add_argument("--h-low", type=int, default=0)
    parser.add_argument("--h-high", type=int, default=8)
    parser.add_argument("--h-low2", type=int, default=170)
    parser.add_argument("--h-high2", type=int, default=179)
    parser.add_argument("--s-min", type=int, default=130)
    parser.add_argument("--v-min", type=int, default=70)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an OpenCV window with the overlay (works in conda grover; rqt needs system PyQt5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.hover_offset < 0.0 or math.isnan(args.hover_offset):
        print("ERROR: --hover-offset must be >= 0", file=sys.stderr)
        return 1

    rclpy.init()
    node = RedCubeDetector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
