#!/usr/bin/env python3
"""LeRobot Camera that reads a ROS 2 sensor_msgs/Image topic.

Bringup already owns the D405 USB device (realsense2_camera). Training used
``intelrealsense`` directly; inference after detect/hover must subscribe to
``/d405_wrist/color/image_raw`` instead of opening the camera again.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode
from lerobot.utils.errors import DeviceNotConnectedError

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: OpenCV (cv2) is required for ros_image_camera.") from exc

logger = logging.getLogger(__name__)


def _imgmsg_to_array(msg: Image, channels: int, dtype: np.dtype) -> np.ndarray:
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    itemsize = dt.itemsize
    packed_step = msg.width * channels * itemsize
    if msg.step < packed_step:
        raise ValueError(f"Image step {msg.step} too small for {msg.width}x{channels} {dt}")
    shape = (msg.height, msg.width) if channels == 1 else (msg.height, msg.width, channels)
    if msg.step == packed_step:
        return (
            np.frombuffer(msg.data, dtype=dt, count=msg.height * msg.width * channels)
            .reshape(shape)
            .copy()
        )
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=msg.height * msg.step)
    rows = raw.reshape(msg.height, msg.step)[:, :packed_step]
    return np.frombuffer(np.ascontiguousarray(rows), dtype=dt).reshape(shape).copy()


def _imgmsg_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding
    if enc in ("rgb8", "8UC3"):
        return np.ascontiguousarray(_imgmsg_to_array(msg, 3, np.uint8))
    if enc == "bgr8":
        return cv2.cvtColor(_imgmsg_to_array(msg, 3, np.uint8), cv2.COLOR_BGR2RGB)
    if enc in ("rgba8", "rgbA8"):
        return cv2.cvtColor(_imgmsg_to_array(msg, 4, np.uint8), cv2.COLOR_RGBA2RGB)
    if enc == "bgra8":
        return cv2.cvtColor(_imgmsg_to_array(msg, 4, np.uint8), cv2.COLOR_BGRA2RGB)
    if enc == "mono8":
        return cv2.cvtColor(_imgmsg_to_array(msg, 1, np.uint8), cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unsupported color encoding '{enc}'")


@CameraConfig.register_subclass("ros_image")
@dataclass
class ROSImageCameraConfig(CameraConfig):
    topic: str = "/d405_wrist/color/image_raw"
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: float = 1.0


class ROSImageCamera(Camera):
    """Latest-frame subscriber on a ROS Image topic (SensorData / BEST_EFFORT)."""

    def __init__(self, config: ROSImageCameraConfig):
        super().__init__(config)
        self.config = config
        self.color_mode = ColorMode(config.color_mode)
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: NDArray[Any] | None = None
        self._new_frame = threading.Event()
        self._connected = False

    def __str__(self) -> str:
        return f"ROSImageCamera({self.config.topic})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._node is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            return
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("lerobot_ros_image_camera")
        self._node.create_subscription(
            Image, self.config.topic, self._on_image, qos_profile_sensor_data
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self._connected = True

        if warmup and self.config.warmup_s > 0:
            deadline = time.monotonic() + float(self.config.warmup_s)
            while time.monotonic() < deadline:
                try:
                    self.async_read(timeout_ms=200)
                    break
                except TimeoutError:
                    continue
            else:
                raise ConnectionError(
                    f"{self} got no frames on {self.config.topic} in {self.config.warmup_s:.1f}s. "
                    "Is so101_bringup running with enable_d405:=true?"
                )
        logger.info("%s connected.", self)

    def _on_image(self, msg: Image) -> None:
        rgb = _imgmsg_to_rgb(msg)
        if self.width and self.height and (rgb.shape[1] != self.width or rgb.shape[0] != self.height):
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.color_mode == ColorMode.BGR:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._latest = rgb
        self._new_frame.set()

    def read(self) -> NDArray[Any]:
        return self.async_read(timeout_ms=1000)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self._new_frame.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms} ms "
                f"(topic {self.config.topic})."
            )
        with self._lock:
            frame = self._latest
            self._new_frame.clear()
        if frame is None:
            raise RuntimeError(f"{self} event set but no frame.")
        return frame

    def disconnect(self) -> None:
        self._connected = False
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        with self._lock:
            self._latest = None
        logger.info("%s disconnected.", self)
