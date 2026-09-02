#!/usr/bin/env python3
"""Qt window for a ROS 2 sensor_msgs/Image topic (SensorData / BEST_EFFORT).

The grover conda ``cv2`` is headless (no GTK), so OpenCV highgui cannot open
a window. This viewer uses PyQt5 instead.

Usage:
    python3 scripts/view_ros_image.py
    python3 scripts/view_ros_image.py --topic /red_cube/overlay
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_red_cube import _color_to_bgr  # noqa: E402


class ImageNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("view_ros_image")
        self.frame: np.ndarray | None = None
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f"Viewing {topic} (SensorData QoS, Qt)")

    def _on_image(self, msg: Image) -> None:
        try:
            self.frame = _color_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=2.0)


class ImageWindow(QMainWindow):
    def __init__(self, node: ImageNode, title: str) -> None:
        super().__init__()
        self._node = node
        self.setWindowTitle(title)
        self._label = QLabel("Waiting for image…")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(640, 480)
        self.setCentralWidget(self._label)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self) -> None:
        if not rclpy.ok():
            QApplication.quit()
            return
        try:
            rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            if not rclpy.ok():
                QApplication.quit()
                return
            raise
        frame = self._node.frame
        if frame is None:
            return
        h, w, _ = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self._label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self._label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )
        frame = self._node.frame
        if frame is None:
            return
        h, w, _ = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self._label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self._label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/d405_wrist/color/image_raw",
        help="Image topic (default: /d405_wrist/color/image_raw).",
    )
    parser.add_argument("--window", default="D405 wrist")
    args = parser.parse_args(argv)

    rclpy.init()
    node = ImageNode(args.topic)
    app = QApplication(sys.argv)
    win = ImageWindow(node, args.window)
    win.show()
    try:
        return int(app.exec_())
    except KeyboardInterrupt:
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
