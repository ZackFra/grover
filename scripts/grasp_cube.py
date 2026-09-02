#!/usr/bin/env python3
"""Open at top_view, close the gripper, then hover → ACT → home.

Starts ``detect_red_cube.py`` and a wrist-camera OpenCV window
(``view_ros_image.py`` on ``/d405_wrist/color/image_raw``). Press Enter to
start each iteration; Ctrl-C quits.

Do not teleop or RViz Plan & Execute during a grasp.

Bringup (D405 on, octomap off)::

    ros2 launch launch/so101_bringup.launch.py is_sim:=False enable_d405:=true enable_octomap:=false

Usage::

    python3 scripts/grasp_cube.py
    python3 scripts/grasp_cube.py --once               # one grasp, no prompt
    python3 scripts/grasp_cube.py --no-detect          # detect already running
    python3 scripts/grasp_cube.py --no-display         # no OpenCV window
    python3 scripts/grasp_cube.py --infer-time-s 15    # extra flags go to hover
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path


def _preload_conda_openssl() -> None:
    lib = Path(sys.prefix) / "lib"
    for name in ("libcrypto.so.3", "libssl.so.3"):
        path = lib / name
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


_preload_conda_openssl()

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from hover_above_cube import main as hover_main  # noqa: E402
from moveit_goto_joints import execute_named_pose  # noqa: E402

import rclpy  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--open-pose",
        default="top_view_open",
        help="Named pose with gripper open (default: top_view_open).",
    )
    parser.add_argument(
        "--closed-pose",
        default="top_view",
        help="Named pose with gripper closed (default: top_view).",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Do not spawn detect_red_cube.py (use if it is already running).",
    )
    parser.add_argument(
        "--detect-warmup-s",
        type=float,
        default=2.0,
        help="Seconds to wait after starting the detector (default 2).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single grasp and exit (default is loop with Enter between).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open the wrist camera OpenCV window.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Ignored; display is on by default. Kept so --show is not passed to hover.",
    )
    parser.add_argument(
        "--view-topic",
        default="/d405_wrist/color/image_raw",
        help="Image topic for the OpenCV window (default: wrist RGB).",
    )
    return parser


def _start_detector() -> subprocess.Popen:
    cmd = [sys.executable, str(_SCRIPTS / "detect_red_cube.py")]
    print(f"Starting detector: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def _start_viewer(topic: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(_SCRIPTS / "view_ros_image.py"),
        "--topic",
        topic,
    ]
    print(f"Starting camera view: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


def _run_cycle(args: argparse.Namespace, hover_argv: list[str]) -> int:
    print(f"\n=== {args.open_pose} ===", flush=True)
    code = execute_named_pose(args.open_pose)
    if code != 0:
        return code

    print(f"\n=== {args.closed_pose} ===", flush=True)
    code = execute_named_pose(args.closed_pose)
    if code != 0:
        return code

    print("\n=== hover → ACT → home ===", flush=True)
    return hover_main(hover_argv)


def main(argv: list[str] | None = None) -> int:
    args, hover_argv = _build_parser().parse_known_args(argv)
    show = not args.no_display and bool(os.environ.get("DISPLAY"))
    if not args.no_display and not show:
        print(
            "No DISPLAY; skipping OpenCV window. "
            "Run  python3 scripts/view_ros_image.py  in a desktop terminal.",
            flush=True,
        )

    detector: subprocess.Popen | None = None
    viewer: subprocess.Popen | None = None
    try:
        if not args.no_detect:
            detector = _start_detector()
            time.sleep(max(args.detect_warmup_s, 0.0))
            if detector.poll() is not None:
                print(
                    f"ERROR: detect_red_cube.py exited with {detector.returncode}",
                    file=sys.stderr,
                )
                return 1
        if show:
            viewer = _start_viewer(args.view_topic)
            time.sleep(1.0)
            if viewer.poll() is not None:
                print(
                    f"ERROR: camera viewer exited with {viewer.returncode}. "
                    "Try: python3 scripts/view_ros_image.py",
                    file=sys.stderr,
                )
                viewer = None
            else:
                print(
                    f"Wrist camera Qt window: {args.view_topic}",
                    flush=True,
                )

        if not rclpy.ok():
            rclpy.init()

        n = 0
        while True:
            n += 1
            if not args.once:
                try:
                    input(
                        f"\n[grasp {n}] Place the cube, then press Enter to start "
                        "(Ctrl-C to quit) ... "
                    )
                except EOFError:
                    print("\nStopped.", flush=True)
                    return 0
            code = _run_cycle(args, hover_argv)
            if args.once:
                return code
            if code != 0:
                print(
                    f"Grasp {n} finished with code {code}. "
                    "Press Enter to try again, or Ctrl-C to quit.",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        return 0
    finally:
        rclpy.try_shutdown()
        _stop_process(viewer)
        _stop_process(detector)


if __name__ == "__main__":
    raise SystemExit(main())
