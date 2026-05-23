#!/usr/bin/env python3
"""Apply a local-frame rotation to a URDF ``rpy`` triple and emit the new ``rpy``.

URDF's ``rpy="r p y"`` is a fixed-axis (extrinsic XYZ) Euler representation:
``R = Rz(y) · Ry(p) · Rx(r)``. Tweaking individual components when ``rpy`` is
non-zero does NOT rotate around the child frame's own axes, and at pitch=±π/2
the representation hits gimbal lock (roll and yaw collide on the same axis).

This script sidesteps both problems. Give it the current ``rpy`` plus a
rotation expressed in the CHILD'S OWN frame (e.g. "rotate 30° around the
camera's local +Z / image-up axis"), and it composes ``R · R_local`` and
re-extracts a clean rpy.

Example — apply +30° yaw around the camera's own image-up (+Z) axis on top of
the current SO-101 wrist camera orientation:

    python3 rotate_rpy.py --rpy "4.7124 1.5708 0" --axis z --degrees 30

Sign convention: positive degrees = right-hand rotation about the chosen
LOCAL axis of the child frame. For the D405 macro: ``x``=lens roll-around,
``y``=image pitch up/down, ``z``=image yaw left/right (from lens POV).
"""

from __future__ import annotations

import argparse

import numpy as np


def _parse_triplet(text: str, name: str) -> np.ndarray:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must have 3 numbers, got {text!r}")
    return np.array([float(x) for x in parts], dtype=float)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy (extrinsic XYZ) → 3x3 rotation R = Rz·Ry·Rx."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Extract URDF rpy from a 3x3 rotation matrix.

    Picks the principal-branch solution where ``pitch ∈ [-π/2, π/2]``. At
    pitch=±π/2 (gimbal lock) the (roll, yaw) split is ambiguous; this function
    arbitrarily attributes the residual to yaw.
    """
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    if np.isclose(np.cos(pitch), 0.0, atol=1e-9):
        # Gimbal-locked output: collapse onto yaw, leave roll at 0.
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    else:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw


def axis_angle_rotation(axis: str, radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise argparse.ArgumentTypeError(f"--axis must be x, y, or z (got {axis!r})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rpy",
        required=True,
        type=lambda s: _parse_triplet(s, "--rpy"),
        help='Current rpy triple in radians, e.g. "1.5708 1.5708 0".',
    )
    parser.add_argument(
        "--axis",
        required=True,
        choices=["x", "y", "z"],
        help="Local (child-frame) axis to rotate around.",
    )
    parser.add_argument(
        "--degrees",
        required=True,
        type=float,
        help="Rotation magnitude in degrees (positive = right-hand rule about --axis).",
    )
    parser.add_argument(
        "--apply",
        choices=["local", "parent"],
        default="local",
        help=(
            "Frame interpretation: 'local' applies R_new = R · R_axis(θ) so the rotation "
            "is about the child's own axis (the usual intent for camera placement); "
            "'parent' applies R_new = R_axis(θ) · R so the rotation is about the parent's axis. "
            "Default: local."
        ),
    )
    args = parser.parse_args()

    current = rpy_to_matrix(args.rpy)
    delta = axis_angle_rotation(args.axis, np.deg2rad(args.degrees))
    new = current @ delta if args.apply == "local" else delta @ current

    r, p, y = matrix_to_rpy(new)
    print(f'rpy="{r:.4f} {p:.4f} {y:.4f}"')
    print(f"# lens (+X cam) in parent frame: {new[:, 0].round(4)}")
    print(f"# left (+Y cam) in parent frame: {new[:, 1].round(4)}")
    print(f"# up   (+Z cam) in parent frame: {new[:, 2].round(4)}")


if __name__ == "__main__":
    main()
