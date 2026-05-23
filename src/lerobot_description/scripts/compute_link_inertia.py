#!/usr/bin/env python3
"""Emit a URDF `<inertial>` block for a printed link plus an optional rigid payload.

Designed for the SO-101 ``gripper`` link after the Wrist_Roll_D405_Holder swap:
the printed holder's mass / CoM / inertia comes from the STL at a given effective
density (PLA-and-infill), and the Intel RealSense D405 (or any other rigid
payload) is added as a uniform box at a user-measured offset.

All combination math is done in the *link* frame (the same frame as the URDF
``<visual>``/``<collision>`` origin). The mesh-local CoM/inertia from trimesh is
transformed into the link frame using the visual ``xyz``/``rpy`` you pass, then
the payload is composed via the parallel-axis theorem.

Example (gripper link with the D405 holder + a D405 measured 4 cm in front of
the wrist-roll output, 2 cm above, centered laterally; PLA @ ~20% infill):

    python3 compute_link_inertia.py \\
      --mesh src/lerobot_description/meshes/so101/Wrist_Roll_D405_Holder.stl \\
      --mesh-scale 0.001 \\
      --density 0.42 \\
      --visual-xyz "5.55e-17 -2.18e-4 9.50e-4" \\
      --visual-rpy "-3.14159 0 0" \\
      --payload-mass 0.060 \\
      --payload-xyz  "0.000 -0.040 0.020" \\
      --payload-box  "0.042 0.042 0.023"

Pass ``--no-payload`` to inspect just the printed body. Pass ``--mesh-scale
0.001`` when the STL is exported in millimeters (Onshape/Fusion default); the
script will warn if a >1 m bounding box at scale 1.0 looks suspicious. Units:
meters, kg, and density in g/cm³.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    import trimesh
except ImportError as e:  # pragma: no cover - import-time hint
    raise SystemExit("trimesh required: pip install trimesh") from e


@dataclass(frozen=True)
class RigidBody:
    """A rigid body's mass properties expressed in a common (link) frame."""

    mass: float
    com: np.ndarray  # shape (3,), meters
    inertia: np.ndarray  # shape (3, 3), kg*m^2, about `com` in link-aligned axes


def _parse_triplet(text: str, name: str) -> np.ndarray:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must have 3 numbers, got {text!r}")
    return np.array([float(x) for x in parts], dtype=float)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy (roll-pitch-yaw, fixed-axis) → 3x3 rotation matrix R = Rz·Ry·Rx."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _parallel_axis(inertia_at_com: np.ndarray, mass: float, r: np.ndarray) -> np.ndarray:
    """Shift a 3x3 inertia tensor from CoM to a point offset by `r` (vector from new origin to CoM)."""
    return inertia_at_com + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def mesh_body(
    mesh_path: str,
    density_g_per_cm3: float,
    visual_xyz: np.ndarray,
    visual_rpy: np.ndarray,
    mesh_scale: float,
) -> RigidBody:
    mesh = trimesh.load_mesh(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit(f"Loaded mesh is not a single Trimesh: {type(mesh).__name__}")
    if not mesh.is_watertight:
        # trimesh still estimates volume via divergence theorem on non-watertight
        # meshes; warn so the user knows to suspect numbers if the STL is broken.
        print(
            "warning: mesh is not watertight; volume/inertia are best-effort. "
            "Repair with `meshlab` (Filters → Cleaning) if numbers look wrong."
        )
    if mesh_scale != 1.0:
        mesh.apply_scale(mesh_scale)
    extents = mesh.bounding_box.extents
    if mesh_scale == 1.0 and any(e > 1.0 for e in extents):
        # ROS URDFs are in meters; a >1 m bounding box on a SO-101 part almost
        # certainly means the STL is in millimeters. Bail loudly rather than emit
        # nonsense numbers (e.g. 27 gigagram CoM offsets).
        print(
            f"warning: mesh bounding box {extents.tolist()} > 1 m on a side. "
            "Your STL is probably in mm; pass --mesh-scale 0.001 (and add "
            'scale="0.001 0.001 0.001" to the <mesh> tag in the URDF too).'
        )
    mesh.density = density_g_per_cm3 * 1000.0  # g/cm³ → kg/m³ (trimesh uses SI)

    mass = float(mesh.mass)
    com_local = np.asarray(mesh.center_mass, dtype=float)
    i_local = np.asarray(mesh.moment_inertia, dtype=float)

    r_link_from_mesh = rpy_to_matrix(visual_rpy)
    com_link = r_link_from_mesh @ com_local + visual_xyz
    i_link = r_link_from_mesh @ i_local @ r_link_from_mesh.T

    return RigidBody(mass=mass, com=com_link, inertia=i_link)


def box_payload(
    mass: float,
    com_link: np.ndarray,
    rpy: np.ndarray,
    dims: np.ndarray,
) -> RigidBody:
    a, b, c = dims
    diag = np.array(
        [
            mass / 12.0 * (b * b + c * c),
            mass / 12.0 * (a * a + c * c),
            mass / 12.0 * (a * a + b * b),
        ]
    )
    i_local = np.diag(diag)
    r_link_from_payload = rpy_to_matrix(rpy)
    i_link = r_link_from_payload @ i_local @ r_link_from_payload.T
    return RigidBody(mass=mass, com=com_link, inertia=i_link)


def combine(bodies: list[RigidBody]) -> RigidBody:
    total_mass = sum(b.mass for b in bodies)
    if total_mass <= 0:
        raise SystemExit("Total mass is non-positive; check density / inputs.")
    com = sum(b.mass * b.com for b in bodies) / total_mass
    inertia = np.zeros((3, 3))
    for b in bodies:
        inertia += _parallel_axis(b.inertia, b.mass, b.com - com)
    return RigidBody(mass=total_mass, com=com, inertia=inertia)


def format_urdf(body: RigidBody, indent: str = "\t\t") -> str:
    cx, cy, cz = body.com
    i = body.inertia
    return (
        f"{indent}<inertial>\n"
        f'{indent}\t<origin xyz="{cx:.6g} {cy:.6g} {cz:.6g}" rpy="0 0 0" />\n'
        f'{indent}\t<mass value="{body.mass:.6g}" />\n'
        f'{indent}\t<inertia ixx="{i[0, 0]:.6g}" ixy="{i[0, 1]:.6g}" ixz="{i[0, 2]:.6g}"\n'
        f'{indent}\t\tiyy="{i[1, 1]:.6g}" iyz="{i[1, 2]:.6g}" izz="{i[2, 2]:.6g}" />\n'
        f"{indent}</inertial>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", required=True, help="Path to printed-body STL.")
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help=(
            "Uniform scale applied to mesh vertices before computing inertia. "
            "Use 0.001 if the STL is exported in millimeters (common with Onshape/Fusion). "
            "Default: 1.0 (mesh already in meters)."
        ),
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.42,
        help="Effective density in g/cm³ (PLA solid ~1.24; PLA @ ~20%% infill ~0.42). Default: 0.42.",
    )
    parser.add_argument(
        "--visual-xyz",
        type=lambda s: _parse_triplet(s, "--visual-xyz"),
        default=np.zeros(3),
        help='URDF <visual>/<collision> origin xyz for the mesh, in meters. Default: "0 0 0".',
    )
    parser.add_argument(
        "--visual-rpy",
        type=lambda s: _parse_triplet(s, "--visual-rpy"),
        default=np.zeros(3),
        help='URDF <visual>/<collision> origin rpy for the mesh, in radians. Default: "0 0 0".',
    )
    parser.add_argument(
        "--no-payload",
        action="store_true",
        help="Skip the rigid payload; emit inertial for the printed body alone.",
    )
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=0.060,
        help="Payload mass in kg. Default: 0.060 (Intel RealSense D405).",
    )
    parser.add_argument(
        "--payload-xyz",
        type=lambda s: _parse_triplet(s, "--payload-xyz"),
        default=None,
        help="Payload CoM xyz in the LINK frame, in meters. Required unless --no-payload.",
    )
    parser.add_argument(
        "--payload-rpy",
        type=lambda s: _parse_triplet(s, "--payload-rpy"),
        default=np.zeros(3),
        help='Payload body-axes orientation in the link frame, in radians. Default: "0 0 0".',
    )
    parser.add_argument(
        "--payload-box",
        type=lambda s: _parse_triplet(s, "--payload-box"),
        default=np.array([0.042, 0.042, 0.023]),
        help='Payload box dimensions in meters (a b c). Default: "0.042 0.042 0.023" (D405 body).',
    )
    parser.add_argument(
        "--indent",
        default="\t\t",
        help="Leading indent for the emitted <inertial> block. Default: two tabs (matches so101_base.xacro).",
    )
    args = parser.parse_args()

    holder = mesh_body(
        mesh_path=args.mesh,
        density_g_per_cm3=args.density,
        visual_xyz=args.visual_xyz,
        visual_rpy=args.visual_rpy,
        mesh_scale=args.mesh_scale,
    )

    bodies: list[RigidBody] = [holder]
    if not args.no_payload:
        if args.payload_xyz is None:
            raise SystemExit("--payload-xyz is required unless --no-payload is passed.")
        bodies.append(
            box_payload(
                mass=args.payload_mass,
                com_link=args.payload_xyz,
                rpy=args.payload_rpy,
                dims=args.payload_box,
            )
        )

    total = combine(bodies)

    print("# Per-body summary (link frame)")
    labels = ["printed_body", "payload"] if not args.no_payload else ["printed_body"]
    for label, b in zip(labels, bodies):
        print(f"  {label:14s}  m={b.mass*1000:7.2f} g  com={np.array2string(b.com, precision=5)}")
    print(
        f"  {'combined':14s}  "
        f"m={total.mass*1000:7.2f} g  com={np.array2string(total.com, precision=5)}"
    )
    print()
    print(format_urdf(total, indent=args.indent))


if __name__ == "__main__":
    main()
