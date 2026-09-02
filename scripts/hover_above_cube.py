#!/usr/bin/env python3
"""Wait for a stable /red_cube/hover_pose, MoveIt there, run ACT, then go home.

Leaves the gripper above the cube (same XY, +Z offset from detect_red_cube.py)
with the orientation copied from the live gripper TF. Then runs the trained
LeRobot ACT policy for ``--infer-time-s`` (wrist RGB from the ROS D405 topic)
and MoveIt-returns to ``home``. Do not run teleop or RViz Plan & Execute
at the same time.

Bringup must include the D405 node (default ``enable_d405:=true``).

Usage:
    # bringup already running, not teleop
    python3 scripts/moveit_goto_joints.py top_view
    python3 scripts/detect_red_cube.py          # leave running; check /red_cube/overlay
    python3 scripts/hover_above_cube.py         # hover → grasp → home
    python3 scripts/hover_above_cube.py --plan-only
    python3 scripts/hover_above_cube.py --no-infer   # hover only
    python3 scripts/hover_above_cube.py --no-home
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path


def _preload_conda_openssl() -> None:
    """Load conda OpenSSL before rclpy/torch.

    Sourcing ROS puts Ubuntu ``libcrypto.so.3`` (3.0) on ``LD_LIBRARY_PATH``.
    Conda CPython ``_ssl`` needs ``OPENSSL_3.3.0`` from the env's libcrypto.
    Preloading that copy first makes later ``import ssl`` / ``import torch``
    bind to 3.3 instead of the system 3.0.
    """
    lib = Path(sys.prefix) / "lib"
    for name in ("libcrypto.so.3", "libssl.so.3"):
        path = lib / name
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


_preload_conda_openssl()

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import JointConstraint
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

# Same directory as this file (python3 scripts/hover_above_cube.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moveit_goto_joints import execute_named_pose  # noqa: E402
from moveit_goto_pose import format_moveit_result, send_pose  # noqa: E402
from ros_image_camera import ROSImageCameraConfig  # noqa: E402

_WRIST_HOLD_JOINTS = ("wrist_roll", "wrist_flex")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET_ROOT = _REPO_ROOT / "outputs" / "datasets" / "so101_d405_wrist"
_DEFAULT_CKPT_ROOT = (
    _REPO_ROOT / "outputs" / "train" / "act_so101_d405_wrist_v2" / "checkpoints"
)
_DEFAULT_POLICY = _DEFAULT_CKPT_ROOT / "100000" / "pretrained_model"


def _hold_joint(name: str, position: float, tol: float) -> JointConstraint:
    jc = JointConstraint()
    jc.joint_name = name
    jc.position = float(position)
    jc.tolerance_above = float(tol)
    jc.tolerance_below = float(tol)
    jc.weight = 1.0
    return jc


def _current_wrist_holds(roll_tol: float, flex_tol: float) -> list[JointConstraint]:
    """Keep wrist_roll/flex at the live top_view angles so hover does not spin."""
    node = Node("hover_above_cube_joints")
    found: dict[str, float] = {}

    def _cb(msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            if n in _WRIST_HOLD_JOINTS:
                found[n] = float(p)

    node.create_subscription(JointState, "/joint_states", _cb, 10)
    deadline = time.monotonic() + 2.0
    try:
        while time.monotonic() < deadline and not all(
            n in found for n in _WRIST_HOLD_JOINTS
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()

    holds: list[JointConstraint] = []
    if "wrist_roll" in found and roll_tol > 0.0:
        holds.append(_hold_joint("wrist_roll", found["wrist_roll"], roll_tol))
    if "wrist_flex" in found and flex_tol > 0.0:
        holds.append(_hold_joint("wrist_flex", found["wrist_flex"], flex_tol))
    return holds


def _default_policy_path() -> Path:
    if _DEFAULT_POLICY.exists():
        return _DEFAULT_POLICY
    last = _DEFAULT_CKPT_ROOT / "last" / "pretrained_model"
    if last.exists():
        return last
    numbered = sorted(
        (p for p in _DEFAULT_CKPT_ROOT.glob("*") if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if numbered:
        return numbered[-1] / "pretrained_model"
    return _DEFAULT_POLICY


def run_grasp_inference(
    *,
    policy_path: Path,
    dataset_root: Path,
    repo_id: str,
    task: str,
    infer_time_s: float,
    fps: int,
    color_topic: str,
) -> None:
    """Run ACT on the parked arm using the ROS D405 stream, then disconnect."""
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.processor import make_default_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.robot_utils import precise_sleep

    register_third_party_plugins()
    from lerobot_robot_ros.config import ActionType, SO101ROSConfig

    if not policy_path.exists():
        raise FileNotFoundError(f"No policy checkpoint at {policy_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"No dataset at {dataset_root} (needed for ACT normalization stats)")

    dataset = LeRobotDataset(repo_id, root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    robot = make_robot_from_config(
        SO101ROSConfig(
            id="gi_jane",
            action_type=ActionType.JOINT_POSITION,
            convert_so101_leader_units=True,
            cameras={
                "wrist": ROSImageCameraConfig(
                    topic=color_topic,
                    fps=fps,
                    width=640,
                    height=480,
                )
            },
        )
    )
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    robot.connect()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    device = get_safe_torch_device(policy.config.device)
    print(
        f"ACT grasp for {infer_time_s:.1f}s from {policy_path} (device={device})",
        flush=True,
    )
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < infer_time_s:
            loop_t = time.perf_counter()
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            act = make_robot_action(action_values, dataset.features)
            robot.send_action(robot_action_processor((act, obs)))
            precise_sleep(max(1.0 / fps - (time.perf_counter() - loop_t), 0.0))
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("ACT grasp finished.", flush=True)


def _span(values: list[float]) -> float:
    return max(values) - min(values) if values else float("inf")


def wait_for_stable_hover(
    topic: str,
    *,
    samples: int,
    max_age_s: float,
    max_span_m: float,
    timeout_s: float,
) -> PoseStamped:
    """Collect `samples` fresh poses whose XYZ box is smaller than `max_span_m`."""
    node = Node("hover_above_cube_wait")
    buffer: list[tuple[float, PoseStamped]] = []

    def _cb(msg: PoseStamped) -> None:
        buffer.append((time.monotonic(), msg))

    node.create_subscription(PoseStamped, topic, _cb, qos_profile_sensor_data)
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    n_seen = 0
    last_span = (float("inf"), float("inf"), float("inf"))
    n_fresh = 0
    stable: PoseStamped | None = None
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            n_seen = len(buffer)
            fresh = [(t, m) for t, m in buffer if (now - t) <= max_age_s]
            buffer[:] = fresh[-max(samples * 4, samples) :]
            n_fresh = len(fresh)
            if n_fresh >= 2:
                window = [m for _, m in fresh[-min(samples, n_fresh) :]]
                xs = [p.pose.position.x for p in window]
                ys = [p.pose.position.y for p in window]
                zs = [p.pose.position.z for p in window]
                last_span = (_span(xs), _span(ys), _span(zs))
            if now - last_print >= 2.0:
                print(
                    f"  waiting: {n_fresh} fresh / {n_seen} total, "
                    f"span dx={last_span[0]:.3f} dy={last_span[1]:.3f} "
                    f"dz={last_span[2]:.3f} m (need {samples} samples, "
                    f"xy span <= {max_span_m:.3f})",
                    flush=True,
                )
                last_print = now
            if n_fresh < samples:
                continue
            window = [m for _, m in fresh[-samples:]]
            xs = [p.pose.position.x for p in window]
            ys = [p.pose.position.y for p in window]
            zs = [p.pose.position.z for p in window]
            last_span = (_span(xs), _span(ys), _span(zs))
            # XY must agree; Z is allowed to be noisier (D405 depth + offset).
            if last_span[0] > max_span_m or last_span[1] > max_span_m:
                continue
            if last_span[2] > max(max_span_m, 0.20):
                continue
            avg = PoseStamped()
            avg.header = window[-1].header
            avg.pose.position.x = sorted(xs)[len(xs) // 2]
            avg.pose.position.y = sorted(ys)[len(ys) // 2]
            avg.pose.position.z = sorted(zs)[len(zs) // 2]
            avg.pose.orientation = window[-1].pose.orientation
            stable = avg
            break
    finally:
        node.destroy_node()

    if stable is None:
        raise TimeoutError(
            f"No stable pose on '{topic}' within {timeout_s:.1f}s "
            f"(got {n_fresh} fresh / {n_seen} total, "
            f"span dx={last_span[0]:.3f} dy={last_span[1]:.3f} "
            f"dz={last_span[2]:.3f} m; need {samples} samples, "
            f"age <= {max_age_s:.2f}s, xy span <= {max_span_m:.3f} m)."
        )
    return stable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/red_cube/hover_pose")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Poses to median-filter once they agree (default 5).",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=1.0,
        help="Drop samples older than this many seconds (default 1.0).",
    )
    parser.add_argument(
        "--max-span",
        type=float,
        default=0.05,
        help="Max XYZ range (m) across the sample window (default 0.05; D405 Z is noisy).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a stable hover pose (default 10).",
    )
    parser.add_argument("--group", default="arm")
    parser.add_argument("--link", default="gripper")
    parser.add_argument("--vel", type=float, default=0.3)
    parser.add_argument("--accel", type=float, default=0.3)
    parser.add_argument(
        "--position-radius",
        type=float,
        default=0.012,
        help="MoveIt goal sphere radius (m); smaller = tighter XY (default 0.012).",
    )
    parser.add_argument(
        "--orient-tol",
        type=float,
        default=0.6,
        help=(
            "Tilt tolerance (rad) so the gripper stays pointing down (default 0.6). "
            "Pass 0 for position-only."
        ),
    )
    parser.add_argument(
        "--wrist-roll-tol",
        type=float,
        default=0.12,
        help="Hold wrist_roll within this many radians of the current angle (default 0.12). 0 disables.",
    )
    parser.add_argument(
        "--wrist-flex-tol",
        type=float,
        default=0.20,
        help="Hold wrist_flex within this many radians of the current angle (default 0.20). 0 disables.",
    )
    parser.add_argument("--planning-time", type=float, default=20.0)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan but don't execute.",
    )
    parser.add_argument(
        "--no-infer",
        action="store_true",
        help="Stop after hover; do not run the ACT policy.",
    )
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Skip the MoveIt return to home after inference.",
    )
    parser.add_argument(
        "--infer-time-s",
        type=float,
        default=12.0,
        help="Seconds to run ACT after hover (default 12).",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help=(
            "ACT checkpoint dir (default: "
            "outputs/train/act_so101_d405_wrist_v2/checkpoints/100000/pretrained_model)."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_DEFAULT_DATASET_ROOT,
        help="Training dataset root (normalization stats).",
    )
    parser.add_argument("--repo-id", default="local/so101_d405_wrist")
    parser.add_argument("--task", default="pick up the cube")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--color-topic",
        default="/d405_wrist/color/image_raw",
        help="ROS D405 RGB topic used as the policy wrist camera.",
    )
    parser.add_argument("--home-pose", default="home")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.samples < 1:
        print("ERROR: --samples must be >= 1", file=sys.stderr)
        return 1

    own_ctx = not rclpy.ok()
    if own_ctx:
        rclpy.init()
    result = None
    try:
        try:
            pose = wait_for_stable_hover(
                args.topic,
                samples=args.samples,
                max_age_s=args.max_age,
                max_span_m=args.max_span,
                timeout_s=args.timeout,
            )
        except TimeoutError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        p = pose.pose.position
        print(
            f"Stable hover in '{pose.header.frame_id}': "
            f"xyz=({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f})"
        )
        wrist_holds = _current_wrist_holds(args.wrist_roll_tol, args.wrist_flex_tol)
        if wrist_holds:
            print(
                "  holding wrist at "
                + ", ".join(
                    f"{jc.joint_name}={jc.position:+.3f}±{jc.tolerance_above:.3f}"
                    for jc in wrist_holds
                ),
                flush=True,
            )
        else:
            print("  warning: no wrist_roll/wrist_flex in /joint_states; wrist not locked")
        result = send_pose(
            pose,
            group=args.group,
            link_name=args.link,
            plan_only=args.plan_only,
            vel_scale=args.vel,
            accel_scale=args.accel,
            position_radius=args.position_radius,
            orient_tol=args.orient_tol,
            planning_time_s=args.planning_time,
            attempts=args.attempts,
            extra_joint_constraints=wrist_holds,
        )

        exit_code, message = format_moveit_result(result)
        print(f"\n{message}")
        if exit_code != 0 or args.plan_only:
            return exit_code

        if args.no_infer:
            print("Parked above the cube. Skipping ACT (--no-infer).")
            return 0

        print("Parked above the cube. Running ACT grasp. Do not teleop or Plan & Execute.")
        infer_failed = False
        try:
            run_grasp_inference(
                policy_path=args.policy_path or _default_policy_path(),
                dataset_root=args.dataset_root,
                repo_id=args.repo_id,
                task=args.task,
                infer_time_s=args.infer_time_s,
                fps=args.fps,
                color_topic=args.color_topic,
            )
        except Exception as exc:
            print(f"ERROR: ACT inference failed: {exc}", file=sys.stderr)
            infer_failed = True

        if args.no_home:
            print("Skipping home (--no-home).")
            return 4 if infer_failed else 0

        print("Returning home.", flush=True)
        home_code = execute_named_pose(args.home_pose, vel=args.vel, accel=args.accel)
        if infer_failed:
            return 4
        return home_code
    finally:
        if own_ctx:
            rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
