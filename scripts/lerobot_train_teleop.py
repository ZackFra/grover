#!/usr/bin/env python3
"""Python equivalent of the stock LeRobot record / train / eval CLI.

Same configs you would pass to ``lerobot-record`` and ``lerobot-train``,
built as dataclasses and handed to ``record()`` / ``train()``. Wrist RGB
comes from the D405 via ``RealSenseCameraConfig`` (``type=intelrealsense``).

Bringup must already own the Feetech bus, but **not** the D405 USB device —
LeRobot opens the camera itself. Launch with:

    ros2 launch launch/so101_bringup.launch.py is_sim:=False enable_d405:=false

CLI this script mirrors
-----------------------
Record teleop demos (right arrow ends an episode early)::

    lerobot-record \\
      --robot.type=so101_ros --robot.id=gi_jane \\
      --robot.action_type=joint_position --robot.convert_so101_leader_units=true \\
      --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: 'Intel RealSense D405', width: 640, height: 480, fps: 30} }" \\
      --teleop.type=so101_leader --teleop.id=gi_joe --teleop.port=/dev/so101_leader \\
      --dataset.repo_id=local/so101_d405_wrist \\
      --dataset.root=outputs/datasets/so101_d405_wrist \\
      --dataset.single_task='pick up the cube' \\
      --dataset.num_episodes=20 --dataset.episode_time_s=60 --dataset.reset_time_s=60 \\
      --dataset.push_to_hub=false --display_data=true

Train ACT::

    lerobot-train \\
      --dataset.repo_id=local/so101_d405_wrist --dataset.root=outputs/datasets/so101_d405_wrist \\
      --policy.type=act --output_dir=outputs/train/act_so101_d405_wrist \\
      --job_name=act_so101_d405_wrist --policy.device=cuda --policy.push_to_hub=false \\
      --steps=20000 --eval_freq=0

Run the policy; **between each episode** ``record()`` gives you ``reset_time_s``
of leader teleop (no frames saved) so you can park the SO-101 in front of the
target again. Right arrow skips the rest of the current episode or reset::

    lerobot-record \\
      --robot.type=so101_ros ...same robot / cameras / teleop... \\
      --policy.path=outputs/train/act_so101_d405_wrist/checkpoints/last/pretrained_model \\
      --dataset.repo_id=local/eval_so101_d405_wrist --dataset.root=outputs/datasets/eval_so101_d405_wrist \\
      --dataset.single_task='pick up the cube' \\
      --dataset.num_episodes=10 --dataset.reset_time_s=60 --dataset.push_to_hub=false

Usage::

    python3 scripts/lerobot_train_teleop.py --mode record --task 'pick up the cube'
    python3 scripts/lerobot_train_teleop.py --mode train
    python3 scripts/lerobot_train_teleop.py --mode eval
    python3 scripts/lerobot_train_teleop.py --mode all --task 'pick up the cube'
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.default import DatasetConfig as TrainDatasetConfig
from lerobot.configs.default import WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.scripts.lerobot_train import train
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

REPO_ROOT = Path(__file__).resolve().parents[1]

ROBOT_ID = "gi_jane"
LEADER_ID = "gi_joe"
LEADER_PORT = "/dev/so101_leader"

# Matches config/realsense-d405/d405.json and so101_bringup 640x480@30.
D405_NAME = "Intel RealSense D405"
D405_WIDTH = 640
D405_HEIGHT = 480
D405_FPS = 30

DEFAULT_REPO_ID = "local/so101_d405_wrist"
DEFAULT_EVAL_REPO_ID = "local/eval_so101_d405_wrist"
DEFAULT_DATASETS_DIR = REPO_ROOT / "outputs" / "datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "train" / "act_so101_d405_wrist"
DEFAULT_TASK = "pick up the cube"


def _dataset_root(datasets_dir: Path, repo_id: str) -> Path:
    """Workspace folder for a repo_id: outputs/datasets/<name> (not HF cache)."""
    return Path(datasets_dir) / repo_id.split("/", 1)[-1]


def _display_data() -> bool:
    return os.environ.get("DISPLAY_DATA", "true") != "false"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _camera_serial(cam: dict) -> str:
    # lerobot 0.5.1 find_cameras() stores the SN as "id"; name lookup still
    # reads "serial_number" and KeyErrors. Always pass a numeric SN to skip that path.
    serial = cam.get("id") or cam.get("serial_number")
    if not serial:
        raise ValueError(f"RealSense listing has no serial: {cam}")
    return str(serial)


def _resolve_d405_serial(serial_or_name: str) -> str:
    if serial_or_name.isdigit():
        return serial_or_name

    cameras = RealSenseCamera.find_cameras()
    if not cameras:
        raise RuntimeError(
            "No RealSense cameras found. Unplug anything else using the D405 "
            "(bringup must use enable_d405:=false) and retry. "
            "Or pass the serial from `lerobot-find-cameras realsense` as --camera."
        )

    exact = [c for c in cameras if str(c.get("name", "")) == serial_or_name]
    if not exact:
        exact = [c for c in cameras if serial_or_name.lower() in str(c.get("name", "")).lower()]
    if not exact and serial_or_name == D405_NAME:
        exact = [c for c in cameras if "d405" in str(c.get("name", "")).lower()]

    if not exact:
        listing = ", ".join(f"{c.get('name')} (SN {c.get('id')})" for c in cameras)
        raise RuntimeError(
            f"No RealSense camera matching '{serial_or_name}'. Connected: {listing}"
        )
    if len(exact) > 1:
        sns = [_camera_serial(c) for c in exact]
        raise RuntimeError(
            f"Multiple cameras match '{serial_or_name}': {sns}. Pass one serial as --camera."
        )

    serial = _camera_serial(exact[0])
    logging.info("Using D405 serial %s (%s)", serial, exact[0].get("name"))
    return serial


def _d405_cameras(serial_number_or_name: str) -> dict[str, RealSenseCameraConfig]:
    serial = _resolve_d405_serial(serial_number_or_name)
    return {
        "wrist": RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=D405_FPS,
            width=D405_WIDTH,
            height=D405_HEIGHT,
        )
    }


def _robot_config(serial_number_or_name: str):
    from lerobot_robot_ros.config import ActionType, SO101ROSConfig

    return SO101ROSConfig(
        id=ROBOT_ID,
        action_type=ActionType.JOINT_POSITION,
        convert_so101_leader_units=True,
        cameras=_d405_cameras(serial_number_or_name),
    )


def _teleop_config() -> SO101LeaderConfig:
    return SO101LeaderConfig(id=LEADER_ID, port=LEADER_PORT)


def _record_config(
    *,
    repo_id: str,
    root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: float,
    reset_time_s: float,
    resume: bool,
    camera: str,
    policy: PreTrainedConfig | None = None,
) -> RecordConfig:
    return RecordConfig(
        robot=_robot_config(camera),
        teleop=_teleop_config(),
        policy=policy,
        dataset=DatasetRecordConfig(
            repo_id=repo_id,
            root=root,
            single_task=task,
            fps=D405_FPS,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
            num_episodes=num_episodes,
            push_to_hub=False,
        ),
        display_data=_display_data(),
        play_sounds=False,
        resume=resume,
    )


def _train_config(
    *, repo_id: str, root: Path, output_dir: Path, steps: int, batch_size: int
) -> TrainPipelineConfig:
    return TrainPipelineConfig(
        dataset=TrainDatasetConfig(repo_id=repo_id, root=str(root)),
        policy=ACTConfig(device=_device(), push_to_hub=False),
        output_dir=output_dir,
        job_name=output_dir.name,
        steps=steps,
        batch_size=batch_size,
        eval_freq=0,
        wandb=WandBConfig(enable=False),
    )


def _load_policy(policy_path: Path) -> PreTrainedConfig:
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    return cfg


def _last_checkpoint(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "last" / "pretrained_model"


def run_record(args: argparse.Namespace) -> None:
    root = _dataset_root(args.datasets_dir, args.repo_id)
    logging.info("Dataset root: %s", root)
    cfg = _record_config(
        repo_id=args.repo_id,
        root=root,
        task=args.task,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        resume=args.resume,
        camera=args.camera,
    )
    logging.info(pformat(asdict(cfg)))
    logging.info(
        "Recording teleop demos. After each episode you get %.0fs of teleop "
        "(not recorded) to put the SO-101 in front of the target again. "
        "Right arrow = next, left arrow = rerecord, Esc = stop.",
        args.reset_time_s,
    )
    record(cfg)


def run_train(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    root = _dataset_root(args.datasets_dir, args.repo_id)
    logging.info("Training from dataset root: %s", root)
    cfg = _train_config(
        repo_id=args.repo_id,
        root=root,
        output_dir=output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    logging.info(pformat(asdict(cfg)))
    train(cfg)
    ckpt = _last_checkpoint(output_dir)
    logging.info("Training finished. Policy checkpoint: %s", ckpt)
    return ckpt


def run_eval(args: argparse.Namespace, policy_path: Path | None = None) -> None:
    path = Path(policy_path or args.policy_path or _last_checkpoint(Path(args.output_dir)))
    if not path.exists():
        raise FileNotFoundError(
            f"No policy at {path}. Train first (--mode train) or pass --policy-path."
        )
    cfg = _record_config(
        repo_id=args.eval_repo_id,
        root=_dataset_root(args.datasets_dir, args.eval_repo_id),
        task=args.task,
        num_episodes=args.num_eval_episodes,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        resume=False,
        camera=args.camera,
        policy=_load_policy(path),
    )
    logging.info(pformat(asdict(cfg)))
    logging.info(
        "Policy episodes from %s. Between each run, teleop the SO-101 in "
        "front of the target (%.0fs, right arrow when ready).",
        path,
        args.reset_time_s,
    )
    record(cfg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("record", "train", "eval", "all"),
        default="all",
        help="record = teleop demos; train = ACT offline; eval = policy + teleop reset; all = three in order.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help="Parent dir for local datasets (default: <workspace>/outputs/datasets).",
    )
    parser.add_argument(
        "--eval-repo-id",
        default=DEFAULT_EVAL_REPO_ID,
        help="Must start with eval_ when --mode eval (LeRobot naming rule).",
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument(
        "--reset-time-s",
        type=float,
        default=60.0,
        help="Teleop window after each episode to aim the wrist at the target again.",
    )
    parser.add_argument("--resume", action="store_true", help="Append episodes to an existing dataset.")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Checkpoint dir for --mode eval (default: <output-dir>/checkpoints/last/pretrained_model).",
    )
    parser.add_argument(
        "--camera",
        default=D405_NAME,
        help="D405 serial or RealSense name (default: 'Intel RealSense D405'). "
        "Override with the serial from `lerobot-find-cameras realsense` if the name is not unique.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    register_third_party_plugins()
    init_logging()

    if args.mode in ("record", "all"):
        run_record(args)
    if args.mode in ("train", "all"):
        ckpt = run_train(args)
    else:
        ckpt = None
    if args.mode in ("eval", "all"):
        run_eval(args, policy_path=ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
