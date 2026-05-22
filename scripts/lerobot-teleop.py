#!/usr/bin/env python3
"""Python equivalent of scripts/lerobot-teleop.sh.

HW teleop: joint_jog -> MoveIt Servo (collision checking) -> /arm_controller/commands.

Same defaults as the bash script. Use this when you want to edit `SO101ROSConfig`
in place, attach a debugger, or extend the teleop loop. For the bash-only path,
see scripts/lerobot-teleop.sh.

Usage:
    ./scripts/lerobot-teleop.py
    DISPLAY_DATA=false ./scripts/lerobot-teleop.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pprint import pformat

import rerun as rr

from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleop_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

ROBOT_ID = "gi_jane"
LEADER_ID = "gi_joe"
LEADER_PORT = "/dev/so101_leader"


def main() -> None:
    register_third_party_plugins()

    # Lazy import: lerobot-ros registers `so101_ros` on import; importing it after
    # register_third_party_plugins keeps any registry-related errors readable.
    from lerobot_robot_ros.config import ActionType, SO101ROSConfig

    display_data = os.environ.get("DISPLAY_DATA", "true") != "false"

    cfg = TeleoperateConfig(
        robot=SO101ROSConfig(
            id=ROBOT_ID,
            action_type=ActionType.JOINT_JOG,
            convert_so101_leader_units=True,
        ),
        teleop=SO101LeaderConfig(id=LEADER_ID, port=LEADER_PORT),
        display_data=display_data,
    )

    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    teleop.connect()
    robot.connect()
    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_proc,
            robot_action_processor=robot_action_proc,
            robot_observation_processor=robot_obs_proc,
            display_compressed_images=cfg.display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
