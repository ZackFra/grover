#!/usr/bin/env python3
"""SOFollower.configure() before ros2_control opens the follower USB port."""

from __future__ import annotations

import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


def configure(port: str) -> None:
    norm = MotorNormMode.DEGREES
    motors = {
        "shoulder_pan": Motor(1, "sts3215", norm),
        "shoulder_lift": Motor(2, "sts3215", norm),
        "elbow_flex": Motor(3, "sts3215", norm),
        "wrist_flex": Motor(4, "sts3215", norm),
        "wrist_roll": Motor(5, "sts3215", norm),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect()
    try:
        with bus.torque_disabled():
            bus.configure_motors()
            for name in bus.motors:
                bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                bus.write("P_Coefficient", name, 16)
                bus.write("I_Coefficient", name, 0)
                bus.write("D_Coefficient", name, 32)
                if name == "gripper":
                    bus.write("Max_Torque_Limit", name, 500)
                    bus.write("Protection_Current", name, 250)
                    bus.write("Overload_Torque", name, 25)
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    port = sys.argv[1]
    configure(port)
    print(f"Configured Feetech motors on {port}")
