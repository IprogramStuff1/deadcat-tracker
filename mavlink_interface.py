#MANGO TUFF PYTHON CODE [anyone reading this ignore]
from __future__ import annotations

import math
import threading
import time
from typing import Any

from pymavlink import mavutil


link = mavutil.mavlink_connection(
    "/dev/serial0",
    baud=57600,
    source_system=245,
)

heartbeat = link.wait_heartbeat(timeout=10)

if heartbeat is None:
    raise TimeoutError("No MAVLink heartbeat received")

def follow_vision_command(forward_m_s, yaw_rad_s):
    link.mav.set_position_target_local_ned_send(
        int(time.monotonic() * 1000) & 0xFFFFFFFF,
        link.target_system,
        link.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        1479,

    0.0,0.0,0.0,
    forward_m_s, 0.0, 0.0,
    0.0,0.0,0.0,
    0.0,
    yaw_rad_s,
    )