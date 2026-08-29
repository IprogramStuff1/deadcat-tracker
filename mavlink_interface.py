from __future__ import annotations

import math
import time

from pymavlink import mavutil

SERIAL_DEVICE = "/dev/serial0"
SERIAL_BAUD = 57600
SOURCE_SYSTEM_ID = 245

FOLLOW_FRAME = mavutil.mavlink.MAV_FRAME_BODY_NED
TYPE_MASK = 1479

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