import pymavlink as mv
from pymavlink import mavutil

link = mavutil.mavlink_connection(
    "/dev/serial0",
    baud=921600,
    source_system=245,
)

link.wait_heartbeat(timeout=10)