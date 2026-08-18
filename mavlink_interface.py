"""MAVLink transport for body-frame follow commands.

This module intentionally does not arm the aircraft or change its flight mode.
The pilot remains responsible for those actions and for having a working RC
override/failsafe path.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from pymavlink import mavutil

