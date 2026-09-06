"""Hardware-independent command validation and ArduCopter tracking gate."""

from __future__ import annotations

import math

from navigation import SimpleProportionalControl


def command_from_sample(sample, now: float, timeout: float):
    """Return a bounded command only for a recent, finite target observation."""
    if sample is None or not math.isfinite(sample.timestamp):
        return None
    if not 0 <= now - sample.timestamp <= timeout:
        return None
    errors = sample.error_vector
    if errors is None or len(errors) != 4 or not all(map(math.isfinite, errors)):
        return None
    forward, yaw = SimpleProportionalControl(errors)
    if not math.isfinite(forward) or not math.isfinite(yaw):
        return None
    return float(forward), float(yaw)


class TrackingGate:
    """Acquire control on an observed non-GUIDED -> armed GUIDED transition.

    None means do not transmit. (0, 0) requests zero velocity/yaw rate while
    this process still owns an armed GUIDED session. Faults never auto-resume.
    """

    def __init__(self, command_timeout=0.5, heartbeat_timeout=3.0,
                 target_loss_timeout=2.0):
        for value in (command_timeout, heartbeat_timeout, target_loss_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be finite and positive")
        self.command_timeout = command_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.target_loss_timeout = target_loss_timeout
        self.last_heartbeat = None
        self.guided = False
        self.armed = False
        self.ready = False
        self.engaged = False
        self.fault = None
        self.session = 0
        self.enabled_at = None
        self.last_target_at = None
        self.status = "waiting for a non-GUIDED heartbeat"

    def latch(self, reason):
        self.ready = False
        if self.engaged and self.fault is None:
            self.fault = reason
        self.status = reason

    def observe_heartbeat(self, now, *, guided, armed):
        # Also detect gaps when a fresh heartbeat arrives before the next tick.
        if (self.last_heartbeat is not None
                and now - self.last_heartbeat > self.heartbeat_timeout):
            self.latch("heartbeat lost; switch out of GUIDED and back to resume")
        was_guided = self.guided
        self.last_heartbeat = now
        self.guided = guided
        self.armed = armed
        if not guided:
            self.engaged = False
            self.ready = True
            self.fault = None
            self.status = "paused; switch to GUIDED after manual takeoff"
        elif not armed:
            self.engaged = False
            self.ready = False
            self.fault = None
            self.status = "disarmed; switch out of GUIDED before enabling"
        elif not was_guided and self.ready:
            self.engaged = True
            self.ready = False
            self.fault = None
            self.session += 1
            self.enabled_at = now
            self.last_target_at = now
            self.status = "enabled; waiting for a fresh target"

    def can_send(self, now):
        return (self.engaged and self.guided and self.armed
                and self.last_heartbeat is not None
                and 0 <= now - self.last_heartbeat <= self.heartbeat_timeout)

    def command(self, sample, now):
        if (self.last_heartbeat is not None
                and now - self.last_heartbeat > self.heartbeat_timeout):
            self.latch("heartbeat lost; switch out of GUIDED and back to resume")
        if not self.can_send(now):
            return None
        if self.fault:
            self.status = self.fault
            return 0.0, 0.0
        # Recovery between ticks must not hide an already elapsed loss timeout.
        if now - self.last_target_at >= self.target_loss_timeout:
            self.latch("target/camera data lost; switch out of GUIDED and back to resume")
            return 0.0, 0.0
        command = command_from_sample(sample, now, self.command_timeout)
        if sample is not None and sample.timestamp < self.enabled_at:
            command = None
        if command is not None:
            self.last_target_at = max(self.last_target_at, sample.timestamp)
            self.status = "following"
            return command
        self.status = "target missing or stale; requesting zero motion"
        return 0.0, 0.0
