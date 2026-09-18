"""Run camera preview, or opt in to RC-switched ArduCopter following."""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import signal
import threading
import time

from control import TrackingGate, command_from_sample
from vision import VisionSource

LOG = logging.getLogger("tracker")
SEND_HZ = 10.0
COMMAND_TIMEOUT = 0.5
HEARTBEAT_TIMEOUT = 3.0
TARGET_LOSS_TIMEOUT = 2.0
CAMERA_STARTUP_TIMEOUT = 60.0


class VisionWorker(threading.Thread):
    """Keep slow camera operations off the control loop; store one latest sample."""

    def __init__(self, source_factory):
        super().__init__(name="vision", daemon=True)
        self.source_factory = source_factory
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.generation = 0
        self.reset_at = float("-inf")

    def reset_tracker(self):
        with self.lock:
            self.generation += 1
            self.reset_at = time.monotonic()
            self.latest = None

    def snapshot(self):
        with self.lock:
            return self.latest, self.error

    def run(self):
        try:
            with self.source_factory() as source:
                generation = -1
                self.ready.set()
                while not self.stop_event.is_set():
                    with self.lock:
                        requested_generation = self.generation
                    if generation != requested_generation:
                        source.reset_tracker()
                        generation = requested_generation
                    sample = source.read_latest()
                    if sample is not None:
                        now = time.monotonic()
                        if not math.isfinite(sample.timestamp) or sample.timestamp > now:
                            raise ValueError("Camera returned an invalid timestamp")
                        with self.lock:
                            if (generation == self.generation
                                    and sample.timestamp >= self.reset_at
                                    and (self.latest is None
                                         or sample.timestamp > self.latest.timestamp)):
                                self.latest = sample
                    self.stop_event.wait(0.005)
        except Exception as exc:
            with self.lock:
                self.latest = None
                self.error = exc
        finally:
            self.ready.set()


class MavlinkSession:
    """Use the existing UART module unchanged, and accept only its FC heartbeat."""

    def __init__(self, interface):
        self.interface = interface
        self.link = interface.link
        self.mavutil = interface.mavutil
        self.system = self.link.target_system
        # pymavlink can leave target_component at 0 (broadcast) after heartbeat.
        # Heartbeat monitoring must match the autopilot's actual source component.
        self.component = self.mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
        if not self.is_controller_heartbeat(interface.heartbeat):
            raise RuntimeError("Expected the Pixhawk ArduCopter heartbeat on the configured UART")

    def is_controller_heartbeat(self, message):
        m = self.mavutil.mavlink
        return (message is not None and message.get_type() == "HEARTBEAT"
                and message.get_srcSystem() == self.system
                and message.get_srcComponent() == self.component
                and message.autopilot == m.MAV_AUTOPILOT_ARDUPILOTMEGA
                and message.type == m.MAV_TYPE_QUADROTOR)

    def observe(self, message, gate, now):
        if self.is_controller_heartbeat(message):
            gate.observe_heartbeat(
                now,
                guided=message.custom_mode == 4,  # ArduCopter GUIDED
                armed=bool(message.base_mode
                           & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
            )

    def poll(self, gate, now):
        # Bound work even when other telemetry is arriving continuously.
        for _ in range(100):
            message = self.link.recv_match(blocking=False)
            if message is None:
                break
            self.observe(message, gate, now)

    def send(self, command):
        self.interface.follow_vision_command(*command)

    def close(self):
        self.link.close()


class VisionDiagnostics:
    """Report observation throughput and age without treating 10 Hz ticks as FPS."""

    def __init__(self):
        self.last_time = None
        self.last_count = 0
        self.fps = 0.0

    def describe(self, sample, now, standoff):
        count = getattr(sample, "frame_count", self.last_count)
        if self.last_time is None:
            self.last_time, self.last_count = now, count
        elif now - self.last_time >= 1.0:
            self.fps = max(0, count - self.last_count) / (now - self.last_time)
            self.last_time, self.last_count = now, count
        if sample is None:
            return f"vision_fps={self.fps:.1f} age_ms=none last_target_xyz_m=none"
        age_ms = (now - sample.timestamp) * 1000
        errors = sample.error_vector
        if errors is None:
            target = "last_target_xyz_m=none"
        else:
            target = (f"last_target_xyz_m=({errors[0] + standoff:.2f},"
                      f"{errors[1]:.2f},{errors[2]:.2f}) "
                      f"yaw_error_deg={math.degrees(errors[3]):.1f}")
        tracking = getattr(sample, "tracking_status", "unknown")
        return f"vision_fps={self.fps:.1f} age_ms={age_ms:.0f} {target} tracker={tracking}"


def run(args, stop_event):
    snapshot_dir = getattr(args, "snapshot_dir", None)
    if args.live and snapshot_dir is not None:
        raise ValueError("Diagnostic snapshots are only available in dry-run mode")
    worker = VisionWorker(lambda: VisionSource(
        standoff=args.standoff, snapshot_dir=snapshot_dir,
        min_depth=getattr(args, "min_depth", 0.3), max_depth=getattr(args, "max_depth", 10.0)))
    gate = TrackingGate(COMMAND_TIMEOUT, HEARTBEAT_TIMEOUT, TARGET_LOSS_TIMEOUT)
    session = None
    worker.start()
    try:
        LOG.info("Starting camera (%s)", "live requested" if args.live else "dry run; no MAVLink")
        deadline = time.monotonic() + CAMERA_STARTUP_TIMEOUT
        while not worker.ready.wait(0.1):
            if stop_event.is_set():
                return 0
            if time.monotonic() >= deadline:
                raise RuntimeError("Camera startup timed out; check OAK USB/power and model availability")
        _, error = worker.snapshot()
        if error:
            raise RuntimeError(f"Camera startup failed: {error}") from error
        if stop_event.is_set():
            return 0
        if args.live:
            # Importing this module opens UART. Never import it for preview/help/tests.
            interface = importlib.import_module("mavlink_interface")
            try:
                session = MavlinkSession(interface)
            except Exception:
                interface.link.close()
                raise
            session.observe(interface.heartbeat, gate, time.monotonic())
            LOG.info("Live: manually take off, then switch out of GUIDED and into GUIDED to enable")
        log_at = 0.0
        previous_status = None
        diagnostics = VisionDiagnostics()
        while not stop_event.is_set():
            now = time.monotonic()
            previous_session = gate.session
            if session:
                session.poll(gate, now)
                if gate.session != previous_session:
                    worker.reset_tracker()
            sample, error = worker.snapshot()
            if error:
                gate.latch("camera failed; restart the program and cycle GUIDED")
                raise RuntimeError(f"Camera failed: {error}") from error
            now = time.monotonic()
            if session:
                command = gate.command(sample, now)
                if command is not None:
                    session.send(command)
                status = gate.status
            else:
                preview = command_from_sample(sample, now, COMMAND_TIMEOUT)
                command = preview if preview is not None else (0.0, 0.0)
                status = "dry run; target present" if preview is not None else "dry run; target missing or stale"
            if status != previous_status or now >= log_at:
                LOG.info("%s | forward/yaw=%s | %s", status, command,
                         diagnostics.describe(sample, now, args.standoff))
                log_at = now + 1.0
                previous_status = status
            # Skip missed periods; never burst old setpoints after a slow operation.
            stop_event.wait(max(0.0, 1.0 / SEND_HZ - (time.monotonic() - now)))
        return 0
    finally:
        if session:
            try:
                session.poll(gate, time.monotonic())
                if gate.can_send(time.monotonic()):
                    session.send((0.0, 0.0))
            except Exception as exc:
                LOG.warning("Could not send final zero-motion request: %s", exc)
            finally:
                try:
                    session.close()
                except Exception as exc:
                    LOG.warning("Could not close MAVLink: %s", exc)
        worker.stop_event.set()
        worker.join(timeout=2.0)
        if worker.is_alive():
            LOG.warning("Camera worker is still blocked; it cannot transmit MAVLink commands")


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="enable RC-gated MAVLink commands on the existing UART")
    parser.add_argument("--standoff", type=positive_number, default=2.0, help="target distance in meters (default: 2)")
    parser.add_argument("--snapshot-dir", help="dry run only: save up to 30 annotated camera images, at most one per second")
    parser.add_argument("--min-depth", type=positive_number, default=0.3, help="minimum accepted target depth in metres (default: 0.3)")
    parser.add_argument("--max-depth", type=positive_number, default=10.0, help="maximum accepted target depth in metres (default: 10)")
    args = parser.parse_args(argv)
    if not 0 < round(args.min_depth * 1000) < round(args.max_depth * 1000) <= 65535:
        parser.error("depth limits must be positive, increasing at millimetre precision, and at most 65.535 m")
    if args.live and args.snapshot_dir:
        parser.error("--snapshot-dir is only available without --live")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop_event = threading.Event()
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda *_: stop_event.set())
    try:
        return run(args, stop_event)
    except Exception as exc:
        LOG.error("Tracker stopped: %s", exc)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
