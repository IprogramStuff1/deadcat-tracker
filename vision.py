"""Person tracking and an explicitly opened, nonblocking OAK camera source."""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import time


PERSON_LABEL = 0  # COCO class used by the existing YOLO model.
MODEL_SOURCE = "luxonis/yolov6-nano:r2-coco-512x288"


@dataclass(frozen=True)
class VisionSample:
    """A camera observation; timestamp shares time.monotonic()'s clock domain.

    A sample with no error vector means a frame arrived without a usable target.
    No sample at all means the output queue has no new frame.
    """

    timestamp: float
    error_vector: tuple[float, float, float, float] | None


def _coordinates(relative_coords):
    """Return finite millimetre coordinates, requiring a positive depth."""
    try:
        coords = tuple(float(getattr(relative_coords, axis)) for axis in ("x", "y", "z"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in coords) or coords[2] <= 0:
        return None
    return coords


class DesignatedTracker:
    """Acquire by confidence, then associate the nearest same-class detection.

    This is positional association, not person identification. Once enough frames
    miss the target, acquisition stays disabled until an explicit reset/designation.
    """

    def __init__(self, label=None, max_distance_meters=1.5, max_missed=10):
        if not math.isfinite(max_distance_meters) or max_distance_meters <= 0:
            raise ValueError("max_distance_meters must be finite and positive")
        if not isinstance(max_missed, int) or max_missed < 1:
            raise ValueError("max_missed must be a positive integer")
        self._configured_label = label
        self.max_distance_squared = (max_distance_meters * 1000.0) ** 2
        self.max_missed = max_missed
        self.reset()

    def reset(self):
        """Forget the target and permit acquisition of the configured class."""
        self.label = self._configured_label
        self.position = None
        self.missed = 0
        self.lost = False

    def designate(self, detection):
        """Explicitly lock onto a valid detection selected by the caller."""
        coords = _coordinates(getattr(detection, "spatialCoordinates", None))
        if coords is None:
            raise ValueError("Cannot designate a detection without valid spatial coordinates")
        if self._configured_label is not None and detection.label != self._configured_label:
            raise ValueError("Detection does not match the configured target class")
        self.label = detection.label
        self.position = coords
        self.missed = 0
        self.lost = False
        return detection

    def update(self, detections):
        """Return a valid target, or None for a missing/invalid/latched-lost target."""
        if self.lost:
            return None
        best_detection = None
        best_distance_squared = self.max_distance_squared
        best_confidence = -1.0
        for detection in detections:
            if self.label is not None and detection.label != self.label:
                continue
            coords = _coordinates(getattr(detection, "spatialCoordinates", None))
            try:
                confidence = float(getattr(detection, "confidence", 0.0))
            except (TypeError, ValueError, OverflowError):
                continue
            if coords is None or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                continue
            if self.position is None:
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = detection
            else:
                deltas = (current - previous for current, previous in zip(coords, self.position))
                distance_squared = sum(delta * delta for delta in deltas)
                if distance_squared < best_distance_squared:
                    best_distance_squared = distance_squared
                    best_detection = detection

        if best_detection is not None:
            return self.designate(best_detection)
        # Do not latch loss before a target has ever been acquired.
        if self.position is not None:
            self.missed += 1
            if self.missed >= self.max_missed:
                self.lost = True
        return None


def camera_to_drone(relative_coords, standoff):
    """Convert camera right/down/forward mm into body forward/right/down metres.

    Assumes the camera is mounted level and faces the aircraft's forward axis.
    The fourth component is a right-positive yaw error in radians.
    """
    coords = _coordinates(relative_coords)
    if coords is None:
        raise ValueError("Camera coordinates must be finite with positive depth")
    if not math.isfinite(standoff) or standoff < 0:
        raise ValueError("standoff must be finite and nonnegative")
    x, y, z = coords
    return z / 1000.0 - standoff, x / 1000.0, y / 1000.0, math.atan2(x, z)


class VisionSource:
    """Own the camera for a with-block; consume only the latest available frame.

    Hardware libraries, USB access and model loading occur only in __enter__.
    One worker thread should own this object, including calls to reset_tracker().
    """

    def __init__(self, standoff: float = 2.0):
        if not math.isfinite(standoff) or standoff < 0:
            raise ValueError("standoff must be finite and nonnegative")
        self.standoff = standoff
        self.tracker = DesignatedTracker(label=PERSON_LABEL)
        self._stack = None
        self._pipeline = None
        self._queue = None
        self._dai = None
        self._reset_at = -math.inf
        self._last_capture_time = -math.inf

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("VisionSource is already open")
        try:
            import depthai as dai
            from depthai_nodes.node import ParsingNeuralNetwork
        except ImportError as exc:
            raise RuntimeError(
                "The camera needs the existing DepthAI/depthai-nodes dependencies. "
                "Activate the project environment and check its installation."
            ) from exc

        try:
            with ExitStack() as stack:
                pipeline = stack.enter_context(dai.Pipeline())
                rgb_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                stereo = pipeline.create(dai.node.StereoDepth)
                mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
                mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
                mono_left.requestOutput((640, 400)).link(stereo.left)
                mono_right.requestOutput((640, 400)).link(stereo.right)
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                stereo.setLeftRightCheck(True)  # Required for RGB depth alignment on RVC2.
                stereo.setSubpixel(True)
                stereo.setOutputSize(640, 400)

                nn = pipeline.create(ParsingNeuralNetwork).build(
                    rgb_cam, nnSource=MODEL_SOURCE, fps=30
                )
                slc = pipeline.create(dai.node.SpatialLocationCalculator)
                slc.setRunOnHost(True)
                slc.inputConfig.setWaitForMessage(False)
                # DepthAI v3 carries image transformations with detections and
                # depth, so SLC maps their differing crop/output sizes itself.
                nn.out.link(slc.inputDetections)
                stereo.depth.link(slc.inputDepth)
                spatial_queue = slc.outputDetections.createOutputQueue(maxSize=1, blocking=False)
                pipeline.start()
                if not pipeline.isRunning():
                    raise RuntimeError("The camera pipeline stopped during startup")
                self._pipeline = pipeline
                self._queue = spatial_queue
                self._dai = dai
                self._stack = stack.pop_all()
                self.reset_tracker()
        except Exception as exc:
            raise RuntimeError(
                "Unable to start the OAK camera. Check its USB data connection, "
                "power and Linux USB permissions; the model must be cached or "
                f"downloadable on first startup. Details: {exc}"
            ) from exc
        return self

    def reset_tracker(self):
        """Permit a new target acquisition after the user enables following."""
        self._reset_at = time.monotonic()
        self._last_capture_time = -math.inf
        self.tracker.reset()

    def read_latest(self) -> VisionSample | None:
        """Read without waiting for a frame; retain capture age for expiry checks."""
        if self._pipeline is None:
            raise RuntimeError("Open VisionSource using a with-block before reading")
        if not self._pipeline.isRunning():
            raise RuntimeError("OAK camera pipeline stopped; check its USB connection and power")
        packet = self._queue.tryGet()
        if packet is None:
            return None
        # getTimestamp() uses the host-synchronised DepthAI monotonic clock.
        # Convert via age because its epoch need not equal Python's monotonic epoch.
        now = time.monotonic()
        try:
            captured_at = packet.getTimestamp()
            age = (self._dai.Clock.now() - captured_at).total_seconds()
            capture_time = captured_at.total_seconds()
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("OAK frame has an invalid capture timestamp") from exc
        timestamp = now - age
        if not all(math.isfinite(value) for value in (age, capture_time, timestamp)) or age < 0:
            raise RuntimeError("OAK frame has an invalid or future capture timestamp")
        # Reject before association: a queued frame from the previous enable
        # session must never designate the new session's person. Compare the
        # original capture clock for ordering to avoid conversion-rounding jitter.
        if timestamp < self._reset_at or capture_time <= self._last_capture_time:
            return None
        self._last_capture_time = capture_time
        tracked_object = self.tracker.update(packet.detections)
        error_vector = None
        if tracked_object is not None:
            error_vector = camera_to_drone(tracked_object.spatialCoordinates, self.standoff)
        return VisionSample(timestamp=timestamp, error_vector=error_vector)

    def __exit__(self, exc_type, exc_value, traceback):
        stack, self._stack = self._stack, None
        self._pipeline = self._queue = self._dai = None
        self.reset_tracker()
        if stack is not None:
            return stack.__exit__(exc_type, exc_value, traceback)
        return False
