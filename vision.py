"""Person tracking and an explicitly opened, nonblocking OAK camera source."""

from contextlib import ExitStack
from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time


PERSON_LABEL = 0  # COCO class used by the existing YOLO model.
MODEL_SOURCE = "luxonis/yolov6-nano:r2-coco-512x288"
LOG = logging.getLogger("tracker.vision")


@dataclass(frozen=True)
class VisionSample:
    """A camera observation; timestamp shares time.monotonic()'s clock domain.

    A sample with no error vector means a frame arrived without a usable target.
    No sample at all means the output queue has no new frame.
    """

    timestamp: float
    error_vector: tuple[float, float, float, float] | None
    frame_count: int = 0  # Accepted camera observations, independent of control ticks.


def configure_model_and_alignment(pipeline, dai, parsing_network, rgb_cam, stereo):
    """Use one model archive for inference and parsing, and align to its input."""
    device = pipeline.getDefaultDevice()
    if device.getPlatform() != dai.Platform.RVC2:
        raise RuntimeError("This camera pipeline is configured for the OAK-D Lite (RVC2)")
    description = dai.NNModelDescription(MODEL_SOURCE)
    description.platform = device.getPlatformAsString()
    archive = dai.NNArchive(dai.getModelFromZoo(description))
    size = archive.getInputSize()
    if size is None or len(size) != 2 or any(value <= 0 for value in size):
        raise RuntimeError("Model archive has no valid image input width/height")
    if len(archive.getConfig().model.inputs) != 1 or len(archive.getConfig().model.heads) != 1:
        raise RuntimeError("Expected a single image input and detection head in the model")

    # Stereo alignment needs an undistorted RGB reference. Request the model's
    # precise input here, so the helper cannot choose a different crop/geometry.
    inference_image = rgb_cam.requestOutput(
        tuple(size), type=dai.ImgFrame.Type.BGR888p,
        resizeMode=dai.ImgResizeMode.CROP, fps=30, enableUndistortion=True,
    )
    nn = pipeline.create(parsing_network).build(inference_image, nnSource=archive)
    # depthai-nodes 0.6.0 only supplies the head to this parser. The full archive
    # also supplies the blob's tensor dimensions, avoiding the RVC2 416x416 fallback.
    # Do not combine setInputImageSize with archive/blob configuration.
    nn.getParser(dai.node.DetectionParser).setNNArchive(archive)
    # RVC2's StereoDepth performs alignment itself. Use the exact inference crop,
    # resize and camera metadata rather than a separate 640x400 RGB viewpoint.
    nn.passthrough.link(stereo.inputAlignTo)
    LOG.info("Model input=%dx%d; parser uses full archive; depth aligned to inference image", *size)
    return nn, tuple(size)


class SnapshotWriter:
    """Save a bounded set of same-frame RGB diagnostics without a GUI dependency."""

    def __init__(self, directory, queue):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.queue = queue
        self.frames = OrderedDict()
        self.next_save = 0.0
        self.saved = 0
        self.run_id = time.time_ns()

    def capture(self, packet, tracked_object, now):
        for frame in self.queue.tryGetAll():
            key = (frame.getSequenceNum(), frame.getTimestamp())
            self.frames[key] = frame
        while len(self.frames) > 16:
            self.frames.popitem(last=False)
        if now < self.next_save or self.saved >= 30:
            return
        frame = self.frames.pop((packet.getSequenceNum(), packet.getTimestamp()), None)
        if frame is None:
            return  # Never draw boxes over a different frame.
        import cv2  # imwrite works with the existing headless OpenCV package.

        image = frame.getCvFrame().copy()
        height, width = image.shape[:2]
        for detection in packet.detections:
            if detection.label != PERSON_LABEL:
                continue
            box = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            if not all(math.isfinite(value) for value in box):
                continue
            x1, y1, x2, y2 = (
                int(max(0, min(1, value)) * (limit - 1))
                for value, limit in zip(box, (width, height, width, height))
            )
            color = (0, 255, 0) if detection is tracked_object else (0, 180, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            coords = _coordinates(detection.spatialCoordinates)
            label = f"z={coords[2] / 1000:.2f}m" if coords else "invalid depth"
            cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        path = self.directory / f"frame-{self.run_id}-{packet.getSequenceNum():08d}.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write diagnostic image: {path}")
        self.saved += 1
        self.next_save = now + 1.0
        LOG.info("Saved camera diagnostic %s (%d/30)", path, self.saved)


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

    def __init__(self, standoff: float = 2.0, snapshot_dir=None):
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
        self._input_size = None
        self._depth_queue = None
        self._depth_transform = None
        self._geometry_checked = False
        self._frame_count = 0
        self._snapshot_dir = snapshot_dir
        self._snapshots = None

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
                stereo.setLeftRightCheck(True)  # Required for RGB depth alignment on RVC2.
                stereo.setSubpixel(True)

                nn, input_size = configure_model_and_alignment(
                    pipeline, dai, ParsingNeuralNetwork, rgb_cam, stereo
                )
                slc = pipeline.create(dai.node.SpatialLocationCalculator)
                slc.setRunOnHost(True)
                slc.inputConfig.setWaitForMessage(False)
                nn.out.link(slc.inputDetections)
                stereo.depth.link(slc.inputDepth)
                spatial_queue = slc.outputDetections.createOutputQueue(maxSize=1, blocking=False)
                # This is already host-side depth. Inspect its geometry once to
                # catch a failed alignment before exposing control observations.
                depth_queue = slc.passthroughDepth.createOutputQueue(maxSize=1, blocking=False)
                snapshots = None
                if self._snapshot_dir is not None:
                    rgb_queue = nn.passthrough.createOutputQueue(maxSize=8, blocking=False)
                    snapshots = SnapshotWriter(self._snapshot_dir, rgb_queue)
                pipeline.start()
                if not pipeline.isRunning():
                    raise RuntimeError("The camera pipeline stopped during startup")
                self._pipeline = pipeline
                self._queue = spatial_queue
                self._dai = dai
                self._input_size = input_size
                self._depth_queue = depth_queue
                self._depth_transform = None
                self._geometry_checked = False
                self._frame_count = 0
                self._snapshots = snapshots
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
        if self._depth_queue is not None:
            depth = self._depth_queue.tryGet()
            if depth is not None and not self._geometry_checked:
                self._depth_transform = depth.getTransformation()
        packet = self._queue.tryGet()
        if packet is None:
            return None
        if self._input_size is not None and not self._geometry_checked:
            if self._depth_transform is None:
                return None  # Wait for the first aligned depth metadata as well.
            transform = packet.getTransformation()
            if (tuple(transform.getSize()) != self._input_size
                    or tuple(self._depth_transform.getSize()) != self._input_size
                    or not self._depth_transform.isEqualTransformation(transform)):
                raise RuntimeError(
                    "Depth/detection geometry does not match the model input; "
                    "camera observations withheld. Check the inference alignment link."
                )
            self._geometry_checked = True
            LOG.info("Verified depth/detection geometry: %dx%d", *self._input_size)
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
        self._frame_count += 1
        if self._snapshots is not None:
            self._snapshots.capture(packet, tracked_object, now)
        return VisionSample(timestamp=timestamp, error_vector=error_vector, frame_count=self._frame_count)

    def __exit__(self, exc_type, exc_value, traceback):
        stack, self._stack = self._stack, None
        self._pipeline = self._queue = self._dai = None
        self._depth_queue = self._depth_transform = self._snapshots = None
        self.reset_tracker()
        if stack is not None:
            return stack.__exit__(exc_type, exc_value, traceback)
        return False
