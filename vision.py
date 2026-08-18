import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
import math
import numpy as np
from navigation import SimpleProportionalControl

# Standard COCO labels for YOLO models
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
class DesignatedTracker:
    __slots__ = ("label", "max_distance_squared", "max_missed", "missed", "position")

    def __init__(self, label=None, max_distance_meters=1.5, max_missed=10):
        self.label = label
        max_distance_mm = max_distance_meters * 1000.0
        self.max_distance_squared = max_distance_mm * max_distance_mm
        self.max_missed = max_missed
        self.missed = 0
        self.position = None

    def designate(self, detection):
        """Lock the tracker onto a detection selected by the caller."""
        coordinates = detection.spatialCoordinates
        self.label = detection.label
        self.position = (coordinates.x, coordinates.y, coordinates.z)
        self.missed = 0
        return detection

    def update(self, detections):
        """Return the same target in a new frame, or None when it is not found."""
        best_detection = None
        best_distance_squared = self.max_distance_squared

        if self.position is None:
            best_confidence = -1.0
            for detection in detections:
                if self.label is not None and detection.label != self.label:
                    continue
                coordinates = detection.spatialCoordinates
                if coordinates.z <= 0:
                    continue
                confidence = getattr(detection, "confidence", 0.0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = detection
        else:
            previous_x, previous_y, previous_z = self.position
            for detection in detections:
                if detection.label != self.label:
                    continue
                coordinates = detection.spatialCoordinates
                if coordinates.z <= 0:
                    continue
                delta_x = coordinates.x - previous_x
                delta_y = coordinates.y - previous_y
                delta_z = coordinates.z - previous_z
                distance_squared = (
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
                )
                if distance_squared < best_distance_squared:
                    best_distance_squared = distance_squared
                    best_detection = detection

        if best_detection is None:
            self.missed += 1
            if self.missed > self.max_missed:
                self.position = None
            return None

        return self.designate(best_detection)
tracker = DesignatedTracker(label=labels.index("person"))
def camera_to_drone(relative_coords, standoff):
    drone_x_error = relative_coords.z / 1000.0 - standoff
    drone_y_error = relative_coords.x / 1000.0
    drone_z_error = relative_coords.y / 1000.0
    drone_yaw_error = math.atan2(relative_coords.x, relative_coords.z)
    return drone_x_error, drone_y_error, drone_z_error, drone_yaw_error
with dai.Pipeline() as pipeline:
    
    # 1. Camera Node (v3 syntax automatically handles ISP scaling)
    rgb_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    # 2. Stereo Depth Generation
    stereo = pipeline.create(dai.node.StereoDepth)
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    
    monoLeft.requestOutput((640, 400)).link(stereo.left)
    monoRight.requestOutput((640, 400)).link(stereo.right)
    
    # Crucial step: Align the depth map to the RGB camera's perspective
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    #stereo.setExtendedDisparity(True) only for really close objects
    stereo.setSubpixel(True) #good for far/medium range objects
    stereo.setOutputSize(640,400)

    # 3. The v3 Parsing Neural Network (Auto-handles YOLO decoding!)
    nn = pipeline.create(ParsingNeuralNetwork).build(
        rgb_cam, 
        nnSource="luxonis/yolov6-nano:r2-coco-512x288", # Pulls directly from the Luxonis Hub
        fps=30
    )

    # 4. Spatial Location Calculator (slc) (v3 way to get X/Y/Z coords)
    slc = pipeline.create(dai.node.SpatialLocationCalculator)
    slc.setRunOnHost(True) # Run on host to avoid sending data back-and-forth to the camera
    
    # Link the parsed YOLO detections and the raw depth map into the calculator
    nn.out.link(slc.inputDetections)
    stereo.depth.link(slc.inputDepth)

    # 5. Output Queue for our Spatial Detections
    spatial_queue = slc.outputDetections.createOutputQueue()

    # Start the pipeline
    pipeline.start()
    
    device = pipeline.getDefaultDevice()
    print(f"DeviceID: {device.getDeviceInfo().getDeviceId()}")

    while pipeline.isRunning():
        
        # Pull the spatial detections from the queue
        in_det = spatial_queue.get()
        
        if in_det is not None:
            tracked_object = tracker.update(in_det.detections)
            if tracked_object is not None:
                relative_coords = tracked_object.spatialCoordinates
                
                
                #pitch = math.degrees(math.atan2(-distance_y_meters,distance_z_meters)) #required pitch up, can be subbed with climb
               
                error_vector = camera_to_drone(relative_coords=relative_coords, standoff=2.0)
                forward_command, yaw_command = SimpleProportionalControl(error_vector=error_vector)