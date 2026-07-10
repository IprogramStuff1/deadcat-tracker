import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
import math

# Standard COCO labels for YOLO models
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

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
            for detection in in_det.detections:
                
                label_id = detection.label
                label_name = labels[label_id] if label_id < len(labels) else str(label_id)
                
                # Z coordinate is the physical distance straight out from the lens, X is hortizontal displacement and Y is vertical displacement
                distance_z_meters = detection.spatialCoordinates.z / 1000.0 
                distance_x_meters = detection.spatialCoordinates.x / 1000.0
                distance_y_meters = detection.spatialCoordinates.y / 1000.0 # Positive is downwards and negative is upwards
                yaw = math.degrees(math.atan2(distance_x_meters,distance_z_meters))
                pitch = math.degrees(math.atan2(distance_y_meters,distance_z_meters))
                print(f"Yaw: {yaw} | Pitch: {pitch}")
                print(f"Detected: {label_name} | X: {distance_x_meters:.2f}m | Y: {distance_y_meters:.2f}m | Z: {distance_z_meters:.2f}m")