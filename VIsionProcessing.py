import numpy as np
import depthai as dai
import matplotlib as plt

with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    # Print DeviceID, USB speed, and available cameras on the device
    print('DeviceID:',device.getDeviceInfo().getDeviceId())
    print('USB speed:',device.getUsbSpeed())
    print('Connected cameras:',device.getConnectedCameras())
    objectTracker = pipeline.create(dai.node.ObjectTracker)
    
    pipeline.start()
pipeline = dai.Pipeline()
camera = pipeline.create(dai.node.Camera).build()
detection = pipeline.create(dai.node.DetectionNetwork).build(camera, dai.NNModelDescription("yolov6-nano"))
detection.setConfidenceThreshold(0.5)
detection.input.setBlocking(False)