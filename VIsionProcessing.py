import numpy as np
import depthai as dai
import matplotlib as plt

visualizer = dai.RemoteConnection()

with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    # Print DeviceID, USB speed, and available cameras on the device
    print('DeviceID:',device.getDeviceInfo().getDeviceId())
    print('USB speed:',device.getUsbSpeed())
    print('Connected cameras:',device.getConnectedCameras())

    objectTracker = pipeline.create(dai.node.ObjectTracker) # object tracker

    rgb_cam = pipeline.create(dai.node.Camera).build()
    rgb_stream = rgb_cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12)
    visualizer.addTopic("rgb", rgb_stream)
    
    pipeline.start()

    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        if visualizer.waitKey(1) == ord('q'):
            pipeline.stop()

pipeline = dai.Pipeline()
camera = pipeline.create(dai.node.Camera).build()
detection = pipeline.create(dai.node.DetectionNetwork).build(camera, dai.NNModelDescription("yolov6-nano"))
detection.setConfidenceThreshold(0.5)
detection.input.setBlocking(False)