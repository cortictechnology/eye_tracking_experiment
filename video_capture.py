import cv2
import numpy as np
import depthai as dai
from PIL import Image
import time

class VideoCapture:
    def __init__(self, use_1080p=False, use_720p_mono=False, extended_disparity=False, use_depth=False, get_depth_map=False):
        self.use_1080p = use_1080p
        if self.use_1080p:
            self.frame_width = 1920
            self.frame_height = 1080
        else:
            self.frame_width = 1280
            self.frame_height = 720
        self.use_720p_mono = use_720p_mono
        self.extended_disparity = extended_disparity
        self.use_depth = use_depth
        self.get_depth_map = get_depth_map
        self.pipeline = dai.Pipeline()
        self.device = dai.Device()
        caliData = self.device.readCalibration()
        self.camera_matrix = np.array(caliData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.frame_width, self.frame_height), dtype=float)
        self.camera_distortion = np.array(caliData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        self.focal_length = (self.camera_matrix[0,0] + self.camera_matrix[1,1]) / 2
        self.build_pipeline()
        self.device.startPipeline(self.pipeline)
        self.qRgb = self.device.getOutputQueue("rgb", 1, blocking=False)
        if self.use_depth:
            self.qSpatial = self.device.getOutputQueue("spatialData", 1, blocking=False)
            self.spatialCalcConfigInQueue = self.device.getInputQueue("spatialCalcConfig")
            if (self.get_depth_map):
                self.qDepth = self.device.getOutputQueue("depth", 1, blocking=False)


    def build_pipeline(self):
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        # camRgb.setFps(self.fps)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        if (not self.use_1080p):
            camRgb.setIspScale(2, 3)
            camRgb.setPreviewSize(self.frame_width, self.frame_height)
        camRgb.setInterleaved(False)
        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        xoutRgb.input.setQueueSize(1)
        xoutRgb.input.setBlocking(False)
        if (self.use_1080p):
            camRgb.video.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)
        if self.use_depth:
            xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
            xoutSpatialData.input.setQueueSize(1)
            xoutSpatialData.input.setBlocking(False)
            xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)
            xoutSpatialData.setStreamName("spatialData")
            xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
            if (self.get_depth_map):
                xoutDepth = self.pipeline.create(dai.node.XLinkOut)
                xoutDepth.input.setQueueSize(1)
                xoutDepth.input.setBlocking(False)
                xoutDepth.setStreamName("depth")

            calibData = self.device.readCalibration()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
            left = self.pipeline.create(dai.node.MonoCamera)
            right = self.pipeline.create(dai.node.MonoCamera)
            stereo = self.pipeline.create(dai.node.StereoDepth)
            spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)
            if (self.use_720p_mono):
                monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
            else:
                monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

            left.setResolution(monoResolution)
            # left.setFps(self.fps)
            left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            right.setResolution(monoResolution)
            # right.setFps(self.fps)
            right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            # LR-check is required for depth alignment
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setExtendedDisparity(self.extended_disparity)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

            # Config
            topLeft = dai.Point2f(0.49, 0.49)
            bottomRight = dai.Point2f(0.51, 0.51)

            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 10000
            config.roi = dai.Rect(topLeft, bottomRight)
            spatialLocationCalculator.inputConfig.setWaitForMessage(False)
            spatialLocationCalculator.setWaitForConfigInput(True)
            spatialLocationCalculator.initialConfig.addROI(config)
            spatialLocationCalculator.inputDepth.setBlocking(False)
            spatialLocationCalculator.inputDepth.setQueueSize(1)
            left.out.link(stereo.left)
            right.out.link(stereo.right)
            stereo.depth.link(spatialLocationCalculator.inputDepth)
            spatialLocationCalculator.out.link(xoutSpatialData.input)
            if (self.get_depth_map):
                stereo.depth.link(xoutDepth.input)
            xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    def get_frame(self, get_depth=False):
        frame = self.qRgb.get().getCvFrame()
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
        if (get_depth):
            depth_frame = self.qDepth.get().getFrame()
            return frame, frame_rgb, depth_frame
        return frame, frame_rgb

    def get_depth(self, rois):
        depth_data = []
        if self.use_depth:
            cfg = dai.SpatialLocationCalculatorConfig()
            for roi in rois:
                config = dai.SpatialLocationCalculatorConfigData()
                config.depthThresholds.lowerThreshold = 100
                config.depthThresholds.upperThreshold = 10000
                topLeft = dai.Point2f(roi[0], roi[1])
                bottomRight = dai.Point2f(roi[2], roi[3])
                config.roi = dai.Rect(topLeft, bottomRight)
                config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
                cfg.addROI(config)
            self.spatialCalcConfigInQueue.send(cfg)
            spatialData = self.qSpatial.get().getSpatialLocations()
            return spatialData
        else:
            print("Video capture is not init with depth enabled")
        return depth_data

    def draw_spatial_data(self, frame, spatialData, color=(255, 0, 0)):
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            cv2.putText(frame, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(frame, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(frame, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
        return frame
