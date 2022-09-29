import cv2
import mediapipe as mp
import numpy as np
import depthai as dai
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image, ImageOps
from custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)
from custom.iris_lm_depth import from_landmarks_to_depth
from face_detector import FaceDetector
from model import L2CS
from utils import draw_gaze
import time

enable_head_pose = True
enable_iris_detection = True
enable_gaze_estimation = True

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

iris_points_idx = [33, 133, 362, 263, 61, 291, 199]
iris_points_idx = list(set(iris_points_idx))
iris_points_idx.sort()

left_eye_landmarks_id = np.array([33, 133])
right_eye_landmarks_id = np.array([362, 263])

dist_coeff = np.zeros((4, 1))

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
SMALL_CIRCLE_SIZE = 1
LARGE_CIRCLE_SIZE = 2

frame_height, frame_width, channels = (720, 1280, 3)
image_size = (frame_width, frame_height)
pipeline = dai.Pipeline()
device = dai.Device()
caliData = device.readCalibration()
camera_matrix = np.array(caliData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, frame_width, frame_height), dtype=float)
focal_length = (camera_matrix[0,0] + camera_matrix[1,1]) / 2
focal_length_mm = focal_length * 6.29 / 4056
print(focal_length_mm)
dist_coeff = np.zeros((4, 1))

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

import tkinter
root = tkinter.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.quit()

screen_width_mm = 1190
screen_height_mm = 670
camera_location_mm = (512, 565, 135)

def projection(screen, left_eye, right_eye, gaze):
    blank_canvas = np.ones((screen_height, screen_width, 3))
    canvas = np.ones((screen_height, screen_width, 3))
    canvas = cv2.circle(blank_canvas, (10, 10), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (10, screen_height - 10), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (screen_width - 10, screen_height - 10), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (screen_width - 10, 10), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (int(screen_width / 2), int(screen_height / 2)), 5, (0, 0,255), -1)

    canvas = cv2.circle(blank_canvas, (750, 750), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (750, screen_height - 750), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (screen_width - 750, screen_height - 750), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (screen_width - 750, 750), 5, (0, 0,255), -1)
    canvas = cv2.circle(blank_canvas, (int(camera_location_mm[0] / screen_width_mm * screen_width), int(camera_location_mm[1] / screen_height_mm * screen_height)), 10, (255, 0,), -1)

    eye_center_x_mm = (left_eye[0] + right_eye[0]) / 2
    eye_center_y_mm = (left_eye[1] + right_eye[1]) / 2
    eye_center_z_mm = (left_eye[2] + right_eye[2]) / 2

    screen_eye_center_x_mm = camera_location_mm[0] - eye_center_x_mm
    screen_eye_center_y_mm = camera_location_mm[1] - eye_center_y_mm
    screen_eye_center_z_mm = camera_location_mm[2] + eye_center_z_mm
    screen_eye_center_z_mm = 974
    screen_eye_center = (int(screen_eye_center_x_mm / screen_width_mm * screen_width), int(screen_eye_center_y_mm / screen_height_mm * screen_height))    
    canvas = cv2.circle(blank_canvas, screen_eye_center, 10, (0, 0,0), -1)

    if (gaze is not None):
        print(gaze[0] / np.pi * 180, gaze[1] / np.pi * 180)
        print(screen_eye_center_z_mm, screen_eye_center_z_mm * np.tan(gaze[0]), screen_eye_center_z_mm * np.tan(gaze[1]))
        screen_gaze_x_mm = screen_eye_center_x_mm + screen_eye_center_z_mm * np.tan(gaze[0])
        screen_gaze_y_mm = screen_eye_center_y_mm - screen_eye_center_z_mm * np.tan(gaze[1])
        gaze_center = (int(screen_gaze_x_mm / screen_width_mm * screen_width), int(screen_gaze_y_mm / screen_height_mm * screen_height)) 
        print(gaze_center)
        if (gaze_center[1] > 0 and gaze_center[1] < screen_height and gaze_center[0] > 0 and gaze_center[0] < screen_width):
            canvas = cv2.circle(blank_canvas, gaze_center, 10, (255, 0, 255), -1)

        # print(screen_gaze_x, screen_gaze_y)

    cv2.imshow(screen, canvas)

def put_eye_coord_from_stereo(frame, spatialData):
    # write depth values into frame
    for depthData in spatialData:
        roi = depthData.config.roi
        roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
        xmin = int(roi.topLeft().x)
        ymin = int(roi.topLeft().y)
        xmax = int(roi.bottomRight().x)
        ymax = int(roi.bottomRight().y)

        depthMin = depthData.depthMin
        depthMax = depthData.depthMax

        fontType = cv2.FONT_HERSHEY_TRIPLEX
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        cv2.putText(frame, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, 255)
        cv2.putText(frame, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, 255)
        cv2.putText(frame, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, 255)
    return frame

def put_eye_coord_from_iris_estimation(frame, left_iris_landmarks, right_iris_landmarks, smooth_left_depth, smooth_right_depth):
    fontType = cv2.FONT_HERSHEY_TRIPLEX
    xmin = 150
    ymin = 50
    x = right_iris_landmarks[0][0]
    y = right_iris_landmarks[0][1]
    x = (x - 0.5) * frame_width / focal_length * smooth_left_depth
    y = (0.5 - y) * frame_height / focal_length * smooth_left_depth
    right_eye_mm = (x, y, smooth_right_depth)
    cv2.putText(frame, f"X: {int(x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, RED)
    cv2.putText(frame, f"Y: {int(y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, RED)
    cv2.putText(frame, f"Z: {int(smooth_right_depth)} mm", (xmin + 10, ymin + 50), fontType, 0.5, RED)

    xmin = 50
    x = left_iris_landmarks[0][0]
    y = left_iris_landmarks[0][1]
    x = (x - 0.5) * frame_width / focal_length * smooth_left_depth
    y = (0.5 - y) * frame_height / focal_length * smooth_left_depth
    left_eye_mm = (x, y, smooth_left_depth)
    cv2.putText(frame, f"X: {int(x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, RED)
    cv2.putText(frame, f"Y: {int(y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, RED)
    cv2.putText(frame, f"Z: {int(smooth_left_depth)} mm", (xmin + 10, ymin + 50), fontType, 0.5, RED)
    return frame, left_eye_mm, right_eye_mm

def send_new_eye_locations(spatialCalcConfigInQueue, config, left_iris_landmarks, right_iris_landmarks):
    cfg = dai.SpatialLocationCalculatorConfig()
    #left
    topLeft = dai.Point2f(np.min(left_iris_landmarks[:, 0]), np.min(left_iris_landmarks[:, 1]))
    bottomRight =  dai.Point2f(np.max(left_iris_landmarks[:, 0]), np.max(left_iris_landmarks[:, 1]))
    config.roi = dai.Rect(topLeft, bottomRight)
    config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
    cfg.addROI(config)

    #right
    topLeft = dai.Point2f(np.min(right_iris_landmarks[:, 0]), np.min(right_iris_landmarks[:, 1]))
    bottomRight =  dai.Point2f(np.max(right_iris_landmarks[:, 0]), np.max(right_iris_landmarks[:, 1]))
    config.roi = dai.Rect(topLeft, bottomRight)
    config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
    cfg.addROI(config)

    spatialCalcConfigInQueue.send(cfg)


def main():
    batch_size = 1
    snapshot_path = "models/L2CSNet_gaze360.pkl"
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    compute_device = torch.device("mps")
    model=getArch("ResNet50", 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path, map_location=compute_device)
    model.load_state_dict(saved_state_dict)
    model.to(compute_device)
    model.eval()

    screen = 'screen'
    cv2.namedWindow(screen, cv2.WND_PROP_FULLSCREEN)

    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData.input.setQueueSize(1)
    xoutSpatialData.input.setBlocking(False)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewSize(frame_width, frame_height)
    camRgb.setIspScale(2, 3)
    camRgb.setInterleaved(False)
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    xoutRgb.input.setQueueSize(1)
    xoutRgb.input.setBlocking(False)

    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Config
    topLeft = dai.Point2f(0.49, 0.49)
    bottomRight = dai.Point2f(0.51, 0.51)

    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(topLeft, bottomRight)
    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)

    camRgb.preview.link(xoutRgb.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(spatialLocationCalculator.inputDepth)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    device.startPipeline(pipeline)

    qRgb = device.getOutputQueue("rgb", 1)
    qSpatial = device.getOutputQueue("spatialData", 1)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    refine_landmarks = True

    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )
    
    landmarks = None
    smooth_left_depth = -1
    smooth_right_depth = -1
    smooth_factor = 0.1
    
    softmax = nn.Softmax(dim=1)
    detector = FaceDetector()
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(compute_device)
    x=0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            start_fps = time.time()
            frame = qRgb.get().getCvFrame()
            spatialData = qSpatial.get().getSpatialLocations()
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks
            pitch_result = None
            yaw_result = None
            if enable_gaze_estimation:
                faces = detector.run_inference(frame)
                if faces is not None:
                    for face in faces:
                        face_coordinates = face['face_coordinates']
                        x_min = max(0, int(face_coordinates[0] * frame_width))
                        y_min = max(0, int(face_coordinates[1] * frame_height))
                        x_max = min(int(face_coordinates[2] * frame_width), frame_width - 1)
                        y_max = min(int(face_coordinates[3] * frame_height), frame_height - 1)

                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min

                        # Crop image
                        img = frame[y_min:y_max, x_min:x_max]
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(img)
                        img=transformations(im_pil)
                        img  = Variable(img).to(compute_device)
                        img  = img.unsqueeze(0)
                        
                        # gaze prediction
                        gaze_pitch, gaze_yaw = model(img)
                        
                        pitch_predicted = softmax(gaze_pitch)
                        yaw_predicted = softmax(gaze_yaw)
                        
                        # Get continuous predictions in degrees.
                        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                        
                        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                        pitch_result = pitch_predicted
                        yaw_result = yaw_predicted

                        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

            if multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks = landmarks.T
                if enable_iris_detection:
                    (
                        left_depth,
                        left_iris_size,
                        left_iris_landmarks,
                        left_eye_contours,
                    ) = from_landmarks_to_depth(
                        frame_rgb,
                        landmarks[:, left_eye_landmarks_id],
                        image_size,
                        is_right_eye=False,
                        focal_length=focal_length,
                    )

                    (
                        right_depth,
                        right_iris_size,
                        right_iris_landmarks,
                        right_eye_contours,
                    ) = from_landmarks_to_depth(
                        frame_rgb,
                        landmarks[:, right_eye_landmarks_id],
                        image_size,
                        is_right_eye=True,
                        focal_length=focal_length,
                    )

                    if smooth_right_depth < 0:
                        smooth_right_depth = right_depth
                    else:
                        smooth_right_depth = (
                            smooth_right_depth * (1 - smooth_factor)
                            + right_depth * smooth_factor
                        )

                    if smooth_left_depth < 0:
                        smooth_left_depth = left_depth
                    else:
                        smooth_left_depth = (
                            smooth_left_depth * (1 - smooth_factor)
                            + left_depth * smooth_factor
                        )

                if refine_landmarks:
                    landmarks = landmarks[:, :468]

                if enable_head_pose:
                    metric_landmarks, pose_transform_mat = get_metric_landmarks(
                        landmarks.copy(), pcf
                    )

                    image_points = (
                        landmarks[0:2, points_idx].T
                        * np.array([frame_width, frame_height])[None, :]
                    )
                    model_points = metric_landmarks[0:3, points_idx].T

                    # see here:
                    # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
                    pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
                    mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
                    mp_translation_vector = pose_transform_mat[:3, 3, None]

                    nose_tip = model_points[0]
                    nose_tip_extended = 2.5 * model_points[0]
                    (nose_pointer2D, jacobian) = cv2.projectPoints(
                        np.array([nose_tip, nose_tip_extended]),
                        mp_rotation_vector,
                        mp_translation_vector,
                        camera_matrix,
                        dist_coeff,
                    )

                    nose_tip_2D, nose_tip_2D_extended = nose_pointer2D.squeeze().astype(int)
                    frame = cv2.line(
                        frame, nose_tip_2D, nose_tip_2D_extended, (255, 0, 0), 2
                    )
                
                if landmarks is not None and enable_iris_detection:

                    # draw subset of facemesh
                    for ii in iris_points_idx:
                        pos = (np.array(image_size) * landmarks[:2, ii]).astype(np.int32)
                        frame = cv2.circle(frame, tuple(pos), LARGE_CIRCLE_SIZE, GREEN, -1)
                    
                    # draw eye contours
                    eye_landmarks = np.concatenate(
                        [
                            right_eye_contours,
                            left_eye_contours,
                        ]
                    )
                    for landmark in eye_landmarks:
                        pos = (np.array(image_size) * landmark[:2]).astype(np.int32)
                        frame = cv2.circle(frame, tuple(pos), SMALL_CIRCLE_SIZE, RED, -1)

                    # draw iris landmarks
                    iris_landmarks = np.concatenate(
                        [
                            right_iris_landmarks,
                            left_iris_landmarks,
                        ]
                    )
                    for landmark in iris_landmarks:
                        pos = (np.array(image_size) * landmark[:2]).astype(np.int32)
                        frame = cv2.circle(frame, tuple(pos), SMALL_CIRCLE_SIZE, YELLOW, -1)

                    send_new_eye_locations(spatialCalcConfigInQueue, config, left_iris_landmarks, right_iris_landmarks)
                    frame = put_eye_coord_from_stereo(frame, spatialData)
                    frame, left_eye_mm, right_eye_mm = put_eye_coord_from_iris_estimation(frame, left_iris_landmarks, right_iris_landmarks, smooth_left_depth, smooth_right_depth)

                    gaze = None
                    if (pitch_result is not None):
                        gaze = (pitch_result, yaw_result)
                    projection(screen, left_eye_mm, right_eye_mm, gaze)

                    # depth_string = "{:.2f}cm, {:.2f}cm".format(
                    #     smooth_left_depth / 10, smooth_right_depth / 10
                    # )
                    # frame = cv2.putText(
                    #     frame,
                    #     depth_string,
                    #     (50, 50),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     1,
                    #     GREEN,
                    #     2,
                    #     cv2.LINE_AA,
                    # )
            FPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(FPS), (frame_width - 150, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
