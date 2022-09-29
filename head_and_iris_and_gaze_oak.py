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
iris_points_idx = list(set(points_idx))
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
    
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewSize(frame_width, frame_height)
    camRgb.setInterleaved(False)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    xoutRgb.input.setQueueSize(1)
    xoutRgb.input.setBlocking(False)
    camRgb.preview.link(xoutRgb.input)
    device.startPipeline(pipeline)
    
    qRgb = device.getOutputQueue("rgb", 1)
    
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
            frame = qRgb.get().getCvFrame()
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks
            if enable_gaze_estimation:
                faces = detector.run_inference(frame)
                if faces is not None:
                    for face in faces:
                        face_coordinates = face['face_coordinates']
                        x_min = int(face_coordinates[0] * frame_width)
                        y_min = int(face_coordinates[1] * frame_height)
                        x_max = int(face_coordinates[2] * frame_width)
                        y_max = int(face_coordinates[3] * frame_height)
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

                    # write depth values into frame
                    depth_string = "{:.2f}cm, {:.2f}cm".format(
                        smooth_left_depth / 10, smooth_right_depth / 10
                    )
                    frame = cv2.putText(
                        frame,
                        depth_string,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        GREEN,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("Eye Tracking", frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
