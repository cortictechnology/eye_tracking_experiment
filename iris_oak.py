import argparse

import cv2
import mediapipe as mp
import numpy as np
import depthai as dai
from PIL import Image
import time
from custom.iris_lm_depth import from_landmarks_to_depth

mp_face_mesh = mp.solutions.face_mesh

points_idx = [33, 133, 362, 263, 61, 291, 199]
points_idx = list(set(points_idx))
points_idx.sort()

left_eye_landmarks_id = np.array([33, 133])
right_eye_landmarks_id = np.array([362, 263])

dist_coeff = np.zeros((4, 1))

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
SMALL_CIRCLE_SIZE = 1
LARGE_CIRCLE_SIZE = 2


def main():
    image_size = (1280, 720)
    pipeline = dai.Pipeline()
    device = dai.Device()
    caliData = device.readCalibration()
    K_col = np.array(caliData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, image_size[0], image_size[1]), dtype=float)
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewSize(image_size[0], image_size[1])
    camRgb.setInterleaved(False)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    xoutRgb.input.setQueueSize(1)
    xoutRgb.input.setBlocking(False)
    camRgb.preview.link(xoutRgb.input)
    device.startPipeline(pipeline)
    
    qRgb = device.getOutputQueue("rgb", 1)

    # pseudo camera internals
    focal_length = (K_col[0,0] + K_col[1, 1]) / 2

    landmarks = None
    smooth_left_depth = -1
    smooth_right_depth = -1
    smooth_factor = 0.1

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
            
        while True:
#            t1 = time.time()
            frame = qRgb.get().getCvFrame()
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks = landmarks.T

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

            if landmarks is not None:

                # draw subset of facemesh
                for ii in points_idx:
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

            cv2.imshow("Iris tracking", frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
