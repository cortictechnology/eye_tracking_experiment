import cv2
import numpy as np
import time
from video_capture import VideoCapture
from face_alignment import FaceAlignment
from facemesh_estimation import FaceMeshEstimation
from iris_detection import IrisDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from utils import eye_converter

def main(use_depth):
    video = VideoCapture(use_depth=use_depth)
    ref_photo = cv2.imread("./face_photos/michael.jpg")
    facemesh_estimator = FaceMeshEstimation()
    aligned_facemesh_estimator = FaceMeshEstimation()
    face_aligner = FaceAlignment(ref_photo, facemesh_estimator)
    iris_detector = IrisDetection(video.frame_width, video.frame_height, video.focal_length)
    head_pose_estimator = HeadPoseEstimation(video.frame_width, video.frame_height, video.camera_matrix)
    gaze_estimator = GazeEstimation(video.frame_width, video.frame_height)
    while True:
        frame, frame_rgb = video.get_frame()
        if frame is not None and frame_rgb is not None:
            landmarks, detected_faces = facemesh_estimator.get_facemesh(frame_rgb)
            aligned_frame = face_aligner.get_aligned_face(frame, landmarks)
            aligned_crop_eyes = None
            if aligned_frame is not None:
                aligned_landmarks, _ = aligned_facemesh_estimator.get_facemesh(aligned_frame, convert_rgb=True)
                aligned_crop_width, aligned_crop_height, aligned_crop_eyes = aligned_facemesh_estimator.crop_eye_image(aligned_frame, aligned_landmarks)
                aligned_right_iris_landmarks, aligned_left_iris_landmarks, _, _ = iris_detector.get_iris(aligned_frame, aligned_landmarks)
                aligned_pixel_distance, _ = eye_converter(aligned_frame, video, aligned_left_iris_landmarks[0], aligned_right_iris_landmarks[0], aligned_landmarks[:, 1], aligned_landmarks[:, 8], warpped=True)
            right_iris_landmarks, left_iris_landmarks, left_depth, right_depth = iris_detector.get_iris(frame_rgb, landmarks)
            metric_landmarks, pose_transform_mat, image_points, model_points, mp_rotation_vector, mp_translation_vector = head_pose_estimator.get_head_pose(landmarks)
            #pitch, yaw = gaze_estimator.get_gaze(frame, detected_faces)
            pixel_distance, metric_distance = eye_converter(frame.copy(), video, left_iris_landmarks[0], right_iris_landmarks[0], landmarks[:, 1], landmarks[:, 8], warpped=False, left_eye_depth_mm=left_depth*10, right_eye_depth_mm=right_depth*10)
            frame = facemesh_estimator.draw_detected_face(frame, detected_faces)
            frame = iris_detector.draw_iris(frame, right_iris_landmarks, left_iris_landmarks, left_depth, right_depth)
            if use_depth:
                left_eye_roi = [np.min(left_iris_landmarks[:, 0]), np.min(left_iris_landmarks[:, 1]), np.max(left_iris_landmarks[:, 0]), np.max(left_iris_landmarks[:, 1])]
                right_eye_roi = [np.min(right_iris_landmarks[:, 0]), np.min(right_iris_landmarks[:, 1]), np.max(right_iris_landmarks[:, 0]), np.max(right_iris_landmarks[:, 1])]
                eyes_depth = video.get_depth([left_eye_roi, right_eye_roi])
                frame = video.draw_spatial_data(frame, eyes_depth)
            frame = head_pose_estimator.draw_head_pose(frame, model_points, mp_rotation_vector, mp_translation_vector)
            #frame = gaze_estimator.draw_gaze(frame, detected_faces, pitch, yaw)
            if face_aligner.cropped_eye_image is not None:
                frame[0:face_aligner.crop_frame_height, (video.frame_width-face_aligner.crop_frame_width):video.frame_width] = face_aligner.cropped_eye_image
            if aligned_crop_eyes is not None:
                frame[0:aligned_crop_height, (video.frame_width-aligned_crop_width):video.frame_width] = cv2.addWeighted(frame[0:aligned_crop_height, (video.frame_width-aligned_crop_width):video.frame_width], 0.5, aligned_crop_eyes, 0.5, 0.0)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

if __name__ == "__main__":
    main(use_depth=True)