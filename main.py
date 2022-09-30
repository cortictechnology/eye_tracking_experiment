import cv2
import numpy as np
from video_capture import VideoCapture
from face_alignment import FaceAlignment
from facemesh_estimation import FaceMeshEstimation
from iris_detection import IrisDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation

def main(use_depth=False):
    video = VideoCapture(use_depth=use_depth)
    face_aligner = FaceAlignment() # Not implemented yet
    facemesh_estimator = FaceMeshEstimation()
    iris_detector = IrisDetection(video.frame_width, video.frame_height, video.focal_length)
    head_pose_estimator = HeadPoseEstimation(video.frame_width, video.frame_height, video.camera_matrix)
    gaze_estimator = GazeEstimation(video.frame_width, video.frame_height)
    while True:
        frame, frame_rgb = video.get_frame()
        if frame is not None and frame_rgb is not None:
            #detected_faces = face_detector.run_inference(frame_rgb)
            landmarks, detected_faces = facemesh_estimator.get_facemesh(frame_rgb)
            right_iris_landmarks, left_iris_landmarks, left_depth, right_depth = iris_detector.get_iris(frame_rgb, landmarks)
            metric_landmarks, pose_transform_mat, image_points, model_points, mp_rotation_vector, mp_translation_vector = head_pose_estimator.get_head_pose(landmarks)
            pitch, yaw = gaze_estimator.get_gaze(frame, detected_faces)
            frame = facemesh_estimator.draw_detected_face(frame, detected_faces)
            frame = iris_detector.draw_iris(frame, right_iris_landmarks, left_iris_landmarks, left_depth, right_depth)
            if use_depth:
                left_eye_roi = [np.min(left_iris_landmarks[:, 0]), np.min(left_iris_landmarks[:, 1]), np.max(left_iris_landmarks[:, 0]), np.max(left_iris_landmarks[:, 1])]
                right_eye_roi = [np.min(right_iris_landmarks[:, 0]), np.min(right_iris_landmarks[:, 1]), np.max(right_iris_landmarks[:, 0]), np.max(right_iris_landmarks[:, 1])]
                eyes_depth = video.get_depth([left_eye_roi, right_eye_roi])
                frame = video.draw_spatial_data(frame, eyes_depth)
            frame = head_pose_estimator.draw_head_pose(frame, model_points, mp_rotation_vector, mp_translation_vector)
            frame = gaze_estimator.draw_gaze(frame, detected_faces, pitch, yaw)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

if __name__ == "__main__":
    main(True)