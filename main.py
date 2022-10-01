import cv2
import numpy as np
import time
from video_capture import VideoCapture
from face_alignment import FaceAlignment
from facemesh_estimation import FaceMeshEstimation
from iris_detection import IrisDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from blink_counter import BlinkCounter
from utils import eye_converter, calculate_avg_ear

EAR_THRESH = 0.1

record_data = False

def save_data_to_csv(pixel_distances, metric_distances, aligned_pixel_distances):
    pass

def main(use_depth):
    global record_data
    video = VideoCapture(use_depth=use_depth)
    ref_photo = cv2.imread("./face_photos/michael.jpg")
    facemesh_estimator = FaceMeshEstimation()
    aligned_facemesh_estimator = FaceMeshEstimation()
    face_aligner = FaceAlignment(ref_photo, facemesh_estimator)
    iris_detector = IrisDetection(video.frame_width, video.frame_height, video.focal_length)
    head_pose_estimator = HeadPoseEstimation(video.frame_width, video.frame_height, video.camera_matrix)
    gaze_estimator = GazeEstimation(video.frame_width, video.frame_height)
    blink_count = BlinkCounter(EAR_THRESH, video.frame_width, video.frame_height)
    pixel_distances = []
    metric_distances = []
    aligned_pixel_distances = []
    while True:
        #t1 = time.time()
        frame, frame_rgb = video.get_frame()
        if frame is not None and frame_rgb is not None:
            landmarks, detected_faces = facemesh_estimator.get_facemesh(frame_rgb)
            num_blinks = blink_count.count_blink(landmarks)
            right_iris_landmarks, left_iris_landmarks, left_depth, right_depth = iris_detector.get_iris(frame_rgb, landmarks)
            if use_depth:
                left_eye_roi = [np.min(left_iris_landmarks[:, 0]), np.min(left_iris_landmarks[:, 1]), np.max(left_iris_landmarks[:, 0]), np.max(left_iris_landmarks[:, 1])]
                right_eye_roi = [np.min(right_iris_landmarks[:, 0]), np.min(right_iris_landmarks[:, 1]), np.max(right_iris_landmarks[:, 0]), np.max(right_iris_landmarks[:, 1])]
                eyes_depth = video.get_depth([left_eye_roi, right_eye_roi])
                #frame = video.draw_spatial_data(frame, eyes_depth)
            pixel_distance, metric_distance = eye_converter(frame.copy(), video, left_iris_landmarks[0], right_iris_landmarks[0], landmarks[:, 1], landmarks[:, 8], warpped=False, left_eye_depth_mm=eyes_depth[0].spatialCoordinates.z, right_eye_depth_mm=eyes_depth[1].spatialCoordinates.z)
            # Perform face alignment
            aligned_frame, aligned_landmarks, aligned_right_iris, aligned_left_iris = face_aligner.get_aligned_face(frame, landmarks, right_iris_landmarks, left_iris_landmarks)
            aligned_crop_eyes = None
            if aligned_frame is not None:
                aligned_crop_width, aligned_crop_height, aligned_crop_eyes = aligned_facemesh_estimator.crop_eye_image(aligned_frame, aligned_landmarks)
                aligned_pixel_distance, _ = eye_converter(aligned_frame.copy(), video, aligned_right_iris[0], aligned_left_iris[0], aligned_landmarks[:, 1], aligned_landmarks[:, 8], warpped=True)
            if record_data:
                pixel_distances.append(pixel_distance)
                metric_distances.append(metric_distance)
                aligned_pixel_distances.append(aligned_pixel_distance)
            #metric_landmarks, pose_transform_mat, image_points, model_points, mp_rotation_vector, mp_translation_vector = head_pose_estimator.get_head_pose(landmarks)
            #pitch, yaw = gaze_estimator.get_gaze(frame, detected_faces)
            frame = blink_count.draw_blinks(frame)
            frame = facemesh_estimator.draw_detected_face(frame, detected_faces)
            frame = iris_detector.draw_iris(frame, right_iris_landmarks, left_iris_landmarks, left_depth, right_depth)
            #frame = head_pose_estimator.draw_head_pose(frame, model_points, mp_rotation_vector, mp_translation_vector)
            #frame = gaze_estimator.draw_gaze(frame, detected_faces, pitch, yaw)
            if face_aligner.cropped_eye_image is not None:
                frame[0:face_aligner.crop_frame_height, (video.frame_width-face_aligner.crop_frame_width):video.frame_width] = face_aligner.cropped_eye_image
            if aligned_crop_eyes is not None:
                frame[0:aligned_crop_height, (video.frame_width-aligned_crop_width):video.frame_width] = cv2.addWeighted(frame[0:aligned_crop_height, (video.frame_width-aligned_crop_width):video.frame_width], 0.5, aligned_crop_eyes, 0.5, 0.0)
            #print("Total time:", time.time() - t1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == "r":
                if record_data:
                    record_data = False
                    save_data_to_csv(pixel_distances, metric_distances, aligned_pixel_distances)
                    pixel_distances = []
                    metric_distances = []
                    aligned_pixel_distances = []
                else:
                    record_data = True

if __name__ == "__main__":
    main(use_depth=True)