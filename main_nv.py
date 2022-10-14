import cv2
import numpy as np
import time
import csv
from video_capture import VideoCapture
from face_alignment import FaceAlignment
from facemesh_estimation import FaceMeshEstimation
from iris_detection import IrisDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation as GazeEstimation2
from gaze_estimation_nv import GazeEstimation
from blink_counter import BlinkCounter
from face_landmarks_detection import FaceLandmarkDetection
from utils import eye_converter

EAR_THRESH = 0.1

record_data = False

def save_data_to_csv(pixel_distances, metric_distances, aligned_pixel_distances):
    right_eye_pixels = []
    left_eye_pixels = []
    right_eye_metric = []
    left_eye_metric = []
    warpped_right_eye_pixels = []
    warpped_left_eye_pixels = []

    for i in range(len(pixel_distances)):
        pixel = pixel_distances[i]
        left_eye_pixels.append([i, pixel[0]])
        right_eye_pixels.append([i, pixel[1]])

    fields = ['Frame', 'Pixel Distance'] 
    with open('right_eye_pixel.csv', 'w') as f:      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(right_eye_pixels)
    with open('left_eye_pixel.csv', 'w') as f:      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(left_eye_pixels)

    for i in range(len(metric_distances)):
        metric = metric_distances[i]
        left_eye_metric.append([i, metric[0]])
        right_eye_metric.append([i, metric[1]])

    fields = ['Frame', 'Metric Distance'] 
    with open('right_eye_metric.csv', 'w') as f:      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(right_eye_metric)
    with open('left_eye_metric.csv', 'w') as f:      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(left_eye_metric)

    for i in range(len(aligned_pixel_distances)):
        pixel = pixel_distances[i]
        warpped_left_eye_pixels.append([i, pixel[0]])
        warpped_right_eye_pixels.append([i, pixel[1]])

    fields = ['Frame', 'Warpped Pixel Distance'] 
    with open('warpped_right_eye_pixel.csv', 'w') as f:      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(warpped_right_eye_pixels)
    with open('warpped_left_eye_pixel.csv', 'w') as f:      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(warpped_left_eye_pixels)

def draw_face_depth(frame, depth):
    bar_width = 50
    bar_height = 280
    margin = 50
    frame = cv2.rectangle(frame, (margin, frame.shape[0] - margin), (margin + bar_width, frame.shape[0] - margin - bar_height), (255, 255, 255), 2)
    frame = cv2.putText(frame, "0 cm", (margin, int(frame.shape[0] - margin / 2.2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (144,238,144), 2)
    frame = cv2.putText(frame, "100 cm", (margin, int(frame.shape[0] - margin - bar_height - margin / 4.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (144,238,144), 2)
    depth_bar_height = int(depth / 1000 * bar_height)
    frame = cv2.rectangle(frame, (margin + 2, frame.shape[0] - margin - 2), (margin + bar_width - 2, frame.shape[0] - margin - depth_bar_height), (144,238,144), -1)
    frame = cv2.putText(frame, str(int(depth/10)) + " cm", (margin + bar_width + 10, frame.shape[0] - margin - depth_bar_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (144, 238, 144), 2)
    return frame

def draw_eye_displacement(frame, pixel_distance, metric_distance):
    right_margin = 350
    top_margin = 50
    text_margin = 180
    cv2.putText(frame, "     Displacement", (frame.shape[1] - right_margin, top_margin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 0, 127), 2)
    cv2.putText(frame, "Left Eye        Right Eye", (frame.shape[1] - right_margin, top_margin + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 0, 127), 2)
    cv2.putText(frame, " " + str(round(pixel_distance[0])) + " pixels" , (frame.shape[1] - right_margin, top_margin + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 0, 127), 2)
    cv2.putText(frame, " " + str(round(pixel_distance[1])) + " pixels" , (frame.shape[1] - right_margin + text_margin, top_margin + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 0, 127), 2)
    cv2.putText(frame, " " + str(round(metric_distance[0])) + " mm", (frame.shape[1] - right_margin, top_margin + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 0, 127), 2)
    cv2.putText(frame, " " + str(round(metric_distance[1])) + " mm" , (frame.shape[1] - right_margin + text_margin, top_margin + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 0, 127), 2)
    return frame

def main(use_depth):
    global record_data
    video = VideoCapture(use_depth=use_depth)
    ref_photo = cv2.imread("./face_photos/michael.png")
    facemesh_estimator = FaceMeshEstimation()
    face_aligner = FaceAlignment(ref_photo, facemesh_estimator)
    iris_detector = IrisDetection(video.frame_width, video.frame_height, video.focal_length,smooth_factor=0.2)
    head_pose_estimator = HeadPoseEstimation(video.frame_width, video.frame_height, video.camera_matrix)
    gaze_estimator = GazeEstimation(video, video.frame_width, video.frame_height)
    face_landmark_estimator = FaceLandmarkDetection()
    # gaze_estimator.calibrate(video, face_landmark_estimator, head_pose_estimator)
    gaze_estimator2 = GazeEstimation2(video.frame_width, video.frame_height)
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
            face_depth = 0
            if use_depth:
                if right_iris_landmarks is not None and left_iris_landmarks is not None:
                    left_eye_roi = [np.min(left_iris_landmarks[:, 0]), np.min(left_iris_landmarks[:, 1]), np.max(left_iris_landmarks[:, 0]), np.max(left_iris_landmarks[:, 1])]
                    right_eye_roi = [np.min(right_iris_landmarks[:, 0]), np.min(right_iris_landmarks[:, 1]), np.max(right_iris_landmarks[:, 0]), np.max(right_iris_landmarks[:, 1])]
                    eyes_depth = video.get_depth([left_eye_roi, right_eye_roi])
                    face_depth = (eyes_depth[0].spatialCoordinates.z + eyes_depth[1].spatialCoordinates.z) / 2
                #frame = video.draw_spatial_data(frame, eyes_depth)
            pixel_distance, metric_distance = eye_converter(frame.copy(), video, left_iris_landmarks[0], right_iris_landmarks[0], landmarks[:, 1], landmarks[:, 8], warpped=False, left_eye_depth_mm=eyes_depth[0].spatialCoordinates.z, right_eye_depth_mm=eyes_depth[1].spatialCoordinates.z)
            # Perform face alignment
            #aligned_frame, aligned_landmarks, aligned_right_iris, aligned_left_iris = face_aligner.get_aligned_face(frame, landmarks, right_iris_landmarks, left_iris_landmarks)
            #aligned_crop_eyes = None
            #if aligned_frame is not None:
                #aligned_crop_width, aligned_crop_height, aligned_crop_eyes = aligned_facemesh_estimator.crop_eye_image(aligned_frame, aligned_landmarks)
            #    aligned_pixel_distance, _ = eye_converter(aligned_frame.copy(), video, aligned_right_iris[0], aligned_left_iris[0], aligned_landmarks[:, 1], aligned_landmarks[:, 8], warpped=True)
            if record_data:
                pixel_distances.append(pixel_distance)
                metric_distances.append(metric_distance)
            #    aligned_pixel_distances.append(aligned_pixel_distance)
            metric_landmarks, pose_transform_mat, image_points, model_points, mp_rotation_vector, mp_translation_vector = head_pose_estimator.get_head_pose(landmarks)
            _, landmark_68 = face_landmark_estimator.detect_landmarks(frame_rgb)
            pitch, yaw = gaze_estimator.get_gaze(frame, detected_faces, landmark_68, show=False)
            pitch2, yaw2 = gaze_estimator2.get_gaze(frame, detected_faces)
            # print("pitch: ", pitch, " yaw: ", yaw)
            
            facemesh_frame = facemesh_estimator.draw_facemesh(frame.copy(), landmarks)
            frame = cv2.addWeighted(frame, 0.6, facemesh_frame, 0.4, 0.0)
            frame = draw_face_depth(frame, face_depth)
            frame = draw_eye_displacement(frame, pixel_distance, metric_distance)
            frame = blink_count.draw_blinks(frame)
            frame = facemesh_estimator.draw_detected_face(frame, detected_faces)
            frame = iris_detector.draw_iris(frame, right_iris_landmarks, left_iris_landmarks, left_depth, right_depth)
            frame = head_pose_estimator.draw_head_pose(frame, model_points, mp_rotation_vector, mp_translation_vector)
            frame = gaze_estimator.draw_gaze(frame, left_iris_landmarks, right_iris_landmarks, pitch, yaw)
            frame = gaze_estimator.draw_gaze(frame, left_iris_landmarks, right_iris_landmarks, pitch2, yaw2, color=(255, 0, 0))
            # if face_aligner.cropped_eye_image is not None:
            #     frame[0:face_aligner.crop_frame_height, (video.frame_width-face_aligner.crop_frame_width):video.frame_width] = face_aligner.cropped_eye_image
            # if aligned_crop_eyes is not None:
            #     frame[0:aligned_crop_height, (video.frame_width-aligned_crop_width):video.frame_width] = cv2.addWeighted(frame[0:aligned_crop_height, (video.frame_width-aligned_crop_width):video.frame_width], 0.5, aligned_crop_eyes, 0.5, 0.0)
            #print("Total time:", time.time() - t1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord("r"):
                if record_data:
                    print("Stop data recording")
                    record_data = False
                    save_data_to_csv(pixel_distances, metric_distances, aligned_pixel_distances)
                    pixel_distances = []
                    metric_distances = []
                    aligned_pixel_distances = []
                else:
                    print("Start recording data")
                    record_data = True

if __name__ == "__main__":
    main(use_depth=True)