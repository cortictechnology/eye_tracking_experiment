import cv2
import numpy as np
from PIL import Image
from face_alignment2 import FaceAlignment
from facemesh_estimation import FaceMeshEstimation
from iris_detection import IrisDetection

def main():
    ref_photo = cv2.imread("./face_photos/michael.png")
    frame = cv2.imread("./face_photos/michael_side_2.png")
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
    facemesh_estimator = FaceMeshEstimation(static_mode=True)
    face_aligner = FaceAlignment(ref_photo, facemesh_estimator)
    iris_detector = IrisDetection(1280, 720, 850, smooth_factor=0.2)

    landmarks, detected_faces = facemesh_estimator.get_facemesh(frame_rgb)
    right_iris_landmarks, left_iris_landmarks, left_depth, right_depth = iris_detector.get_iris(frame_rgb, landmarks)
    aligned_frame, aligned_landmarks, aligned_right_iris, aligned_left_iris = face_aligner.get_aligned_face(frame, landmarks, right_iris_landmarks, left_iris_landmarks)
    aligned_crop_width, aligned_crop_height, aligned_crop_eyes  = facemesh_estimator.crop_eye_image(aligned_frame, face_aligner.start_x, face_aligner.end_x, face_aligner.start_y, face_aligner.end_y)
    frame[0:face_aligner.crop_frame_height, (1280-face_aligner.crop_frame_width):1280] = face_aligner.cropped_eye_image
    frame[0:aligned_crop_height, (1280-aligned_crop_width):1280] = cv2.addWeighted(frame[0:aligned_crop_height, (1280-aligned_crop_width):1280], 0.5, aligned_crop_eyes, 0.5, 0.0)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
