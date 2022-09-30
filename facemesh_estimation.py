import mediapipe as mp
import numpy as np
import cv2
from PIL import Image

class FaceMeshEstimation:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_estimator = self.mp_face_mesh.FaceMesh(static_image_mode=False, 
                                                              refine_landmarks=True,
                                                              min_detection_confidence=self.min_detection_confidence, 
                                                              min_tracking_confidence=self.min_tracking_confidence)
        self.landmarks = None
        self.face = None
        
    def get_facemesh(self, frame_rgb):
        results = self.face_mesh_estimator.process(frame_rgb)
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            self.landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )
            self.landmarks = self.landmarks.T
            self.face = [[np.min(self.landmarks[0, :]), np.min(self.landmarks[1, :]), np.max(self.landmarks[0, :]), np.max(self.landmarks[1, :])]]
        return self.landmarks, self.face

    def draw_disconnected_rect(self, frame, pt1, pt2, color, thickness):
        img = frame
        width = pt2[0] - pt1[0]
        height = pt2[1] - pt1[1]
        line_width = min(20, width // 4)
        line_height = min(20, height // 4)
        line_length = max(line_width, line_height)
        cv2.line(img, pt1, (pt1[0] + line_length, pt1[1]), color, thickness)
        cv2.line(img, pt1, (pt1[0], pt1[1] + line_length), color, thickness)
        cv2.line(
            img, (pt2[0] - line_length, pt1[1]
                  ), (pt2[0], pt1[1]), color, thickness
        )
        cv2.line(
            img, (pt2[0], pt1[1]), (pt2[0], pt1[1] +
                                    line_length), color, thickness
        )
        cv2.line(
            img, (pt1[0], pt2[1]), (pt1[0] +
                                    line_length, pt2[1]), color, thickness
        )
        cv2.line(
            img, (pt1[0], pt2[1] - line_length), (pt1[0],
                                                  pt2[1]), color, thickness
        )
        cv2.line(img, pt2, (pt2[0] - line_length, pt2[1]), color, thickness)
        cv2.line(img, (pt2[0], pt2[1] - line_length), pt2, color, thickness)
        return img

    def norm2abs(self, x_y, frame_size, pad_w, pad_h):
        x = int(x_y[0] * frame_size - pad_w)
        y = int(x_y[1] * frame_size - pad_h)
        return (x, y)

    def draw_detected_face(self, frame, detected_faces):
        if frame is not None and detected_faces is not None:
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            for face_coordinates in detected_faces:    
                x1 = int(face_coordinates[0] * frame_width)
                y1 = int(face_coordinates[1] * frame_height)
                x2 = int(face_coordinates[2] * frame_width)
                y2 = int(face_coordinates[3] * frame_height)
                frame = self.draw_disconnected_rect(frame, (x1, y1), (x2, y2), (234, 187, 105), 2)
        return frame
        