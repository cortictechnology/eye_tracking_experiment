import mediapipe as mp
import numpy as np
import cv2
from data import triangulation
from PIL import Image

COLOR = [(0, 153, 0), (234, 187, 105), (80, 190, 168), (0, 0, 255)]

# index of crop top left: 70
# index of crop bottom right: 340

class FaceMeshEstimation:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_mode=False, smooth_factor=0.1):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_estimator = self.mp_face_mesh.FaceMesh(static_image_mode=static_mode, 
                                                              refine_landmarks=True,
                                                              min_detection_confidence=self.min_detection_confidence, 
                                                              min_tracking_confidence=self.min_tracking_confidence)
        self.landmarks = None
        self.face = None
        self.smooth_factor = smooth_factor
        
    def get_facemesh(self, frame, convert_rgb=False):
        frame_rgb = frame
        if convert_rgb:
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
        results = self.face_mesh_estimator.process(frame_rgb)
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )
            if self.landmarks is None:
                self.landmarks = face_landmarks.T
            else:
                self.landmarks = face_landmarks.T * (1 - self.smooth_factor) + self.smooth_factor * self.landmarks
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
                frame = self.draw_disconnected_rect(frame, (x1, y1), (x2, y2), (234, 105, 105), 2)
        return frame

    def draw_facemesh(self, frame, landmarks):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        for i in range(int(triangulation.shape[0] / 3)):
            pt1 = np.array(landmarks[0:2, triangulation[i * 3]], np.float32)
            pt2 = np.array(landmarks[0:2, triangulation[i * 3 + 1]], np.float32)
            pt3 = np.array(landmarks[0:2, triangulation[i * 3 + 2]], np.float32)
            pt1[0] = int(pt1[0] * frame_width)
            pt1[1] = int(pt1[1] * frame_height)
            pt2[0] = int(pt2[0] * frame_width)
            pt2[1] = int(pt2[1] * frame_height)
            pt3[0] = int(pt3[0] * frame_width)
            pt3[1] = int(pt3[1] * frame_height)
            pt1 = pt1.astype(np.int32)
            pt2 = pt2.astype(np.int32)
            pt3 = pt3.astype(np.int32)
            cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), COLOR[1], thickness=1)
            cv2.line(frame, (pt2[0], pt2[1]), (pt3[0], pt3[1]), COLOR[1], thickness=1)
            cv2.line(frame, (pt3[0], pt3[1]), (pt1[0], pt1[1]), COLOR[1], thickness=1)
        return frame

    def crop_eye_image(self, frame, xmin, xmax, ymin, ymax):
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax >= frame.shape[1]:
            xmax = frame.shape[1] - 1
        if ymax >= frame.shape[0]:
            ymax = frame.shape[0] - 1
        crop_frame_width = xmax - xmin
        crop_frame_height = ymax - ymin
        cropped_eye_image = frame[ymin:ymax, xmin:xmax]

        return crop_frame_width, crop_frame_height, cropped_eye_image
        