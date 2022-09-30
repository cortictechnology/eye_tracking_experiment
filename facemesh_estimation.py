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
        
    def get_facemesh(self, frame_rgb):
        results = self.face_mesh_estimator.process(frame_rgb)
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            self.landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )
            self.landmarks = self.landmarks.T
        return self.landmarks

        