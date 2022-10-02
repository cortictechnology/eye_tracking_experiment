import cv2
import numpy as np
from custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

class HeadPoseEstimation:
    def __init__(self, frame_width, frame_height, camera_matrix, smooth_factor=0.1):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_matrix = camera_matrix
        self.points_idx = [33, 263, 61, 291, 199]
        self.points_idx = self.points_idx + [key for (key, val) in procrustes_landmark_basis]
        self.points_idx = list(set(self.points_idx))
        self.points_idx.sort()
        self.dist_coeff = np.zeros((4, 1))
        self.pcf = PCF(
            near=1,
            far=10000,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            fy=self.camera_matrix[1, 1],
        )
        self.mp_rotation_vector = None
        self.mp_translation_vector = None
        self.smooth_factor = smooth_factor

    def get_head_pose(self, landmarks):
        metric_landmarks = None
        pose_transform_mat = None
        image_points = None
        model_points = None
        mp_rotation_vector = None
        mp_translation_vector = None
        if landmarks is not None:
            refined_landmarks = landmarks[:, :468]
            metric_landmarks, pose_transform_mat = get_metric_landmarks(
                refined_landmarks.copy(), self.pcf
            )

            image_points = (
                landmarks[0:2, self.points_idx].T
                * np.array([self.frame_width, self.frame_height])[None, :]
            )
            model_points = metric_landmarks[0:3, self.points_idx].T

            # see here:
            # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
            pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
            mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
            mp_translation_vector = pose_transform_mat[:3, 3, None]
            if self.mp_rotation_vector is None:
                self.mp_rotation_vector = mp_rotation_vector
            else:
                self.mp_rotation_vector = mp_rotation_vector * (1 - self.smooth_factor) + self.mp_rotation_vector * self.smooth_factor
            
            if self.mp_translation_vector is None:
                self.mp_translation_vector = mp_translation_vector
            else:
                self.mp_translation_vector = mp_translation_vector * (1 - self.smooth_factor) + self.mp_translation_vector * self.smooth_factor
        return metric_landmarks, pose_transform_mat, image_points, model_points, self.mp_rotation_vector, self.mp_translation_vector

    def draw_head_pose(self, frame, model_points, mp_rotation_vector, mp_translation_vector):
        if frame is not None and model_points is not None and mp_rotation_vector is not None and mp_translation_vector is not None:
            nose_tip = model_points[0]
            nose_tip_extended = 2.5 * model_points[0]
            (nose_pointer2D, jacobian) = cv2.projectPoints(
                np.array([nose_tip, nose_tip_extended]),
                mp_rotation_vector,
                mp_translation_vector,
                self.camera_matrix,
                self.dist_coeff,
            )

            nose_tip_2D, nose_tip_2D_extended = nose_pointer2D.squeeze().astype(int)
            frame = cv2.line(
                frame, nose_tip_2D, nose_tip_2D_extended, (255, 0, 0), 2
            )
        return frame
