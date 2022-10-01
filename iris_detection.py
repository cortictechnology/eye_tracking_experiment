import cv2
import numpy as np
from custom.iris_lm_depth import from_landmarks_to_depth
from PIL import Image

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
SMALL_CIRCLE_SIZE = 1
LARGE_CIRCLE_SIZE = 2

class IrisDetection:
    def __init__(self, frame_width, frame_height, focal_length):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.image_size = (self.frame_width, self.frame_height)
        self.focal_length = focal_length
        self.left_eye_landmarks_id = np.array([33, 133])
        self.right_eye_landmarks_id = np.array([362, 263])
        self.points_idx = [33, 133, 362, 263, 61, 291, 199]
        self.points_idx = list(set(self.points_idx))
        self.points_idx.sort()
        self.left_eye_landmarks_id = np.array([33, 133])
        self.right_eye_landmarks_id = np.array([362, 263])
        self.dist_coeff = np.zeros((4, 1))
        self.smooth_left_depth = -1
        self.smooth_right_depth = -1
        self.smooth_factor = 0.1

    def get_iris(self, frame, landmarks, convert_rgb=False):
        eye_landmarks = None
        iris_landmarks = None
        if frame is not None and landmarks is not None:
            frame_rgb = frame
            if convert_rgb:
                cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
            (
                left_depth,
                left_iris_size,
                left_iris_landmarks,
                left_eye_contours,
            ) = from_landmarks_to_depth(
                frame_rgb,
                landmarks[:, self.left_eye_landmarks_id],
                self.image_size,
                is_right_eye=False,
                focal_length=self.focal_length,
            )

            (
                right_depth,
                right_iris_size,
                right_iris_landmarks,
                right_eye_contours,
            ) = from_landmarks_to_depth(
                frame_rgb,
                landmarks[:, self.right_eye_landmarks_id],
                self.image_size,
                is_right_eye=True,
                focal_length=self.focal_length,
            )

            if self.smooth_right_depth < 0:
                self.smooth_right_depth = right_depth
            else:
                self.smooth_right_depth = (
                    self.smooth_right_depth * (1 - self.smooth_factor)
                    + right_depth * self.smooth_factor
                )

            if self.smooth_left_depth < 0:
                self.smooth_left_depth = left_depth
            else:
                self.smooth_left_depth = (
                    self.smooth_left_depth * (1 - self.smooth_factor)
                    + left_depth * self.smooth_factor
                )
        return right_iris_landmarks, left_iris_landmarks, self.smooth_left_depth / 10, self.smooth_right_depth / 10


    def draw_iris(self, frame, right_iris_landmarks, left_iris_landmarks, left_depth, right_depth):
        if frame is None or right_iris_landmarks is None or left_iris_landmarks is None:
            return frame
        iris_landmarks = np.concatenate(
            [
                right_iris_landmarks,
                left_iris_landmarks,
            ]
        )
        for landmark in iris_landmarks:
            pos = (np.array(self.image_size) * landmark[:2]).astype(np.int32)
            frame = cv2.circle(frame, tuple(pos), SMALL_CIRCLE_SIZE, YELLOW, -1)

        # write depth values into frame
        depth_string = "{:.2f}cm, {:.2f}cm".format(
            left_depth, right_depth
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
        return frame