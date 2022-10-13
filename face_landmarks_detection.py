import numpy as np
import cv2
from PIL import Image
from landmarks_lib.api.facer import FaceAna

class FaceLandmarkDetection:

    def __init__(self):
        self.facer = FaceAna()

    def detect_landmarks(self, frame, convert_rgb=False):
        frame_rgb = frame.copy()
        if convert_rgb:
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=np.uint8)
        boxes = None
        landmarks = None
        if frame_rgb is not None:
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, landmarks, states = self.facer.run(frame_rgb)
            if (landmarks.shape[0] > 0):
                landmarks = landmarks[0, :, :]

                landmarks[:, 0] = landmarks[:, 0] / frame_rgb.shape[1]
                landmarks[:, 1] = landmarks[:, 1] / frame_rgb.shape[0]

        return boxes, landmarks

    def draw_landmarks(self, frame, landmarks):
        if frame is not None and landmarks is not None:
            for landmarks_index in range(landmarks.shape[0]):
                x_y = landmarks[landmarks_index]
                x_y[0] = x_y[0] * frame.shape[1]
                x_y[1] = x_y[1] * frame.shape[0]
                frame = cv2.circle(frame, (int(x_y[0]), int(x_y[1])), 2,
                        (222, 222, 222), -1)
                #frame = cv2.putText(frame, str(landmarks_index ), (int(x_y[0]), int(x_y[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
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
