import cv2
from utils import calculate_avg_ear

class BlinkCounter:
    def __init__(self, ear_threshold, frame_width, frame_height):
        self.ear_threshold = ear_threshold
        self.ear_deadzone_threshold = self.ear_threshold * 1.2
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.blink_counter_reset = False
        self.num_blinks = 0
        self.chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
        self.chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
        self.all_chosen_idxs = self.chosen_left_eye_idxs + self.chosen_right_eye_idxs

    def count_blink(self, landmarks):
        EAR, _ = calculate_avg_ear(landmarks.T, self.chosen_left_eye_idxs, self.chosen_right_eye_idxs, self.frame_width, self.frame_height)
        if self.blink_counter_reset:
            if EAR < self.ear_threshold:
                self.num_blinks += 1
                self.blink_counter_reset = False
        if EAR > self.ear_deadzone_threshold:
            if not self.blink_counter_reset:
                self.blink_counter_reset = True
        return self.num_blinks

    def draw_blinks(self, frame):
        right_margin = 350
        top_margin = 200
        frame = cv2.putText(
            frame,
            "Number of blinks:  " + str(self.num_blinks),
            (frame.shape[1] - right_margin, top_margin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (127, 0, 127),
            2
        )
        return frame
