import cv2
import numpy as np
from skimage import transform as trans

class FaceAlignment:
    def __init__(self):
        self.refernce_landmarks = []