import cv2
import numpy as np
from skimage import transform as trans
import time
# index of upper head center: 8
# indeox of nose tip: 1
# index of left mouth: 57
# index of right mouth: 287
# index of mouth-nose: 164
# index of left inner_eye: 244
# index of right inner_eye: 464
# index of left outer_eye: 130
# index of right outer_eye: 359

class FaceAlignment:
    def __init__(self, ref_photo, facemesh_estimator):
        self.ref_photo = ref_photo
        self.facemesh_estimator = facemesh_estimator
        landmarks, detected_faces = self.facemesh_estimator.get_facemesh(self.ref_photo, convert_rgb=True)
        #detected_face = detected_faces[0]
        #self.face_width = int((detected_face[2] - detected_face[0]) * self.ref_photo.shape[1])
        #self.face_height = int((detected_face[3] - detected_face[1]) * self.ref_photo.shape[0])
        self.ref_landmarks = np.array([landmarks[:2, 164], landmarks[:2, 244], landmarks[:2, 464], landmarks[:2, 130], landmarks[:2, 359], landmarks[:2, 57], landmarks[:2, 287]], dtype=np.float32)
        #self.ref_landmarks[:, 0] = self.ref_landmarks[:, 0] * self.ref_photo.shape[1] - detected_face[0] *  self.ref_photo.shape[1]
        #self.ref_landmarks[:, 1] = self.ref_landmarks[:, 1] * self.ref_photo.shape[0] - detected_face[1] * self.ref_photo.shape[0]
        self.ref_landmarks[:, 0] = self.ref_landmarks[:, 0] * self.ref_photo.shape[1]
        self.ref_landmarks[:, 1] = self.ref_landmarks[:, 1] * self.ref_photo.shape[0]
        self.ref_landmarks = np.expand_dims(self.ref_landmarks, axis=0)
        self.crop_frame_width, self.crop_frame_height, self.cropped_eye_image = self.facemesh_estimator.crop_eye_image(self.ref_photo, landmarks)


    def get_aligned_face(self, frame, landmarks):
        aligned_frame = None
        if frame is not None and landmarks is not None:
            extracted_landmarks = np.array([landmarks[:2, 164], landmarks[:2, 244], landmarks[:2, 464], landmarks[:2, 130], landmarks[:2, 359], landmarks[:2, 57], landmarks[:2, 287]], dtype=np.float32)
            extracted_landmarks[:, 0] = extracted_landmarks[:, 0] * frame.shape[1]
            extracted_landmarks[:, 1] = extracted_landmarks[:, 1] * frame.shape[0]
            transformation = self.get_transform(extracted_landmarks[:, :2], self.ref_landmarks)
            aligned_frame = cv2.warpAffine(
                frame, transformation, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
        # aligned_landmarks = cv2.warpAffine(
        #     landmarks_2d, transformation, (self.face_width, self.face_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        # )
        return aligned_frame

    def estimate_norm(self, lmk, ref_landmarks):
        assert lmk.shape == (ref_landmarks.shape[1], ref_landmarks.shape[2])
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(ref_landmarks.shape[1]), axis=1)
        min_M = []
        min_index = []
        min_error = float("inf")
        src = ref_landmarks
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def get_transform(self, landmark, ref_landmarks):
        M, _ = self.estimate_norm(landmark, ref_landmarks)
        return M