import cv2
import numpy as np
from skimage import transform as trans
import time


selected_face_landmarks = [33, 133, 362, 263, 61, 291]

# index of upper head center: 8
# indeox of nose tip: 1
# index of left mouth: 61
# index of right mouth: 291
# index of mouth-nose: 164
# index of left inner_eye: 133
# index of right inner_eye: 362
# index of left outer_eye: 33
# index of right outer_eye: 263

class FaceAlignment:
    def __init__(self, ref_photo, facemesh_estimator):
        self.ref_photo = ref_photo
        self.facemesh_estimator = facemesh_estimator
        landmarks, detected_faces = self.facemesh_estimator.get_facemesh(self.ref_photo, convert_rgb=True)
        #detected_face = detected_faces[0]
        #self.face_width = int((detected_face[2] - detected_face[0]) * self.ref_photo.shape[1])
        #self.face_height = int((detected_face[3] - detected_face[1]) * self.ref_photo.shape[0])
        landmark_list = [landmarks[:2, i] for i in selected_face_landmarks]
        self.ref_landmarks = np.array(landmark_list, dtype=np.float32)
        #self.ref_landmarks[:, 0] = self.ref_landmarks[:, 0] * self.ref_photo.shape[1] - detected_face[0] *  self.ref_photo.shape[1]
        #self.ref_landmarks[:, 1] = self.ref_landmarks[:, 1] * self.ref_photo.shape[0] - detected_face[1] * self.ref_photo.shape[0]
        self.ref_landmarks[:, 0] = self.ref_landmarks[:, 0] * self.ref_photo.shape[1]
        self.ref_landmarks[:, 1] = self.ref_landmarks[:, 1] * self.ref_photo.shape[0]
        self.crop_x_extend_ratio = 0.05
        self.crop_y_extend_ratio = 0.1

        x_min = np.min(self.ref_landmarks[:, 0])
        x_max = np.max(self.ref_landmarks[:, 0])
        y_min = np.min(self.ref_landmarks[:4, 1])
        y_max = np.max(self.ref_landmarks[:4, 1])
        self.start_x = int(x_min * (1-self.crop_x_extend_ratio))
        self.end_x = int(x_max * (1+self.crop_x_extend_ratio))
        self.start_y = int(y_min * (1-self.crop_y_extend_ratio))
        self.end_y = int(y_max * (1+self.crop_y_extend_ratio))
        
        for i in range(self.ref_landmarks.shape[0]):
            print("(Reference)Point: ", str(i), ": x:", int(self.ref_landmarks[i, 0]), ", y:", self.ref_landmarks[i, 1])
            cv2.circle(self.ref_photo, (int(self.ref_landmarks[i, 0]), int(self.ref_landmarks[i, 1])), 2, (0, 255, 0), -1)
            cv2.putText(self.ref_photo, str(i), (int(self.ref_landmarks[i, 0]), int(self.ref_landmarks[i, 1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        cv2.imshow("Ref", self.ref_photo)
        self.ref_landmarks = np.expand_dims(self.ref_landmarks, axis=0)
        self.crop_frame_width, self.crop_frame_height, self.cropped_eye_image = self.facemesh_estimator.crop_eye_image(self.ref_photo, self.start_x, self.end_x, self.start_y, self.end_y)

    def get_aligned_face(self, frame, landmarks, right_iris, left_iris):
        aligned_frame = None
        if frame is not None and landmarks is not None:
            landmarks_2d = landmarks.copy()
            landmarks_2d[0, :] = landmarks_2d[0, :] * frame.shape[1]
            landmarks_2d[1, :] = landmarks_2d[1, :] * frame.shape[0]
            landmarks_2d[2, :] = 1.0
            right_iris_scaled = right_iris.copy()
            left_iris_scaled = left_iris.copy()
            right_iris_scaled[:, 0] = right_iris_scaled[:, 0] * frame.shape[1]
            right_iris_scaled[:, 1] = right_iris_scaled[:, 1] * frame.shape[0]
            right_iris_scaled[:, 2] = 1.0
            left_iris_scaled[:, 0] = left_iris_scaled[:, 0] * frame.shape[1]
            left_iris_scaled[:, 1] = left_iris_scaled[:, 1] * frame.shape[0]
            left_iris_scaled[:, 2] = 1.0
            landmark_list = [landmarks[:2, i] for i in selected_face_landmarks]
            extracted_landmarks = np.array(landmark_list)
            #extracted_landmarks = np.array([landmarks[:2, 164], landmarks[:2, 244], landmarks[:2, 464], landmarks[:2, 130], landmarks[:2, 359], landmarks[:2, 61], landmarks[:2, 291]], dtype=np.float32)
            extracted_landmarks[:, 0] = extracted_landmarks[:, 0] * frame.shape[1]
            extracted_landmarks[:, 1] = extracted_landmarks[:, 1] * frame.shape[0]
            extracted_landmarks_2d = extracted_landmarks.copy().T
            transformation = self.get_transform(extracted_landmarks[:, :2], self.ref_landmarks)
            aligned_frame = cv2.warpPerspective(
                frame, transformation, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR
            )

            for i in range(extracted_landmarks.shape[0]):
                print("(Original)Point: ", str(i), ": x:", int(extracted_landmarks[i, 0]), ", y:", int(extracted_landmarks[i, 1]))
                cv2.circle(frame, (int(extracted_landmarks[i, 0]), int(extracted_landmarks[i, 1])), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(extracted_landmarks[i, 0]), int(extracted_landmarks[i, 1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

            extracted_landmarks_2d = np.insert(extracted_landmarks_2d, 2, values=np.ones(extracted_landmarks_2d.shape[1]), axis=0)
            aligned_extracted_points = np.dot(transformation, extracted_landmarks_2d)
            aligned_extracted_points[0, :] = aligned_extracted_points[0, :] / aligned_extracted_points[2, :]
            aligned_extracted_points[1, :] = aligned_extracted_points[1, :] / aligned_extracted_points[2, :]

            for i in range(aligned_extracted_points.shape[1]):
                print("(Warpped)Point: ", str(i), ": x:", int(aligned_extracted_points[0, i]), ", y:", int(aligned_extracted_points[1, i]))
                cv2.circle(aligned_frame, (int(aligned_extracted_points[0, i]), int(aligned_extracted_points[1, i])), 2, (0, 255, 0), -1)
                cv2.putText(aligned_frame, str(i), (int(aligned_extracted_points[0, i]), int(aligned_extracted_points[1, i])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            cv2.imshow("Aligned", aligned_frame)

            aligned_landmarks_2d = np.dot(transformation, landmarks_2d)
            aligned_landmarks_2d[0, :] = aligned_landmarks_2d[0, :] / aligned_landmarks_2d[2, :]
            aligned_landmarks_2d[1, :] = aligned_landmarks_2d[1, :] / aligned_landmarks_2d[2, :]
            aligned_landmarks_2d[0, :]  = aligned_landmarks_2d[0, :]  / frame.shape[1]
            aligned_landmarks_2d[1, :] = aligned_landmarks_2d[1, :] / frame.shape[0]

            aligned_right_iris = np.dot(transformation, right_iris_scaled.T)
            aligned_right_iris[0, :] = aligned_right_iris[0, :] / aligned_right_iris[2, :]
            aligned_right_iris[1, :] = aligned_right_iris[1, :] / aligned_right_iris[2, :]
            aligned_right_iris[0, :]  = aligned_right_iris[0, :]  / frame.shape[1]
            aligned_right_iris[1, :] = aligned_right_iris[1, :] / frame.shape[0]

            aligned_left_iris = np.dot(transformation, left_iris_scaled.T)
            aligned_left_iris[0, :] = aligned_left_iris[0, :] / aligned_left_iris[2, :]
            aligned_left_iris[1, :] = aligned_left_iris[1, :] / aligned_left_iris[2, :]
            aligned_left_iris[0, :]  = aligned_left_iris[0, :]  / frame.shape[1]
            aligned_left_iris[1, :] = aligned_left_iris[1, :] / frame.shape[0]

        return aligned_frame, aligned_landmarks_2d, aligned_right_iris.T, aligned_left_iris.T

    def estimate_norm(self, lmk, ref_landmarks):
        M, mask = cv2.findHomography(lmk, ref_landmarks, cv2.RANSAC, 5.0)
        
        # assert lmk.shape == (ref_landmarks.shape[1], ref_landmarks.shape[2])
        # tform = trans.ProjectiveTransform()
        # lmk_tran = np.insert(lmk, 2, values=np.ones(ref_landmarks.shape[1]), axis=1)
        # # min_M = []
        # # min_index = []
        # min_error = float("inf")
        # src = ref_landmarks
        # for i in np.arange(src.shape[0]):
        #     results = np.dot(M, lmk_tran.T)
        #     results[0,:] = results[0,:] / results[2,:]
        #     results[1,:] = results[1,:] / results[2,:]
        #     results = results[0:2, :]
        #     results = results.T
        #     error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #     if error < min_error:
        #         min_error = error
        # print(min_error)
        return M, mask

    def get_transform(self, landmark, ref_landmarks):
        M, _ = self.estimate_norm(landmark, ref_landmarks)
        return M