import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import pickle

from PIL import Image
from l2cs_model import L2CS
from utils import draw_gaze
from few_shot_gaze.undistorter import Undistorter
from few_shot_gaze.models import DTED
from few_shot_gaze.normalization import normalize
from few_shot_gaze.KalmanFilter1D import Kalman1D
from few_shot_gaze.monitor import monitor
from few_shot_gaze.person_calibration import collect_data, fine_tune
from few_shot_gaze.head import PnPHeadPoseEstimator
class GazeEstimation:
    def __init__(self, video, frame_width, frame_height):
        self.video = video
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ted_parameters_path = 'few_shot_gaze/weights/weights_ted.pth.tar'
        self.maml_parameters_path = 'few_shot_gaze/weights/weights_maml'
        k = 9
        self.model = gaze_network = DTED(
            growth_rate=32,
            z_dim_app=64,
            z_dim_gaze=2,
            z_dim_head=16,
            decoder_input_c=32,
            normalize_3d_codes=True,
            normalize_3d_codes_axis=1,
            backprop_gaze_to_encoder=False,
        )
        self.transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.compute_device = torch.device("cpu")
        print('Loading snapshot.')
        # Load T-ED weights if available
        print('> Loading: %s' % self.ted_parameters_path)
        ted_weights = torch.load(self.ted_parameters_path, map_location=self.compute_device)
        if next(iter(ted_weights.keys())).startswith('module.'):
            ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

        #####################################

        # Load MAML MLP weights if available
        full_maml_parameters_path = self.maml_parameters_path +'/%02d.pth.tar' % k
        print('> Loading: %s' % full_maml_parameters_path)
        maml_weights = torch.load(full_maml_parameters_path, map_location=self.compute_device)
        ted_weights.update({  # rename to fit
            'gaze1.weight': maml_weights['layer01.weights'],
            'gaze1.bias':   maml_weights['layer01.bias'],
            'gaze2.weight': maml_weights['layer02.weights'],
            'gaze2.bias':   maml_weights['layer02.bias'],
        })
        self.model.load_state_dict(ted_weights)

        self.model.to(self.compute_device)
        self.model.eval()

        self.kalman_filters = list()
        for point in range(2):
            # initialize kalman filters for different coordinates
            # will be used for face detection over a single object
            self.kalman_filters.append(Kalman1D(sz=100, R=0.01 ** 2))

        # self.kalman_filters_landm = list()
        # for point in range(68):
        #     # initialize Kalman filters for different coordinates
        #     # will be used to smooth landmarks over the face for a single face tracking
        #     self.kalman_filters_landm.append(Kalman1D(sz=100, R=0.005 ** 2))

        self.kalman_filter_gaze = list()
        self.kalman_filter_gaze.append(Kalman1D(sz=100, R=0.01 ** 2))
        self.mon = monitor()
        self.undistorter = Undistorter(self.video.camera_matrix, self.video.camera_distortion)
        self.head_pose_estimator = PnPHeadPoseEstimator()

        # self.softmax = nn.Softmax(dim=1)
        # self.idx_tensor = [idx for idx in range(90)]
        # self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.compute_device)
    def process(self, subject, cap, face_landmark_estimator, head_pose_estimator, mon, device, gaze_network, por_available=False, show=False):

        g_t = None
        data = {'image_a': [], 'gaze_a': [], 'head_a': [], 'R_gaze_a': [], 'R_head_a': []}
        if por_available:
            f = open('./%s_calib_target.pkl' % subject, 'rb')
            targets = pickle.load(f)

        frames_read = 0
        ret, img = cap.read()
        # img = cv2.imread('./michael_face.jpg')
        while ret:
            print(0)
            img = self.undistorter.apply(img)
            if por_available:
                g_t = targets[frames_read]
            frames_read += 1
            print(0.1)

            # detect face
            detected_faces, landmarks = face_landmark_estimator.detect_landmarks(img)
            print(0.2)

            if len(detected_faces) > 0:
                face_location = detected_faces[0]
                face_location[0] *= self.video.frame_width
                face_location[1] *= self.video.frame_height
                face_location[2] *= self.video.frame_width
                face_location[3] *= self.video.frame_height
                print(face_location)
    
                # use kalman filter to smooth bounding box position
                # assume work with complex numbers:
                output_tracked = self.kalman_filters[0].update(face_location[0] + 1j * face_location[1])
                face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
                output_tracked = self.kalman_filters[1].update(face_location[2] + 1j * face_location[3])
                face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)


                landmarks[:, 0] *= self.video.frame_width
                landmarks[:, 1] *= self.video.frame_height
                print("landmarks", landmarks)
                # # run Kalman filter on landmarks to smooth them
                # for i in range(68):
                #     kalman_filters_landm_complex = self.kalman_filters_landm[i].update(pts[i, 0] + 1j * pts[i, 1])
                #     pts[i, 0], pts[i, 1] = np.real(kalman_filters_landm_complex), np.imag(kalman_filters_landm_complex)

                # compute head pose
                # fx, _, cx, _, fy, cy, _, _, _ = self.video.camera_matrix.flatten()
                # camera_parameters = np.asarray([fx, fy, cx, cy])

                fx, _, cx, _, fy, cy, _, _, _ = self.video.camera_matrix.flatten()
                camera_parameters = np.asarray([fx, fy, cx, cy])
                rvec, tvec = head_pose_estimator.fit_func(landmarks, camera_parameters)

                # scaled_landmarks = landmarks.copy()
                # scaled_landmarks[0, :] *= self.video.frame_width
                # scaled_landmarks[1, :] *= self.video.frame_height
                # metric_landmarks, pose_transform_mat, image_points, model_points, rvec, tvec = head_pose_estimator.get_head_pose(landmarks)

                ######### GAZE PART #########

                # create normalized eye patch and gaze and head pose value,
                # if the ground truth point of regard is given
                head_pose = (rvec, tvec)
                print("r", rvec)
                print("t", tvec)
                por = None
                if por_available:
                    por = np.zeros((3, 1))
                    por[0] = g_t[0]
                    por[1] = g_t[1]
                entry = {
                        'full_frame': img,
                        '3d_gaze_target': por,
                        'camera_parameters': self.video.camera_matrix,
                        'full_frame_size': (img.shape[0], img.shape[1]),
                        'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                                              int(face_location[2] - face_location[0]),
                                              int(face_location[3] - face_location[1]))
                        }
                [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose)
                # cv2.imshow('raw patch', patch)

                # estimate the PoR using the gaze network
                processed_patch = self.preprocess_image(patch)
                processed_patch = processed_patch[np.newaxis, :, :, :]

                # Functions to calculate relative rotation matrices for gaze dir. and head pose
                def R_x(theta):
                    sin_ = np.sin(theta)
                    cos_ = np.cos(theta)
                    return np.array([
                        [1., 0., 0.],
                        [0., cos_, -sin_],
                        [0., sin_, cos_]
                    ]).astype(np.float32)

                def R_y(phi):
                    sin_ = np.sin(phi)
                    cos_ = np.cos(phi)
                    return np.array([
                        [cos_, 0., sin_],
                        [0., 1., 0.],
                        [-sin_, 0., cos_]
                    ]).astype(np.float32)

                def calculate_rotation_matrix(e):
                    return np.matmul(R_y(e[1]), R_x(e[0]))

                def pitchyaw_to_vector(pitchyaw):

                    vector = np.zeros((3, 1))
                    vector[0, 0] = np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
                    vector[1, 0] = np.sin(pitchyaw[0])
                    vector[2, 0] = np.cos(pitchyaw[0]) * np.cos(pitchyaw[1])
                    return vector

                # compute the ground truth POR if the
                # ground truth is available
                print(1)
                R_head_a = calculate_rotation_matrix(h_n)
                R_gaze_a = np.zeros((1, 3, 3))
                print(2)
                if type(g_n) is np.ndarray:
                    R_gaze_a = calculate_rotation_matrix(g_n)

                    print(3)
                    # verify that g_n can be transformed back
                    # to the screen's pixel location shown
                    # during calibration
                    gaze_n_vector = pitchyaw_to_vector(g_n)
                    gaze_n_forward = -gaze_n_vector
                    g_cam_forward = inverse_M * gaze_n_forward

                    # compute the POR on z=0 plane
                    d = -gaze_cam_origin[2] / g_cam_forward[2]
                    por_cam_x = gaze_cam_origin[0] + d * g_cam_forward[0]
                    por_cam_y = gaze_cam_origin[1] + d * g_cam_forward[1]
                    por_cam_z = 0.0

                    x_pixel_gt, y_pixel_gt = mon.camera_to_monitor(por_cam_x, por_cam_y)
                    # verified for correctness of calibration targets
                print(4)

                input_dict = {
                    'image_a': processed_patch,
                    'gaze_a': g_n,
                    'head_a': h_n,
                    'R_gaze_a': R_gaze_a,
                    'R_head_a': R_head_a,
                }
                if por_available:
                    data['image_a'].append(processed_patch)
                    data['gaze_a'].append(g_n)
                    data['head_a'].append(h_n)
                    data['R_gaze_a'].append(R_gaze_a)
                    data['R_head_a'].append(R_head_a)

            # read the next frame
            ret, img = cap.read()

        return data

    def calibrate(self, video, face_landmark_estimator, head_pose_estimator):
        data = collect_data(video, self.mon, calib_points=9, rand_points=4)
        # adjust steps and lr for best results
        # To debug calibration, set show=True
        self.model = fine_tune("test", data, self.process, face_landmark_estimator, self.head_pose_estimator, self.mon, self.compute_device, self.model, 9, steps=1000, lr=1e-5, show=False)


    def preprocess_image(self, image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        cv2.imshow('processed patch', image)
        cv2.waitKey(0)
        image = np.transpose(image, [2, 0, 1])  # CxHxW
        image = 2.0 * image / 255.0 - 1
        return image
    def get_gaze(self, frame, detected_faces, head_pose, por_available=False, show=True):
        pitch = []
        yaw = []
        if frame is not None and detected_faces is not None:
            for face_coordinates in detected_faces:
                x_min = int(face_coordinates[0] * self.frame_width)
                y_min = int(face_coordinates[1] * self.frame_height)
                x_max = int(face_coordinates[2] * self.frame_width)
                y_max = int(face_coordinates[3] * self.frame_height)
                if x_min < 0:
                    x_min = 0
                if y_min < 0:
                    y_min = 0
                if x_max >= self.frame_width:
                    x_max = self.frame_width - 1
                if y_max >= self.frame_height:
                    y_max = self.frame_height - 1


                # create normalized eye patch and gaze and head pose value,
                # if the ground truth point of regard is given
                # head_pose = (rvec, tvec)
                por = None
                # if por_available:
                #     por = np.zeros((3, 1))
                #     por[0] = g_t[0]
                #     por[1] = g_t[1]
                entry = {
                        'full_frame': frame,
                        '3d_gaze_target': por,
                        'camera_parameters': self.video.camera_matrix,
                        'full_frame_size': (frame.shape[0], frame.shape[1]),
                        'face_bounding_box': (x_min, y_min, x_max - x_min, y_max - y_min)
                        }
                [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose)
                # cv2.imshow('raw patch', patch)

                # estimate the PoR using the gaze network
                processed_patch = self.preprocess_image(patch)
                processed_patch = processed_patch[np.newaxis, :, :, :]

                # Functions to calculate relative rotation matrices for gaze dir. and head pose
                def R_x(theta):
                    sin_ = np.sin(theta)
                    cos_ = np.cos(theta)
                    return np.array([
                        [1., 0., 0.],
                        [0., cos_, -sin_],
                        [0., sin_, cos_]
                    ]).astype(np.float32)

                def R_y(phi):
                    sin_ = np.sin(phi)
                    cos_ = np.cos(phi)
                    return np.array([
                        [cos_, 0., sin_],
                        [0., 1., 0.],
                        [-sin_, 0., cos_]
                    ]).astype(np.float32)

                def calculate_rotation_matrix(e):
                    return np.matmul(R_y(e[1]), R_x(e[0]))

                def pitchyaw_to_vector(pitchyaw):

                    vector = np.zeros((3, 1))
                    vector[0, 0] = np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
                    vector[1, 0] = np.sin(pitchyaw[0])
                    vector[2, 0] = np.cos(pitchyaw[0]) * np.cos(pitchyaw[1])
                    return vector

                # compute the ground truth POR if the
                # ground truth is available
                R_head_a = calculate_rotation_matrix(h_n)
                R_gaze_a = np.zeros((1, 3, 3))
                if type(g_n) is np.ndarray:
                    R_gaze_a = calculate_rotation_matrix(g_n)

                    # verify that g_n can be transformed back
                    # to the screen's pixel location shown
                    # during calibration
                    gaze_n_vector = pitchyaw_to_vector(g_n)
                    gaze_n_forward = -gaze_n_vector
                    g_cam_forward = inverse_M * gaze_n_forward

                    # compute the POR on z=0 plane
                    d = -gaze_cam_origin[2] / g_cam_forward[2]
                    por_cam_x = gaze_cam_origin[0] + d * g_cam_forward[0]
                    por_cam_y = gaze_cam_origin[1] + d * g_cam_forward[1]
                    por_cam_z = 0.0

                    x_pixel_gt, y_pixel_gt = self.mon.camera_to_monitor(por_cam_x, por_cam_y)
                    # verified for correctness of calibration targets

                input_dict = {
                    'image_a': processed_patch,
                    'gaze_a': g_n,
                    'head_a': h_n,
                    'R_gaze_a': R_gaze_a,
                    'R_head_a': R_head_a,
                }
                                    # compute eye gaze and point of regard
                for k, v in input_dict.items():
                    input_dict[k] = torch.FloatTensor(v).to(self.compute_device).detach()
                if por_available:
                    data['image_a'].append(processed_patch)
                    data['gaze_a'].append(g_n)
                    data['head_a'].append(h_n)
                    data['R_gaze_a'].append(R_gaze_a)
                    data['R_head_a'].append(R_head_a)

                self.model.eval()
                output_dict = self.model(input_dict)
                output = output_dict['gaze_a_hat']
                output_np = np.squeeze(output.cpu().detach().numpy(), axis=0)
                output_np /= np.linalg.norm(output_np)
                yaw_predicted = np.arctan(output_np[1] / output_np[0])
                pitch_predicted = np.arctan(np.sqrt(output_np[0] ** 2 + output_np[1]**2) / output_np[2])
                pitch.append(pitch_predicted)
                yaw.append(yaw_predicted)
                if show:
                    g_cnn = output.data.cpu().numpy()
                    g_cnn = g_cnn.reshape(3, 1)
                    g_cnn /= np.linalg.norm(g_cnn)

                    # compute the POR on z=0 plane
                    g_n_forward = -g_cnn
                    g_cam_forward = inverse_M * g_n_forward
                    g_cam_forward = g_cam_forward / np.linalg.norm(g_cam_forward)

                    d = -gaze_cam_origin[2] / g_cam_forward[2]
                    por_cam_x = gaze_cam_origin[0] + d * g_cam_forward[0]
                    por_cam_y = gaze_cam_origin[1] + d * g_cam_forward[1]
                    por_cam_z = 0.0

                    x_pixel_hat, y_pixel_hat = self.mon.camera_to_monitor(por_cam_x, por_cam_y)

                    # output_tracked = self.kalman_filter_gaze[0].update(x_pixel_hat + 1j * y_pixel_hat)
                    # x_pixel_hat, y_pixel_hat = np.ceil(np.real(output_tracked)), np.ceil(np.imag(output_tracked))

                    # show point of regard on screen
                    display = np.ones((self.mon.h_pixels, self.mon.w_pixels, 3), np.float32)
                    h, w, c = patch.shape
                    display[0:h, int(self.mon.w_pixels/2 - w/2):int(self.mon.w_pixels/2 + w/2), :] = 1.0 * patch / 255.0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if type(g_n) is np.ndarray:
                        cv2.putText(display, '.', (x_pixel_gt, y_pixel_gt), font, 0.5, (0, 0, 0), 10, cv2.LINE_AA)
                    cv2.putText(display, '.', (int(x_pixel_hat), int(y_pixel_hat)), font, 0.5, (0, 0, 255), 10, cv2.LINE_AA)
                    cv2.namedWindow("por", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("por", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('por', display)

                    # also show the face:
                    # cv2.rectangle(img, (int(face_location[0]), int(face_location[1])),
                    #               (int(face_location[2]), int(face_location[3])), (255, 0, 0), 2)
                    # self.landmarks_detector.plot_markers(img, pts)
                    # self.head_pose_estimator.drawPose(img, rvec, tvec, self.cam_calib['mtx'], np.zeros((1, 4)))
                    # cv2.imshow('image', img)
        return pitch, yaw

    def draw_gaze(self, frame, left_eye, right_eye, pitch, yaw):
        if frame is not None and left_eye is not None and right_eye is not None and pitch is not None and yaw is not None:
            frame = draw_gaze(left_eye, right_eye, frame, (pitch[0], yaw[0]), color=(0,0,255))
            # for i in range(len(detected_faces)):
            #     face_coordinates = detected_faces[i]
            #     x_min = int(face_coordinates[0] * self.frame_width)
            #     y_min = int(face_coordinates[1] * self.frame_height)
            #     x_max = int(face_coordinates[2] * self.frame_width)
            #     y_max = int(face_coordinates[3] * self.frame_height)
            #     if x_min < 0:
            #         x_min = 0
            #     if y_min < 0:
            #         y_min = 0
            #     if x_max >= self.frame_width:
            #         x_max = self.frame_width - 1
            #     if y_max >= self.frame_height:
            #         y_max = self.frame_height - 1
            #     bbox_width = x_max - x_min
            #     bbox_height = y_max - y_min
                
        return frame