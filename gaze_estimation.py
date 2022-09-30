import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision
from PIL import Image
from l2cs_model import L2CS
from utils import draw_gaze

class GazeEstimation:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.snapshot_path = "models/L2CSNet_gaze360.pkl"
        self.model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90) # ResNet-50 model trained on Gaze360
        self.transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.compute_device = torch.device("mps")
        print('Loading snapshot.')
        saved_state_dict = torch.load(self.snapshot_path, map_location=self.compute_device)
        self.model.load_state_dict(saved_state_dict)
        self.model.to(self.compute_device)
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.compute_device)

    def get_gaze(self, frame, detected_faces):
        pitch = []
        yaw = []
        if frame is not None and detected_faces is not None:
            for face in detected_faces:
                face_coordinates = face['face_coordinates']
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

                # Crop image
                img = frame[y_min:y_max, x_min:x_max]
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                img = self.transformations(im_pil)
                img  = Variable(img).to(self.compute_device)
                img  = img.unsqueeze(0)
                
                # gaze prediction
                gaze_pitch, gaze_yaw = self.model(img)
                
                pitch_predicted = self.softmax(gaze_pitch)
                yaw_predicted = self.softmax(gaze_yaw)
                
                # Get continuous predictions in degrees.
                pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 4 - 180
                yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 4 - 180
                
                pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                pitch.append(pitch_predicted)
                yaw.append(yaw_predicted)

        return pitch, yaw

    def draw_gaze(self, frame, detected_faces, pitch, yaw):
        if frame is not None and detected_faces is not None and pitch is not None and yaw is not None:
            for i in range(len(detected_faces)):
                face = detected_faces[i]
                face_coordinates = face['face_coordinates']
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
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                frame = draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch[i], yaw[i]), color=(0,0,255))
        return frame