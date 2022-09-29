import argparse
import numpy as np
import cv2
import time
import depthai as dai
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import draw_gaze
from PIL import Image, ImageOps

from face_detector import FaceDetector
from model import L2CS


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    args = parse_args()

    arch=args.arch
    batch_size = 1
    snapshot_path = "models/L2CSNet_gaze360.pkl"
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    compute_device = torch.device("mps")
    model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path, map_location=compute_device)
    model.load_state_dict(saved_state_dict)
    model.to(compute_device)
    model.eval()
    
    image_size = (1280, 720)
    pipeline = dai.Pipeline()
    device = dai.Device()
    caliData = device.readCalibration()
    K_col = np.array(caliData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, image_size[0], image_size[1]), dtype=float)
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewSize(image_size[0], image_size[1])
    camRgb.setInterleaved(False)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    xoutRgb.input.setQueueSize(1)
    xoutRgb.input.setBlocking(False)
    camRgb.preview.link(xoutRgb.input)
    device.startPipeline(pipeline)
    
    qRgb = device.getOutputQueue("rgb", 1)

    softmax = nn.Softmax(dim=1)
    detector = FaceDetector()
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(compute_device)
    x=0

    with torch.no_grad():
        while True:
            frame = qRgb.get().getCvFrame()
            start_fps = time.time()  
           
            faces = detector.run_inference(frame)
            if faces is not None:
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                for face in faces:
                    face_coordinates = face['face_coordinates']
                    x_min = int(face_coordinates[0] * frame_width)
                    y_min = int(face_coordinates[1] * frame_height)
                    x_max = int(face_coordinates[2] * frame_width)
                    y_max = int(face_coordinates[3] * frame_height)
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    img  = Variable(img).to(compute_device)
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)
                    
                    
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0
                    
                    draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Gaze Estimation",frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
