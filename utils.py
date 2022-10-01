import numpy as np
import torch
import torch.nn as nn
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin
from pathlib import Path
import subprocess
import re
from l2cs_model import L2CS
import torchvision
import sys

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def eye_converter(video, left_eye_2d, right_eye_2d, face_center_p1_2d, face_center_p2_2d, warpped=False, left_eye_depth_mm=None, right_eye_depth_mm=None):
    p1 = face_center_p1_2d[:2]
    p2 = face_center_p2_2d[:2]
    # frame = cv2.line(frame, (int(p1[0] * video.frame_width), int(p1[1] * video.frame_height)), (int(p2[0] * video.frame_width), int(p2[1] * video.frame_height)), (0, 0, 255), 1) 
    p3 = left_eye_2d[:2]
    p4 = right_eye_2d[:2]
    # frame = cv2.line(frame, (int(p3[0] * video.frame_width), int(p3[1] * video.frame_height)), (int(p4[0] * video.frame_width), int(p4[1] * video.frame_height)), (0, 255, 0), 1) 

    denom = ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
    origin_x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom
    origin_y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom
    # frame = cv2.circle(frame, (int(origin_x * video.frame_width), int(origin_y * video.frame_height)), 5, (255, 0, 0), -1)
    if (warpped):
        left_eye_dist_px = np.sqrt((((p3[0] - origin_x) * video.frame_width) ** 2 + ((p3[1] - origin_y) * video.frame_height) ** 2))
        right_eye_dist_px = np.sqrt((((p4[0] - origin_x) * video.frame_width) ** 2 + ((p4[1] - origin_y) * video.frame_height) ** 2))
        frame = cv2.putText(frame, f"{int(left_eye_dist_px)} px", (int(p3[0] * video.frame_width), int(p3[1] * video.frame_height) + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        frame = cv2.putText(frame, f"{int(right_eye_dist_px)} px", (int(p4[0] * video.frame_width), int(p4[1] * video.frame_height) + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        # cv2.imshow("Eye distance", frame)
        return (left_eye_dist_px, right_eye_dist_px), None
    else:

        left_eye_dist_px = np.sqrt((((p3[0] - origin_x) * video.frame_width) ** 2 + ((p3[1] - origin_y) * video.frame_height) ** 2))
        right_eye_dist_px = np.sqrt((((p4[0] - origin_x) * video.frame_width) ** 2 + ((p4[1] - origin_y) * video.frame_height) ** 2))

        eye_dist_2d_px = left_eye_dist_px + right_eye_dist_px
        eye_dist_2d_mm = eye_dist_2d_px / video.focal_length * left_eye_depth_mm
        print(eye_dist_2d_mm)
        
        eye_dist_mm = np.sqrt(eye_dist_2d_mm ** 2 + (left_eye_depth_mm - right_eye_depth_mm) ** 2)
        left_eye_dist_mm = left_eye_dist_px / eye_dist_2d_px * eye_dist_mm
        right_eye_dist_mm = eye_dist_mm - left_eye_dist_mm
        # frame = cv2.putText(frame, f"{int(left_eye_dist_px)}px, {int(left_eye_dist_mm)}mm", (int(p3[0] * video.frame_width) - 50, int(p3[1] * video.frame_height) + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        # frame = cv2.putText(frame, f"{int(right_eye_dist_px)}px, {int(right_eye_dist_mm)}mm", (int(p4[0] * video.frame_width), int(p4[1] * video.frame_height) + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0))
        # cv2.imshow("Eye distance", frame)
        return (left_eye_dist_px, right_eye_dist_px), (left_eye_dist_mm, right_eye_dist_mm)



def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w/2
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out    

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv3 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    return torch.device('cuda:0' if cuda else 'cpu')

def spherical2cartesial(x):
    
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])

    return output
    
def compute_angular_error(input,target):

    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1,3,1)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180*torch.mean(output_dot)/math.pi
    return output_dot

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result
   
def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository
        

