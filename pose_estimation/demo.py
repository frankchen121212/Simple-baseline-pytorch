

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import json

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_final_preds
from utils.utils import create_logger
from utils.transforms import get_affine_transform

import dataset
import models
from models.pose_resnet import get_pose_net
import numpy as np
import cv2


def _load_coco_person_detection_results(bbox_file,image_names):
    all_boxes = None
    with open(bbox_file, 'r') as f:
        all_boxes = json.load(f)

    if not all_boxes:
        print('error!')
        return None

    print('=> Total boxes: {}'.format(len(all_boxes)))

    kpt_db = []
    num_boxes = 0
    for n_img in range(0, len(all_boxes)):
        det_res = all_boxes[n_img]
        if det_res['category_id'] != 1:
            continue
        imag_names = self.image_path_from_index(det_res['image_id'])
        box = det_res['bbox']
        score = det_res['score']

        if score < config.TEST.IMAGE_THRE:
            continue

        num_boxes = num_boxes + 1

        center, scale = _box2cs(box)
        joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
        joints_3d_vis = np.ones(
            (self.num_joints, 3), dtype=np.float)
        kpt_db.append({
            'image': img_name,
            'center': center,
            'scale': scale,
            'score': score,
            'joints_3d': joints_3d,
            'joints_3d_vis': joints_3d_vis,
        })

    print('=> Total boxes after fliter low score@{}: {}'.format(
        config.TEST.IMAGE_THRE, num_boxes))
    return kpt_db

def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

## load model
update_config('../experiments/coco/resnet50/test.yaml')
config.TEST.MODEL_FILE = '/share1/home/chunyang/files/flowtrack/lib/estimation/models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar'

model = get_pose_net(config, is_train=False)
model.load_state_dict(torch.load(config.TEST.MODEL_FILE))

## Load an image
image_file = 'COCO_test2015_000000001669.jpg'
img_files = json.load(open('../data/posetrack/posetrack_data/annotations/val/000342_mpii_test.json'))
image_names = []
for i in range(len(img_files['images'])):
    image_names.append(img_files['images'][i]['file_name'])

data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
if data_numpy is None:
    raise ValueError('Fail to read {}'.format(image_file))

# object detection box
box = [300, 100, 200, 250]
c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
r = 0

trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
input = cv2.warpAffine(
    data_numpy,
    trans,
    (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
    flags=cv2.INTER_LINEAR)

# vis transformed image
cv2.imshow('image', input)
cv2.waitKey(1000)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
input = transform(input).unsqueeze(0)

# switch to evaluate mode
model.eval()

with torch.no_grad():
    # compute output heatmap
    output = model(input)

    # compute coordinate
    preds, maxvals = get_final_preds(
        config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))

    # plot
    image = data_numpy.copy()
    for mat in preds[0]:
        x, y = int(mat[0]), int(mat[1])
        cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

    # vis result
    cv2.imshow('res', image)
    cv2.waitKey(0)