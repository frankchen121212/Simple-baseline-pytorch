# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )

def save_whole_images_with_dt_and_flow_bbox(data_numpy,meta,c_flow,s_flow,c,s,perfix):
    file_name='{}_bbox.jpg'.format(perfix)
    img=data_numpy
    flow_bbox_width=int(s_flow[0]*160)
    dt_bbox_width=int(s[0]*160)
    flow_bbox_height=int(s_flow[1]*160)
    dt_bbox_height = int(s[1] * 160)
    cv2.putText(data_numpy, 'image_id'+str(meta['image_id'][0].numpy()),
                (int(data_numpy.shape[0]*0.5),50), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 0])


    flow_bbox_left_up=(int(c_flow[0]-flow_bbox_width*0.5),int(c_flow[1]-flow_bbox_height*0.5))
    flow_bbox_righ_up=(int(c_flow[0]+flow_bbox_width*0.5),int(c_flow[1]-flow_bbox_height*0.5))
    flow_bbox_left_down=(int(c_flow[0]-flow_bbox_width*0.5),int(c_flow[1]+flow_bbox_height*0.5))
    flow_bbox_righ_down=(int(c_flow[0]+flow_bbox_width*0.5),int(c_flow[1]+flow_bbox_height*0.5))

    dt_bbox_left_up = (int(c[0] - dt_bbox_width * 0.5), int(c[1] - dt_bbox_height * 0.5))
    dt_bbox_righ_up = (int(c[0] + dt_bbox_width * 0.5), int(c[1] - dt_bbox_height * 0.5))
    dt_bbox_left_down = (int(c[0] - dt_bbox_width * 0.5), int(c[1] + dt_bbox_height * 0.5))
    dt_bbox_righ_down = (int(c[0] + dt_bbox_width * 0.5), int(c[1] + dt_bbox_height * 0.5))

    cv2.putText(data_numpy,'flow_bbox',flow_bbox_left_up,cv2.FONT_HERSHEY_COMPLEX,1,[0,255,255])
    cv2.line(data_numpy,flow_bbox_left_up,flow_bbox_righ_up,[0,255,255],2)
    cv2.line(data_numpy,flow_bbox_left_up,flow_bbox_left_down,[0,255,255],2)
    cv2.line(data_numpy, flow_bbox_left_down, flow_bbox_righ_down, [0, 255, 255], 2)
    cv2.line(data_numpy, flow_bbox_righ_up, flow_bbox_righ_down, [0, 255, 255], 2)

    cv2.putText(data_numpy, 'dt_bbox', dt_bbox_left_up, cv2.FONT_HERSHEY_COMPLEX, 1,[0, 0, 255])
    cv2.line(data_numpy, dt_bbox_left_up, dt_bbox_righ_up, [0, 0, 255], 2)
    cv2.line(data_numpy, dt_bbox_left_up, dt_bbox_left_down, [0, 0, 255], 2)
    cv2.line(data_numpy, dt_bbox_left_down, dt_bbox_righ_down, [0, 0, 255], 2)
    cv2.line(data_numpy, dt_bbox_righ_up, dt_bbox_righ_down, [0, 0, 255], 2)

    cv2.imwrite(file_name,img)