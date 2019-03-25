# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
import tqdm

import scipy.optimize
import scipy.spatial
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from collections import deque
from collections import Iterable
import mmcv
import copy
import imageio

from core.flow import get_cs_from_flow
from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds, get_max_preds
from utils.transforms import flip_back, get_affine_transform, fliplr_joints
from utils.vis import save_debug_images
from utils.vis import  save_whole_images_with_dt_and_flow_bbox

import torch.nn as nn
from tensorboardX import  SummaryWriter
import json

logger = logging.getLogger(__name__)
MAX_TRACK_IDS = 999
FIRST_TRACK_ID = 0


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input_, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            if meta['image_id'] != 10003420000:
                continue
            root = config.DATASET.ROOT
            file_name = index_to_path(root, meta['image_id'][0].item())
            data_numpy = cv2.imread(file_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            c_dt = meta['center'][0].numpy()
            s_dt = meta['scale'][0].numpy()
            r = 0
            trans = get_affine_transform(c_dt, s_dt, r, config.MODEL.IMAGE_SIZE)
            input = cv2.warpAffine(
                data_numpy,
                trans,
                (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
                flags=cv2.INTER_LINEAR)

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

            input = transform(input)
            # print(type(input))
            # print(input.shape)

            new_input = np.zeros([1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]])
            new_input[0, :, :, :] = input[:, :, :]
            input = torch.from_numpy(new_input).float()



            output = model(input)
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            c_d = meta['center'].numpy()
            s_d = meta['scale'].numpy()


            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c_d, s_d)

            print('id--{},\nkpts:\n{}'.format(meta['image_id'],preds[0]))
            # time.sleep(10)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            # if config.DATASET.DATASET == 'posetrack':
            #     filenames.extend(meta['filename'])
            #     imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# def test(config, val_loader, val_dataset, model, criterion, output_dir,
#              tb_log_dir, writer_dict=None):
#
#     batch_time = AverageMeter()
#     model.eval()
#     num_samples = len(val_dataset)
#     all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
#                          dtype=np.float32)
#     all_boxes = np.zeros((num_samples, 6))
#     image_path = []
#     filenames = []
#     imgnums = []
#     idx = 0
#     with torch.no_grad():
#         end = time.time()
#         for i, (input, _, _, meta) in enumerate(val_loader):
#             # compute output
#             output = model(input)
#             if config.TEST.FLIP_TEST:
#                 # this part is ugly, because pytorch has not supported negative index
#                 # input_flipped = model(input[:, :, :, ::-1])
#                 input_flipped = np.flip(input.cpu().numpy(), 3).copy()
#                 input_flipped = torch.from_numpy(input_flipped).cuda()
#                 output_flipped = model(input_flipped)
#                 output_flipped = flip_back(output_flipped.cpu().numpy(),
#                                            val_dataset.flip_pairs)
#                 output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
#
#                 # feature is not aligned, shift flipped heatmap for higher accuracy
#                 if config.TEST.SHIFT_HEATMAP:
#                     output_flipped[:, :, :, 1:] = \
#                         output_flipped.clone()[:, :, :, 0:-1]
#                     # output_flipped[:, :, :, 0] = 0
#
#                 output = (output + output_flipped) * 0.5
#
#             pred, _ = get_max_preds(output.detach().cpu().numpy())
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             num_images = input.size(0)
#
#             c = meta['center'].numpy()
#             s = meta['scale'].numpy()
#             score = meta['score'].numpy()
#
#             preds, maxvals = get_final_preds(
#                 config, output.clone().cpu().numpy(), c, s)
#
#             all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
#             all_preds[idx:idx + num_images, :, 2:3] = maxvals
#             # double check this all_boxes parts
#             all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
#             all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
#             all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
#             all_boxes[idx:idx + num_images, 5] = score
#             image_path.extend(meta['image'])
#
#             # if config.DATASET.DATASET == 'posetrack':
#             #     import mmcv
#             #     file_name = meta['image']
#             #     fl_name = file_name.split('/')
#             #     if int(fl_name[-1].split['.'][0]) > 0:
#             #         flowfile = '%06d.flo' % (int(fl_name[-1].split['.'][0]) - 1) # we use 000000.flo to optimize the bboxes of 000001.jpg
#             #         flowpath = os.path.join('/share1/home/chunyang/Server/Dataset/posetrack/Flowimgs/flow',
#             #                                fl_name[-3], fl_name[-2], flowfile)
#             #         flow = mmcv.flowread(flowpath)
#             #         loc_x = preds[:, :, 0]
#             #         loc_y = preds[:, :, 1]
#             #         new_x = []
#             #         new_y = []
#             #         for x, y in zip(loc_x, loc_y):
#             #             new_x.append(x + flow[x][y][0])
#             #             new_y.append(y + flow[x][y][1])
#             #         x = min(new_x)
#             #         y = min(new_y)
#             #         w = (max(new_x) - x) / 0.85
#             #         h = (max(new_y) - y) / 0.85
#             #         new_box = [x,y,w,h]
#
#
#
#
#
#
#             # if config.DATASET.DATASET == 'posetrack':
#             #      filenames.extend(meta['filename'])
#             #     imgnums.extend(meta['imgnum'].numpy())
#
#             idx += num_images
#
#             if i % config.PRINT_FREQ == 0:
#                 msg = 'Test: [{0}/{1}]\t' \
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
#                           i, len(val_loader), batch_time=batch_time)
#                 prefix = '{}_{}'.format(os.path.join(output_dir, 'test'), i)
#                 logger.info(msg)
#                 save_images(config, input, meta, pred * 4, output,
#                                   prefix)
#
#         name_values, perf_indicator = val_dataset.evaluate(
#                 config, all_preds, output_dir, all_boxes, image_path,
#                 filenames, imgnums)
#     return perf_indicator


# markdown format output

def validate_with_opticalflow(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    root = config.DATASET.ROOT
    batch_time = AverageMeter()
    # losses = AverageMeter()
    # acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    ### load det bboxs ###
    if os.path.exists(os.path.join(root,'new_full_det_bboxs_'+str(config.TEST.IMAGE_THRE)+'.npy')):
        print('loading new_full_det_bboxs.npy from {}...'.format(os.path.join(root,'new_full_det_bboxs_'+str(config.TEST.IMAGE_THRE)+'.npy')))
        full_bboxs = np.load(os.path.join(root,'new_full_det_bboxs_'+str(config.TEST.IMAGE_THRE)+'.npy')).item()
        print('detection bboxes loaded')
        ids = sorted(full_bboxs.keys())
    else:

        print('creating  new_full_det_bboxs.npy...')

        full_bboxs = {}
        ids = []
        for _, meta in val_loader:
            # print(type(input))
            # print(input.shape)

            image_id = meta['image_id'][0].item()
            if image_id not in ids:
                ids.append(int(image_id))

        #generate ids

        ids = sorted(ids)

        #fulfill the missing ids
        pre_im_id=ids[0]
        for im_id in ids:
            if (im_id-pre_im_id)>1 and (im_id-pre_im_id)<60:
                for i in range(im_id-pre_im_id-1):
                    pre_im_id = pre_im_id + 1
                    if pre_im_id not in ids:
                        ids.append(int(pre_im_id))
                    logger.info('adding missing image_id--{}'.format(pre_im_id))
            pre_im_id=im_id
        ids = sorted(ids)

        temp_key={}
        temp_key['ids']=ids
        # with open(os.path.join(root,'temp_id_vis.json'),'w') as f :
        #     json.dump(temp_key,f,indent=4)
        # print('finish writing temp_id_vis.json')

        for im_id in ids:
            full_bboxs[im_id] = []


        for _, meta in val_loader:
            image_id = meta['image_id'][0].item()
            center = meta['center'].numpy()
            scale = meta['scale'].numpy()
            score = meta['score'].numpy()
            box_sc = np.array(meta['box_sc']) # [[x,y,w,h,score]]


            box = (center, scale, score, box_sc)

            full_bboxs[int(image_id)].append(box)  # {1003420000:[(c1,s1,score1,[x1,y1,w1,h1,score1]),
        np.save(os.path.join(root, 'new_full_det_bboxs'+str(config.TEST.IMAGE_THRE)+'.npy'), full_bboxs)
        print('detection bboxes loaded')


    with torch.no_grad():
        end = time.time()
        batch_time.update(time.time() - end)
        image_path = []
        frames = []
        num_box = 0
        pres, vals, c, s, sc , track_IDs= [], [], [], [], [],[]
        Q = deque(maxlen=config.TEST.TRACK_FRAME_LEN)  # tracked instances queue

        next_track_id=FIRST_TRACK_ID

        for i, im_id in enumerate(ids):
            file_name = index_to_path(root, im_id)
            data_numpy = cv2.imread(file_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            frame_boxs = full_bboxs[im_id]  # get all boxes information in this frame
            boxs = np.array([item[-1] for item in frame_boxs ])
            keep = bbox_nms(boxs, 0.5)  # do the nms for each frame
            if len(keep) == 0:
                nmsed_boxs = frame_boxs
            else:
                nmsed_boxs = [frame_boxs[_keep] for _keep in keep]
            print('current im_id_{}'.format(im_id))

            next_id = im_id + 1
            if next_id in ids:
                image_id = str(im_id)
                root_flow = os.path.join(root,config.TEST.FLOW_PATH)
                folder_name = image_id[1:7] + '_mpii_test'
                flow_name = '00' + image_id[-4:] + '.flo'
                flow_path = os.path.join(root_flow, folder_name, flow_name)
                flow = mmcv.flowread(flow_path)  # [h,w,2]

            instance = []
            which_box=0
            #compute each box
            for box in nmsed_boxs:
                person = {}
                person_flow = {}
                num_box += 1
                c_d = box[0]
                s_d = box[1]
                c_dt = box[0][0]
                s_dt = box[1][0]
                # print('type:{}, value:{}'.format(type(s_d),s_d))
                score = box[2]

                c.append(c_dt)
                s.append(s_dt)
                sc.append(score)
                r = 0

                image_path.append(file_name)


                h, w = data_numpy.shape[0], data_numpy.shape[1]
                trans = get_affine_transform(c_dt, s_dt, r, config.MODEL.IMAGE_SIZE)

                input = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
                    flags=cv2.INTER_LINEAR)

                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])

                input = transform(input)
                new_input=np.zeros([1,3,config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]])
                new_input[0, :, :, :] = input[:, :, :]
                input = torch.from_numpy(new_input).float()


                output = model(input)
                if config.TEST.FLIP_TEST:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    output_flipped = model(input_flipped)
                    output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]
                        # output_flipped[:, :, :, 0] = 0

                    output = (output + output_flipped) * 0.5

                batch_time.update(time.time() - end)
                end = time.time()

                preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c_d, s_d)


                #preds--(1, 17, 2)
                #macvals--(1, 17, 1)
                # if the value of each pred<0.4 set the joint into invisible
                preds_set_invisible , maxvals_set_invisible = get_visible_joints(preds.copy(),maxvals.copy(),config.TEST.IN_VIS_THRE)
                individual = np.concatenate((preds_set_invisible.squeeze(), maxvals_set_invisible.reshape(-1, 1)),axis=1).flatten()  # array with shape(num_keypoints x 3,)
                person['image'] = input
                person['keypoints'] = individual
                # person['bbox'] = bbox[:-1]
                person['score'] = score[0]
                person['track_id'] = None
                person['bbox'] , person['area'] = get_bbox_sc_from_cs(c_dt,s_dt,score)

                instance.append(person)


                pres.append(preds_set_invisible)
                vals.append(maxvals_set_invisible)#[1,17,1]


                #get the avg_joint_score for each box
                joint_score=0
                for joint_i in range(maxvals.shape[1]):
                    joint_score +=maxvals[0][joint_i]
                avg_joint_score = joint_score/maxvals.shape[1]
                #get center and scale from flow
                c_flow, s_flow, box_sc_flow, is_ignore,scale_flow, flow_kps = get_cs_from_flow(flow, preds_set_invisible, h, w, avg_joint_score,config.TEST.FLOW_THRE)  # TODO

                ### save debug bboxes ###

                if (i % config.PRINT_FREQ) == 0 and config.DEBUG.SAVE_ALL_BOXES:
                    file = data_numpy.copy()
                    save_all_boxes_with_joints(file, im_id,which_box, c_dt, s_dt, c_flow[0], s_flow[0],preds_set_invisible,score,avg_joint_score,output_dir)
                which_box += 1

                if is_ignore or next_id not in ids:
                    continue

                box_flow = (c_flow, s_flow, avg_joint_score, box_sc_flow)
                full_bboxs[next_id].append(box_flow)
                individual_flow = np.concatenate((flow_kps, maxvals.reshape(-1, 1)), axis=1).flatten()
                person_flow['keypoints'] = individual_flow
                person_flow['bbox'] = box_sc_flow[:-1]
                person_flow['area'] = scale_flow
                person_flow['track_id'] = None
                person_flow['image'] = input
                person_flow['score'] = score[0]
                instance = instance[:-1]
                instance.append(person_flow)
            ### Assign the Track ID for all instances in one frame ###

            #when the image id is in ids but detector and flow detect nobody in the frame
            if len(instance)==0 :
                continue

            # the last frame of a video
            # go into another video
            if next_id not in ids:
                if config.DEBUG.SAVE_VIDEO_TRACKING:
                    frames = save_image_with_skeleton(data_numpy, im_id, instance, output_dir, frames, next_id, ids)
                IDs,next_track_id = assignID(Q, instance,next_track_id, similarity_thresh=config.TEST.TRACK_SIMILARITY_THRE)
                for i_person, each_person in enumerate(instance):
                    each_person['track_id'] = IDs[i_person]
                track_IDs.extend(IDs)
                Q = deque(maxlen=config.TEST.TRACK_FRAME_LEN)
                next_track_id=FIRST_TRACK_ID
                logger.info('current_im_id{}--go in to next video--next_track_id{}'.format(im_id,next_track_id))
                continue# init the Deque for the next video


            IDs ,next_track_id= assignID(Q, instance,next_track_id, similarity_thresh=config.TEST.TRACK_SIMILARITY_THRE)
            print('IDs--{}'.format(IDs))
            for i_person, each_person in enumerate(instance):
                each_person['track_id'] = IDs[i_person]

            #########################save image with joints and skeletons###################

            if config.DEBUG.SAVE_VIDEO_TRACKING :
                frames = save_image_with_skeleton(data_numpy,im_id,instance,output_dir,frames,next_id,ids)

            track_IDs.extend(IDs)
            Q.append(instance)  # Q:[[{},{},{}],[{},{}],...]

            #print
            if i % (config.PRINT_FREQ) == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      .format(
                    i, len(ids), batch_time=batch_time,)
                logger.info(msg)



        logger.info('boxes number:{}\t'.format(num_box))
        pres = np.array(pres)
        vals = np.array(vals)
        c, s, sc = np.array(c), np.array(s), np.array(sc)
        np.save(os.path.join(root, 'full_pose_results.npy'), pres)
        np.save(os.path.join(root, 'full_pose_scores.npy'), vals)
        total_bboxes = np.zeros((num_box, 6))
        total_preds = np.zeros((num_box, config.MODEL.NUM_JOINTS, 3)
                               , dtype=np.float32)  # num_box x 17 x 3
        total_track_IDs = np.zeros((num_box))

        for i in range(num_box):
            total_preds[i:i+1, :, 0:2] = pres[i, :, :, 0:2]
            total_preds[i:i + 1, :, 2:3] = vals[i]

            total_bboxes[i:i + 1, 0:2] = c[i][0:2]
            total_bboxes[i:i + 1, 2:4] = s[i][0:2]
            total_bboxes[i:i + 1, 4] = np.prod(s*200, 1)[i]
            total_bboxes[i:i + 1, 5] = sc[i]
            total_track_IDs[i] = track_IDs[i]

        name_values, perf_indicator = val_dataset.evaluate(
            config, total_preds, output_dir, total_bboxes, image_path,total_track_IDs)

    return perf_indicator


def flatten(lst):
    result = []
    def fly(lst):
        for item in lst:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes,dict)):
                fly(item)
            else:
                result.append(item)
    fly(lst)
    return result
# compute the OKS between detected pose and propagated pose using flow
def get_bbox_sc_from_cs(center,scale,score):

    pixel_std = 200
    scale = scale / 1.25
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    x = center[0] - w * 0.5
    y = center[1] - h * 0.5
    box_sc = [x, y, w, h, score]
    area = w * h
    return box_sc,area

def computeOks(Queque, instances):
    # dimention here should be Nxm
    gts = Queque
    dts = instances
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0.4)  ### when vg > 0.4, the kpt is visible
        bb = gt['bbox']
        x0 = bb[0] ; x1 = bb[0] + bb[2]
        y0 = bb[1] ; y1 = bb[1] + bb[3]
        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                # dx = np.zeros_like(xd)
                # dy = np.zeros_like(yd)

            e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious

def computeIOU(Queque, instances):
    gts = Queque
    dts = instances
    ious = np.zeros((len(dts), len(gts)))
    #box_sc = [x, y, w, h, score]
    for j_gt, gt in enumerate(gts):
        gt_box = gt['bbox']
        x0_gt,y0_gt,x1_gt,y1_gt = gt_box[0],gt_box[1],gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]
        rec1 = [y0_gt,x0_gt,y1_gt,x1_gt]
        #(y0,x0,y1,x1)
        #(top,left,bottom,right)
        for i_dt, dt in enumerate(dts):
            dt_box = dt['bbox']
            x0, y0, x1, y1 = dt_box[0], dt_box[1], dt_box[0] + dt_box[2], dt_box[1] + dt_box[3]
            rec2 = [y0, x0, y1, x1]

            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])

            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                ious[i_dt,j_gt]=0
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                ious[i_dt, j_gt] = intersect / (sum_area - intersect)
    return ious

def assignID(Queque, instances,next_track_id, similarity_thresh):
    IDs = []  # store the IDs used in this frame
    Queque_flattened = flatten(Queque)
    if len(Queque_flattened) == 0: # the first frame
        matches = -np.ones((len(instances),))
    else:
        #################   compute oks   ######################
        oks = computeOks(Queque_flattened, instances)
        # oks.shape[instances , Queque ]
        iou = computeIOU(Queque_flattened, instances)
        # iou.shape[instances, Queque ]
        C_similarity = oks*0.5 + iou*0.5
        prev_inds = np.argmax(C_similarity, axis=1)
        matches = -np.ones((len(instances),), dtype=np.int32)
        for i_nex, prev in enumerate(prev_inds):
            if C_similarity[i_nex][prev] > similarity_thresh :
                matches[i_nex] = Queque_flattened[prev]['track_id']
            else :
                matches[i_nex]=-1
    #assign ids
    for m in matches:
        #no match set m = -1
        if m == -1 :
            IDs.append(next_track_id)
            next_track_id += 1
            if next_track_id >= MAX_TRACK_IDS:
                next_track_id %= MAX_TRACK_IDS
        else:
            IDs.append(int(m))

    return IDs , next_track_id

def get_visible_joints(preds,maxvals,invisible_thre):
    preds_set_invisible = preds.copy()
    maxvals_set_invisible = maxvals.copy()
    num_invisible_joints = 0
    for i_sc_j in range(maxvals.shape[1]):
        if maxvals[0][i_sc_j][:] < invisible_thre:
            preds_set_invisible[0][i_sc_j][0] = 0
            preds_set_invisible[0][i_sc_j][1] = 0
            num_invisible_joints += 1
    if num_invisible_joints == maxvals.shape[1]:
        the_best_joint_score = np.max(maxvals, axis=1)
        for i_sc_j in range(maxvals.shape[1]):
            if maxvals_set_invisible[0][i_sc_j][:] == the_best_joint_score:
                preds_set_invisible[0][i_sc_j][0] = preds[0][i_sc_j][0]
                preds_set_invisible[0][i_sc_j][1] = preds[0][i_sc_j][1]
    return preds_set_invisible ,maxvals_set_invisible

def generate_target(config,joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((17, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    # assert self.target_type == 'gaussian', \
    #     'Only support gaussian map now!'

    #if self.target_type == 'gaussian':
    target = np.zeros((config.MODEL.NUM_JOINTS,
                       config.MODEL.EXTRA.HEATMAP_SIZE[1],
                       config.MODEL.EXTRA.HEATMAP_SIZE[0]),
                      dtype=np.float32)



    tmp_size = config.MODEL.EXTRA.SIGMA * 3

    for joint_id in range(config.MODEL.NUM_JOINTS):
        feat_stride = config.MODEL.IMAGE_SIZE / config.MODEL.EXTRA.HEATMAP_SIZE
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= config.MODEL.EXTRA.HEATMAP_SIZE[0] or ul[1] >= config.MODEL.EXTRA.HEATMAP_SIZE[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * config.MODEL.EXTRA.SIGMA ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], config.MODEL.EXTRA.HEATMAP_SIZE[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], config.MODEL.EXTRA.HEATMAP_SIZE[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], config.MODEL.EXTRA.HEATMAP_SIZE[0])
        img_y = max(0, ul[1]), min(br[1], config.MODEL.EXTRA.HEATMAP_SIZE[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


def index_to_path(root, index):
    # get image path from image id. only for the val set

    image_set = 'val'

    file_name = '%s.jpg' % str(index)[-4:].zfill(6)  # 10003420099 -> 99 -> 000099.jpg
    suffix = '_mpii_test'
    # TODO: deal with the test dataset
    folder = str(index)[1:-4] + suffix
    image_path = os.path.join(
        root, 'images', image_set, folder, file_name)

    assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)

    return image_path


def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def save_all_boxes_with_joints(file,im_id,which_box,c,s,c_flow,s_flow,batch_joints,score,avg_joint_score,output_dir):

    file_name = '{}_box_{}_joints.png'.format(str(im_id),str(which_box))
    flow_bbox_width = int(s_flow[0] * 160)
    dt_bbox_width = int(s[0] * 160)

    flow_bbox_height = int(s_flow[1] * 160)
    dt_bbox_height = int(s[1] * 160)

    cv2.putText(file, 'image_id' + str(im_id),
                (int(file.shape[0] * 0.5), 50), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 0])

    dt_bbox_left_up = (int(c[0] - dt_bbox_width * 0.5), int(c[1] - dt_bbox_height * 0.5))
    dt_bbox_righ_down = (int(c[0] + dt_bbox_width * 0.5), int(c[1] + dt_bbox_height * 0.5))

    flow_bbox_left_up = (int(c_flow[0] - flow_bbox_width * 0.5), int(c_flow[1] - flow_bbox_height * 0.5))
    flow_bbox_righ_up = (int(c_flow[0] + flow_bbox_width * 0.5), int(c_flow[1] - flow_bbox_height * 0.5+15))
    flow_bbox_righ_down = (int(c_flow[0] + flow_bbox_width * 0.5), int(c_flow[1] + flow_bbox_height * 0.5))

    cv2.putText(file,'{}/{}'.format(str(score),str(avg_joint_score)),flow_bbox_righ_up,cv2.FONT_HERSHEY_COMPLEX, 0.6, [0, 255, 255])

    cv2.putText(file, 'flow', flow_bbox_left_up, cv2.FONT_HERSHEY_COMPLEX, 0.6, [0, 255, 255])
    cv2.rectangle(file, (flow_bbox_left_up[0],flow_bbox_left_up[1]), (flow_bbox_righ_down[0], flow_bbox_righ_down[1]), (0, 255, 255), 2)

    cv2.putText(file, '{}'.format('dt'), dt_bbox_left_up, cv2.FONT_HERSHEY_COMPLEX, 0.6, [0, 0, 255])
    cv2.rectangle(file, (dt_bbox_left_up[0], dt_bbox_left_up[1]), (dt_bbox_righ_down[0], dt_bbox_righ_down[1]),
                  (0, 0, 255), 2)

    joints = batch_joints[0]
    for i in range(joints.shape[0]):
        joint_x=joints[i][0]
        joint_y=joints[i][1]
        cv2.circle(file,(int(joint_x), int(joint_y)), 2, [255, 0, 0], 2)
    cv2.imwrite(os.path.join(output_dir, file_name), file)

def save_image_with_skeleton(data_numpy,im_id,instance,output_dir,frames,next_id,ids):
    video_name = str(im_id)[1:7]
    if not os.path.exists(os.path.join(output_dir,'videos',video_name)):
        os.makedirs(os.path.join(output_dir,'videos',video_name))
    file_name = '{}.png'.format(str(im_id))
    cv2.putText(data_numpy, 'image_id' + str(im_id),
                (int(data_numpy.shape[0] * 0.5), 50), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 0])
    for i_person, person in enumerate(instance):
        keypoints = person['keypoints'].reshape((17,3))
        track_id = person['track_id']
        nose            =(int(keypoints[0][0]),int(keypoints[0][1]))
        head_bottom     =(int(keypoints[1][0]),int(keypoints[1][1]))
        head_top        =(int(keypoints[2][0]),int(keypoints[2][1]))
        left_ear        =(int(keypoints[3][0]),int(keypoints[3][1]))
        right_ear       =(int(keypoints[4][0]),int(keypoints[4][1]))
        left_shoulder   =(int(keypoints[5][0]),int(keypoints[5][1]))
        right_shoulder  =(int(keypoints[6][0]),int(keypoints[6][1]))
        left_elbow      =(int(keypoints[7][0]),int(keypoints[7][1]))
        right_elbow     =(int(keypoints[8][0]),int(keypoints[8][1]))
        left_wrist      =(int(keypoints[9][0]),int(keypoints[9][1]))
        right_wrist     =(int(keypoints[10][0]),int(keypoints[10][1]))
        left_hip        =(int(keypoints[11][0]),int(keypoints[11][1]))
        right_hip       =(int(keypoints[12][0]),int(keypoints[12][1]))
        left_knee       =(int(keypoints[13][0]),int(keypoints[13][1]))
        right_knee      =(int(keypoints[14][0]),int(keypoints[14][1]))
        left_ankle      =(int(keypoints[15][0]),int(keypoints[15][1]))
        right_ankle     =(int(keypoints[16][0]),int(keypoints[16][1]))
        #right part
        if right_ankle !=(0,0) and right_knee !=(0,0):
            cv2.line(data_numpy, right_ankle, right_knee, [255, 0, 0], 3)
        if right_knee != (0, 0) and right_hip != (0, 0):
            cv2.line(data_numpy, right_knee, right_hip, [255, 0, 0], 3)
        if right_wrist != (0, 0) and right_elbow != (0, 0):
            cv2.line(data_numpy, right_wrist, right_elbow, [255, 0, 0], 3)
        if right_elbow != (0, 0) and right_shoulder != (0, 0):
            cv2.line(data_numpy, right_elbow, right_shoulder, [255, 0, 0], 3)
        if right_hip != (0, 0) and right_shoulder != (0, 0):
            cv2.line(data_numpy, right_hip, right_shoulder, [255, 0, 0], 3)
            #left part
        if left_ankle != (0, 0) and left_knee != (0, 0):
            cv2.line(data_numpy, left_ankle, left_knee, [255, 0, 0], 3)
        if left_knee != (0, 0) and left_hip != (0, 0):
            cv2.line(data_numpy, left_knee, left_hip, [255, 0, 0], 3)
        if left_wrist != (0, 0) and left_elbow != (0, 0):
            cv2.line(data_numpy, left_wrist, left_elbow, [255, 0, 0], 3)
        if left_elbow != (0, 0) and left_shoulder != (0, 0):
            cv2.line(data_numpy, left_elbow, left_shoulder, [255, 0, 0], 3)
        if left_hip != (0, 0) and left_shoulder != (0, 0):
            cv2.line(data_numpy, left_hip, left_shoulder, [255, 0, 0], 3)
            #shoulder
        if left_shoulder != (0, 0) and head_bottom != (0, 0):
            cv2.line(data_numpy, left_shoulder, head_bottom, [255, 0, 0], 3)
        if right_shoulder != (0, 0) and head_bottom != (0, 0):
            cv2.line(data_numpy, right_shoulder, head_bottom, [255, 0, 0], 3)
            #bottom
        if left_hip != (0, 0) and right_hip != (0, 0):
            cv2.line(data_numpy, left_hip, right_hip, [255, 0, 0], 3)
            #head
        if head_bottom != (0, 0) and head_top != (0, 0):
            cv2.line(data_numpy, head_bottom, head_top, [0, 0, 255], 3)

        cv2.putText(data_numpy, 'id_{}'.format(track_id), head_top, cv2.FONT_HERSHEY_COMPLEX, 0.6, [0, 255, 255])
    cv2.imwrite(os.path.join(output_dir,'videos',video_name, file_name), data_numpy)
    before_resized = Image.open(os.path.join(output_dir,'videos',video_name, file_name))
    (x,y)=before_resized.size
    new_x = 400
    ratio = y/x
    new_y = new_x * ratio
    resized = before_resized.resize((int(new_x),int(new_y)))
    resized.save(os.path.join(output_dir, 'videos', video_name, file_name))

    if next_id in ids :
        frames.append(imageio.imread(os.path.join(output_dir,'videos',video_name, file_name)))
        return frames
    else:
        gif_name = 'video_{}.gif'.format(video_name)
        logger.info('saving video-->{}.gif'.format(video_name))
        imageio.mimsave(os.path.join(output_dir,'videos',video_name,gif_name),frames,'GIF',duration=0.05)
        return []

def bbox_nms(dets, thresh):
    """
        greedily select boxes with high confidence and overlap with current maximum <= thresh
        rule out overlap >= thresh
        :param dets: array[[x, y, w, h, score]...]
        :param thresh: retain overlap < thresh
        :return: indexes to keep
        """
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    w = dets[:, 2]
    h = dets[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    scores = dets[:, -1]
    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_i = np.maximum(0.0, xx2 - xx1 + 1)
        h_i = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w_i * h_i
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
