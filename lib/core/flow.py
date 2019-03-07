import mmcv
import numpy as np
import json as js
import cv2
import os


def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)


def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = 192/256  # w/h
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

def get_cs_from_flow(flow,pre_frame_pred, h, w,score,flow_thre):
    # example
    # image_id--10003420000
    # 000342_mpii_test/000000.flo

    # image_id = str(image_id)
    # root_flow = '/share1/home/chunyang/files/human-pose-estimation.pytorch/data/posetrack/Flowimgs/flow/val/inference'
    # folder_name = image_id[1:7]+'_mpii_test'
    # file_name = '00'+image_id[-4:]+'.flo'
    # flow_path = os.path.join(root_flow,folder_name,file_name)
    # flow = mmcv.flowread(flow_path)  # [h,w,2]
    if score<flow_thre :
        is_ignore = True
        final_center = np.zeros([1, 2])
        final_scale = np.zeros([1, 2])
        box_sc_flow = None
        scale_flow = None
        kps = pre_frame_pred[0]
        flow_kps = kps
        return final_center, final_scale, box_sc_flow, is_ignore, scale_flow, flow_kps


    h_boarder = h - 1
    w_boarder = w - 1
    # print('flow_path--{}'.format(folder_name +'/' + file_name))
    # print('h--{}, w--{}'.format(h_boarder,w_boarder))
    kps = pre_frame_pred[0]
    flow_kps = kps  # num_keypoints x 2 array

    lc_x, lc_y = [], []
    for x in kps[:, 0]:
        lc_x.append(int(x))
    for y in kps[:, 1]:
        lc_y.append(int(y))

    new_x = []
    new_y = []

    # print('dx--{}'.format(dx))
    # print('dy--{}'.format(dy))
    # print(flow.shape)
    #TODO

    for i, (x, y) in enumerate(zip(lc_x, lc_y)):
        if x <= 0 or y <= 0 or x > w_boarder or y > h_boarder:
            flow_kps[i] = np.array([x, y])
            continue
        else:
            new_x.append(x+flow[y][x][0])
            new_y.append(y+flow[y][x][1])


    is_ignore = False

    if len(new_x) * len(new_y) == 0:
        is_ignore = True
        scale_flow = None
        final_center = np.zeros([1, 2])
        final_scale = np.zeros([1, 2])
        box_sc_flow = None

    elif max(new_x) < 0 or max(new_y) < 0 or min(new_x) > w_boarder or min(new_y) > h_boarder:
        is_ignore = True
        scale_flow = None
        final_center = np.zeros([1, 2])
        final_scale = np.zeros([1, 2])
        box_sc_flow = None

    else:

        x = max(min(min(new_x), w_boarder), 0)

        y = max(min(min(new_y), h_boarder), 0)

        w = min((max(new_x) - x) / 0.85, w_boarder - x)

        h = min((max(new_y) - y) / 0.85, h_boarder - y)

        scale_flow = w * h

        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))

        bbox = [x, y, w, h]

        box_sc_flow = [x, y, w, h, score]

        center, scale = _box2cs(bbox)

        final_center = np.zeros([1, 2])

        final_scale = np.zeros([1, 2])

        final_center[0] = center

        final_scale[0] = scale

    return final_center, final_scale, box_sc_flow, is_ignore, scale_flow, flow_kps

    # img = mmcv.imread('/share1/home/chunyang/files/flowtrack/img_pair/frame_000089.png')
    # im = cv2.rectangle(img,(x,y),(x+w,y+h),(55,255,155),5)

    # flow = mmcv.flowread('/share1/home/chunyang/files/flownet2-pytorch/img0.flo')
    # kps = js.load(open('/share1/home/chunyang/files/flowtrack/alphapose-results.json','r'))
    # location = [i for i in kps[0]['keypoints'] if not((kps[0]['keypoints'].index(i)+1) % 3 == 0)]
    # loc = [int(i) for i in location]
    # loc_y = [int(i) for i in location if(location.index(i)%2==1)]
    # loc_x = [int(i) for i in location if(location.index(i)%2==0)]
    #
    # new_x = []
    # new_y = []
    # for x,y in zip(loc_x,loc_y):
    #          new_x.append(x+flow[x][y][0])
    #          new_y.append(y+flow[x][y][1])
    # x = min(new_x)
    # y = min(new_y)
    # w = (max(new_x)-x)/0.85
    # h = (max(new_y) -y)/0.85
    #
    # x,y,w,h = int(round(x)),int(round(y)),int(round(w)),int(round(h))
    # bbox = [x,y,w,h]
    #
    # img = mmcv.imread('/share1/home/chunyang/files/flowtrack/img_pair/frame_000089.png')
    # im = cv2.rectangle(img,(x,y),(x+w,y+h),(55,255,155),5)





