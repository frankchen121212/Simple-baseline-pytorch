from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from collections import OrderedDict
import logging
import os
import sys
import json
from collections import defaultdict
import time
import pickle
import itertools
import numpy as np
from cocoapi.PythonAPI.pycocotools import mask as maskUtils

#from cocoapi.PythonAPI.pycocotools.cocoeval_posetrack import COCOeval
from core.cocoeval_posetrack import COCOeval
from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms

###  poseval API  ###

sys.path.append('poseval/py-motmetrics')

from poseval.py.evaluateAP import evaluateAP
from poseval.py import eval_helpers




# "keypoints": [
#                 "nose",             0
#                 "head_bottom",      1
#                 "head_top",         2
#                 "left_ear",         3
#                 "right_ear",        4
#                 "left_shoulder",    5
#                 "right_shoulder",   6
#                 "left_elbow",       7
#                 "right_elbow",      8
#                 "left_wrist",       9
#                 "right_wrist",      10
#                 "left_hip",         11
#                 "right_hip",        12
#                 "left_knee",        13
#                 "right_knee",       14
#                 "left_ankle",       15
#                 "right_ankle"       16
#             ],
logger = logging.getLogger(__name__)


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class PoseTrackDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super(PoseTrackDataset, self).__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.coco=COCO(self._get_ann_file())

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]


        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        #print(self.image_set_index)

        self.num_images = len(self.image_set_index)
        print('=> num_images: {}'.format(self.num_images))

        self.num_joints = 17
        self.flip_pairs = [[3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None

        self.db = self._get_db()

        '''if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)'''

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file(self):
        """ self.root / posetrack_data/annotations / posetrack_val.json """
        prefix = 'posetrack_instance' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.root, 'posetrack_data', 'annotations',
                            prefix + '_' + self.image_set + '_0003420000.json')


    def _load_image_set_index(self):
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        pass
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db
    def select_data(self):
        pass
    def _load_coco_keypoint_annotations(self):
        gt_db = []
        for index in self.image_set_index:
            '''
            10_15860_0140, 10158600141, 
            10158600142, 10158600143, 10158600144, 
            10158600145, 10158600146, 10158600147, 10158600148
            index
            '''
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
            #index--every frame id
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        '''im_ann----{'has_no_densepose': True, 
        'is_labeled': True, 
        'file_name': 'images/val/004707_mpii_test/000000.jpg', 
        'nframes': 126, 
        'frame_id': 10_047070000, 
        'vid_id': '004707', 
        'ignore_regions_y':  65, 
        'id': 10_047070000, 
        'width': 1920, 
        'height': 1080}
        '''

        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index)
        objs = self.coco.loadAnns(annIds)

        '''annIds=1004707000000'''

        '''
        objs=
        [{'bbox_head': [654, 157, 244, 330], 
        'keypoints': [0], 
        'track_id': 0, 
        'image_id': 10047070000,
         'bbox': [312.25, 75.60604548999993, 734.5, 1070.40186162], 
         'scores': [], 
         'category_id': 1, 
         'id': 1004707000000, 
         'segmentation': [],
          'num_keypoints': 17,
         'area': 786210.1673598901}]
        '''


        valid_objs = []
        for obj in objs:
            if('bbox' not in obj):
                continue

            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            id_global = obj['id']
            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])

            rec.append({
                'image': self.image_path_from_index(index),
                #'image_id':  id_global,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'image_id': index,
                'imgnum': 0,
            })

        return rec

    #only when use detector
    def _load_coco_person_detection_results(self):
        pass
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'image_id': det_res['image_id'],
                'box': box,  # to do the bounding box NMS
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    def image_path_from_index(self, index):

        file_name = '%06d.jpg' % int(str(index)[-4:])  # 10003420099 -> 99 -> 000099.jpg
        if 'train' in self.image_set:
            bonn = ['000001', '000002', '000003', '000010', '000015', '000017', '000022', '000023', '000026', '000027', \
                    '000028', '000029', '000036', '000048']
            if str(index)[1:-4] in bonn:
                suffix = '_bonn_train'
            else:
                suffix = '_mpii_train'
        elif 'val' in self.image_set:
            suffix = '_mpii_test'
        else:
            pass  # TODO: deal with the test dataset

        folder = str(index)[1:-4] + suffix

        image_path = os.path.join(
            self.root, 'images', self.image_set, folder, file_name)

        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)

        return image_path

    def evaluate(self, cfg, preds, output_dir, all_boxes, image_path,
                 file_names, *args):
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.image_set)

        # person x (keypoints)
        _kpts = []


        for idx, kpt in enumerate(preds):

            #image_id=file_names[idx]
            #print((img_path[idx]))
            '''
            example=>(img_path[idx])data/posetrack/images/valid/000522_mpii_test/000080.jpg
            '''

            image_id = image_path[idx][-8:-4]
            folder_name = image_path[idx].split('/')[-2][0:6]

            if(self.image_set == 'valid' or 'val'):
                prefix = '1'
            else:
                prefix = '0'

            image_id = int(prefix+folder_name+image_id)
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                #'image': (img_path[idx][-10:-4]),
                #'image': img_path[idx],
                #'image_id'  :image_path[idx]
                'image_id': image_id
                #int(img_path[idx][-10:-4])#
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image_id']].append(kpt)

        #logger.info('=>kpts--{}'.format(kpts))
        # rescoring and oks nms
        logger.info('=>len--kpts.keys()--{}'.format(len(kpts.keys())))


        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []


        for img in kpts.keys():

            img_kpts = kpts[img]
            for n_p in img_kpts:
                #logger.info('n_p--{}'.format(n_p))

                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                           oks_thre)
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)

            else:

                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])



        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)

        pred_folder = os.path.join(self.root,'posetrack_example_results', 'test_pred/')
        gt_folder = os.path.join(self.root, 'posetrack_data', 'anns/')

        if 'test' not in self.image_set:
         info_str = self._do_python_keypoint_eval(
             res_file, res_folder)

         if "val" in self.image_set:
             self._do_posetrack_keypoint_eval(gt_folder, pred_folder)
         name_value = OrderedDict(info_str)
         return name_value, name_value['AP']
        else:
         return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):

        data_pack = [{'cat_id': 1,
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'keypoints',
                      'keypoints': keypoints
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)
        ### DIY results process###


        if "val" or 'valid' in self.image_set:
            pred = json.load(open(res_file))
            for root, _, files in os.walk(os.path.join(self.root, 'posetrack_data', 'anns')):  #TODO
                for f in files:
                    test_pred = {}
                    test_pred['annotations'], test_pred['images'], test_pred['categories'] = [], [], []
                    temp = {}
                    test_pred['categories'].append(temp)
                    test_pred['categories'][0]["supercategory"] = "person"
                    test_pred['categories'][0]["name"] = "person"
                    test_pred['categories'][0]["id"] = 1
                    test_pred['categories'][0]['keypoints'] = [u'nose', u'head_bottom', u'head_top', u'left_shoulder',
                                                               u'right_shoulder', u'left_elbow', u'right_elbow',
                                                               u'left_wrist', u'right_wrist', u'left_hip', u'right_hip',
                                                               u'left_knee', u'right_knee', u'left_ankle', u'right_ankle']
                    assert len(test_pred["categories"]) == 1

                    ids = []

                    for p in pred:
                        res = str(p['image_id'])[1:7] + '_mpii_test.json'

                        if res == f:
                            ann = self._process(p)
                            test_pred['annotations'].append(ann)
                            ids.append(p['image_id'])

                    ids = list(set(ids))

                    gt = json.load(open(os.path.join(root, f)))
                    gt_ids = []
                    for item in gt['images']:
                        gt_ids.append(item['frame_id'])

                    ### if detector cnnot detect anything in an image, add the image id manually ###
                    for i in gt_ids:
                        if i not in ids:
                            ids.append(i)
                            print("image %s not in prediction files!" % {i})


                    for i in sorted(ids):
                        img = {}
                        img['id'] = int(i)
                        img["file_name"] = self.image_path_from_index(str(int(i)))
                        test_pred['images'].append(img)

                    if len(test_pred['annotations']) == 0:
                        print(f)
                        continue

                    else:
                        with open(os.path.join(self.root,'posetrack_example_results', 'test_pred/' + f), 'w') as result:
                            json.dump(test_pred, result, sort_keys= True, indent=4)

    def _process(self,item):
        ann = {}
        del item['keypoints'][9:15]
        ann["keypoints"] = []
        ann["scores"] = item['keypoints'][2::3]
        # ann["scores"] = [i if i > 0.4 else 0 for i in ann["scores"]]
        ann["scores"] = [i for i in ann["scores"]]
        ann["keypoints"] = item['keypoints']
        if len(ann['keypoints']) != 45:
            print(len(ann['keypoints']))
        # for k, kp in enumerate(ann["keypoints"]):
        #     if (k + 1) % 3 == 0:
        #         ann["keypoints"][k] = int(round(kp + 0.1))  # > 0.4 --> 1
        ann["track_id"] = np.random.randint(1, 999)
        ann["image_id"] = item['image_id']

        return ann

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float)

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [{'image_id': int(img_kpts[k]['image_id']),
                       'category_id': cat_id,
                       'keypoints': list(key_points[k]),
                       'score': img_kpts[k]['score'],
                       'center': list(img_kpts[k]['center']),
                       'scale': list(img_kpts[k]['scale'])

                       } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results


    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)

        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')

        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        #print('info_str---{}'.format(info_str))
        eval_file = os.path.join(
            res_folder, 'keypoints_%s_results.pkl' % self.image_set)

        with open(eval_file, 'wb') as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        logger.info('=> coco eval results saved to %s' % eval_file)

        return info_str

    def _do_posetrack_keypoint_eval(self, gt_folder, pred_folder):
        argv = ['', gt_folder, pred_folder]
        outputDir = 'lib/poseval/py/out'

        print("Loading data")
        gtFramesAll, prFramesAll = eval_helpers.load_data_dir(argv)
        print("# gt frames  :", len(gtFramesAll))
        print("# pred frames:", len(prFramesAll))
        # evaluate per-frame multi-person pose estimation (AP)

        # compute AP
        print("Evaluation of per-frame multi-person pose estimation")
        apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll, outputDir, True, False)

        # print AP
        print("Average Precision (AP) metric:")
        eval_helpers.printTable(apAll)







##################################################################################
#
#self-write coco-API
#
##################################################################################

class COCO:

    def __init__(self, annotation_file=None):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        if not annotation_file == None:
            print('Dataloader*****loading {} for training into memory...'.format(annotation_file))
            tic = time.time()

            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Preparing evalutate results...')
        tic = time.time()
        if type(resFile) == str :
            #print('result file is string format!!!')
            anns = json.load(open(resFile))
        else:
            anns = resFile


        assert type(anns) == list, 'results in not an array of objects'
        #annsImgIds = [ann['id'] for ann in anns]
        #assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        #       'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1


                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()

        return res

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        # imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        # if len(imgIds) == 0:
        #     anns = self.dataset['annotations']
        # else:
        #     if not len(imgIds) == 0:
        #         lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
        #         anns = list(itertools.chain.from_iterable(lists))
        #     else:
        #         anns = self.dataset['annotations']
        #
        # anns = list(itertools.chain.from_iterable(lists))
        #
        # # if len(imgIds) != 0:
        # #     print('imgIds[0]--{}'.format(imgIds[0]))
        #
        # ids = [ann['id'] for ann in anns]
        """
                Get ann ids that satisfy given filter conditions. default skips that filter
                :param imgIds  (int array)     : get anns for given imgs
                       catIds  (int array)     : get anns for given cats
                       areaRng (float array)   : get anns for given area range (e.g. [0 inf])
                       iscrowd (boolean)       : get anns for given crowd label (False or True)
                :return: ids (int array)       : integer array of ann ids
                """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids


    def getImgIds(self, imgIds=[], catIds=[]):
        ids = self.imgs.keys()
        return list(ids)



