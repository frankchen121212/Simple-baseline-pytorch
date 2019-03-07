from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



import _init_paths
import cv2
import json
import os
import os.path as osp
import scipy.optimize
import scipy.spatial
import numpy as np
import utils.keypoints as kps_utils
import utils.image as img_utils
from core.config import config as cfg



MAX_TRACK_IDS = 999
FIRST_TRACK_ID = 0



def _load_json_file(json_fpath):
    with open(json_fpath, 'rb') as fin:
        return json.load(fin)

def _write_json_file(json_fpath, dict_file):
    with open(json_fpath, 'w') as fin:
        json.dump(dict_file, fin)

# save the json file
def _write_detwithTracks_file(pred, trackid, test_output_dir):
    # use trackid for pred and save the file into the output_dir
    temp_track_id = []
    cnt = 0
    for i in range(len(trackid)):
        for j in range(len(trackid[i])):
            temp_track_id.append(trackid[i][j])
    print (temp_track_id)
    res_annotwithTracks = {'images': [], 'annotations': [], 'categories': []}
    pred_annot = pred['annotations']
    for i in range(len(pred_annot)):
        pred_annot[i]['track_id'] = temp_track_id[cnt]
        cnt += 1
    res_annotwithTracks['images'] = pred['images']
    res_annotwithTracks['annotations'] = pred_annot
    res_annotwithTracks['categories'] = pred['categories']
    json_name = pred['images'][0]['file_name'].split('/')[2] + '.json'
    json_output_path = osp.join(test_output_dir, json_name)
    print (json_output_path)
    _write_json_file(json_output_path, res_annotwithTracks)
    # with open( json_output_path, 'wb') as fin:
    #     json.dump(pred_annot, fin)



def _get_annot_frames_id(video_json_data):
    res_frames_id = []
    annot = video_json_data['annotations']
    for i in range(len(annot)):
        res_frames_id.append(annot[i]['image_id'])
    res_frames_id = set(res_frames_id)
    return list(res_frames_id)


def _get_gt_boxes_poses(video_json_data, img_id):
    res_bbox = []
    res_pose = []
    flag = 0
    annot = video_json_data['annotations']
    for i in range(len(annot)):
        if img_id == annot[i]['image_id']:
            flag = 1
            res_bbox.append(annot[i]['bbox'])
            res_pose.append(annot[i]['keypoints'])
    return res_bbox, res_pose, flag


def _get_boxes():
    pass

def _get_poses():
    pass

def _get_video_json_data(video_json_data, annot_frames, index):
    # res1 = video_json_data['images'][index]
    # res2 = video_json_data['images'][index]['id']
    for i in range(len(video_json_data['images'])):
        if video_json_data['images'][i]['id'] == annot_frames[index]:
            res1 = video_json_data['images'][i]
            break
    res2 = annot_frames[index]
    return res1, res2


def _compute_pairwise_iou():
    pass

def _compute_deep_features(imname, boxes):
    import utils.cnn_features as cnn_utils
    # imname is 'file_name' in 'images'
    print (imname)
    I = cv2.imread(imname) # the path of image
    if I is None:
        raise ValueError('Image not found {}'.format(imname))
    all_feats = []
    print ('Now is the {} info'.format(imname))
    print ('bbox info:', boxes)
    for box in boxes:
        patch = I[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2]), :]
        all_feats.append(cnn_utils.extract_features(
            patch, layers = (cfg.TRACKING.CNN_MATCHING_LAYER,)))
    return np.stack(all_feats) if len(all_feats) > 0 else np.zeros((0, ))



def _compute_pairwise_deep_cosine_dist(a_imname, a, b_imname, b):
    f1 = _compute_deep_features(a_imname, a)
    f2 = _compute_deep_features(b_imname, b)
    if f1.size * f2.size == 0:
        return np.zeros((f1.shape[0], f2.shape[0]))
    return scipy.spatial.distance.cdist(
        f1.reshape((f1.shape[0], -1)), f2.reshape((f2.shape[0], -1)),
        'cosine')



def _compute_pairwise_kpt_distance_pck(a, b, kpt_names):
    res = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            res[i, j] = kps_utils.pck_distance(a[i], b[j], kpt_names)
    return res


def _compute_pairwise_kps_distance_oks(a, b, kpt_names):
    pass



def _compute_distance_matrix(
    prev_json_data, prev_boxes, prev_poses,
    cur_json_data, cur_boxes, cur_poses,
    cost_types, cost_weights):
    assert (len(cost_weights) == len(cost_types))
    all_Cs = []
    for cost_type, cost_weight in zip(cost_types, cost_weights):
        if cost_weight == 0:
            continue
        if cost_type == 'bbox-overlap':
            all_Cs.append((1 - _compute_pairwise_iou(pre_boxes, cur_boxes)))
        elif cost_type == 'cnn-cosdist':
            all_Cs.append(_compute_pairwise_deep_cosine_dist(
                img_utils.get_image_path(prev_json_data), prev_boxes,
                img_utils.get_image_path(cur_json_data), cur_boxes))
        elif cost_type == 'pose-pck':
            all_Cs.append((_compute_pairwise_kpt_distance_pck(
                prev_poses, cur_poses)))
        elif cost_type == 'pose-oks':
            all_Cs.append(_compute_pairwise_kpt_distance_oks(
                prev_poses, cur_poses, kps_names))
            pass
        else:
            raise NotImplementedError('Unknown cost type {}'.format(cost_type))
        all_Cs[-1] *= cost_weight
    return np.sum(np.stack(all_Cs, axis = 0), axis = 0)



def _compute_matches(prev_frame_data, cur_frame_data, prev_boxes, cur_boxes,
                     prev_poses, cur_poses,
                     cost_types, cost_weights,
                     bipart_match_algo, C = None):
    if C is None:
        nboxes = cur_boxes.shape[0]
        matches = np.ones((nboxes,), dtype = np.int32)
        C = _compute_distance_matrix(
            prev_frame_data, prev_boxes, prev_poses,
            cur_frame_data, cur_boxes, cur_poses,
            cost_types = cost_types,
            cost_weights = cost_weights)
    else:
        matches = np.ones((C.shape[1],), dtype = np.int32)

    if bipart_match_algo == 'hungarian':
        prev_inds, next_inds = scipy.optimize.linear_sum_assignment(C)
    elif bipart_match_algo == 'greedy':
        prev_inds, next_inds = bipartite_matching_greedy(C)
    else:
        raise NotImplementedError('Unknown matching algo: {}'.format(bipart_match_algo))
    assert (len(prev_inds) == len(next_inds))
    for i in range(len(prev_inds)):
        matches[next_inds[i]] = prev_inds[i]
    return matches


def _compute_tracks_video(video_json_data, dets):
    #video_json_data = _load_json_file(video_json_data)
    nframes = len(video_json_data['images']) # the number of frames in one video

    video_tracks = [] # track_id for all of frames. [[], [], [], [], []]
    next_track_id = FIRST_TRACK_ID
    annot_frames = _get_annot_frames_id(video_json_data)
    annot_frames.sort()
    print ('The annoted info in the video: ', annot_frames)
    annot_frames_id = 0
    n_annoted_frames = len(annot_frames)
    # for frame_id in range(nframes):
    for annot_frames_id in range(n_annoted_frames):
        frame_tracks = [] # frame_tracks []

        # det_id is the 'id' in 'images'
        # So I should get boxes and poses from the image_id
        #frame_data, det_id = video_json_data[frame_id], convert to a function
        frame_data, img_id = _get_video_json_data(video_json_data, annot_frames, annot_frames_id)
        # cur_boxes = _get_boxes(video_json_data, img_id)
        # cur_poses = _get_poses(video_json_data, img_id)
        cur_boxes, cur_poses, flag = _get_gt_boxes_poses(video_json_data, img_id)
        cur_boxes = np.array(cur_boxes)
        cur_poses = np.array(cur_poses)
        if flag == 0: # flag == 0 means no gt info for boxes and poses
            continue


        if annot_frames_id == 0:
            matches = -np.ones((cur_boxes.shape[0], ))
        else:
            cur_frame_data = frame_data
            # prev_boxes = _get_boxes(video_json_data, img_id)
            # prev_poses = _get_poses(video_json_data, img_id)
            prev_frame_data, prev_img_id = _get_video_json_data(video_json_data, annot_frames, annot_frames_id - 1)
            print('annot_frames_id', annot_frames_id)
            prev_boxes, prev_poses, _ = _get_gt_boxes_poses(video_json_data, prev_img_id)
            # prev_frame_data = video_json_data['images'][frame_id - 1][0] # ???
            matches = _compute_matches(
                prev_frame_data, cur_frame_data,
                prev_boxes, cur_boxes, prev_poses, cur_poses,
                cost_types = cfg.TRACKING.DISTANCE_METRICS,
                cost_weights = cfg.TRACKING.DISTANCE_METRICS_WTS,
                bipart_match_algo = cfg.TRACKING.BIPARTITE_MATCHING_ALGO)

        print ('Now is {} for matching'.format(annot_frames_id))
        print ('The matching ID is', matches)
        prev_tracks = video_tracks[annot_frames_id - 1] if annot_frames_id > 0 else None


        for m in matches:
            if m == -1:
                frame_tracks.append(next_track_id)
                next_track_id += 1
                if next_track_id >= MAX_TRACK_IDS:
                    next_track_id %= MAX_TRACK_IDS
            else:
                frame_tracks.append(prev_tracks[m])
        video_tracks.append(frame_tracks)

    return video_tracks


def compute_matches_tracks(json_data, dets):
    # num_imgs = len(json_data) # the number of pic
    # all_tracks = [[]] * num_imgs
    vid_id = json_data['images'][0]['vid_id']
    # print (type(vid_id))
    # vid_id = int(vid_id)

    print ('Computing tracks for {} video.'.format(vid_id)) # for this video

    tracks = _compute_tracks_video(json_data, dets)
    return tracks


def run_posetrack_tracking(test_output_dir, gt_json_data, pred_json_data):
    """
    video: 000522_mpii_test.json
    metric: cnn-cosdist
    test_output_dir: each entry in annotations should have a track_id
    only for one-video
    gt_json_data is gt_info for one video
    pred_json_data is pred_info for one video
    test_output_dir is the information with bbox
    """

    # det_file has
    # print (cfg)
    if len(cfg.TRACKING.DETECTIONS_FILE ):
        det_file = cfg.TRACKING.DETECTIONS_FILE
    else:
        det_file = None

    # the file that has kps, bbox, track_id
    #out_detwitTracks_file = osp.join(test_output_dir, 'detect_withTracks.json')

    conf = cfg.TRACKING.CONF_FILTER_INITIAL_DETS
    print ('Pruning detections with less than {} confidence'.format(conf))
    # TODO
    gt_json_data = _load_json_file(gt_json_data)
    det_withTracks = compute_matches_tracks(gt_json_data, det_file)
    print (det_withTracks) # [[],[],[],[],[]]
    # TODO: how to allocate the track_id for each annotations.(solved)
    pred_json_data = _load_json_file(pred_json_data)

    _write_detwithTracks_file(pred_json_data, det_withTracks, test_output_dir) # save the track_id for this video



def main():
    test_output_dir = '/home/users/yang.bai/project/analysis_result_tf_pytorch/pytorch_detwithTracks'
    gt_json_data = '/mnt/data-1/data/yang.bai/PoseTrack2018/images/posetrack_data/annotations_original/val/000522_mpii_test.json'
    pred_json_data = '/home/users/yang.bai/project/analysis_result_tf_pytorch/pytorch_track_id/000522_mpii_test.json'
    run_posetrack_tracking(test_output_dir, gt_json_data, pred_json_data)

if __name__ == '__main__':
    main()