# Simple Baselines for Human Pose Estimation and Tracking

## Introduction
This is a recurrent pytorch implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). 
State-of-the-art results are achieved on challenging benchmarks. On posetrack keypoints valid dataset, our best **single model** achieves **73.5 of mAP**and**61.1 of MOTA**. You can reproduce our results using this repo.    </br>

## Pose-Track2018 Tracking demo
![image](https://github.com/frankchen121212/Simpe-baseline/blob/master/example/video_001007.gif)

## Main Results
### Results on Posetrack val
1. Task1: Multi-Person Pose Estimation (mAP)

| Method | Head mAP | Shoulder mAP | Elbow mAP | Wrist mAP | Hip mAP | Knee mAP | Ankle mAP | Total mAP |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| **ours** | **78.3** | **79.7** | **74.9** | **66.6** | **73.5** | **73.2** | **66.2** | **73.5** |


 Note:
- Flip test is used
2. Task2: Pose Tracking (MOTA)

| Method | Head MOTA | Shoulder MOTA | Elbow MOTA | Wrist MOTA | Hip MOTA | Knee MOTA | Ankle MOTA | Total MOTA | Total MOTP | Prec Total | Rec Total |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| **ours** | **70.6** | **71.6** | **57.7** | **50.4** | **62.7** | **60.5** | **49.5** | **61.1** | **67.3**| **83.6**| **77.6** |

 Note:
- Flip test is used
- Person detector has person AP of 56.4 on COCO val2017 dataset 

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/)
2. Disable cudnn for batch_norm
   ```
   # PYTORCH=/path/to/pytorch
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```
   Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick a path where you'd like to have pytorch installed  and then set an environment variable (PYTORCH in this case) accordingly.
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}
2. Install dependencies.
   ```
   pip install -r requirements.txt
   ```
3. Make libs
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
3. Download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo). 
4. Download our posetrack pretrained models from [OneDrive](https://1drv.ms/f/s!Ap3dsRxBx6KvhnZYD1T3UZHvy2xQ). Please download them under ${POSE_ROOT}/models/pytorch, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
             -- pose_coco
                |-- pose_resnet_50_256x192.pth.tar
            `-- pose_posetrack
                |-- finetune_final_10loss_.tar

   ```

5. Init output(training model output directory) and log(tensorboard log directory) directory.

   ```
   mkdir output 
   mkdir log
   ```

   and your directory tree should look like this

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```
   
### Data preparation
**For Posetrack data**, please download from [Posetrack2018 download](https://posetrack.net/users/download.php).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- posetrack
    `-- |-- annotations
        |-- Flowimgs
        |-- posetrack_data
        |   |-- annotations
            |-- train
                |-- 000001_bonn_train.json
                |-- 000002_bonn_train.json
                |-- 000003_bonn_train.json
                |-- ... 
            |-- val
                |-- 000342_mpii_test.json
                |-- 000522_mpii_test.json
                |-- 000583_mpii_test.json
                |-- ... 
            |--detections_full_val_results_fpn_dcn.json
            |--posetrack_instance_train.json
            |--posetrack_instance_val.json
        `-- images
            |-- train
                |-- 000001_bonn_train
                |-- 000002_bonn_train
                |-- 000003_bonn_train
                |-- ... 
            |-- val
                |-- 000342_mpii_test
                |-- 000522_mpii_test
                |-- 000583_mpii_test
                |-- ... 
```

### Valid on Posetrack2018 using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```

### Fintune on Posetrack2018 

```
CUDA_VISIBLE_DEVICES=0 python pose_estimation/valid_with_opticalfolw.py \
    --cfg experiments/posetrack/resnet50/valid_with_opticalflow_tracking.yaml \
    --flip-test \
    --model-file    models/pytorch/pose_posetrack/finetune_final_10loss_.tar
```

