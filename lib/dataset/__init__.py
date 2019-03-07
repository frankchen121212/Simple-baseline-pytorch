# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
# from .posetrack import PoseTrackDataset as posetrack
# from .posetrack_chunyang import PoseTrackDataset as posetrack  # for pose estimation
from .posetrack_opticalfolw import PoseTrackDataset as posetrack  # for flow_base pose tracking