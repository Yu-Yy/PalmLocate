# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
# from easydict import EasyDict as edict
from yacs.config import CfgNode as CN

config = CN()
config.OUTPUT_DIR = ''
config.GPUS = '0,1,2'
config.NUM_WORKERS = 8
config.PRINT_FREQ = 100
config.DEBUG_FREQ = 200

config.MODEL = CN()
config.MODEL.PRETRAINED = ''
config.MODEL.ALIGN_CORNERS = True
config.MODEL.EXTRA = CN(new_allowed=True)

config.DATASET = CN()
config.DATASET.NUM_CLASSES = 19
config.DATASET.ROOT = ''

config.TRAIN = CN()
config.TRAIN.BATCH_SIZE_PER_GPU = 8
config.TRAIN.SHUFFLE = True
config.TRAIN.EPOCHS = 100
config.TRAIN.LR = 0.001
config.TRAIN.PATCH_SIZE = 128
config.TRAIN.RESUME = True

config.TEST = CN()
config.TEST.BATCH_SIZE_PER_GPU = 1

def update_config(cfg, args):
    cfg.defrost() 
    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
