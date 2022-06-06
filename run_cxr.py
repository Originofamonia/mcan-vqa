# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# modify this to our CXR VQA dataset
# --------------------------------------------------------

import os
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors

from cfgs.base_cfgs import Cfgs
from core.exec import ExecuteCXR
import argparse, yaml


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--run', dest='run_mode',
                      choices=['train', 'val', 'test', 'visualize'],
                      type=str, default='train')

    parser.add_argument('--model', dest='model',
                      choices=['small', 'large'],
                      default='small', type=str)

    parser.add_argument('--split', dest='train_split',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "eg.'train', 'train+val+vg'"
                           "set 'train' can trigger the "
                           "eval after every epoch",
                      type=str)

    parser.add_argument('--eval_every_epoch', default=False,
                      help='set True to evaluate the '
                           'val split when an epoch finished'
                           "(only work when train with "
                           "'train' split)",
                      type=bool)

    parser.add_argument('--test_save_pred',
                      help='set True to save the '
                           'prediction vectors'
                           '(only work in testing)',
                      type=bool)

    parser.add_argument('--batch_size', default=256,  # was 256
                      help='batch size during training',
                      type=int)

    parser.add_argument('--max_epoch', default=13,
                      help='max training epoch',
                      type=int)

    parser.add_argument('--preload',
                      help='pre-load the features into memory'
                           'to increase the I/O speed',
                      type=bool)

    parser.add_argument('--gpu', default='0,1',
                      help="gpu select, eg.'0, 1, 2'",
                      type=str)

    parser.add_argument('--seed', default=444,
                      help='fix random seed',
                      type=int)

    parser.add_argument('--version',
                      help='version control',
                      type=str)

    parser.add_argument('--resume',
                      help='resume training', default=False,
                      type=bool)

    parser.add_argument('--ckpt_version',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--ckpt_epoch',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--ckpt_path',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'ckpt_version and ckpt_epoch '
                           'instead',
                      type=str)

    parser.add_argument('--grad_accu_steps',
                      help='reduce gpu memory usage',
                      type=int)

    parser.add_argument('--num_workers',
                      help='multithreaded loading',
                      type=int)

    parser.add_argument('--pin_mem',
                      help='use pin memory',
                      type=bool)

    parser.add_argument('--verbose',
                      help='verbose print',
                      type=bool)

    parser.add_argument('--dataset_path',
                      help='vqav2 dataset root path',
                      type=str)

    parser.add_argument('--feature_path',
                      help='bottom up features root path',
                      type=str)

    args = parser.parse_args()
    return args


def main():
    opt = Cfgs()

    args = parse_args()
    args_dict = opt.parse_to_dict(args)

    cfg_file = "cfgs/{}_model.yml".format(args.model)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    args_dict = {**yaml_dict, **args_dict}
    opt.add_args(args_dict)
    opt.proc()

    print('Hyper Parameters:')
    print(opt)

    opt.check_cxr_path()

    execution = ExecuteCXR(opt)
    execution.run(opt.run_mode)


if __name__ == '__main__':
     main()