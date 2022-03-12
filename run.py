# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from cfgs.base_cfgs import Cfgs
from core.exec import Execution
import argparse, yaml


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--run', dest='run_mode',
                      choices=['train', 'val', 'test', 'visualize'],
                      type=str, default='val')

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

    parser.add_argument('--batch_size', default=1,  # was 256
                      help='batch size during training',
                      type=int)

    parser.add_argument('--max_epoch',
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
                      help='resume training',
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

    opt.check_path()

    execution = Execution(opt)
    execution.run(opt.run_mode)


def text_layout():
     fig = plt.figure()

     left, width = .25, .5
     bottom, height = .25, .5
     right = left + width
     top = bottom + height
     
     # Draw a rectangle in figure coordinates ((0, 0) is bottom left and (1, 1) is
     # upper right).
     p = Rectangle((left, bottom), width, height, fill=False)
     fig.add_artist(p)
     # Figure.text (aka. plt.figtext) behaves like Axes.text (aka. plt.text), with
     # the sole exception that the coordinates are relative to the figure ((0, 0) is
     # bottom left and (1, 1) is upper right).
     fig.text(left, bottom, 'left top',
          horizontalalignment='left', verticalalignment='top')
     fig.text(left, bottom, 'left bottom',
          horizontalalignment='left', verticalalignment='bottom')
     fig.text(right, top, 'right bottom',
          horizontalalignment='right', verticalalignment='bottom')
     fig.text(right, top, 'right top',
          horizontalalignment='right', verticalalignment='top')
     fig.text(right, bottom, 'center top',
          horizontalalignment='center', verticalalignment='top')
     fig.text(left, 0.5*(bottom+top), 'right center',
          horizontalalignment='right', verticalalignment='center',
          rotation='vertical')
     fig.text(left, 0.5*(bottom+top), 'left center',
          horizontalalignment='left', verticalalignment='center',
          rotation='vertical')
     fig.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
          horizontalalignment='center', verticalalignment='center',
          fontsize=20, color='red')
     fig.text(right, 0.5*(bottom+top), 'centered',
          horizontalalignment='center', verticalalignment='center',
          rotation='vertical')
     fig.text(left, top, 'rotated\nwith newlines',
          horizontalalignment='center', verticalalignment='center',
          rotation=45)

     caption = f'text layout'
     fig.text(0.01, 0.91, f'{caption}')
     f1 = os.path.join(os.getcwd(), f'results/val_imgs/text_layout.jpg')
     plt.savefig(f1)
     plt.close()


if __name__ == '__main__':
     main()
     # text_layout()
