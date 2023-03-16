# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# modify this to our VQA dataset
# --------------------------------------------------------

import os
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors

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

    parser.add_argument('--gpu', default='1',
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
    # compute some interesting data
    x0, x1 = -5, 5
    y0, y1 = -3, 3
    x = np.linspace(x0, x1, 500)
    y = np.linspace(y0, y1, 500)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2

    # Set up a colormap:
    # use copy so that we do not mutate the global colormap instance
    palette = copy(plt.cm.gray)
    palette.set_over('r', 1.0)
    palette.set_under('g', 1.0)
    palette.set_bad('b', 1.0)
    # Alternatively, we could use
    # palette.set_bad(alpha = 0.0)
    # to make the bad region transparent.  This is the default.
    # If you comment out all the palette.set* lines, you will see
    # all the defaults; under and over will be colored with the
    # first and last colors in the palette, respectively.
    Zm = np.ma.masked_where(Z > 1.2, Z)

    # By setting vmin and vmax in the norm, we establish the
    # range to which the regular palette color scale is applied.
    # Anything above that range is colored based on palette.set_over, etc.

    # set up the Axes objects
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 5.4))

    # plot using 'continuous' color map
    im = ax1.imshow(Zm, interpolation='bilinear',
              cmap=palette,
              norm=colors.Normalize(vmin=-1.0, vmax=1.0),
              aspect='auto',
              origin='lower',
              extent=[x0, x1, y0, y1])
    ax1.set_title('Green=low, Red=high, Blue=masked')
    cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax1)
    cbar.set_label('uniform')
    for ticklabel in ax1.xaxis.get_ticklabels():
        ticklabel.set_visible(False)

    # Plot using a small number of colors, with unevenly spaced boundaries.
    im = ax2.imshow(Zm, interpolation='nearest',
              cmap=palette,
              norm=colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                                       ncolors=palette.N),
              aspect='auto',
              origin='lower',
              extent=[x0, x1, y0, y1])
    ax2.set_title('With BoundaryNorm')
    cbar = fig.colorbar(im, extend='both', spacing='proportional',
                   shrink=0.9, ax=ax2)
    cbar.set_label('proportional')

    fig.suptitle('imshow, with out-of-range and masked data')

    f1 = os.path.join(os.getcwd(), f'results/val_imgs/dark_mask.jpg')
    plt.savefig(f1)
    plt.close()


if __name__ == '__main__':
     main()
     # text_layout()
