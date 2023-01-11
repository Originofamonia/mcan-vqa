# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from cfgs.path_cfgs import PATH
# from path_cfgs import PATH

import os, torch, random
import numpy as np
from types import MethodType


class Cfgs(PATH):
    def __init__(self):
        super(Cfgs, self).__init__()

        # Set Devices
        # If use multi-gpu training, set e.g.'0, 1, 2' instead
        self.gpu = '0,1'

        # Set RNG For CPU And GPUs
        self.seed = 444

        # -------------------------
        # ---- Version Control ----
        # -------------------------

        # Define a specific name to start new training
        # self.VERSION = 'Anonymous_' + str(self.SEED)
        self.version = str(self.seed)

        # Resume training
        self.resume = False

        # Used in Resume training and testing
        self.ckpt_version = self.version
        self.ckpt_epoch = 0

        # Absolutely checkpoint path, 'CKPT_VERSION' and 'CKPT_EPOCH' will be overridden
        self.ckpt_path = '/drive/qiyuan/mcan-vqa/'

        # Print loss every step
        self.verbose = True

        # ------------------------------
        # ---- Data Provider Params ----
        # ------------------------------

        # {'train', 'val', 'test'}
        self.run_mode = 'train'

        # Set True to evaluate offline
        self.eval_every_epoch = True

        # Set True to save the prediction vector (Ensemble)
        self.test_save_pred = False

        # Pre-load the features into memory to increase the I/O speed
        self.preload = False

        # Define the 'train' 'val' 'test' data split
        # (EVAL_EVERY_EPOCH triggered when set {'train': 'train'})
        self.split = {
            'train': 'train',
            'val': 'val',
            'test': 'test',
        }

        # A external method to set train split
        self.train_split = 'train+val+vg'

        # Set True to use pretrained word embedding
        # (GloVe: spaCy https://spacy.io/)
        self.use_glove = True

        # Word embedding matrix size
        # (token size x WORD_EMBED_SIZE)
        self.word_embed_size = 300

        # Max length of question sentences
        self.max_token = 14

        # Filter the answer by occurrence
        # self.ANS_FREQ = 8

        # Max length of extracted faster-rcnn 2048D features
        # (bottom-up and Top-down: https://github.com/peteanderson80/bottom-up-attention)
        self.img_feat_pad_size = 60

        # Faster-rcnn 2048D features
        self.img_feat_size = 1024  # was 2048

        # Default training batch size: 64
        self.batch_size = 64

        # Multi-thread I/O
        self.num_workers = 4

        # Use pin memory
        # (Warning: pin memory can accelerate GPU loading but may
        # increase the CPU memory usage when NUM_WORKS is large)
        self.pin_mem = True

        # Large model can not training with batch size 64
        # Gradient accumulate can split batch to reduce gpu memory usage
        # (Warning: BATCH_SIZE should be divided by GRAD_ACCU_STEPS)
        self.grad_accu_steps = 1

        # Set 'external': use external shuffle method to implement training shuffle
        # Set 'internal': use pytorch dataloader default shuffle method
        self.shuffle_mode = 'internal'


        # ------------------------
        # ---- Network Params ----
        # ------------------------

        # Model deeps
        # (Encoder and Decoder will be same deeps)
        self.layer = 4  # only 6 is best, 8 is bad

        # Model hidden size
        # (512 as default, bigger will be a sharp increase of gpu memory usage)
        self.hidden_size = 512  # was 512 for att pooling

        # Multi-head number in MCA layers
        # (Warning: HIDDEN_SIZE should be divided by MULTI_HEAD)
        self.multi_head = 8

        # Dropout rate for all dropout layers
        # (dropout can prevent overfitting： [Dropout: a simple way to prevent neural networks from overfitting])
        self.dropout_rate = 0.1

        # MLP size in flatten layers
        self.flat_mlp_size = 512

        # Flatten the last hidden to vector with {n} attention glimpses
        self.flat_glimpses = 1
        self.flat_out_size = 1024  # was 1024

        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------

        # The base learning rate
        self.lr_base = 1e-4  # vit is 3e-5

        # Learning rate decay ratio
        self.lr_decay_rate = 0.2

        # Learning rate decay at {x, y, z...} epoch
        self.lr_decay_list = [8, 12]

        # Max training epoch
        self.max_epoch = 13

        # Gradient clip
        # (default: -1 means not using)
        self.grad_norm_clip = -1

        # Adam optimizer betas and eps
        self.opt_betas = (0.9, 0.98)
        self.opt_eps = 1e-9


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict


    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


    def proc(self):
        """
        set args
        """
        assert self.run_mode in ['train', 'val', 'test']

        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        self.n_gpu = len(str(self.gpu).split(','))
        # self.devices = [_ for _ in range(self.n_gpu)]
        # torch.set_num_threads(2)

        # ------------ Seed setup
        # fix pytorch seed
        torch.manual_seed(self.seed)
        if self.n_gpu < 2:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True

        # fix numpy seed
        np.random.seed(self.seed)

        # fix random seed
        random.seed(self.seed)

        if self.ckpt_path is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')
            self.ckpt_version = self.ckpt_path.split('/')[-1] + '_'


        # ------------ Split setup
        self.split['train'] = self.train_split
        # if 'val' in self.split['train'].split('+') or self.run_mode not in ['train']:
        #     self.eval_every_epoch = False

        if self.run_mode not in ['test']:
            self.test_save_pred = False


        # ------------ Gradient accumulate setup
        assert self.batch_size % self.grad_accu_steps == 0
        self.sub_batch_size = int(self.batch_size / self.grad_accu_steps)

        # Use a small eval batch will reduce gpu memory usage
        self.eval_batch_size = self.sub_batch_size


        # ------------ Networks setup
        # FeedForwardNet size in every MCA layer
        self.ff_size = int(self.hidden_size * 4)

        # A pipe line hidden size in attention compute
        assert self.hidden_size % self.multi_head == 0
        self.hidden_size_head = int(self.hidden_size / self.multi_head)


    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''
