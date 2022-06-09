# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import h5py
import pickle
import numpy as np
import pandas as pd
import glob, json, torch, time
from torch.utils.data import Dataset, DataLoader

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import pad_img_feat, proc_ques, proc_ans, proc_mimic_ans


class CustomDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # ---- Raw data loading ----
        # Loading all image paths
        # if self.opt.preload:
        self.img_feat_path_list = []
        split_list = opt.split[opt.run_mode].split('+')  # change this for split
        # split_list = ['test']
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(opt.img_feat_path[split] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(opt.question_path['train'], 'r'))['questions'] + \
            json.load(open(opt.question_path['val'], 'r'))['questions'] + \
            json.load(open(opt.question_path['test'], 'r'))['questions'] + \
            json.load(open(opt.question_path['vg'], 'r'))['questions']

        # Loading answer word list
        # self.stat_ans_list = \
        #     json.load(open(__C.answer_path['train'], 'r'))['annotations'] + \
        #     json.load(open(__C.answer_path['val'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        # split_list = opt.split[opt.run_mode].split('+')
        # split_list = ['train', 'val', 'test']
        for split in split_list:
            self.ques_list += json.load(open(opt.question_path[split], 'r'))['questions']
            # if opt.run_mode in ['train']:
            self.ans_list += json.load(open(opt.answer_path[split], 'r'))['annotations']

        # Define run data size
        if opt.run_mode in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)

        # ---- Data statistic ----
        # {image id} -> {image feature absolutely path}
        if self.opt.preload:
            print('==== Pre-Loading features ...')
            time_start = time.time()
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print('==== Finished in {}s'.format(int(time_end-time_start)))
        else:
            self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, opt.use_glove)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('load dataset finished.')

    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.opt.run_mode in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            if self.opt.preload:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                img_feats = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                img_feat_x = img_feats['x'].transpose((1, 0))
            img_feat_iter = pad_img_feat(img_feat_x, self.opt.img_feat_pad_size)
            boxes = pad_img_feat(img_feats['bbox'], self.opt.img_feat_pad_size)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.opt.max_token)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)
            return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter), torch.from_numpy(boxes), torch.tensor([idx]), self.opt.run_mode

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            # # Process image feature from (.npz) file
            # img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            # img_feat_x = img_feat['x'].transpose((1, 0))
            # Process image feature from (.npz) file
            if self.opt.preload:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feats = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                img_feat_x = img_feats['x'].transpose((1, 0))
            img_feat_iter = pad_img_feat(img_feat_x, self.opt.img_feat_pad_size)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.opt.max_token)
            # only works for batchsize=1
            return torch.from_numpy(img_feat_iter), \
                torch.from_numpy(ques_ix_iter), \
                torch.from_numpy(ans_iter), img_feats, idx


    def __len__(self):
        return self.data_size


class MIMICDatasetBase(Dataset):
    """
    MIMIC dataset base: includes everything but (train/val/test) qa pickles
    """
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        f1 = h5py.File(opt.cxr_img_feat_path, 'r')
        print(f'keys: {f1.keys()}')
        self.image_features = f1['image_features']  # [377110, 60, 1024]
        self.bbox_label = f1['bbox_label']  # [377k, 60]
        self.image_adj_matrix = f1['image_adj_matrix']  # [377k, 100, 100]
        self.image_bb = f1['image_bb']  # [377k, 60, 4]
        self.pos_boxes = f1['pos_boxes']  # [377k, 2]
        self.semantic_adj_matrix = f1['semantic_adj_matrix']  # [377k, 100, 100]
        self.spatial_features = f1['spatial_features']  # [377k, 60, 6]

        self.v_dim = self.image_features.chunks[-1]  # visual feat dim
        self.s_dim = self.spatial_features.chunks[-1]  # spatial dim

        with open(opt.mimic_ans_dict_path['ans2idx'], 'rb') as f3:
            self.ans_to_ix = pickle.load(f3)
        # because no_finding becomes yes or no, so 15 labels
        with open(opt.mimic_ans_dict_path['idx2ans'], 'rb') as f4:
            self.ix_to_ans = pickle.load(f4)
        self.ans_size = self.ans_to_ix.__len__()
        
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('load mimic base dataset finished.')


class MIMICDatasetSplit(MIMICDatasetBase):
    """
    train/val/test split of MIMIC QA dataset
    """
    def __init__(self, opt) -> None:
        super().__init__(opt)
        with open(opt.mimic_qa_path[opt.run_mode], 'rb') as f2:
            self.qa = pickle.load(f2)  # qa pairs
        
        self.token_to_ix, self.pretrained_emb = tokenize(self.qa, opt.use_glove)
        self.token_size = self.token_to_ix.__len__()
        self.data_size = self.qa.__len__()
        print('== Question token vocab size:', self.token_size)
    
    def __getitem__(self, idx):
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        qa = self.qa[idx]
        # raw_question = qa["question"]
        # image_id = qa["dicom_id"]

        if self.opt.run_mode in ['train']:
            img_feats = np.array(self.image_features[qa['image']])
            img_feat_iter = pad_img_feat(img_feats, self.opt.img_feat_pad_size)
            # boxes = pad_img_feat(self.image_bb[qa['image']], self.opt.img_feat_pad_size)

            # Process question
            ques_ix_iter = proc_ques(qa, self.token_to_ix, self.opt.max_token)

            # Process answer
            ans_iter = proc_mimic_ans(qa['answer'], self.ans_to_ix)

            return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), torch.from_numpy(ans_iter), \
               torch.tensor([idx]), self.opt.run_mode

        else:  # ['val', 'test']
            img_feats = np.array(self.image_features[qa['image']])
            img_feat_iter = pad_img_feat(img_feats, self.opt.img_feat_pad_size)
            boxes = pad_img_feat(self.image_bb[qa['image']], self.opt.img_feat_pad_size)

            # Process question
            ques_ix_iter = proc_ques(qa, self.token_to_ix, self.opt.max_token)

            # Process answer
            ans_iter = proc_mimic_ans(qa['answer'], self.ans_to_ix)
            # only works for batchsize=1
            return torch.from_numpy(img_feat_iter), \
                torch.from_numpy(ques_ix_iter), torch.from_numpy(ans_iter), \
                img_feats, boxes, torch.tensor([idx])

    def __len__(self):
        return self.data_size


class CustomLoader(DataLoader):
    def __init__(self, dataset, opt):
        # self.dataset = dataset
        self.opt = opt
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': self.opt.batch_size,
            'shuffle': True,
            'collate_fn': self.collate_fn,
            'num_workers': self.opt.num_workers,
            'pin_memory': self.opt.pin_mem,
            'drop_last': True,
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        if data[0][-1] in ['train']:
            img_feat_iter, ques_ix_iter, ans_iter, idx, _ = zip(*data)
            img_feat_iter = torch.stack(img_feat_iter, dim=0)
            ques_ix_iter = torch.stack(ques_ix_iter, dim=0)
            ans_iter = torch.stack(ans_iter, dim=0)
            idx = torch.stack(idx, dim=0)
            return img_feat_iter, ques_ix_iter, ans_iter, idx
        else:
            img_feat_iter, ques_ix_iter, ans_iter, img_feats, boxes, idx = zip(*data)
            img_feat_iter = torch.stack(img_feat_iter, dim=0)
            ques_ix_iter = torch.stack(ques_ix_iter, dim=0)
            ans_iter = torch.stack(ans_iter, dim=0)
            idx = torch.stack(idx, dim=0)
            return img_feat_iter, ques_ix_iter, ans_iter, img_feats, boxes, idx
