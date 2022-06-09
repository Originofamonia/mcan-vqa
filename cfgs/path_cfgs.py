# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.dataset_path = './datasets/vqa/'

        # bottom up features root path
        self.feature_path = './datasets/coco_extract/'
        self.cxr_data_path = f'/home/xinyue/VQA_ReGat/data/mimic/mimic_'

        self.init_path()


    def init_path(self):

        self.img_feat_path = {
            'train': self.feature_path + 'train2014/',
            'val': self.feature_path + 'val2014/',
            'test': self.feature_path + 'test2015/',
        }

        self.cxr_img_feat_path = f'/home/xinyue/VQA_ReGat/data/mimic/cmb_bbox_features_full.hdf5'

        self.mimic_qa_path = {
            'train': self.cxr_data_path + 'dataset_train_full.pkl',
            'val': self.cxr_data_path + 'dataset_val_full.pkl',
            'test': self.cxr_data_path + 'dataset_test_full.pkl',
        }

        self.mimic_ans_dict_path = {
            'ans2idx': self.cxr_data_path + 'ans2label_full.pkl',
            'idx2ans': self.cxr_data_path + 'label2ans_full.pkl',
        }

        self.question_path = {
            'train': self.dataset_path + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'val': self.dataset_path + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'test': self.dataset_path + 'v2_OpenEnded_mscoco_test2015_questions.json',
            'vg': self.dataset_path + 'VG_questions.json',
        }

        self.answer_path = {
            'train': self.dataset_path + 'v2_mscoco_train2014_annotations.json',
            'val': self.dataset_path + 'v2_mscoco_val2014_annotations.json',
            'vg': self.dataset_path + 'VG_annotations.json',
        }

        self.result_path = './results/result_test/'
        self.pred_path = './results/pred/'
        self.cache_path = './results/cache/'
        self.log_path = f'/drive/qiyuan/mcan-vqa/'
        self.ckpt_path = './ckpt/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')

    def check_path(self):
        print('Checking dataset ...')

        for mode in self.img_feat_path:
            if not os.path.exists(self.img_feat_path[mode]):
                print(self.img_feat_path[mode] + 'not exist')
                exit(-1)

        for mode in self.question_path:
            if not os.path.exists(self.question_path[mode]):
                print(self.question_path[mode] + 'not exist')
                exit(-1)

        for mode in self.answer_path:
            if not os.path.exists(self.answer_path[mode]):
                print(self.answer_path[mode] + 'not exist')
                exit(-1)

        print('check path finished')
    
    def check_mimic_path(self):
        print('Check MIMIC VQA dataset')

        if not os.path.exists(self.cxr_img_feat_path):
            print(self.cxr_img_feat_path + 'not exist')
            exit(-1)

        for mode in self.mimic_qa_path:
            if not os.path.exists(self.mimic_qa_path[mode]):
                print(self.mimic_qa_path[mode] + 'not exist')
                exit(-1)

        print('MIMIC check path finished')
