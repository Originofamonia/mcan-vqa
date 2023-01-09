# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import os, json, torch, datetime, pickle, copy, shutil, time
from textwrap import wrap
from os.path import exists
import shutil
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score
import wandb

from core.data.load_data import CustomDataset, CustomLoader, MIMICDatasetBase, MIMICDatasetSplit
from core.model.net import Net, ClassifierNet, Net2
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval


class Execution:
    def __init__(self, opt):
        self.opt = opt
        self.model = None
        print('Load training set ........')
        self.dataset = CustomDataset(opt)

        self.dataset_eval = None
        if opt.eval_every_epoch:
            test_opt = copy.deepcopy(opt)
            setattr(test_opt, 'run_mode', 'test')

            print('Loading validation set for per-epoch evaluation ...')
            print('Load test set for evaluation ...')
            self.dataset_test = MIMICDatasetSplit(test_opt)

    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        self.model = Net(
            self.opt,
            pretrained_emb,
            token_size,
            ans_size
        )
        self.model.cuda()
        self.model.train()
        
        # Define the multi-gpu training if needed
        if self.opt.n_gpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.devices)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum')

        # Load checkpoint if resume training
        if self.opt.resume:
            print('Resume training')

            if self.opt.ckpt_path is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.opt.ckpt_path
            else:
                path = self.opt.ckpt_path + \
                       'ckpt_' + self.opt.ckpt_version + \
                       '/epoch' + str(self.opt.ckpt_epoch) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            self.model.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.opt, self.model, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.opt.batch_size * self.opt.ckpt_epoch)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.opt.ckpt_epoch

        else:
            ckpt_path_latter = 'ckpt_' + str(self.opt.seed)
            ckpt_path_full = os.path.join(self.opt.ckpt_path, ckpt_path_latter)
            if ckpt_path_latter in os.listdir(self.opt.ckpt_path):
                shutil.rmtree(ckpt_path_full)

            os.mkdir(ckpt_path_full)

            optim = get_optim(self.opt, self.model, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(self.model.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        # if self.opt.shuffle_mode in ['external']:
        #     dataloader = DataLoader(
        #         dataset,
        #         batch_size=self.opt.batch_size,
        #         shuffle=False,
        #         num_workers=self.opt.num_workers,
        #         pin_memory=self.opt.pin_mem,
        #         drop_last=True
        #     )
        # else:
        dataloader = CustomLoader(
            dataset, self.opt
            # batch_size=self.opt.batch_size,
            # shuffle=True,
            # num_workers=self.opt.num_workers,
            # pin_memory=self.opt.pin_mem,
            # drop_last=True
        )
        
        # Save log information
        logfile = open(
            self.opt.log_path +
            f'log_run_{str(self.opt.seed)}.txt',
            'w'
        )
        logfile.write(
            f"NowTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        logfile.write(json.dumps(self.opt.__dict__))

        # Training script
        for epoch in range(start_epoch, self.opt.max_epoch):
            self.model.train()
            # Learning Rate Decay
            if epoch in self.opt.lr_decay_list:
                adjust_lr(optim, self.opt.lr_decay_rate)

            # Externally shuffle
            if self.opt.shuffle_mode == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            pbar = tqdm(dataloader)
            for step, batch in enumerate(pbar):
                img_feat_iter, ques_ix_iter, ans_iter, idx = batch
                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                for accu_step in range(self.opt.grad_accu_steps):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.opt.sub_batch_size:
                                      (accu_step + 1) * self.opt.sub_batch_size]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.opt.sub_batch_size:
                                     (accu_step + 1) * self.opt.sub_batch_size]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.opt.sub_batch_size:
                                 (accu_step + 1) * self.opt.sub_batch_size]

                    logits, _, _, _, _, _, _, _ = self.model(
                        sub_img_feat_iter, sub_ques_ix_iter)  # was sub_ques_ix_iter

                    loss = loss_fn(logits, sub_ans_iter)
                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.opt.GRAD_ACCU_STEPS
                    loss.backward()
                    loss_sum += loss.item() * self.opt.grad_accu_steps

                    if self.opt.verbose:
                        descr = f"e: {epoch}; s: {step}/{int(data_size / self.opt.batch_size)}; " +\
                              f"loss: {loss.item()/self.opt.sub_batch_size:.3f}, lr: {optim._rate:.2e}"
                        pbar.set_description(descr)

                # Gradient norm clipping
                if self.opt.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.opt.grad_norm_clip
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.opt.grad_accu_steps
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print(f'Train epoch {epoch} finished in {int(time_end-time_start)}s')

            epoch_finish = epoch + 1

            # Logging
            logfile.write(
                f'epoch = {str(epoch_finish)}; loss = {str(loss_sum / data_size)}; ' +
                f'lr = {str(optim._rate)}\n\n'
            )
            # wandb.log({'train_loss': loss.item(),})

            # Eval after every epoch
            if self.opt.eval_every_epoch:
                if epoch % 2 == 0:
                    perclass_roc, micro_roc, macro_roc = self.eval(self.dataset,)
                    logfile.write(
                        f"train perclass_roc: {perclass_roc};\n" +
                        f"micro_roc: {micro_roc:.3f}; macro roc: {macro_roc:.3f}.\n"
                    )
                    # wandb.log({'train macro roc': macro_roc})
                perclass_roc, micro_roc, macro_roc = self.eval(self.dataset_test,)
                logfile.write(
                    f"test perclass_roc: {perclass_roc};\n" +
                    f"micro_roc: {micro_roc:.3f}; macro roc: {macro_roc:.3f}.\n"
                )
                # wandb.log({'test macro roc': macro_roc})
            loss_sum = 0
            grad_norm = np.zeros(len(named_params))

        # Save checkpoint
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': optim.optimizer.state_dict(),
            'lr_base': optim.lr_base
        }
        try:
            torch.save(
                state,
                os.path.join(self.opt.ckpt_path, 'ckpt_' + str(self.opt.seed) +
                '/epoch' + str(epoch_finish) + '.pt')
            )
        except:
            print('error in saving model')
        return self.model

    def visualize(self, dataset, state_dict=None, valid=False):
        """
        plot 1. img+box; 2. q-q; 3. v-v; 4. v-q; 5. v-a; 6. q-a.
        """
        # Load parameters
        path = f'{self.opt.ckpt_path}epoch{str(self.opt.ckpt_epoch)}_512.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('load model finished.')

        # Store the prediction list
        task = 'val'
        src_path = f'/home/qiyuan/2022spring/mcan-vqa/datasets/{task}2014/COCO_{task}2014_000000'
        tgt_path = f'/home/qiyuan/2022spring/mcan-vqa/results/val_imgs/'
        question = 'What is the green vegetable?'
        # iid = '228749'
        qid_list = [str(item['question_id']) for item in dataset.ques_list if item['question'].startswith('How many')]
        # for item in q_list:
        #     iid = str(item['image_id'])
        #     if len(iid) < 6:
        #         l_iid = len(iid)
        #         iid = '0' * (6 - l_iid) + iid
        #     img_path = f'{src_path}'+str(item['image_id'])+'.jpg'
        #     try:
        #         if exists(img_path):
        #             shutil.copy(img_path, tgt_path)
        #     except:
        #         print(f"{item['image_id']} doesnt exist")

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(self.opt, pretrained_emb, token_size, ans_size)
        net.cuda()
        net.eval()

        if self.opt.n_gpu > 1:
            net = nn.DataParallel(net, device_ids=self.opt.devices)

        net.load_state_dict(state_dict)

        dataloader = DataLoader(
            dataset,
            batch_size=self.opt.eval_batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            pin_memory=True
        )

        for step, batch in enumerate(dataloader):
            if step > 1000:
                break
            else:
                img_feat_iter, ques_ix_iter, ans_iter, img_feats, idx = batch
                # img_feat_iter, ques_ix_iter, ans_iter, boxes, idx = manual_batch(question, iid, dataset)

                iid, q, qid = [str(item) for item in dataloader.dataset.ques_list[idx].values()]
                if qid in qid_list:
                    if len(iid) < 6:
                        l_iid = len(iid)
                        iid = '0' * (6 - l_iid) + iid
                    im_file = f'{os.getcwd()}/datasets/{task}2014/COCO_{task}2014_000000{iid}.jpg'
                    if exists(im_file):
                        img_feat_iter = img_feat_iter.cuda()
                        ques_ix_iter = ques_ix_iter.cuda()
                        # img_feat: [B, 100, 512]
                        logits, v_feat, v_mask, v_w, q_feat, q_mask, q_w, a = net(
                            img_feat_iter, ques_ix_iter)  # lang_feat: [B, 14, 512]
                        pred_np = logits.cpu().data.numpy()
                        pred_argmax = np.argmax(pred_np, axis=1)
                        topk_preds = pred_np[0].argsort()[-4:][::-1]

                        # Save the answer index
                        if pred_argmax.shape[0] != self.opt.eval_batch_size:
                            pred_argmax = np.pad(
                                pred_argmax,
                                (0, self.opt.eval_batch_size - pred_argmax.shape[0]),
                                mode='constant',
                                constant_values=-1
                            )

                        # pred = dataloader.dataset.ix_to_ans[str(pred_argmax[0])]
                        preds = [dataloader.dataset.ix_to_ans[str(item)] for item in topk_preds]
                        ans = [item['answer'] for item in dataloader.dataset.ans_list[idx]['answers']]

                        qq, qa, va_values, va_indices, vv, vq = calc_mats_v2(v_feat[0], 
                            v_mask[0], v_w[0], q_feat[0], q_mask[0], q_w[0], a)
                        plot_boxesv2(im_file, iid, q, preds, ans, img_feats['bbox'][0], 
                            qq, qa, va_values, va_indices, vv, vq)

    def eval(self, dataset,):
        """
        eval on MIMIC val/test dataset
        """
        # load model
        if self.model:
            self.model.eval()
            net = self.model
        else:
            net = Net(self.opt, pretrained_emb, token_size, ans_size)
            net.cuda()
            net.eval()
            if self.opt.n_gpu > 1:
                net = nn.DataParallel(net, device_ids=self.opt.devices)
            path = f'/drive/qiyuan/mcan-vqackpt_50729920/epoch13.pkl'
            state_dict = torch.load(path)['state_dict']
            net.load_state_dict(state_dict)

        # data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        dataloader = CustomLoader(
            dataset,
            self.opt,
            # batch_size=self.opt.eval_batch_size,
            # shuffle=False,
            # num_workers=self.opt.num_workers,
            # pin_memory=True
        )
        preds = []
        targets = []

        for step, batch in enumerate(dataloader):
            img_feat_iter, ques_ix_iter, ans_iter, img_feats, boxes, idx = batch
            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            results = net(
                img_feat_iter,
                ques_ix_iter
            )
            logits, _, _, _, _, _, _, _ = results  # [B, 15]
            preds.append(logits.detach().cpu().numpy())
            targets.append(ans_iter.detach().cpu().numpy())
        
        targets = np.vstack(targets)
        preds = np.vstack(preds)
        try:
            perclass_roc = roc_auc_score(targets, preds, average=None)
            print(f'per class ROC: {perclass_roc}')
            micro_roc = roc_auc_score(targets, preds, average='micro')
            print(f'micro ROC: {micro_roc}')
            macro_roc = roc_auc_score(targets, preds, average='macro')
            print(f'macro ROC: {macro_roc}')
            return perclass_roc, micro_roc, macro_roc
        except:
            print(f"except in {self.opt.run_mode}")

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.opt.seed)
            self.train(self.dataset, self.dataset_eval)
        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)
            # self.visualize(self.dataset, valid=True)
        elif run_mode == 'test':
            self.eval(self.dataset)

    def empty_log(self, version):
        print('Initialize log file')
        if (os.path.exists(self.opt.log_path + 'log_run_' + str(version) + '.txt')):
            os.remove(self.opt.log_path + 'log_run_' + str(version) + '.txt')
        print('Initialize log file finished!\n')


class ExecuteMIMIC(Execution):
    def __init__(self, opt):
        """
        init mimic dataset here
        """
        self.opt = opt
        self.model = None  # trained model
        print('Load mimic training set')
        self.dataset = MIMICDatasetSplit(opt)

        val_opt = copy.deepcopy(opt)
        setattr(val_opt, 'run_mode', 'val')
        print('Load validation set for evaluation ...')
        self.dataset_val = MIMICDatasetSplit(val_opt)

        test_opt = copy.deepcopy(val_opt)
        setattr(test_opt, 'run_mode', 'test')
        print('Load test set for evaluation ...')
        self.dataset_test = MIMICDatasetSplit(test_opt)
    
    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.opt.seed)
            self.train(self.dataset,)
        elif run_mode == 'val':
            self.eval(self.dataset_val)
            # self.visualize(self.dataset, valid=True)
        elif run_mode == 'test':
            self.eval(self.dataset_test)
    
    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        self.model = Net2(
            self.opt,
            pretrained_emb,
            token_size,
            ans_size
        )
        # self.model.cuda()
        self.model.to(self.opt.gpu)
        self.model.train()
        
        # Define the multi-gpu training if needed
        if self.opt.n_gpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.devices)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum')

        # Load checkpoint if resume training
        if self.opt.resume:
            print('Resume training')

            if self.opt.ckpt_path is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.opt.ckpt_path
            else:
                path = self.opt.ckpt_path + \
                       'ckpt_' + self.opt.ckpt_version + \
                       '/epoch' + str(self.opt.ckpt_epoch) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            self.model.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.opt, self.model, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.opt.batch_size * self.opt.ckpt_epoch)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.opt.ckpt_epoch

        else:
            ckpt_path_latter = 'ckpt_' + str(self.opt.seed)
            ckpt_path_full = os.path.join(self.opt.ckpt_path, ckpt_path_latter)
            if ckpt_path_latter in os.listdir(self.opt.ckpt_path):
                shutil.rmtree(ckpt_path_full)

            os.mkdir(ckpt_path_full)

            optim = get_optim(self.opt, self.model, data_size)
            start_epoch = 0

        # loss_sum = 0
        reg_crit = nn.SmoothL1Loss()  # add regularization
        named_params = list(self.model.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        # if self.opt.shuffle_mode in ['external']:
        #     dataloader = DataLoader(
        #         dataset,
        #         batch_size=self.opt.batch_size,
        #         shuffle=False,
        #         num_workers=self.opt.num_workers,
        #         pin_memory=self.opt.pin_mem,
        #         drop_last=True
        #     )
        # else:
        dataloader = CustomLoader(
            dataset, self.opt
            # batch_size=self.opt.batch_size,
            # shuffle=True,
            # num_workers=self.opt.num_workers,
            # pin_memory=self.opt.pin_mem,
            # drop_last=True
        )
        
        # Save log information
        logfile = open(
            self.opt.log_path +
            f'log_run_{str(self.opt.seed)}.txt',
            'w'
        )
        logfile.write(
            f"NowTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        logfile.write(json.dumps(self.opt.__dict__))

        # Training script
        for epoch in range(start_epoch, self.opt.max_epoch):
            self.model.train()
            # Learning Rate Decay
            if epoch in self.opt.lr_decay_list:
                adjust_lr(optim, self.opt.lr_decay_rate)

            # Externally shuffle
            if self.opt.shuffle_mode == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            pbar = tqdm(dataloader)
            for step, batch in enumerate(pbar):
                img_feat_iter, ques_ix_iter, ans_iter, idx = batch
                optim.zero_grad()

                img_feat_iter = img_feat_iter.to(self.opt.gpu)
                ques_ix_iter = ques_ix_iter.to(self.opt.gpu)
                ans_iter = ans_iter.to(self.opt.gpu)

                for accu_step in range(self.opt.grad_accu_steps):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.opt.sub_batch_size:
                                      (accu_step + 1) * self.opt.sub_batch_size]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.opt.sub_batch_size:
                                     (accu_step + 1) * self.opt.sub_batch_size]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.opt.sub_batch_size:
                                 (accu_step + 1) * self.opt.sub_batch_size]

                    logits, _, _, _, _ = self.model(
                        sub_img_feat_iter, sub_ques_ix_iter)  # was sub_ques_ix_iter

                    main_loss = loss_fn(logits, sub_ans_iter)
                    reg_loss = 0
                    if self.opt.reg_factor > 0:
                        for param in self.model.parameters():
                            reg_loss += compute_l1_loss(param)
                            reg_loss += compute_l2_loss(param)
                        loss = self.opt.reg_factor * reg_loss + main_loss
                        loss.backward()
                    else:
                        main_loss.backward()
                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.opt.GRAD_ACCU_STEPS
                    # loss.backward()
                    # loss_sum += loss.item() * self.opt.grad_accu_steps

                    if self.opt.verbose:
                        descr = f"e: {epoch}; reg_loss: {self.opt.reg_factor * reg_loss:.3f}; " +\
                              f"main_loss: {main_loss.item():.3f}, lr: {optim._rate:.2e}"
                        pbar.set_description(descr)

                    if step % self.opt.eval_interval == 1:
                        
                        # Eval after every epoch
                        if self.dataset_test is not None:
                            # tentatively eval on train set?
                            perclass_roc, micro_roc, macro_roc = self.eval(self.dataset_test,)
                            logfile.write(f'step: {step}; ' + 
                                f"perclass_roc: {perclass_roc};\n" +
                                f"micro_roc: {micro_roc:.3f}; macro roc: {macro_roc:.3f}.\n"
                            )
                            # wandb.log({'main_loss': main_loss.item(), 'test macro roc': macro_roc})
                        self.model.train()

                # Gradient norm clipping
                if self.opt.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.opt.grad_norm_clip
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.opt.grad_accu_steps
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print(f'Train epoch {epoch} finished in {int(time_end-time_start)}s')

            epoch_finish = epoch + 1

            # Logging
            logfile.write(
                f'epoch = {str(epoch_finish)}; loss = {main_loss.item()}; ' +
                f'lr = {str(optim._rate)}\n\n'
            )
            # wandb.log({'train_loss': main_loss.item(),})

            # Eval after every epoch
            if self.opt.eval_every_epoch:
                if epoch % 2 == 0:
                    perclass_roc, micro_roc, macro_roc = self.eval(self.dataset,)
                    logfile.write(
                        f"train perclass_roc: {perclass_roc};\n" +
                        f"micro_roc: {micro_roc:.3f}; macro roc: {macro_roc:.3f}.\n"
                    )
                    # wandb.log({'train macro roc': macro_roc})

                perclass_roc, micro_roc, macro_roc = self.eval(self.dataset_test,)
                logfile.write(
                    f"test perclass_roc: {perclass_roc};\n" +
                    f"micro_roc: {micro_roc:.3f}; macro roc: {macro_roc:.3f}.\n"
                )
                # wandb.log({'test macro roc': macro_roc})

            grad_norm = np.zeros(len(named_params))

        # Save checkpoint
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': optim.optimizer.state_dict(),
            'lr_base': optim.lr_base
        }
        try:
            torch.save(
                state,
                os.path.join(self.opt.ckpt_path, 'ckpt_' + str(self.opt.seed) +
                '/epoch' + str(epoch_finish) + '.pt')
            )
        except:
            print('error in saving model')
        
    
    def eval(self, dataset):
        """
        eval on MIMIC val/test dataset
        """
        # load model
        if self.model:
            self.model.eval()
            net = self.model
        else:
            net = Net(self.opt, pretrained_emb, token_size, ans_size)
            net.to(self.opt.gpu)
            net.eval()
            if self.opt.n_gpu > 1:
                net = nn.DataParallel(net, device_ids=self.opt.devices)
            path = f'/drive/qiyuan/mcan-vqackpt_50729920/epoch13.pkl'
            state_dict = torch.load(path)['state_dict']
            net.load_state_dict(state_dict)

        # data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        dataloader = DataLoader(  # shouldn't use CustomLoader cuz no shuffle & drop_last=True
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True
        )
        preds = []
        targets = []
        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            img_feat_iter, ques_ix_iter, ans_iter, idx = batch
            img_feat_iter = img_feat_iter.to(self.opt.gpu)
            ques_ix_iter = ques_ix_iter.to(self.opt.gpu)

            results = net(img_feat_iter, ques_ix_iter,)
            logits, _, _, _, _ = results  # [B, 15]
            preds.append(logits.detach().cpu().numpy())
            targets.append(ans_iter.detach().cpu().numpy())
            pbar.set_description(f'eval on {self.opt.run_mode}; s: ' +
                f'{step}/{int(dataset.data_size / self.opt.batch_size)}')

        targets = np.vstack(targets)
        preds = np.vstack(preds)

        perclass_roc = roc_auc_score(targets, preds, average=None)
        print(f'per class ROC: {perclass_roc}')
        micro_roc = roc_auc_score(targets, preds, average='micro')
        print(f'micro ROC: {micro_roc:.3f}')
        macro_roc = roc_auc_score(targets, preds, average='macro')
        print(f'macro ROC: {macro_roc:.3f}')
        return perclass_roc, micro_roc, macro_roc


class ExecClassify(Execution):
    def __init__(self, opt):
        self.opt = opt
        self.model = None  # trained model
        print('Load mimic training set')
        self.dataset = MIMICDatasetSplit(opt)

        eval_opt = copy.deepcopy(opt)
        setattr(eval_opt, 'run_mode', 'val')
        print('Load validation set for evaluation ...')
        self.dataset_val = MIMICDatasetSplit(eval_opt)

        test_opt = copy.deepcopy(eval_opt)
        setattr(test_opt, 'run_mode', 'test')
        print('Load test set for evaluation ...')
        self.dataset_test = MIMICDatasetSplit(test_opt)
    
    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.opt.seed)
            self.train(self.dataset,)
        elif run_mode == 'val':
            self.eval(self.dataset_val)
            # self.visualize(self.dataset, valid=True)
        elif run_mode == 'test':
            self.eval(self.dataset_test)

    def train(self, dataset, dataset_val=None):

        # Obtain needed information
        data_size = dataset.data_size
        # token_size = dataset.token_size
        ans_size = dataset.ans_size
        # pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        self.model = ClassifierNet(
            self.opt,
            # pretrained_emb,
            # token_size,
            ans_size
        )
        self.model.cuda()
        self.model.train()
        
        # Define the multi-gpu training if needed
        if self.opt.n_gpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.devices)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')  # very bad
        loss_fn = torch.nn.BCELoss(reduction='sum')

        # Load checkpoint if resume training
        if self.opt.resume:
            print('Resume training')

            if self.opt.ckpt_path is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.opt.ckpt_path
            else:
                path = self.opt.ckpt_path + \
                       'ckpt_' + self.opt.ckpt_version + \
                       '/epoch' + str(self.opt.ckpt_epoch) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            self.model.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.opt, self.model, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.opt.batch_size * self.opt.ckpt_epoch)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.opt.ckpt_epoch

        else:
            ckpt_path_latter = 'ckpt_' + str(self.opt.seed)
            ckpt_path_full = os.path.join(self.opt.ckpt_path, ckpt_path_latter)
            if ckpt_path_latter in os.listdir(self.opt.ckpt_path):
                shutil.rmtree(ckpt_path_full)

            os.mkdir(ckpt_path_full)

            optim = get_optim(self.opt, self.model, data_size)
            start_epoch = 0

        reg_crit = nn.SmoothL1Loss()  # add regularization
        named_params = list(self.model.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        # if self.opt.shuffle_mode in ['external']:
        #     dataloader = DataLoader(
        #         dataset,
        #         batch_size=self.opt.batch_size,
        #         shuffle=False,
        #         num_workers=self.opt.num_workers,
        #         pin_memory=self.opt.pin_mem,
        #         drop_last=True
        #     )
        # else:
        dataloader = CustomLoader(
            dataset, self.opt
            # batch_size=self.opt.batch_size,
            # shuffle=True,
            # num_workers=self.opt.num_workers,
            # pin_memory=self.opt.pin_mem,
            # drop_last=True
        )
        
        # Save log information
        logfile = open(
            self.opt.log_path +
            f'log_run_{str(self.opt.seed)}.txt',
            'w'
        )
        logfile.write(
            f"NowTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        logfile.write(json.dumps(self.opt.__dict__))

        # Training script
        for epoch in range(start_epoch, self.opt.max_epoch):
            self.model.train()
            # Learning Rate Decay
            if epoch in self.opt.lr_decay_list:
                adjust_lr(optim, self.opt.lr_decay_rate)

            # Externally shuffle
            if self.opt.shuffle_mode == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            pbar = tqdm(dataloader)
            for step, batch in enumerate(pbar):
                img_feat_iter, ques_ix_iter, ans_iter, idx, multilabel = batch
                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                multilabel = multilabel.cuda()

                for accu_step in range(self.opt.grad_accu_steps):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.opt.sub_batch_size:
                                      (accu_step + 1) * self.opt.sub_batch_size]
                    multilabel_iter = \
                        multilabel[accu_step * self.opt.sub_batch_size:
                                     (accu_step + 1) * self.opt.sub_batch_size]
                    # sub_ans_iter = \
                    #     ans_iter[accu_step * self.opt.sub_batch_size:
                    #              (accu_step + 1) * self.opt.sub_batch_size]

                    logits, _, _, _, _ = self.model(
                        sub_img_feat_iter)  # was sub_ques_ix_iter

                    loss = loss_fn(logits, multilabel_iter)

                    if self.opt.reg_factor > 0:
                        reg_loss = 0
                        for param in self.model.parameters():
                            reg_loss += reg_crit(param, target=torch.zeros_like(param))
                        loss += self.opt.reg_factor * reg_loss

                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.opt.GRAD_ACCU_STEPS
                    loss.backward()
                    # loss_sum += loss.item() * self.opt.grad_accu_steps

                    if self.opt.verbose:
                        descr = f"train: e: {epoch+1}; s: {step}/{int(data_size / self.opt.batch_size)}; " +\
                              f"l: {loss.item()/self.opt.sub_batch_size:.3f}; lr: {optim._rate:.2e}"
                        pbar.set_description(descr)
                    
                    if step % self.opt.eval_interval == 0:
                        
                        # Eval after every epoch
                        if self.dataset_test is not None:
                            # tentatively eval on train set?
                            perclass_roc, micro_roc, macro_roc = self.eval(self.dataset_test,)
                            logfile.write(f'step: {step}; ' + 
                                f"perclass_roc: {perclass_roc};\n" +
                                f"micro_roc: {micro_roc:.3f}; macro roc: {macro_roc:.3f}.\n"
                            )
                            # wandb.log({'train_loss': loss.item(), 'test macro roc': macro_roc})
                        self.model.train()

                # Gradient norm clipping
                if self.opt.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.opt.grad_norm_clip
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.opt.grad_accu_steps

                optim.step()

            time_end = time.time()
            print(f'Train epoch {epoch} finished in {int(time_end-time_start)}s')

            perclass_roc, micro_roc, macro_roc = self.eval(self.dataset,)
            # wandb.log({'train macro roc': macro_roc})
            perclass_roc, micro_roc, macro_roc = self.eval(self.dataset_val,)
            # wandb.log({'valid macro roc': macro_roc})
            perclass_roc, micro_roc, macro_roc = self.eval(self.dataset_test,)
            # wandb.log({'test macro roc': macro_roc})
            self.model.train()
            epoch_finish = epoch + 1

            # Logging
            logfile.write(
                f'epoch = {str(epoch_finish)}; loss = {str(loss.item() / data_size)}; ' +
                f'lr = {str(optim._rate)}\n\n'
            )

            # Eval after every epoch
            if self.dataset_test is not None:
                perclass_roc, micro_roc, macro_roc = self.eval(self.dataset_test,)
                logfile.write(
                    f"perclass_roc: {perclass_roc};\n" +
                    f"micro_roc: {micro_roc:.3f}; macro roc: {macro_roc:.3f}.\n"
                )
            # loss_sum = 0
            grad_norm = np.zeros(len(named_params))

        # Save checkpoint
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': optim.optimizer.state_dict(),
            'lr_base': optim.lr_base
        }
        try:
            torch.save(
                state,
                os.path.join(self.opt.ckpt_path, 'ckpt_' + str(self.opt.seed) +
                '/epoch' + str(epoch_finish) + '.pt')
            )
        except:
            print('error in saving model')
        # return self.model

    def eval(self, dataset):
        """
        eval for classify 14 classes
        """
        # load model
        if self.model:
            self.model.eval()
            net = self.model
        else:
            net = Net(self.opt, pretrained_emb, token_size, ans_size)
            net.cuda()
            net.eval()
            if self.opt.n_gpu > 1:
                net = nn.DataParallel(net, device_ids=self.opt.devices)
            path = f'/drive/qiyuan/mcan-vqackpt_50729920/epoch13.pkl'
            state_dict = torch.load(path)['state_dict']
            net.load_state_dict(state_dict)

        # data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        dataloader = CustomLoader(
            dataset,
            self.opt,
        )
        preds = []
        targets = []
        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            img_feat_iter, ques_ix_iter, ans_iter, idx, multilabel = batch
            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            results = net(img_feat_iter)
            logits, _, _, _, _ = results  # [B, 15]
            preds.append(logits.detach().cpu().numpy())
            targets.append(multilabel.detach().cpu().numpy())
            desc = f'eval batch: {step}/ {len(dataset)/self.opt.batch_size}'
            pbar.set_description(desc)

        targets = np.vstack(targets)
        preds = np.vstack(preds)
        try:
            print(f'run_mode: {self.opt.run_mode}')
            perclass_roc = roc_auc_score(targets, preds, average=None)
            print(f'per class ROC: {perclass_roc}')
            micro_roc = roc_auc_score(targets, preds, average='micro')
            print(f'micro ROC: {micro_roc:.3f}')
            macro_roc = roc_auc_score(targets, preds, average='macro')
            print(f'macro ROC: {macro_roc:.3f}')
            return perclass_roc, micro_roc, macro_roc
        except:
            print(f"except in {self.opt.run_mode}")

def plot_boxes(im_file, iid, q, preds, ans, boxes, qq, qa, va_values, va_indices, vv, vq):
    """
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    plot all 6 figures, put make axs in here
    """
    if type(preds) is str:
        pred = preds
    else:
        pred = preds[0]
    va_indices = va_indices.squeeze()  # .detach().cpu().numpy()
    selected_boxes = boxes[va_indices]
    min_box = va_values.min()
    box_range = va_values.max() - min_box

    im = plt.imread(im_file) / 255
    fig = plt.figure()
    r = fig.canvas.get_renderer()
    gs = GridSpec(4, 4, fig)
    ax0 = fig.add_subplot(gs[:3, :3])
    # ax0.imshow(im)  # 1. img+box
    # background = im / 255 / 7
    all_masks = np.zeros(im.shape)
    for i, box in enumerate(selected_boxes):  # is xyxy, verified
        c = np.random.rand(3,)
        left = int(box[0].item())
        top = int(box[1].item())
        w, h = int((box[2]-box[0]).item()), int((box[3]-box[1]).item())
        box_weights = ((va_values[i] - min_box) / box_range).item()
        dark_mask = np.zeros(im.shape)
        dark_mask[top:(top + h), left:(left + w)] = box_weights
        all_masks = np.maximum(dark_mask, all_masks)
        all_masks = np.clip(all_masks, 0, 1)
        # rect = Rectangle((left, top), w, h, linewidth=1.5 * box_weights, 
        #     edgecolor=c, facecolor='none', alpha=box_weights)
        # ax0.add_artist(rect)

        ax0.text(left, top, va_indices[i].item(), horizontalalignment='right',
            verticalalignment='bottom', color=c)
    ax0.imshow(im * all_masks)

    ax1 = fig.add_subplot(gs[0, 3])
    ax1.imshow(vv.detach().cpu().numpy())
    ax1.set_xticks(np.arange(len(vv)))
    ax1.set_xticklabels(va_indices.detach().cpu().numpy())
    ax1.set_yticks(np.arange(len(vv)))
    ax1.set_yticklabels(va_indices.detach().cpu().numpy())
    
    q_words = q.split(' ')
    ax2 = fig.add_subplot(gs[1, 3])
    im2 = ax2.imshow(qq.detach().cpu().numpy())
    ax2.set_xticks(np.arange(len(q_words)))
    ax2.set_xticklabels(q_words)
    ax2.set_yticks(np.arange(len(q_words)))
    ax2.set_yticklabels(q_words)
    plt.setp(ax2.get_xticklabels(), rotation=-45, ha='left', rotation_mode='anchor')
    # cbar = ax2.figure.colorbar(im2, ax=ax2,)
    # cbar.ax.set_ylabel(ylabel='', cbarlabel='', rotation=-90, va="bottom")

    ax3 = fig.add_subplot(gs[3, 0])
    qa_weights = qa.detach().cpu().numpy()
    qa_weights = qa_weights / qa_weights.max()
    ax3.imshow(qa_weights)
    ax3.set_yticks(np.arange(len(q_words)))
    ax3.set_yticklabels(q_words)
    ax3.set_xticks([0])
    ax3.set_xticklabels([pred])

    ax4 = fig.add_subplot(gs[3, 1])
    ax4.imshow(va_values.detach().cpu().numpy())
    ax4.set_yticks(np.arange(len(vv)))
    ax4.set_yticklabels(va_indices.detach().cpu().numpy())
    ax4.set_xticks([0])
    ax4.set_xticklabels([pred])

    ax5 = fig.add_subplot(gs[3, 3])
    ax5.imshow(vq.detach().cpu().numpy())
    ax5.set_yticks(np.arange(len(vv)))
    ax5.set_yticklabels(va_indices.detach().cpu().numpy())
    ax5.set_xticks(np.arange(len(q_words)))
    ax5.set_xticklabels(q_words)
    plt.setp(ax5.get_xticklabels(), rotation=-45, ha='left', rotation_mode='anchor')

    x, y = 0.01, 0.89
    t_w, t_y = 0.01, 0.96
    if len(q_words) > 14:
        q_words = q_words[:14]
    for i, w_q in enumerate(q_words):
        t1 = fig.text(t_w, t_y, f'{w_q} ', alpha=qa_weights[i][0])
        bb = t1.get_window_extent(renderer=r)
        t_w += bb.width / im.shape[1]
    
    caption2 = f'preds: {preds}\nans: {ans}'
    t2 = fig.text(x, y, f'{caption2}', wrap=False)
    f1 = os.path.join(os.getcwd(), f'results/val_imgs/{iid}.jpg')
    plt.savefig(f1)
    plt.close()


def plot_boxesv2(im_file, iid, q, preds, ans, boxes, qq, qa, va_values, va_indices, vv, vq):
    """
    plot as fig8
    """
    if type(preds) is str:
        pred = preds
    else:
        pred = preds[0]

    va_indices = va_indices.squeeze()  # .detach().cpu().numpy()
    selected_boxes = boxes[va_indices]
    min_box = va_values.min()
    box_range = va_values.max() - min_box

    im = plt.imread(im_file) / 255
    fig = plt.figure()
    r = fig.canvas.get_renderer()
    gs = GridSpec(4, 4, fig)
    ax0 = fig.add_subplot(gs[:2, :2])
    ax0.imshow(im)  # 1. img+box
    
    ax1 = fig.add_subplot(gs[:2, 2:])
    all_masks = np.zeros(im.shape)
    for i, box in enumerate(selected_boxes):  # is xyxy, verified
        c = np.random.rand(3,)
        left = int(box[0].item())
        top = int(box[1].item())
        w, h = int((box[2]-box[0]).item()), int((box[3]-box[1]).item())
        box_weights = ((va_values[i] - min_box) / box_range).item()
        dark_mask = np.zeros(im.shape)
        dark_mask[top:(top + h), left:(left + w)] = box_weights
        all_masks = np.maximum(dark_mask, all_masks)
        all_masks = np.clip(all_masks, 0, 1)

        ax1.text(left, top, va_indices[i].item(), horizontalalignment='right',
            verticalalignment='bottom', color=c)
    ax1.imshow(im * all_masks)

    empty_im = np.ones(im.shape)
    ax2 = fig.add_subplot(gs[2:, :2])
    ax2.imshow(empty_im)
    ax2.axis('off')
    x0, y0 = 0.01, 0.4
    q_cap = f'q: {q}'
    t0 = fig.text(x0, y0, q_cap, wrap=True)

    x1, y1 = 0.01, 0.15
    a_cap = f'a: {ans}'
    t1 = fig.text(x1, y1, a_cap, wrap=True)

    q_words = q.split(' ')
    qa_weights = qa.detach().cpu().numpy()
    qa_weights = qa_weights / qa_weights.max()
    ax3 = fig.add_subplot(gs[2:, 2:])
    ax3.imshow(empty_im)
    ax3.axis('off')

    t_x, t_y = 0.5, 0.37
    if len(q_words) > 14:
        q_words = q_words[:14]
    for i, w_q in enumerate(q_words):
        if i < len(qa_weights):
            t2 = fig.text(t_x, t_y, f'{w_q} ', alpha=qa_weights[i][0], wrap=True)
            bb = t2.get_window_extent(renderer=r)
            t_x += bb.width / im.shape[1]
            # if t_x >= im.shape[1]:
            #     t_y -= bb.height
            #     t_x = 0.5

    x, y = 0.5, 0.25
    caption2 = f'p: {preds}'
    t3 = fig.text(x, y, f'{caption2}', wrap=True)
    f1 = os.path.join(os.getcwd(), f'results/val_imgs/{iid}.jpg')
    plt.savefig(f1)
    plt.close()


def calc_mats(v, v_mask, v_w, q, q_mask, q_w, a):
    """
    calc all matrices scores: vv, qq, qa, va, vq
    1. use ans select img_feat: no, doesn't make sense
    2. use q select v
    ans * v to select v, tentatively select 7 boxes
    ans * q to highlight q and select len(q) rois for visualize
    """
    v_mask = ~v_mask.squeeze()
    v = v[v_mask]  # [41, 512]
    q_mask = ~q_mask.squeeze()
    q = q[q_mask]  # [4, 512]
    qq_scores = torch.matmul(q, q.permute(1, 0))
    # len_q = q_feat.size(0)  # tentatively omit
    qa_scores = torch.matmul(q, a.permute(1, 0))  # [4, 1]
    va_scores = torch.matmul(v, a.permute(1, 0))  # [41, 1]
    va_values, va_indices = torch.topk(va_scores, k=7, dim=0)

    selected_v = v[va_indices.squeeze()]
    vv_scores = torch.matmul(selected_v, selected_v.permute(1, 0))
    vq_scores = torch.matmul(selected_v, q.permute(1, 0))
    return qq_scores, qa_scores, va_values, va_indices, vv_scores, vq_scores


def calc_mats_v2(v, v_mask, v_w, q, q_mask, q_w, a):
    """
    use att_w (v_w, q_w) to select v and q
    
    """
    v_mask = ~v_mask.squeeze()
    v = v[v_mask]  # [41, 512]
    q_mask = ~q_mask.squeeze()
    q = q[q_mask]  # [4, 512]
    qq_scores = torch.matmul(q, q.permute(1, 0))
    # len_q = q_feat.size(0)  # tentatively omit
    qa_scores = q_w[q_mask]  # [4, 1]
    va_values, va_indices = torch.topk(v_w[v_mask], k=7, dim=0)  # [7, 1]

    selected_v = v[va_indices.squeeze()]
    vv_scores = torch.matmul(selected_v, selected_v.permute(1, 0))
    vq_scores = torch.matmul(selected_v, q.permute(1, 0))
    return qq_scores, qa_scores, va_values, va_indices, vv_scores, vq_scores


def manual_batch(q, iid, dataset):
    idx = [dataset.ques_list.index(item) for item in dataset.ques_list if str(item['image_id']) == iid and item['question'] == q]
    idx = idx[0]
    img_feat_iter, ques_ix_iter, ans_iter, img_feats, idx = dataset[idx]
    img_feat_iter = img_feat_iter.unsqueeze(0)
    ques_ix_iter = ques_ix_iter.unsqueeze(0)
    ans_iter = ans_iter.unsqueeze(0)
    idx = torch.tensor([idx])
    return img_feat_iter, ques_ix_iter, ans_iter, torch.from_numpy(img_feats['bbox']).cuda(), idx

# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md
def compute_l1_loss(w):
      return torch.abs(w).sum()


def compute_l2_loss(w):
    return torch.square(w).sum()
