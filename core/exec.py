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
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib.gridspec import GridSpec

from core.data.load_data import CustomDataset, CustomLoader
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval


class Execution:
    def __init__(self, opt):
        self.opt = opt

        print('Loading training set ........')
        self.dataset = CustomDataset(opt)

        self.dataset_eval = None
        if opt.eval_every_epoch:
            eval_opt = copy.deepcopy(opt)
            setattr(eval_opt, 'run_mode', 'val')

            print('Loading validation set for per-epoch evaluation ...')
            # self.dataset_eval = CustomDataset(eval_opt)


    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        net = Net(
            self.opt,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.opt.n_gpu > 1:
            net = nn.DataParallel(net, device_ids=self.opt.devices)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Load checkpoint if resume training
        if self.opt.resume:
            print(' ========== Resume training')

            if self.opt.ckpt_path is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.opt.ckpt_path
            else:
                path = self.opt.ckpts_path + \
                       'ckpt_' + self.opt.ckpt_version + \
                       '/epoch' + str(self.opt.ckpt_epoch) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.opt, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.opt.batch_size * self.opt.ckpt_epoch)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.opt.ckpt_epoch

        else:
            if ('ckpt_' + self.opt.version) in os.listdir(self.opt.ckpts_path):
                shutil.rmtree(self.opt.ckpts_path + 'ckpt_' + self.opt.version)

            os.mkdir(self.opt.ckpts_path + 'ckpt_' + self.opt.version)

            optim = get_optim(self.opt, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
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

        # Training script
        for epoch in range(start_epoch, self.opt.max_epoch):

            # Save log information
            logfile = open(
                self.opt.log_path +
                'log_run_' + self.opt.version + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.opt.lr_decay_list:
                adjust_lr(optim, self.opt.lr_decay_rate)

            # Externally shuffle
            if self.opt.shuffle_mode == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            for step, batch in enumerate(dataloader):
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


                    logits, _, _, _ = net(
                        sub_img_feat_iter, sub_ques_ix_iter)

                    loss = loss_fn(logits, sub_ans_iter)
                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.__C.GRAD_ACCU_STEPS
                    loss.backward()
                    loss_sum += loss.cpu().data.numpy() * self.opt.grad_accu_steps

                    if self.opt.verbose:
                        if dataset_eval is not None:
                            mode_str = self.opt.split['train'] + '->' + self.opt.split['val']
                        else:
                            mode_str = self.opt.split['train'] + '->' + self.opt.split['test']

                        print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.opt.version,
                            epoch + 1,
                            step,
                            int(data_size / self.opt.batch_size),
                            mode_str,
                            loss.cpu().data.numpy() / self.opt.sub_batch_size,
                            optim._rate
                        ), end='          ')

                # Gradient norm clipping
                if self.opt.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
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
            print('Finished in {}s'.format(int(time_end-time_start)))

            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }

            torch.save(
                state,
                self.opt.ckpts_path +
                'ckpt_' + self.opt.version +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.opt.log_path +
                'log_run_' + self.opt.version + '.txt',
                'a+'
            )

            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True)

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))

    def visualize(self, dataset, state_dict=None, valid=False):
        """
        plot 1. img+box; 2. q-q; 3. v-v; 4. v-q; 5. v-a; 6. q-a.
        """
        # Load parameters
        path = f'{self.opt.ckpts_path}epoch{str(self.opt.ckpt_epoch)}_512.pkl'

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
        iid = '228749'
        # q_list = [item for item in dataset.ques_list if item['question'] == question]
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
        ans_ix_list = []
        pred_list = []

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
            if step > 300:
                break
            else:
                # img_feat_iter, ques_ix_iter, ans_iter, img_feats, idx = batch
                img_feat_iter, ques_ix_iter, ans_iter, boxes, idx = manual_batch_from_iid(question, iid, dataset)

                iid, q, qid = [str(item) for item in dataloader.dataset.ques_list[idx].values()]
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

                    # ans_ix_list.append(pred_argmax)
                    # pred = dataloader.dataset.ix_to_ans[str(pred_argmax[0])]
                    preds = [dataloader.dataset.ix_to_ans[str(item)] for item in topk_preds]
                    ans = [item['answer'] for item in dataloader.dataset.ans_list[idx]['answers']]

                    qq, qa, va_values, va_indices, vv, vq = calc_mats_v2(v_feat[0], 
                        v_mask[0], v_w[0], q_feat[0], q_mask[0], q_w[0], a)
                    plot_boxesv2(im_file, iid, q, preds, ans, boxes, 
                        qq, qa, va_values, va_indices, vv, vq)

    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        if self.opt.ckpt_path is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.opt.ckpt_path
        else:
            path = self.opt.ckpts_path + \
                    'epoch' + str(self.opt.ckpt_epoch) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('load model finished.')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []

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
            img_feat_iter, ques_ix_iter, ans_iter, idx = batch
            print("\rEvaluation: [step %4d/%4d]" % (
                step, int(data_size / self.opt.eval_batch_size),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            logits = net(
                img_feat_iter,
                ques_ix_iter
            )
            pred_np = logits.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # Save the answer index
            if pred_argmax.shape[0] != self.opt.eval_batch_size:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.opt.eval_batch_size - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)
            # pred = dataloader.dataset.ix_to_ans[str(pred_argmax[0])]

            # Save the whole prediction vector
            if self.opt.test_save_pred:
                if pred_np.shape[0] != self.opt.eval_batch_size:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.opt.eval_batch_size - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

        ans_ix_list = np.array(ans_ix_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])
        }for qix in range(qid_list.__len__())]

        # Write the results to result file
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.opt.cache_path + \
                    'result_run_' + self.opt.ckpt_version + \
                    '.json'
            else:
                result_eval_file = \
                    self.opt.cache_path + \
                    'result_run_' + self.opt.version + \
                    '.json'

        else:
            if self.opt.ckpt_path is not None:
                result_eval_file = \
                    self.opt.result_path + \
                    'result_run_' + self.opt.ckpt_version + \
                    '.json'
            else:
                result_eval_file = \
                    self.opt.result_path + \
                    'result_run_' + self.opt.ckpt_version + \
                    '_epoch' + str(self.opt.ckpt_epoch) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.opt.test_save_pred:

            if self.opt.ckpt_path is not None:
                ensemble_file = \
                    self.opt.pred_path + \
                    'result_run_' + self.opt.ckpt_version + \
                    '.json'
            else:
                ensemble_file = \
                    self.opt.pred_path + \
                    'result_run_' + self.opt.ckpt_version + \
                    '_epoch' + str(self.opt.ckpt_epoch) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.opt.question_path['val']
            ans_file_path = self.opt.answer_path['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            vqaEval.evaluate()

            # print accuracies
            print("\n")
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # print("Per Question Type Accuracy is the following:")
            # for quesType in vqaEval.accuracy['perQuestionType']:
            #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            # print("\n")
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")

            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.opt.log_path +
                    'log_run_' + self.opt.ckpt_version + '.txt',
                    'a+')
                )

                logfile = open(
                    self.opt.log_path +
                    'log_run_' + self.opt.ckpt_version + '.txt',
                    'a+'
                )

            else:
                print('Write to log file: {}'.format(
                    self.opt.log_path +
                    'log_run_' + self.opt.version + '.txt',
                    'a+')
                )

                logfile = open(
                    self.opt.log_path +
                    'log_run_' + self.opt.version + '.txt',
                    'a+'
                )

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n\n")
            logfile.close()


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.opt.version)
            self.train(self.dataset, self.dataset_eval)
        elif run_mode == 'val':
            # self.eval(self.dataset, valid=True)
            self.visualize(self.dataset, valid=True)
        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.opt.log_path + 'log_run_' + version + '.txt')):
            os.remove(self.opt.log_path + 'log_run_' + version + '.txt')
        print('Finished!\n')


def plot_boxes(im_file, iid, q, preds, ans, boxes, qq, qa, va_values, va_indices, vv, vq):
    """
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    TODO: plot all 6 figures, put make axs in here
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
    
    ax3 = fig.add_subplot(gs[:2, :2])
    ax3.imshow(empty_im)
    ax3.axis('off')

    q_words = q.split(' ')
    qa_weights = qa.detach().cpu().numpy()
    qa_weights = qa_weights / qa_weights.max()

    x0, y0 = 0.01, 0.4
    q_cap = f'q: {q}'
    t0 = fig.text(x0, y0, q_cap, wrap=True)

    x1, y1 = 0.01, 0.2
    a_cap = f'a: {ans}'
    t1 = fig.text(x1, y1, a_cap, wrap=True)

    t_x, t_y = 0.5, 0.4
    if len(q_words) > 14:
        q_words = q_words[:14]
    for i, w_q in enumerate(q_words):
        if i < len(qa_weights):
            t2 = fig.text(t_x, t_y, f'{w_q} ', alpha=qa_weights[i][0])
            bb = t2.get_window_extent(renderer=r)
            t_x += bb.width / im.shape[1]

    x, y = 0.5, 0.29
    caption2 = f'preds: {preds}'
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


def manual_batch_from_iid(q, iid, dataset):
    idx = [dataset.ques_list.index(item) for item in dataset.ques_list if str(item['image_id']) == iid and item['question'] == q]
    idx = idx[0]
    img_feat_iter, ques_ix_iter, ans_iter, img_feats, idx = dataset[idx]
    img_feat_iter = img_feat_iter.unsqueeze(0)
    ques_ix_iter = ques_ix_iter.unsqueeze(0)
    ans_iter = ans_iter.unsqueeze(0)
    idx = torch.tensor([idx])
    return img_feat_iter, ques_ix_iter, ans_iter, torch.from_numpy(img_feats['bbox']).cuda(), idx
