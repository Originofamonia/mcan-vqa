# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import os, json, torch, datetime, pickle, copy, shutil, time
from os.path import exists
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
import matplotlib.path as mpath

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
            self.dataset_eval = CustomDataset(eval_opt)


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
                adjust_lr(optim, self.opt.lr_decay_r)

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

            # print('')
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
                    valid=True
                )

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))

    def visualize(self, dataset, state_dict=None, valid=False):
        """
        plot 1. img+box; 2. q-q; 3. v-v; 4. v-q; 5. v-a; 6. q-a.
        """
        # Load parameters
        path = self.opt.ckpts_path + 'epoch' + str(self.opt.ckpt_epoch) + '.pkl'

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
            if step > 200:
                break
            else:
                img_feat_iter, ques_ix_iter, ans_iter, img_feats, idx = batch
                iid, q, qid = [str(item) for item in dataloader.dataset.ques_list[idx].values()]
                if len(iid) < 6:
                    l_iid = len(iid)
                    iid = '0' * (6 - l_iid) + iid
                im_file = f'{os.getcwd()}/datasets/val2014/COCO_val2014_000000{iid}.jpg'
                if exists(im_file):
                    im = plt.imread(im_file)

                    img_feat_iter = img_feat_iter.cuda()
                    ques_ix_iter = ques_ix_iter.cuda()

                    logits, img_feat, lang_feat, ans_feat = net(  # img_feat: [B, 100, 512]
                        img_feat_iter, ques_ix_iter)  # lang_feat: [B, 14, 512]
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
                    pred = dataloader.dataset.ix_to_ans[str(pred_argmax[0])]
                    ans = [item['answer'] for item in dataloader.dataset.ans_list[idx]['answers']]
                    fig, axs = plt.subplots(2, 3)
                    # fig, axs = plt.subplots()
                    axs[0, 0].imshow(im)  # 1. img+box
                    plot_boxes(axs[0, 0], img_feats[0], 9)

                    caption = f'q: {q}\n pred: {pred}\nans: {ans}'
                    fig.text(0.01, 0.91, f'{caption}')
                    f1 = os.path.join(os.getcwd(), f'results/val_imgs/{iid}.jpg')
                    plt.savefig(f1)
                    plt.close()


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


def plot_boxes(ax, boxes, n_box):
    for i, box in enumerate(boxes[:n_box]):
        box_w, box_h = (box[2] - box[0]).item(), (box[3] - box[0]).item()  # xyxy, xywh or center-xywh not sure
        x = box[0] - box[2] / 2
        y = box[1] - box[3] / 2
        rect = Rectangle((box[0].item(), box[1].item()), box_w, box_h, linewidth=2, 
            edgecolor=np.random.rand(3,), facecolor='none', alpha=1)
        ax.add_patch(rect)
