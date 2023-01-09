# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.weight_norm import weight_norm
# from block import fusions

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED, MCAClassifier

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):  # att pooling
    def __init__(self, opt):
        super(AttFlat, self).__init__()
        self.opt = opt

        self.mlp = MLP(
            in_size=opt.hidden_size,
            mid_size=opt.flat_mlp_size,
            out_size=opt.flat_glimpses,  # att heads
            dropout_rate=opt.dropout_rate,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            opt.hidden_size * opt.flat_glimpses,
            opt.flat_out_size
        )

    def forward(self, x, x_mask):
        att_w = self.mlp(x)
        att_w = att_w.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att_w = F.softmax(att_w, dim=1)  # att weights

        att_list = []
        for i in range(self.opt.flat_glimpses):
            att_list.append(
                torch.sum(att_w[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted, att_w


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, opt, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=opt.word_embed_size
        )

        # Loading the GloVe embedding weights
        if opt.use_glove:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=opt.word_embed_size,
            hidden_size=opt.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            opt.img_feat_size,
            opt.hidden_size
        )

        self.backbone = MCA_ED(opt)

        self.attflat_img = AttFlat(opt)
        self.attflat_lang = AttFlat(opt)

        self.proj_norm = LayerNorm(opt.flat_out_size)
        self.proj = nn.Linear(opt.flat_out_size, answer_size)


    def forward(self, v, ques_ix):

        # Make mask
        q_mask = self.make_mask(ques_ix.unsqueeze(2))
        v_mask = self.make_mask(v)

        # Pre-process Language Feature
        q = self.embedding(ques_ix)
        q, _ = self.lstm(q)

        # Pre-process Image Feature
        v = self.img_feat_linear(v)

        # Backbone Framework
        q, v = self.backbone(  # was q, v, q_mask, v_mask
            q,
            v,
            q_mask,
            v_mask
        )

        lang_feat_flat, q_w = self.attflat_lang(  # [B, 512]
            q,
            q_mask
        )

        img_feat_flat, v_w = self.attflat_img(  # [B, 512]
            v,
            v_mask
        )

        a = lang_feat_flat + img_feat_flat
        a = self.proj_norm(a)  # [B, 512]
        logits = torch.sigmoid(self.proj(a))

        return logits, v, v_mask, v_w, q, q_mask, q_w, a


    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), 
            dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class ClassifierNet(nn.Module):
    def __init__(self, opt, answer_size) -> None:
        super(ClassifierNet, self).__init__()
        self.img_feat_linear = nn.Linear(
            opt.img_feat_size,
            opt.hidden_size
        )

        self.backbone = MCAClassifier(opt)

        self.attflat_img = AttFlat(opt)
        self.attflat_lang = AttFlat(opt)

        self.proj_norm = LayerNorm(opt.flat_out_size)
        self.proj = nn.Linear(opt.flat_out_size, answer_size)


    def forward(self, v):

        # Make mask
        # q_mask = self.make_mask(ques_ix.unsqueeze(2))
        v_mask = self.make_mask(v)

        # Pre-process Language Feature
        # q = self.embedding(ques_ix)
        # q, _ = self.lstm(q)

        # Pre-process Image Feature
        v = self.img_feat_linear(v)

        # Backbone Framework
        v = self.backbone(
            v,
            v_mask
        )

        # lang_feat_flat, q_w = self.attflat_lang(  # [B, 512]
        #     q,
        #     q_mask
        # )

        img_feat_flat, v_w = self.attflat_img(  # [B, 512]
            v,
            v_mask
        )

        a = img_feat_flat
        a = self.proj_norm(a)  # [B, 512]
        logits = torch.sigmoid(self.proj(a))

        return logits, v, v_mask, v_w, a


    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), 
            dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),
                                      dim=None))
            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),
                                  dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# class MuTAN_Attention(nn.Module):
#     def __init__(self, dim_v, dim_q, dim_out, method="Mutan", mlp_glimpses=0):
#         super(MuTAN_Attention, self).__init__()
#         self.mlp_glimpses = mlp_glimpses
#         self.fusion = getattr(fusions, method)(
#                         [dim_q, dim_v], dim_out, mm_dim=1200,
#                         dropout_input=0.1)
#         if self.mlp_glimpses > 0:
#             self.linear0 = FCNet([dim_out, 512], '', 0)
#             self.linear1 = FCNet([512, mlp_glimpses], '', 0)

#     def forward(self, q, v):
#         alpha = self.process_attention(q, v)

#         if self.mlp_glimpses > 0:
#             alpha = self.linear0(alpha)
#             alpha = F.relu(alpha)
#             alpha = self.linear1(alpha)

#         alpha = F.softmax(alpha, dim=1)

#         if alpha.size(2) > 1:  # nb_glimpses > 1
#             alphas = torch.unbind(alpha, dim=2)
#             v_outs = []
#             for alpha in alphas:
#                 alpha = alpha.unsqueeze(2).expand_as(v)
#                 v_out = alpha*v
#                 v_out = v_out.sum(1)
#                 v_outs.append(v_out)
#             v_out = torch.cat(v_outs, dim=1)
#         else:
#             alpha = alpha.expand_as(v)
#             v_out = alpha*v
#             v_out = v_out.sum(1)
#         return v_out

#     def process_attention(self, q, v):
#         batch_size = q.size(0)
#         n_regions = v.size(1)
#         # q = q[:, None, :].expand(q.size(0), n_regions, q.size(1)).clone()
#         q = torch.unsqueeze(q, dim=1).repeat(1, n_regions, 1)
#         alpha = self.fusion([
#             q.contiguous().view(batch_size * n_regions, -1),
#             v.contiguous().view(batch_size * n_regions, -1)
#         ])
#         alpha = alpha.view(batch_size, n_regions, -1)
#         return alpha


# class MuTAN(nn.Module):
#     def __init__(self, v_relation_dim, num_hid, num_ans_candidates, gamma):
#         super(MuTAN, self).__init__()
#         self.gamma = gamma
#         self.attention = MuTAN_Attention(v_relation_dim, num_hid,
#                                          dim_out=360, method="Mutan",
#                                          mlp_glimpses=gamma)
#         self.fusion = getattr(fusions, "Mutan")(
#                         [num_hid, v_relation_dim*2], num_ans_candidates,
#                         mm_dim=1200, dropout_input=0.1)

#     def forward(self, v_relation, q_emb):
#         # b: bounding box features, not used for this fusion method
#         att = self.attention(q_emb, v_relation)
#         logits = self.fusion([q_emb, att])
#         return logits, att


class Net2(nn.Module):
    """
    replace att pooling with Mutan
    TODO: replace att pooling w Mutan
    """
    def __init__(self, opt, pretrained_emb, token_size, answer_size):
        super(Net2, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=opt.word_embed_size
        )

        # Loading the GloVe embedding weights
        if opt.use_glove:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=opt.word_embed_size,
            hidden_size=opt.hidden_size,
            num_layers=1,
            dropout=opt.dropout_rate,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            opt.img_feat_size,
            opt.hidden_size
        )

        self.backbone = MCA_ED(opt)
        # was relation_dim: 1024, num_hid: 1024, mutan_gamma: 2
        # self.joint_embedding = MuTAN(opt.img_feat_size, opt.img_feat_size,
        #                              answer_size, opt.mutan_gamma)

        self.attflat_img = AttFlat(opt)
        self.attflat_lang = AttFlat(opt)

        self.proj_norm = LayerNorm(opt.flat_out_size)
        self.proj = nn.Linear(opt.flat_out_size, answer_size)


    def forward(self, v, ques_ix):

        # Make mask
        q_mask = self.make_mask(ques_ix.unsqueeze(2))
        v_mask = self.make_mask(v)

        # Pre-process Language Feature
        q = self.embedding(ques_ix)
        q, (hn, cn) = self.lstm(q)

        # Pre-process Image Feature
        v = self.img_feat_linear(v)

        # Backbone Framework
        q, v = self.backbone(  # was q, v, q_mask, v_mask
            q,
            v,
            q_mask,
            v_mask
        )
        # v_emb: [B, 36, 1024], q_emb_self_att: [B, 1024]
        

        lang_feat_flat, q_w = self.attflat_lang(  # [B, 512]
            q,
            q_mask
        )
        # logits, att = self.joint_embedding(v, lang_feat_flat)

        img_feat_flat, v_w = self.attflat_img(  # [B, 512]
            v,
            v_mask
        )

        a = lang_feat_flat + img_feat_flat
        a = self.proj(self.proj_norm(a))  # [B, 512]
        logits = torch.sigmoid(a)

        return logits, v, v_mask, q, q_mask


    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), 
            dim=-1) == 0).unsqueeze(1).unsqueeze(2)
