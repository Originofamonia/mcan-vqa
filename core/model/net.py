# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
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
        q, v = self.backbone(
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
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
