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
            out_size=opt.flat_glimpses,
            dropout_rate=opt.dropout_rate,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            opt.hidden_size * opt.flat_glimpses,
            opt.flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.opt.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


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


    def forward(self, img_feat, ques_ix):

        # Make mask
        q_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        q_feat = self.embedding(ques_ix)
        q_feat, _ = self.lstm(q_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        q_feat, img_feat = self.backbone(
            q_feat,
            img_feat,
            q_feat_mask,
            img_feat_mask
        )

        lang_feat_flat = self.attflat_lang(
            q_feat,
            q_feat_mask
        )

        img_feat_flat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        ans_feat = lang_feat_flat + img_feat_flat
        ans_feat = self.proj_norm(ans_feat)
        logits = torch.sigmoid(self.proj(ans_feat))

        return logits, img_feat, img_feat_mask, q_feat, q_feat_mask, ans_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
