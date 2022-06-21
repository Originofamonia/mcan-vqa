# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_k = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_q = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_merge = nn.Linear(opt.hidden_size, opt.hidden_size)

        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt.multi_head,
            self.opt.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt.multi_head,
            self.opt.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt.multi_head,
            self.opt.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=opt.hidden_size,
            mid_size=opt.ff_size,
            out_size=opt.hidden_size,
            dropout_rate=opt.dropout_rate,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, opt):
        super(SA, self).__init__()

        self.mhatt = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt.dropout_rate)
        self.norm1 = LayerNorm(opt.hidden_size)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(opt.dropout_rate)
        self.norm2 = LayerNorm(opt.hidden_size)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))
        # x = self.activation(x)  # add gelu
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, opt):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt.dropout_rate)
        self.norm1 = LayerNorm(opt.hidden_size)

        self.dropout2 = nn.Dropout(opt.dropout_rate)
        self.norm2 = LayerNorm(opt.hidden_size)

        self.dropout3 = nn.Dropout(opt.dropout_rate)
        self.norm3 = LayerNorm(opt.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, opt):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(opt) for _ in range(opt.layer)])
        self.dec_list = nn.ModuleList([SGA(opt) for _ in range(opt.layer)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y


class MCAClassifier(nn.Module):
    """
    inputs only img features, for multi-label classification
    """
    def __init__(self, opt):
        super(MCAClassifier, self).__init__()

        self.enc_list = nn.ModuleList([SA(opt) for _ in range(opt.layer)])
        # self.dec_list = nn.ModuleList([SGA(opt) for _ in range(opt.layer)])

    def forward(self, y, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # for dec in self.dec_list:
        #     y = dec(y, y, y_mask, y_mask)

        return y