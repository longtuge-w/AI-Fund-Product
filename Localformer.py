# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import copy
import math

import torch
import torch.nn as nn

from torch.nn.modules.container import ModuleList


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class LocalformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, d_model):
        super(LocalformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.conv = _get_clones(nn.Conv1d(d_model, d_model, 3, 1, 1), num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask):
        output = src
        out = src

        for i, mod in enumerate(self.layers):
            # [T, N, F] --> [N, T, F] --> [N, F, T]
            out = output.transpose(1, 0).transpose(2, 1)
            out = self.conv[i](out).transpose(2, 1).transpose(1, 0)

            output = mod(output + out, src_mask=mask)

        return output + out


class Localformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, output_size=1, dropout=0.5, device=None):
        super(Localformer, self).__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = LocalformerEncoder(self.encoder_layer, num_layers=num_layers, d_model=d_model)
        self.decoder_layer = nn.Linear(d_model, output_size)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        output, _ = self.rnn(output)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()
