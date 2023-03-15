# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.input_size = input_size
        self.name = 'LSTM'

    def forward(self, x):
        # [N, T, F]
        out, _ = self.rnn(x)
        out = self.fc_out(out).squeeze()
        return out
