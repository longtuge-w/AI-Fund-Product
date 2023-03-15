import math
import torch
import torch.nn as nn


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


class Transformer(nn.Module):
    def __init__(self, input_size=6, d_model=64, nhead=4, output_size=10, num_layers=15, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.name = 'Transformer'
        self.feature_layer = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, output_size)
        self.norm1 = nn.BatchNorm1d(input_size)
        if output_size != 1:
            self.norm2 = nn.BatchNorm1d(output_size)
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.name = 'Transformer'

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        # src = src.reshape(len(src), self.input_size, -1).permute(0, 2, 1)
        src = src.permute(0, 2, 1)
        src = self.norm1(src)
        src = src.permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        output = output.squeeze()
        if self.output_size != 1:
            output = self.norm2(output)

        return output
